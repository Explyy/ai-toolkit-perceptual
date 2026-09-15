import io
import json
from pathlib import Path

from training_automation.worker import run_worker


class FakeProcess:
    def __init__(self, output: str, exit_code: int):
        self.stdout = io.StringIO(output)
        self.exit_code = exit_code

    def wait(self):
        return self.exit_code


def worker_env(tmp_path: Path):
    return {
        "TRAINING_STORAGE_ROOT": str(tmp_path / "storage"),
        "TRAINING_RUN_ID": "run-1",
        "TRAINING_SHARD_ID": "a",
    }


def test_failed_child_is_logged_and_held_without_process_exit(tmp_path):
    calls = []
    held = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess("actual trainer traceback\n", 7)

    console = io.StringIO()
    result = run_worker(
        env=worker_env(tmp_path), popen=popen,
        hold=lambda: held.append(True), console=console,
    )
    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text()
    assert result == 7
    assert held == [True]
    assert "actual trainer traceback" in log
    assert "parallel worker initial exit=7" in log
    assert "manual cleanup is required" in console.getvalue()
    assert calls[0][0][-2:] == ["training_automation", "parallel-run"]
    assert calls[0][1]["env"]["PYTHONUNBUFFERED"] == "1"


def test_successful_child_returns_without_hold(tmp_path):
    held = []
    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("completed\n", 0),
        hold=lambda: held.append(True),
        console=io.StringIO(),
    )
    assert result == 0
    assert held == []


def test_durable_log_open_failure_holds_instance(monkeypatch, tmp_path):
    held = []

    def fail_open(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(Path, "open", fail_open)
    console = io.StringIO()
    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("child must not start without a durable log")
        ),
        hold=lambda: held.append(True),
        console=console,
    )
    assert result == 127
    assert held == [True]
    assert "could not initialize its durable log" in console.getvalue()
    assert "storage unavailable" in console.getvalue()


def test_completed_queue_failure_retries_archive_only_and_records_backoff(tmp_path):
    calls = []
    sleeps = []
    held = []

    def popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            run_root = tmp_path / "storage/automation/run-1/a"
            (run_root / "bootstrap-state.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "run-1",
                "shard_id": "a",
                "status": "failed",
                "delete_requested": False,
            }), encoding="utf-8")
            (run_root / "queue-state.json").write_text(json.dumps({
                "schema_version": 2,
                "jobs": {
                    "subject-job": {
                        "status": "completed",
                        "training_status": "completed",
                        "evaluation_status": "completed",
                    }
                },
            }), encoding="utf-8")
            return FakeProcess("archive upload interrupted\n", 7)
        return FakeProcess("archive verified and delete accepted\n", 0)

    result = run_worker(
        env=worker_env(tmp_path),
        popen=popen,
        hold=lambda: held.append(True),
        sleep=sleeps.append,
        console=io.StringIO(),
    )

    recovery = json.loads(
        (tmp_path / "storage/automation/run-1/a/worker-recovery-state.json")
        .read_text(encoding="utf-8")
    )
    assert result == 0
    assert len(calls) == 2
    assert sleeps == [5]
    assert held == []
    assert recovery["status"] == "completed"
    assert recovery["attempts_started"] == 1


def test_supervisor_restart_never_resets_terminal_recovery_budget(tmp_path):
    run_root = tmp_path / "storage/automation/run-1/a"
    run_root.mkdir(parents=True)
    (run_root / "worker-recovery-state.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-1",
        "shard_id": "a",
        "status": "completed",
        "attempts_started": 1,
        "last_exit_code": 0,
    }), encoding="utf-8")
    held = []

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal recovery must not launch another child")
        ),
        hold=lambda: held.append(True),
        console=io.StringIO(),
    )

    assert result == 127
    assert held == [True]
