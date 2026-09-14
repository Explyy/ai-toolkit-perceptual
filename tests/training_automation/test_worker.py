import io
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
    assert "parallel worker exit=7" in log
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
