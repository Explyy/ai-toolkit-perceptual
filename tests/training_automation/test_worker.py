import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import training_automation.worker as worker
from training_automation.backup import BackupError
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


def test_held_instance_deletes_itself_only_after_the_printed_deadline(tmp_path):
    events = []
    console = io.StringIO()

    result = run_worker(
        env={**worker_env(tmp_path), "TRAINING_FAILURE_HOLD_SECONDS": "7200"},
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        now=lambda: 0.0,
        console=console,
    )

    printed = console.getvalue()
    assert result == 9
    assert events == ["hold", "delete"]
    assert "holding this instance for diagnosis" in printed
    assert "1970-01-01T02:00:00+00:00" in printed
    assert "(7200s from now)" in printed
    assert "self-delete outcome: accepted" in printed
    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text()
    assert "self-delete outcome: accepted" in log


def test_successful_run_never_requests_a_self_delete(tmp_path):
    events = []
    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("completed\n", 0),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=io.StringIO(),
    )
    assert result == 0
    assert events == []


def test_bounded_hold_waits_for_the_whole_deadline_and_not_longer():
    clock = [1000.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    worker.hold_for(150, sleep=sleep, monotonic=lambda: clock[0])

    assert sum(slept) == 150
    assert slept == [60, 60, 30]


def test_hold_limit_is_configurable_with_a_bounded_conservative_default():
    assert worker.failure_hold_seconds({}) == worker.FAILURE_HOLD_SECONDS
    assert worker.failure_hold_seconds({"TRAINING_FAILURE_HOLD_SECONDS": "90"}) == 90
    assert worker.failure_hold_seconds({"TRAINING_FAILURE_HOLD_SECONDS": "0"}) == 0
    assert (
        worker.failure_hold_seconds({"TRAINING_FAILURE_HOLD_SECONDS": "not-a-number"})
        == worker.FAILURE_HOLD_SECONDS
    )


def test_self_delete_uses_the_persisted_binding_and_records_the_request(tmp_path):
    run_root = tmp_path / "storage/automation/run-1/a"
    run_root.mkdir(parents=True)
    state_path = run_root / "bootstrap-state.json"
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-1",
        "shard_id": "a",
        "status": "failed",
        "error": "BackupError: evaluation backend did not load",
        "delete_requested": False,
        "instance_id": 42,
        "instance_hash_id": "hash-42",
        "instance_notes": "training-run:run-1;shard:a",
    }), encoding="utf-8")
    seen = {}

    class Pod:
        def __init__(self, token, **kwargs):
            seen["token"] = token

        def instance(self, instance_id):
            seen["verified"] = instance_id
            return {
                "id": instance_id, "hashId": "hash-42",
                "notes": "training-run:run-1;shard:a",
            }

        def delete(self, instance_id):
            seen["state_at_delete"] = json.loads(state_path.read_text(encoding="utf-8"))
            return "accepted"

    import training_automation.lifecycle as lifecycle

    original = lifecycle.SimplePodClient
    lifecycle.SimplePodClient = Pod
    try:
        outcome = worker.release_instance({
            **worker_env(tmp_path), "SIMPLEPOD_API_TOKEN": "token-value",
        })
    finally:
        lifecycle.SimplePodClient = original

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert outcome == "accepted"
    assert seen["token"] == "token-value"
    assert seen["verified"] == 42
    # The request is durable before it is issued, and the failure stays readable.
    assert seen["state_at_delete"]["delete_requested"] is True
    assert persisted["status"] == "failed"
    assert persisted["error"] == "BackupError: evaluation backend did not load"
    assert persisted["delete_requested_by"] == "diagnostic-hold-deadline"


def test_self_delete_refuses_an_instance_it_cannot_identify(tmp_path):
    run_root = tmp_path / "storage/automation/run-1/a"
    run_root.mkdir(parents=True)
    (run_root / "bootstrap-state.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "run-1", "shard_id": "a",
        "status": "failed", "delete_requested": False,
    }), encoding="utf-8")

    # Not retryable: waiting never produces a binding this container never had,
    # so the outer self-delete loop must fail loudly at the first attempt.
    with pytest.raises(ValueError, match="no verified instance binding") as raised:
        worker.release_instance({
            **worker_env(tmp_path), "SIMPLEPOD_API_TOKEN": "token-value",
        })
    assert worker._delete_is_retryable(raised.value) is False
    assert worker._delete_is_retryable(BackupError("delete is unconfirmed")) is True


def test_default_hold_waits_the_real_limit_and_retries_an_unconfirmed_delete(tmp_path):
    """Cover the branch production takes: `hold` is omitted, as in main().

    Every other worker test injects `hold`, so the wiring between the configured
    limit and the actual waiting is only exercised here. It also covers the
    outer self-delete budget: the inner budget can end unconfirmed during a
    provider outage, and the instance must not keep billing because of it.
    """
    clock = [500.0]
    slept = []
    events = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    attempts = []

    def delete_instance(env):
        attempts.append(sum(slept))
        events.append(f"delete-{len(attempts)}")
        if len(attempts) < 3:
            raise BackupError("SimplePod delete of instance 42 is unconfirmed")
        return "accepted"

    console = io.StringIO()
    result = run_worker(
        env={**worker_env(tmp_path), "TRAINING_FAILURE_HOLD_SECONDS": "120"},
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        sleep=sleep,
        monotonic=lambda: clock[0],
        now=lambda: 0.0,
        delete_instance=delete_instance,
        self_delete_attempts=3,
        self_delete_retry_seconds=90,
        console=console,
    )

    printed = console.getvalue()
    assert result == 9
    # First the configured limit, then one bounded re-hold per unconfirmed
    # attempt, and every delete happens strictly after its own hold.
    assert sum(slept) == 120 + 90 + 90
    assert attempts == [120, 210, 300]
    assert "self-delete attempt 1/3" in printed
    assert "self-delete 1/3 did not confirm" in printed
    assert "self-delete attempt 3/3" in printed
    assert "self-delete outcome: accepted" in printed
    assert "manual cleanup is required" not in printed


def test_exhausted_self_delete_budget_stops_instead_of_retrying_forever(tmp_path):
    clock = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    calls = []

    def delete_instance(env):
        calls.append(True)
        raise BackupError("SimplePod delete of instance 42 is unconfirmed")

    console = io.StringIO()
    run_worker(
        env={**worker_env(tmp_path), "TRAINING_FAILURE_HOLD_SECONDS": "60"},
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        sleep=sleep,
        monotonic=lambda: clock[0],
        delete_instance=delete_instance,
        self_delete_attempts=2,
        self_delete_retry_seconds=30,
        console=console,
    )

    assert len(calls) == 2
    assert sum(slept) == 60 + 30
    assert "manual cleanup is required" in console.getvalue()


def test_unretryable_self_delete_failure_is_not_re_held(tmp_path):
    held = []
    calls = []

    def delete_instance(env):
        calls.append(True)
        raise ValueError("this container has no verified instance binding to delete")

    console = io.StringIO()
    run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        hold=lambda: held.append(True),
        delete_instance=delete_instance,
        console=console,
    )

    assert calls == [True]
    assert held == [True]
    assert "manual cleanup is required" in console.getvalue()


class LiveChild:
    """A child whose pipe dies while the process itself keeps running."""

    def __init__(self, *, stops_on_terminate=True):
        self.stops_on_terminate = stops_on_terminate
        self.signals = []
        self.alive = True

    @property
    def stdout(self):
        raise OSError("read pipe collapsed")

    def poll(self):
        return None if self.alive else 143

    def terminate(self):
        self.signals.append("terminate")
        if self.stops_on_terminate:
            self.alive = False

    def kill(self):
        self.signals.append("kill")
        self.alive = False

    def wait(self, timeout=None):
        if self.alive:
            raise TimeoutError("child did not stop")
        return 143


def test_pipe_failure_terminates_the_child_before_any_deletion(tmp_path):
    child = LiveChild()
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: child,
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    assert result == 127
    # Nothing is signalled when the pipe breaks: a transient read error must not
    # destroy an otherwise healthy shard. The termination happens at the
    # deadline, after the child has had the whole hold.
    assert child.signals == ["terminate"]
    assert child.alive is False
    assert events == ["hold", "delete"]
    assert "terminating the supervised child" in console.getvalue()


def test_a_child_that_will_not_die_blocks_the_deletion(tmp_path):
    class Unkillable(LiveChild):
        def kill(self):
            self.signals.append("kill")

    child = Unkillable(stops_on_terminate=False)
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: child,
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    assert result == 127
    assert child.signals == ["terminate", "kill"]
    # The child keeps the whole hold; only at the deadline does the supervisor
    # insist on its death, and evidence outranks billing when it cannot get it:
    # an instance whose child may still be publishing is never deleted.
    assert events == ["hold"]
    assert "refuses to delete this instance while a supervised child" in printed
    assert "manual cleanup is required" in printed


GRANDCHILD = (
    "import sys, time\n"
    "target = sys.argv[1]\n"
    "while True:\n"
    "    with open(target, 'a') as handle:\n"
    "        handle.write('x')\n"
    "    time.sleep(0.02)\n"
)

SUPERVISED_CHILD = (
    "import subprocess, sys, time\n"
    "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])\n"
    "time.sleep(600)\n"
)


def _grew(path: Path, *, seconds: float) -> bool:
    before = path.stat().st_size
    time.sleep(seconds)
    return path.stat().st_size > before


def test_group_termination_reaches_the_grandchild_that_writes_evidence(tmp_path):
    """The queue runs the trainer as its own subprocess (queue.py:378-380).

    Signalling the supervised pid alone kills the queue and orphans that
    trainer, which keeps writing while the guard reports success. The guard must
    address the process group, and establish death by observing the group.
    """
    marker = tmp_path / "evidence.txt"
    marker.write_text("", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", SUPERVISED_CHILD, str(marker), GRANDCHILD],
        start_new_session=True,
    )
    pgid = worker.child_process_group(process)
    notes = []
    try:
        deadline = time.monotonic() + 10
        while marker.stat().st_size == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.stat().st_size > 0, "the grandchild never started writing"
        assert _grew(marker, seconds=0.2), "the grandchild is not writing"

        assert worker._ensure_child_stopped(process, pgid=pgid, emit=notes.append) is True

        assert not _grew(marker, seconds=0.5), "the grandchild survived the guard"
        assert worker._group_is_gone(pgid) is True
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            pass


def test_supervised_child_starts_its_own_session(tmp_path):
    seen = {}

    def popen(command, **kwargs):
        seen.update(kwargs)
        return FakeProcess("done\n", 0)

    run_worker(env=worker_env(tmp_path), popen=popen, console=io.StringIO())

    # Without this the child shares the supervisor's group and no group signal
    # can reach the trainer without also hitting this process.
    assert seen["start_new_session"] is True


class RelaunchChild:
    """A first attempt whose pipe collapses while the process stays alive."""

    def __init__(self):
        self.signals = []
        self.pid = None

    @property
    def stdout(self):
        raise OSError("read pipe collapsed")

    def poll(self):
        return None

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")

    def wait(self, timeout=None):
        raise TimeoutError("child did not stop")


def test_recovery_relaunch_never_clears_an_earlier_live_child(tmp_path):
    """The reset WAVE-008 describes: a live first child, then an eligible relaunch.

    `queue-state.json` shows every job completed, so the supervisor relaunches
    over the still-live child. The second attempt exiting says nothing about the
    first, and the instance must not be deleted on its word.
    """
    run_root = tmp_path / "storage/automation/run-1/a"
    run_root.mkdir(parents=True)
    (run_root / "bootstrap-state.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "run-1", "shard_id": "a",
        "status": "failed", "delete_requested": False,
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

    live = RelaunchChild()
    started = []

    def popen(command, **kwargs):
        started.append(command)
        if len(started) == 1:
            return live
        return FakeProcess("archive failed again\n", 9)

    events = []
    console = io.StringIO()
    result = run_worker(
        env=worker_env(tmp_path),
        popen=popen,
        hold=lambda: events.append("hold"),
        sleep=lambda _: None,
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    # Initial attempt plus the archive-only recovery budget: every relaunch here
    # happens over the still-live first child.
    assert len(started) >= 2, "the recovery relaunch this finding needs did not happen"
    assert live.signals == ["terminate", "kill"]
    assert result == 9
    assert events == ["hold"]
    assert "refuses to delete this instance while a supervised child" in console.getvalue()


def test_retry_classification_separates_transient_from_permanent():
    from training_automation.lifecycle import InstanceIdentityError, PermanentApiError

    retryable = [
        BackupError("SimplePod delete of instance 42 is unconfirmed after 4 attempts"),
        BackupError("SimplePod GET /instances/42 failed with HTTP 503"),
        OSError("the storage mount is temporarily unreachable"),
    ]
    permanent = [
        PermanentApiError("SimplePod delete of instance 42 was refused with HTTP 401"),
        InstanceIdentityError("SimplePod response hashId does not match the explicit binding"),
        FileNotFoundError("bootstrap-state.json"),
        ValueError("this container has no verified instance binding to delete"),
    ]
    assert [worker._delete_is_retryable(exc) for exc in retryable] == [True, True, True]
    assert [worker._delete_is_retryable(exc) for exc in permanent] == [False] * 4


def test_a_transient_state_read_failure_is_re_held(tmp_path):
    calls = []

    def delete_instance(env):
        calls.append(True)
        if len(calls) < 2:
            raise OSError("the storage mount is temporarily unreachable")
        return "accepted"

    console = io.StringIO()
    run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        hold=lambda: None,
        delete_instance=delete_instance,
        console=console,
    )

    assert len(calls) == 2
    assert "self-delete outcome: accepted" in console.getvalue()
    assert "manual cleanup is required" not in console.getvalue()
