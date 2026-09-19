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
    """A child whose pipe dies while the process itself keeps training.

    The production shape of 2026-09-18: the supervisor can no longer read the
    output, the trainer on the GPU is untouched by that and goes on to its own
    ending. `polls` is how many times it answers "still running" before it
    reaches that ending.
    """

    def __init__(self, *, exit_code=0, polls=2):
        self.exit_code = exit_code
        self.polls = polls
        self.signals = []
        self.alive = True

    @property
    def stdout(self):
        raise OSError("read pipe collapsed")

    def poll(self):
        return None if self.alive else self.exit_code

    def terminate(self):
        self.signals.append("terminate")
        self.alive = False

    def kill(self):
        self.signals.append("kill")
        self.alive = False

    def wait(self, timeout=None):
        if self.polls > 0:
            self.polls -= 1
            raise TimeoutError("the child is still running")
        self.alive = False
        return self.exit_code


def test_a_lost_pipe_over_a_live_child_never_reports_it_dead(tmp_path):
    """The loss this batch exists for: 1.7 USD and two hours of training.

    The supervisor lost the output of a healthy child, called it exit=127,
    scheduled the hold and deleted the instance nine steps before the first
    checkpoint. Losing the pipe is a fact about the supervisor; the child's
    exit status is the only fact about the child.
    """
    child = LiveChild(exit_code=0, polls=2)
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
    assert result == 0
    # No hold, no deletion, and nothing signalled at any point: the child was
    # allowed to finish and it did.
    assert events == []
    assert child.signals == []
    assert "lost the supervised child's output stream" in printed
    assert "the child is still running" in printed
    assert "exit=127" not in printed
    assert "parallel worker initial exit=0" in printed


def test_a_lost_pipe_over_a_child_that_then_fails_still_holds_and_deletes(tmp_path):
    """A genuine failure is still a failure, even when the pipe broke first."""
    child = LiveChild(exit_code=7, polls=1)
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
    assert result == 7
    assert events == ["hold", "delete"]
    assert "real exit status after losing its output: exit=7" in printed
    assert "parallel worker initial exit=7" in printed


def test_a_child_that_cannot_be_questioned_holds_nothing_and_deletes_nothing(tmp_path):
    """No output and no answerable exit status: the fate stays unknown.

    Unknown is not failure. The supervisor says so with its own status instead
    of borrowing 127 from a child it never saw end, and it schedules neither
    the hold nor the deletion that follows it.
    """

    class SilentChild:
        stdout = None

        def poll(self):
            return None

    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: SilentChild(),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    assert result == worker.CAPTURE_UNRESOLVED_EXIT
    assert result != 127
    assert events == []
    assert "could not establish how the supervised child ended" in printed
    assert "holds nothing and deletes nothing" in printed
    # WAVE-013: the console says what actually happens next. `main()` returns
    # here, which ends the container command; the child is not protected by
    # this supervisor staying alive, because it does not stay alive.
    assert "ends the container command" in printed
    assert "the instance stays up" not in printed


class BrokenSink:
    """A sink that fails the way a full volume or a dead consumer fails."""

    def __init__(self, error=OSError(28, "No space left on device")):
        self.error = error
        self.writes = 0

    def write(self, message):
        self.writes += 1
        raise self.error

    def flush(self):
        raise self.error


def test_a_broken_console_never_ends_the_container(tmp_path):
    """WAVE-011. The supervisor is the container command (`exec python -m ...`).

    An exception escaping the capture path ends the container and takes a
    still-training child with it, before the hold and the self-delete that
    exist for exactly this failure can run. A console that cannot be written
    is not a reason to do that, and the durable log must survive it.
    """
    console = BrokenSink()
    events = []

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert result == 9
    assert console.writes > 0, "the console this test breaks was never used"
    # The failed shard still reaches its hold and its self-delete.
    assert events == ["hold", "delete"]
    # Losing one sink never costs the line in the other.
    assert "trainer exploded" in log
    assert "parallel worker initial exit=9" in log
    assert "self-delete outcome: accepted" in log


def test_a_broken_console_over_a_live_child_concludes_nothing(tmp_path):
    """The reproduction, at the call site that repeats forever.

    The per-poll notice of the wait loop is emitted for as long as the child
    lives, so a sink that fails there would raise out of `run_worker` on the
    one path built to keep a healthy child alive.
    """
    child = LiveChild(exit_code=0, polls=2)
    console = BrokenSink()
    events = []

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: child,
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert result == 0
    assert events == []
    assert child.signals == []
    assert "the child is still running" in log


def test_a_broken_durable_log_never_ends_the_container(tmp_path, monkeypatch):
    """The mirror image: the volume is gone, the console is all there is."""
    handles = []

    class Handle(BrokenSink):
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def close(self):
            return None

    original = Path.open

    def open_handle(self, *args, **kwargs):
        if self.name == "worker.log":
            handles.append(Handle())
            return handles[-1]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_handle)

    console = io.StringIO()
    events = []
    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("training complete\n", 0),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    assert result == 0
    assert events == []
    assert "training complete" in printed
    assert "parallel worker initial exit=0" in printed
    assert handles and handles[0].writes > 0, "the log this test breaks was never used"


class UnanswerableChild:
    """A child this supervisor cannot question: no output, no exit status."""

    stdout = None

    def poll(self):
        return None


def _completed_queue_state(tmp_path):
    """The state that makes an archive-only relaunch eligible."""
    run_root = tmp_path / "storage/automation/run-1/a"
    run_root.mkdir(parents=True, exist_ok=True)
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
    return run_root


def test_an_unobserved_child_is_never_relaunched_over(tmp_path):
    """WAVE-012. Eligibility is not the thing that stops this relaunch.

    With a completed `queue-state.json` the archive-only recovery would fire,
    and a second trainer would start on a GPU where the first may still be
    running. The supervisor stops because it does not know how the first child
    ended, not because the queue said no.
    """
    run_root = _completed_queue_state(tmp_path)
    started = []

    def popen(command, **kwargs):
        started.append(command)
        return UnanswerableChild()

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

    assert worker._recovery_eligible(worker_env(tmp_path))[0] is True, (
        "this test must run with a relaunch that would otherwise be eligible"
    )
    assert len(started) == 1, "a second child was started over an unobserved one"
    assert result == worker.CAPTURE_UNRESOLVED_EXIT
    assert events == []
    # No terminal state is written on a guess either.
    assert not (run_root / "worker-recovery-state.json").exists()
    assert "could not establish how the supervised child ended" in console.getvalue()


class FlakyStream:
    """A stream whose read fails, then delivers the rest of the output."""

    def __init__(self, *, failures):
        self.failures = failures
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        if self.reads <= self.failures:
            def interrupted():
                yield f"step {self.reads} of training\n"
                raise OSError("read interrupted")

            return interrupted()
        return iter(["training complete\n"])


class StreamProcess:
    def __init__(self, stream, exit_code):
        self.stdout = stream
        self.exit_code = exit_code

    def wait(self, timeout=None):
        return self.exit_code


def test_a_failed_read_is_retried_instead_of_abandoning_the_pipe(tmp_path):
    """WAVE-012. A pipe nobody drains eventually blocks the child writing it."""
    stream = FlakyStream(failures=1)
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: StreamProcess(stream, 0),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert result == 0
    assert events == []
    assert stream.reads == 2
    assert "retrying that read 1/2" in log
    # The read recovered, so the rest of the child's output is still evidence
    # and the supervisor never lost the stream at all.
    assert "training complete" in log
    assert "lost the supervised child's output stream" not in log


def test_the_read_retry_budget_is_bounded_and_exact(tmp_path):
    """The other side of the same counter: it retries twice, then stops."""
    stream = FlakyStream(failures=worker.CAPTURE_DRAIN_ATTEMPTS)
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: StreamProcess(stream, 0),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert stream.reads == worker.CAPTURE_DRAIN_ATTEMPTS == 3
    assert "retrying that read 1/2" in log
    assert "retrying that read 2/2" in log
    assert "retrying that read 3/2" not in log
    # Out of retries, the supervisor stops reading and asks the child instead;
    # it does not invent a status for it.
    assert "lost the supervised child's output stream" in log
    assert result == 0
    assert events == []


def test_the_stop_guard_still_reports_a_child_object_it_cannot_stop():
    """The `klein-stop-determinism` guard itself, on a process with no group."""

    class Unkillable:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

        def wait(self, timeout=None):
            raise TimeoutError("child did not stop")

    class Surviving(Unkillable):
        def wait(self, timeout=None):
            return 0

    child = Unkillable()
    notes = []
    stopped = worker._ensure_child_stopped(child, pgid=None, emit=notes.append)

    printed = "".join(notes)
    assert stopped is False
    assert child.signals == ["terminate", "kill"]
    assert "terminating the supervised child" in printed
    assert "could not confirm the supervised child stopped" in printed

    survivor = Surviving()
    notes = []
    stopped = worker._ensure_child_stopped(survivor, pgid=None, emit=notes.append)

    assert stopped is False
    assert "still sees a live supervised child" in "".join(notes)


def test_a_child_that_will_not_die_blocks_the_deletion(tmp_path, monkeypatch):
    """A failed shard whose group cannot be confirmed empty is never deleted.

    Evidence outranks billing: an instance whose trainer may still be
    publishing truncates work that costs GPU hours to reproduce.
    """
    visited = []

    monkeypatch.setattr(worker, "child_process_group", lambda process: 4242)
    monkeypatch.setattr(
        worker,
        "_ensure_child_stopped",
        lambda process, **kwargs: visited.append(process) or False,
    )

    events = []
    console = io.StringIO()
    result = run_worker(
        env=worker_env(tmp_path),
        popen=lambda *args, **kwargs: FakeProcess("trainer exploded\n", 9),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    assert result == 9
    assert len(visited) == 1
    # The child keeps the whole hold; only at the deadline does the supervisor
    # insist on its death, and it refuses to delete when it cannot get it.
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


def test_the_child_stream_is_decoded_leniently(tmp_path):
    """Strict decoding makes an ordinary progress bar able to raise."""
    seen = {}

    def popen(command, **kwargs):
        seen.update(kwargs)
        return FakeProcess("done\n", 0)

    run_worker(env=worker_env(tmp_path), popen=popen, console=io.StringIO())

    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"


PROGRESS_BAR_CHILD = (
    "import sys\n"
    "out = sys.stdout.buffer\n"
    "out.write(('Downloading klein-9b: |' + '\\u2588' * 24 + '| 61%').encode('utf-8'))\n"
    "out.write(b'\\n')\n"
    "out.write(b'tqdm fragment: \\xe2\\x96 and a lone \\xff byte\\n')\n"
    "out.flush()\n"
    "print('training complete')\n"
)

FAILING_CHILD = (
    "import sys\n"
    "print('trainer exploded')\n"
    "sys.exit(9)\n"
)


def real_child(script: str):
    """Run a real child through exactly the kwargs the supervisor passes.

    Only `cwd` and `env` are dropped: the image path does not exist on a test
    machine, and the child needs nothing from the shard environment. Everything
    that decides how the stream is read is left exactly as production sets it.
    """

    def popen(command, **kwargs):
        kwargs.pop("cwd", None)
        kwargs.pop("env", None)
        return subprocess.Popen([sys.executable, "-c", script], **kwargs)

    return popen


def test_undecodable_progress_bytes_are_not_a_failed_shard(tmp_path):
    """The mechanism behind the loss, on a real pipe.

    The trigger in production was a multi-byte block glyph split across a read
    boundary, which is a race; a lone invalid byte is its deterministic
    equivalent. Under strict decoding either one raises inside the capture loop
    and used to be reported as the child's own exit status.
    """
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=real_child(PROGRESS_BAR_CHILD),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert result == 0
    assert events == []
    assert "lost the supervised child's output stream" not in printed
    assert "Downloading klein-9b:" in log
    assert "training complete" in log
    assert "parallel worker initial exit=0" in log


def test_a_real_failed_child_still_holds_terminates_its_group_and_deletes(tmp_path):
    """Everything `klein-stop-determinism` delivered, on a real process group."""
    events = []
    console = io.StringIO()

    result = run_worker(
        env=worker_env(tmp_path),
        popen=real_child(FAILING_CHILD),
        hold=lambda: events.append("hold"),
        delete_instance=lambda env: events.append("delete") or "accepted",
        console=console,
    )

    printed = console.getvalue()
    log = (tmp_path / "storage/automation/run-1/a/worker.log").read_text(encoding="utf-8")
    assert result == 9
    assert events == ["hold", "delete"]
    assert "trainer exploded" in log
    assert "parallel worker initial exit=9" in log
    assert "refuses to delete this instance" not in printed


def test_recovery_relaunch_never_clears_an_earlier_child(tmp_path, monkeypatch):
    """The reset WAVE-008 describes, as it can still happen.

    A relaunch over a *live* child is now impossible: a lost pipe is followed to
    the child's real exit status before anything else happens. What survives is
    a first child that really exited while the trainer it started keeps writing
    in its group. The second attempt exiting says nothing about the first, so
    the guard runs over every child this supervisor ever started, and the
    instance is not deleted on the last one's word.
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

    first = LiveChild(exit_code=9, polls=1)
    started = []

    def popen(command, **kwargs):
        started.append(command)
        if len(started) == 1:
            return first
        return FakeProcess("archive failed again\n", 9)

    visited = []
    monkeypatch.setattr(worker, "child_process_group", lambda process: 4242)
    monkeypatch.setattr(
        worker,
        "_ensure_child_stopped",
        lambda process, **kwargs: visited.append(process) or process is not first,
    )

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

    printed = console.getvalue()
    # Initial attempt plus the archive-only recovery budget.
    assert len(started) >= 2, "the recovery relaunch this finding needs did not happen"
    # The relaunch waits for the first child's real exit status; it never runs
    # over a child that is still going.
    assert printed.index("real exit status after losing its output: exit=9") < printed.index(
        "starting recovery-1 supervised queue attempt"
    )
    assert first.signals == []
    # The guard visits every child, not just the last one to exit.
    assert first in visited and len(visited) >= 2
    assert result == 9
    assert events == ["hold"]
    assert "refuses to delete this instance while a supervised child" in printed


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
