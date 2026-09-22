from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .state import atomic_write_json


COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FINALIZATION_RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_SECONDS = (5, 15, 30)
# A held instance keeps billing. The diagnostic value of the held container is
# real but finite, so the hold is a deadline, not a state: conservative enough
# that a human woken by the alert still finds the console, bounded so that a
# human who is not woken does not pay for days.
FAILURE_HOLD_SECONDS = 6 * 3600
FAILURE_HOLD_ENV = "TRAINING_FAILURE_HOLD_SECONDS"
HOLD_POLL_SECONDS = 60
# The deletion itself already retries inside its own budget; this is the outer
# budget for a provider that is still unreachable when that one ends. Bounded on
# both sides: a transient outage must not strand the instance, and a permanent
# one must not turn into an infinite retry.
SELF_DELETE_ATTEMPTS = 3
SELF_DELETE_RETRY_SECONDS = 1800
CHILD_TERMINATION_SECONDS = 30
CHILD_TERMINATION_POLL_SECONDS = 0.2
# How often a supervisor that lost the child's output asks the child itself what
# it is doing. Only an exit status ends a shard, so this is a question, not a
# deadline: a child that is still running is still running.
CHILD_STATUS_POLL_SECONDS = 60
# A read that failed once may not have failed for good, and a pipe nobody drains
# eventually blocks the writer. The drain is retried a bounded number of times
# before the supervisor stops reading and only waits.
CAPTURE_DRAIN_ATTEMPTS = 3
# 127 is a statement about the child: it means the child ended that way. A
# supervisor that never saw the child end says so with its own status instead of
# borrowing the child's.
CAPTURE_UNRESOLVED_EXIT = 125


def _component(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name, ""))
    if not COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{name} must be one safe path component")
    return value


def worker_log_path(env: Mapping[str, str]) -> Path:
    storage_root = Path(env.get("TRAINING_STORAGE_ROOT", "/storage")).resolve()
    return (
        storage_root
        / "automation"
        / _component(env, "TRAINING_RUN_ID")
        / _component(env, "TRAINING_SHARD_ID")
        / "worker.log"
    )


def failure_hold_seconds(env: Mapping[str, str]) -> float:
    """Return the configured diagnostic hold, defaulting conservatively."""
    raw = str(env.get(FAILURE_HOLD_ENV, "")).strip()
    if not raw:
        return float(FAILURE_HOLD_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(FAILURE_HOLD_SECONDS)
    return max(0.0, value)


def hold_for(
    seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Hold this instance until the deadline and not one moment longer."""
    deadline = monotonic() + max(0.0, float(seconds))
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        sleep(min(float(HOLD_POLL_SECONDS), remaining))


def _delete_is_retryable(exc: BaseException) -> bool:
    """Retry what time can change; give up at once on what it cannot.

    An unconfirmed provider answer, a transport failure or a 5xx is exactly what
    a second hold is for, and so is a local mount that refused one read of the
    persisted binding. Revoked credentials, a refused method and an instance
    whose identity does not match this process are permanent by construction:
    spending the whole outer budget on them is spending money to learn nothing.
    A binding that was never persisted is permanent in the same way.
    """
    from .backup import BackupError
    from .lifecycle import InstanceIdentityError, PermanentApiError

    if isinstance(exc, (PermanentApiError, InstanceIdentityError)):
        return False
    if isinstance(exc, BackupError):
        return True
    if isinstance(exc, FileNotFoundError):
        return False
    if isinstance(exc, OSError):
        return True
    return False


def child_process_group(process: Any) -> int | None:
    """Return the process group of a real supervised child, if it has one.

    The child is started with `start_new_session=True`, so it leads its own
    group and the trainer it runs as its own subprocess inherits that group.
    Addressing the group is the difference between signalling the supervisor
    and signalling the process that is actually writing evidence.
    """
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def _group_is_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except Exception:
        # Existing but unsignalable is not gone.
        return False
    return False


def _reap(process: Any, wait_seconds: float) -> None:
    wait = getattr(process, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=wait_seconds)
    except Exception:
        pass


def _await_group_exit(
    pgid: int,
    *,
    process: Any,
    seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> bool:
    deadline = monotonic() + max(0.0, float(seconds))
    while True:
        # The direct child must be reaped or its zombie keeps the group alive.
        _reap(process, 0.1)
        if _group_is_gone(pgid):
            return True
        if monotonic() >= deadline:
            return False
        sleep(CHILD_TERMINATION_POLL_SECONDS)


def _stop_process_object(process: Any, *, emit: Callable[[str], None], wait_seconds: float) -> bool:
    """Fallback for a child with no addressable group."""
    poll = getattr(process, "poll", None)
    terminate = getattr(process, "terminate", None)
    kill = getattr(process, "kill", None)
    if poll is None or terminate is None or kill is None:
        emit(
            "parallel worker cannot inspect the supervised child, so it will not "
            "delete this instance\n"
        )
        return False
    try:
        if poll() is not None:
            return True
        emit("parallel worker terminating the supervised child before any deletion\n")
        terminate()
        try:
            process.wait(timeout=wait_seconds)
        except Exception:
            kill()
            process.wait(timeout=wait_seconds)
        stopped = poll() is not None
    except Exception as exc:
        emit(
            "parallel worker could not confirm the supervised child stopped "
            f"({type(exc).__name__}: {exc})\n"
        )
        return False
    if not stopped:
        emit("parallel worker still sees a live supervised child\n")
    return stopped


def _ensure_child_stopped(
    process: Any,
    *,
    pgid: int | None = None,
    emit: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wait_seconds: float = CHILD_TERMINATION_SECONDS,
) -> bool:
    """Return True only when nothing this supervisor started can still write.

    Signalling the supervised pid alone is not enough and is worse than no
    guard at all: the queue process runs the trainer as its own subprocess, so
    SIGTERM to the leader leaves an orphaned trainer writing into the same
    output while the guard reports success. The whole group is signalled, and
    death is established by observing that the group is gone, not by reading
    the leader's return code.
    """
    if process is None:
        return True
    if pgid is None:
        return _stop_process_object(process, emit=emit, wait_seconds=wait_seconds)
    if _await_group_exit(
        pgid, process=process, seconds=0, sleep=sleep, monotonic=monotonic
    ):
        return True
    emit(
        f"parallel worker terminating supervised process group {pgid} "
        "before any deletion\n"
    )
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception as exc:
        emit(
            f"parallel worker could not signal process group {pgid} "
            f"({type(exc).__name__}: {exc})\n"
        )
        return False
    if _await_group_exit(
        pgid, process=process, seconds=wait_seconds / 2, sleep=sleep, monotonic=monotonic
    ):
        return True
    emit(f"parallel worker killing supervised process group {pgid}\n")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception as exc:
        emit(
            f"parallel worker could not kill process group {pgid} "
            f"({type(exc).__name__}: {exc})\n"
        )
        return False
    stopped = _await_group_exit(
        pgid, process=process, seconds=wait_seconds / 2, sleep=sleep, monotonic=monotonic
    )
    if not stopped:
        emit(f"parallel worker still sees a live process in group {pgid}\n")
    return stopped


def _drain_output(stream: Any, emit: Callable[[str], None]) -> str | None:
    """Copy the child's output through, or describe why it could not be read."""
    try:
        for line in stream:
            emit(line)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _capture_output(
    stream: Any,
    emit: Callable[[str], None],
    *,
    attempts: int = CAPTURE_DRAIN_ATTEMPTS,
) -> str | None:
    """Follow the child's output to its end, retrying a failed read.

    The child is decoded leniently, so ordinary progress-bar bytes cannot raise
    here at all: a tqdm block glyph split across a read boundary is exactly the
    stream this supervisor is built to carry. What remains are real read
    failures, and those are retried, because abandoning a pipe the child is
    still writing to eventually blocks the child itself.
    """
    budget = max(1, int(attempts))
    failure: str | None = None
    for attempt in range(1, budget + 1):
        failure = _drain_output(stream, emit)
        if failure is None:
            return None
        if attempt < budget:
            emit(
                "parallel worker could not read the supervised child's output "
                f"({failure}); retrying that read {attempt}/{budget - 1}\n"
            )
    return failure


def _observe_child_exit(
    process: Any,
    *,
    emit: Callable[[str], None],
    poll_seconds: float = CHILD_STATUS_POLL_SECONDS,
) -> int | None:
    """Return the child's real exit status, or None when it cannot be observed.

    Losing the output stream says nothing about the process that was writing
    into it. The supervisor therefore asks the child itself, and keeps asking
    for as long as the child is alive: an exit status is the only evidence
    allowed to end a shard. Waiting on a child that never exits is the same
    unbounded wait a readable pipe already implies for a hung trainer; what is
    not allowed is converting a living child into a failure.
    """
    wait = getattr(process, "wait", None)
    if wait is None:
        return None
    timed = True
    while True:
        try:
            status = wait(timeout=max(0.0, float(poll_seconds))) if timed else wait()
        except (subprocess.TimeoutExpired, TimeoutError):
            emit(
                "parallel worker has lost the supervised child's output and the "
                "child is still running; no failure is concluded and nothing is "
                "scheduled against it\n"
            )
            continue
        except TypeError:
            if not timed:
                return None
            # A child object whose wait() takes no timeout: ask it the blocking
            # way rather than inventing a status on its behalf.
            timed = False
            continue
        except Exception as exc:
            emit(
                "parallel worker could not read the supervised child's exit status "
                f"({type(exc).__name__}: {exc})\n"
            )
            return None
        try:
            return int(status)
        except (TypeError, ValueError):
            return None


def _deadline_text(now: Callable[[], float], seconds: float) -> str:
    return datetime.fromtimestamp(now() + seconds, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def release_instance(env: Mapping[str, str]) -> str:
    """Delete this instance through the same lifecycle path as the success route.

    The identity comes from the binding the bootstrap verified and persisted
    before any paid work; nothing is deleted on an identity this worker cannot
    check against the provider itself. The delete request is recorded durably
    before it is issued, exactly as the success route records it.
    """
    from .lifecycle import InstanceBinding, SimplePodClient, delete_verified_instance

    token = str(env.get("SIMPLEPOD_API_TOKEN", ""))
    if not token:
        # A precondition that no amount of waiting can satisfy, answered before
        # any read whose failure would deserve another hold.
        raise ValueError("SIMPLEPOD_API_TOKEN is unset, so this container cannot delete itself")
    run_id = _component(env, "TRAINING_RUN_ID")
    shard_id = _component(env, "TRAINING_SHARD_ID")
    state_path = worker_log_path(env).with_name("bootstrap-state.json")
    state = _load_mapping(state_path)
    if (
        state.get("schema_version") != 1
        or state.get("run_id") != run_id
        or state.get("shard_id") != shard_id
    ):
        raise ValueError("bootstrap state identity does not match this worker")
    try:
        binding = InstanceBinding.from_document({
            "schema_version": 1,
            "run_id": run_id,
            "shard_id": shard_id,
            "instance_id": state.get("instance_id", 0),
            "instance_hash_id": state.get("instance_hash_id", ""),
            "instance_notes": state.get("instance_notes", ""),
        })
    except Exception as exc:
        # Not retryable: no amount of waiting creates a binding this container
        # never received.
        raise ValueError(
            f"this container has no verified instance binding to delete: {exc}"
        ) from None
    client = SimplePodClient(token)

    def record_delete_request() -> None:
        # The failure itself stays readable in "status"; only the delete
        # request is added, so a container restart still refuses to resume on
        # an outcome it cannot know.
        state["delete_requested"] = True
        state["delete_requested_by"] = "diagnostic-hold-deadline"
        atomic_write_json(state_path, state)

    return delete_verified_instance(client, binding, before_request=record_delete_request)


def _load_mapping(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return document


def _recovery_eligible(env: Mapping[str, str]) -> tuple[bool, str]:
    log_path = worker_log_path(env)
    run_id = _component(env, "TRAINING_RUN_ID")
    shard_id = _component(env, "TRAINING_SHARD_ID")
    try:
        bootstrap = _load_mapping(log_path.with_name("bootstrap-state.json"))
        queue = _load_mapping(log_path.with_name("queue-state.json"))
    except Exception as exc:
        return False, f"durable completion state unavailable: {type(exc).__name__}: {exc}"
    if (
        bootstrap.get("schema_version") != 1
        or bootstrap.get("run_id") != run_id
        or bootstrap.get("shard_id") != shard_id
    ):
        return False, "bootstrap state identity does not match this worker"
    if bootstrap.get("delete_requested"):
        return False, "a prior delete request has an uncertain outcome"
    jobs = queue.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        return False, "persisted queue has no completed jobs"
    if any(
        not isinstance(item, Mapping)
        or item.get("status") != "completed"
        or item.get("training_status") != "completed"
        or item.get("evaluation_status") != "completed"
        for item in jobs.values()
    ):
        return False, "training or evaluation is incomplete"
    return True, "completed queue is eligible for archive-only finalization recovery"


def run_worker(
    *,
    env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    hold: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], float] = time.time,
    delete_instance: Callable[[Mapping[str, str]], str] = release_instance,
    self_delete_attempts: int = SELF_DELETE_ATTEMPTS,
    self_delete_retry_seconds: float = SELF_DELETE_RETRY_SECONDS,
    recovery_attempts: int = FINALIZATION_RECOVERY_ATTEMPTS,
    console: TextIO | None = None,
) -> int:
    values = dict(os.environ if env is None else env)
    output = console or sys.stdout
    limit = failure_hold_seconds(values)
    attempts_allowed = max(1, int(self_delete_attempts))
    # Every child this supervisor ever started, with the group it leads. The
    # guard runs over all of them: a relaunch after a lost pipe can leave an
    # earlier child alive, and the later child exiting says nothing about it.
    children: list[dict[str, Any]] = []

    def every_child_stopped(emit: Callable[[str], None]) -> bool:
        stopped = True
        for child in children:
            if child["pgid"] is None and child["reaped"]:
                # No group to address and its exit status was already collected:
                # this one is observably finished.
                continue
            if not _ensure_child_stopped(
                child["process"],
                pgid=child["pgid"],
                emit=emit,
                sleep=sleep,
                monotonic=monotonic,
            ):
                stopped = False
        return stopped

    def wait_out(seconds: float) -> None:
        if hold is not None:
            hold()
        else:
            hold_for(seconds, sleep=sleep, monotonic=monotonic)

    def stop_after_hold(emit: Callable[[str], None], reason: str) -> None:
        """Hold this instance for diagnosis, then stop paying for it.

        The deletion has its own bounded retry budget; this is the outer one. A
        provider outage that spans the whole inner budget must not leave the
        instance billing forever, which is the failure this supervisor exists to
        remove, so the hold is repeated a fixed number of times before the
        container gives up out loud.
        """
        seconds = limit
        for attempt in range(1, attempts_allowed + 1):
            emit(
                f"parallel worker holding this instance for diagnosis because {reason}; "
                f"the hold ends at {_deadline_text(now, seconds)} "
                f"({int(seconds)}s from now) and the instance then deletes itself "
                f"(self-delete attempt {attempt}/{attempts_allowed})\n"
            )
            wait_out(seconds)
            emit(
                "parallel worker diagnostic hold reached its deadline; requesting "
                f"self-delete {attempt}/{attempts_allowed}\n"
            )
            # Death is established here, at the deadline, and not when the pipe
            # broke: a healthy child keeps the whole hold to finish its training
            # or its publication, and only the instant before the delete request
            # does the supervisor insist that nothing can still be writing.
            if not every_child_stopped(emit):
                emit(
                    "parallel worker refuses to delete this instance while a supervised "
                    "child may still be writing evidence; manual cleanup is required\n"
                )
                return
            try:
                outcome = delete_instance(values)
            except Exception as exc:
                emit(
                    f"parallel worker self-delete {attempt}/{attempts_allowed} did not "
                    f"confirm ({type(exc).__name__}: {exc})\n"
                )
                if not _delete_is_retryable(exc) or attempt == attempts_allowed:
                    emit(
                        "parallel worker could not delete its own instance; "
                        "manual cleanup is required\n"
                    )
                    return
                seconds = max(0.0, float(self_delete_retry_seconds))
                continue
            emit(f"parallel worker self-delete outcome: {outcome}\n")
            return

    try:
        log_path = worker_log_path(values)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8", buffering=1)
    except Exception as exc:
        def emit_console(message: str) -> None:
            # Best-effort for the same reason: this path exists to hold and
            # then delete an instance whose durable log is already gone, and a
            # console write must not be what prevents it.
            try:
                output.write(message)
                output.flush()
            except Exception:
                pass

        emit_console(f"parallel worker could not initialize its durable log: {type(exc).__name__}: {exc}\n")
        stop_after_hold(
            emit_console,
            f"its durable log could not be initialized: {type(exc).__name__}: {exc}",
        )
        return 127

    child_env = dict(values)
    child_env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "training_automation", "parallel-run"]
    exit_code = 127
    recovery_path = log_path.with_name("worker-recovery-state.json")
    recovery_limit = max(0, min(int(recovery_attempts), len(RECOVERY_BACKOFF_SECONDS)))
    with log:
        def emit(message: str) -> None:
            # Two sinks, written independently and both best-effort. A console
            # whose consumer went away and a volume that filled up while it was
            # receiving a 9B checkpoint are ordinary events on a training pod,
            # and neither may raise out of here: this supervisor is the
            # container command, so an exception escaping the capture path ends
            # the container, takes a still-training child with it and leaves the
            # instance billing before the hold or the self-delete can run. The
            # durable log goes first, because it is the sink that still exists
            # tomorrow, and losing one sink never costs the line in the other.
            try:
                log.write(message)
            except Exception:
                pass
            try:
                output.write(message)
                output.flush()
            except Exception:
                pass

        recovery_started = 0
        terminal_recovery_state = False
        # False only when this supervisor never saw how the child ended. Every
        # other path either observed an exit status or never started a child.
        shard_outcome_known = True
        terminal_reason = "no archive-only recovery was possible"
        if recovery_path.is_file():
            try:
                recovery_state = _load_mapping(recovery_path)
                if (
                    recovery_state.get("schema_version") != 1
                    or recovery_state.get("run_id")
                    != _component(values, "TRAINING_RUN_ID")
                    or recovery_state.get("shard_id")
                    != _component(values, "TRAINING_SHARD_ID")
                ):
                    raise ValueError("recovery state identity does not match this worker")
                recovery_started = int(recovery_state.get("attempts_started", 0))
                if recovery_state.get("status") in {"completed", "exhausted"}:
                    terminal_recovery_state = True
                    exit_code = (
                        127
                        if recovery_state.get("status") == "completed"
                        else int(recovery_state.get("last_exit_code", 127))
                    )
                    terminal_reason = (
                        "the recovery budget already ended in "
                        f"{recovery_state.get('status')}"
                    )
                    emit(
                        "parallel worker preserving terminal recovery state "
                        f"{recovery_state.get('status')} without another child launch\n"
                    )
                else:
                    eligible, reason = _recovery_eligible(values)
                    if not eligible or recovery_started >= recovery_limit:
                        recovery_state.update({
                            "status": "exhausted",
                            "last_exit_code": int(
                                recovery_state.get("last_exit_code", 127)
                            ),
                            "reason": reason,
                        })
                        atomic_write_json(recovery_path, recovery_state)
                        terminal_recovery_state = True
                        exit_code = int(recovery_state["last_exit_code"])
                        terminal_reason = reason
                        emit(f"parallel worker recovery stopped: {reason}\n")
                    else:
                        recovery_started += 1
                        backoff = RECOVERY_BACKOFF_SECONDS[recovery_started - 1]
                        recovery_state.update({
                            "status": "waiting",
                            "attempts_started": recovery_started,
                            "backoff_seconds": backoff,
                            "reason": reason,
                        })
                        atomic_write_json(recovery_path, recovery_state)
                        emit(
                            "parallel worker resuming archive-only recovery "
                            f"{recovery_started}/{recovery_limit} after {backoff}s\n"
                        )
                        sleep(backoff)
                        recovery_state["status"] = "running"
                        atomic_write_json(recovery_path, recovery_state)
            except Exception as exc:
                terminal_recovery_state = True
                exit_code = 127
                terminal_reason = (
                    f"recovery could not be safely resumed: {type(exc).__name__}: {exc}"
                )
                emit(
                    "parallel worker could not safely resume recovery: "
                    f"{type(exc).__name__}: {exc}\n"
                )
        while not terminal_recovery_state:
            label = "initial" if recovery_started == 0 else f"recovery-{recovery_started}"
            emit(f"parallel worker starting {label} supervised queue attempt\n")
            process = None
            try:
                process = popen(
                    command,
                    cwd="/app/ai-toolkit",
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # The child prints tqdm progress bars whose glyphs are
                    # multi-byte, and a read boundary can fall inside one of
                    # them. Strict decoding turns that ordinary byte into an
                    # exception, so the stream is decoded leniently: an
                    # unreadable character is a damaged character, never a
                    # verdict on the run.
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except Exception as exc:
                # No child was started, so this failure is the shard's own and
                # nothing alive can be harmed by treating it as one.
                emit(
                    "parallel worker launch failure: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                exit_code = 127
            else:
                child = {
                    "process": process,
                    "pgid": child_process_group(process),
                    "reaped": False,
                }
                children.append(child)
                capture_failure: str | None = None
                try:
                    stream = process.stdout
                except Exception as exc:
                    stream = None
                    capture_failure = f"{type(exc).__name__}: {exc}"
                else:
                    capture_failure = (
                        None
                        if stream is not None
                        else "the child was started without a readable output stream"
                    )
                if capture_failure is None:
                    capture_failure = _capture_output(stream, emit)
                if capture_failure is None:
                    try:
                        exit_code = int(process.wait())
                        child["reaped"] = True
                    except Exception as exc:
                        capture_failure = f"{type(exc).__name__}: {exc}"
                if capture_failure is not None:
                    # The supervisor lost the output, not the child. Nothing is
                    # signalled and nothing is concluded until the child itself
                    # says how it ended.
                    emit(
                        "parallel worker lost the supervised child's output stream "
                        f"({capture_failure}); the child is unaffected and this "
                        "supervisor now waits for its real exit status\n"
                    )
                    observed = _observe_child_exit(process, emit=emit)
                    if observed is None:
                        shard_outcome_known = False
                        exit_code = CAPTURE_UNRESOLVED_EXIT
                        emit(
                            "parallel worker could not establish how the supervised "
                            "child ended, so it records no failure for this shard\n"
                        )
                    else:
                        exit_code = observed
                        child["reaped"] = True
                        emit(
                            "parallel worker recovered the supervised child's real "
                            f"exit status after losing its output: exit={exit_code}\n"
                        )
            emit(f"parallel worker {label} exit={exit_code}\n")
            if not shard_outcome_known:
                # An unobserved child is not a failed shard: no recovery
                # relaunch over a process that may still be running, and no
                # terminal state written on a guess.
                break
            if exit_code == 0:
                if recovery_started:
                    atomic_write_json(recovery_path, {
                        "schema_version": 1,
                        "run_id": _component(values, "TRAINING_RUN_ID"),
                        "shard_id": _component(values, "TRAINING_SHARD_ID"),
                        "status": "completed",
                        "attempts_started": recovery_started,
                        "last_exit_code": 0,
                    })
                break
            eligible, reason = _recovery_eligible(values)
            if not eligible or recovery_started >= recovery_limit:
                if recovery_started:
                    atomic_write_json(recovery_path, {
                        "schema_version": 1,
                        "run_id": _component(values, "TRAINING_RUN_ID"),
                        "shard_id": _component(values, "TRAINING_SHARD_ID"),
                        "status": "exhausted",
                        "attempts_started": recovery_started,
                        "last_exit_code": exit_code,
                        "reason": reason,
                    })
                emit(f"parallel worker recovery stopped: {reason}\n")
                terminal_reason = reason
                break
            recovery_started += 1
            backoff = RECOVERY_BACKOFF_SECONDS[recovery_started - 1]
            atomic_write_json(recovery_path, {
                "schema_version": 1,
                "run_id": _component(values, "TRAINING_RUN_ID"),
                "shard_id": _component(values, "TRAINING_SHARD_ID"),
                "status": "waiting",
                "attempts_started": recovery_started,
                "last_exit_code": exit_code,
                "backoff_seconds": backoff,
                "reason": reason,
            })
            emit(
                f"parallel worker scheduling archive-only recovery "
                f"{recovery_started}/{recovery_limit} after {backoff}s: {reason}\n"
            )
            sleep(backoff)
            recovery_state = _load_mapping(recovery_path)
            recovery_state["status"] = "running"
            atomic_write_json(recovery_path, recovery_state)
        if not shard_outcome_known:
            # The hold exists to preserve a failure for diagnosis and then stop
            # paying for it. A child whose exit was never observed may still be
            # training or publishing, and destroying it to save the bill is the
            # one outcome this supervisor must never produce.
            emit(
                "parallel worker holds nothing and deletes nothing because the "
                "supervised child's outcome was never observed; this supervisor "
                "exits here, which ends the container command without deleting the "
                "instance and without protecting the child from a provider restart, "
                "so this instance needs manual inspection now\n"
            )
        elif exit_code != 0:
            stop_after_hold(
                emit,
                f"the supervised queue ended with exit={exit_code}: {terminal_reason}",
            )
    return exit_code


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
