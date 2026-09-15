from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .state import atomic_write_json


COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FINALIZATION_RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_SECONDS = (5, 15, 30)


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


def hold_forever() -> None:
    while True:
        time.sleep(3600)


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
    hold: Callable[[], None] = hold_forever,
    sleep: Callable[[float], None] = time.sleep,
    recovery_attempts: int = FINALIZATION_RECOVERY_ATTEMPTS,
    console: TextIO | None = None,
) -> int:
    values = dict(os.environ if env is None else env)
    output = console or sys.stdout
    try:
        log_path = worker_log_path(values)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8", buffering=1)
    except Exception as exc:
        output.write(f"parallel worker could not initialize its durable log: {type(exc).__name__}: {exc}\n")
        output.flush()
        hold()
        return 127

    child_env = dict(values)
    child_env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "training_automation", "parallel-run"]
    exit_code = 127
    recovery_path = log_path.with_name("worker-recovery-state.json")
    recovery_limit = max(0, min(int(recovery_attempts), len(RECOVERY_BACKOFF_SECONDS)))
    with log:
        def emit(message: str) -> None:
            output.write(message)
            output.flush()
            log.write(message)

        recovery_started = 0
        terminal_recovery_state = False
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
                emit(
                    "parallel worker could not safely resume recovery: "
                    f"{type(exc).__name__}: {exc}\n"
                )
        while not terminal_recovery_state:
            label = "initial" if recovery_started == 0 else f"recovery-{recovery_started}"
            emit(f"parallel worker starting {label} supervised queue attempt\n")
            try:
                process = popen(
                    command,
                    cwd="/app/ai-toolkit",
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if process.stdout is None:
                    raise RuntimeError("parallel worker could not capture child output")
                for line in process.stdout:
                    emit(line)
                exit_code = int(process.wait())
            except Exception as exc:
                emit(
                    "parallel worker launch/capture failure: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                exit_code = 127
            emit(f"parallel worker {label} exit={exit_code}\n")
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
                emit(
                    "parallel worker preserving this instance for diagnosis; "
                    "manual cleanup is required\n"
                )
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
    if exit_code != 0:
        hold()
    return exit_code


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
