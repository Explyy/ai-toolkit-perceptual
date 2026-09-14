from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, TextIO


COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def run_worker(
    *,
    env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    hold: Callable[[], None] = hold_forever,
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
    with log:
        def emit(message: str) -> None:
            output.write(message)
            output.flush()
            log.write(message)

        emit("parallel worker starting one supervised queue attempt\n")
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
            emit(f"parallel worker launch/capture failure: {type(exc).__name__}: {exc}\n")
            exit_code = 127
        emit(f"parallel worker exit={exit_code}\n")
        if exit_code != 0:
            emit("parallel worker preserving this instance for diagnosis; manual cleanup is required\n")
    if exit_code != 0:
        hold()
    return exit_code


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
