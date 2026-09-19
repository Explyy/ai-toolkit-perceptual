from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .state import atomic_write_json
from .unified import run_unified_workflow


def supervise(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    run: Callable[..., Mapping[str, Any]] = run_unified_workflow,
    sleep: Callable[[float], None] = time.sleep,
    once: bool = False,
) -> int:
    values = dict(os.environ if env is None else env)
    worker_id = int(values.get("TRAINING_WORKER_ID", "0"))
    if worker_id < 0:
        raise ValueError("TRAINING_WORKER_ID cannot be negative")
    status_path = Path(
        values.get(
            "TRAINING_UNIFIED_STATUS",
            f"/storage/automation/unified/worker-{worker_id}/supervisor-state.json",
        )
    )
    interval = float(values.get("TRAINING_UNIFIED_POLL_SECONDS", "60"))
    if interval < 1:
        raise ValueError("TRAINING_UNIFIED_POLL_SECONDS must be at least one second")
    attempts = 0
    while True:
        attempts += 1
        try:
            result = dict(run(config_path, env=values))
            status = {
                "schema_version": 1,
                "status": result.get("status", "unknown"),
                "attempts": attempts,
                "result": result,
            }
            code = 0
        except Exception as exc:
            status = {
                "schema_version": 1,
                "status": "held",
                "attempts": attempts,
                "error": f"{type(exc).__name__}: {exc}",
            }
            code = 1
        atomic_write_json(status_path, status)
        print(f"unified automation status={status['status']} attempts={attempts}", flush=True)
        if once:
            return code
        sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="training-automation-supervisor")
    parser.add_argument("config", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    return supervise(args.config, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
