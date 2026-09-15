from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from training_automation.backends import UltralyticsPoseCPUBackend
from training_automation.evaluation import preserve_cuda_visibility


ROOT = "/opt/training-automation-models"
KEY = "CUDA_VISIBLE_DEVICES"


def child_visibility() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", f"import os; print(os.environ.get('{KEY}', '<absent>'))"],
        text=True,
    ).strip()


def infer(pose: UltralyticsPoseCPUBackend) -> None:
    status = pose.metrics(np.zeros((64, 64, 3), dtype=np.uint8))["status"]
    assert status in {"missing", "available", "ambiguous", "occluded", "degenerate"}


def main() -> None:
    sources = json.load(open(f"{ROOT}/SOURCES.json", encoding="utf-8"))

    os.environ[KEY] = "0"
    with preserve_cuda_visibility():
        pose = UltralyticsPoseCPUBackend(
            model_path=sources["pose"]["path"],
            expected_sha256=sources["pose"]["sha256"],
        )
        infer(pose)
    assert os.environ[KEY] == "0"
    assert child_visibility() == "0"

    os.environ.pop(KEY)
    with preserve_cuda_visibility():
        infer(pose)
    assert KEY not in os.environ
    assert child_visibility() == "<absent>"


if __name__ == "__main__":
    main()
