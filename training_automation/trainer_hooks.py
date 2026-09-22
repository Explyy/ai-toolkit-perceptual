from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from .backup import CheckpointBackup


def _resolve_repo_id(config: Mapping[str, Any]) -> str:
    if config.get("repo_id"):
        return str(config["repo_id"])
    env_name = str(config.get("repo_id_env", "HF_REPO_ID"))
    return os.environ.get(env_name, "")


def initialize_checkpoint_backup(process: Any) -> CheckpointBackup | None:
    config = process.get_conf("checkpoint_backup", {}) or {}
    if not config.get("enabled", False):
        return None
    state_path = Path(config.get("state_path") or Path(process.save_root) / ".automation" / "backup-state.json")
    backup = CheckpointBackup(
        repo_id=_resolve_repo_id(config),
        repo_type=str(config.get("repo_type", "model")),
        state_path=state_path,
        remote_prefix=str(config.get("remote_prefix", "training-backups")),
        token_env=str(config.get("token_env", "HF_TOKEN")),
        max_attempts=int(config.get("max_attempts", 4)),
        backoff_seconds=float(config.get("backoff_seconds", 2)),
        catalog_metadata={
            "name": (config.get("catalog") or {}).get("name") or str(process.job.name),
            "base_arch": (config.get("catalog") or {}).get("base_arch") or getattr(process.model_config, "arch", None),
            "base_model": (config.get("catalog") or {}).get("base_model") or getattr(process.model_config, "name_or_path", None),
            "trigger_word": (config.get("catalog") or {}).get("trigger_word", process.get_conf("trigger_word", None)),
            "destination_kind": (config.get("catalog") or {}).get("destination_kind", "loras"),
            "expected_id": (config.get("catalog") or {}).get("expected_id"),
        },
    )
    backup.validate_destination()
    backup.resume_pending()
    return backup


def _checkpoint_paths(process: Any, checkpoint_path: Path, checkpoint_id: str, final: bool) -> list[Path]:
    root = Path(process.save_root)
    paths = [checkpoint_path]
    if not final:
        paths.extend(item for item in root.iterdir() if checkpoint_id in item.name)
    else:
        paths.extend(
            item for item in root.iterdir()
            if item != checkpoint_path
            and (
                (item.is_dir() and item.name.startswith(str(process.job.name)))
                or (item.is_file() and item.suffix in {".safetensors", ".pt"})
            )
            and re.search(r"_\d{9}(?:_|\.|$)", item.name) is None
        )
    for name in ("optimizer.pt", "config.yaml", "learnable_snr.json"):
        item = root / name
        if item.exists():
            paths.append(item)
    config = process.get_conf("checkpoint_backup", {}) or {}
    paths.extend(Path(item) for item in config.get("extra_paths", []))
    return paths


def protect_training_checkpoint(process: Any, checkpoint_path: str, requested_step: int | None) -> None:
    backup = getattr(process, "_training_checkpoint_backup", None)
    if backup is None:
        return
    final = requested_step is None
    step = int(process.step_num if final else requested_step)
    checkpoint_id = f"step-{step:09d}" + ("-final" if final else "")
    backup.protect(
        job_id=str(process.job.name),
        checkpoint_id=checkpoint_id,
        step=step,
        paths=_checkpoint_paths(process, Path(checkpoint_path), checkpoint_id.replace("step-", "_"), final),
        final=final,
    )


def checkpoint_may_be_removed(process: Any, path: str) -> bool:
    backup = getattr(process, "_training_checkpoint_backup", None)
    return True if backup is None else backup.can_delete(Path(path))
