from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from .backup import BackupConfigurationError, BackupError, LocalArtifact, sha256_file
from .state import atomic_write_json


CATALOG_SCHEMA = 2
ALLOWED_DESTINATIONS = {"loras", "diffusion_models", "vae", "text_encoders"}


class CatalogClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...
    def read_remote_file(self, repo_id: str, repo_type: str, path: str) -> tuple[bytes | None, str]: ...
    def commit_files(self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact], message: str, parent_commit: str | None = None) -> str: ...
    def download_file(self, repo_id: str, repo_type: str, path: str, revision: str, destination: Path) -> None: ...
    def path_metadata(self, repo_id: str, repo_type: str, paths: list[str], revision: str) -> Mapping[str, Mapping[str, Any]]: ...


def safe_name(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise BackupConfigurationError("catalog name must contain a letter or number")
    return result


def safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise BackupConfigurationError(f"unsafe catalog relative path: {value!r}")
    return Path(*posix.parts)


class CatalogStore:
    def __init__(self, *, client: CatalogClient, repo_id: str, repo_type: str, catalog_path: str, work_dir: Path):
        self.client = client
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.catalog_path = str(PurePosixPath(catalog_path))
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def read(self) -> tuple[dict[str, Any], str]:
        payload, revision = self.client.read_remote_file(self.repo_id, self.repo_type, self.catalog_path)
        catalog = {"schema_version": CATALOG_SCHEMA, "models": []} if payload is None else json.loads(payload)
        if catalog.get("schema_version") == 1 and isinstance(catalog.get("models"), list):
            for model in catalog["models"]:
                model.setdefault("base_model", None)
                model.setdefault("selection", None)
                for checkpoint in model.get("checkpoints", []):
                    for artifact in checkpoint.get("weights", []):
                        artifact.setdefault(
                            "generation_relative_path", artifact.get("relative_path")
                        )
                    if checkpoint.get("imported_existing"):
                        checkpoint["resume_artifacts"] = []
                    else:
                        for artifact in checkpoint.get("resume_artifacts", []):
                            legacy = artifact.get("relative_path", "")
                            artifact.setdefault(
                                "training_relative_path", Path(legacy).name
                            )
            catalog["schema_version"] = CATALOG_SCHEMA
        if catalog.get("schema_version") != CATALOG_SCHEMA or not isinstance(catalog.get("models"), list):
            raise BackupError("unsupported or invalid remote model catalog")
        return catalog, revision

    def _write(self, catalog: Mapping[str, Any], parent: str, message: str) -> str:
        local = self.work_dir / "catalog.json"
        atomic_write_json(local, catalog)
        artifact = LocalArtifact(str(local), self.catalog_path, local.stat().st_size, sha256_file(local))
        return self.client.commit_files(
            self.repo_id, self.repo_type, [artifact], message, parent_commit=parent
        )

    def ensure_model(self, metadata: Mapping[str, Any], attempts: int = 5) -> tuple[dict[str, Any], str]:
        name = str(metadata["name"])
        destination_kind = str(metadata.get("destination_kind", "loras"))
        if destination_kind not in ALLOWED_DESTINATIONS:
            raise BackupConfigurationError(f"unsupported destination kind: {destination_kind}")
        last_error: Exception | None = None
        for _ in range(attempts):
            catalog, parent = self.read()
            existing = next((item for item in catalog["models"] if item["name"] == name), None)
            if existing:
                changed = False
                for key in ("base_arch", "base_model", "trigger_word", "destination_kind"):
                    if key == "base_model" and existing.get(key) is None and metadata.get(key) is not None:
                        existing[key] = metadata[key]
                        changed = True
                        continue
                    if existing.get(key) != metadata.get(key):
                        raise BackupConfigurationError(f"catalog model {name!r} has conflicting {key}")
                if not changed:
                    return existing, parent
                try:
                    revision = self._write(catalog, parent, f"Record exact base model for {name}")
                    return existing, revision
                except Exception as exc:
                    last_error = exc
                    continue
            numeric_id = max([int(item["id"]) for item in catalog["models"]] or [0]) + 1
            model = {
                "id": numeric_id,
                "name": name,
                "folder": f"{numeric_id:04d}-{safe_name(name)}",
                "base_arch": metadata.get("base_arch"),
                "base_model": metadata.get("base_model"),
                "trigger_word": metadata.get("trigger_word"),
                "destination_kind": destination_kind,
                "checkpoints": [],
                "selected_checkpoint_id": None,
                "selection": None,
            }
            catalog["models"].append(model)
            catalog["models"].sort(key=lambda item: int(item["id"]))
            try:
                revision = self._write(catalog, parent, f"Reserve catalog model {numeric_id:04d} {name}")
                return model, revision
            except Exception as exc:
                last_error = exc
        raise BackupError("catalog reservation conflicted repeatedly") from last_error

    def add_checkpoint(self, model_name: str, checkpoint: Mapping[str, Any], attempts: int = 5) -> str:
        last_error: Exception | None = None
        for _ in range(attempts):
            catalog, parent = self.read()
            model = next((item for item in catalog["models"] if item["name"] == model_name), None)
            if model is None:
                raise BackupError(f"catalog model disappeared: {model_name}")
            existing = next((item for item in model["checkpoints"] if item["checkpoint_id"] == checkpoint["checkpoint_id"]), None)
            if existing:
                if existing != checkpoint:
                    raise BackupError("checkpoint id already exists with different immutable evidence")
                return parent
            model["checkpoints"].append(dict(checkpoint))
            model["checkpoints"].sort(key=lambda item: (int(item["step"]), item["checkpoint_id"]))
            try:
                return self._write(catalog, parent, f"Index {model_name} {checkpoint['checkpoint_id']}")
            except Exception as exc:
                last_error = exc
        raise BackupError("catalog checkpoint update conflicted repeatedly") from last_error

    def select(
        self,
        identifier: str | int,
        checkpoint_id: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        attempts: int = 5,
    ) -> str:
        last_error: Exception | None = None
        for _ in range(attempts):
            catalog, parent = self.read()
            model = resolve_model(catalog, identifier)
            checkpoint = next((item for item in model["checkpoints"] if item["checkpoint_id"] == checkpoint_id), None)
            if checkpoint is None:
                raise BackupError(f"unknown checkpoint {checkpoint_id!r}")
            model["selected_checkpoint_id"] = checkpoint_id
            model["selection"] = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_revision": checkpoint["revision"],
                "evidence": dict(evidence or {}),
            }
            try:
                return self._write(catalog, parent, f"Select {model['name']} {checkpoint_id}")
            except Exception as exc:
                last_error = exc
        raise BackupError("catalog selection conflicted repeatedly") from last_error

    def import_existing(
        self,
        *,
        metadata: Mapping[str, Any],
        remote_path: str,
        checkpoint_id: str,
        step: int,
        final: bool,
    ) -> str:
        """Index an existing remote weight without moving or replacing it."""
        model, _ = self.ensure_model(metadata)
        _, revision = self.read()
        safe_relative_path(remote_path)
        fd, temporary_name = tempfile.mkstemp(prefix="catalog-import-", dir=self.work_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            self.client.download_file(
                self.repo_id, self.repo_type, remote_path, revision, temporary
            )
            record = {
                "remote_path": remote_path,
                "generation_relative_path": f"{model['folder']}/{Path(remote_path).name}",
                "size": temporary.stat().st_size,
                "sha256": sha256_file(temporary),
            }
        finally:
            temporary.unlink(missing_ok=True)
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "step": int(step),
            "final": bool(final),
            "revision": revision,
            "weights": [record],
            "resume_artifacts": [],
            "imported_existing": True,
            "training_resume_available": False,
        }
        return self.add_checkpoint(str(metadata["name"]), checkpoint)


def resolve_model(catalog: Mapping[str, Any], identifier: str | int) -> dict[str, Any]:
    text = str(identifier)
    numeric = int(text) if text.isdigit() else None
    matches = [
        item for item in catalog["models"]
        if (numeric is not None and int(item["id"]) == numeric)
        or item["name"] == text or item["folder"] == text
    ]
    if len(matches) != 1:
        raise BackupError(f"model identifier {identifier!r} did not resolve uniquely")
    return matches[0]


def _install(client: CatalogClient, repo_id: str, repo_type: str, artifact: Mapping[str, Any], revision: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and target.stat().st_size == int(artifact["size"]) and sha256_file(target) == artifact["sha256"]:
            return "already-present"
        raise BackupError(f"refusing to overwrite conflicting local file: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        client.download_file(repo_id, repo_type, artifact["remote_path"], revision, temporary)
        if temporary.stat().st_size != int(artifact["size"]) or sha256_file(temporary) != artifact["sha256"]:
            raise BackupError(f"download verification failed for {artifact['remote_path']}")
        try:
            os.link(temporary, target)
            return "installed"
        except FileExistsError:
            if (
                target.is_file()
                and target.stat().st_size == int(artifact["size"])
                and sha256_file(target) == artifact["sha256"]
            ):
                return "already-present"
            raise BackupError(f"refusing to overwrite concurrently created file: {target}")
    finally:
        temporary.unlink(missing_ok=True)


def _target(root: Path, relative: Path) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve(strict=False)
    if not target.is_relative_to(resolved_root):
        raise BackupConfigurationError(f"restore target escapes configured root: {relative}")
    return target


def restore_generation(*, client: CatalogClient, repo_id: str, repo_type: str, catalog: Mapping[str, Any], identifier: str | int, roots: Mapping[str, Path], checkpoint_id: str | None = None) -> list[dict[str, str]]:
    model = resolve_model(catalog, identifier)
    selected = checkpoint_id or model.get("selected_checkpoint_id")
    if not selected:
        raise BackupError("model has no selected checkpoint; specify one explicitly")
    checkpoint = next((item for item in model["checkpoints"] if item["checkpoint_id"] == selected), None)
    if checkpoint is None:
        raise BackupError("selected checkpoint is absent from catalog")
    kind = model["destination_kind"]
    if kind not in roots:
        raise BackupConfigurationError(f"no local root configured for destination kind {kind}")
    results = []
    for artifact in checkpoint["weights"]:
        relative_value = artifact.get("generation_relative_path") or artifact.get("relative_path")
        relative = safe_relative_path(relative_value)
        target = _target(Path(roots[kind]), relative)
        status = _install(client, repo_id, repo_type, artifact, checkpoint["revision"], target)
        results.append({"path": str(target), "status": status})
    return results


def restore_training(*, client: CatalogClient, repo_id: str, repo_type: str, catalog: Mapping[str, Any], identifier: str | int, target_root: Path, checkpoint_id: str | None = None) -> list[dict[str, str]]:
    model = resolve_model(catalog, identifier)
    selected = checkpoint_id or model.get("selected_checkpoint_id")
    checkpoint = next((item for item in model["checkpoints"] if item["checkpoint_id"] == selected), None)
    if checkpoint is None:
        raise BackupError("training checkpoint is absent or not selected")
    if not checkpoint.get("resume_artifacts"):
        raise BackupError("selected checkpoint is weights-only and has no trainer resume state")
    results = []
    for artifact in checkpoint["resume_artifacts"]:
        relative_value = artifact.get("training_relative_path")
        if relative_value is None:
            relative_value = Path(artifact.get("relative_path", "")).name
        target = _target(target_root, safe_relative_path(relative_value))
        status = _install(client, repo_id, repo_type, artifact, checkpoint["revision"], target)
        results.append({"path": str(target), "status": status})
    return results
