from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from .backup import BackupError, sha256_file
from .catalog import safe_relative_path
from .state import atomic_write_json, read_json


STAGING_SCHEMA = 1
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StagingClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...

    def download_file(
        self, repo_id: str, repo_type: str, path: str, revision: str, destination: Path
    ) -> None: ...


def _safe_target(root: Path, relative_value: str) -> Path:
    relative = safe_relative_path(relative_value)
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve(strict=False)
    if not target.is_relative_to(resolved_root):
        raise BackupError(f"staging path escapes its root: {relative_value!r}")
    cursor = resolved_root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BackupError(f"staging path traverses a symlink: {cursor}")
    return target


def _install_no_clobber(
    client: StagingClient,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    remote_path: str,
    target: Path,
    expected_size: int,
    expected_sha256: str,
) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            target.is_file()
            and target.stat().st_size == expected_size
            and sha256_file(target) == expected_sha256
        ):
            return "already-present"
        raise BackupError(f"refusing to overwrite conflicting staged file: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        client.download_file(repo_id, repo_type, remote_path, revision, temporary)
        if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_sha256:
            raise BackupError(f"staged download verification failed for {remote_path}")
        try:
            os.link(temporary, target)
            return "installed"
        except FileExistsError:
            if (
                target.is_file()
                and target.stat().st_size == expected_size
                and sha256_file(target) == expected_sha256
            ):
                return "already-present"
            raise BackupError(f"refusing to overwrite concurrently created staged file: {target}")
    finally:
        temporary.unlink(missing_ok=True)


class PinnedDatasetStager:
    def __init__(
        self,
        *,
        client: StagingClient,
        repo_id: str,
        repo_type: str,
        revision: str,
        target_root: Path,
        state_path: Path,
    ):
        if repo_type not in {"model", "dataset"}:
            raise BackupError("staging repo_type must be model or dataset")
        if not COMMIT_RE.fullmatch(revision):
            raise BackupError("dataset staging revision must be an immutable 40-character commit SHA")
        if not client.repo_is_private(repo_id, repo_type):
            raise BackupError("refusing to stage from a non-private repository")
        self.client = client
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.target_root = target_root
        self.state_path = state_path

    def _state(self) -> dict[str, Any]:
        source = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }
        state = read_json(
            self.state_path,
            {"schema_version": STAGING_SCHEMA, "source": source, "files": {}},
        )
        if state.get("schema_version") != STAGING_SCHEMA or state.get("source") != source:
            raise BackupError("staging state belongs to a different source or schema")
        return state

    def stage_dataset(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        dataset_id = str(dataset.get("id", ""))
        safe_id = safe_relative_path(dataset_id)
        if len(safe_id.parts) != 1:
            raise BackupError("dataset id must be one safe path component")
        root = self.target_root / safe_id
        state = self._state()
        results = []
        for spec in dataset.get("files", []):
            remote_path = safe_relative_path(str(spec.get("remote_path", ""))).as_posix()
            relative_path = safe_relative_path(str(spec.get("relative_path", ""))).as_posix()
            expected_size = int(spec.get("size", -1))
            expected_sha = str(spec.get("sha256", "")).lower()
            if expected_size < 0 or not SHA256_RE.fullmatch(expected_sha):
                raise BackupError(f"invalid size or sha256 for staged file {remote_path!r}")
            key = f"{dataset_id}/{relative_path}"
            target = _safe_target(root, relative_path)
            state["files"][key] = {
                "status": "pending",
                "remote_path": remote_path,
                "relative_path": relative_path,
                "size": expected_size,
                "sha256": expected_sha,
                "target": str(target),
            }
            atomic_write_json(self.state_path, state)
            try:
                install_status = _install_no_clobber(
                    self.client,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    remote_path=remote_path,
                    target=target,
                    expected_size=expected_size,
                    expected_sha256=expected_sha,
                )
            except Exception as exc:
                state["files"][key].update(
                    {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
                atomic_write_json(self.state_path, state)
                raise
            state["files"][key].update({"status": "verified", "install_status": install_status})
            state["files"][key].pop("error", None)
            atomic_write_json(self.state_path, state)
            results.append(state["files"][key])
        if not results:
            raise BackupError(f"dataset {dataset_id!r} has no files")
        return {"dataset_id": dataset_id, "folder": str(root), "files": results}
