from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from .backup import BackupError, LocalArtifact, sha256_file
from .catalog import safe_relative_path
from .state import atomic_write_json, read_json


ARCHIVE_STATE_SCHEMA = 1


class ArchiveClient(Protocol):
    def commit_files(
        self,
        repo_id: str,
        repo_type: str,
        artifacts: list[LocalArtifact],
        message: str,
        parent_commit: str | None = None,
    ) -> str: ...

    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]: ...


class EvidenceArchive:
    def __init__(
        self,
        *,
        client: ArchiveClient,
        repo_id: str,
        repo_type: str,
        remote_prefix: str,
        run_id: str,
        shard_id: str,
        state_path: Path,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.remote_prefix = safe_relative_path(remote_prefix).as_posix()
        self.run_id = safe_relative_path(run_id).as_posix()
        self.shard_id = safe_relative_path(shard_id).as_posix()
        if "/" in self.run_id or "/" in self.shard_id:
            raise BackupError("run_id and shard_id must each be one safe path component")
        self.state_path = state_path
        self.max_attempts = max(1, int(max_attempts))
        self.sleep = sleep

    def _destination(self) -> dict[str, str]:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "remote_prefix": self.remote_prefix,
            "run_id": self.run_id,
            "shard_id": self.shard_id,
        }

    def _state(self) -> dict[str, Any]:
        state = read_json(
            self.state_path,
            {
                "schema_version": ARCHIVE_STATE_SCHEMA,
                "destination": self._destination(),
                "status": "pending",
            },
        )
        if (
            state.get("schema_version") != ARCHIVE_STATE_SCHEMA
            or state.get("destination") != self._destination()
        ):
            raise BackupError("archive state belongs to a different run destination")
        return state

    def _verify(self, artifacts: Sequence[LocalArtifact], revision: str) -> None:
        metadata = self.client.path_metadata(
            self.repo_id,
            self.repo_type,
            [item.remote_path for item in artifacts],
            revision,
        )
        for artifact in artifacts:
            remote = metadata.get(artifact.remote_path)
            if remote is None or int(remote.get("size", -1)) != artifact.size:
                raise BackupError(f"archive size verification failed for {artifact.remote_path}")
            remote_sha = remote.get("sha256")
            if remote_sha and str(remote_sha).removeprefix("sha256:") != artifact.sha256:
                raise BackupError(f"archive hash verification failed for {artifact.remote_path}")

    def publish(
        self,
        files: Sequence[tuple[Path, str]],
        completion: Mapping[str, Any],
    ) -> dict[str, Any]:
        remote_root = PurePosixPath(self.remote_prefix) / self.run_id / self.shard_id
        artifacts = []
        seen_remote: set[str] = set()
        for local, relative_value in files:
            if not local.is_file():
                raise BackupError(f"archive evidence file is missing: {local}")
            relative = safe_relative_path(relative_value).as_posix()
            remote = str(remote_root / "evidence" / PurePosixPath(relative))
            if remote in seen_remote:
                raise BackupError(f"duplicate archive evidence destination: {relative}")
            seen_remote.add(remote)
            artifacts.append(
                LocalArtifact(
                    str(local), remote, local.stat().st_size, sha256_file(local),
                    role="evidence", relative_path=relative,
                )
            )
        if not artifacts:
            raise BackupError("completion archive requires at least one evidence file")
        fingerprint = hashlib.sha256(
            "\n".join(f"{item.relative_path}:{item.size}:{item.sha256}" for item in artifacts).encode()
        ).hexdigest()
        state = self._state()
        if state.get("status") == "completed" and state.get("fingerprint") == fingerprint:
            return state
        state.update({
            "status": "pending",
            "fingerprint": fingerprint,
            "artifacts": [item.__dict__ for item in artifacts],
        })
        atomic_write_json(self.state_path, state)
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                evidence_revision = self.client.commit_files(
                    self.repo_id,
                    self.repo_type,
                    artifacts,
                    f"Archive training evidence {self.run_id}/{self.shard_id}",
                )
                self._verify(artifacts, evidence_revision)
                completion_path = self.state_path.with_name("completion-manifest.json")
                completion_document = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "shard_id": self.shard_id,
                    "status": "completed",
                    "evidence_revision": evidence_revision,
                    "evidence_fingerprint": fingerprint,
                    "evidence": [
                        {
                            "remote_path": item.remote_path,
                            "relative_path": item.relative_path,
                            "size": item.size,
                            "sha256": item.sha256,
                        }
                        for item in artifacts
                    ],
                    "completion": dict(completion),
                }
                atomic_write_json(completion_path, completion_document)
                completion_artifact = LocalArtifact(
                    str(completion_path),
                    str(remote_root / "completion.json"),
                    completion_path.stat().st_size,
                    sha256_file(completion_path),
                    role="completion",
                    relative_path="completion.json",
                )
                completion_revision = self.client.commit_files(
                    self.repo_id,
                    self.repo_type,
                    [completion_artifact],
                    f"Complete training shard {self.run_id}/{self.shard_id}",
                )
                self._verify([completion_artifact], completion_revision)
                state.update({
                    "status": "completed",
                    "evidence_revision": evidence_revision,
                    "completion_revision": completion_revision,
                    "completion_remote_path": completion_artifact.remote_path,
                    "attempts": attempt + 1,
                })
                state.pop("error", None)
                atomic_write_json(self.state_path, state)
                return state
            except Exception as exc:
                last_error = exc
                state.update({
                    "status": "failed",
                    "attempts": attempt + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                atomic_write_json(self.state_path, state)
                if attempt + 1 < self.max_attempts:
                    self.sleep(2**attempt)
        raise BackupError(f"evidence archive failed after {self.max_attempts} attempts") from last_error
