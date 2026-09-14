from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol

from .state import atomic_write_json, read_json


STATE_SCHEMA = 2


class BackupError(RuntimeError):
    pass


class BackupConfigurationError(BackupError):
    pass


@dataclass(frozen=True)
class LocalArtifact:
    local_path: str
    remote_path: str
    size: int
    sha256: str
    role: str = "resume"
    relative_path: str = ""


class BackupClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...

    def commit_files(
        self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact], message: str,
        parent_commit: str | None = None,
    ) -> str: ...

    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def read_remote_file(self, repo_id: str, repo_type: str, path: str) -> tuple[bytes | None, str]: ...

    def download_file(self, repo_id: str, repo_type: str, path: str, revision: str, destination: Path) -> None: ...


class HuggingFaceBackupClient:
    """Small adapter around huggingface_hub, imported only when enabled."""

    def __init__(self, token: str):
        from huggingface_hub import HfApi

        self._token = token
        self._api = HfApi(token=token)

    def repo_is_private(self, repo_id: str, repo_type: str) -> bool:
        info = self._api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return bool(getattr(info, "private", False))

    def commit_files(
        self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact], message: str,
        parent_commit: str | None = None,
    ) -> str:
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=item.remote_path, path_or_fileobj=item.local_path)
            for item in artifacts
        ]
        result = self._api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=message,
            parent_commit=parent_commit,
        )
        commit_id = getattr(result, "oid", None) or getattr(result, "commit_id", None)
        if not commit_id:
            raise BackupError("Hugging Face did not return a commit id")
        return str(commit_id)

    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]:
        entries = self._api.get_paths_info(
            repo_id=repo_id, repo_type=repo_type, paths=paths, revision=revision
        )
        result: dict[str, dict[str, Any]] = {}
        for entry in entries:
            lfs = getattr(entry, "lfs", None)
            lfs_sha = None
            if isinstance(lfs, dict):
                lfs_sha = lfs.get("sha256") or lfs.get("oid")
            elif lfs is not None:
                lfs_sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            result[str(entry.path)] = {
                "size": getattr(entry, "size", None),
                "sha256": lfs_sha,
            }
        return result

    def read_remote_file(self, repo_id: str, repo_type: str, path: str) -> tuple[bytes | None, str]:
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.errors import EntryNotFoundError
        except ImportError:
            from huggingface_hub.utils import EntryNotFoundError

        revision = str(self._api.repo_info(repo_id=repo_id, repo_type=repo_type).sha)
        try:
            local = hf_hub_download(
                repo_id=repo_id, repo_type=repo_type, filename=path,
                revision=revision, token=self._token,
            )
        except EntryNotFoundError:
            return None, revision
        return Path(local).read_bytes(), revision

    def download_file(self, repo_id: str, repo_type: str, path: str, revision: str, destination: Path) -> None:
        from huggingface_hub import hf_hub_download
        import shutil

        local = hf_hub_download(
            repo_id=repo_id, repo_type=repo_type, filename=path,
            revision=revision, token=self._token,
        )
        with Path(local).open("rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote_artifacts(
    client: BackupClient,
    *,
    repo_id: str,
    repo_type: str,
    artifacts: Iterable[LocalArtifact],
    revision: str,
    context: str,
) -> None:
    artifacts = list(artifacts)
    metadata = client.path_metadata(
        repo_id, repo_type, [item.remote_path for item in artifacts], revision
    )
    for artifact in artifacts:
        remote = metadata.get(artifact.remote_path)
        if remote is None or int(remote.get("size", -1)) != artifact.size:
            raise BackupError(f"{context} size verification failed for {artifact.remote_path}")
        remote_sha = remote.get("sha256")
        if remote_sha:
            if str(remote_sha).removeprefix("sha256:") != artifact.sha256:
                raise BackupError(f"{context} hash verification failed for {artifact.remote_path}")
            continue
        fd, temporary_name = tempfile.mkstemp(prefix="training-automation-verify-")
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            client.download_file(
                repo_id, repo_type, artifact.remote_path, revision, temporary
            )
            if (
                temporary.stat().st_size != artifact.size
                or sha256_file(temporary) != artifact.sha256
            ):
                raise BackupError(
                    f"{context} downloaded-byte hash verification failed for {artifact.remote_path}"
                )
        finally:
            temporary.unlink(missing_ok=True)


def _files(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(item for item in sorted(path.rglob("*")) if item.is_file())
        elif path.is_file():
            expanded.append(path)
        else:
            raise BackupError(f"backup artifact does not exist: {path}")
    unique = {str(path.resolve()): path.resolve() for path in expanded}
    return [unique[key] for key in sorted(unique)]


class CheckpointBackup:
    def __init__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        state_path: Path,
        remote_prefix: str = "training-backups",
        token_env: str = "HF_TOKEN",
        max_attempts: int = 4,
        backoff_seconds: float = 2.0,
        client: BackupClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        catalog_metadata: Mapping[str, Any] | None = None,
    ):
        if repo_type not in {"model", "dataset"}:
            raise BackupConfigurationError("repo_type must be 'model' or 'dataset'")
        if not repo_id:
            raise BackupConfigurationError("an existing Hugging Face repo_id is required")
        token = os.environ.get(token_env)
        if client is None and not token:
            raise BackupConfigurationError(f"missing Hugging Face credential in {token_env}")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.state_path = state_path
        self.remote_prefix = remote_prefix.strip("/")
        prefix_path = PurePosixPath(self.remote_prefix)
        if not self.remote_prefix or prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise BackupConfigurationError("remote_prefix must be a safe relative repository path")
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.client = client or HuggingFaceBackupClient(token=token or "")
        self.sleep = sleep
        self.catalog_metadata = dict(catalog_metadata or {})
        self._destination_checked = False

    def validate_destination(self) -> None:
        if self._destination_checked:
            return
        try:
            private = self.client.repo_is_private(self.repo_id, self.repo_type)
        except Exception as exc:
            raise BackupConfigurationError(
                f"cannot verify existing Hugging Face {self.repo_type} repository {self.repo_id!r}"
            ) from exc
        if not private:
            raise BackupConfigurationError(
                f"refusing checkpoint backup to non-private repository {self.repo_id!r}"
            )
        self._destination_checked = True

    def _state(self) -> dict[str, Any]:
        destination = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "remote_prefix": self.remote_prefix,
        }
        state = read_json(
            self.state_path,
            {"schema_version": STATE_SCHEMA, "destination": destination, "checkpoints": {}},
        )
        if state.get("schema_version") == 1:
            if state.get("checkpoints"):
                raise BackupConfigurationError(
                    "legacy backup state has unbound receipts; preserve it and configure a new state_path"
                )
            state = {"schema_version": STATE_SCHEMA, "destination": destination, "checkpoints": {}}
        if state.get("schema_version") != STATE_SCHEMA:
            raise BackupError(f"unsupported backup state schema in {self.state_path}")
        if state.get("destination") != destination:
            raise BackupConfigurationError(
                "backup state belongs to a different repo_id, repo_type, or remote_prefix"
            )
        state.setdefault("checkpoints", {})
        return state

    def _save(self, state: Mapping[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def protect(
        self,
        *,
        job_id: str,
        checkpoint_id: str,
        step: int,
        paths: Iterable[Path],
        final: bool,
    ) -> str:
        self.validate_destination()
        path_list = list(paths)
        local_files = _files(path_list)
        primary = Path(path_list[0]).resolve()
        fingerprint_source = "\n".join(
            f"{path.resolve()}:{path.stat().st_size}:{sha256_file(path)}" for path in local_files
        )
        content_fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
        qualified_checkpoint_id = f"{job_id}--{checkpoint_id}--{content_fingerprint[:12]}"
        key = qualified_checkpoint_id
        state = self._state()
        existing = state["checkpoints"].get(key)
        if existing and existing.get("status") == "backed_up":
            expected = {
                str(path.resolve()): (path.stat().st_size, sha256_file(path)) for path in local_files
            }
            recorded = {
                str(Path(item["local_path"]).resolve()): (int(item["size"]), item["sha256"])
                for item in existing.get("artifacts", [])
                if item.get("role") != "manifest"
            }
            if expected == recorded:
                return str(existing["commit_id"])
            raise BackupError("verified checkpoint receipt does not match current local artifacts")

        from .catalog import CatalogStore

        model_name = str(self.catalog_metadata.get("name") or job_id)
        catalog = CatalogStore(
            client=self.client,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            catalog_path=str(PurePosixPath(self.remote_prefix) / "catalog.json"),
            work_dir=self.state_path.parent,
        )
        model, _ = catalog.ensure_model(
            {
                "name": model_name,
                "base_arch": self.catalog_metadata.get("base_arch"),
                "base_model": self.catalog_metadata.get("base_model"),
                "trigger_word": self.catalog_metadata.get("trigger_word"),
                "destination_kind": self.catalog_metadata.get("destination_kind", "loras"),
            }
        )
        expected_id = self.catalog_metadata.get("expected_id")
        if expected_id is not None and int(model["id"]) != int(expected_id):
            raise BackupConfigurationError(
                f"catalog model {model_name!r} has id {model['id']}, expected pre-reserved id {expected_id}"
            )
        remote_root = PurePosixPath(self.remote_prefix) / "models" / model["folder"] / "checkpoints" / qualified_checkpoint_id
        artifacts: list[LocalArtifact] = []
        used: set[str] = set()
        for path in local_files:
            is_primary = path.resolve() == primary or (primary.is_dir() and primary in path.resolve().parents)
            step_weight = (
                checkpoint_id.replace("step-", "_").replace("-final", "") in path.name
                and path.suffix in {".safetensors", ".pt"}
                and not path.name.endswith("_optimizer.pt")
                and path.name != "optimizer.pt"
            )
            category = "weights" if is_primary or step_weight else "resume"
            if primary.is_dir() and (path.resolve() == primary or primary in path.resolve().parents):
                local_relative = Path(primary.name) / path.resolve().relative_to(primary)
            else:
                local_relative = Path(path.name)
            remote_name = str(remote_root / category / PurePosixPath(*local_relative.parts))
            if remote_name in used:
                remote_name = str(remote_root / category / f"{sha256_file(path)[:12]}-{path.name}")
                local_relative = Path(f"{sha256_file(path)[:12]}-{path.name}")
            used.add(remote_name)
            artifacts.append(
                LocalArtifact(
                    str(path), remote_name, path.stat().st_size, sha256_file(path),
                    role=category, relative_path=local_relative.as_posix(),
                )
            )

        manifest_path = self.state_path.parent / "manifests" / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"
        manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "catalog_checkpoint_id": qualified_checkpoint_id,
            "step": int(step),
            "final": bool(final),
            "artifacts": [item.__dict__ for item in artifacts],
        }
        atomic_write_json(manifest_path, manifest)
        manifest_remote = str(remote_root / "manifest.json")
        artifacts.append(
            LocalArtifact(
                str(manifest_path), manifest_remote, manifest_path.stat().st_size, sha256_file(manifest_path),
                role="manifest", relative_path="manifest.json",
            )
        )
        state["checkpoints"][key] = {
            "status": "pending",
            "job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "catalog_checkpoint_id": qualified_checkpoint_id,
            "content_fingerprint": content_fingerprint,
            "step": int(step),
            "final": bool(final),
            "artifacts": [item.__dict__ for item in artifacts],
            "attempts": int(existing.get("attempts", 0)) if existing else 0,
            "catalog_name": model_name,
            "catalog_folder": model["folder"],
        }
        self._save(state)
        return self._upload_pending(key)

    def resume_pending(self) -> list[str]:
        self.validate_destination()
        completed = []
        for key, entry in sorted(self._state()["checkpoints"].items()):
            if entry.get("status") in {"pending", "uploaded_pending_catalog"}:
                completed.append(self._upload_pending(key))
        return completed

    def _upload_pending(self, key: str) -> str:
        from .catalog import CatalogStore

        last_error: Exception | None = None
        for attempt_index in range(self.max_attempts):
            state = self._state()
            entry = state["checkpoints"][key]
            catalog = CatalogStore(
                client=self.client,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                catalog_path=str(PurePosixPath(self.remote_prefix) / "catalog.json"),
                work_dir=self.state_path.parent,
            )
            if entry.get("status") == "uploaded_pending_catalog":
                try:
                    catalog.add_checkpoint(entry["catalog_name"], entry["checkpoint_record"])
                    state = self._state()
                    state["checkpoints"][key].update(
                        {"status": "backed_up", "commit_id": entry["upload_commit_id"], "verified": True, "cataloged": True}
                    )
                    self._save(state)
                    return str(entry["upload_commit_id"])
                except Exception as exc:
                    last_error = exc
                    if attempt_index + 1 < self.max_attempts:
                        self.sleep(self.backoff_seconds * (2**attempt_index))
                    continue
            artifacts = [LocalArtifact(**item) for item in entry["artifacts"]]
            for artifact in artifacts:
                path = Path(artifact.local_path)
                if not path.is_file() or path.stat().st_size != artifact.size or sha256_file(path) != artifact.sha256:
                    raise BackupError(f"pending artifact changed or disappeared: {path}")
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            self._save(state)
            try:
                commit_id = self.client.commit_files(
                    self.repo_id,
                    self.repo_type,
                    artifacts,
                    f"Backup {entry['job_id']} checkpoint {entry['checkpoint_id']}",
                )
                verify_remote_artifacts(
                    self.client,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    artifacts=artifacts,
                    revision=commit_id,
                    context="remote",
                )
                state = self._state()
                entry = state["checkpoints"][key]
                weights = []
                resume_artifacts = []
                for artifact in artifacts:
                    if artifact.role == "manifest":
                        continue
                    common = {
                        "remote_path": artifact.remote_path,
                        "size": artifact.size,
                        "sha256": artifact.sha256,
                    }
                    resume_artifacts.append(
                        {**common, "training_relative_path": artifact.relative_path}
                    )
                    if artifact.role == "weights":
                        weights.append(
                            {
                                **common,
                                "generation_relative_path": f"{entry['catalog_folder']}/{artifact.relative_path}",
                            }
                        )
                checkpoint_record = {
                    "checkpoint_id": entry["catalog_checkpoint_id"],
                    "source_checkpoint_id": entry["checkpoint_id"],
                    "job_id": entry["job_id"],
                    "step": int(entry["step"]),
                    "final": bool(entry["final"]),
                    "revision": commit_id,
                    "training_layout": {
                        "root_kind": "ai_toolkit_job_save_root",
                        "job_id": entry["job_id"],
                    },
                    "weights": weights,
                    "resume_artifacts": resume_artifacts,
                }
                entry.update({"status": "uploaded_pending_catalog", "upload_commit_id": commit_id, "checkpoint_record": checkpoint_record})
                self._save(state)
                catalog.add_checkpoint(entry["catalog_name"], checkpoint_record)
                state = self._state()
                entry = state["checkpoints"][key]
                entry.update({"status": "backed_up", "commit_id": commit_id, "verified": True, "cataloged": True})
                self._save(state)
                return commit_id
            except Exception as exc:
                last_error = exc
                if attempt_index + 1 < self.max_attempts:
                    self.sleep(self.backoff_seconds * (2**attempt_index))
        raise BackupError(f"checkpoint backup failed after {self.max_attempts} attempts") from last_error

    def can_delete(self, path: Path) -> bool:
        """Return true only when current bytes are present in a verified backup."""
        candidates = _files([path]) if path.exists() else []
        sidecar = path.with_suffix(".yaml")
        if sidecar.is_file():
            candidates.append(sidecar.resolve())
        backed: dict[str, tuple[int, str]] = {}
        for entry in self._state()["checkpoints"].values():
            if entry.get("status") != "backed_up" or not entry.get("verified"):
                continue
            for item in entry.get("artifacts", []):
                backed[str(Path(item["local_path"]).resolve())] = (int(item["size"]), item["sha256"])
        return bool(candidates) and all(
            str(item) in backed
            and item.stat().st_size == backed[str(item)][0]
            and sha256_file(item) == backed[str(item)][1]
            for item in candidates
        )
