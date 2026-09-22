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
PATH_METADATA_BATCH_SIZE = 100
EVIDENCE_JOB_CONFIG_ENV = "TRAINING_AUTOMATION_JOB_CONFIG"
# Provenance of the per-checkpoint evidence contract. Every receipt this version
# creates carries it, so a consumer can tell a receipt that must have its own
# evidence from one written before the contract existed. It is an additive
# optional field: STATE_SCHEMA deliberately stays 2, because bumping it would
# make `_state` refuse every existing schema-2 state that already holds receipts,
# including the one on the pod that is training right now.
EVIDENCE_CONTRACT = 1


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
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(paths), PATH_METADATA_BATCH_SIZE):
            entries = self._api.get_paths_info(
                repo_id=repo_id,
                repo_type=repo_type,
                paths=paths[start:start + PATH_METADATA_BATCH_SIZE],
                revision=revision,
            )
            for entry in entries:
                lfs = getattr(entry, "lfs", None)
                lfs_sha = None
                if isinstance(lfs, dict):
                    lfs_sha = lfs.get("sha256") or lfs.get("oid")
                elif lfs is not None:
                    lfs_sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
                metadata = {
                    "size": getattr(entry, "size", None),
                    "sha256": lfs_sha,
                }
                blob_id = getattr(entry, "blob_id", None)
                if blob_id:
                    metadata["blob_id"] = blob_id
                result[str(entry.path)] = metadata
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

    def list_repo_files(self, repo_id: str, repo_type: str, revision: str) -> list[str]:
        return list(self._api.list_repo_files(
            repo_id=repo_id, repo_type=repo_type, revision=revision,
        ))


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
                self._flush_evidence_quietly()
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
            "evidence_contract": EVIDENCE_CONTRACT,
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
        commit_id = self._upload_pending(key)
        # The images of this step do not exist yet; this pass publishes the
        # evidence of the checkpoints whose sample run completed in the meantime.
        self._flush_evidence_quietly()
        return commit_id

    def _flush_evidence_quietly(self) -> None:
        """Publish ready checkpoint evidence without ever stopping training.

        A failure here is recorded in the backup state and retried by the next
        checkpoint. The strict pass the queue runs after training refuses to
        complete a job whose checkpoints still have no published evidence.
        """
        try:
            settings = self._evidence_settings(None)
            if settings is None or not settings["during_training"]:
                # The queue still runs the strict pass once training has exited,
                # so the evidence is complete either way; only the protection
                # against losing the machine mid-run is traded away.
                return
            self.publish_pending_evidence(strict=False)
        except Exception as exc:  # pragma: no cover - defensive during paid training
            print(f"training_automation: deferred checkpoint evidence failed: {type(exc).__name__}: {exc}")

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

    # ------------------------------------------------------------------
    # Per-checkpoint evidence
    #
    # The trainer saves a checkpoint and only then samples that same step, so the
    # images of step N do not exist yet when step N is protected. Evidence is
    # therefore published in a separate pass: every protect() call flushes the
    # checkpoints whose sample run has since completed, and the queue runs one
    # strict pass after the training process exits so the last checkpoint is
    # covered too. Nothing already published is deleted or overwritten.
    # ------------------------------------------------------------------

    def _evidence_settings(self, job_config_path: Path | None) -> dict[str, Any] | None:
        import yaml

        if job_config_path is None:
            value = os.environ.get(EVIDENCE_JOB_CONFIG_ENV)
            if not value:
                return None
            job_config_path = Path(value)
        job_config_path = Path(job_config_path)
        if not job_config_path.is_file():
            raise BackupConfigurationError(
                f"checkpoint evidence needs the generated job config: {job_config_path}"
            )
        document = yaml.safe_load(job_config_path.read_text(encoding="utf-8"))
        try:
            process = document["config"]["process"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackupConfigurationError(
                f"generated job config has no config.process[0]: {job_config_path}"
            ) from exc
        evidence = (process.get("checkpoint_backup") or {}).get("evidence") or {}
        if not evidence.get("enabled", False):
            return None
        output_dir = evidence.get("output_dir")
        if not output_dir:
            raise BackupConfigurationError("checkpoint evidence requires an explicit output_dir")
        # Lineage of a declared refinement phase. It is an additive optional
        # field written only by a job that declares one: EVIDENCE_CONTRACT stays
        # 1 because nothing an earlier version wrote becomes unreadable, and an
        # older image reading a newer queue configuration refuses the declaration
        # by name instead of training something it cannot describe.
        refinement_phase = (process.get("checkpoint_backup") or {}).get("refinement_phase")
        return {
            "job_config_path": job_config_path,
            "output_dir": Path(str(output_dir)),
            "reference_images": [Path(str(item)) for item in evidence.get("reference_images", [])],
            "evaluation": dict(evidence.get("evaluation") or {}),
            "during_training": bool(evidence.get("during_training", True)),
            "refinement_phase": dict(refinement_phase) if refinement_phase else None,
        }

    def _evidence_evaluator(self, settings: Mapping[str, Any]):
        from .evaluation import CheckpointEvidenceEvaluator

        cached = getattr(self, "_cached_evidence_evaluator", None)
        if cached is not None and cached[0] == str(settings["job_config_path"]):
            return cached[1]
        evaluator = CheckpointEvidenceEvaluator(
            job_config_path=settings["job_config_path"],
            output_dir=settings["output_dir"],
            reference_images=list(settings["reference_images"]),
            config=settings["evaluation"],
        )
        self._cached_evidence_evaluator = (str(settings["job_config_path"]), evaluator)
        return evaluator

    def publish_pending_evidence(
        self, *, job_config_path: Path | None = None, strict: bool = False
    ) -> list[dict[str, Any]]:
        """Upload samples and the evaluation record of every ready checkpoint.

        With ``strict`` false a checkpoint whose sample run is not complete yet is
        left pending and retried by the next pass. With ``strict`` true every
        verified checkpoint must have its own images; the honest incomplete status
        is published as it is, and a checkpoint with no image at all is an error.
        """
        settings = self._evidence_settings(job_config_path)
        if settings is None:
            return []
        published: list[dict[str, Any]] = []
        for key in sorted(self._state()["checkpoints"]):
            result = self._publish_checkpoint_evidence(key, settings, strict=strict)
            if result is not None:
                published.append(result)
        return published

    def _publish_checkpoint_evidence(
        self, key: str, settings: Mapping[str, Any], *, strict: bool
    ) -> dict[str, Any] | None:
        state = self._state()
        entry = state["checkpoints"].get(key)
        if entry is None:
            return None
        if (
            entry.get("status") != "backed_up"
            or not entry.get("verified")
            or not entry.get("cataloged")
        ):
            return None
        existing = entry.get("evidence") or {}
        if existing.get("status") == "published":
            return None
        evaluator = self._evidence_evaluator(settings)
        record = evaluator.evaluate(step=int(entry["step"]), final=bool(entry["final"]))
        if settings.get("refinement_phase"):
            record["refinement_phase"] = dict(settings["refinement_phase"])
        if record["sample_run_status"] == "missing":
            if not strict:
                return None
            raise BackupError(
                f"checkpoint {entry['catalog_checkpoint_id']} has no sample image at step {entry['step']}"
            )
        if record["sample_run_status"] != "complete" and not strict:
            state = self._state()
            state["checkpoints"][key]["evidence"] = {
                "status": "pending",
                "reason": "the sample run of this step is not complete yet",
                "sample_run_status": record["sample_run_status"],
            }
            self._save(state)
            return None
        try:
            return self._upload_checkpoint_evidence(key, entry, record)
        except Exception as exc:
            state = self._state()
            state["checkpoints"][key]["evidence"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "sample_run_status": record["sample_run_status"],
            }
            self._save(state)
            if strict:
                raise
            return None

    def _upload_checkpoint_evidence(
        self, key: str, entry: Mapping[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        self.validate_destination()
        remote_root = (
            PurePosixPath(self.remote_prefix)
            / "models" / str(entry["catalog_folder"])
            / "checkpoints" / str(entry["catalog_checkpoint_id"])
        )
        artifacts: list[LocalArtifact] = []
        used: set[str] = set()
        for sample in record["samples"]:
            local = Path(sample["path"])
            if not local.is_file():
                raise BackupError(f"sample evidence disappeared before upload: {local}")
            name = local.name
            if name in used:
                raise BackupError(f"duplicate sample evidence filename: {name}")
            used.add(name)
            remote_path = str(remote_root / "samples" / name)
            digest = sha256_file(local)
            size = local.stat().st_size
            sample.update({
                "file": name,
                "remote_path": remote_path,
                "size": size,
                "sha256": digest,
            })
            artifacts.append(
                LocalArtifact(
                    str(local), remote_path, size, digest,
                    role="evidence-sample", relative_path=f"samples/{name}",
                )
            )
        record.update({
            "job_id": entry["job_id"],
            "checkpoint_id": entry["checkpoint_id"],
            "catalog_checkpoint_id": entry["catalog_checkpoint_id"],
            "catalog_name": entry["catalog_name"],
            "catalog_folder": entry["catalog_folder"],
            "checkpoint_revision": entry.get("commit_id"),
            "destination": {
                "repo_id": self.repo_id,
                "repo_type": self.repo_type,
                "remote_prefix": self.remote_prefix,
            },
        })
        record_path = (
            self.state_path.parent / "evidence" / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"
        )
        atomic_write_json(record_path, record)
        record_remote = str(remote_root / "evaluation.json")
        artifacts.append(
            LocalArtifact(
                str(record_path), record_remote, record_path.stat().st_size,
                sha256_file(record_path), role="evidence", relative_path="evaluation.json",
            )
        )
        last_error: Exception | None = None
        for attempt_index in range(self.max_attempts):
            try:
                commit_id = self.client.commit_files(
                    self.repo_id,
                    self.repo_type,
                    artifacts,
                    f"Evidence {entry['job_id']} checkpoint {entry['checkpoint_id']}",
                )
                verify_remote_artifacts(
                    self.client,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    artifacts=artifacts,
                    revision=commit_id,
                    context="checkpoint evidence",
                )
                evidence = {
                    "status": "published",
                    "commit_id": commit_id,
                    "verified": True,
                    "sample_run_status": record["sample_run_status"],
                    "sample_count": len(record["samples"]),
                    "expected_sample_count": int(record["expected_sample_count"]),
                    "score": record["score"],
                    "evaluation_remote_path": record_remote,
                    "sample_remote_paths": [
                        item.remote_path for item in artifacts if item.role == "evidence-sample"
                    ],
                    "local_record_path": str(record_path),
                }
                state = self._state()
                state["checkpoints"][key]["evidence"] = evidence
                self._save(state)
                return {"checkpoint_key": key, **evidence}
            except Exception as exc:
                last_error = exc
                if attempt_index + 1 < self.max_attempts:
                    self.sleep(self.backoff_seconds * (2**attempt_index))
        raise BackupError(
            f"checkpoint evidence upload failed after {self.max_attempts} attempts"
        ) from last_error

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
