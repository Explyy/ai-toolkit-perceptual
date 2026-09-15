from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image

from .backup import BackupError, LocalArtifact, sha256_file
from .catalog import safe_name, safe_relative_path
from .state import atomic_write_json, read_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_NAME_RE = re.compile(r"^(?!\.)(?!.*[\\/])(?!.*[\x00-\x1f]).{1,128}$")
LEDGER_SCHEMA = 1


class DiscoveryClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...
    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...
    def commit_files(
        self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact],
        message: str, parent_commit: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class DatasetSnapshot:
    folder: str
    name: str
    trigger_word: str
    fingerprint: str
    image_count: int
    files: tuple[str, ...]
    loader_batches_per_epoch: int
    loader_epochs: int
    training_steps: int
    original_image_exposures: int

    def accounting(self) -> dict[str, Any]:
        return {
            "source_image_count": self.image_count,
            "loader_batches_per_epoch": self.loader_batches_per_epoch,
            "loader_epochs": self.loader_epochs,
            "resolution_repeats": [16, 4, 1],
            "batch_size": 4,
            "original_image_exposures": self.original_image_exposures,
            "partial_bucket_batches": "un-padded",
        }


def _bucket(width: int, height: int, resolution: int, divisibility: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise BackupError("dataset image has invalid dimensions")
    pixels = width * height
    target = min(pixels, resolution * resolution)
    scale = (target / pixels) ** 0.5
    raw_width = width * scale / divisibility
    raw_height = height * scale / divisibility
    choices = [
        (width_round * divisibility, height_round * divisibility)
        for width_round in (math.floor(raw_width), math.ceil(raw_width))
        for height_round in (math.floor(raw_height), math.ceil(raw_height))
        if width_round > 0
        and height_round > 0
        and width_round * height_round * divisibility * divisibility <= resolution * resolution
    ]
    if not choices:
        choices = [(
            max(divisibility, math.floor(raw_width) * divisibility),
            max(divisibility, math.floor(raw_height) * divisibility),
        )]
    return min(choices, key=lambda size: abs(size[0] * size[1] - target))


def _snapshot_folder(
    folder: Path,
    *,
    exposures: int,
    batch_size: int,
    resolutions: Sequence[int],
    repeats: Sequence[int],
    bucket_divisibility: int,
    default_trigger_word: str,
) -> DatasetSnapshot:
    if folder.is_symlink() or folder.resolve().parent != folder.parent.resolve():
        raise BackupError(f"dataset folder is a symlink or escapes its root: {folder.name}")
    if not DATASET_NAME_RE.fullmatch(folder.name):
        raise BackupError(f"dataset folder name contains unsupported characters: {folder.name}")
    entries = sorted(folder.iterdir(), key=lambda path: path.name.casefold())
    casefolded = [path.name.casefold() for path in entries]
    if len(casefolded) != len(set(casefolded)):
        raise BackupError(f"dataset folder contains case-ambiguous filenames: {folder.name}")
    if any(path.is_symlink() for path in entries):
        raise BackupError(f"dataset folder contains a symlink: {folder.name}")
    if any(path.is_dir() for path in entries):
        raise BackupError(f"dataset folder must contain files directly: {folder.name}")
    images = [path for path in entries if path.suffix.casefold() in IMAGE_SUFFIXES]
    if not images:
        raise BackupError(f"dataset folder contains no supported images: {folder.name}")
    allowed = {path.name for path in images}
    stems = [path.stem.casefold() for path in images]
    if len(stems) != len(set(stems)):
        raise BackupError(f"dataset folder contains images with duplicate caption stems: {folder.name}")
    captions = []
    for image in images:
        caption = image.with_suffix(".txt")
        if not caption.is_file() or caption.is_symlink():
            raise BackupError(f"dataset image has no safe matching caption: {image.name}")
        if not caption.read_text(encoding="utf-8").strip():
            raise BackupError(f"dataset caption is empty: {caption.name}")
        captions.append(caption)
        allowed.add(caption.name)
    metadata_path = folder / ".training-automation.json"
    metadata: Mapping[str, Any] = {}
    if metadata_path.is_file() and not metadata_path.is_symlink():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping) or set(metadata) - {"catalog_name", "trigger_word"}:
            raise BackupError(f"dataset metadata has unsupported fields: {folder.name}")
        allowed.add(metadata_path.name)
    name = str(metadata.get("catalog_name") or folder.name).strip()
    trigger_word = str(metadata.get("trigger_word") or default_trigger_word).strip()
    if not name or not trigger_word:
        raise BackupError(f"dataset catalog name and trigger word must be nonempty: {folder.name}")
    extras = [path.name for path in entries if path.name not in allowed and not path.name.startswith(".")]
    if extras:
        raise BackupError(f"dataset folder contains unsupported files: {folder.name}: {extras}")

    digest = hashlib.sha256()
    buckets: dict[tuple[int, int, int], int] = {}
    for image in images:
        try:
            with Image.open(image) as opened:
                opened.load()
                width, height = opened.size
        except Exception as exc:
            raise BackupError(f"dataset image is unreadable: {image.name}") from exc
        for resolution, repeat in zip(resolutions, repeats):
            width_bucket, height_bucket = _bucket(
                width, height, int(resolution), bucket_divisibility
            )
            key = (int(resolution), width_bucket, height_bucket)
            buckets[key] = buckets.get(key, 0) + int(repeat)
    for path in sorted([*images, *captions], key=lambda item: item.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    loader_batches = sum(math.ceil(count / batch_size) for count in buckets.values())
    loader_epochs = max(1, round(exposures / sum(int(value) for value in repeats)))
    return DatasetSnapshot(
        folder=folder.name,
        name=name,
        trigger_word=trigger_word,
        fingerprint=digest.hexdigest(),
        image_count=len(images),
        files=tuple(path.name for path in sorted([*images, *captions], key=lambda item: item.name.casefold())),
        loader_batches_per_epoch=loader_batches,
        loader_epochs=loader_epochs,
        training_steps=loader_batches * loader_epochs,
        original_image_exposures=sum(int(value) for value in repeats) * loader_epochs,
    )


def scan_dataset_root(
    root: Path,
    *,
    exposures: int = 126,
    batch_size: int = 4,
    resolutions: Sequence[int] = (512, 768, 1024),
    repeats: Sequence[int] = (16, 4, 1),
    bucket_divisibility: int = 16,
    default_trigger_word: str = "Owhx",
) -> list[DatasetSnapshot]:
    root = root.resolve()
    if not root.is_dir():
        raise BackupError(f"configured dataset root is missing: {root}")
    if len(resolutions) != len(repeats) or not resolutions or any(int(value) <= 0 for value in repeats):
        raise BackupError("dataset resolutions and repeats must be nonempty matching positive lists")
    snapshots = [
        _snapshot_folder(
            path,
            exposures=exposures,
            batch_size=batch_size,
            resolutions=resolutions,
            repeats=repeats,
            bucket_divisibility=bucket_divisibility,
            default_trigger_word=default_trigger_word,
        )
        for path in root.iterdir()
        if path.is_dir() or path.is_symlink()
    ]
    normalized = [safe_name(item.name) for item in snapshots]
    if len(normalized) != len(set(normalized)):
        raise BackupError("dataset folders resolve to duplicate normalized model names")
    return sorted(snapshots, key=lambda item: (safe_name(item.name), item.folder.casefold()))


def assigned_worker(fingerprint: str, worker_count: int, folder: str = "") -> int:
    if worker_count <= 0:
        raise BackupError("worker_count must be positive")
    identity = hashlib.sha256(f"{folder}\0{fingerprint}".encode("utf-8")).hexdigest()
    return int(identity[:16], 16) % worker_count


class WorkflowLedgerStore:
    def __init__(
        self, *, client: DiscoveryClient, repo_id: str, repo_type: str,
        remote_path: str, local_path: Path,
    ):
        self.client = client
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.remote_path = safe_relative_path(remote_path).as_posix()
        self.local_path = local_path

    def read(self) -> tuple[dict[str, Any], str]:
        payload, revision = self.client.read_remote_file(
            self.repo_id, self.repo_type, self.remote_path
        )
        ledger = (
            {"schema_version": LEDGER_SCHEMA, "datasets": {}}
            if payload is None else json.loads(payload)
        )
        if ledger.get("schema_version") != LEDGER_SCHEMA or not isinstance(ledger.get("datasets"), dict):
            raise BackupError("unsupported or invalid remote workflow ledger")
        atomic_write_json(self.local_path, ledger)
        return ledger, revision

    def write(self, ledger: Mapping[str, Any], parent: str, message: str) -> str:
        atomic_write_json(self.local_path, ledger)
        artifact = LocalArtifact(
            str(self.local_path), self.remote_path, self.local_path.stat().st_size,
            sha256_file(self.local_path), role="workflow-ledger",
        )
        return self.client.commit_files(
            self.repo_id, self.repo_type, [artifact], message, parent_commit=parent
        )

    def reconcile(
        self,
        snapshots: Sequence[DatasetSnapshot],
        *,
        worker_count: int,
        quiet_seconds: float,
        now: float,
        legacy_completed: Mapping[str, Any] | None = None,
        verified_stable_fingerprints: Sequence[str] = (),
        attempts: int = 5,
    ) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        verified_fingerprints = set(verified_stable_fingerprints)
        for _ in range(attempts):
            ledger, parent = self.read()
            datasets = ledger["datasets"]
            for folder, record in (legacy_completed or {}).items():
                already_migrated = any(
                    item.get("fingerprint") == record.get("fingerprint")
                    for item in datasets.values()
                )
                if folder not in datasets and not already_migrated:
                    datasets[folder] = dict(record)
            for snapshot in snapshots:
                current = datasets.get(snapshot.folder)
                if current is None and snapshot.fingerprint in verified_fingerprints:
                    aliases = [
                        (folder, item) for folder, item in datasets.items()
                        if item.get("fingerprint") == snapshot.fingerprint
                    ]
                    if len(aliases) > 1:
                        raise BackupError(
                            f"persisted dataset identity is ambiguous for {snapshot.folder}"
                        )
                    if aliases:
                        previous_folder, current = aliases[0]
                        if previous_folder != snapshot.folder:
                            datasets.pop(previous_folder)
                            current["folder"] = snapshot.folder
                            current["name"] = snapshot.name
                            current["trigger_word"] = snapshot.trigger_word
                            current["folder_aliases"] = sorted(set([
                                *current.get("folder_aliases", []), previous_folder,
                            ]))
                            datasets[snapshot.folder] = current
                snapshot_data = asdict(snapshot)
                snapshot_data["files"] = list(snapshot_data["files"])
                if current is None:
                    verified_remote = snapshot.fingerprint in verified_fingerprints
                    datasets[snapshot.folder] = {
                        **snapshot_data,
                        "status": "ready" if verified_remote else "observing",
                        "first_observed_at": now,
                        **({"stable_at": now, "source": "verified-remote-dataset"} if verified_remote else {}),
                        "worker": assigned_worker(
                            snapshot.fingerprint, worker_count, snapshot.folder
                        ),
                    }
                    continue
                if current.get("fingerprint") != snapshot.fingerprint:
                    if current.get("status") == "observing":
                        datasets[snapshot.folder] = {
                            **snapshot_data,
                            "status": "observing",
                            "first_observed_at": now,
                            "worker": assigned_worker(
                                snapshot.fingerprint, worker_count, snapshot.folder
                            ),
                        }
                        continue
                    current.update({
                        "status": "changed",
                        "observed_fingerprint": snapshot.fingerprint,
                        "observed_at": now,
                        "reason": "dataset content differs from its persisted identity",
                    })
                    continue
                if current.get("status") == "observing":
                    if now - float(current.get("first_observed_at", now)) >= quiet_seconds:
                        current["status"] = "ready"
                        current["stable_at"] = now
                for field, value in snapshot_data.items():
                    if field not in current and current.get("legacy"):
                        current[field] = value
                    elif current.get(field) != value:
                        raise BackupError(f"persisted dataset snapshot field drift: {snapshot.folder}/{field}")
            ledger["datasets"] = dict(sorted(datasets.items(), key=lambda item: item[0].casefold()))
            try:
                revision = self.write(ledger, parent, "Reconcile automatic training datasets")
                return ledger, revision
            except Exception as exc:
                last_error = exc
        raise BackupError("workflow ledger conflicted repeatedly") from last_error

    def update_status(
        self, folder: str, fingerprint: str, status: str,
        *, details: Mapping[str, Any] | None = None, attempts: int = 5,
    ) -> str:
        last_error: Exception | None = None
        for _ in range(attempts):
            ledger, parent = self.read()
            record = ledger["datasets"].get(folder)
            if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
                raise BackupError(f"workflow ledger identity changed for {folder}")
            if record.get("status") == "completed" and status != "completed":
                raise BackupError(f"refusing to regress completed workflow state for {folder}")
            record["status"] = status
            record.update(dict(details or {}))
            try:
                return self.write(ledger, parent, f"Record {folder} workflow status {status}")
            except Exception as exc:
                last_error = exc
        raise BackupError("workflow ledger status update conflicted repeatedly") from last_error


def load_legacy_completed(
    client: Any, *, repo_id: str, repo_type: str, remote_path: str,
    revision: str, work_dir: Path,
) -> dict[str, Any]:
    from .sync import _json_at_revision

    document = _json_at_revision(
        client, repo_id=repo_id, repo_type=repo_type,
        remote_path=safe_relative_path(remote_path).as_posix(), revision=revision,
        work_dir=work_dir,
    )
    if document.get("schema_version") != 1 or not isinstance(document.get("datasets"), dict):
        raise BackupError("legacy completed dataset index is invalid")
    result: dict[str, Any] = {}
    for folder, item in document["datasets"].items():
        safe_relative_path(folder)
        if not isinstance(item, Mapping) or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("fingerprint", ""))):
            raise BackupError("legacy completed dataset index contains invalid identity")
        result[folder] = {**dict(item), "folder": folder, "status": "completed", "legacy": True}
    return result
