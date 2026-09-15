from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from .backup import BackupError, LocalArtifact, sha256_file, verify_remote_artifacts
from .catalog import CatalogStore, resolve_model, safe_name, safe_relative_path
from .discovery import IMAGE_SUFFIXES, DatasetSnapshot, _snapshot_folder
from .state import atomic_write_json


DATASET_CATALOG_SCHEMA = 1
DATASET_MANIFEST_SCHEMA = 1
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
REMOTE_MANIFEST = ".training-automation.json"


class DatasetStorageClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...
    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...
    def list_repo_files(
        self, repo_id: str, repo_type: str, revision: str
    ) -> list[str]: ...
    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]: ...
    def download_file(
        self, repo_id: str, repo_type: str, path: str,
        revision: str, destination: Path,
    ) -> None: ...
    def commit_files(
        self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact],
        message: str, parent_commit: str | None = None,
    ) -> str: ...


def _display_name(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if (
        not text or len(text) > 128 or "/" in text or "\\" in text
        or any(ord(character) < 32 for character in text)
    ):
        raise BackupError(f"remote dataset {field} is invalid")
    safe_name(text)
    return text


def _immutable_revision(value: Any, field: str) -> str:
    revision = str(value or "")
    if not COMMIT_RE.fullmatch(revision):
        raise BackupError(f"{field} must be an immutable 40-character commit SHA")
    return revision


def _git_blob_sha(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_download(path: Path, spec: Mapping[str, Any]) -> str:
    if path.stat().st_size != int(spec["size"]):
        raise BackupError(f"remote dataset size verification failed: {spec['remote_path']}")
    digest = sha256_file(path)
    expected = spec.get("sha256")
    if expected and digest != expected:
        raise BackupError(f"remote dataset SHA-256 verification failed: {spec['remote_path']}")
    blob_id = str(spec.get("blob_id") or "")
    if not expected and blob_id and _git_blob_sha(path) != blob_id:
        raise BackupError(f"remote dataset Git blob verification failed: {spec['remote_path']}")
    if not expected and not blob_id:
        raise BackupError(f"remote dataset file has no verifiable object identity: {spec['remote_path']}")
    return digest


def _download(
    client: DatasetStorageClient, *, repo_id: str, repo_type: str,
    revision: str, spec: Mapping[str, Any], destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            return _verify_download(destination, spec)
        except BackupError:
            destination.unlink()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        client.download_file(
            repo_id, repo_type, str(spec["remote_path"]), revision, temporary
        )
        digest = _verify_download(temporary, spec)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _verify_download(destination, spec) != digest:
                raise BackupError(f"conflicting staged dataset file: {destination}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(
    client: DatasetStorageClient, *, repo_id: str, repo_type: str,
    revision: str, spec: Mapping[str, Any], work_dir: Path,
) -> Mapping[str, Any]:
    target = work_dir / f"manifest-{hashlib.sha256(str(spec['remote_path']).encode()).hexdigest()}.json"
    _download(
        client, repo_id=repo_id, repo_type=repo_type, revision=revision,
        spec=spec, destination=target,
    )
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BackupError(f"remote dataset manifest is invalid: {spec['remote_path']}") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise BackupError(f"remote dataset manifest schema is invalid: {spec['remote_path']}")
    return document


def observe_remote_datasets(
    *, client: DatasetStorageClient, repo_id: str, repo_type: str,
    revision: str, remote_prefix: str, work_dir: Path,
    default_trigger_word: str = "Owhx",
) -> dict[str, dict[str, Any]]:
    """Describe every direct remote dataset folder at one immutable snapshot."""
    revision = _immutable_revision(revision, "remote dataset source revision")
    prefix = PurePosixPath(safe_relative_path(remote_prefix).as_posix())
    paths = [
        path for path in client.list_repo_files(repo_id, repo_type, revision)
        if PurePosixPath(path).parts[: len(prefix.parts)] == prefix.parts
        and len(PurePosixPath(path).parts) > len(prefix.parts)
    ]
    metadata = client.path_metadata(repo_id, repo_type, paths, revision)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        relative = PurePosixPath(path).relative_to(prefix)
        folder = relative.parts[0]
        grouped.setdefault(folder, []).append({
            "relative_path": PurePosixPath(*relative.parts[1:]).as_posix(),
            "remote_path": path,
            **dict(metadata.get(path) or {}),
        })
    work_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}
    for folder, raw_files in sorted(grouped.items(), key=lambda item: item[0].casefold()):
        remote_folder = str(prefix / folder)
        try:
            _display_name(folder, "source folder")
            if any(
                not item["relative_path"]
                or len(PurePosixPath(item["relative_path"]).parts) != 1
                for item in raw_files
            ):
                raise BackupError("remote dataset folders may contain direct files only")
            if len({str(item["relative_path"]).casefold() for item in raw_files}) != len(raw_files):
                raise BackupError("remote dataset contains case-ambiguous filenames")
            for item in raw_files:
                if not isinstance(item.get("size"), int) or int(item["size"]) < 0:
                    raise BackupError(f"remote dataset file lacks a valid size: {item['remote_path']}")
                sha = str(item.get("sha256") or "").removeprefix("sha256:").lower()
                blob = str(item.get("blob_id") or "").lower()
                if sha and not SHA256_RE.fullmatch(sha):
                    raise BackupError(f"remote dataset file has invalid SHA-256: {item['remote_path']}")
                if not sha and not GIT_OID_RE.fullmatch(blob):
                    raise BackupError(f"remote dataset file lacks an immutable object id: {item['remote_path']}")
                item["sha256"] = sha or None
                item["blob_id"] = blob or None
            manifest_spec = next(
                (item for item in raw_files if item["relative_path"] == REMOTE_MANIFEST), None
            )
            training = [item for item in raw_files if item is not manifest_spec]
            images = [
                item for item in training
                if Path(str(item["relative_path"])).suffix.casefold() in IMAGE_SUFFIXES
            ]
            captions = [
                item for item in training
                if Path(str(item["relative_path"])).suffix.casefold() == ".txt"
            ]
            if len(images) + len(captions) != len(training):
                extras = sorted(
                    item["relative_path"] for item in training
                    if item not in images and item not in captions
                )
                raise BackupError(f"remote dataset contains unsupported files: {extras}")
            if not images:
                raise BackupError("remote dataset contains no supported images")
            image_stems = {Path(str(item["relative_path"])).stem.casefold() for item in images}
            caption_stems = {Path(str(item["relative_path"])).stem.casefold() for item in captions}
            if len(image_stems) != len(images) or image_stems != caption_stems:
                raise BackupError("remote dataset images require one matching non-ambiguous caption")
            name = folder
            trigger_word = default_trigger_word
            declared: dict[str, Mapping[str, Any]] = {}
            if manifest_spec is not None:
                document = _manifest(
                    client, repo_id=repo_id, repo_type=repo_type, revision=revision,
                    spec=manifest_spec, work_dir=work_dir,
                )
                name = _display_name(document.get("catalog_name"), "catalog_name")
                trigger_word = _display_name(document.get("trigger_word"), "trigger_word")
                manifest_files = document.get("files")
                if not isinstance(manifest_files, list):
                    raise BackupError("remote dataset manifest requires files")
                for item in manifest_files:
                    if not isinstance(item, Mapping):
                        raise BackupError("remote dataset manifest file entry must be a mapping")
                    relative = safe_relative_path(str(item.get("relative_path") or ""))
                    if len(relative.parts) != 1 or relative.as_posix() in declared:
                        raise BackupError("remote dataset manifest repeats or nests a file")
                    sha = str(item.get("sha256") or "").lower()
                    if not SHA256_RE.fullmatch(sha) or not isinstance(item.get("size"), int):
                        raise BackupError("remote dataset manifest file identity is invalid")
                    declared[relative.as_posix()] = item
                if set(declared) != {str(item["relative_path"]) for item in training}:
                    raise BackupError("remote dataset manifest does not exactly cover its payload")
                for item in training:
                    declared_item = declared[str(item["relative_path"])]
                    if int(declared_item["size"]) != int(item["size"]):
                        raise BackupError("remote dataset manifest size differs from repository metadata")
                    if item.get("sha256") and item["sha256"] != declared_item["sha256"]:
                        raise BackupError("remote dataset manifest hash differs from repository metadata")
                    item["sha256"] = str(declared_item["sha256"])
            name = _display_name(name, "catalog_name")
            trigger_word = _display_name(trigger_word, "trigger_word")
            stable = {
                "remote_folder": remote_folder,
                "name": name,
                "trigger_word": trigger_word,
                "files": [
                    {
                        key: item.get(key)
                        for key in ("relative_path", "remote_path", "size", "sha256", "blob_id")
                    }
                    for item in sorted(training, key=lambda value: str(value["relative_path"]).casefold())
                ],
            }
            observation = hashlib.sha256(
                json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            result[remote_folder] = {
                **stable, "revision": revision,
                "observation_fingerprint": observation, "status": "valid",
            }
        except Exception as exc:
            result[remote_folder] = {
                "remote_folder": remote_folder, "revision": revision,
                "status": "held", "reason": f"{type(exc).__name__}: {exc}",
            }
    return result


class DatasetCatalogStore:
    def __init__(
        self, *, client: DatasetStorageClient, repo_id: str, repo_type: str,
        remote_path: str, work_dir: Path,
    ):
        self.client = client
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.remote_path = safe_relative_path(remote_path).as_posix()
        self.work_dir = work_dir

    def read(self) -> tuple[dict[str, Any], str]:
        payload, revision = self.client.read_remote_file(
            self.repo_id, self.repo_type, self.remote_path
        )
        catalog = (
            {"schema_version": DATASET_CATALOG_SCHEMA, "datasets": []}
            if payload is None else json.loads(payload)
        )
        if (
            not isinstance(catalog, Mapping)
            or catalog.get("schema_version") != DATASET_CATALOG_SCHEMA
            or not isinstance(catalog.get("datasets"), list)
        ):
            raise BackupError("remote dataset catalog is invalid")
        seen_ids: set[int] = set()
        seen_fingerprints: set[str] = set()
        seen_sources: set[str] = set()
        for item in catalog["datasets"]:
            if not isinstance(item, Mapping):
                raise BackupError("remote dataset catalog entry is invalid")
            numeric_id = int(item.get("id", 0))
            name = _display_name(item.get("name"), "catalog name")
            trigger_word = _display_name(item.get("trigger_word"), "catalog trigger_word")
            fingerprint = str(item.get("fingerprint") or "")
            canonical = str(item.get("canonical_folder") or "")
            if (
                numeric_id <= 0 or numeric_id in seen_ids
                or not SHA256_RE.fullmatch(fingerprint)
                or fingerprint in seen_fingerprints
                or canonical != f"{numeric_id:04d}-{safe_name(name)}"
                or not trigger_word
                or not isinstance(item.get("image_count"), int)
                or int(item["image_count"]) <= 0
                or not isinstance(item.get("sources"), list)
            ):
                raise BackupError("remote dataset catalog has conflicting stable identities")
            seen_ids.add(numeric_id)
            seen_fingerprints.add(fingerprint)
            for source in item["sources"]:
                folder = safe_relative_path(str(source.get("remote_folder") or "")).as_posix()
                _immutable_revision(source.get("revision"), "dataset catalog source revision")
                if folder in seen_sources or not isinstance(source.get("files"), list):
                    raise BackupError("remote dataset catalog repeats a source folder")
                seen_sources.add(folder)
                seen_files: set[str] = set()
                for file in source["files"]:
                    if not isinstance(file, Mapping):
                        raise BackupError("remote dataset catalog source file is invalid")
                    relative = safe_relative_path(str(file.get("relative_path") or ""))
                    remote = safe_relative_path(str(file.get("remote_path") or ""))
                    sha = str(file.get("sha256") or "")
                    if (
                        len(relative.parts) != 1
                        or relative.as_posix() in seen_files
                        or remote.as_posix() != f"{folder}/{relative.as_posix()}"
                        or not isinstance(file.get("size"), int)
                        or int(file["size"]) < 0
                        or not SHA256_RE.fullmatch(sha)
                    ):
                        raise BackupError("remote dataset catalog source file identity is invalid")
                    seen_files.add(relative.as_posix())
                if not seen_files:
                    raise BackupError("remote dataset catalog source contains no files")
        return dict(catalog), revision

    def _write(
        self, catalog: Mapping[str, Any], parent: str,
        *, model_catalog: Mapping[str, Any] | None = None,
        model_catalog_path: str | None = None,
    ) -> str:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        local = self.work_dir / "dataset-catalog.json"
        atomic_write_json(local, catalog)
        artifacts = [LocalArtifact(
            str(local), self.remote_path, local.stat().st_size, sha256_file(local),
            role="dataset-catalog",
        )]
        if model_catalog is not None:
            if not model_catalog_path:
                raise BackupError("model catalog path is required for atomic dataset registration")
            model_local = self.work_dir / "model-catalog.json"
            atomic_write_json(model_local, model_catalog)
            artifacts.append(LocalArtifact(
                str(model_local), model_catalog_path,
                model_local.stat().st_size, sha256_file(model_local), role="model-catalog",
            ))
        return self.client.commit_files(
            self.repo_id, self.repo_type, artifacts,
            "Register immutable training dataset", parent_commit=parent,
        )

    def source_record(self, remote_folder: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
        catalog, _ = self.read()
        matches = [
            (item, source) for item in catalog["datasets"] for source in item["sources"]
            if source.get("remote_folder") == remote_folder
        ]
        if len(matches) > 1:
            raise BackupError("remote dataset source resolves ambiguously")
        return matches[0] if matches else None

    def ensure(
        self, *, snapshot: DatasetSnapshot, source: Mapping[str, Any],
        model_store: CatalogStore, model_metadata: Mapping[str, Any], attempts: int = 5,
    ) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        for _ in range(attempts):
            catalog, parent = self.read()
            by_fingerprint = [
                item for item in catalog["datasets"]
                if item["fingerprint"] == snapshot.fingerprint
            ]
            by_source = [
                item for item in catalog["datasets"]
                if any(entry["remote_folder"] == source["remote_folder"] for entry in item["sources"])
            ]
            normalized = [
                item for item in catalog["datasets"]
                if safe_name(str(item["name"])) == safe_name(snapshot.name)
            ]
            if by_source and by_source != by_fingerprint:
                raise BackupError("known remote dataset source changed content; automatic retraining is held")
            if len(by_fingerprint) > 1 or len(normalized) > 1:
                raise BackupError("dataset identity resolves ambiguously")
            if by_fingerprint:
                record = by_fingerprint[0]
                model_catalog, _ = model_store.read()
                existing_model = resolve_model(model_catalog, int(record["id"]))
                if (
                    int(existing_model["id"]) != int(record["id"])
                    or existing_model["folder"] != record["canonical_folder"]
                    or existing_model["name"] != record["name"]
                    or existing_model.get("trigger_word") != record.get("trigger_word")
                ):
                    raise BackupError("dataset and model catalogs disagree")
                if not by_source:
                    record["sources"].append(dict(source))
                else:
                    return dict(record), parent
                model_catalog = None
            else:
                if normalized:
                    raise BackupError("known dataset name changed content; automatic retraining is held")
                model_catalog, model_parent = model_store.read()
                if model_parent != parent:
                    continue
                normalized_models = [
                    item for item in model_catalog["models"]
                    if safe_name(str(item["name"])) == safe_name(snapshot.name)
                ]
                if normalized_models:
                    raise BackupError(
                        "model name already exists without a matching immutable dataset identity"
                    )
                numeric_id = max(
                    [int(item["id"]) for item in model_catalog["models"]]
                    + [int(item["id"]) for item in catalog["datasets"]]
                    + [0]
                ) + 1
                model = {
                    "id": numeric_id, "name": snapshot.name,
                    "folder": f"{numeric_id:04d}-{safe_name(snapshot.name)}",
                    "base_arch": model_metadata.get("base_arch"),
                    "base_model": model_metadata.get("base_model"),
                    "trigger_word": snapshot.trigger_word,
                    "destination_kind": "loras", "checkpoints": [],
                    "selected_checkpoint_id": None, "selection": None,
                }
                model_catalog["models"].append(model)
                model_catalog["models"].sort(key=lambda item: int(item["id"]))
                record = {
                    "id": int(model["id"]), "name": str(model["name"]),
                    "canonical_folder": str(model["folder"]),
                    "trigger_word": snapshot.trigger_word,
                    "fingerprint": snapshot.fingerprint,
                    "image_count": snapshot.image_count,
                    "sources": [dict(source)],
                }
                catalog["datasets"].append(record)
                catalog["datasets"].sort(key=lambda item: int(item["id"]))
            try:
                return dict(record), self._write(
                    catalog, parent, model_catalog=model_catalog,
                    model_catalog_path=(model_store.catalog_path if model_catalog is not None else None),
                )
            except Exception as exc:
                last_error = exc
        raise BackupError("dataset catalog update conflicted repeatedly") from last_error


def _source_from_files(candidate: Mapping[str, Any], files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "remote_folder": candidate["remote_folder"],
        "revision": candidate["revision"],
        "files": [
            {
                "relative_path": item["relative_path"],
                "remote_path": item["remote_path"],
                "size": int(item["size"]),
                "sha256": item["sha256"],
            }
            for item in files
        ],
    }


def _local_snapshot(folder: Path, *, name: str, trigger_word: str, exposures: int) -> DatasetSnapshot:
    metadata = folder / REMOTE_MANIFEST
    atomic_write_json(metadata, {"catalog_name": name, "trigger_word": trigger_word})
    return _snapshot_folder(
        folder, exposures=exposures, batch_size=4,
        resolutions=(512, 768, 1024), repeats=(16, 4, 1),
        bucket_divisibility=16, default_trigger_word=trigger_word,
    )


def _existing_target(folder: Path, record: Mapping[str, Any], exposures: int) -> bool:
    if not folder.exists():
        return False
    if not folder.is_dir() or folder.is_symlink():
        raise BackupError(f"canonical dataset target conflicts with an existing path: {folder}")
    snapshot = _snapshot_folder(
        folder, exposures=exposures, batch_size=4,
        resolutions=(512, 768, 1024), repeats=(16, 4, 1),
        bucket_divisibility=16, default_trigger_word=str(record["trigger_word"]),
    )
    if snapshot.fingerprint != record["fingerprint"]:
        raise BackupError(f"canonical dataset target has conflicting content: {folder}")
    return True


def _known_source_matches(
    *, client: DatasetStorageClient, repo_id: str, repo_type: str,
    candidate: Mapping[str, Any], source: Mapping[str, Any], work_dir: Path,
) -> bool:
    expected = {item["relative_path"]: item for item in source["files"]}
    observed = {item["relative_path"]: item for item in candidate["files"]}
    if set(expected) != set(observed):
        return False
    for relative, item in observed.items():
        known = expected[relative]
        if int(item["size"]) != int(known["size"]):
            return False
        if item.get("sha256"):
            if item["sha256"] != known["sha256"]:
                return False
            continue
        target = work_dir / "source-check" / hashlib.sha256(item["remote_path"].encode()).hexdigest()
        digest = _download(
            client, repo_id=repo_id, repo_type=repo_type,
            revision=str(candidate["revision"]), spec=item, destination=target,
        )
        if digest != known["sha256"]:
            return False
    return True


def sync_remote_datasets(
    *, client: DatasetStorageClient, repo_id: str, repo_type: str,
    first: Mapping[str, Mapping[str, Any]], second: Mapping[str, Mapping[str, Any]],
    dataset_root: Path, work_dir: Path, catalog_path: str, model_catalog_path: str,
    base_arch: str, base_model: str, exposures: int = 126,
    minimum_free_bytes: int = 1_073_741_824,
) -> dict[str, Any]:
    """Install only remote folders stable across two immutable observations."""
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("remote dataset storage requires a private repository")
    dataset_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_store = DatasetCatalogStore(
        client=client, repo_id=repo_id, repo_type=repo_type,
        remote_path=catalog_path, work_dir=work_dir / "catalog",
    )
    model_store = CatalogStore(
        client=client, repo_id=repo_id, repo_type=repo_type,
        catalog_path=model_catalog_path, work_dir=work_dir / "model-catalog",
    )
    installed = []
    held = []
    for remote_folder in sorted(set(first) | set(second)):
        known: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        before = first.get(remote_folder)
        candidate = second.get(remote_folder)
        if not before or not candidate:
            held.append({"remote_folder": remote_folder, "reason": "remote upload is not stable across both observations"})
            continue
        if candidate.get("status") != "valid":
            held.append({"remote_folder": remote_folder, "reason": candidate.get("reason")})
            continue
        if (
            before.get("status") != "valid"
            or before.get("observation_fingerprint") != candidate.get("observation_fingerprint")
        ):
            held.append({"remote_folder": remote_folder, "reason": "remote upload changed during the quiet interval"})
            continue
        try:
            known = dataset_store.source_record(remote_folder)
            if known is not None:
                record, source = known
                candidate = {
                    **dict(candidate), "name": record["name"],
                    "trigger_word": record["trigger_word"],
                }
                target = dataset_root / str(record["canonical_folder"])
                if _existing_target(target, record, exposures) and _known_source_matches(
                    client=client, repo_id=repo_id, repo_type=repo_type,
                    candidate=candidate, source=source, work_dir=work_dir,
                ):
                    installed.append({
                        "remote_folder": remote_folder,
                        "canonical_folder": record["canonical_folder"],
                        "dataset_id": int(record["id"]), "status": "already-present",
                        "fingerprint": record["fingerprint"],
                    })
                    continue
            required = sum(int(item["size"]) for item in candidate["files"])
            free = shutil.disk_usage(work_dir).free
            if free - required < int(minimum_free_bytes):
                raise BackupError(
                    f"insufficient disk space for {remote_folder}: need {required} bytes plus "
                    f"{int(minimum_free_bytes)} reserved, have {free}"
                )
            stage = work_dir / "payloads" / str(candidate["observation_fingerprint"])
            stage.mkdir(parents=True, exist_ok=True)
            verified_files = []
            for spec in candidate["files"]:
                relative = safe_relative_path(str(spec["relative_path"]))
                if len(relative.parts) != 1:
                    raise BackupError("remote dataset payload must be direct files")
                target = stage / relative
                digest = _download(
                    client, repo_id=repo_id, repo_type=repo_type,
                    revision=str(candidate["revision"]), spec=spec, destination=target,
                )
                verified_files.append({**dict(spec), "sha256": digest})
            snapshot = _local_snapshot(
                stage, name=str(candidate["name"]),
                trigger_word=str(candidate["trigger_word"]), exposures=exposures,
            )
            source = _source_from_files(candidate, verified_files)
            record, catalog_revision = dataset_store.ensure(
                snapshot=snapshot, source=source, model_store=model_store,
                model_metadata={
                    "name": snapshot.name, "base_arch": base_arch,
                    "base_model": base_model, "trigger_word": snapshot.trigger_word,
                    "destination_kind": "loras",
                },
            )
            atomic_write_json(
                stage / REMOTE_MANIFEST,
                {"catalog_name": record["name"], "trigger_word": record["trigger_word"]},
            )
            target = dataset_root / str(record["canonical_folder"])
            if _existing_target(target, record, exposures):
                shutil.rmtree(stage)
                status = "already-present"
            else:
                try:
                    os.rename(stage, target)
                except OSError as exc:
                    raise BackupError(
                        "atomic dataset install failed; cache and dataset root must share a filesystem"
                    ) from exc
                status = "installed"
            installed.append({
                "remote_folder": remote_folder,
                "canonical_folder": record["canonical_folder"],
                "dataset_id": int(record["id"]), "status": status,
                "fingerprint": record["fingerprint"],
                "catalog_revision": catalog_revision,
            })
        except Exception as exc:
            failure = {
                "remote_folder": remote_folder,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            if known is not None:
                record, _ = known
                failure.update({
                    "canonical_folder": record["canonical_folder"],
                    "dataset_id": int(record["id"]),
                    "fingerprint": record["fingerprint"],
                })
            held.append(failure)
    return {"schema_version": 1, "installed": installed, "held": held}


def upload_dataset_folder(
    *, client: DatasetStorageClient, repo_id: str, repo_type: str,
    folder: Path, remote_folder_name: str | None = None,
    catalog_name: str | None = None, trigger_word: str = "Owhx",
    remote_prefix: str = "datasets", exposures: int = 126,
    work_dir: Path,
) -> dict[str, Any]:
    """Atomically upload one validated image/caption folder plus its final manifest."""
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("dataset upload requires a private repository")
    if folder.is_symlink():
        raise BackupError("dataset upload folder must not be a symlink")
    folder = folder.resolve()
    name = _display_name(catalog_name or folder.name, "catalog_name")
    trigger = _display_name(trigger_word, "trigger_word")
    snapshot = _snapshot_folder(
        folder, exposures=exposures, batch_size=4,
        resolutions=(512, 768, 1024), repeats=(16, 4, 1),
        bucket_divisibility=16, default_trigger_word=trigger,
    )
    allowed_local = {*snapshot.files, REMOTE_MANIFEST}
    omitted = sorted(path.name for path in folder.iterdir() if path.name not in allowed_local)
    if omitted:
        raise BackupError(f"dataset upload refuses unlisted local files: {omitted}")
    source_name = _display_name(remote_folder_name or folder.name, "source folder")
    prefix = PurePosixPath(safe_relative_path(remote_prefix).as_posix()) / source_name
    _, parent = client.read_remote_file(repo_id, repo_type, f"{prefix}/{REMOTE_MANIFEST}")
    _immutable_revision(parent, "dataset upload parent revision")
    existing = [
        path for path in client.list_repo_files(repo_id, repo_type, parent)
        if PurePosixPath(path).parts[: len(prefix.parts)] == prefix.parts
    ]
    if existing:
        raise BackupError(f"refusing to overwrite existing remote dataset folder: {prefix}")
    files = []
    artifacts = []
    for relative in snapshot.files:
        local = folder / relative
        remote = str(prefix / relative)
        digest = sha256_file(local)
        files.append({"relative_path": relative, "size": local.stat().st_size, "sha256": digest})
        artifacts.append(LocalArtifact(
            str(local), remote, local.stat().st_size, digest,
            role="dataset-source", relative_path=relative,
        ))
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / (
        "dataset-manifest-"
        f"{hashlib.sha256(str(prefix).encode()).hexdigest()}-{snapshot.fingerprint}.json"
    )
    atomic_write_json(manifest_path, {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "catalog_name": name, "trigger_word": trigger, "files": files,
    })
    artifacts.append(LocalArtifact(
        str(manifest_path), str(prefix / REMOTE_MANIFEST),
        manifest_path.stat().st_size, sha256_file(manifest_path),
        role="dataset-manifest", relative_path=REMOTE_MANIFEST,
    ))
    revision = client.commit_files(
        repo_id, repo_type, artifacts,
        f"Upload complete training dataset {source_name}", parent_commit=parent,
    )
    _immutable_revision(revision, "dataset upload revision")
    verify_remote_artifacts(
        client, repo_id=repo_id, repo_type=repo_type,
        artifacts=artifacts, revision=revision, context="dataset upload",
    )
    return {
        "schema_version": 1, "status": "uploaded", "remote_folder": str(prefix),
        "revision": revision, "fingerprint": snapshot.fingerprint,
        "image_count": snapshot.image_count,
    }
