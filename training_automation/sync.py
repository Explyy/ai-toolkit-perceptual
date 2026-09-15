from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from .backup import BackupError, sha256_file
from .catalog import resolve_model, safe_relative_path
from .staging import COMMIT_RE


class SyncClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...

    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def download_file(
        self, repo_id: str, repo_type: str, path: str, revision: str,
        destination: Path,
    ) -> None: ...

    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...


def _pinned_download(
    client: SyncClient,
    *,
    repo_id: str,
    repo_type: str,
    remote_path: str,
    revision: str,
    destination: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    safe_relative_path(remote_path)
    metadata = client.path_metadata(repo_id, repo_type, [remote_path], revision)
    remote = metadata.get(remote_path)
    if remote is None or remote.get("size") is None:
        raise BackupError(f"pinned sync source metadata is missing: {remote_path}")
    size = int(remote["size"])
    if expected_size is not None and size != int(expected_size):
        raise BackupError(f"pinned sync source size differs from evidence: {remote_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(repo_id, repo_type, remote_path, revision, destination)
    digest = sha256_file(destination)
    if destination.stat().st_size != size:
        raise BackupError(f"pinned sync download size mismatch: {remote_path}")
    metadata_sha = str(remote.get("sha256") or "").removeprefix("sha256:")
    if metadata_sha and digest != metadata_sha:
        raise BackupError(f"pinned sync metadata hash mismatch: {remote_path}")
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise BackupError(f"pinned sync content hash differs from evidence: {remote_path}")
    return {"size": size, "sha256": digest}


def _json_at_revision(
    client: SyncClient,
    *,
    repo_id: str,
    repo_type: str,
    remote_path: str,
    revision: str,
    work_dir: Path,
) -> dict[str, Any]:
    local = work_dir / (PurePosixPath(remote_path).name + ".download")
    _pinned_download(
        client,
        repo_id=repo_id,
        repo_type=repo_type,
        remote_path=remote_path,
        revision=revision,
        destination=local,
    )
    try:
        document = json.loads(local.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError(f"pinned sync JSON is invalid: {remote_path}") from exc
    if not isinstance(document, dict):
        raise BackupError(f"pinned sync JSON must be an object: {remote_path}")
    return document


def _local_status(target: Path, *, size: int, sha256: str) -> str:
    if not target.exists():
        return "planned"
    if target.is_file() and target.stat().st_size == size and sha256_file(target) == sha256:
        return "reused"
    return "conflict"


def _install(
    client: SyncClient,
    *,
    repo_id: str,
    repo_type: str,
    remote_path: str,
    revision: str,
    target: Path,
    size: int,
    sha256: str,
) -> str:
    status = _local_status(target, size=size, sha256=sha256)
    if status != "planned":
        return status
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        _pinned_download(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_path=remote_path,
            revision=revision,
            destination=temporary,
            expected_size=size,
            expected_sha256=sha256,
        )
        try:
            os.link(temporary, target)
            return "downloaded"
        except FileExistsError:
            return _local_status(target, size=size, sha256=sha256)
    finally:
        temporary.unlink(missing_ok=True)


def sync_ranked_loras(
    *,
    client: SyncClient,
    repo_id: str,
    repo_type: str,
    source_revision: str,
    run_id: str,
    loras_root: Path,
    work_dir: Path,
    ranks: Sequence[int] = (1,),
    model_ids: Sequence[int] = (),
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Synchronize explicit automatic ranks from one immutable private Hub snapshot."""
    if not COMMIT_RE.fullmatch(source_revision):
        raise BackupError("sync requires an immutable 40-character source revision")
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("refusing ranked sync from a non-private repository")
    run_component = safe_relative_path(run_id)
    if len(run_component.parts) != 1:
        raise BackupError("sync run_id must be one safe path component")
    normalized_ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not normalized_ranks or any(rank not in {1, 2, 3} for rank in normalized_ranks):
        raise BackupError("sync ranks must contain only 1, 2, or 3")
    filters = {int(value) for value in model_ids}
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json"
    catalog = _json_at_revision(
        client,
        repo_id=repo_id,
        repo_type=repo_type,
        remote_path=catalog_path,
        revision=source_revision,
        work_dir=work_dir,
    )
    models = sorted(catalog.get("models") or [], key=lambda item: int(item["id"]))
    if filters:
        missing = filters - {int(model["id"]) for model in models}
        if missing:
            raise BackupError(f"sync model ids are absent from the pinned catalog: {sorted(missing)}")
        models = [model for model in models if int(model["id"]) in filters]
    records = []
    counts = {"downloaded": 0, "reused": 0, "planned": 0, "conflict": 0}
    for model_document in models:
        model = resolve_model(catalog, int(model_document["id"]))
        if model.get("destination_kind") != "loras":
            continue
        folder = safe_relative_path(str(model["folder"]))
        if len(folder.parts) != 1 or not folder.name.startswith(f"{int(model['id']):04d}-"):
            raise BackupError("pinned catalog model has an invalid numeric-name folder")
        index_root = PurePosixPath(safe_relative_path(results_prefix).as_posix()) / run_id / folder.name
        index = _json_at_revision(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_path=str(index_root / "index.json"),
            revision=source_revision,
            work_dir=work_dir / folder.name,
        )
        if (
            index.get("run_id") != run_id
            or int((index.get("model") or {}).get("id", -1)) != int(model["id"])
            or (index.get("model") or {}).get("folder") != folder.name
        ):
            raise BackupError(f"pinned result index identity mismatch: {folder.name}")
        candidates = {int(item["rank"]): item for item in index.get("candidates") or []}
        for rank in normalized_ranks:
            candidate = candidates.get(rank)
            if candidate is None:
                raise BackupError(f"pinned result index has no top-{rank}: {folder.name}")
            selection_path = str(index_root / f"top-{rank}" / "selection.json")
            selection = _json_at_revision(
                client,
                repo_id=repo_id,
                repo_type=repo_type,
                remote_path=selection_path,
                revision=source_revision,
                work_dir=work_dir / folder.name / f"top-{rank}",
            )
            checkpoint_id = str(candidate.get("catalog_checkpoint_id") or "")
            if (
                selection.get("run_id") != run_id
                or selection.get("rank") != rank
                or selection.get("catalog_checkpoint_id") != checkpoint_id
                or int((selection.get("model") or {}).get("id", -1)) != int(model["id"])
            ):
                raise BackupError(f"pinned top-{rank} selection identity mismatch: {folder.name}")
            checkpoint = next(
                (
                    item for item in model.get("checkpoints") or []
                    if item.get("checkpoint_id") == checkpoint_id
                ),
                None,
            )
            if checkpoint is None:
                raise BackupError(f"pinned top-{rank} checkpoint is absent from catalog: {folder.name}")
            checkpoint_weights = {
                str(item["remote_path"]): item for item in checkpoint.get("weights") or []
            }
            seen_names = set()
            for weight in selection.get("source_weights") or []:
                source = checkpoint_weights.get(str(weight.get("source_remote_path")))
                if (
                    source is None
                    or checkpoint.get("revision") != weight.get("source_revision")
                    or int(source["size"]) != int(weight.get("size", -1))
                    or source["sha256"] != weight.get("sha256")
                ):
                    raise BackupError(f"pinned top-{rank} weight differs from catalog: {folder.name}")
                result_path = safe_relative_path(str(weight["destination_remote_path"]))
                expected_parent = Path(*index_root.parts) / f"top-{rank}" / "weights"
                if result_path.parent != expected_parent:
                    raise BackupError(f"pinned top-{rank} weight path escapes its result folder")
                name = result_path.name
                if name in seen_names:
                    raise BackupError(f"pinned top-{rank} repeats a weight basename")
                seen_names.add(name)
                checkpoint_folder = safe_relative_path(checkpoint_id)
                if len(checkpoint_folder.parts) != 1:
                    raise BackupError("ranked sync checkpoint id must be one safe path component")
                target = (
                    loras_root.resolve() / folder / checkpoint_folder / name
                ).resolve(strict=False)
                if not target.is_relative_to(loras_root.resolve()):
                    raise BackupError("ranked sync target escapes configured LoRA root")
                if dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary_name = tempfile.mkstemp(prefix=".sync-dry-run-", dir=work_dir)
                    os.close(fd)
                    temporary = Path(temporary_name)
                    try:
                        _pinned_download(
                            client,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            remote_path=result_path.as_posix(),
                            revision=source_revision,
                            destination=temporary,
                            expected_size=int(weight["size"]),
                            expected_sha256=str(weight["sha256"]),
                        )
                    finally:
                        temporary.unlink(missing_ok=True)
                    status = _local_status(
                        target, size=int(weight["size"]), sha256=str(weight["sha256"])
                    )
                else:
                    status = _install(
                        client,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        remote_path=result_path.as_posix(),
                        revision=source_revision,
                        target=target,
                        size=int(weight["size"]),
                        sha256=str(weight["sha256"]),
                    )
                counts[status] += 1
                records.append({
                    "model_id": int(model["id"]),
                    "model_folder": folder.name,
                    "rank": rank,
                    "checkpoint_id": checkpoint_id,
                    "source_revision": source_revision,
                    "path": str(target),
                    "status": status,
                })
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "run_id": run_id,
        "source_revision": source_revision,
        "dry_run": bool(dry_run),
        "counts": counts,
        "files": records,
        "status": "conflicts" if counts["conflict"] else "completed",
    }


def sync_latest_loras(
    *,
    client: SyncClient,
    repo_id: str,
    repo_type: str,
    source_revision: str,
    loras_root: Path,
    work_dir: Path,
    ranks: Sequence[int] = (1,),
    model_ids: Sequence[int] = (),
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync per-model latest pointers from one immutable repository revision."""
    catalog = _json_at_revision(
        client,
        repo_id=repo_id,
        repo_type=repo_type,
        remote_path=f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json",
        revision=source_revision,
        work_dir=work_dir / "catalog",
    )
    requested = {int(value) for value in model_ids}
    available = {int(model["id"]) for model in catalog.get("models") or []}
    missing = requested - available
    if missing:
        raise BackupError(f"latest sync model ids are absent from catalog: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    totals = {"downloaded": 0, "reused": 0, "planned": 0, "conflict": 0}
    skipped = []
    for model in sorted(catalog.get("models") or [], key=lambda item: int(item["id"])):
        model_id = int(model["id"])
        if requested and model_id not in requested:
            continue
        folder = safe_relative_path(str(model["folder"]))
        pointer_path = (
            f"{safe_relative_path(results_prefix).as_posix()}/latest/{folder.name}.json"
        )
        current, _ = client.read_remote_file(repo_id, repo_type, pointer_path)
        if current is None:
            if requested:
                raise BackupError(f"latest result pointer is missing: {folder.name}")
            skipped.append({"model_id": model_id, "reason": "latest pointer unavailable"})
            continue
        pointer = _json_at_revision(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_path=pointer_path,
            revision=source_revision,
            work_dir=work_dir / "latest" / folder.name,
        )
        if (
            pointer.get("schema_version") != 1
            or pointer.get("status") != "completed"
            or int((pointer.get("model") or {}).get("id", -1)) != model_id
            or (pointer.get("model") or {}).get("folder") != folder.name
            or pointer.get("index_path")
            != f"{safe_relative_path(results_prefix).as_posix()}/{pointer.get('run_id')}/{folder.name}/index.json"
        ):
            raise BackupError(f"latest result pointer identity is invalid: {folder.name}")
        result = sync_ranked_loras(
            client=client,
            repo_id=repo_id,
            repo_type=repo_type,
            source_revision=source_revision,
            run_id=str(pointer["run_id"]),
            loras_root=loras_root,
            work_dir=work_dir / "runs" / str(pointer["run_id"]) / folder.name,
            ranks=ranks,
            model_ids=(model_id,),
            catalog_prefix=catalog_prefix,
            results_prefix=results_prefix,
            dry_run=dry_run,
        )
        for key in totals:
            totals[key] += int(result["counts"][key])
        records.extend(result["files"])
    return {
        "schema_version": 1,
        "source_revision": source_revision,
        "counts": totals,
        "files": records,
        "skipped": skipped,
        "status": "conflicts" if totals["conflict"] else "completed",
    }
