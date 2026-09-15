from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .archive import EvidenceArchive
from .backup import BackupConfigurationError, BackupError, HuggingFaceBackupClient
from .catalog import CatalogStore, safe_name, safe_relative_path
from .discovery import (
    DatasetSnapshot,
    WorkflowLedgerStore,
    load_legacy_completed,
    scan_dataset_root,
)
from .queue import TrainingQueue
from .results import publish_ranked_results
from .state import atomic_write_json
from .sync import sync_latest_loras, sync_ranked_loras


UNIFIED_SCHEMA = 1


def _expanded(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in env.items():
            value = value.replace(f"${{{key}}}", replacement)
        if "${" in value:
            raise BackupConfigurationError(f"unresolved environment placeholder: {value}")
        return value
    if isinstance(value, list):
        return [_expanded(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _expanded(item, env) for key, item in value.items()}
    return value


def load_unified_config(path: Path, env: Mapping[str, str]) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != UNIFIED_SCHEMA:
        raise BackupConfigurationError("unified workflow config requires schema_version 1")
    return _expanded(document, env)


def _existing_path(value: Any, description: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not str(value or "") or not path.is_dir():
        raise BackupConfigurationError(f"{description} is not an existing directory: {value}")
    return path


def resolve_dataset_root(config: Mapping[str, Any], env: Mapping[str, str]) -> Path:
    configured = []
    if config.get("dataset_root"):
        configured.append(("workflow dataset_root", config["dataset_root"]))
    if env.get("DATASETS_FOLDER"):
        configured.append(("GUI DATASETS_FOLDER", env["DATASETS_FOLDER"]))
    bridge = config.get("gui_settings_file")
    if bridge:
        document = json.loads(Path(str(bridge)).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise BackupConfigurationError("GUI settings bridge must contain a JSON object")
        value = document.get("DATASETS_FOLDER") or document.get("datasets_folder")
        if value:
            configured.append(("GUI settings dataset folder", value))
    if not configured:
        raise BackupConfigurationError(
            "dataset root is unknown; set DATASETS_FOLDER or unified dataset_root"
        )
    resolved = [(label, _existing_path(value, label)) for label, value in configured]
    if len({path for _, path in resolved}) != 1:
        values = ", ".join(f"{label}={path}" for label, path in resolved)
        raise BackupConfigurationError(f"GUI and automation dataset roots differ: {values}")
    return resolved[0][1]


def resolve_loras_root(config: Mapping[str, Any], env: Mapping[str, str]) -> Path:
    explicit = config.get("loras_root") or env.get("LORAS_ROOT")
    if explicit:
        return _existing_path(explicit, "LoRA root")
    comfy = config.get("comfyui_root") or env.get("COMFYUI_ROOT")
    if comfy:
        return _existing_path(Path(str(comfy)) / "models" / "loras", "ComfyUI LoRA root")
    storage = Path(str(config.get("storage_root", "/storage"))).resolve()
    if not storage.is_dir():
        raise BackupConfigurationError("storage root is unavailable for bounded ComfyUI discovery")
    candidates = [
        path / "models" / "loras"
        for path in storage.iterdir()
        if path.is_dir() and path.name.casefold() == "comfyui"
        and (path / "models" / "loras").is_dir()
    ]
    if len(candidates) != 1:
        raise BackupConfigurationError(
            "set COMFYUI_ROOT or LORAS_ROOT; bounded /storage discovery found "
            f"{len(candidates)} ComfyUI LoRA roots"
        )
    return candidates[0].resolve()


@contextmanager
def _controller_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError("unified automation is already active on this storage") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _repo(config: Mapping[str, Any], env: Mapping[str, str]) -> tuple[str, str, str]:
    hub = config.get("hub") or {}
    repo_id = str(hub.get("repo_id") or env.get(str(hub.get("repo_id_env", "HF_REPO_ID")), ""))
    token_env = str(hub.get("token_env", "HF_TOKEN"))
    token = str(env.get(token_env, ""))
    repo_type = str(hub.get("repo_type", "dataset"))
    if not repo_id or not token or repo_type not in {"dataset", "model"}:
        raise BackupConfigurationError("private Hub repo, repo type, and credential environment are required")
    return repo_id, repo_type, token


def _sync_startup(
    *, client: Any, config: Mapping[str, Any], repo_id: str, repo_type: str,
    loras_root: Path, work_root: Path, dry_run: bool,
) -> list[dict[str, Any]]:
    sync_config = config.get("sync") or {}
    catalog_prefix = str(sync_config.get("catalog_prefix", "training-backups"))
    results_prefix = str(sync_config.get("results_prefix", "training-results"))
    _, revision = client.read_remote_file(
        repo_id, repo_type, f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json"
    )
    if not revision:
        raise BackupError("Hub catalog has no immutable source revision")
    records = [sync_latest_loras(
        client=client, repo_id=repo_id, repo_type=repo_type,
        source_revision=revision, loras_root=loras_root,
        work_dir=work_root / "sync" / "latest", ranks=(1,),
        catalog_prefix=catalog_prefix, results_prefix=results_prefix, dry_run=dry_run,
    )]
    for seed in sync_config.get("legacy_runs") or []:
        if not isinstance(seed, Mapping):
            raise BackupConfigurationError("sync legacy_runs entries must be mappings")
        records.append(sync_ranked_loras(
            client=client, repo_id=repo_id, repo_type=repo_type,
            source_revision=str(seed["revision"]), run_id=str(seed["run_id"]),
            loras_root=loras_root, work_dir=work_root / "sync" / "legacy" / str(seed["run_id"]),
            ranks=(1,), model_ids=tuple(int(value) for value in seed.get("model_ids") or []),
            catalog_prefix=catalog_prefix, results_prefix=results_prefix, dry_run=dry_run,
        ))
    if any(item["status"] == "conflicts" for item in records):
        raise BackupError("LoRA startup sync found conflicting local files")
    return records


def _legacy_index(
    *, client: Any, config: Mapping[str, Any], repo_id: str, repo_type: str,
    work_root: Path,
) -> dict[str, Any]:
    value = (config.get("discovery") or {}).get("legacy_completed_index")
    if not value:
        return {}
    return load_legacy_completed(
        client, repo_id=repo_id, repo_type=repo_type,
        remote_path=str(value["path"]), revision=str(value["revision"]),
        work_dir=work_root / "legacy-completed",
    )


def _ordered_candidates(
    ledger: Mapping[str, Any], *, worker_id: int, explicit_order: Sequence[str],
    include_incomplete: bool,
) -> list[dict[str, Any]]:
    candidates = [
        dict(item) for item in ledger["datasets"].values()
        if int(item.get("worker", -1)) == worker_id
        and item.get("status") in (
            {"ready", "queued", "incomplete"} if include_incomplete else {"ready", "queued"}
        )
    ]
    lookup: dict[str, int] = {}
    for index, value in enumerate(explicit_order):
        key = safe_name(str(value))
        if key in lookup:
            raise BackupConfigurationError("explicit dataset order contains duplicates")
        lookup[key] = index
    known = {safe_name(str(item["name"])) for item in candidates} | {
        safe_name(str(item["folder"])) for item in candidates
    }
    missing = set(lookup) - known
    if missing:
        raise BackupConfigurationError(f"explicit dataset order names no pending dataset: {sorted(missing)}")
    return sorted(
        candidates,
        key=lambda item: (
            min(
                lookup.get(safe_name(str(item["name"])), len(lookup)),
                lookup.get(safe_name(str(item["folder"])), len(lookup)),
            ),
            safe_name(str(item["name"])),
        ),
    )


def _write_queue(
    *, config: Mapping[str, Any], items: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]], dataset_root: Path,
    work_root: Path, repo_id: str, repo_type: str, worker_id: int,
) -> Path:
    queue_config = config.get("queue") or {}
    evaluation = dict(config.get("evaluation") or {})
    datasets = []
    for item in items:
        model = models[str(item["folder"])]
        folder = dataset_root / str(item["folder"])
        references = [
            str(folder / name) for name in item["files"]
            if Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        datasets.append({
            "name": str(item["folder"]),
            "catalog_name": str(model["name"]),
            "expected_catalog_id": int(model["id"]),
            "folder": str(folder),
            "trigger_word": str(item["trigger_word"]),
            "destination_kind": "loras",
            "dataset_revision": str(item["fingerprint"]),
            "reference_images": references,
            "training_steps": int(item["training_steps"]),
            "training_accounting": {
                "source_image_count": int(item["image_count"]),
                "loader_batches_per_epoch": int(item["loader_batches_per_epoch"]),
                "loader_epochs": int(item["loader_epochs"]),
                "resolution_repeats": [16, 4, 1],
                "batch_size": 4,
                "original_image_exposures": int(item["original_image_exposures"]),
                "partial_bucket_batches": "un-padded",
            },
            "trainer_dataset": {
                "caption_ext": "txt", "caption_dropout_rate": 0.05,
                "cache_latents_to_disk": True, "resolution": [512, 768, 1024],
                "num_repeats": [16, 4, 1], "bucket_tolerance": 16,
                "scale": 1, "square_crop": False, "random_crop": False,
            },
        })
    path = work_root / "queue.yaml"
    document = {
        "schema_version": 1,
        "trainer_yaml": str(Path(str(queue_config["trainer_yaml"])).resolve()),
        "repo_root": str(Path(str(queue_config.get("repo_root", "/app/ai-toolkit"))).resolve()),
        "generated_dir": str(work_root / "generated"),
        "state_path": str(work_root / "queue-state.json"),
        "continue_on_error": False,
        "training_folder": str(Path(str(queue_config.get("output_root", "/storage/output"))).resolve()),
        "checkpoint_policy": dict(config.get("checkpoint_policy") or {"save_every": 100, "max_local_step_saves": 5}),
        "checkpoint_backup": {
            "enabled": True, "repo_id": repo_id, "repo_type": repo_type,
            "token_env": str((config.get("hub") or {}).get("token_env", "HF_TOKEN")),
            "remote_prefix": str((config.get("sync") or {}).get("catalog_prefix", "training-backups")),
            "max_attempts": 4, "backoff_seconds": 2,
            "catalog": {"destination_kind": "loras"},
        },
        "evaluation": evaluation,
        "datasets": datasets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def run_unified_workflow(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
    queue_factory: Callable[[Path], TrainingQueue] = TrainingQueue,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    dry_run: bool = False,
    startup_sync: Callable[..., list[dict[str, Any]]] | None = None,
    publisher: Callable[..., tuple[dict[str, Any], list[tuple[Path, str]]]] = publish_ranked_results,
    latest_sync: Callable[..., Mapping[str, Any]] = sync_latest_loras,
    archive_factory: Callable[..., Any] = EvidenceArchive,
) -> dict[str, Any]:
    values = dict(os.environ if env is None else env)
    config = load_unified_config(config_path, values)
    dataset_root = resolve_dataset_root(config, values)
    loras_root = resolve_loras_root(config, values)
    repo_id, repo_type, token = _repo(config, values)
    hub = client or HuggingFaceBackupClient(token)
    if not hub.repo_is_private(repo_id, repo_type):
        raise BackupConfigurationError("unified workflow requires a private Hub repository")
    work_root = Path(str(config.get("work_root", "/storage/automation/unified"))).resolve()
    worker = config.get("worker") or {}
    worker_id = int(worker.get("id", values.get("TRAINING_WORKER_ID", 0)))
    worker_count = int(worker.get("count", values.get("TRAINING_WORKER_COUNT", 1)))
    if worker_count <= 0 or worker_id < 0 or worker_id >= worker_count:
        raise BackupConfigurationError("worker id must fall within configured worker count")
    discovery = config.get("discovery") or {}
    quiet_seconds = float(discovery.get("quiet_seconds", 60))
    if quiet_seconds < 0:
        raise BackupConfigurationError("discovery quiet_seconds cannot be negative")
    ledger_store = WorkflowLedgerStore(
        client=hub, repo_id=repo_id, repo_type=repo_type,
        remote_path=str(discovery.get("ledger_path", "training-automation/workflow-ledger.json")),
        local_path=work_root / "workflow-ledger.json",
    )
    with _controller_lock(work_root / f"controller-worker-{worker_id}.lock"):
        sync_records = (startup_sync or _sync_startup)(
            client=hub, config=config, repo_id=repo_id, repo_type=repo_type,
            loras_root=loras_root, work_root=work_root, dry_run=dry_run,
        )
        first = scan_dataset_root(
            dataset_root, exposures=int(discovery.get("target_exposures", 126)),
            default_trigger_word=str((config.get("queue") or {}).get("trigger_word", "Owhx")),
        )
        start = clock()
        ledger_store.reconcile(
            first, worker_count=worker_count, quiet_seconds=quiet_seconds, now=start,
            legacy_completed=_legacy_index(
                client=hub, config=config, repo_id=repo_id,
                repo_type=repo_type, work_root=work_root,
            ),
        )
        if quiet_seconds:
            sleep(quiet_seconds)
        second = scan_dataset_root(
            dataset_root, exposures=int(discovery.get("target_exposures", 126)),
            default_trigger_word=str((config.get("queue") or {}).get("trigger_word", "Owhx")),
        )
        ledger, ledger_revision = ledger_store.reconcile(
            second, worker_count=worker_count, quiet_seconds=quiet_seconds,
            now=max(clock(), start + quiet_seconds),
        )
        candidates = _ordered_candidates(
            ledger, worker_id=worker_id,
            explicit_order=tuple(discovery.get("order") or []),
            include_incomplete=bool(discovery.get("retry_incomplete", False)),
        )
        summary: dict[str, Any] = {
            "schema_version": 1, "worker_id": worker_id,
            "worker_count": worker_count, "dataset_root": str(dataset_root),
            "loras_root": str(loras_root), "ledger_revision": ledger_revision,
            "sync": sync_records, "pending": [item["folder"] for item in candidates],
        }
        if not candidates or dry_run:
            held = [
                {"folder": item["folder"], "status": item["status"], "reason": item.get("reason")}
                for item in ledger["datasets"].values()
                if item.get("status") in {"changed", "incomplete", "observing"}
                and int(item.get("worker", worker_id)) == worker_id
            ]
            summary["held"] = held
            summary["status"] = "ready" if candidates else ("held" if held else "idle")
            summary["dry_run"] = dry_run
            return summary

        store = CatalogStore(
            client=hub, repo_id=repo_id, repo_type=repo_type,
            catalog_path=f"{safe_relative_path(str((config.get('sync') or {}).get('catalog_prefix', 'training-backups'))).as_posix()}/catalog.json",
            work_dir=work_root / "catalog",
        )
        recipe = yaml.safe_load(Path(str((config.get("queue") or {})["trainer_yaml"])).read_text(encoding="utf-8"))
        process = recipe["config"]["process"][0]
        models: dict[str, Mapping[str, Any]] = {}
        for item in candidates:
            model, _ = store.ensure_model({
                "name": item["name"], "base_arch": process["model"]["arch"],
                "base_model": process["model"]["name_or_path"],
                "trigger_word": str(item["trigger_word"]),
                "destination_kind": "loras",
            })
            models[str(item["folder"])] = model
            ledger_store.update_status(
                str(item["folder"]), str(item["fingerprint"]), "queued",
                details={"catalog_id": int(model["id"]), "catalog_folder": model["folder"]},
            )
        fingerprint = hashlib.sha256(
            "\n".join(str(item["fingerprint"]) for item in candidates).encode()
        ).hexdigest()[:16]
        run_id = f"unified-{fingerprint}"
        run_root = work_root / "runs" / run_id / f"worker-{worker_id}"
        queue_path = _write_queue(
            config=config, items=candidates, models=models, dataset_root=dataset_root,
            work_root=run_root, repo_id=repo_id, repo_type=repo_type, worker_id=worker_id,
        )
        before_training = {item.folder: item.fingerprint for item in scan_dataset_root(
            dataset_root, exposures=int(discovery.get("target_exposures", 126)),
            default_trigger_word=str((config.get("queue") or {}).get("trigger_word", "Owhx")),
        )}
        for item in candidates:
            if before_training.get(str(item["folder"])) != item["fingerprint"]:
                raise BackupError(f"dataset changed after quiet-period acceptance: {item['folder']}")
        queue = queue_factory(queue_path)
        jobs = queue.materialize()
        if len(jobs) != len(candidates):
            raise BackupError("materialized queue does not match assigned dataset count")
        state = queue.run()
        if any((state["jobs"].get(job.job_id) or {}).get("status") != "completed" for job in jobs):
            for item, job in zip(candidates, jobs):
                job_status = (state["jobs"].get(job.job_id) or {}).get("status", "unknown")
                ledger_store.update_status(
                    str(item["folder"]), str(item["fingerprint"]), "incomplete",
                    details={"job_id": job.job_id, "job_status": job_status, "run_id": run_id},
                )
            raise BackupError("one or more unified training jobs did not complete")

        archive_files: list[tuple[Path, str]] = [(queue_path, "queue.yaml"), (queue.state_path, "queue-state.json")]
        exports = []
        for item, job in zip(candidates, jobs):
            output = job.output_root / job.job_id
            record, evidence = publisher(
                client=hub, repo_id=repo_id, repo_type=repo_type,
                run_id=run_id, job_id=job.job_id,
                report_path=output / ".automation" / "evaluation.json",
                sample_root=output / "samples", work_dir=run_root / "results" / job.job_id,
                catalog_prefix=str((config.get("sync") or {}).get("catalog_prefix", "training-backups")),
            )
            exports.append(record)
            archive_files.extend(evidence)
            archive_files.append((output / ".automation" / "evaluation.json", f"jobs/{job.job_id}/evaluation.json"))
            archive_files.extend(
                (path, f"jobs/{job.job_id}/samples/{path.name}")
                for path in sorted((output / "samples").iterdir()) if path.is_file()
            )
            latest_result = latest_sync(
                client=hub, repo_id=repo_id, repo_type=repo_type,
                source_revision=str(record["revision"]), loras_root=loras_root,
                work_dir=run_root / "sync-after-export" / job.job_id,
                model_ids=(int(models[str(item["folder"])]["id"]),), ranks=(1,),
                catalog_prefix=str((config.get("sync") or {}).get("catalog_prefix", "training-backups")),
                results_prefix=str((config.get("sync") or {}).get("results_prefix", "training-results")),
            )
            if latest_result["status"] != "completed":
                raise BackupError(f"post-export top1 sync failed: {item['folder']}")
        archive = archive_factory(
            client=hub, repo_id=repo_id, repo_type=repo_type,
            remote_prefix=str((config.get("archive") or {}).get("remote_prefix", "training-archives")),
            run_id=run_id, shard_id=f"worker-{worker_id}",
            state_path=run_root / "archive-state.json",
        ).publish(archive_files, {
            "status": "completed", "run_id": run_id, "worker_id": worker_id,
            "jobs": [{"job_id": job.job_id, "dataset_fingerprint": item["fingerprint"]}
                     for item, job in zip(candidates, jobs)],
        })
        for item, job, record in zip(candidates, jobs, exports):
            ledger_store.update_status(
                str(item["folder"]), str(item["fingerprint"]), "completed",
                details={
                    "job_id": job.job_id, "run_id": run_id,
                    "result_revision": record["revision"],
                    "archive_completion_revision": archive["completion_revision"],
                },
            )
        summary.update({"status": "completed", "run_id": run_id, "exports": exports, "archive": archive})
        return summary
