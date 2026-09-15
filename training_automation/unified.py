from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
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
from .state import atomic_write_json, read_json
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


def _gui_database(config: Mapping[str, Any]) -> Path:
    queue = config.get("queue") or {}
    repo_root = Path(str(queue.get("repo_root", "/app/ai-toolkit"))).resolve()
    return Path(str(config.get("gui_database", repo_root / "aitk_db.db"))).resolve()


def _read_gui_settings(config: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
    database = _gui_database(config)
    if not database.is_file():
        return {}, False
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='Settings'"
        ).fetchone()
        if table is None:
            return {}, False
        return {
            str(key): str(value)
            for key, value in connection.execute("SELECT key, value FROM Settings")
            if value is not None and str(value)
        }, True
    finally:
        connection.close()


def _initialize_gui_dataset_root(config: Mapping[str, Any], desired: Path) -> None:
    database = _gui_database(config)
    if not database.parent.is_dir():
        raise BackupConfigurationError(
            f"GUI database parent directory is unavailable: {database.parent}"
        )
    queue = config.get("queue") or {}
    native_default = Path(
        str(Path(str(queue.get("repo_root", "/app/ai-toolkit"))).resolve() / "datasets")
    ).resolve()
    if native_default != desired and native_default.is_dir() and any(native_default.iterdir()):
        raise BackupConfigurationError(
            "refusing to redirect GUI DATASETS_FOLDER while the native dataset root contains data"
        )
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS Settings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)"
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT value FROM Settings WHERE key = ?", ("DATASETS_FOLDER",)
        ).fetchone()
        if existing is not None and str(existing[0] or ""):
            if Path(str(existing[0])).expanduser().resolve() != desired:
                raise BackupConfigurationError(
                    "GUI DATASETS_FOLDER appeared with a conflicting value during initialization"
                )
        else:
            connection.execute(
                "INSERT INTO Settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value "
                "WHERE Settings.value = ''",
                ("DATASETS_FOLDER", str(desired)),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def resolve_dataset_root(config: Mapping[str, Any], env: Mapping[str, str]) -> Path:
    desired = []
    if config.get("dataset_root"):
        desired.append(("workflow dataset_root", config["dataset_root"]))
    if env.get("DATASETS_FOLDER"):
        desired.append(("DATASETS_FOLDER environment", env["DATASETS_FOLDER"]))
    resolved_desired = [(label, _existing_path(value, label)) for label, value in desired]
    if len({path for _, path in resolved_desired}) > 1:
        values = ", ".join(f"{label}={path}" for label, path in resolved_desired)
        raise BackupConfigurationError(f"configured dataset roots differ: {values}")

    actual = []
    bridge = config.get("gui_settings_file")
    if bridge:
        document = json.loads(Path(str(bridge)).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise BackupConfigurationError("GUI settings bridge must contain a JSON object")
        value = document.get("DATASETS_FOLDER") or document.get("datasets_folder")
        if value:
            actual.append(("GUI settings bridge", _existing_path(value, "GUI settings dataset folder")))
    gui_settings, _ = _read_gui_settings(config)
    database_value = gui_settings.get("DATASETS_FOLDER")
    if database_value:
        actual.append(("GUI database DATASETS_FOLDER", _existing_path(database_value, "GUI database DATASETS_FOLDER")))
    queue = config.get("queue") or {}
    native_default = Path(str(queue.get("repo_root", "/app/ai-toolkit"))).resolve() / "datasets"
    desired_path = resolved_desired[0][1] if resolved_desired else None
    if not actual and desired_path is not None and desired_path != native_default.resolve():
        if not bool(config.get("initialize_gui_dataset_root", False)):
            raise BackupConfigurationError(
                "GUI has no DATASETS_FOLDER setting; enable initialize_gui_dataset_root "
                "or use the native GUI dataset root"
            )
        _initialize_gui_dataset_root(config, desired_path)
        actual.append(("initialized GUI database DATASETS_FOLDER", desired_path))
    if not actual:
        actual.append(("native GUI dataset root", _existing_path(native_default, "native GUI dataset root")))
    if len({path for _, path in actual}) != 1:
        values = ", ".join(f"{label}={path}" for label, path in actual)
        raise BackupConfigurationError(f"GUI dataset roots differ: {values}")
    if desired_path is not None and actual[0][1] != desired_path:
        values = f"configured={desired_path}, GUI={actual[0][1]}"
        raise BackupConfigurationError(f"GUI and automation dataset roots differ: {values}")
    return actual[0][1]


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


def prepare_unified_environment(
    config_path: Path, *, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Resolve and, when explicitly enabled, initialize the GUI storage bridge."""
    values = dict(os.environ if env is None else env)
    config = load_unified_config(config_path, values)
    return {
        "schema_version": 1,
        "dataset_root": str(resolve_dataset_root(config, values)),
        "loras_root": str(resolve_loras_root(config, values)),
        "gui_database": str(_gui_database(config)),
    }


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
    gui_settings, _ = _read_gui_settings(config)
    token = str(env.get(token_env, "") or gui_settings.get("HF_TOKEN", ""))
    repo_type = str(hub.get("repo_type", "dataset"))
    if not repo_id or not token or repo_type not in {"dataset", "model"}:
        raise BackupConfigurationError("private Hub repo, repo type, and credential environment are required")
    return repo_id, repo_type, token


def _sync_startup(
    *, client: Any, config: Mapping[str, Any], repo_id: str, repo_type: str,
    loras_root: Path, work_root: Path, dry_run: bool,
    ranks: Sequence[int] = (1,), model_ids: Sequence[int] = (),
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
        work_dir=work_root / "sync" / "latest", ranks=ranks,
        model_ids=model_ids, skip_missing_requested=True,
        catalog_prefix=catalog_prefix, results_prefix=results_prefix, dry_run=dry_run,
    )]
    latest_model_ids = {
        int(item["model_id"]) for item in records[0].get("files") or []
    }
    for seed in sync_config.get("legacy_runs") or []:
        if not isinstance(seed, Mapping):
            raise BackupConfigurationError("sync legacy_runs entries must be mappings")
        configured_model_ids = tuple(int(value) for value in seed.get("model_ids") or [])
        if not configured_model_ids:
            raise BackupConfigurationError("each historical sync seed requires explicit model_ids")
        seed_model_ids = tuple(
            value for value in configured_model_ids
            if int(value) not in latest_model_ids
            and (not model_ids or int(value) in {int(item) for item in model_ids})
        )
        if not seed_model_ids:
            continue
        records.append(sync_ranked_loras(
            client=client, repo_id=repo_id, repo_type=repo_type,
            source_revision=str(seed.get("revision") or revision), run_id=str(seed["run_id"]),
            loras_root=loras_root, work_dir=work_root / "sync" / "legacy" / str(seed["run_id"]),
            ranks=ranks, model_ids=seed_model_ids,
            catalog_prefix=catalog_prefix, results_prefix=results_prefix, dry_run=dry_run,
        ))
    if any(item["status"] == "conflicts" for item in records):
        raise BackupError("LoRA startup sync found conflicting local files")
    if model_ids:
        synced = {
            int(item["model_id"])
            for record in records for item in record.get("files") or []
        }
        missing = {int(value) for value in model_ids} - synced
        if missing:
            raise BackupError(
                f"requested models have neither latest pointers nor explicit historical seeds: {sorted(missing)}"
            )
    return records


def sync_unified_loras(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
    dry_run: bool = False,
    ranks: Sequence[int] = (1,),
    model_ids: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Run the exact startup latest-plus-explicit-history sync without discovery/training."""
    values = dict(os.environ if env is None else env)
    config = load_unified_config(config_path, values)
    loras_root = resolve_loras_root(config, values)
    repo_id, repo_type, token = _repo(config, values)
    hub = client or HuggingFaceBackupClient(token)
    if not hub.repo_is_private(repo_id, repo_type):
        raise BackupConfigurationError("unified workflow requires a private Hub repository")
    worker = config.get("worker") or {}
    worker_id = int(values.get("TRAINING_WORKER_ID", worker.get("id", 0)))
    worker_count = int(values.get("TRAINING_WORKER_COUNT", worker.get("count", 1)))
    if worker_count <= 0 or worker_id < 0 or worker_id >= worker_count:
        raise BackupConfigurationError("worker id must fall within configured worker count")
    work_root = Path(str(config.get("work_root", "/storage/automation/unified"))).resolve()
    worker_root = work_root / f"worker-{worker_id}"
    with _controller_lock(worker_root / "sync-controller.lock"):
        return _sync_startup(
            client=hub, config=config, repo_id=repo_id, repo_type=repo_type,
            loras_root=loras_root, work_root=worker_root, dry_run=dry_run,
            ranks=ranks, model_ids=model_ids,
        )


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
    eligible_statuses = (
        {"ready", "queued", "incomplete"} if include_incomplete else {"ready", "queued"}
    )
    eligible = [
        dict(item) for item in ledger["datasets"].values()
        if item.get("status") in eligible_statuses
    ]
    candidates = [item for item in eligible if int(item.get("worker", -1)) == worker_id]
    lookup: dict[str, int] = {}
    for index, value in enumerate(explicit_order):
        key = safe_name(str(value))
        if key in lookup:
            raise BackupConfigurationError("explicit dataset order contains duplicates")
        lookup[key] = index
    known = {safe_name(str(item["name"])) for item in eligible} | {
        safe_name(str(item["folder"])) for item in eligible
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


def _execution_identity(
    item: Mapping[str, Any], model: Mapping[str, Any], trainer_yaml: Path, worker_id: int
) -> tuple[dict[str, Any], str]:
    identity = {
        "dataset_folder": str(item["folder"]),
        "dataset_fingerprint": str(item["fingerprint"]),
        "catalog_id": int(model["id"]),
        "catalog_folder": str(model["folder"]),
        "trainer_sha256": hashlib.sha256(trainer_yaml.read_bytes()).hexdigest(),
        "worker_id": int(worker_id),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity, f"unified-{digest[:16]}"


def _phase_state(path: Path, identity: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    existed = path.is_file()
    state = read_json(path, {})
    if existed:
        if (
            state.get("schema_version") != 1
            or state.get("run_id") != run_id
            or state.get("identity") != dict(identity)
        ):
            raise BackupError("durable dataset execution state belongs to a different identity")
        return state
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "identity": dict(identity),
        "phase": "prepared",
    }
    atomic_write_json(path, state)
    return state


def _record_phase(path: Path, state: dict[str, Any], phase: str, **details: Any) -> None:
    state.update({"phase": phase, **details})
    atomic_write_json(path, state)


def _completed_at(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(float(clock()), timezone.utc).isoformat()


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
    worker_id = int(values.get("TRAINING_WORKER_ID", worker.get("id", 0)))
    worker_count = int(values.get("TRAINING_WORKER_COUNT", worker.get("count", 1)))
    if worker_count <= 0 or worker_id < 0 or worker_id >= worker_count:
        raise BackupConfigurationError("worker id must fall within configured worker count")
    discovery = config.get("discovery") or {}
    quiet_seconds = float(discovery.get("quiet_seconds", 60))
    if quiet_seconds < 0:
        raise BackupConfigurationError("discovery quiet_seconds cannot be negative")
    worker_root = work_root / f"worker-{worker_id}"
    ledger_store = WorkflowLedgerStore(
        client=hub, repo_id=repo_id, repo_type=repo_type,
        remote_path=str(discovery.get("ledger_path", "training-automation/workflow-ledger.json")),
        local_path=worker_root / "ledger" / "workflow-ledger.json",
    )
    with _controller_lock(worker_root / "controller.lock"):
        sync_records = (startup_sync or _sync_startup)(
            client=hub, config=config, repo_id=repo_id, repo_type=repo_type,
            loras_root=loras_root, work_root=worker_root, dry_run=dry_run,
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
                repo_type=repo_type, work_root=worker_root,
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
        invalid_assignments = [
            item["folder"] for item in ledger["datasets"].values()
            if item.get("status") not in {"completed", "changed"}
            and (
                not isinstance(item.get("worker"), int)
                or int(item["worker"]) < 0
                or int(item["worker"]) >= worker_count
            )
        ]
        if invalid_assignments:
            raise BackupError(
                "persisted worker assignments do not fit configured worker_count: "
                f"{sorted(invalid_assignments)}"
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
            work_dir=worker_root / "catalog",
        )
        trainer_yaml = Path(str((config.get("queue") or {})["trainer_yaml"])).resolve()
        recipe = yaml.safe_load(trainer_yaml.read_text(encoding="utf-8"))
        process = recipe["config"]["process"][0]
        exports: list[dict[str, Any]] = []
        archives: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for item in candidates:
            model, _ = store.ensure_model({
                "name": item["name"], "base_arch": process["model"]["arch"],
                "base_model": process["model"]["name_or_path"],
                "trigger_word": str(item["trigger_word"]),
                "destination_kind": "loras",
            })
            identity, run_id = _execution_identity(item, model, trainer_yaml, worker_id)
            run_root = worker_root / "runs" / run_id
            phase_path = run_root / "execution-state.json"
            if (
                not phase_path.is_file()
                and item.get("status") in {"queued", "incomplete"}
                and item.get("run_id")
            ):
                raise BackupError(
                    f"durable execution state is missing for previously dispatched dataset: {item['folder']}"
                )
            phase = _phase_state(phase_path, identity, run_id)
            ledger_store.update_status(
                str(item["folder"]), str(item["fingerprint"]), "queued",
                details={
                    "catalog_id": int(model["id"]), "catalog_folder": model["folder"],
                    "run_id": run_id, "execution_identity": identity,
                    "execution_phase": str(phase["phase"]),
                },
            )
            queue_path = _write_queue(
                config=config, items=(item,), models={str(item["folder"]): model},
                dataset_root=dataset_root, work_root=run_root,
                repo_id=repo_id, repo_type=repo_type, worker_id=worker_id,
            )
            before_training = {snapshot.folder: snapshot.fingerprint for snapshot in scan_dataset_root(
                dataset_root, exposures=int(discovery.get("target_exposures", 126)),
                default_trigger_word=str((config.get("queue") or {}).get("trigger_word", "Owhx")),
            )}
            if before_training.get(str(item["folder"])) != item["fingerprint"]:
                raise BackupError(f"dataset changed after quiet-period acceptance: {item['folder']}")
            queue = queue_factory(queue_path)
            jobs = queue.materialize()
            if len(jobs) != 1:
                raise BackupError("per-dataset queue must materialize exactly one job")
            job = jobs[0]
            state = queue.run()
            job_status = (state["jobs"].get(job.job_id) or {}).get("status", "unknown")
            if job_status != "completed":
                _record_phase(
                    phase_path, phase, "queue_incomplete", job_id=job.job_id,
                    job_status=job_status,
                )
                ledger_store.update_status(
                    str(item["folder"]), str(item["fingerprint"]), "incomplete",
                    details={
                        "job_id": job.job_id, "job_status": job_status,
                        "run_id": run_id, "execution_phase": "queue_incomplete",
                    },
                )
                raise BackupError(f"unified training job did not complete: {item['folder']}")
            completed_at = str(phase.get("completed_at") or _completed_at(clock))
            _record_phase(
                phase_path, phase, "training_and_evaluation_completed",
                job_id=job.job_id, completed_at=completed_at,
            )
            output = job.output_root / job.job_id
            record, evidence = publisher(
                client=hub, repo_id=repo_id, repo_type=repo_type,
                run_id=run_id, job_id=job.job_id,
                report_path=output / ".automation" / "evaluation.json",
                sample_root=output / "samples", work_dir=run_root / "results" / job.job_id,
                catalog_prefix=str((config.get("sync") or {}).get("catalog_prefix", "training-backups")),
                completed_at=completed_at,
            )
            _record_phase(
                phase_path, phase, "exported", result_revision=record["revision"],
                result_status=record.get("status"), completed_at=completed_at,
            )
            if record.get("status") == "available":
                latest_result = latest_sync(
                    client=hub, repo_id=repo_id, repo_type=repo_type,
                    source_revision=str(record["revision"]), loras_root=loras_root,
                    work_dir=run_root / "sync-after-export",
                    model_ids=(int(model["id"]),), ranks=(1,),
                    catalog_prefix=str((config.get("sync") or {}).get("catalog_prefix", "training-backups")),
                    results_prefix=str((config.get("sync") or {}).get("results_prefix", "training-results")),
                )
                if latest_result["status"] != "completed":
                    raise BackupError(f"post-export top1 sync failed: {item['folder']}")
                record["top1_sync_status"] = "completed"
            else:
                record["top1_sync_status"] = "unavailable"
            _record_phase(phase_path, phase, "synced", completed_at=completed_at)
            archive_files: list[tuple[Path, str]] = [
                (queue_path, "queue.yaml"), (queue.state_path, "queue-state.json"),
                (phase_path, "execution-state.json"),
                *evidence,
                (output / ".automation" / "evaluation.json", f"jobs/{job.job_id}/evaluation.json"),
                *(
                    (path, f"jobs/{job.job_id}/samples/{path.name}")
                    for path in sorted((output / "samples").iterdir()) if path.is_file()
                ),
            ]
            archive = archive_factory(
                client=hub, repo_id=repo_id, repo_type=repo_type,
                remote_prefix=str((config.get("archive") or {}).get("remote_prefix", "training-archives")),
                run_id=run_id, shard_id=job.job_id,
                state_path=run_root / "archive-state.json",
            ).publish(archive_files, {
                "status": "completed", "run_id": run_id, "worker_id": worker_id,
                "completed_at": completed_at,
                "jobs": [{"job_id": job.job_id, "dataset_fingerprint": item["fingerprint"]}],
            })
            _record_phase(
                phase_path, phase, "archived", completed_at=completed_at,
                archive_completion_revision=archive["completion_revision"],
            )
            ledger_store.update_status(
                str(item["folder"]), str(item["fingerprint"]), "completed",
                details={
                    "job_id": job.job_id, "run_id": run_id,
                    "execution_phase": "completed", "completed_at": completed_at,
                    "result_revision": record["revision"],
                    "archive_completion_revision": archive["completion_revision"],
                },
            )
            _record_phase(phase_path, phase, "completed", completed_at=completed_at)
            exports.append(record)
            archives.append(archive)
            runs.append({"folder": item["folder"], "run_id": run_id, "job_id": job.job_id})
        summary.update({
            "status": "completed", "runs": runs, "exports": exports,
            "archives": archives,
        })
        if len(runs) == 1:
            summary.update({"run_id": runs[0]["run_id"], "archive": archives[0]})
        return summary
