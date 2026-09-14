from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .archive import EvidenceArchive
from .backup import BackupError, HuggingFaceBackupClient
from .catalog import CatalogStore, resolve_model, safe_relative_path
from .lifecycle import SimplePodClient, verify_instance_identity, wait_for_binding
from .queue import TrainingQueue
from .staging import COMMIT_RE, PinnedDatasetStager
from .state import atomic_write_json


MANIFEST_SCHEMA = 1
RECIPE_ID = "subject_likeness_masked_flux2_klein9b"
DEFAULT_RECIPE_PATH = Path(
    "/app/ai-toolkit/config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml"
)
COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MODEL_SOURCE_KINDS = {"base_model", "depth_model", "text_encoder", "vae"}
REQUIRED_MODEL_SOURCE_KINDS = {"base_model", "depth_model"}
NATIVE_KLEIN_FILENAME = "flux-2-klein-base-9b.safetensors"
FLUX2_VAE_FILENAME = "ae.safetensors"


def _component(value: Any, field: str) -> str:
    text = str(value or "")
    if not COMPONENT_RE.fullmatch(text):
        raise BackupError(f"{field} must be one safe path component")
    return text


def validate_manifest(
    manifest: Mapping[str, Any], *, run_id: str, shard_id: str
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise BackupError("unsupported private deployment manifest schema")
    if manifest.get("recipe_id") != RECIPE_ID:
        raise BackupError(f"deployment manifest must select recipe {RECIPE_ID}")
    if str(manifest.get("run_id")) != run_id:
        raise BackupError("deployment manifest run_id does not match TRAINING_RUN_ID")
    if not COMMIT_RE.fullmatch(str(manifest.get("dataset_revision", ""))):
        raise BackupError("deployment manifest requires an immutable dataset_revision")
    if int((manifest.get("storage") or {}).get("minimum_free_bytes", 0)) <= 0:
        raise BackupError("deployment manifest requires storage.minimum_free_bytes")
    expected_jobs = int(manifest.get("expected_jobs_per_shard", 3))
    if expected_jobs != 3:
        raise BackupError("parallel deployment requires exactly three jobs per shard")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or any(not isinstance(item, Mapping) for item in datasets):
        raise BackupError("deployment manifest datasets must be a list of mappings")
    if len(datasets) != 2 * expected_jobs:
        raise BackupError("parallel deployment requires exactly six datasets")
    normalized = [dict(item) for item in datasets]
    shard_ids = [_component(item.get("shard_id"), "dataset shard_id") for item in normalized]
    groups = set(shard_ids)
    if len(groups) != 2 or any(shard_ids.count(group) != expected_jobs for group in groups):
        raise BackupError("parallel deployment requires exactly two disjoint three-job shards")
    if shard_id not in groups:
        raise BackupError(f"shard {shard_id!r} is absent from the deployment manifest")
    ids = [_component(item.get("id"), "dataset id") for item in normalized]
    catalog_names = [str(item.get("catalog_name") or "") for item in normalized]
    try:
        catalog_ids = [int(item.get("expected_catalog_id", 0)) for item in normalized]
    except (TypeError, ValueError) as exc:
        raise BackupError("expected catalog ids must be positive integers") from exc
    if len(ids) != len(set(ids)):
        raise BackupError("dataset ids must be globally unique across shards")
    normalized_names = [name.casefold() for name in catalog_names]
    if any(not name for name in catalog_names) or len(normalized_names) != len(set(normalized_names)):
        raise BackupError("catalog names must be nonempty and globally unique across shards")
    if any(value <= 0 for value in catalog_ids) or len(catalog_ids) != len(set(catalog_ids)):
        raise BackupError("expected catalog ids must be positive and globally unique across shards")
    for item in normalized:
        if int(item.get("expected_catalog_id", 0)) <= 0:
            raise BackupError("each dataset requires its pre-reserved expected_catalog_id")
        if not item.get("catalog_name") or not item.get("trigger_word"):
            raise BackupError("each dataset requires catalog_name and trigger_word")
        references = item.get("reference_paths")
        if not isinstance(references, list) or not references:
            raise BackupError("each dataset requires at least one in-training reference path")
        file_paths = {
            safe_relative_path(str(spec.get("relative_path", ""))).as_posix()
            for spec in item.get("files", [])
        }
        if not file_paths:
            raise BackupError("each dataset requires at least one staged file")
        for reference in references:
            if safe_relative_path(str(reference)).as_posix() not in file_paths:
                raise BackupError("every reference_path must name a staged dataset file")
    selected = [item for item in normalized if str(item["shard_id"]) == shard_id]
    sources = manifest.get("model_sources")
    if not isinstance(sources, list) or any(not isinstance(item, Mapping) for item in sources):
        raise BackupError("model_sources must be a list of mappings")
    source_kinds = [str(item.get("kind")) for item in sources]
    if (
        not REQUIRED_MODEL_SOURCE_KINDS.issubset(source_kinds)
        or len(source_kinds) != len(set(source_kinds))
        or not set(source_kinds).issubset(MODEL_SOURCE_KINDS)
    ):
        raise BackupError(
            "model_sources require unique base_model and depth_model roles with optional text_encoder and vae"
        )
    for source in sources:
        if not source.get("repo_id"):
            raise BackupError("every model source requires repo_id")
        if not COMMIT_RE.fullmatch(str(source.get("revision", ""))):
            raise BackupError("every model source requires an immutable 40-character revision")
        safe_relative_path(str(source.get("local_path", "")))
        patterns = source.get("allow_patterns")
        if not isinstance(patterns, list) or not patterns:
            raise BackupError("every model source requires nonempty allow_patterns")
        artifact_path = source.get("artifact_path")
        if artifact_path is not None:
            if source["kind"] not in {"base_model", "vae"}:
                raise BackupError("artifact_path is supported only for base_model and vae file roles")
            safe_relative_path(str(artifact_path))
    evaluation = manifest.get("evaluation") or {}
    if (
        evaluation.get("enabled") is not True
        or evaluation.get("require_identity_available") is not True
        or evaluation.get("reference_provenance") != "training-set"
        or evaluation.get("face_backend")
        != "training_automation.backends:InsightFaceCPUBackend"
        or not (evaluation.get("face_backend_options") or {}).get("model_dir")
    ):
        raise BackupError(
            "parallel deployment requires training-set-provenance InsightFace identity evaluation"
        )
    return selected


def _validate_catalog_reservations(
    client: HuggingFaceBackupClient,
    *,
    repo_id: str,
    repo_type: str,
    remote_prefix: str,
    selected: list[Mapping[str, Any]],
    work_dir: Path,
    canonical_base_model: str,
) -> None:
    catalog, _ = CatalogStore(
        client=client,
        repo_id=repo_id,
        repo_type=repo_type,
        catalog_path=f"{safe_relative_path(remote_prefix).as_posix()}/catalog.json",
        work_dir=work_dir,
    ).read()
    for item in selected:
        model = resolve_model(catalog, str(item["catalog_name"]))
        expected = {
            "id": int(item["expected_catalog_id"]),
            "base_arch": "flux2_klein_9b",
            "base_model": canonical_base_model,
            "trigger_word": str(item["trigger_word"]),
            "destination_kind": str(item.get("destination_kind", "loras")),
        }
        for field, value in expected.items():
            actual = int(model[field]) if field == "id" else model.get(field)
            if actual != value:
                raise BackupError(
                    f"pre-reserved catalog entry {item['catalog_name']!r} has {field}={actual!r}, expected {value!r}"
                )


def _load_manifest(
    client: HuggingFaceBackupClient,
    *,
    repo_id: str,
    repo_type: str,
    path: str,
    revision: str,
    work_dir: Path,
) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(revision):
        raise BackupError("HF_MANIFEST_REVISION must be an immutable 40-character commit SHA")
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("private deployment manifest repository is not private")
    safe_path = safe_relative_path(path).as_posix()
    work_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="deployment-manifest-", dir=work_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        client.download_file(repo_id, repo_type, safe_path, revision, temporary)
        document = json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
    if not isinstance(document, dict):
        raise BackupError("deployment manifest must be a JSON object")
    return document


def _stage_model_sources(
    sources: list[Mapping[str, Any]],
    *,
    models_root: Path,
    token: str,
    snapshot_fetch: Callable[..., str] | None = None,
) -> dict[str, str]:
    if snapshot_fetch is None:
        from huggingface_hub import snapshot_download

        snapshot_fetch = snapshot_download
    resolved: dict[str, str] = {}
    for source in sources:
        kind = str(source["kind"])
        target = (models_root.resolve() / safe_relative_path(str(source["local_path"]))).resolve()
        if not target.is_relative_to(models_root.resolve()):
            raise BackupError("model source local_path escapes models root")
        target.mkdir(parents=True, exist_ok=True)
        marker = target / ".training-automation-source.json"
        expected_marker = {
            "repo_id": str(source["repo_id"]),
            "revision": str(source["revision"]),
            "allow_patterns": list(source["allow_patterns"]),
        }
        if source.get("artifact_path") is not None:
            expected_marker["artifact_path"] = str(source["artifact_path"])
        lock_path = target.with_name(target.name + ".snapshot.lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if marker.is_file():
                if json.loads(marker.read_text(encoding="utf-8")) != expected_marker:
                    raise BackupError(f"model staging target has conflicting source metadata: {target}")
            snapshot_fetch(
                repo_id=source["repo_id"],
                revision=source["revision"],
                allow_patterns=list(source["allow_patterns"]),
                local_dir=str(target),
                token=token,
            )
            atomic_write_json(marker, expected_marker)
        resolved[kind] = str(target)
    return resolved


def _selected_artifact(root: Path, value: str, *, role: str) -> Path:
    relative = safe_relative_path(value)
    resolved_root = root.resolve()
    artifact = (resolved_root / relative).resolve()
    if not artifact.is_relative_to(resolved_root):
        raise BackupError(f"{role} artifact_path escapes its staged source")
    if not artifact.is_file():
        raise BackupError(f"{role} artifact is missing after pinned staging: {artifact}")
    return artifact


def _resolve_model_paths(
    sources: list[Mapping[str, Any]], staged_roots: Mapping[str, str]
) -> dict[str, str]:
    by_kind = {str(source["kind"]): source for source in sources}
    resolved: dict[str, str] = {}

    base_root = Path(staged_roots["base_model"])
    base_artifact = _selected_artifact(
        base_root,
        str(by_kind["base_model"].get("artifact_path", NATIVE_KLEIN_FILENAME)),
        role="native Klein transformer",
    )
    if base_artifact.name != NATIVE_KLEIN_FILENAME:
        raise BackupError(
            f"native Klein transformer must be named {NATIVE_KLEIN_FILENAME}"
        )
    resolved["base_model"] = str(base_artifact.parent)

    depth_root = Path(staged_roots["depth_model"])
    _selected_artifact(depth_root, "config.json", role="depth_model")
    _selected_artifact(depth_root, "model.safetensors", role="depth_model")
    resolved["depth_model"] = str(depth_root)

    if "text_encoder" in by_kind:
        text_encoder_root = Path(staged_roots["text_encoder"])
        _selected_artifact(text_encoder_root, "config.json", role="text_encoder")
        weights = [
            path for path in text_encoder_root.glob("*.safetensors")
            if path.is_file() and path.resolve().is_relative_to(text_encoder_root.resolve())
        ]
        if not weights:
            raise BackupError("text_encoder staged directory has no safetensors weights")
        resolved["text_encoder"] = str(text_encoder_root)

    if "vae" in by_kind:
        resolved["vae"] = str(
            _selected_artifact(
                Path(staged_roots["vae"]),
                str(by_kind["vae"].get("artifact_path", FLUX2_VAE_FILENAME)),
                role="Flux2 VAE",
            )
        )
    return resolved


def _write_queue_config(
    *,
    manifest: Mapping[str, Any],
    selected: list[dict[str, Any]],
    staged: Mapping[str, Mapping[str, Any]],
    model_paths: Mapping[str, str],
    run_root: Path,
    output_root: Path,
    repo_root: Path,
    repo_id: str,
    repo_type: str,
    shard_id: str,
    recipe_path: Path,
) -> Path:
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    process = recipe["config"]["process"][0]
    process.pop("sqlite_db_path", None)
    process["logging"]["use_ui_logger"] = False
    canonical_base_model = str(process["model"]["name_or_path"])
    process["model"]["name_or_path"] = model_paths["base_model"]
    if "text_encoder" in model_paths:
        process["model"]["te_name_or_path"] = model_paths["text_encoder"]
    if "vae" in model_paths:
        process["model"]["vae_path"] = model_paths["vae"]
    process["depth_consistency"]["model_id"] = model_paths["depth_model"]
    trainer_path = run_root / "trainer.yaml"
    trainer_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    datasets = []
    for item in selected:
        dataset_root = Path(staged[str(item["id"])]["folder"])
        datasets.append({
            "name": str(item["id"]),
            "catalog_name": str(item["catalog_name"]),
            "expected_catalog_id": int(item["expected_catalog_id"]),
            "folder": str(dataset_root),
            "trigger_word": str(item["trigger_word"]),
            "destination_kind": str(item.get("destination_kind", "loras")),
            "dataset_revision": str(item.get("dataset_revision", manifest["dataset_revision"])),
            "shard_id": shard_id,
            "reference_images": [str(dataset_root / safe_relative_path(path)) for path in item["reference_paths"]],
        })
    evaluation = dict(manifest.get("evaluation") or {})
    evaluation.setdefault("enabled", True)
    evaluation.setdefault("require_identity_available", True)
    evaluation.setdefault("reference_provenance", "training-set")
    queue = {
        "schema_version": 1,
        "trainer_yaml": str(trainer_path),
        "repo_root": str(repo_root),
        "generated_dir": str(run_root / "generated"),
        "state_path": str(run_root / "queue-state.json"),
        "training_folder": str(output_root),
        "shard_id": shard_id,
        "continue_on_error": False,
        "checkpoint_backup": {
            "enabled": True,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "token_env": "HF_TOKEN",
            "remote_prefix": str((manifest.get("catalog") or {}).get("remote_prefix", "training-backups")),
            "max_attempts": 4,
            "backoff_seconds": 2,
            "catalog": {
                "base_arch": "flux2_klein_9b",
                "base_model": canonical_base_model,
                "destination_kind": "loras",
            },
        },
        "evaluation": evaluation,
        "datasets": datasets,
    }
    queue_path = run_root / "queue.yaml"
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return queue_path


def _completion_evidence(
    queue: TrainingQueue,
    queue_state: Mapping[str, Any],
    *,
    require_identity: bool,
    staging_state: Path,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    jobs = queue.materialize()
    files: list[tuple[Path, str]] = [
        (queue.state_path, "queue-state.json"),
        (staging_state, "staging-state.json"),
    ]
    completion_jobs = []
    for job in jobs:
        entry = queue_state["jobs"].get(job.job_id, {})
        if (
            entry.get("status") != "completed"
            or entry.get("training_status") != "completed"
            or entry.get("evaluation_status") != "completed"
        ):
            raise BackupError(f"job {job.job_id} did not complete training and evaluation")
        output = job.output_root / job.job_id
        backup_path = output / ".automation" / "backup-state.json"
        report_path = output / ".automation" / "evaluation.json"
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        receipts = list(backup.get("checkpoints", {}).values())
        if (
            not receipts
            or not any(item.get("final") for item in receipts)
            or any(
                item.get("status") != "backed_up"
                or not item.get("verified")
                or not item.get("cataloged")
                for item in receipts
            )
        ):
            raise BackupError(f"job {job.job_id} has incomplete checkpoint backup receipts")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_checkpoints = report.get("checkpoints")
        if not isinstance(report_checkpoints, list) or not report_checkpoints:
            raise BackupError(f"job {job.job_id} has no evaluated checkpoint samples")
        trainer = yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
        process = trainer["config"]["process"][0]
        sample_config = process.get("sample") or {}
        configured_samples = sample_config.get("samples")
        expected_sample_count = len(configured_samples) if isinstance(configured_samples, list) else 0
        final_step = int(process.get("train", {}).get("steps", 0))
        sample_every = int(sample_config.get("sample_every", 0))
        if expected_sample_count != 23 or final_step != 1200 or sample_every != 100:
            raise BackupError(f"job {job.job_id} does not retain the required evaluation schedule")
        checkpoints_by_step = {int(item.get("step", -1)): item for item in report_checkpoints}
        expected_steps = set(range(sample_every, final_step + 1, sample_every))
        if not expected_steps.issubset(checkpoints_by_step):
            raise BackupError(f"job {job.job_id} is missing scheduled checkpoint evaluations")
        for step in sorted(expected_steps):
            checkpoint = checkpoints_by_step[step]
            if (
                checkpoint.get("sample_run_status") != "complete"
                or len(checkpoint.get("samples") or []) != expected_sample_count
            ):
                raise BackupError(
                    f"job {job.job_id} has incomplete configured sample evidence at step {step}"
                )
        if report.get("ranking", {}).get("status") != "available":
            raise BackupError(f"job {job.job_id} has incomplete prompt/seed evaluation coverage")
        if require_identity:
            if report.get("reference_identity_status", {}).get("status") != "available":
                raise BackupError(f"job {job.job_id} has no successful reference identity evaluation")
            ranking_status = report.get("identity_ranking", {}).get("status")
            if ranking_status not in {"available", "unavailable"}:
                raise BackupError(f"job {job.job_id} has no completed identity ranking outcome")
        prefix = f"jobs/{job.job_id}"
        files.extend([
            (job.config_path, f"{prefix}/trainer.yaml"),
            (backup_path, f"{prefix}/backup-state.json"),
            (report_path, f"{prefix}/evaluation.json"),
        ])
        samples = sorted(path for path in (output / "samples").glob("*") if path.is_file())
        if not samples:
            raise BackupError(f"job {job.job_id} has no sample files to archive")
        archived_samples = {path.resolve() for path in samples}
        selected_sample_paths = {
            Path(sample["path"]).resolve()
            for checkpoint in report_checkpoints
            for sample in (checkpoint.get("samples") or [])
        }
        if not selected_sample_paths or not selected_sample_paths.issubset(archived_samples):
            raise BackupError(f"job {job.job_id} evaluation report references missing sample evidence")
        files.extend((path, f"{prefix}/samples/{path.name}") for path in samples)
        completion_jobs.append({
            "job_id": job.job_id,
            "evaluation_report": f"{prefix}/evaluation.json",
            "backup_receipts": len(receipts),
            "sample_files": len(samples),
            "identity_ranking_status": report.get("identity_ranking", {}).get("status"),
            "reference_provenance": report.get("reference_provenance", "unspecified"),
        })
    return files, {"jobs": completion_jobs, "job_count": len(completion_jobs)}


def run_parallel_bootstrap(
    *,
    env: Mapping[str, str] | None = None,
    hub_client: HuggingFaceBackupClient | None = None,
    pod_client: SimplePodClient | None = None,
    queue_factory: Callable[[Path], TrainingQueue] = TrainingQueue,
    snapshot_fetch: Callable[..., str] | None = None,
    recipe_path: Path | None = None,
) -> dict[str, Any]:
    values = dict(os.environ if env is None else env)
    required = [
        "HF_REPO_ID", "HF_MANIFEST_PATH", "HF_MANIFEST_REVISION", "HF_TOKEN",
        "TRAINING_RUN_ID", "TRAINING_SHARD_ID", "SIMPLEPOD_API_TOKEN",
    ]
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise BackupError(f"parallel bootstrap missing required environment: {', '.join(missing)}")
    repo_id = values["HF_REPO_ID"]
    repo_type = values.get("HF_REPO_TYPE", "dataset")
    run_id = _component(values["TRAINING_RUN_ID"], "TRAINING_RUN_ID")
    shard_id = _component(values["TRAINING_SHARD_ID"], "TRAINING_SHARD_ID")
    storage_root = Path(values.get("TRAINING_STORAGE_ROOT", "/storage")).resolve()
    run_root = storage_root / "automation" / run_id / shard_id
    output_root = storage_root / "output" / run_id / shard_id
    datasets_root = storage_root / "datasets" / run_id / shard_id
    models_root = storage_root / "models"
    state_path = run_root / "bootstrap-state.json"
    run_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "shard_id": shard_id,
        "status": "starting", "delete_requested": False,
    }
    atomic_write_json(state_path, state)
    client = hub_client or HuggingFaceBackupClient(values["HF_TOKEN"])
    simplepod = pod_client or SimplePodClient(values["SIMPLEPOD_API_TOKEN"])
    try:
        manifest = _load_manifest(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            path=values["HF_MANIFEST_PATH"],
            revision=values["HF_MANIFEST_REVISION"],
            work_dir=run_root,
        )
        manifest_path = run_root / "deployment-manifest.json"
        atomic_write_json(manifest_path, manifest)
        selected = validate_manifest(manifest, run_id=run_id, shard_id=shard_id)
        recipe = recipe_path or Path(values.get("TRAINING_RECIPE_PATH", DEFAULT_RECIPE_PATH))
        canonical_base_model = str(
            yaml.safe_load(recipe.read_text(encoding="utf-8"))["config"]["process"][0]["model"]["name_or_path"]
        )
        catalog_prefix = str((manifest.get("catalog") or {}).get("remote_prefix", "training-backups"))
        _validate_catalog_reservations(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_prefix=catalog_prefix,
            selected=selected,
            work_dir=run_root,
            canonical_base_model=canonical_base_model,
        )
        binding_config = manifest.get("binding") or {}
        binding_path = str(binding_config.get("remote_path", "")).replace("{shard_id}", shard_id)
        safe_relative_path(binding_path)
        binding = wait_for_binding(
            source=client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_path=binding_path,
            run_id=run_id,
            shard_id=shard_id,
            wait_seconds=float(binding_config.get("wait_seconds", 900)),
            poll_seconds=float(binding_config.get("poll_seconds", 5)),
        )
        verify_instance_identity(simplepod, binding)
        minimum_free = int(manifest["storage"]["minimum_free_bytes"])
        available = shutil.disk_usage(storage_root).free
        if available < minimum_free:
            raise BackupError(
                f"storage preflight failed: {available} bytes free, {minimum_free} required"
            )
        state.update({"status": "staging", "instance_id": binding.instance_id})
        atomic_write_json(state_path, state)
        staging_state = run_root / "staging-state.json"
        stager = PinnedDatasetStager(
            client=client,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=str(manifest["dataset_revision"]),
            target_root=datasets_root,
            state_path=staging_state,
        )
        staged = {str(item["id"]): stager.stage_dataset(item) for item in selected}
        staged_model_roots = _stage_model_sources(
            list(manifest["model_sources"]),
            models_root=models_root,
            token=values["HF_TOKEN"],
            snapshot_fetch=snapshot_fetch,
        )
        model_paths = _resolve_model_paths(
            list(manifest["model_sources"]), staged_model_roots
        )
        queue_path = _write_queue_config(
            manifest=manifest,
            selected=selected,
            staged=staged,
            model_paths=model_paths,
            run_root=run_root,
            output_root=output_root,
            repo_root=Path(values.get("TRAINING_REPO_ROOT", "/app/ai-toolkit")),
            repo_id=repo_id,
            repo_type=repo_type,
            shard_id=shard_id,
            recipe_path=recipe,
        )
        state["status"] = "training"
        atomic_write_json(state_path, state)
        queue = queue_factory(queue_path)
        queue_state = queue.run()
        files, completion = _completion_evidence(
            queue,
            queue_state,
            require_identity=bool((manifest.get("evaluation") or {}).get("require_identity_available", True)),
            staging_state=staging_state,
        )
        state["status"] = "archiving"
        atomic_write_json(state_path, state)
        files.extend([
            (manifest_path, "deployment-manifest.json"),
            (state_path, "bootstrap-state.json"),
        ])
        for source in manifest["model_sources"]:
            marker = Path(staged_model_roots[str(source["kind"])]) / ".training-automation-source.json"
            files.append((marker, f"model-sources/{source['kind']}.json"))
        completion.update({
            "manifest_revision": values["HF_MANIFEST_REVISION"],
            "dataset_revision": manifest["dataset_revision"],
            "reference_provenance": str(
                (manifest.get("evaluation") or {}).get("reference_provenance", "unspecified")
            ),
            "model_sources": [
                {
                    "kind": source["kind"],
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "local_path": source["local_path"],
                    "allow_patterns": source["allow_patterns"],
                    **(
                        {"artifact_path": source["artifact_path"]}
                        if source.get("artifact_path") is not None else {}
                    ),
                }
                for source in manifest["model_sources"]
            ],
        })
        archive = EvidenceArchive(
            client=client,
            repo_id=repo_id,
            repo_type=repo_type,
            remote_prefix=str((manifest.get("archive") or {}).get("remote_prefix", "training-runs")),
            run_id=run_id,
            shard_id=shard_id,
            state_path=run_root / "archive-state.json",
        )
        archive_state = archive.publish(files, completion)
        if archive_state.get("status") != "completed":
            raise BackupError("remote completion archive did not verify")
        verify_instance_identity(simplepod, binding)
        state.update({
            "status": "delete-requested",
            "archive_completion_revision": archive_state["completion_revision"],
            "delete_requested": True,
        })
        atomic_write_json(state_path, state)
        simplepod.delete(binding.instance_id)
        state["status"] = "delete-accepted"
        atomic_write_json(state_path, state)
        return state
    except Exception as exc:
        state.update({
            "status": "failed",
            "delete_requested": bool(state.get("delete_requested", False)),
            "error": f"{type(exc).__name__}: {exc}",
        })
        atomic_write_json(state_path, state)
        raise
