from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .archive import EvidenceArchive
from .backup import EVIDENCE_CONTRACT, BackupError, HuggingFaceBackupClient
from .catalog import CatalogStore, resolve_model, safe_relative_path
from .evaluation import preflight_evaluation_backends
from .lifecycle import (
    InstanceBinding,
    SimplePodClient,
    delete_verified_instance,
    verify_instance_identity,
    wait_for_binding,
)
from .queue import (
    EXTENSION_OPTIONAL_FIELDS,
    EXTENSION_REQUIRED_FIELDS,
    QueueConfigurationError,
    TrainingQueue,
    normalized_phase,
)
from .results import publish_ranked_results
from .staging import COMMIT_RE, PinnedDatasetStager
from .state import atomic_write_json, read_json
from .sync import sync_ranked_loras


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
LEGACY_TRAINING_STEPS = 1200
DEFAULT_SAMPLE_EVERY = 200
SUPPORTED_SAMPLE_CADENCES = {100, DEFAULT_SAMPLE_EVERY}
RESOLUTION_REPEATS = [16, 4, 1]
TRAIN_BATCH_SIZE = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _component(value: Any, field: str) -> str:
    text = str(value or "")
    if not COMPONENT_RE.fullmatch(text):
        raise BackupError(f"{field} must be one safe path component")
    return text


def _load_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise BackupError(f"{description} is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise BackupError(f"{description} must be a JSON object")
    return document


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackupError(f"{field} must be a positive integer")
    return value


def _declared_phase_fields(item: Mapping[str, Any]) -> set[str]:
    """The phase fields this dataset declares, as far as they can be read.

    This only decides whether the schedule may carry a reordered exposure
    budget. The authoritative validation of the declaration stays in
    ``_validate_extension``, which runs on the same dataset right after and
    refuses a malformed phase, so an unreadable declaration relaxes nothing that
    survives the manifest check.
    """
    raw = item.get("extend_from")
    if not isinstance(raw, Mapping):
        return set()
    phase = raw.get("phase")
    if not isinstance(phase, Mapping):
        return set()
    changes = phase.get("changes")
    if not isinstance(changes, (list, tuple)):
        return set()
    return {str(field) for field in changes}


def _is_reordering(value: Any, reference: list[int]) -> bool:
    """True when ``value`` holds exactly the same items as ``reference``."""
    if not isinstance(value, list) or len(value) != len(reference):
        return False
    remaining = list(reference)
    for item in value:
        if item not in remaining:
            return False
        remaining.remove(item)
    return not remaining


def _validate_training_schedule(
    item: Mapping[str, Any], *, declared_fields: set[str] | frozenset[str] = frozenset()
) -> None:
    has_steps = "training_steps" in item
    has_accounting = "training_accounting" in item
    if has_steps != has_accounting:
        raise BackupError("training_steps and training_accounting must be supplied together")
    if not has_steps:
        return
    steps = _positive_int(item["training_steps"], "training_steps")
    accounting = item["training_accounting"]
    required = {
        "source_image_count", "loader_batches_per_epoch", "loader_epochs",
        "resolution_repeats", "batch_size", "original_image_exposures",
        "partial_bucket_batches",
    }
    if not isinstance(accounting, Mapping) or set(accounting) != required:
        raise BackupError(f"training_accounting requires exactly {sorted(required)}")
    source_count = _positive_int(accounting["source_image_count"], "source_image_count")
    batches = _positive_int(accounting["loader_batches_per_epoch"], "loader_batches_per_epoch")
    epochs = _positive_int(accounting["loader_epochs"], "loader_epochs")
    exposures = _positive_int(accounting["original_image_exposures"], "original_image_exposures")
    if "datasets.*.num_repeats" in declared_fields:
        # A declared refinement phase may move the exposure budget toward the
        # highest resolution, but not change the budget: the same repeats in a
        # different order keep sum(repeats) identical, so the exposures identity
        # checked below is unaffected. A dataset without that declaration stays
        # pinned to the exact list.
        if not _is_reordering(accounting["resolution_repeats"], RESOLUTION_REPEATS):
            raise BackupError(
                f"a declared phase may reorder {RESOLUTION_REPEATS}, not change it: "
                f"{accounting['resolution_repeats']!r}"
            )
    elif accounting["resolution_repeats"] != RESOLUTION_REPEATS:
        raise BackupError(f"resolution_repeats must equal {RESOLUTION_REPEATS}")
    if "train.batch_size" in declared_fields:
        # A declared convergence phase cuts the same exposure budget into more,
        # smaller gradient steps; the batch size is then the phase's own, and the
        # two identities checked below still hold because neither depends on it.
        # A dataset without that declaration stays pinned to the exact value.
        _positive_int(accounting["batch_size"], "batch_size")
    elif accounting["batch_size"] != TRAIN_BATCH_SIZE:
        raise BackupError(f"batch_size must equal {TRAIN_BATCH_SIZE}")
    if accounting["partial_bucket_batches"] != "un-padded":
        raise BackupError("partial_bucket_batches must be 'un-padded'")
    staged_images = sum(
        Path(str(spec.get("relative_path", ""))).suffix.casefold() in IMAGE_SUFFIXES
        for spec in item.get("files", [])
    )
    if source_count != staged_images:
        raise BackupError("source_image_count must equal staged image file count")
    if steps != batches * epochs:
        raise BackupError("training_steps must equal loader_batches_per_epoch * loader_epochs")
    if exposures != sum(RESOLUTION_REPEATS) * epochs:
        raise BackupError("original_image_exposures must equal resolution repeats * loader epochs")


def _validate_extension(item: Mapping[str, Any]) -> None:
    raw = item.get("extend_from")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise BackupError("extend_from must be a mapping")
    # The queue owns what an extension may declare; the manifest reuses that set
    # instead of keeping a second copy that can fall behind it, which is how a
    # declared refinement phase used to be refused here by name before the queue
    # could ever see it.
    required = EXTENSION_REQUIRED_FIELDS
    unknown = set(raw) - required - EXTENSION_OPTIONAL_FIELDS
    if unknown:
        raise BackupError(f"extend_from has unsupported fields: {sorted(unknown)}")
    if not required.issubset(raw):
        raise BackupError(f"extend_from requires {sorted(required)}")
    if raw.get("phase") is not None:
        try:
            normalized_phase(raw["phase"])
        except QueueConfigurationError as exc:
            raise BackupError(str(exc)) from exc
    if not str(raw["model"]).strip():
        raise BackupError("extend_from.model must name an existing catalog model")
    if not re.fullmatch(r"[0-9a-f]{64}", str(raw["dataset_fingerprint"])):
        raise BackupError(
            "extend_from.dataset_fingerprint must be the 64-character dataset content hash"
        )
    base_steps = _positive_int(raw["base_training_steps"], "extend_from.base_training_steps")
    if "training_steps" not in item:
        raise BackupError("an extended dataset must declare its new training_steps")
    if _positive_int(item["training_steps"], "training_steps") <= base_steps:
        raise BackupError("an extension must run past its base checkpoint step")


def _validate_checkpoint_policy(manifest: Mapping[str, Any]) -> None:
    raw = manifest.get("checkpoint_policy")
    if raw is None:
        return
    if not isinstance(raw, Mapping) or set(raw) != {"save_every", "max_local_step_saves"}:
        raise BackupError("checkpoint_policy requires exactly save_every and max_local_step_saves")
    save_every = _positive_int(raw["save_every"], "checkpoint_policy.save_every")
    _positive_int(raw["max_local_step_saves"], "checkpoint_policy.max_local_step_saves")
    if save_every not in SUPPORTED_SAMPLE_CADENCES:
        raise BackupError(
            "checkpoint_policy.save_every must be one of "
            f"{sorted(SUPPORTED_SAMPLE_CADENCES)}"
        )


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
        _validate_training_schedule(item, declared_fields=_declared_phase_fields(item))
        _validate_extension(item)
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
    reference_filter = evaluation.get("reference_identity_filter")
    if reference_filter is not None:
        if not isinstance(reference_filter, Mapping) or set(reference_filter) != {
            "single_face_only", "minimum_valid_count", "minimum_valid_fraction"
        }:
            raise BackupError("reference_identity_filter has unsupported fields")
        if reference_filter["single_face_only"] is not True:
            raise BackupError("reference_identity_filter requires single_face_only=true")
        if _positive_int(reference_filter["minimum_valid_count"], "minimum_valid_count") < 3:
            raise BackupError("minimum_valid_count must be at least three")
        fraction = reference_filter["minimum_valid_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.5 <= fraction <= 1:
            raise BackupError("minimum_valid_fraction must be at least 0.5 and at most one")
    landmark = evaluation.get("landmark_backend")
    if landmark is not None:
        if landmark != "training_automation.backends:UltralyticsPoseCPUBackend":
            raise BackupError("parallel deployment supports only the pinned Ultralytics pose backend")
        options = evaluation.get("landmark_backend_options") or {}
        if not options.get("model_path") or not re.fullmatch(r"[0-9a-f]{64}", str(options.get("expected_sha256", ""))):
            raise BackupError("Ultralytics pose backend requires model_path and expected_sha256")
    _validate_checkpoint_policy(manifest)
    shard_datasets = [item for item in normalized if str(item["shard_id"]) == shard_id]
    if "selected_dataset_ids" not in manifest:
        return shard_datasets
    selected_ids = manifest["selected_dataset_ids"]
    if not isinstance(selected_ids, list) or not selected_ids:
        raise BackupError("selected_dataset_ids must be a nonempty list of dataset id strings")
    if any(not isinstance(dataset_id, str) for dataset_id in selected_ids):
        raise BackupError("selected_dataset_ids must contain only dataset id strings")
    if len(selected_ids) != len(set(selected_ids)):
        raise BackupError("selected_dataset_ids must contain unique dataset ids")
    known_ids = set(ids)
    unknown_ids = sorted(set(selected_ids) - known_ids)
    if unknown_ids:
        raise BackupError(f"selected_dataset_ids contains unknown dataset ids: {unknown_ids}")
    shard_dataset_ids = {str(item["id"]) for item in shard_datasets}
    cross_shard_ids = sorted(set(selected_ids) - shard_dataset_ids)
    if cross_shard_ids:
        raise BackupError(
            f"selected_dataset_ids contains dataset ids outside shard {shard_id!r}: "
            f"{cross_shard_ids}"
        )
    selected_id_set = set(selected_ids)
    return [item for item in shard_datasets if str(item["id"]) in selected_id_set]


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
    run_id: str,
    shard_id: str,
    recipe_path: Path,
) -> Path:
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    process = recipe["config"]["process"][0]
    checkpoint_policy = manifest.get("checkpoint_policy")
    cadence = (
        int(checkpoint_policy["save_every"])
        if checkpoint_policy is not None
        else int((process.get("sample") or {}).get("sample_every", DEFAULT_SAMPLE_EVERY))
    )
    process.setdefault("save", {})["save_every"] = cadence
    process.setdefault("sample", {})["sample_every"] = cadence
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
            **({"training_steps": item["training_steps"]} if "training_steps" in item else {}),
            **({"training_accounting": item["training_accounting"]} if "training_accounting" in item else {}),
            **({"extend_from": dict(item["extend_from"])} if item.get("extend_from") else {}),
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
        # Per-checkpoint evidence during training and per-job publication after
        # each job, so an interrupted shard keeps what it already finished.
        "checkpoint_evidence": {"enabled": True},
        "results": {
            "enabled": True,
            "run_id": run_id,
            "work_dir": str(run_root / "results"),
            "catalog_prefix": str(
                (manifest.get("catalog") or {}).get("remote_prefix", "training-backups")
            ),
            "results_prefix": str(
                (manifest.get("results") or {}).get("remote_prefix", "training-results")
            ),
        },
        **({"checkpoint_policy": checkpoint_policy} if checkpoint_policy is not None else {}),
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
        # A receipt that carries the evidence contract was created by a version
        # that also had to publish that checkpoint's own samples and record, so it
        # must have them. A receipt written before the contract existed cannot be
        # completed retroactively: it is tolerated and reported as uncovered,
        # never silently counted as covered.
        covered = [item for item in receipts if item.get("evidence_contract") is not None]
        legacy = [item for item in receipts if item.get("evidence_contract") is None]
        unevidenced = sorted(
            int(item.get("step", -1))
            for item in covered
            if (item.get("evidence") or {}).get("status") != "published"
            or not (item.get("evidence") or {}).get("verified")
        )
        if unevidenced:
            raise BackupError(
                f"job {job.job_id} has checkpoints without verified per-checkpoint evidence "
                f"at steps {unevidenced}"
            )
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
        save_every = int((process.get("save") or {}).get("save_every", 0))
        if (
            expected_sample_count <= 0
            or final_step <= 0
            or sample_every not in SUPPORTED_SAMPLE_CADENCES
            or save_every != sample_every
        ):
            raise BackupError(f"job {job.job_id} does not retain the required evaluation schedule")
        checkpoints_by_step = {int(item.get("step", -1)): item for item in report_checkpoints}
        # An extended run resumes at its base step and is scheduled to produce
        # only the checkpoints past it; the earlier steps belong to the base run
        # and have no evidence here. The reduction is authorized only by a
        # completed extension record that agrees with the declaration, so an
        # ordinary job keeps base_step 0 and still requires its full set.
        base_step = 0
        if job.extend_from is not None:
            extension = entry.get("extension") or {}
            if entry.get("extension_status") != "completed" or not extension:
                raise BackupError(
                    f"job {job.job_id} declares an extension without a completed extension record"
                )
            base_step = int(extension["base_step"])
            if base_step != int(job.extend_from["base_training_steps"]):
                raise BackupError(
                    f"job {job.job_id} extension record is at step {base_step}, its declaration "
                    f"states {job.extend_from['base_training_steps']}"
                )
            if base_step >= final_step:
                raise BackupError(
                    f"job {job.job_id} extension base step {base_step} is not below its "
                    f"final step {final_step}"
                )
        expected_steps = {
            step for step in range(sample_every, final_step, sample_every) if step > base_step
        } | {final_step}
        if not expected_steps.issubset(checkpoints_by_step):
            raise BackupError(f"job {job.job_id} is missing scheduled checkpoint evaluations")
        for step in sorted(expected_steps):
            checkpoint = checkpoints_by_step[step]
            if (
                checkpoint.get("sample_run_status") != "complete"
                or len(checkpoint.get("samples") or []) != expected_sample_count
                or checkpoint.get("remote_association", {}).get("status") != "unique"
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
        landmark_configured = bool((queue.config.get("evaluation") or {}).get("landmark_backend"))
        pose_status = (report.get("pose_backend") or {}).get("status")
        if landmark_configured:
            if pose_status != "available":
                raise BackupError(f"job {job.job_id} pose backend did not execute")
            allowed_pose = {"available", "missing", "ambiguous", "occluded", "degenerate"}
            if any(
                sample.get("pose_body_landmarks", {}).get("status") not in allowed_pose
                for step in expected_steps
                for sample in checkpoints_by_step[step].get("samples", [])
            ):
                raise BackupError(f"job {job.job_id} has incomplete pose evaluation evidence")
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
            "resumed_from_step": base_step,
            "scheduled_checkpoint_steps": sorted(expected_steps),
            "checkpoint_evidence_contract": EVIDENCE_CONTRACT,
            "checkpoint_evidence_uncovered_legacy_steps": sorted(
                int(item.get("step", -1)) for item in legacy
            ),
            "checkpoint_evidence": [
                {
                    "step": int(item["step"]),
                    "final": bool(item.get("final")),
                    "catalog_checkpoint_id": item.get("catalog_checkpoint_id"),
                    "evidence_commit_id": (item.get("evidence") or {}).get("commit_id"),
                    "evaluation_remote_path": (item.get("evidence") or {}).get(
                        "evaluation_remote_path"
                    ),
                    "sample_run_status": (item.get("evidence") or {}).get("sample_run_status"),
                    "sample_count": (item.get("evidence") or {}).get("sample_count"),
                }
                for item in sorted(receipts, key=lambda value: int(value.get("step", 0)))
            ],
            "sample_files": len(samples),
            "identity_ranking_status": report.get("identity_ranking", {}).get("status"),
            "reference_provenance": report.get("reference_provenance", "unspecified"),
            "pose_backend_status": pose_status,
            "training_schedule": (
                trainer.get("meta", {}).get("training_automation_schedule")
                or {
                    "base_training_steps": LEGACY_TRAINING_STEPS,
                    "resolved_training_steps": final_step,
                    "training_accounting": None,
                    "checkpoint_policy": None,
                }
            ),
        })
    return files, {"jobs": completion_jobs, "job_count": len(completion_jobs)}


def _persist_instance_binding(
    state: dict[str, Any], state_path: Path, binding: InstanceBinding
) -> None:
    """Record the verified identity of this instance before any paid work.

    The supervisor cannot resolve the remote binding document by itself, so
    without this record a failed shard has no identity-verified way to stop its
    own instance. The write happens the moment the binding is verified, long
    before any delete request, so it never moves a durable write past one.
    """
    state.update({
        "instance_id": binding.instance_id,
        "instance_hash_id": binding.instance_hash_id,
        "instance_notes": binding.instance_notes,
    })
    atomic_write_json(state_path, state)


def _binding_for_manifest(
    *,
    client: HuggingFaceBackupClient,
    simplepod: SimplePodClient,
    manifest: Mapping[str, Any],
    repo_id: str,
    repo_type: str,
    run_id: str,
    shard_id: str,
):
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
    return binding


def _validate_recovery_queue(
    queue: TrainingQueue,
    *,
    run_root: Path,
    output_root: Path,
    repo_id: str,
    repo_type: str,
    shard_id: str,
) -> dict[str, Any]:
    expected = {
        "state_path": run_root / "queue-state.json",
        "generated_dir": run_root / "generated",
        "training_folder": output_root,
    }
    for field, path in expected.items():
        if Path(queue.config.get(field, "")).resolve() != path.resolve():
            raise BackupError(f"archive recovery queue has unexpected {field}")
    backup = queue.config.get("checkpoint_backup") or {}
    if (
        str(queue.config.get("shard_id")) != shard_id
        or backup.get("repo_id") != repo_id
        or backup.get("repo_type") != repo_type
    ):
        raise BackupError("archive recovery queue identity does not match this process")
    queue_state = read_json(queue.state_path, {})
    jobs = queue_state.get("jobs") if isinstance(queue_state, Mapping) else None
    if not isinstance(jobs, Mapping) or not jobs:
        raise BackupError("archive recovery requires a completed persisted queue")
    if any(
        item.get("status") != "completed"
        or item.get("training_status") != "completed"
        or item.get("evaluation_status") != "completed"
        for item in jobs.values()
    ):
        raise BackupError("automatic restart is limited to completed training and evaluation")
    return dict(queue_state)


def _reuse_job_publication(
    *, job_id: str, work_dir: Path, queue_state: Mapping[str, Any]
) -> tuple[dict[str, Any], list[tuple[Path, str]]] | None:
    """Return the publication this job already made, when it is still intact."""
    entry = (queue_state.get("jobs") or {}).get(job_id) or {}
    manifest_path = work_dir / "published-evidence.json"
    if entry.get("results_status") != "completed" or not manifest_path.is_file():
        return None
    manifest = _load_object(manifest_path, f"published evidence for {job_id}")
    record = manifest.get("record")
    entries = manifest.get("files")
    if (
        manifest.get("job_id") != job_id
        or not isinstance(record, Mapping)
        or not isinstance(entries, list)
        or not entries
    ):
        raise BackupError(f"job {job_id} has an unusable per-job publication manifest")
    evidence_files: list[tuple[Path, str]] = []
    for item in entries:
        if not isinstance(item, list) or len(item) != 2:
            raise BackupError(f"job {job_id} publication manifest has an invalid file entry")
        local = Path(str(item[0]))
        if not local.is_file():
            raise BackupError(
                f"job {job_id} published evidence is missing on disk: {local}"
            )
        evidence_files.append((local, str(item[1])))
    return {**dict(record), "publication": "reused-from-job"}, evidence_files


def _publish_job_results(
    *,
    queue: TrainingQueue,
    queue_state: Mapping[str, Any],
    files: list[tuple[Path, str]],
    completion: dict[str, Any],
    client: HuggingFaceBackupClient,
    repo_id: str,
    repo_type: str,
    run_id: str,
    run_root: Path,
    catalog_prefix: str,
    completed_at: str,
    loras_root: Path | None,
) -> None:
    completion_by_job = {item["job_id"]: item for item in completion["jobs"]}
    for job in queue.materialize():
        output = job.output_root / job.job_id
        work_dir = run_root / "results" / job.job_id
        reused = _reuse_job_publication(
            job_id=job.job_id, work_dir=work_dir, queue_state=queue_state
        )
        if reused is not None:
            record, evidence_files = reused
        else:
            record, evidence_files = publish_ranked_results(
                client=client,
                repo_id=repo_id,
                repo_type=repo_type,
                run_id=run_id,
                job_id=job.job_id,
                report_path=output / ".automation" / "evaluation.json",
                sample_root=output / "samples",
                work_dir=work_dir,
                catalog_prefix=catalog_prefix,
                completed_at=completed_at,
            )
        files.extend(evidence_files)
        completion_by_job[job.job_id]["automatic_result_export"] = record
        if loras_root is not None:
            sync_receipt = sync_ranked_loras(
                client=client,
                repo_id=repo_id,
                repo_type=repo_type,
                source_revision=str(record["revision"]),
                run_id=run_id,
                loras_root=loras_root,
                work_dir=run_root / "results" / job.job_id / "lora-sync",
                ranks=(1, 2, 3),
                model_ids=(int(record["model"]["id"]),),
                catalog_prefix=catalog_prefix,
            )
            if sync_receipt.get("status") != "completed":
                raise BackupError(f"job {job.job_id} ranked LoRA sync did not complete")
            sync_path = run_root / "results" / job.job_id / "ranked-lora-sync.json"
            atomic_write_json(sync_path, sync_receipt)
            files.append((sync_path, f"jobs/{job.job_id}/ranked-lora-sync.json"))
            completion_by_job[job.job_id]["ranked_lora_sync"] = sync_receipt


def _verified_source_marker(source: Mapping[str, Any], models_root: Path) -> Path:
    marker = (
        models_root / safe_relative_path(str(source["local_path"]))
        / ".training-automation-source.json"
    )
    expected = {
        "repo_id": str(source["repo_id"]),
        "revision": str(source["revision"]),
        "allow_patterns": list(source["allow_patterns"]),
    }
    if source.get("artifact_path") is not None:
        expected["artifact_path"] = str(source["artifact_path"])
    if _load_object(marker, f"model source marker {source['kind']}") != expected:
        raise BackupError(f"model source marker differs from manifest: {source['kind']}")
    return marker


def _finalize_completed_run(
    *,
    values: Mapping[str, str],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    queue: TrainingQueue,
    queue_state: Mapping[str, Any],
    staging_state: Path,
    client: HuggingFaceBackupClient,
    simplepod: SimplePodClient,
    binding: Any,
    state: dict[str, Any],
    state_path: Path,
    run_root: Path,
    models_root: Path,
    repo_id: str,
    repo_type: str,
    run_id: str,
    shard_id: str,
) -> dict[str, Any]:
    files, completion = _completion_evidence(
        queue,
        queue_state,
        require_identity=bool(
            (manifest.get("evaluation") or {}).get("require_identity_available", True)
        ),
        staging_state=staging_state,
    )
    completed_at = str(state.get("completed_at") or "")
    if not completed_at:
        completed_at = datetime.now(timezone.utc).isoformat()
        state["completed_at"] = completed_at
    state.update({"status": "publishing-results", "instance_id": binding.instance_id})
    state.pop("error", None)
    atomic_write_json(state_path, state)
    catalog_prefix = str(
        (manifest.get("catalog") or {}).get("remote_prefix", "training-backups")
    )
    loras_root_value = values.get("TRAINING_LORAS_ROOT")
    loras_root = Path(loras_root_value).expanduser().resolve() if loras_root_value else None
    if loras_root is not None and not loras_root.is_dir():
        raise BackupError(f"TRAINING_LORAS_ROOT is not an existing directory: {loras_root}")
    _publish_job_results(
        queue=queue,
        queue_state=queue_state,
        files=files,
        completion=completion,
        client=client,
        repo_id=repo_id,
        repo_type=repo_type,
        run_id=run_id,
        run_root=run_root,
        catalog_prefix=catalog_prefix,
        completed_at=completed_at,
        loras_root=loras_root,
    )
    state.update({"status": "archiving", "instance_id": binding.instance_id})
    state.pop("error", None)
    atomic_write_json(state_path, state)
    files.extend([
        (manifest_path, "deployment-manifest.json"),
        (state_path, "bootstrap-state.json"),
    ])
    worker_recovery_state = run_root / "worker-recovery-state.json"
    if worker_recovery_state.is_file():
        files.append((worker_recovery_state, "worker-recovery-state.json"))
    for source in manifest["model_sources"]:
        marker = _verified_source_marker(source, models_root)
        files.append((marker, f"model-sources/{source['kind']}.json"))
    completion.update({
        "completed_at": completed_at,
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
        remote_prefix=str(
            (manifest.get("archive") or {}).get("remote_prefix", "training-runs")
        ),
        run_id=run_id,
        shard_id=shard_id,
        state_path=run_root / "archive-state.json",
    )
    archive_state = archive.publish(files, completion)
    if archive_state.get("status") != "completed":
        raise BackupError("remote completion archive did not verify")

    def record_delete_request() -> None:
        state.update({
            "status": "delete-requested",
            "archive_completion_revision": archive_state["completion_revision"],
            "delete_requested": True,
        })
        atomic_write_json(state_path, state)

    outcome = delete_verified_instance(
        simplepod, binding, before_request=record_delete_request
    )
    state.update({"status": "delete-accepted", "delete_outcome": outcome})
    atomic_write_json(state_path, state)
    return state


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
    has_prior_state = state_path.is_file()
    prior_state = read_json(state_path, {}) if has_prior_state else None
    if has_prior_state and (
        not isinstance(prior_state, Mapping)
        or prior_state.get("schema_version") != 1
        or prior_state.get("run_id") != run_id
        or prior_state.get("shard_id") != shard_id
    ):
        raise BackupError("persisted bootstrap state belongs to a different run or shard")
    if has_prior_state and bool(prior_state.get("delete_requested")):
        raise BackupError(
            "automatic restart refused because a prior delete request has an uncertain outcome"
        )
    state: dict[str, Any] = (
        dict(prior_state)
        if has_prior_state
        else {
            "schema_version": 1,
            "run_id": run_id,
            "shard_id": shard_id,
            "status": "starting",
            "delete_requested": False,
        }
    )
    if not has_prior_state:
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
        if has_prior_state:
            local_manifest = _load_object(manifest_path, "persisted deployment manifest")
            if local_manifest != manifest:
                raise BackupError("persisted deployment manifest differs from its immutable revision")
            validate_manifest(manifest, run_id=run_id, shard_id=shard_id)
            binding = _binding_for_manifest(
                client=client,
                simplepod=simplepod,
                manifest=manifest,
                repo_id=repo_id,
                repo_type=repo_type,
                run_id=run_id,
                shard_id=shard_id,
            )
            _persist_instance_binding(state, state_path, binding)
            queue = queue_factory(run_root / "queue.yaml")
            queue_state = _validate_recovery_queue(
                queue,
                run_root=run_root,
                output_root=output_root,
                repo_id=repo_id,
                repo_type=repo_type,
                shard_id=shard_id,
            )
            state["status"] = "archive-recovery"
            state.pop("error", None)
            atomic_write_json(state_path, state)
            return _finalize_completed_run(
                values=values,
                manifest=manifest,
                manifest_path=manifest_path,
                queue=queue,
                queue_state=queue_state,
                staging_state=run_root / "staging-state.json",
                client=client,
                simplepod=simplepod,
                binding=binding,
                state=state,
                state_path=state_path,
                run_root=run_root,
                models_root=models_root,
                repo_id=repo_id,
                repo_type=repo_type,
                run_id=run_id,
                shard_id=shard_id,
            )
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
        binding = _binding_for_manifest(
            client=client,
            simplepod=simplepod,
            manifest=manifest,
            repo_id=repo_id,
            repo_type=repo_type,
            run_id=run_id,
            shard_id=shard_id,
        )
        _persist_instance_binding(state, state_path, binding)
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
        preflight_evaluation_backends(manifest.get("evaluation") or {})
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
            run_id=run_id,
            shard_id=shard_id,
            recipe_path=recipe,
        )
        state["status"] = "training"
        atomic_write_json(state_path, state)
        queue = queue_factory(queue_path)
        queue_state = queue.run()
        return _finalize_completed_run(
            values=values,
            manifest=manifest,
            manifest_path=manifest_path,
            queue=queue,
            queue_state=queue_state,
            staging_state=staging_state,
            client=client,
            simplepod=simplepod,
            binding=binding,
            state=state,
            state_path=state_path,
            run_root=run_root,
            models_root=models_root,
            repo_id=repo_id,
            repo_type=repo_type,
            run_id=run_id,
            shard_id=shard_id,
        )
    except Exception as exc:
        state.update({
            "status": "failed",
            "delete_requested": bool(state.get("delete_requested", False)),
            "error": f"{type(exc).__name__}: {exc}",
        })
        atomic_write_json(state_path, state)
        raise
