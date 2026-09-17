from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .evaluation import evaluate_job, preserve_cuda_visibility
from .state import atomic_write_json, read_json


QUEUE_CONFIG_SCHEMA = 1
QUEUE_STATE_SCHEMA = 2
DATASET_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_CAPTION_SUFFIXES = {".txt"}
EXTENSION_REQUIRED_FIELDS = {"model", "dataset_fingerprint", "base_training_steps"}
EXTENSION_OPTIONAL_FIELDS = {"checkpoint_id", "reason", "phase"}
# A declared refinement phase is the only way a continuation may change the
# training shape instead of just its duration, and it may name only these
# fields. '*' stands for exactly one path segment, so datasets.*.num_repeats
# covers every configured dataset without covering anything else.
EXTENSION_PHASE_FIELDS = (
    "train.timestep_type",
    "train.content_or_style",
    "train.lr",
    "datasets.*.num_repeats",
    "datasets.*.resolution",
)
EXTENSION_PHASE_FIELD_KEYS = {"name", "changes"}
# Fields that may legitimately differ between the base run and its extension:
# local staging paths, the automation block, the duration itself, and the
# sampling/saving cadence, which never enters the trained weights.
EXTENSION_IGNORED_FIELDS = (
    "training_folder",
    "sqlite_db_path",
    "device",
    "performance_log_every",
    "checkpoint_backup",
    "logging",
    "sample",
    "save",
    "train.steps",
    "model.name_or_path",
    "model.te_name_or_path",
    "model.vae_path",
    "depth_consistency.model_id",
    "datasets.0.folder_path",
    "datasets.0.mask_path",
)
EXTENSION_REPORTED_DIFFERENCES = 12


class QueueConfigurationError(ValueError):
    pass


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "dataset"


def _load_one_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        documents = [item for item in yaml.safe_load_all(handle) if item is not None]
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise QueueConfigurationError(f"{path} must contain exactly one YAML document")
    return documents[0]


ORDERED_DICT_TAG = "tag:yaml.org,2002:python/object/apply:collections.OrderedDict"


class TrainerConfigLoader(yaml.SafeLoader):
    """Read the config.yaml the trainer saves next to its checkpoints.

    The trainer dumps its ``OrderedDict`` configuration with the default PyYAML
    dumper, so the file carries one Python tag. This loader stays a SafeLoader
    and adds a constructor for that single tag only; no other Python object can
    be instantiated from a remote file.
    """


def _construct_ordered_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> dict[str, Any]:
    arguments = loader.construct_sequence(node, deep=True)
    if not arguments:
        return {}
    pairs = arguments[0]
    if not isinstance(pairs, list):
        raise QueueConfigurationError("saved trainer configuration has an unreadable mapping")
    mapping: dict[str, Any] = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise QueueConfigurationError("saved trainer configuration has an unreadable mapping")
        mapping[pair[0]] = pair[1]
    return mapping


TrainerConfigLoader.add_constructor(ORDERED_DICT_TAG, _construct_ordered_mapping)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dataset_content_fingerprint(folder: Path) -> str:
    """Hash every image and caption by name, size and content.

    This is the same content identity the discovery path records for a dataset,
    so an extension can prove that it continues on exactly the same photographs
    and captions instead of trusting a folder name.
    """
    from .backup import sha256_file

    folder = Path(folder)
    if not folder.is_dir():
        raise QueueConfigurationError(f"dataset folder does not exist: {folder}")
    files = [
        item for item in folder.iterdir()
        if item.is_file()
        and not item.name.startswith(".")
        and item.suffix.casefold() in DATASET_IMAGE_SUFFIXES | DATASET_CAPTION_SUFFIXES
    ]
    if not files:
        raise QueueConfigurationError(f"dataset folder has no image or caption file: {folder}")
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda path: path.name.casefold()):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}}
        flattened: dict[str, Any] = {}
        for key in value:
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], child))
        return flattened
    if isinstance(value, (list, tuple)):
        if not value:
            return {prefix: []}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{prefix}.{index}"))
        return flattened
    return {prefix: value}


_MISSING = object()


def _matches_field(key: str, pattern: str) -> bool:
    """Match a flattened config key against one declared phase field.

    Segments are compared one by one and '*' stands for exactly one segment, so
    ``datasets.*.num_repeats`` covers ``datasets.0.num_repeats.2`` while
    ``train.lr`` never covers a differently named neighbour such as
    ``train.lr_scheduler``.
    """
    pattern_parts = pattern.split(".")
    key_parts = key.split(".")
    if len(key_parts) < len(pattern_parts):
        return False
    return all(
        part == "*" or part == key_parts[index] for index, part in enumerate(pattern_parts)
    )


def _training_differences(
    base_process: Mapping[str, Any], new_process: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    base = _flatten(dict(base_process))
    current = _flatten(dict(new_process))
    differences: list[tuple[str, Any, Any]] = []
    for key in sorted(set(base) | set(current)):
        if any(key == item or key.startswith(f"{item}.") for item in EXTENSION_IGNORED_FIELDS):
            continue
        before = base.get(key, _MISSING)
        after = current.get(key, _MISSING)
        if before != after:
            differences.append((key, before, after))
    return differences


def recipe_differences(
    base_process: Mapping[str, Any],
    new_process: Mapping[str, Any],
    *,
    declared_fields: Sequence[str] = (),
) -> list[str]:
    """Return the training-relevant fields that differ between two job configs.

    ``declared_fields`` names the fields of a declared refinement phase: those
    are reported by ``phase_deviations`` instead of refusing the continuation.
    Without a declaration the result is exactly the historical one.
    """
    differences = []
    for key, before, after in _training_differences(base_process, new_process):
        if any(_matches_field(key, item) for item in declared_fields):
            continue
        differences.append(
            f"{key}: base={'<absent>' if before is _MISSING else before!r} "
            f"extension={'<absent>' if after is _MISSING else after!r}"
        )
    return differences


def phase_deviations(
    base_process: Mapping[str, Any],
    new_process: Mapping[str, Any],
    declared_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Return the differences a declared refinement phase actually produced.

    The declaration states intent; this states what the base and the refined
    configuration really hold, so the lineage recorded with the run is the
    observed deviation and not only the promise. A field that is absent on one
    side simply has no entry for that side.
    """
    deviations: list[dict[str, Any]] = []
    for key, before, after in _training_differences(base_process, new_process):
        if not any(_matches_field(key, item) for item in declared_fields):
            continue
        entry: dict[str, Any] = {"field": key}
        if before is not _MISSING:
            entry["base"] = before
        if after is not _MISSING:
            entry["extension"] = after
        deviations.append(entry)
    return deviations


def normalized_phase(raw: Any) -> dict[str, Any]:
    """Validate the declaration of a refinement phase.

    The declaration is deliberately narrow: a phase names itself and the exact
    training-shape fields it intends to change, chosen from a fixed set. Every
    other field of the base recipe still has to match.

    This is the single validation of a phase in the project: the deployment
    manifest calls it too, so a manifest and a queue configuration cannot drift
    apart on what a phase is allowed to declare.
    """
    if not isinstance(raw, Mapping):
        raise QueueConfigurationError("extend_from.phase must be a mapping")
    unknown = set(raw) - EXTENSION_PHASE_FIELD_KEYS
    if unknown:
        raise QueueConfigurationError(f"extend_from.phase has unsupported fields: {sorted(unknown)}")
    missing = EXTENSION_PHASE_FIELD_KEYS - set(raw)
    if missing:
        raise QueueConfigurationError(f"extend_from.phase requires {sorted(missing)}")
    name = str(raw["name"]).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise QueueConfigurationError(
            "extend_from.phase.name must be a short lowercase slug of letters, digits and dashes"
        )
    changes = raw["changes"]
    if not isinstance(changes, (list, tuple)) or not changes:
        raise QueueConfigurationError(
            "extend_from.phase.changes must list at least one field this phase changes"
        )
    declared = [str(item) for item in changes]
    if len(set(declared)) != len(declared):
        raise QueueConfigurationError("extend_from.phase.changes must not repeat a field")
    unsupported = sorted(set(declared) - set(EXTENSION_PHASE_FIELDS))
    if unsupported:
        raise QueueConfigurationError(
            f"extend_from.phase.changes may only name {list(EXTENSION_PHASE_FIELDS)}; "
            f"unsupported: {unsupported}"
        )
    return {"name": name, "changes": sorted(declared)}


def _normalized_extension(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise QueueConfigurationError("extend_from must be a mapping")
    unknown = set(raw) - EXTENSION_REQUIRED_FIELDS - EXTENSION_OPTIONAL_FIELDS
    if unknown:
        raise QueueConfigurationError(f"extend_from has unsupported fields: {sorted(unknown)}")
    missing = EXTENSION_REQUIRED_FIELDS - set(raw)
    if missing:
        raise QueueConfigurationError(f"extend_from requires {sorted(missing)}")
    fingerprint = str(raw["dataset_fingerprint"])
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise QueueConfigurationError(
            "extend_from.dataset_fingerprint must be the 64-character dataset content hash"
        )
    base_steps = raw["base_training_steps"]
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps <= 0:
        raise QueueConfigurationError("extend_from.base_training_steps must be a positive integer")
    model = str(raw["model"]).strip()
    if not model:
        raise QueueConfigurationError("extend_from.model must name an existing catalog model")
    normalized = {
        "model": model,
        "dataset_fingerprint": fingerprint,
        "base_training_steps": int(base_steps),
    }
    if raw.get("checkpoint_id"):
        normalized["checkpoint_id"] = str(raw["checkpoint_id"])
    if raw.get("reason"):
        normalized["reason"] = str(raw["reason"])
    if raw.get("phase") is not None:
        normalized["phase"] = normalized_phase(raw["phase"])
    return normalized


def _install_local_copy(source: Path, target: Path) -> str:
    from .backup import sha256_file

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            target.is_file()
            and source.is_file()
            and target.stat().st_size == source.stat().st_size
            and sha256_file(target) == sha256_file(source)
        ):
            return "already-present"
        raise QueueConfigurationError(f"refusing to overwrite existing training state: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return "installed"


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    config_path: Path
    output_root: Path
    reference_images: tuple[Path, ...]
    training_steps: int | None = None
    training_accounting: Mapping[str, Any] | None = None
    checkpoint_policy: Mapping[str, int] | None = None
    dataset_folder: Path | None = None
    extend_from: Mapping[str, Any] | None = None


class TrainingQueue:
    def __init__(
        self,
        config_path: Path,
        *,
        run_command: Callable[[Sequence[str], Mapping[str, str]], int] | None = None,
    ):
        self.config_path = config_path.resolve()
        self.config = _load_one_yaml(self.config_path)
        if self.config.get("schema_version") != QUEUE_CONFIG_SCHEMA:
            raise QueueConfigurationError("unsupported or missing queue schema_version")
        base = self.config_path.parent
        self.trainer_path = (base / self.config["trainer_yaml"]).resolve()
        self.generated_dir = (base / self.config.get("generated_dir", "generated")).resolve()
        self.state_path = (base / self.config.get("state_path", "queue-state.json")).resolve()
        self.repo_root = Path(self.config.get("repo_root", Path(__file__).resolve().parents[1])).resolve()
        self.python = str(self.config.get("python", sys.executable))
        self.shard_id = self.config.get("shard_id")
        self._run_command = run_command or self._subprocess

    @staticmethod
    def _subprocess(command: Sequence[str], env: Mapping[str, str]) -> int:
        return subprocess.run(list(command), env=dict(env), check=False).returncode

    def _datasets(self) -> list[dict[str, Any]]:
        datasets = self.config.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise QueueConfigurationError("datasets must be a non-empty list")
        if any(not isinstance(item, dict) for item in datasets):
            raise QueueConfigurationError("each dataset must be a mapping")
        if self.shard_id is not None:
            if any("shard_id" not in item for item in datasets):
                raise QueueConfigurationError("every dataset requires shard_id in a sharded queue")
            datasets = [item for item in datasets if item.get("shard_id") == self.shard_id]
            if not datasets:
                raise QueueConfigurationError(f"no datasets assigned to shard {self.shard_id!r}")
        return datasets

    @staticmethod
    def _positive_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise QueueConfigurationError(f"{field} must be a positive integer")
        return value

    def _checkpoint_policy(self, process: Mapping[str, Any]) -> dict[str, int] | None:
        raw = self.config.get("checkpoint_policy")
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {"save_every", "max_local_step_saves"}:
            raise QueueConfigurationError(
                "checkpoint_policy requires exactly save_every and max_local_step_saves"
            )
        save_every = self._positive_int(raw["save_every"], "checkpoint_policy.save_every")
        max_local = self._positive_int(
            raw["max_local_step_saves"], "checkpoint_policy.max_local_step_saves"
        )
        sample_every = self._positive_int(
            (process.get("sample") or {}).get("sample_every"), "sample.sample_every"
        )
        if save_every != sample_every:
            raise QueueConfigurationError(
                "checkpoint_policy.save_every must equal sample.sample_every"
            )
        return {"save_every": save_every, "max_local_step_saves": max_local}

    def _assert_pending_duration_change(
        self, job_id: str, target: Path, requested_steps: int
    ) -> None:
        if not target.is_file():
            return
        existing = _load_one_yaml(target)
        try:
            existing_steps = int(existing["config"]["process"][0]["train"]["steps"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise QueueConfigurationError(
                f"existing generated config for {job_id} has no valid train.steps"
            ) from exc
        if existing_steps == requested_steps:
            return
        state = read_json(self.state_path, {"jobs": {}})
        entry = (state.get("jobs") or {}).get(job_id, {})
        training_status = entry.get("training_status")
        if training_status is None:
            legacy = entry.get("status")
            training_status = "pending" if legacy in {None, "pending"} else legacy
        if training_status != "pending":
            raise QueueConfigurationError(
                f"training_steps for {job_id} may change only while training is pending "
                f"({existing_steps} -> {requested_steps}, status={training_status})"
            )

    def materialize(self) -> list[QueueJob]:
        template_bytes = self.trainer_path.read_bytes()
        template = _load_one_yaml(self.trainer_path)
        try:
            base_process = template["config"]["process"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise QueueConfigurationError("trainer_yaml has no config.process[0]") from exc
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[QueueJob] = []
        seen: set[str] = set()
        for raw in self._datasets():
            if not isinstance(raw, dict) or not raw.get("folder"):
                raise QueueConfigurationError("each dataset requires folder")
            identity = {
                "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
                "folder": str((self.config_path.parent / raw["folder"]).resolve()),
                "trigger_word": raw.get("trigger_word"),
                "reference_images": [
                    str((self.config_path.parent / item).resolve())
                    for item in raw.get("reference_images", [])
                ],
                "name": raw.get("name"),
                "trainer_dataset": raw.get("trainer_dataset", {}),
                "dataset_revision": raw.get("dataset_revision"),
            }
            if self.shard_id is not None:
                identity["shard_id"] = self.shard_id
            extension = (
                _normalized_extension(raw["extend_from"])
                if raw.get("extend_from") is not None else None
            )
            if extension is not None:
                # An extension is an explicit, separate run identity. The key is
                # added only when requested so existing job ids stay stable.
                identity["extend_from"] = extension
            digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:12]
            job_id = f"{_slug(str(raw.get('name') or Path(raw['folder']).name))}-{digest}"
            if job_id in seen:
                raise QueueConfigurationError(f"duplicate derived job id: {job_id}")
            seen.add(job_id)
            document = copy.deepcopy(template)
            document["config"]["name"] = job_id
            process = document["config"]["process"][0]
            process["trigger_word"] = raw.get("trigger_word")
            if self.config.get("training_folder"):
                process["training_folder"] = str(self.config["training_folder"])
            dataset = copy.deepcopy(raw.get("trainer_dataset", {}))
            if base_process.get("datasets"):
                defaults = copy.deepcopy(base_process["datasets"][0])
                defaults.update(dataset)
                dataset = defaults
            dataset["folder_path"] = identity["folder"]
            process["datasets"] = [dataset]
            base_steps = self._positive_int(
                process.get("train", {}).get("steps"), "train.steps"
            )
            requested_steps = self._positive_int(
                raw.get("training_steps", base_steps), "training_steps"
            )
            accounting = copy.deepcopy(raw.get("training_accounting"))
            policy = self._checkpoint_policy(process)
            output_root = Path(process.get("training_folder", "output"))
            if not output_root.is_absolute():
                output_root = (self.repo_root / output_root).resolve()
            backup = copy.deepcopy(self.config.get("checkpoint_backup", {}))
            if backup.get("enabled"):
                catalog = backup.setdefault("catalog", {})
                catalog["name"] = str(raw.get("catalog_name") or raw.get("name") or job_id)
                catalog.setdefault("base_arch", process.get("model", {}).get("arch"))
                catalog.setdefault("base_model", process.get("model", {}).get("name_or_path"))
                catalog["trigger_word"] = raw.get("trigger_word")
                catalog["destination_kind"] = str(raw.get("destination_kind", catalog.get("destination_kind", "loras")))
                if raw.get("expected_catalog_id") is not None:
                    catalog["expected_id"] = int(raw["expected_catalog_id"])
                backup.setdefault(
                    "state_path", str(output_root / job_id / ".automation" / "backup-state.json")
                )
                if extension is not None and extension.get("phase") is not None:
                    # Lineage: the per-checkpoint evidence published for this run
                    # carries the declared phase, so a reader of the catalog can
                    # tell a refined model from a plain continuation. The key is
                    # written only for a declared phase and lives in the
                    # automation block, which is already exempt from the recipe
                    # comparison, so no other job changes by a single byte.
                    backup["refinement_phase"] = copy.deepcopy(extension["phase"])
                evidence = copy.deepcopy(self.config.get("checkpoint_evidence") or {})
                evaluation_config = copy.deepcopy(self.config.get("evaluation") or {})
                if (
                    evidence.get("enabled", True)
                    and evaluation_config.get("enabled", True)
                ):
                    backup["evidence"] = {
                        "enabled": True,
                        "during_training": bool(evidence.get("during_training", True)),
                        "output_dir": str(output_root / job_id),
                        "reference_images": list(identity["reference_images"]),
                        "evaluation": evaluation_config,
                    }
                process["checkpoint_backup"] = backup
            if extension is not None and requested_steps <= extension["base_training_steps"]:
                raise QueueConfigurationError(
                    f"extension of {job_id} must run past its base checkpoint "
                    f"({requested_steps} <= {extension['base_training_steps']})"
                )
            target = self.generated_dir / f"{job_id}.yaml"
            self._assert_pending_duration_change(job_id, target, requested_steps)
            process["train"]["steps"] = requested_steps
            if policy is not None:
                save = process.setdefault("save", {})
                save["save_every"] = policy["save_every"]
                save["max_step_saves_to_keep"] = policy["max_local_step_saves"]
            if requested_steps != base_steps or accounting is not None or policy is not None:
                document.setdefault("meta", {})["training_automation_schedule"] = {
                    "base_training_steps": base_steps,
                    "resolved_training_steps": requested_steps,
                    "training_accounting": accounting,
                    "checkpoint_policy": policy,
                }
            target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            jobs.append(
                QueueJob(
                    job_id,
                    target,
                    output_root,
                    tuple(Path(item) for item in identity["reference_images"]),
                    requested_steps,
                    accounting,
                    policy,
                    Path(identity["folder"]),
                    extension,
                )
            )
        return jobs

    def _state(self, jobs: list[QueueJob]) -> dict[str, Any]:
        state = read_json(self.state_path, {"schema_version": QUEUE_STATE_SCHEMA, "jobs": {}})
        if state.get("schema_version") == 1:
            for entry in state.get("jobs", {}).values():
                legacy = entry.get("status", "pending")
                entry["training_status"] = "completed" if legacy == "completed" else ("failed" if legacy == "failed" else "pending")
                entry["evaluation_status"] = "completed" if legacy == "completed" else "pending"
                if legacy == "running":
                    entry["interrupted"] = True
            state["schema_version"] = QUEUE_STATE_SCHEMA
        if state.get("schema_version") != QUEUE_STATE_SCHEMA:
            raise QueueConfigurationError("unsupported queue state schema")
        state.setdefault("jobs", {})
        for job in jobs:
            entry = state["jobs"].setdefault(
                job.job_id,
                {
                    "status": "pending",
                    "training_status": "pending",
                    "evaluation_status": "pending",
                    "attempts": 0,
                },
            )
            entry["resolved_training_steps"] = job.training_steps
            if job.training_accounting is not None:
                entry["training_accounting"] = dict(job.training_accounting)
            if job.checkpoint_policy is not None:
                entry["checkpoint_policy"] = dict(job.checkpoint_policy)
            if job.extend_from is not None:
                entry["extend_from"] = dict(job.extend_from)
            if entry.get("training_status") == "running":
                entry["training_status"] = "pending"
                entry["interrupted"] = True
            if entry.get("evaluation_status") == "running":
                entry["evaluation_status"] = "pending"
                entry["evaluation_interrupted"] = True
        atomic_write_json(self.state_path, state)
        return state

    # ------------------------------------------------------------------
    # Private-Hub connection shared by extension, evidence and publication
    # ------------------------------------------------------------------

    def _backup_config(self) -> dict[str, Any]:
        backup = self.config.get("checkpoint_backup") or {}
        if not backup.get("enabled"):
            raise QueueConfigurationError(
                "this operation requires an enabled checkpoint_backup destination"
            )
        repo_id = backup.get("repo_id") or os.environ.get(
            str(backup.get("repo_id_env", "HF_REPO_ID")), ""
        )
        token = os.environ.get(str(backup.get("token_env", "HF_TOKEN")), "")
        if not repo_id or not token:
            raise QueueConfigurationError(
                "a private repository id and its environment credential are required"
            )
        return {
            "repo_id": str(repo_id),
            "repo_type": str(backup.get("repo_type", "model")),
            "remote_prefix": str(backup.get("remote_prefix", "training-backups")).strip("/"),
            "token": str(token),
            "max_attempts": int(backup.get("max_attempts", 4)),
            "backoff_seconds": float(backup.get("backoff_seconds", 2)),
        }

    def _checkpoint_backup(self, job: QueueJob):
        from .backup import CheckpointBackup

        connection = self._backup_config()
        backup = self.config.get("checkpoint_backup") or {}
        state_path = Path(
            backup.get("state_path")
            or job.output_root / job.job_id / ".automation" / "backup-state.json"
        )
        return CheckpointBackup(
            repo_id=connection["repo_id"],
            repo_type=connection["repo_type"],
            state_path=state_path,
            remote_prefix=connection["remote_prefix"],
            token_env=str(backup.get("token_env", "HF_TOKEN")),
            max_attempts=connection["max_attempts"],
            backoff_seconds=connection["backoff_seconds"],
        )

    # ------------------------------------------------------------------
    # Extension of an existing training
    # ------------------------------------------------------------------

    def prepare_extension(self, job: QueueJob) -> dict[str, Any]:
        """Continue an existing model instead of retraining it from zero.

        Only the duration may change, unless the continuation declares a
        refinement phase and names the training-shape fields it changes. Dataset
        bytes, catalog identity and every other training-relevant recipe field
        are verified against the immutable evidence of the base checkpoint; an
        incompatible base is refused.
        """
        from .backup import HuggingFaceBackupClient
        from .catalog import CatalogStore, resolve_model, restore_training, safe_relative_path

        declaration = _normalized_extension(job.extend_from)
        connection = self._backup_config()
        process = _load_one_yaml(job.config_path)["config"]["process"][0]
        requested_steps = int(process["train"]["steps"])
        if requested_steps <= declaration["base_training_steps"]:
            raise QueueConfigurationError(
                f"extension of {job.job_id} must run past its base checkpoint "
                f"({requested_steps} <= {declaration['base_training_steps']})"
            )
        if declaration.get("phase") is not None:
            self._assert_phase_evidence_reachable(job, process, declaration)
        if job.dataset_folder is None:
            raise QueueConfigurationError(f"extension of {job.job_id} has no resolved dataset folder")
        actual_fingerprint = dataset_content_fingerprint(job.dataset_folder)
        if actual_fingerprint != declaration["dataset_fingerprint"]:
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: the dataset content hash is "
                f"{actual_fingerprint}, the declared base used {declaration['dataset_fingerprint']}"
            )
        client = HuggingFaceBackupClient(connection["token"])
        if not client.repo_is_private(connection["repo_id"], connection["repo_type"]):
            raise QueueConfigurationError("refusing extension against a non-private repository")
        store = CatalogStore(
            client=client,
            repo_id=connection["repo_id"],
            repo_type=connection["repo_type"],
            catalog_path=f"{connection['remote_prefix']}/catalog.json",
            work_dir=self.state_path.parent / "extensions" / job.job_id / ".catalog",
        )
        catalog, _ = store.read()
        model = resolve_model(catalog, declaration["model"])
        catalog_config = (process.get("checkpoint_backup") or {}).get("catalog") or {}
        expected_id = catalog_config.get("expected_id")
        if expected_id is not None and int(model["id"]) != int(expected_id):
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: catalog model {model['name']!r} has id "
                f"{model['id']}, this job is pre-reserved for id {expected_id}"
            )
        if str(model.get("trigger_word") or "") != str(process.get("trigger_word") or ""):
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: base trigger word "
                f"{model.get('trigger_word')!r} differs from {process.get('trigger_word')!r}"
            )
        recipe_arch = (process.get("model") or {}).get("arch")
        if model.get("base_arch") != recipe_arch:
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: base architecture {model.get('base_arch')!r} "
                f"differs from {recipe_arch!r}"
            )
        checkpoint = self._extension_checkpoint(model, declaration)
        base_process = self._base_process(client, connection, checkpoint, job)
        phase = declaration.get("phase")
        declared_fields = tuple(phase["changes"]) if phase is not None else ()
        differences = recipe_differences(base_process, process, declared_fields=declared_fields)
        if differences:
            shown = differences[:EXTENSION_REPORTED_DIFFERENCES]
            more = len(differences) - len(shown)
            allowance = (
                "only the duration may change"
                if phase is None
                else f"only the duration and the declared {phase['name']} phase fields "
                f"{list(declared_fields)} may change"
            )
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: the base checkpoint was trained with a "
                f"different recipe; {allowance}, while sampling prompts, save "
                "and sample cadence and local paths are already exempt. Differences: "
                + "; ".join(shown)
                + (f"; and {more} more" if more else "")
            )
        deviations = (
            phase_deviations(base_process, process, declared_fields) if phase is not None else []
        )
        staging = self.state_path.parent / "extensions" / job.job_id / "base"
        staging.mkdir(parents=True, exist_ok=True)
        restore_training(
            client=client,
            repo_id=connection["repo_id"],
            repo_type=connection["repo_type"],
            catalog=catalog,
            identifier=int(model["id"]),
            target_root=staging,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
        )
        base_job_id = str(checkpoint["job_id"])
        weight_remote = {str(item["remote_path"]) for item in checkpoint.get("weights", [])}
        target_root = job.output_root / job.job_id
        target_root.mkdir(parents=True, exist_ok=True)
        installed: list[dict[str, str]] = []
        resumable_weights = 0
        optimizer_installed = False
        for artifact in checkpoint["resume_artifacts"]:
            relative_value = artifact.get("training_relative_path")
            if relative_value is None:
                relative_value = Path(str(artifact.get("relative_path", ""))).name
            relative = safe_relative_path(str(relative_value))
            parts = list(relative.parts)
            if parts[0] == "config.yaml":
                # The extended run writes its own configuration; the base copy is
                # kept in staging as compatibility evidence only.
                continue
            is_weight = str(artifact["remote_path"]) in weight_remote
            if parts[0].startswith(base_job_id):
                parts[0] = job.job_id + parts[0][len(base_job_id):]
            elif is_weight:
                raise QueueConfigurationError(
                    f"refusing extension of {job.job_id}: base weight {relative.as_posix()!r} does "
                    f"not carry the base job name {base_job_id!r}, so the trainer could not resume it"
                )
            source = staging / relative
            target = target_root.joinpath(*parts)
            status = _install_local_copy(source, target)
            installed.append({
                "source_relative_path": relative.as_posix(),
                "path": str(target),
                "status": status,
                "role": "weights" if is_weight else "resume",
            })
            if is_weight and target.name.startswith(job.job_id):
                resumable_weights += 1
            if parts[-1] == "optimizer.pt":
                optimizer_installed = True
        if resumable_weights == 0:
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: no base weight was installed under this "
                "job name, so training would silently restart from zero"
            )
        if not optimizer_installed:
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: the base checkpoint has no optimizer.pt"
            )
        record = {
            "status": "prepared",
            "declaration": declaration,
            "base_model": {
                "id": int(model["id"]),
                "name": model["name"],
                "folder": model["folder"],
            },
            "base_checkpoint_id": str(checkpoint["checkpoint_id"]),
            "base_job_id": base_job_id,
            "base_step": int(checkpoint["step"]),
            "base_revision": str(checkpoint["revision"]),
            "dataset_fingerprint": actual_fingerprint,
            "resolved_training_steps": requested_steps,
            "installed": installed,
            "recipe_comparison": (
                "training-relevant fields identical to the base configuration"
                if phase is None
                else "training-relevant fields identical to the base configuration except the "
                f"fields declared by the {phase['name']} refinement phase"
            ),
        }
        if phase is not None:
            record["refinement_phase"] = {**phase, "deviations": deviations}
        atomic_write_json(target_root / ".automation" / "extension.json", record)
        return record

    @staticmethod
    def _assert_phase_evidence_reachable(
        job: QueueJob, process: Mapping[str, Any], declaration: Mapping[str, Any]
    ) -> None:
        """Refuse a refinement whose completion evidence could never be produced.

        The archive gate requires the configured sample set at every scheduled
        step past the base checkpoint and at the final step. That verdict only
        arrives when the whole paid run is already over, so a refinement that
        changes the training shape has its schedule checked here, before the
        first GPU hour, instead of failing after all of them.
        """
        sample = process.get("sample") or {}
        samples = sample.get("samples")
        if not isinstance(samples, list) or not samples:
            raise QueueConfigurationError(
                f"refusing refinement of {job.job_id}: the recipe configures no evaluation sample, "
                "so this run could never produce its completion evidence"
            )
        # The archive gate accepts only these cadences, so the pre-flight uses
        # its constant instead of a looser copy of it. bootstrap imports this
        # module, so the import stays local to this call.
        from .bootstrap import SUPPORTED_SAMPLE_CADENCES

        sample_every = sample.get("sample_every")
        save_every = (process.get("save") or {}).get("save_every")
        if (
            isinstance(sample_every, bool)
            or not isinstance(sample_every, int)
            or sample_every not in SUPPORTED_SAMPLE_CADENCES
            or save_every != sample_every
        ):
            raise QueueConfigurationError(
                f"refusing refinement of {job.job_id}: sample.sample_every and save.save_every "
                f"must be the same cadence, one of {sorted(SUPPORTED_SAMPLE_CADENCES)} "
                f"({sample_every!r} and {save_every!r})"
            )
        requested_steps = int(process["train"]["steps"])
        if requested_steps % sample_every:
            raise QueueConfigurationError(
                f"refusing refinement of {job.job_id}: the final step {requested_steps} is not on "
                f"the {sample_every}-step sampling cadence, so the final checkpoint would have no "
                "scheduled evaluation and the run could never be archived"
            )
        base_steps = int(declaration["base_training_steps"])
        if base_steps >= requested_steps:
            raise QueueConfigurationError(
                f"refusing refinement of {job.job_id}: base step {base_steps} is not below the "
                f"final step {requested_steps}"
            )

    @staticmethod
    def _extension_checkpoint(
        model: Mapping[str, Any], declaration: Mapping[str, Any]
    ) -> dict[str, Any]:
        resumable = [
            item for item in model.get("checkpoints", [])
            if item.get("resume_artifacts") and not item.get("imported_existing")
        ]
        if not resumable:
            raise QueueConfigurationError(
                f"catalog model {model['name']!r} has no checkpoint with trainer resume state"
            )
        requested = declaration.get("checkpoint_id")
        if requested:
            checkpoint = next(
                (item for item in resumable if str(item["checkpoint_id"]) == requested), None
            )
            if checkpoint is None:
                raise QueueConfigurationError(
                    f"checkpoint {requested!r} is absent from catalog model {model['name']!r} "
                    "or has no trainer resume state"
                )
        else:
            checkpoint = max(resumable, key=lambda item: (int(item["step"]), str(item["checkpoint_id"])))
        if int(checkpoint["step"]) != int(declaration["base_training_steps"]):
            raise QueueConfigurationError(
                f"refusing extension: base checkpoint {checkpoint['checkpoint_id']!r} is at step "
                f"{checkpoint['step']}, the declaration states {declaration['base_training_steps']}"
            )
        return dict(checkpoint)

    @staticmethod
    def _base_process(
        client: Any, connection: Mapping[str, Any], checkpoint: Mapping[str, Any], job: QueueJob
    ) -> dict[str, Any]:
        import tempfile

        artifact = next(
            (
                item for item in checkpoint["resume_artifacts"]
                if str(item.get("training_relative_path") or "") == "config.yaml"
            ),
            None,
        )
        if artifact is None:
            raise QueueConfigurationError(
                f"refusing extension of {job.job_id}: base checkpoint "
                f"{checkpoint['checkpoint_id']!r} has no saved trainer configuration, so its "
                "recipe cannot be proven compatible"
            )
        from .backup import sha256_file

        handle, name = tempfile.mkstemp(prefix="extension-base-config-")
        os.close(handle)
        temporary = Path(name)
        try:
            client.download_file(
                connection["repo_id"], connection["repo_type"],
                str(artifact["remote_path"]), str(checkpoint["revision"]), temporary,
            )
            if (
                temporary.stat().st_size != int(artifact["size"])
                or sha256_file(temporary) != str(artifact["sha256"])
            ):
                raise QueueConfigurationError(
                    "base trainer configuration failed size or hash verification"
                )
            document = yaml.load(
                temporary.read_text(encoding="utf-8"), Loader=TrainerConfigLoader
            )
        finally:
            temporary.unlink(missing_ok=True)
        try:
            return dict(document["config"]["process"][0])
        except (KeyError, IndexError, TypeError) as exc:
            raise QueueConfigurationError(
                "base trainer configuration has no config.process[0]"
            ) from exc

    # ------------------------------------------------------------------
    # Per-checkpoint evidence and per-job publication
    # ------------------------------------------------------------------

    def _evidence_enabled(self) -> bool:
        backup = self.config.get("checkpoint_backup") or {}
        evidence = self.config.get("checkpoint_evidence") or {}
        evaluation = self.config.get("evaluation") or {}
        return bool(
            backup.get("enabled")
            and evidence.get("enabled", True)
            and evaluation.get("enabled", True)
        )

    def flush_checkpoint_evidence(self, job: QueueJob) -> list[dict[str, Any]]:
        """Publish the evidence of every checkpoint of a finished training run."""
        backup = self._checkpoint_backup(job)
        return backup.publish_pending_evidence(job_config_path=job.config_path, strict=True)

    def publish_job_results(self, job: QueueJob, completed_at: str) -> dict[str, Any]:
        """Publish the ranked grids and top three of this single job."""
        from .backup import HuggingFaceBackupClient
        from .results import publish_ranked_results

        results = self.config.get("results") or {}
        connection = self._backup_config()
        output = job.output_root / job.job_id
        work_dir = Path(
            results.get("work_dir") or self.state_path.parent / "results"
        ) / job.job_id
        record, evidence_files = publish_ranked_results(
            client=HuggingFaceBackupClient(connection["token"]),
            repo_id=connection["repo_id"],
            repo_type=connection["repo_type"],
            run_id=str(results["run_id"]),
            job_id=job.job_id,
            report_path=output / ".automation" / "evaluation.json",
            sample_root=output / "samples",
            work_dir=work_dir,
            catalog_prefix=str(results.get("catalog_prefix", connection["remote_prefix"])),
            results_prefix=str(results.get("results_prefix", "training-results")),
            completed_at=completed_at,
        )
        # The shard archive reuses this exact publication instead of redoing it.
        atomic_write_json(
            work_dir / "published-evidence.json",
            {
                "schema_version": 1,
                "job_id": job.job_id,
                "completed_at": completed_at,
                "record": record,
                "files": [[str(local), name] for local, name in evidence_files],
            },
        )
        return record

    @contextmanager
    def _lock(self):
        import fcntl

        lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"training queue is already running: {self.state_path}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        jobs = self.materialize()
        if dry_run:
            return {
                "dry_run": True,
                "jobs": [job.job_id for job in jobs],
                "state_path": str(self.state_path),
            }
        with self._lock():
            state = self._state(jobs)
            continue_on_error = bool(self.config.get("continue_on_error", False))
            for job in jobs:
                entry = state["jobs"][job.job_id]
                if entry.get("status") == "completed":
                    continue
                if entry.get("training_status") != "completed":
                    if job.extend_from is not None and entry.get("extension_status") != "completed":
                        entry.update({"status": "extending", "extension_status": "running"})
                        atomic_write_json(self.state_path, state)
                        try:
                            extension = self.prepare_extension(job)
                        except Exception as exc:
                            entry.update({
                                "status": "failed",
                                "extension_status": "failed",
                                "extension_error": f"{type(exc).__name__}: {exc}",
                            })
                            atomic_write_json(self.state_path, state)
                            if not continue_on_error:
                                break
                            continue
                        entry.update({"extension_status": "completed", "extension": extension})
                        entry.pop("extension_error", None)
                        atomic_write_json(self.state_path, state)
                    entry.update({
                        "status": "training",
                        "training_status": "running",
                        "attempts": int(entry.get("attempts", 0)) + 1,
                    })
                    atomic_write_json(self.state_path, state)
                    env = dict(os.environ)
                    env["TRAINING_AUTOMATION_JOB_CONFIG"] = str(job.config_path)
                    code = self._run_command(
                        [self.python, str(self.repo_root / "run.py"), str(job.config_path)], env
                    )
                    if code != 0:
                        entry.update({"status": "failed", "training_status": "failed", "exit_code": int(code)})
                        atomic_write_json(self.state_path, state)
                        if not continue_on_error:
                            break
                        continue
                    entry.update({"status": "training_completed", "training_status": "completed", "exit_code": 0})
                    atomic_write_json(self.state_path, state)

                if self._evidence_enabled() and entry.get("evidence_status") != "completed":
                    entry.update({"status": "publishing-evidence", "evidence_status": "running"})
                    atomic_write_json(self.state_path, state)
                    try:
                        published = self.flush_checkpoint_evidence(job)
                    except Exception as exc:
                        entry.update({
                            "status": "evidence_failed",
                            "evidence_status": "failed",
                            "evidence_error": f"{type(exc).__name__}: {exc}",
                        })
                        atomic_write_json(self.state_path, state)
                        if not continue_on_error:
                            break
                        continue
                    entry.update({
                        "status": "training_completed",
                        "evidence_status": "completed",
                        "evidence_published_now": len(published),
                    })
                    entry.pop("evidence_error", None)
                    atomic_write_json(self.state_path, state)

                evaluation = self.config.get("evaluation", {}) or {}
                if not evaluation.get("enabled", True):
                    entry.update({"status": "completed", "evaluation_status": "skipped", "evaluation_report": None})
                    atomic_write_json(self.state_path, state)
                    continue
                if entry.get("evaluation_status") == "completed":
                    if not self._publish_completed_job(job, entry, state):
                        if not continue_on_error:
                            break
                        continue
                    entry["status"] = "completed"
                    atomic_write_json(self.state_path, state)
                    continue
                entry.update({"status": "evaluating", "evaluation_status": "running"})
                atomic_write_json(self.state_path, state)
                try:
                    with preserve_cuda_visibility():
                        report_path = evaluate_job(
                            job_config_path=job.config_path,
                            output_dir=job.output_root / job.job_id,
                            reference_images=list(job.reference_images),
                            config=evaluation,
                        )
                except Exception as exc:
                    entry.update({
                        "status": "evaluation_failed",
                        "evaluation_status": "failed",
                        "evaluation_error": f"{type(exc).__name__}: {exc}",
                    })
                    atomic_write_json(self.state_path, state)
                    if not continue_on_error:
                        break
                    continue
                entry.update({
                    "status": "evaluated",
                    "evaluation_status": "completed",
                    "evaluation_report": str(report_path),
                })
                entry.pop("evaluation_error", None)
                atomic_write_json(self.state_path, state)
                if not self._publish_completed_job(job, entry, state):
                    if not continue_on_error:
                        break
                    continue
                entry["status"] = "completed"
                atomic_write_json(self.state_path, state)
            return state

    def _publish_completed_job(
        self, job: QueueJob, entry: dict[str, Any], state: Mapping[str, Any]
    ) -> bool:
        """Publish this job's ranked evidence as soon as the job itself is done.

        Interrupting the queue after this point no longer costs the publication of
        the jobs that already finished.
        """
        results = self.config.get("results") or {}
        if not results.get("enabled") or entry.get("results_status") == "completed":
            return True
        completed_at = str(
            entry.get("job_completed_at") or datetime.now(timezone.utc).isoformat()
        )
        entry.update({
            "status": "publishing-results",
            "results_status": "running",
            "job_completed_at": completed_at,
        })
        atomic_write_json(self.state_path, state)
        try:
            record = self.publish_job_results(job, completed_at)
        except Exception as exc:
            entry.update({
                "status": "publish_failed",
                "results_status": "failed",
                "results_error": f"{type(exc).__name__}: {exc}",
            })
            atomic_write_json(self.state_path, state)
            return False
        entry.update({
            "results_status": "completed",
            "automatic_result_export": record,
        })
        entry.pop("results_error", None)
        atomic_write_json(self.state_path, state)
        return True
