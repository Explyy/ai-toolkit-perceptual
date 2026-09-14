from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .evaluation import evaluate_job
from .state import atomic_write_json, read_json


QUEUE_CONFIG_SCHEMA = 1
QUEUE_STATE_SCHEMA = 2


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


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    config_path: Path
    output_root: Path
    reference_images: tuple[Path, ...]
    training_steps: int | None = None
    training_accounting: Mapping[str, Any] | None = None
    checkpoint_policy: Mapping[str, int] | None = None


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
                process["checkpoint_backup"] = backup
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
            if entry.get("training_status") == "running":
                entry["training_status"] = "pending"
                entry["interrupted"] = True
            if entry.get("evaluation_status") == "running":
                entry["evaluation_status"] = "pending"
                entry["evaluation_interrupted"] = True
        atomic_write_json(self.state_path, state)
        return state

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

                evaluation = self.config.get("evaluation", {}) or {}
                if not evaluation.get("enabled", True):
                    entry.update({"status": "completed", "evaluation_status": "skipped", "evaluation_report": None})
                    atomic_write_json(self.state_path, state)
                    continue
                if entry.get("evaluation_status") == "completed":
                    entry["status"] = "completed"
                    atomic_write_json(self.state_path, state)
                    continue
                entry.update({"status": "evaluating", "evaluation_status": "running"})
                atomic_write_json(self.state_path, state)
                try:
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
                    "status": "completed",
                    "evaluation_status": "completed",
                    "evaluation_report": str(report_path),
                })
                entry.pop("evaluation_error", None)
                atomic_write_json(self.state_path, state)
            return state
