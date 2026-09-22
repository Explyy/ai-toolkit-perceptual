from __future__ import annotations

import importlib
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import yaml
from PIL import Image

from .state import atomic_write_json


REPORT_SCHEMA = 1
CHECKPOINT_EVIDENCE_SCHEMA = 1
CHECKPOINT_SCORE_METHOD = (
    "mean face cosine similarity over this checkpoint's own images with exactly one "
    "valid face, plus mean clipping fraction proxy; single-checkpoint evidence, not a "
    "cross-checkpoint ranking"
)
SAMPLE_RE = re.compile(r"^(?P<time>\d+)__(?P<step>\d+)_(?P<index>\d+)\.(?:png|jpe?g|webp)$", re.I)


class FaceBackend(Protocol):
    def embeddings(self, image: np.ndarray) -> list[np.ndarray]: ...


@contextmanager
def preserve_cuda_visibility():
    """Restore the caller's exact GPU visibility after CPU-only evaluation code."""
    key = "CUDA_VISIBLE_DEVICES"
    was_present = key in os.environ
    previous = os.environ.get(key)
    try:
        yield
    finally:
        if was_present:
            os.environ[key] = previous if previous is not None else ""
        else:
            os.environ.pop(key, None)


def _backend_provenance(backend: Any | None, spec: str | None) -> dict[str, Any]:
    if backend is None:
        return {"status": "unavailable", "backend": spec, "reason": "backend not configured"}
    details = backend.provenance() if hasattr(backend, "provenance") else {}
    return {"status": "available", "backend": spec, **details}


def preflight_evaluation_backends(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load configured backends before a paid training process starts."""
    with preserve_cuda_visibility():
        face = _load_backend(config.get("face_backend"), config.get("face_backend_options", {}))
        landmark = _load_backend(config.get("landmark_backend"), config.get("landmark_backend_options", {}))
        return {
            "face_backend": _backend_provenance(face, config.get("face_backend")),
            "pose_backend": _backend_provenance(landmark, config.get("landmark_backend")),
        }


def _load_backend(spec: str | None, options: Mapping[str, Any]) -> Any | None:
    if not spec:
        return None
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("backend must use module:factory syntax")
    return getattr(importlib.import_module(module_name), factory_name)(**dict(options))


def _image_metrics(image: np.ndarray) -> dict[str, Any]:
    rgb = image.astype(np.float32) / 255.0
    clipping = float(np.mean((rgb <= 1 / 255) | (rgb >= 254 / 255)))
    gray = np.mean(rgb, axis=2)
    laplace = -4 * gray
    laplace[1:, :] += gray[:-1, :]
    laplace[:-1, :] += gray[1:, :]
    laplace[:, 1:] += gray[:, :-1]
    laplace[:, :-1] += gray[:, 1:]
    return {
        "clipping_fraction_proxy": clipping,
        "clipping_metric_note": "fraction of near-black or near-white channels; proxy only",
        "sharpness_laplacian_variance_proxy": float(np.var(laplace)),
        "sharpness_metric_note": "Laplacian variance; proxy affected by grain, scale, and compression",
    }


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return None if not math.isfinite(norm) or norm <= 0 else vector / norm


def _reference_identity(
    reference_images: list[Path], backend: FaceBackend | None,
    filter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if backend is None:
        return {"status": "unavailable", "embedding": None, "reason": "face backend not configured"}
    embeddings = []
    excluded: list[dict[str, str]] = []
    for path in reference_images:
        try:
            faces = backend.embeddings(np.asarray(Image.open(path).convert("RGB")))
        except Exception as exc:
            if filter_config is None:
                return {"status": "unavailable", "embedding": None, "reason": f"invalid reference {path.name}: {type(exc).__name__}"}
            excluded.append({"file": path.name, "reason": "invalid"})
            continue
        if len(faces) == 0:
            if filter_config is not None:
                excluded.append({"file": path.name, "reason": "missing"})
                continue
            return {"status": "missing", "embedding": None, "reason": f"no face in reference {path.name}"}
        if len(faces) != 1:
            if filter_config is not None:
                excluded.append({"file": path.name, "reason": "multiple"})
                continue
            return {"status": "ambiguous", "embedding": None, "reason": f"multiple faces in reference {path.name}"}
        vector = _unit(faces[0])
        if vector is None:
            if filter_config is not None:
                excluded.append({"file": path.name, "reason": "invalid"})
                continue
            return {"status": "unavailable", "embedding": None, "reason": "invalid reference embedding"}
        embeddings.append(vector)
    if not embeddings and filter_config is None:
        return {"status": "missing", "embedding": None, "reason": "no reference images supplied"}
    counts = {
        reason: sum(item["reason"] == reason for item in excluded)
        for reason in ("missing", "multiple", "invalid")
    }
    coverage = len(embeddings) / len(reference_images) if reference_images else 0.0
    filter_evidence = None
    if filter_config is not None:
        filter_evidence = {
            "mode": "exactly-one-valid-face",
            "total": len(reference_images), "valid": len(embeddings),
            "coverage_fraction": coverage, "excluded_counts": counts,
            "excluded": excluded,
            "minimum_valid_count": int(filter_config["minimum_valid_count"]),
            "minimum_valid_fraction": float(filter_config["minimum_valid_fraction"]),
        }
        if (
            len(embeddings) < int(filter_config["minimum_valid_count"])
            or coverage < float(filter_config["minimum_valid_fraction"])
        ):
            return {
                "status": "unavailable", "embedding": None,
                "reason": "single-face reference coverage is below the configured minimum",
                "filter": filter_evidence,
            }
    average = _unit(np.mean(embeddings, axis=0))
    if average is None:
        return {"status": "unavailable", "embedding": None, "reason": "invalid averaged reference embedding"}
    return {
        "status": "available", "embedding": average, "reason": None,
        **({"filter": filter_evidence} if filter_evidence is not None else {}),
    }


def _identity_metric(image: np.ndarray, backend: FaceBackend | None, reference: Mapping[str, Any]) -> dict[str, Any]:
    if reference["status"] != "available":
        return {"status": reference["status"], "cosine_similarity": None, "reason": reference["reason"]}
    faces = backend.embeddings(image) if backend is not None else []
    if len(faces) == 0:
        return {"status": "missing", "cosine_similarity": None, "reason": "no face detected"}
    if len(faces) != 1:
        return {"status": "ambiguous", "cosine_similarity": None, "reason": "multiple faces detected"}
    vector = _unit(faces[0])
    if vector is None:
        return {"status": "unavailable", "cosine_similarity": None, "reason": "invalid face embedding"}
    return {"status": "available", "cosine_similarity": float(np.dot(vector, reference["embedding"])), "reason": None}


def _sample_expectations(
    job_config_path: Path,
) -> tuple[dict[int, dict[str, Any]], int, dict[str, Any]]:
    with job_config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    process = document["config"]["process"][0]
    sample = process.get("sample", {})
    raw = sample.get("samples")
    if raw is None:
        raw = [{"prompt": prompt} for prompt in sample.get("prompts", [])]
    base_seed = int(sample.get("seed", 42))
    walk = bool(sample.get("walk_seed", False))
    meta = document.get("meta") or {}
    cases = meta.get("evaluation_cases") or []
    if cases and len(cases) != len(raw):
        raise ValueError("evaluation case metadata does not match configured sample count")
    expected = {}
    for index, item in enumerate(raw):
        case = cases[index] if cases else {}
        expected[index] = {
            "prompt": item.get("prompt"),
            "seed": int(case.get("seed", item.get("seed", base_seed + index if walk else base_seed))),
            "case_id": case.get("case_id", f"legacy-{index:02d}"),
            "category": case.get("category", "legacy"),
        }
    return expected, int(process.get("train", {}).get("steps", 0)), {
        "id": str(meta.get("evaluation_cohort", "subject-likeness-legacy-v1")),
        "version": str(meta.get("version", "1.0")),
    }


def rank_checkpoints(checkpoints: list[dict[str, Any]], expected_signature: set[tuple[int, int]] | None = None) -> tuple[list[dict[str, Any]], str | None]:
    if not checkpoints:
        return [], "no checkpoint sample groups found"
    signatures = [
        {(sample["prompt_index"], sample["seed"]) for sample in checkpoint["samples"]}
        for checkpoint in checkpoints
    ]
    if not signatures[0] or any(signature != signatures[0] for signature in signatures[1:]):
        return [], "prompt/seed sets are not comparable across checkpoints"
    if expected_signature is not None and signatures[0] != expected_signature:
        return [], "checkpoint samples do not contain the complete configured prompt/seed set"
    ranked = []
    for checkpoint in checkpoints:
        clipping = [sample["metrics"]["clipping_fraction_proxy"] for sample in checkpoint["samples"]]
        score = -(sum(clipping) / len(clipping))
        ranked.append({"step": checkpoint["step"], "clipping_proxy_score": score})
    ranked.sort(key=lambda item: (-item["clipping_proxy_score"], item["step"]))
    return ranked, None


def rank_identity(checkpoints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    available_sets = [
        {
            sample["prompt_index"]
            for sample in checkpoint["samples"]
            if sample["identity"]["status"] == "available"
        }
        for checkpoint in checkpoints
    ]
    common = set.intersection(*available_sets) if available_sets else set()
    if len(common) < 3:
        return [], "fewer than three prompt indices have valid faces at every checkpoint"
    ranked = []
    for checkpoint in checkpoints:
        values = [
            sample["identity"]["cosine_similarity"]
            for sample in checkpoint["samples"]
            if sample["prompt_index"] in common and sample["identity"]["status"] == "available"
        ]
        ranked.append(
            {
                "step": checkpoint["step"],
                "mean_face_cosine_similarity": sum(values) / len(values),
                "prompt_indices": sorted(common),
            }
        )
    ranked.sort(key=lambda item: (-item["mean_face_cosine_similarity"], item["step"]))
    return ranked, None


def shortlist_checkpoints(
    checkpoints: list[dict[str, Any]], identity_ranked: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_step = {item["step"]: item for item in identity_ranked}
    items = []
    for checkpoint in checkpoints:
        identity = by_step.get(checkpoint["step"])
        if identity is None:
            continue
        clipping = [sample["metrics"]["clipping_fraction_proxy"] for sample in checkpoint["samples"]]
        items.append({
            **identity,
            "mean_clipping_fraction_proxy": sum(clipping) / len(clipping),
        })
    items.sort(key=lambda item: (
        -item["mean_face_cosine_similarity"],
        item["mean_clipping_fraction_proxy"], item["step"],
    ))
    return items[:3]


def shortlist_exclusion_reason(checkpoint: Mapping[str, Any]) -> str | None:
    """Return why this checkpoint cannot be a shortlist candidate, or None.

    A shortlisted checkpoint is exported as weights, so it must resolve to
    exactly one verified remote object. Ambiguous means two or more catalog
    checkpoints exist for this job and step and the sample filenames do not say
    which bytes produced the images; unavailable means this job published no
    verified receipt for that step at all. Either way the checkpoint cannot be
    shipped - but that is a property of that checkpoint alone, not of the run.
    """
    association = checkpoint.get("remote_association") or {}
    status = str(association.get("status") or "unavailable")
    if status == "unique":
        return None
    return (
        f"remote association is {status}: "
        f"{association.get('reason') or 'no reason recorded'}"
    )


def partition_shortlist_candidates(
    checkpoints: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the evaluated checkpoints into shippable candidates and exclusions."""
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        reason = shortlist_exclusion_reason(checkpoint)
        if reason is None:
            eligible.append(checkpoint)
        else:
            excluded.append({
                "step": int(checkpoint["step"]),
                "remote_association_status": str(
                    (checkpoint.get("remote_association") or {}).get("status")
                    or "unavailable"
                ),
                "reason": reason,
            })
    return eligible, excluded


def shortlist_unavailable_reason(
    checkpoints: list[dict[str, Any]],
    *, ranking_error: str | None, identity_ranking_error: str | None,
) -> str | None:
    if ranking_error:
        return f"configured prompt/seed evidence is not comparable: {ranking_error}"
    incomplete = [
        int(checkpoint["step"])
        for checkpoint in checkpoints
        if checkpoint.get("sample_run_status") != "complete"
    ]
    if incomplete:
        return f"checkpoint sample runs are incomplete at steps {incomplete}"
    # A checkpoint without a unique verified remote association is excluded from
    # the candidates, not treated as a verdict on the other checkpoints. Voiding
    # the whole shortlist for one such step is what made an extended run publish
    # no top three at all while every other checkpoint of the same run was
    # completely and uniquely associated.
    eligible, excluded = partition_shortlist_candidates(checkpoints)
    if not eligible:
        steps = [item["step"] for item in excluded]
        return (
            "no checkpoint has a unique verified remote association "
            f"(steps {steps})"
        )
    if identity_ranking_error:
        return identity_ranking_error
    return None


def shortlist_partial_coverage_note(excluded: list[dict[str, Any]]) -> str | None:
    """Say out loud which checkpoints could not compete for the top three."""
    if not excluded:
        return None
    return (
        "ranked only the checkpoints with a unique verified remote association; "
        "excluded steps "
        + ", ".join(f"{item['step']} ({item['reason']})" for item in excluded)
    )


def _choose_latest_complete_run(
    samples: list[dict[str, Any]], expected_indices: set[int]
) -> tuple[list[dict[str, Any]], str, int]:
    """Partition chronological files when an index repeats and choose the latest complete run."""
    runs: list[dict[int, dict[str, Any]]] = []
    current: dict[int, dict[str, Any]] = {}
    for sample in sorted(samples, key=lambda item: (item["timestamp_ms"], item["prompt_index"], item["path"])):
        if sample["prompt_index"] in current:
            runs.append(current)
            current = {}
        current[sample["prompt_index"]] = sample
    if current:
        runs.append(current)
    complete = [run for run in runs if set(run) == expected_indices]
    selected = complete[-1] if complete else (runs[-1] if runs else {})
    discarded = len(samples) - len(selected)
    status = "complete" if complete else "incomplete"
    return [selected[index] for index in sorted(selected)], status, discarded


def _evaluate_job(
    *,
    job_config_path: Path,
    output_dir: Path,
    reference_images: list[Path],
    config: Mapping[str, Any],
) -> Path:
    expected, final_step, evaluation_cohort = _sample_expectations(job_config_path)
    with job_config_path.open("r", encoding="utf-8") as handle:
        job_name = yaml.safe_load(handle)["config"]["name"]
    face = _load_backend(config.get("face_backend"), config.get("face_backend_options", {}))
    landmark = _load_backend(config.get("landmark_backend"), config.get("landmark_backend_options", {}))
    reference = _reference_identity(
        reference_images, face, config.get("reference_identity_filter")
    )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((output_dir / "samples").glob("*")):
        match = SAMPLE_RE.match(path.name)
        if not match:
            continue
        step, index = int(match.group("step")), int(match.group("index"))
        expectation = expected.get(index, {"prompt": None, "seed": None})
        image = np.asarray(Image.open(path).convert("RGB"))
        landmarks = (
            landmark.metrics(image)
            if landmark is not None
            else {"status": "unavailable", "values": None, "reason": "landmark backend not configured"}
        )
        grouped.setdefault(step, []).append(
            {
                "path": str(path),
                "timestamp_ms": int(match.group("time")),
                "prompt_index": index,
                "prompt": expectation["prompt"],
                "seed": expectation["seed"],
                "case_id": expectation.get("case_id"),
                "category": expectation.get("category"),
                "metrics": _image_metrics(image),
                "identity": _identity_metric(image, face, reference),
                "pose_body_landmarks": landmarks,
            }
        )
    checkpoint_steps: set[int] = set()
    checkpoint_paths: dict[int, list[str]] = {}
    checkpoint_remote: dict[int, list[dict[str, Any]]] = {}
    step_pattern = re.compile(r"_(\d{9})(?:_|\.|$)")
    for path in output_dir.iterdir() if output_dir.is_dir() else []:
        if path.name.startswith(".") or path.name in {"samples", "optimizer.pt", "config.yaml"}:
            continue
        match = step_pattern.search(path.name)
        if match:
            step = int(match.group(1))
        elif path.name.startswith(job_name) and (path.is_dir() or path.suffix in {".safetensors", ".pt"}):
            step = final_step
        else:
            continue
        checkpoint_steps.add(step)
        checkpoint_paths.setdefault(step, []).append(str(path))
    backup_state_path = Path(
        config.get("backup_state_path") or output_dir / ".automation" / "backup-state.json"
    )
    if backup_state_path.is_file():
        backup_state = json.loads(backup_state_path.read_text(encoding="utf-8"))
        for entry in backup_state.get("checkpoints", {}).values():
            if (
                entry.get("status") != "backed_up"
                or not entry.get("verified")
                or not entry.get("cataloged")
                or entry.get("job_id") != job_name
            ):
                continue
            step = int(entry["step"])
            checkpoint_steps.add(step)
            checkpoint_remote.setdefault(step, []).append({
                "catalog_checkpoint_id": entry.get("catalog_checkpoint_id"),
                "commit_id": entry.get("commit_id"),
                "cataloged": True,
                "destination": backup_state.get("destination"),
                "weight_sha256": sorted(
                    item["sha256"]
                    for item in entry.get("artifacts", [])
                    if item.get("role") == "weights" and item.get("sha256")
                ),
            })
    expected_indices = set(expected)
    checkpoints = []
    for step in sorted(checkpoint_steps & grouped.keys()):
        selected_samples, sample_run_status, discarded_samples = _choose_latest_complete_run(
            grouped[step], expected_indices
        )
        remote_candidates = sorted(
            checkpoint_remote.get(step, []),
            key=lambda item: (str(item.get("catalog_checkpoint_id")), str(item.get("commit_id"))),
        )
        if len(remote_candidates) == 1:
            remote_association = {"status": "unique", "reason": None}
            remote_checkpoint = remote_candidates[0]
        elif len(remote_candidates) > 1:
            remote_association = {
                "status": "ambiguous",
                "reason": "multiple verified catalog checkpoints exist for this job and step; sample filenames do not identify checkpoint bytes",
            }
            remote_checkpoint = None
        else:
            remote_association = {"status": "unavailable", "reason": "no verified catalog checkpoint receipt for this job and step"}
            remote_checkpoint = None
        checkpoints.append({
            "step": step,
            "final": step == final_step,
            "checkpoint_paths": sorted(checkpoint_paths.get(step, [])),
            "catalog_checkpoint_id": remote_checkpoint.get("catalog_checkpoint_id") if remote_checkpoint else None,
            "remote_checkpoint": remote_checkpoint,
            "remote_checkpoint_candidates": remote_candidates,
            "remote_association": remote_association,
            "samples": selected_samples,
            "sample_run_status": sample_run_status,
            "discarded_duplicate_or_partial_samples": discarded_samples,
        })
    expected_signature = {(index, item["seed"]) for index, item in expected.items()}
    ranked, ranking_error = rank_checkpoints(checkpoints, expected_signature)
    identity_ranked, identity_ranking_error = rank_identity(checkpoints)
    shortlist_error = shortlist_unavailable_reason(
        checkpoints, ranking_error=ranking_error,
        identity_ranking_error=identity_ranking_error,
    )
    eligible, excluded_candidates = partition_shortlist_candidates(checkpoints)
    shortlist = shortlist_checkpoints(eligible, identity_ranked) if not shortlist_error else []
    report = {
        "schema_version": REPORT_SCHEMA,
        "job_config": str(job_config_path),
        "evaluation_cohort": evaluation_cohort,
        "final_step": final_step,
        "reference_identity_status": {key: value for key, value in reference.items() if key != "embedding"},
        "reference_provenance": str(config.get("reference_provenance", "unspecified")),
        "face_backend": _backend_provenance(face, config.get("face_backend")),
        "pose_backend": _backend_provenance(landmark, config.get("landmark_backend")),
        "checkpoints": checkpoints,
        "ranking": {
            "status": "available" if not ranking_error else "unavailable",
            "reason": ranking_error,
            "method": "negative mean clipping fraction proxy; heuristic only, not ground truth about overtraining",
            "items": ranked,
        },
        "identity_ranking": {
            "status": "available" if not identity_ranking_error else "unavailable",
            "reason": identity_ranking_error,
            "method": "mean face cosine similarity on the intersection of prompt indices valid at every checkpoint; heuristic only",
            "items": identity_ranked,
        },
        "automatic_shortlist": {
            "status": "available" if shortlist else "unavailable",
            "reason": shortlist_error or shortlist_partial_coverage_note(excluded_candidates),
            "excluded": excluded_candidates,
            "method": "top three by mean face cosine descending on the common valid-face prompt subset, then mean clipping ascending, then step ascending; evidence shortlist only",
            "items": shortlist,
        },
        "pose_body_limit": "optional backend output is viewpoint- and visibility-dependent and is not a 3D body measurement",
    }
    report_path = output_dir / ".automation" / "evaluation.json"
    atomic_write_json(report_path, report)
    return report_path


class CheckpointEvidenceEvaluator:
    """Score one checkpoint's own samples while the trainer is still running.

    The job report produced by :func:`evaluate_job` compares every checkpoint of a
    finished job. This evaluator answers a narrower question with the same fixed
    models and the same sample association rules: what is the evidence for one
    checkpoint, right now, so that the checkpoint can be uploaded as a
    self-sufficient unit instead of waiting for the end of the shard.
    """

    def __init__(
        self,
        *,
        job_config_path: Path,
        output_dir: Path,
        reference_images: list[Path],
        config: Mapping[str, Any],
    ):
        self.job_config_path = Path(job_config_path)
        self.output_dir = Path(output_dir)
        self.config = dict(config or {})
        self.reference_images = [Path(item) for item in reference_images]
        self._expected, self._final_step, self._cohort = _sample_expectations(self.job_config_path)
        self._loaded = False
        self._face: Any | None = None
        self._landmark: Any | None = None
        self._reference: dict[str, Any] = {}

    @property
    def final_step(self) -> int:
        return self._final_step

    @property
    def expected_sample_count(self) -> int:
        return len(self._expected)

    def _ensure_backends(self) -> None:
        if self._loaded:
            return
        self._face = _load_backend(
            self.config.get("face_backend"), self.config.get("face_backend_options", {})
        )
        self._landmark = _load_backend(
            self.config.get("landmark_backend"), self.config.get("landmark_backend_options", {})
        )
        self._reference = _reference_identity(
            self.reference_images, self._face, self.config.get("reference_identity_filter")
        )
        self._loaded = True

    def _files_for_step(self, step: int) -> list[dict[str, Any]]:
        samples_dir = self.output_dir / "samples"
        if not samples_dir.is_dir():
            return []
        found = []
        for path in sorted(samples_dir.glob("*")):
            match = SAMPLE_RE.match(path.name)
            if not match or int(match.group("step")) != int(step):
                continue
            found.append({
                "path": path,
                "timestamp_ms": int(match.group("time")),
                "prompt_index": int(match.group("index")),
            })
        return found

    def evaluate(self, *, step: int, final: bool) -> dict[str, Any]:
        """Return the evidence record for one checkpoint step."""
        with preserve_cuda_visibility():
            return self._evaluate(step=int(step), final=bool(final))

    def _evaluate(self, *, step: int, final: bool) -> dict[str, Any]:
        expected_indices = set(self._expected)
        raw = self._files_for_step(step)
        record: dict[str, Any] = {
            "schema_version": CHECKPOINT_EVIDENCE_SCHEMA,
            "job_config": str(self.job_config_path),
            "evaluation_cohort": self._cohort,
            "step": int(step),
            "final": bool(final),
            "final_step": self._final_step,
            "expected_sample_count": len(expected_indices),
            "reference_provenance": str(self.config.get("reference_provenance", "unspecified")),
            "metric_limits": (
                "raw face cosine similarity is not a percentage or probability; clipping, "
                "sharpness and image-plane pose values are evidence proxies, not quality or "
                "anatomical ground truth"
            ),
        }
        if not raw:
            record.update({
                "sample_run_status": "missing",
                "discarded_duplicate_or_partial_samples": 0,
                "samples": [],
                "reference_identity_status": {"status": "unavailable", "reason": "not evaluated"},
                "face_backend": _backend_provenance(None, self.config.get("face_backend")),
                "pose_backend": _backend_provenance(None, self.config.get("landmark_backend")),
                "score": {
                    "status": "unavailable",
                    "reason": "no sample image exists for this step yet",
                    "mean_face_cosine_similarity": None,
                    "face_sample_count": 0,
                    "mean_clipping_fraction_proxy": None,
                    "method": CHECKPOINT_SCORE_METHOD,
                },
            })
            return record
        self._ensure_backends()
        scored = []
        for item in raw:
            path = item["path"]
            index = item["prompt_index"]
            expectation = self._expected.get(index, {"prompt": None, "seed": None})
            image = np.asarray(Image.open(path).convert("RGB"))
            landmarks = (
                self._landmark.metrics(image)
                if self._landmark is not None
                else {"status": "unavailable", "values": None, "reason": "landmark backend not configured"}
            )
            scored.append({
                "path": str(path),
                "file": path.name,
                "timestamp_ms": item["timestamp_ms"],
                "prompt_index": index,
                "prompt": expectation["prompt"],
                "seed": expectation["seed"],
                "case_id": expectation.get("case_id"),
                "category": expectation.get("category"),
                "metrics": _image_metrics(image),
                "identity": _identity_metric(image, self._face, self._reference),
                "pose_body_landmarks": landmarks,
            })
        selected, sample_run_status, discarded = _choose_latest_complete_run(scored, expected_indices)
        cosines = [
            sample["identity"]["cosine_similarity"]
            for sample in selected
            if sample["identity"]["status"] == "available"
        ]
        clipping = [sample["metrics"]["clipping_fraction_proxy"] for sample in selected]
        record.update({
            "sample_run_status": sample_run_status,
            "discarded_duplicate_or_partial_samples": discarded,
            "samples": selected,
            "reference_identity_status": {
                key: value for key, value in self._reference.items() if key != "embedding"
            },
            "face_backend": _backend_provenance(self._face, self.config.get("face_backend")),
            "pose_backend": _backend_provenance(self._landmark, self.config.get("landmark_backend")),
            "score": {
                "status": "available" if cosines else "unavailable",
                "reason": None if cosines else "no generated image has exactly one valid face",
                "mean_face_cosine_similarity": (sum(cosines) / len(cosines)) if cosines else None,
                "face_sample_count": len(cosines),
                "mean_clipping_fraction_proxy": (sum(clipping) / len(clipping)) if clipping else None,
                "method": CHECKPOINT_SCORE_METHOD,
            },
        })
        return record


def evaluate_job(
    *,
    job_config_path: Path,
    output_dir: Path,
    reference_images: list[Path],
    config: Mapping[str, Any],
) -> Path:
    with preserve_cuda_visibility():
        return _evaluate_job(
            job_config_path=job_config_path,
            output_dir=output_dir,
            reference_images=reference_images,
            config=config,
        )


def persist_selection(report_path: Path, step: int, note: str) -> Path:
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = next((item for item in report["checkpoints"] if int(item["step"]) == int(step)), None)
    if checkpoint is None:
        raise ValueError(f"step {step} is absent from evaluation report")
    selection_path = report_path.with_name("selection.json")
    atomic_write_json(
        selection_path,
        {
            "schema_version": 1,
            "selected_step": int(step),
            "human_note": note,
            "evidence": checkpoint,
            "source_report": str(report_path),
        },
    )
    return selection_path
