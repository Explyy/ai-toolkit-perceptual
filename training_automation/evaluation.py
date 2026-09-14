from __future__ import annotations

import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import yaml
from PIL import Image

from .state import atomic_write_json


REPORT_SCHEMA = 1
SAMPLE_RE = re.compile(r"^(?P<time>\d+)__(?P<step>\d+)_(?P<index>\d+)\.(?:png|jpe?g|webp)$", re.I)


class FaceBackend(Protocol):
    def embeddings(self, image: np.ndarray) -> list[np.ndarray]: ...


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


def _reference_identity(reference_images: list[Path], backend: FaceBackend | None) -> dict[str, Any]:
    if backend is None:
        return {"status": "unavailable", "embedding": None, "reason": "face backend not configured"}
    embeddings = []
    for path in reference_images:
        faces = backend.embeddings(np.asarray(Image.open(path).convert("RGB")))
        if len(faces) == 0:
            return {"status": "missing", "embedding": None, "reason": f"no face in reference {path.name}"}
        if len(faces) != 1:
            return {"status": "ambiguous", "embedding": None, "reason": f"multiple faces in reference {path.name}"}
        vector = _unit(faces[0])
        if vector is None:
            return {"status": "unavailable", "embedding": None, "reason": "invalid reference embedding"}
        embeddings.append(vector)
    if not embeddings:
        return {"status": "missing", "embedding": None, "reason": "no reference images supplied"}
    average = _unit(np.mean(embeddings, axis=0))
    if average is None:
        return {"status": "unavailable", "embedding": None, "reason": "invalid averaged reference embedding"}
    return {"status": "available", "embedding": average, "reason": None}


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


def _sample_expectations(job_config_path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    with job_config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    process = document["config"]["process"][0]
    sample = process.get("sample", {})
    raw = sample.get("samples")
    if raw is None:
        raw = [{"prompt": prompt} for prompt in sample.get("prompts", [])]
    base_seed = int(sample.get("seed", 42))
    walk = bool(sample.get("walk_seed", False))
    expected = {}
    for index, item in enumerate(raw):
        expected[index] = {
            "prompt": item.get("prompt"),
            "seed": int(item.get("seed", base_seed + index if walk else base_seed)),
        }
    return expected, int(process.get("train", {}).get("steps", 0))


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
    if not available_sets or not available_sets[0]:
        return [], "no checkpoint has a valid identity subset"
    if any(indices != available_sets[0] for indices in available_sets[1:]):
        return [], "valid identity subsets are not comparable across checkpoints"
    ranked = []
    for checkpoint in checkpoints:
        values = [
            sample["identity"]["cosine_similarity"]
            for sample in checkpoint["samples"]
            if sample["prompt_index"] in available_sets[0]
        ]
        ranked.append(
            {
                "step": checkpoint["step"],
                "mean_face_cosine_similarity": sum(values) / len(values),
                "prompt_indices": sorted(available_sets[0]),
            }
        )
    ranked.sort(key=lambda item: (-item["mean_face_cosine_similarity"], item["step"]))
    return ranked, None


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


def evaluate_job(
    *,
    job_config_path: Path,
    output_dir: Path,
    reference_images: list[Path],
    config: Mapping[str, Any],
) -> Path:
    expected, final_step = _sample_expectations(job_config_path)
    with job_config_path.open("r", encoding="utf-8") as handle:
        job_name = yaml.safe_load(handle)["config"]["name"]
    face = _load_backend(config.get("face_backend"), config.get("face_backend_options", {}))
    landmark = _load_backend(config.get("landmark_backend"), config.get("landmark_backend_options", {}))
    reference = _reference_identity(reference_images, face)
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
                "metrics": _image_metrics(image),
                "identity": _identity_metric(image, face, reference),
                "pose_body_landmarks": landmarks,
            }
        )
    checkpoint_steps: set[int] = set()
    checkpoint_paths: dict[int, list[str]] = {}
    checkpoint_remote: dict[int, dict[str, Any]] = {}
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
            checkpoint_remote[step] = {
                "catalog_checkpoint_id": entry.get("catalog_checkpoint_id"),
                "commit_id": entry.get("commit_id"),
                "cataloged": True,
                "destination": backup_state.get("destination"),
            }
    expected_indices = set(expected)
    checkpoints = []
    for step in sorted(checkpoint_steps & grouped.keys()):
        selected_samples, sample_run_status, discarded_samples = _choose_latest_complete_run(
            grouped[step], expected_indices
        )
        checkpoints.append({
            "step": step,
            "final": step == final_step,
            "checkpoint_paths": sorted(checkpoint_paths.get(step, [])),
            "catalog_checkpoint_id": checkpoint_remote.get(step, {}).get("catalog_checkpoint_id"),
            "remote_checkpoint": checkpoint_remote.get(step),
            "samples": selected_samples,
            "sample_run_status": sample_run_status,
            "discarded_duplicate_or_partial_samples": discarded_samples,
        })
    expected_signature = {(index, item["seed"]) for index, item in expected.items()}
    ranked, ranking_error = rank_checkpoints(checkpoints, expected_signature)
    identity_ranked, identity_ranking_error = rank_identity(checkpoints)
    report = {
        "schema_version": REPORT_SCHEMA,
        "job_config": str(job_config_path),
        "final_step": final_step,
        "reference_identity_status": {key: value for key, value in reference.items() if key != "embedding"},
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
            "method": "mean face cosine similarity on the same valid prompt-index subset for every checkpoint; heuristic only",
            "items": identity_ranked,
        },
        "pose_body_limit": "optional backend output is viewpoint- and visibility-dependent and is not a 3D body measurement",
    }
    report_path = output_dir / ".automation" / "evaluation.json"
    atomic_write_json(report_path, report)
    return report_path


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
