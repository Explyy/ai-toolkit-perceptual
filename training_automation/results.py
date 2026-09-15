from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import tempfile
import textwrap
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont

from .backup import BackupError, LocalArtifact, sha256_file, verify_remote_artifacts
from .catalog import CatalogStore, resolve_model, safe_relative_path
from .gallery import render_gallery
from .state import atomic_write_json


COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ResultsClient(Protocol):
    def repo_is_private(self, repo_id: str, repo_type: str) -> bool: ...

    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...

    def commit_files(
        self, repo_id: str, repo_type: str, artifacts: list[LocalArtifact], message: str,
        parent_commit: str | None = None,
    ) -> str: ...

    def path_metadata(
        self, repo_id: str, repo_type: str, paths: list[str], revision: str
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def download_file(
        self, repo_id: str, repo_type: str, path: str, revision: str, destination: Path
    ) -> None: ...


def _component(value: Any, field: str) -> str:
    text = str(value or "")
    if text in {".", ".."} or not COMPONENT_RE.fullmatch(text):
        raise BackupError(f"{field} must be one safe path component")
    return text


def _load_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise BackupError(f"{description} is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise BackupError(f"{description} must be a JSON object")
    return document


def report_job_id(report_path: Path) -> str:
    report = _load_object(report_path, "evaluation report")
    configured = str(report.get("job_config") or "")
    if not configured:
        raise BackupError("evaluation report does not record its stable job config")
    return _component(Path(configured).stem, "job_id")


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else "unavailable"


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return [(prefix or "value", value)]
    flattened = []
    for key in sorted(value, key=str):
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value[key], Mapping):
            flattened.extend(_flatten(value[key], label))
        else:
            flattened.append((label, value[key]))
    return flattened


def _sample_lines(sample: Mapping[str, Any]) -> list[str]:
    identity = sample.get("identity") or {}
    metrics = sample.get("metrics") or {}
    pose = sample.get("pose_body_landmarks") or {}
    identity_value = (
        _number(identity.get("cosine_similarity"))
        if identity.get("status") == "available"
        else str(identity.get("status", "unavailable"))
    )
    lines = [
        f"Prompt {sample.get('prompt_index')} | seed {sample.get('seed')}",
        f"Raw face cosine: {identity_value}",
        f"Clipping fraction proxy: {_number(metrics.get('clipping_fraction_proxy'))}",
        f"Sharpness proxy: {_number(metrics.get('sharpness_laplacian_variance_proxy'))}",
        f"Pose status: {pose.get('status', 'unavailable')}",
    ]
    if pose.get("status") == "available":
        lines.extend(f"{key}: {_number(value)}" for key, value in _flatten(pose.get("values") or {}))
    prompt = str(sample.get("prompt") or "prompt unavailable")
    lines.extend(["", *textwrap.wrap(prompt, width=46)])
    return lines


def _safe_sample(sample_root: Path, raw_path: Any) -> tuple[Path, str]:
    name = PurePosixPath(str(raw_path or "").replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise BackupError("result sample path has no safe basename")
    root = sample_root.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise BackupError(f"result sample is missing or escapes sample root: {name}")
    return path, name


def _sample_manifest(sample: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {
        "sample_basename": name,
        "prompt_index": sample.get("prompt_index"),
        "prompt": sample.get("prompt"),
        "seed": sample.get("seed"),
        "metrics": sample.get("metrics"),
        "identity": sample.get("identity"),
        "pose_body_landmarks": sample.get("pose_body_landmarks"),
    }


def render_contact_sheet(
    checkpoint: Mapping[str, Any],
    *,
    sample_root: Path,
    output_path: Path,
    heading: str,
    aggregate: Mapping[str, Any],
) -> Path:
    """Create a deterministic two-column raster sheet without cropping sample images."""
    sample_root = sample_root.resolve()
    font = ImageFont.load_default()
    cards = []
    seen = set()
    samples = sorted(
        checkpoint.get("samples") or [],
        key=lambda item: (int(item.get("prompt_index", -1)), int(item.get("seed", -1))),
    )
    for sample in samples:
        path, name = _safe_sample(sample_root, sample.get("path"))
        if name in seen:
            raise BackupError(f"contact sheet has duplicate sample basename: {name}")
        seen.add(name)
        try:
            with Image.open(path) as opened:
                opened.load()
                media = opened.convert("RGB")
        except Exception as exc:
            raise BackupError(f"contact sheet sample is not a readable image: {name}") from exc
        media.thumbnail((330, 360))
        lines = _sample_lines(sample)
        text = "\n".join(lines)
        probe = Image.new("RGB", (1, 1))
        bounds = ImageDraw.Draw(probe).multiline_textbbox((0, 0), text, font=font, spacing=3)
        text_height = bounds[3] - bounds[1]
        card_height = max(media.height, text_height) + 28
        card = Image.new("RGB", (720, card_height), "#121722")
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card.width - 1, card.height - 1), outline="#344055")
        card.paste(media, (14, 14))
        draw.multiline_text((366, 14), text, fill="#eef1f5", font=font, spacing=3)
        cards.append(card)
    if not cards:
        raise BackupError("contact sheet checkpoint has no samples")
    columns = 2
    gap = 14
    title_lines = [
        heading,
        f"Checkpoint aggregate raw mean face cosine: {_number(aggregate.get('mean_face_cosine_similarity'))}",
        f"Checkpoint aggregate mean clipping proxy: {_number(aggregate.get('mean_clipping_fraction_proxy'))}",
        "Common valid-face prompts used by aggregate: "
        + ", ".join(str(value) for value in aggregate.get("prompt_indices") or []),
        "Individual values appear beside each uncropped image; pose values are image-plane proxies.",
    ]
    title_text = "\n".join(title_lines)
    title_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_bounds = title_probe.multiline_textbbox(
        (0, 0), title_text, font=font, spacing=4
    )
    title_height = title_bounds[3] - title_bounds[1] + 2 * gap
    row_heights = [
        max(card.height for card in cards[row:row + columns])
        for row in range(0, len(cards), columns)
    ]
    width = columns * 720 + (columns + 1) * gap
    height = title_height + sum(row_heights) + (len(row_heights) + 1) * gap
    sheet = Image.new("RGB", (width, height), "#0c0e12")
    draw = ImageDraw.Draw(sheet)
    draw.multiline_text(
        (gap, gap), title_text, fill="#eef1f5", font=font, spacing=4
    )
    y = title_height + gap
    for row_index, row_height in enumerate(row_heights):
        for column, card in enumerate(cards[row_index * columns:(row_index + 1) * columns]):
            sheet.paste(card, (gap + column * (720 + gap), y))
        y += row_height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    return output_path


def _catalog_identity(
    catalog: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint_ids = {
        str(item.get("catalog_checkpoint_id"))
        for item in report.get("checkpoints") or []
        if item.get("catalog_checkpoint_id")
    }
    if not checkpoint_ids:
        raise BackupError("result export report has no verified catalog checkpoint identities")
    matches = []
    for model in catalog.get("models") or []:
        model_ids = {
            str(item.get("checkpoint_id")) for item in model.get("checkpoints") or []
        }
        if checkpoint_ids.issubset(model_ids):
            matches.append(model)
    if len(matches) != 1:
        raise BackupError("result export checkpoints do not resolve to one catalog model")
    return resolve_model(catalog, matches[0]["id"])


def _shortlist_checkpoints(
    report: Mapping[str, Any]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], str | None]:
    shortlist = report.get("automatic_shortlist") or {}
    if shortlist.get("status") != "available":
        return [], str(shortlist.get("reason") or "automatic shortlist unavailable")
    if (
        (report.get("ranking") or {}).get("status") != "available"
        or (report.get("identity_ranking") or {}).get("status") != "available"
    ):
        raise BackupError(
            "available automatic shortlist requires completed image and identity rankings"
        )
    items = shortlist.get("items") or []
    checkpoint_documents = report.get("checkpoints") or []
    if len(items) != min(3, len(checkpoint_documents)) or not items:
        raise BackupError(
            "available automatic shortlist must contain the deterministic top candidates"
        )
    checkpoints = {int(item["step"]): item for item in checkpoint_documents}
    if len(checkpoints) != len(checkpoint_documents):
        raise BackupError("evaluation report contains duplicate checkpoint steps")
    signatures = []
    for checkpoint in checkpoint_documents:
        samples = checkpoint.get("samples") or []
        signature = [
            (int(sample["prompt_index"]), int(sample["seed"])) for sample in samples
        ]
        if not signature or len(signature) != len(set(signature)):
            raise BackupError("checkpoint samples lack a unique prompt/seed set")
        signatures.append(set(signature))
        if (
            checkpoint.get("sample_run_status") != "complete"
            or checkpoint.get("remote_association", {}).get("status") != "unique"
        ):
            raise BackupError(
                "available automatic shortlist requires complete uniquely associated checkpoints"
            )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise BackupError(
            "available automatic shortlist requires comparable checkpoint prompt/seed sets"
        )
    identity_items = {
        int(item["step"]): item
        for item in (report.get("identity_ranking") or {}).get("items") or []
    }
    if set(identity_items) != set(checkpoints):
        raise BackupError(
            "available automatic shortlist requires identity aggregates for every checkpoint"
        )
    expected = []
    for step, checkpoint in checkpoints.items():
        clipping = [
            float(sample["metrics"]["clipping_fraction_proxy"])
            for sample in checkpoint["samples"]
        ]
        expected.append({
            **identity_items[step],
            "mean_clipping_fraction_proxy": sum(clipping) / len(clipping),
        })
    expected.sort(key=lambda item: (
        -float(item["mean_face_cosine_similarity"]),
        float(item["mean_clipping_fraction_proxy"]),
        int(item["step"]),
    ))
    if items != expected[:3]:
        raise BackupError("automatic shortlist differs from the deterministic report ranking")
    result = []
    seen_steps = set()
    for item in items:
        step = int(item["step"])
        if step in seen_steps:
            raise BackupError("automatic shortlist contains a duplicate checkpoint step")
        seen_steps.add(step)
        checkpoint = checkpoints.get(step)
        if checkpoint is None:
            raise BackupError(f"shortlisted step {step} is absent from checkpoint evidence")
        checkpoint_id = str(checkpoint.get("catalog_checkpoint_id") or "")
        if not checkpoint_id:
            raise BackupError(f"shortlisted step {step} has no catalog checkpoint identity")
        result.append((item, checkpoint))
    return result, None


def _weight_artifacts(
    *,
    client: ResultsClient,
    repo_id: str,
    repo_type: str,
    model: Mapping[str, Any],
    shortlist: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    remote_root: PurePosixPath,
    download_root: Path,
) -> tuple[list[LocalArtifact], list[list[dict[str, Any]]]]:
    catalog_by_id = {
        str(item.get("checkpoint_id")): item for item in model.get("checkpoints") or []
    }
    uploads = []
    manifests = []
    used_destinations = set()
    for rank, (_, report_checkpoint) in enumerate(shortlist, 1):
        checkpoint_id = str(report_checkpoint["catalog_checkpoint_id"])
        checkpoint = catalog_by_id.get(checkpoint_id)
        if checkpoint is None:
            raise BackupError(f"shortlisted checkpoint disappeared from catalog: {checkpoint_id}")
        revision = str(checkpoint.get("revision") or "")
        reported_revision = str(
            (report_checkpoint.get("remote_checkpoint") or {}).get("commit_id") or ""
        )
        if not revision or revision != reported_revision:
            raise BackupError(f"shortlisted checkpoint revision mismatch: {checkpoint_id}")
        weights = checkpoint.get("weights") or []
        if not weights:
            raise BackupError(f"shortlisted checkpoint has no generation weights: {checkpoint_id}")
        expected = [
            LocalArtifact(
                local_path="",
                remote_path=str(weight["remote_path"]),
                size=int(weight["size"]),
                sha256=str(weight["sha256"]),
                role="weights",
            )
            for weight in weights
        ]
        verify_remote_artifacts(
            client,
            repo_id=repo_id,
            repo_type=repo_type,
            artifacts=expected,
            revision=revision,
            context=f"top-{rank} source checkpoint",
        )
        rank_manifest = []
        used_names = set()
        for index, weight in enumerate(weights):
            source_path = safe_relative_path(str(weight["remote_path"]))
            name = source_path.name
            if name in used_names:
                raise BackupError(f"shortlisted checkpoint has duplicate weight basename: {name}")
            used_names.add(name)
            local = download_root / f"top-{rank}" / f"{index:02d}-{name}"
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(repo_id, repo_type, str(weight["remote_path"]), revision, local)
            if (
                local.stat().st_size != int(weight["size"])
                or sha256_file(local) != str(weight["sha256"])
            ):
                raise BackupError(
                    f"shortlisted checkpoint download verification failed: {source_path}"
                )
            destination = str(remote_root / f"top-{rank}" / "weights" / name)
            if destination in used_destinations:
                raise BackupError(f"duplicate result weight destination: {destination}")
            used_destinations.add(destination)
            uploads.append(LocalArtifact(
                str(local), destination, local.stat().st_size, sha256_file(local),
                role="weights", relative_path=name,
            ))
            rank_manifest.append({
                "source_remote_path": str(weight["remote_path"]),
                "source_revision": revision,
                "destination_remote_path": destination,
                "size": int(weight["size"]),
                "sha256": str(weight["sha256"]),
            })
        manifests.append(rank_manifest)
    return uploads, manifests


def publish_ranked_results(
    *,
    client: ResultsClient,
    repo_id: str,
    repo_type: str,
    run_id: str,
    job_id: str,
    report_path: Path,
    sample_root: Path,
    work_dir: Path,
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
    max_attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    """Publish deterministic shortlist evidence and verified weight copies without selection."""
    run_id = _component(run_id, "run_id")
    job_id = _component(job_id, "job_id")
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("refusing result export to a non-private repository")
    report = _load_object(report_path, "evaluation report")
    report_sha256 = sha256_file(report_path)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    catalog_store = CatalogStore(
        client=client,
        repo_id=repo_id,
        repo_type=repo_type,
        catalog_path=f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json",
        work_dir=work_dir / ".catalog",
    )
    download_root = Path(tempfile.mkdtemp(prefix="result-weights-", dir=work_dir))
    try:
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                catalog, parent = catalog_store.read()
                model = _catalog_identity(catalog, report)
                model_folder = safe_relative_path(str(model["folder"]))
                if (
                    len(model_folder.parts) != 1
                    or not re.fullmatch(r"[0-9]{4}-[a-z0-9][a-z0-9-]*", model_folder.name)
                ):
                    raise BackupError("catalog model folder must be one safe numeric-name component")
                gallery_path = render_gallery(
                    report_path,
                    sample_root=sample_root,
                    output_path=work_dir / "evaluation-gallery.html",
                    title=f"{model['folder']} · {model['name']} · evaluation gallery",
                )
                shortlist, unavailable_reason = _shortlist_checkpoints(report)
                remote_root = (
                    PurePosixPath(safe_relative_path(results_prefix).as_posix())
                    / run_id / model_folder.as_posix()
                )
                existing_index, _ = client.read_remote_file(
                    repo_id,
                    repo_type,
                    str(remote_root / "index.json"),
                )
                if existing_index is not None:
                    try:
                        existing = json.loads(existing_index)
                    except (TypeError, ValueError) as exc:
                        raise BackupError(
                            "existing result index is not valid JSON"
                        ) from exc
                    if (
                        not isinstance(existing, Mapping)
                        or existing.get("run_id") != run_id
                        or existing.get("job_id") != job_id
                        or existing.get("evaluation_report_sha256") != report_sha256
                    ):
                        raise BackupError(
                            "result destination already contains different evidence"
                        )
                shutil.rmtree(download_root, ignore_errors=True)
                download_root.mkdir(parents=True)
                weight_uploads, weight_manifests = _weight_artifacts(
                    client=client,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    model=model,
                    shortlist=shortlist,
                    remote_root=remote_root,
                    download_root=download_root,
                )
                local_uploads: list[LocalArtifact] = []
                evidence_files: list[tuple[Path, str]] = []
                candidate_records = []
                for rank, ((aggregate, checkpoint), weights) in enumerate(
                    zip(shortlist, weight_manifests), 1
                ):
                    step = int(checkpoint["step"])
                    samples = []
                    for sample in checkpoint.get("samples") or []:
                        _, name = _safe_sample(sample_root, sample.get("path"))
                        samples.append(_sample_manifest(sample, name))
                    rank_dir = work_dir / f"top-{rank}"
                    sheet = render_contact_sheet(
                        checkpoint,
                        sample_root=sample_root,
                        output_path=rank_dir / "contact-sheet.png",
                        heading=f"{model['folder']} · top-{rank} · step {step}",
                        aggregate=aggregate,
                    )
                    metrics = {
                        "schema_version": 1,
                        "selection_kind": "automatic-evidence-rank",
                        "rank": rank,
                        "run_id": run_id,
                        "job_id": job_id,
                        "model": {
                            key: model.get(key)
                            for key in (
                                "id", "name", "folder", "base_arch", "base_model",
                                "trigger_word", "destination_kind",
                            )
                        },
                        "catalog_checkpoint_id": checkpoint["catalog_checkpoint_id"],
                        "canonical_selection_unchanged": True,
                        "step": step,
                        "final": bool(checkpoint.get("final")),
                        "checkpoint_aggregate": dict(aggregate),
                        "source_weights": weights,
                        "samples": samples,
                        "metric_limits": {
                            "identity": "raw cosine similarity, not a percentage or probability",
                            "image_and_pose": "heuristic image-plane proxies, not ground-truth quality or anatomy",
                        },
                    }
                    restore_argv = [
                        "python",
                        "-m",
                        "training_automation",
                        "restore",
                        f"{int(model['id']):04d}",
                        "--checkpoint-id",
                        str(checkpoint["catalog_checkpoint_id"]),
                        "--repo-type",
                        repo_type,
                        "--mode",
                        "generation",
                        "--comfy-root",
                        "/workspace/ComfyUI",
                    ]
                    metrics["restore"] = {
                        "argv": restore_argv,
                        "command": shlex.join(restore_argv),
                        "required_environment": ["HF_TOKEN", "HF_REPO_ID"],
                    }
                    metrics_path = rank_dir / "selection.json"
                    atomic_write_json(metrics_path, metrics)
                    for local, name in (
                        (sheet, "contact-sheet.png"),
                        (metrics_path, "selection.json"),
                    ):
                        remote = str(remote_root / f"top-{rank}" / name)
                        local_uploads.append(LocalArtifact(
                            str(local), remote, local.stat().st_size, sha256_file(local),
                            role="evidence", relative_path=name,
                        ))
                        evidence_files.append((local, f"results/{model['folder']}/top-{rank}/{name}"))
                    candidate_records.append({
                        "rank": rank,
                        "step": step,
                        "final": bool(checkpoint.get("final")),
                        "catalog_checkpoint_id": checkpoint["catalog_checkpoint_id"],
                        "checkpoint_aggregate": dict(aggregate),
                        "remote_folder": str(remote_root / f"top-{rank}"),
                        "weights": weights,
                    })
                index = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "job_id": job_id,
                    "evaluation_report_sha256": report_sha256,
                    "model": {
                        key: model.get(key)
                        for key in (
                            "id", "name", "folder", "base_arch", "base_model",
                            "trigger_word", "destination_kind",
                        )
                    },
                    "status": "available" if shortlist else "unavailable",
                    "reason": unavailable_reason,
                    "exported_candidate_count": len(shortlist),
                    "method": (report.get("automatic_shortlist") or {}).get("method"),
                    "canonical_selected_checkpoint_id": model.get("selected_checkpoint_id"),
                    "canonical_selection_unchanged": True,
                    "candidates": candidate_records,
                }
                index_path = work_dir / "index.json"
                atomic_write_json(index_path, index)
                for local, name in ((index_path, "index.json"), (gallery_path, "evaluation-gallery.html")):
                    remote = str(remote_root / name)
                    local_uploads.append(LocalArtifact(
                        str(local), remote, local.stat().st_size, sha256_file(local),
                        role="evidence", relative_path=name,
                    ))
                    evidence_files.append((local, f"results/{model['folder']}/{name}"))
                uploads = [*local_uploads, *weight_uploads]
                revision = client.commit_files(
                    repo_id,
                    repo_type,
                    uploads,
                    f"Publish automatic top checkpoints for {run_id}/{model['folder']}",
                    parent_commit=parent,
                )
                verify_remote_artifacts(
                    client,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    artifacts=uploads,
                    revision=revision,
                    context="ranked result export",
                )
                record = {
                    **index,
                    "revision": revision,
                    "remote_root": str(remote_root),
                }
                return record, evidence_files
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, int(max_attempts)):
                    sleep(2**attempt)
        raise BackupError("ranked result export failed after bounded retries") from last_error
    finally:
        shutil.rmtree(download_root, ignore_errors=True)


def publish_archived_run_results(
    *,
    client: ResultsClient,
    repo_id: str,
    repo_type: str,
    run_id: str,
    archive_root: Path,
    work_dir: Path,
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
) -> list[dict[str, Any]]:
    """Publish every direct job directory from a reconstructed immutable archive."""
    root = archive_root.resolve()
    jobs_root = (root / "jobs").resolve()
    if jobs_root.parent != root or not jobs_root.is_dir():
        raise BackupError("archived run root must contain one direct jobs directory")
    jobs = sorted(path for path in jobs_root.iterdir() if path.is_dir())
    if not jobs:
        raise BackupError("archived run has no job directories")
    records = []
    for job_root in jobs:
        resolved_job = job_root.resolve()
        if resolved_job.parent != jobs_root:
            raise BackupError("archived job directory escapes jobs root")
        job_id = _component(job_root.name, "archived job_id")
        report_path = resolved_job / "evaluation.json"
        if report_job_id(report_path) != job_id:
            raise BackupError(
                f"archived report identity does not match its job directory: {job_id}"
            )
        record, _ = publish_ranked_results(
            client=client,
            repo_id=repo_id,
            repo_type=repo_type,
            run_id=run_id,
            job_id=job_id,
            report_path=report_path,
            sample_root=resolved_job / "samples",
            work_dir=work_dir / job_id,
            catalog_prefix=catalog_prefix,
            results_prefix=results_prefix,
        )
        records.append(record)
    return records
