from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont

from .backup import BackupError, LocalArtifact, sha256_file, verify_remote_artifacts
from .catalog import CatalogStore, resolve_model, safe_relative_path
from .gallery import render_gallery
from .reporting import render_checkpoint_pages, render_top_comparison
from .state import atomic_write_json


COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
BODY_FONT_SIZE = 20
SAMPLE_HEADING_FONT_SIZE = 24
SHEET_HEADING_FONT_SIZE = 30
CARD_WIDTH = 860
MEDIA_MAX_SIZE = (360, 420)
CARD_PADDING = 18
CARD_TEXT_X = 402
CARD_TEXT_WIDTH = CARD_WIDTH - CARD_TEXT_X - CARD_PADDING
GRID_GAP = 18


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

    def list_repo_files(
        self, repo_id: str, repo_type: str, revision: str
    ) -> Sequence[str]: ...


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


def _completion_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackupError("result completion timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise BackupError("result completion timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _scalable_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        for candidate in (
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    raise BackupError(f"contact sheet requires a scalable font at {size}px")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text or " ", font=font)
    return box[2] - box[0]


def _wrap_text_pixels(
    draw: ImageDraw.ImageDraw,
    text: Any,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap untrusted text to a measured pixel width, splitting long tokens safely."""
    words = str(text).replace("\r", " ").replace("\n", " \n ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = f"{current} {word}".strip()
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        fragment = ""
        for character in word:
            candidate = fragment + character
            if fragment and _text_width(draw, candidate, font) > max_width:
                lines.append(fragment)
                fragment = character
            else:
                fragment = candidate
        current = fragment
    if current or not lines:
        lines.append(current)
    return lines


def _line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return box[3] - box[1]


def _sample_text_rows(sample: Mapping[str, Any]) -> list[tuple[str, str]]:
    identity = sample.get("identity") or {}
    metrics = sample.get("metrics") or {}
    pose = sample.get("pose_body_landmarks") or {}
    identity_value = (
        _number(identity.get("cosine_similarity"))
        if identity.get("status") == "available"
        else str(identity.get("status", "unavailable"))
    )
    rows = [
        ("heading", f"Prompt {sample.get('prompt_index')} · seed {sample.get('seed')}"),
        ("primary", f"Raw face cosine  {identity_value}"),
        ("metric", f"Clipping proxy  {_number(metrics.get('clipping_fraction_proxy'))}"),
        ("metric", f"Sharpness proxy  {_number(metrics.get('sharpness_laplacian_variance_proxy'))}"),
        ("status", f"Pose status  {pose.get('status', 'unavailable')}"),
    ]
    if pose.get("status") == "available":
        ratios = ((pose.get("values") or {}).get("ratios") or {})
        for key, label in (
            ("shoulder_to_hip_width", "Shoulder / hip width"),
            ("left_thigh_to_torso", "Left thigh / torso"),
            ("right_thigh_to_torso", "Right thigh / torso"),
        ):
            if ratios.get(key) is not None:
                rows.append(("metric", f"{label}  {_number(ratios[key])}"))
    rows.append(("prompt", f"Prompt text  {sample.get('prompt') or 'unavailable'}"))
    return rows


def _layout_rows(
    draw: ImageDraw.ImageDraw,
    rows: Sequence[tuple[str, str]],
    *,
    fonts: Mapping[str, ImageFont.ImageFont],
    max_width: int,
) -> tuple[list[tuple[str, str, ImageFont.ImageFont]], int]:
    lines = []
    height = 0
    for style, text in rows:
        font = fonts[style]
        for line in _wrap_text_pixels(draw, text, font=font, max_width=max_width):
            lines.append((style, line, font))
            height += _line_height(font) + 7
        height += 5 if style in {"heading", "status"} else 0
    return lines, height


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
    body_font = _scalable_font(BODY_FONT_SIZE)
    sample_heading_font = _scalable_font(SAMPLE_HEADING_FONT_SIZE)
    sheet_heading_font = _scalable_font(SHEET_HEADING_FONT_SIZE)
    fonts = {
        "heading": sample_heading_font,
        "primary": body_font,
        "metric": body_font,
        "status": body_font,
        "prompt": body_font,
    }
    colors = {
        "heading": "#ffffff",
        "primary": "#8fe3d0",
        "metric": "#d9dee7",
        "status": "#f0c875",
        "prompt": "#aeb8c8",
    }
    probe_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
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
        media.thumbnail(MEDIA_MAX_SIZE)
        lines, text_height = _layout_rows(
            probe_draw,
            _sample_text_rows(sample),
            fonts=fonts,
            max_width=CARD_TEXT_WIDTH,
        )
        card_height = max(media.height, text_height) + 2 * CARD_PADDING
        card = Image.new("RGB", (CARD_WIDTH, card_height), "#121722")
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card.width - 1, card.height - 1), outline="#344055")
        card.paste(media, (CARD_PADDING, CARD_PADDING))
        y = CARD_PADDING
        for style, line, font in lines:
            draw.text((CARD_TEXT_X, y), line, fill=colors[style], font=font)
            y += _line_height(font) + 7
            if style in {"heading", "status"}:
                y += 5
        cards.append(card)
    if not cards:
        raise BackupError("contact sheet checkpoint has no samples")
    columns = 2
    header_rows = [
        ("heading", heading),
        (
            "primary",
            "CHECKPOINT AGGREGATE · raw mean face cosine  "
            + _number(aggregate.get("mean_face_cosine_similarity")),
        ),
        (
            "metric",
            "Mean clipping proxy  "
            + _number(aggregate.get("mean_clipping_fraction_proxy")),
        ),
        (
            "metric",
            "Common valid-face prompts  "
            + ", ".join(str(value) for value in aggregate.get("prompt_indices") or []),
        ),
        (
            "prompt",
            "Individual values sit beside each uncropped image. Pose values are image-plane proxies.",
        ),
    ]
    header_fonts = {
        **fonts,
        "heading": sheet_heading_font,
    }
    width = columns * CARD_WIDTH + (columns + 1) * GRID_GAP
    header_lines, header_text_height = _layout_rows(
        probe_draw,
        header_rows,
        fonts=header_fonts,
        max_width=width - 2 * GRID_GAP,
    )
    title_height = header_text_height + 2 * GRID_GAP
    row_heights = [
        max(card.height for card in cards[row:row + columns])
        for row in range(0, len(cards), columns)
    ]
    height = (
        title_height
        + sum(row_heights)
        + (len(row_heights) + 1) * GRID_GAP
    )
    sheet = Image.new("RGB", (width, height), "#0c0e12")
    draw = ImageDraw.Draw(sheet)
    y = GRID_GAP
    for style, line, font in header_lines:
        draw.text((GRID_GAP, y), line, fill=colors[style], font=font)
        y += _line_height(font) + 7
        if style in {"heading", "status"}:
            y += 5
    y = title_height + GRID_GAP
    for row_index, row_height in enumerate(row_heights):
        for column, card in enumerate(cards[row_index * columns:(row_index + 1) * columns]):
            sheet.paste(
                card,
                (GRID_GAP + column * (CARD_WIDTH + GRID_GAP), y),
            )
        y += row_height + GRID_GAP
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
    completed_at: str | None = None,
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
                checkpoints_by_step = {
                    int(item["step"]): item for item in report.get("checkpoints") or []
                }
                for rank, ((aggregate, checkpoint), weights) in enumerate(
                    zip(shortlist, weight_manifests), 1
                ):
                    step = int(checkpoint["step"])
                    samples = []
                    for sample in checkpoint.get("samples") or []:
                        _, name = _safe_sample(sample_root, sample.get("path"))
                        samples.append(_sample_manifest(sample, name))
                    rank_dir = work_dir / f"top-{rank}"
                    rendered = render_checkpoint_pages(
                        checkpoint,
                        sample_root=sample_root,
                        output_dir=rank_dir,
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
                        "report": {
                            "entry_preview": "contact-sheet.png",
                            "index": "report-index.json",
                            "pages": [
                                f"pages/{path.name}" for path in rendered.page_paths
                            ],
                        },
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
                    report_files = [
                        (rendered.entry_path, "contact-sheet.png"),
                        (rendered.index_path, "report-index.json"),
                        (metrics_path, "selection.json"),
                        *[(path, f"pages/{path.name}") for path in rendered.page_paths],
                    ]
                    for local, name in report_files:
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
                        "report": metrics["report"],
                    })
                overview_path = render_top_comparison(
                    report=report,
                    sample_root=sample_root,
                    shortlist=[item for item, _ in shortlist],
                    checkpoints=checkpoints_by_step,
                    output_path=work_dir / "overview.png",
                    heading=f"{model['folder']} · top checkpoint",
                )
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
                    "overview": "overview.png",
                }
                index_path = work_dir / "index.json"
                atomic_write_json(index_path, index)
                for local, name in (
                    (index_path, "index.json"),
                    (gallery_path, "evaluation-gallery.html"),
                    (overview_path, "overview.png"),
                ):
                    remote = str(remote_root / name)
                    local_uploads.append(LocalArtifact(
                        str(local), remote, local.stat().st_size, sha256_file(local),
                        role="evidence", relative_path=name,
                    ))
                    evidence_files.append((local, f"results/{model['folder']}/{name}"))
                pointer_path = (
                    PurePosixPath(safe_relative_path(results_prefix).as_posix())
                    / "latest" / f"{model_folder.name}.json"
                )
                resolved_completed_at = completed_at
                pointer_artifact = None
                if shortlist:
                    candidate_time = _completion_time(resolved_completed_at)
                    prior_pointer, _ = client.read_remote_file(
                        repo_id, repo_type, str(pointer_path)
                    )
                    prior = None
                    if prior_pointer is not None:
                        try:
                            prior = json.loads(prior_pointer)
                        except (TypeError, ValueError) as exc:
                            raise BackupError("existing latest result pointer is invalid JSON") from exc
                        if (
                            prior.get("run_id") == run_id
                            and prior.get("job_id") == job_id
                            and prior.get("evaluation_report_sha256") == report_sha256
                        ):
                            resolved_completed_at = str(prior.get("completed_at") or resolved_completed_at or "")
                            candidate_time = _completion_time(resolved_completed_at)
                    promote = candidate_time is not None
                    if prior is not None and not (
                        prior.get("run_id") == run_id
                        and prior.get("job_id") == job_id
                        and prior.get("evaluation_report_sha256") == report_sha256
                    ):
                        prior_time = _completion_time(prior.get("completed_at"))
                        promote = (
                            candidate_time is not None
                            and prior_time is not None
                            and candidate_time > prior_time
                        )
                    if promote:
                        pointer = {
                            "schema_version": 1,
                            "status": "completed",
                            "run_id": run_id,
                            "job_id": job_id,
                            "completed_at": resolved_completed_at,
                            "evaluation_report_sha256": report_sha256,
                            "index_path": str(remote_root / "index.json"),
                            "model": index["model"],
                        }
                        pointer_local = work_dir / "latest-result-pointer.json"
                        atomic_write_json(pointer_local, pointer)
                        pointer_artifact = LocalArtifact(
                            str(pointer_local), str(pointer_path), pointer_local.stat().st_size,
                            sha256_file(pointer_local), role="result-pointer",
                            relative_path=f"latest/{model_folder.name}.json",
                        )
                uploads = [*local_uploads, *weight_uploads]
                if pointer_artifact is not None:
                    uploads.append(pointer_artifact)
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
                    "latest_pointer": str(pointer_path) if pointer_artifact is not None else None,
                    "completed_at": resolved_completed_at,
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


def publish_refreshed_report(
    *,
    client: ResultsClient,
    repo_id: str,
    repo_type: str,
    run_id: str,
    report_path: Path,
    sample_root: Path,
    work_dir: Path,
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
) -> dict[str, Any]:
    """Render and publish additive report-v2 evidence without copying model weights."""
    run_id = _component(run_id, "run_id")
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("refusing report refresh to a non-private repository")
    report = _load_object(report_path, "evaluation report")
    report_digest = sha256_file(report_path)
    store = CatalogStore(
        client=client,
        repo_id=repo_id,
        repo_type=repo_type,
        catalog_path=f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json",
        work_dir=work_dir / ".catalog",
    )
    catalog, parent = store.read()
    model = _catalog_identity(catalog, report)
    folder = safe_relative_path(str(model["folder"]))
    if len(folder.parts) != 1:
        raise BackupError("report refresh model folder must be one component")
    remote_root = (
        PurePosixPath(safe_relative_path(results_prefix).as_posix())
        / run_id / folder.name / "reports-v2" / report_digest[:16]
    )
    checkpoints = {
        int(item["step"]): item for item in report.get("checkpoints") or []
    }
    if not checkpoints:
        raise BackupError("report refresh requires checkpoint evidence")
    shortlist, unavailable_reason = _shortlist_checkpoints(report)
    uploads: list[LocalArtifact] = []
    rendered_records = []
    for step, checkpoint in sorted(checkpoints.items()):
        aggregate = next(
            (
                item for item in (report.get("identity_ranking") or {}).get("items") or []
                if int(item["step"]) == step
            ),
            {"step": step, "prompt_indices": []},
        )
        rendered = render_checkpoint_pages(
            checkpoint,
            sample_root=sample_root,
            output_dir=work_dir / "reports-v2" / f"step-{step:09d}",
            heading=f"{folder.name} · step {step}",
            aggregate=aggregate,
        )
        local_files = [
            (rendered.entry_path, "contact-sheet.png"),
            (rendered.index_path, "report-index.json"),
            *[(path, f"pages/{path.name}") for path in rendered.page_paths],
        ]
        for local, relative in local_files:
            remote = str(remote_root / f"step-{step:09d}" / relative)
            uploads.append(LocalArtifact(
                str(local), remote, local.stat().st_size, sha256_file(local),
                role="refreshed-report", relative_path=f"step-{step:09d}/{relative}",
            ))
        rendered_records.append({
            "step": step,
            "sample_count": len(checkpoint.get("samples") or []),
            "pages": len(rendered.page_paths),
            "remote_root": str(remote_root / f"step-{step:09d}"),
        })
    overview = render_top_comparison(
        report=report,
        sample_root=sample_root,
        shortlist=[item for item, _ in shortlist],
        checkpoints=checkpoints,
        output_path=work_dir / "reports-v2" / "overview.png",
        heading=f"{folder.name} · top checkpoint",
    )
    uploads.append(LocalArtifact(
        str(overview), str(remote_root / "overview.png"), overview.stat().st_size,
        sha256_file(overview), role="refreshed-report", relative_path="overview.png",
    ))
    manifest = {
        "schema_version": 2,
        "kind": "additive-report-refresh",
        "run_id": run_id,
        "model": {key: model.get(key) for key in ("id", "name", "folder")},
        "source_evaluation_sha256": report_digest,
        "source_evidence_unchanged": True,
        "model_weights_transferred": False,
        "shortlist_status": "available" if shortlist else "unavailable",
        "shortlist_reason": unavailable_reason,
        "overview": "overview.png",
        "checkpoints": rendered_records,
    }
    manifest_path = work_dir / "reports-v2" / "report-manifest.json"
    atomic_write_json(manifest_path, manifest)
    uploads.append(LocalArtifact(
        str(manifest_path), str(remote_root / "report-manifest.json"),
        manifest_path.stat().st_size, sha256_file(manifest_path),
        role="refreshed-report", relative_path="report-manifest.json",
    ))
    revision = client.commit_files(
        repo_id, repo_type, uploads,
        f"Publish additive report v2 for {run_id}/{folder.name}",
        parent_commit=parent,
    )
    verify_remote_artifacts(
        client, repo_id=repo_id, repo_type=repo_type, artifacts=uploads,
        revision=revision, context="refreshed report",
    )
    return {**manifest, "revision": revision, "remote_root": str(remote_root)}


def refresh_archived_run_reports(
    *,
    client: ResultsClient,
    repo_id: str,
    repo_type: str,
    run_id: str,
    work_dir: Path,
    job_ids: Sequence[str] = (),
    source_revision: str | None = None,
    archive_prefix: str = "training-archives",
    catalog_prefix: str = "training-backups",
    results_prefix: str = "training-results",
) -> dict[str, Any]:
    """Rebuild additive reports from hash-verified immutable Hub archive evidence."""
    run_id = _component(run_id, "run_id")
    requested = {_component(value, "job_id") for value in job_ids}
    if not client.repo_is_private(repo_id, repo_type):
        raise BackupError("refusing archive report refresh from a non-private repository")
    if source_revision is None:
        catalog_path = f"{safe_relative_path(catalog_prefix).as_posix()}/catalog.json"
        catalog_payload, source_revision = client.read_remote_file(
            repo_id, repo_type, catalog_path
        )
        if catalog_payload is None or not source_revision:
            raise BackupError("cannot pin archive refresh to a catalog repository revision")
    source_revision = str(source_revision)
    archive_root = (
        PurePosixPath(safe_relative_path(archive_prefix).as_posix()) / run_id
    )
    completion_paths = sorted(
        path for path in client.list_repo_files(
            repo_id, repo_type, source_revision
        )
        if PurePosixPath(path).parent.parent == archive_root
        and PurePosixPath(path).name == "completion.json"
    )
    if not completion_paths:
        raise BackupError(f"no immutable archive completion exists for run {run_id}")
    staging_parent = work_dir.resolve()
    staging_parent.mkdir(parents=True, exist_ok=True)
    operation_root = Path(tempfile.mkdtemp(
        prefix=f"{run_id}-{source_revision[:12]}-", dir=staging_parent,
    )).resolve()
    records = []
    found_jobs: set[str] = set()
    for completion_path in completion_paths:
        shard = _component(PurePosixPath(completion_path).parent.name, "archive shard")
        completion_local = operation_root / shard / "completion.json"
        completion_local.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(
            repo_id, repo_type, completion_path, source_revision, completion_local
        )
        completion = _load_object(completion_local, "archive completion")
        if (
            completion.get("schema_version") != 1
            or completion.get("status") != "completed"
            or completion.get("run_id") != run_id
            or completion.get("shard_id") != shard
            or not completion.get("evidence_revision")
        ):
            raise BackupError(f"archive completion identity is invalid: {completion_path}")
        completion_jobs = {
            _component(item.get("job_id"), "archived job_id")
            for item in ((completion.get("completion") or {}).get("jobs") or [])
            if isinstance(item, Mapping)
        }
        if not completion_jobs:
            raise BackupError(f"archive completion has no job identities: {completion_path}")
        selected_jobs = completion_jobs & requested if requested else completion_jobs
        if found_jobs & selected_jobs:
            raise BackupError("an archived job is claimed by multiple completion manifests")
        evidence_revision = str(completion["evidence_revision"])
        evidence_by_relative: dict[str, Mapping[str, Any]] = {}
        for item in completion.get("evidence") or []:
            if not isinstance(item, Mapping):
                raise BackupError("archive completion evidence entry must be a mapping")
            relative = safe_relative_path(str(item.get("relative_path") or "")).as_posix()
            remote_path = safe_relative_path(str(item.get("remote_path") or "")).as_posix()
            expected_remote = str(
                archive_root / shard / "evidence" / PurePosixPath(relative)
            )
            if remote_path != expected_remote:
                raise BackupError(f"archive evidence remote path is outside its completion: {relative}")
            if relative in evidence_by_relative:
                raise BackupError(f"archive completion repeats evidence path: {relative}")
            if not isinstance(item.get("size"), int) or int(item["size"]) < 0:
                raise BackupError(f"archive evidence has invalid size: {relative}")
            sha = str(item.get("sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise BackupError(f"archive evidence has invalid SHA-256: {relative}")
            evidence_by_relative[relative] = {
                "remote_path": remote_path, "size": int(item["size"]), "sha256": sha,
            }
        for job_id in sorted(selected_jobs):
            prefix = f"jobs/{job_id}/"
            if any(
                relative.startswith(f"jobs/{job_id}/samples/")
                and len(PurePosixPath(relative).parts) != 4
                for relative in evidence_by_relative
            ):
                raise BackupError(f"archive sample evidence is not a direct file: {job_id}")
            selected = {
                relative: item for relative, item in evidence_by_relative.items()
                if relative == f"jobs/{job_id}/evaluation.json"
                or (
                    relative.startswith(f"jobs/{job_id}/samples/")
                    and len(PurePosixPath(relative).parts) == 4
                )
            }
            report_relative = f"jobs/{job_id}/evaluation.json"
            sample_entries = {
                relative: item for relative, item in selected.items()
                if relative.startswith(f"jobs/{job_id}/samples/")
            }
            if report_relative not in selected or not sample_entries:
                raise BackupError(f"archive job lacks evaluation or sample evidence: {job_id}")
            job_root = operation_root / shard / "jobs" / job_id
            sample_root = job_root / "samples"
            for relative, item in sorted(selected.items()):
                suffix = safe_relative_path(relative.removeprefix(prefix))
                destination = (job_root / suffix).resolve()
                if job_root.resolve() not in destination.parents:
                    raise BackupError("archive evidence destination escapes reconstructed job")
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(
                    repo_id, repo_type, str(item["remote_path"]),
                    evidence_revision, destination,
                )
                if (
                    destination.stat().st_size != int(item["size"])
                    or sha256_file(destination) != str(item["sha256"])
                ):
                    raise BackupError(f"archive evidence hash verification failed: {relative}")
            report_path = job_root / "evaluation.json"
            if report_job_id(report_path) != job_id:
                raise BackupError(f"archive report identity does not match completion: {job_id}")
            refreshed = publish_refreshed_report(
                client=client, repo_id=repo_id, repo_type=repo_type,
                run_id=run_id, report_path=report_path, sample_root=sample_root,
                work_dir=operation_root / shard / "rendered" / job_id,
                catalog_prefix=catalog_prefix, results_prefix=results_prefix,
            )
            records.append({
                **refreshed, "job_id": job_id, "archive_shard": shard,
                "archive_completion_path": completion_path,
                "archive_completion_revision": source_revision,
                "archive_evidence_revision": evidence_revision,
            })
            found_jobs.add(job_id)
    missing = requested - found_jobs
    if missing:
        raise BackupError(f"requested jobs are absent from completed archive: {sorted(missing)}")
    return {
        "schema_version": 1,
        "status": "completed",
        "archive_format": "training-archive-completion-v1",
        "run_id": run_id,
        "source_revision": source_revision,
        "jobs": records,
        "model_weights_transferred": False,
    }
