from __future__ import annotations

import base64
import html
import io
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PIL import Image

from .backup import BackupError


PREVIEW_SIZE = (420, 420)
IMAGE_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return _text(value)


def _sample_basename(value: Any) -> str:
    raw = str(value or "").replace("\\", "/")
    name = PurePosixPath(raw).name
    if not name or name in {".", ".."}:
        raise BackupError("gallery sample path has no safe basename")
    return name


def _resolve_sample(sample_root: Path, value: Any) -> tuple[Path, str]:
    name = _sample_basename(value)
    root = sample_root.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise BackupError(f"gallery sample is missing or escapes sample root: {name}")
    return candidate, name


def _image_data(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            media_type = IMAGE_MEDIA_TYPES.get(str(image.format).upper())
            if media_type is None:
                raise BackupError(f"gallery sample uses unsupported image format: {path.name}")
            preview = image.convert("RGB")
            preview.thumbnail(PREVIEW_SIZE)
            encoded_preview = io.BytesIO()
            preview.save(encoded_preview, format="JPEG", quality=85, optimize=True)
    except BackupError:
        raise
    except Exception as exc:
        raise BackupError(f"gallery sample is not a readable image: {path.name}") from exc
    original = base64.b64encode(payload).decode("ascii")
    thumbnail = base64.b64encode(encoded_preview.getvalue()).decode("ascii")
    return (
        f"data:image/jpeg;base64,{thumbnail}",
        f"data:{media_type};base64,{original}",
    )


def _flatten_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return [(prefix or "value", value)]
    flattened = []
    for key in sorted(value, key=str):
        label = f"{prefix}.{key}" if prefix else str(key)
        nested = value[key]
        if isinstance(nested, Mapping):
            flattened.extend(_flatten_values(nested, label))
        else:
            flattened.append((label, nested))
    return flattened


def _metric(label: str, value: Any) -> str:
    return f"<dt>{_text(label)}</dt><dd>{_number(value)}</dd>"


def _status_metric(label: str, document: Mapping[str, Any], value_key: str) -> str:
    status = str(document.get("status", "unavailable"))
    value = document.get(value_key)
    if status == "available" and value is not None:
        rendered = _number(value)
    else:
        reason = document.get("reason")
        rendered = _text(status + (f": {reason}" if reason else ""))
    return f"<dt>{_text(label)}</dt><dd>{rendered}</dd>"


def _job_label(report: Mapping[str, Any], report_path: Path) -> str:
    configured = str(report.get("job_config") or "")
    if configured:
        return Path(configured).stem
    if report_path.parent.name == ".automation":
        return report_path.parent.parent.name
    return report_path.parent.name


def _checkpoint_order(report: Mapping[str, Any]) -> dict[int, int]:
    ranking = report.get("identity_ranking") or {}
    if ranking.get("status") != "available":
        return {}
    return {
        int(item["step"]): index
        for index, item in enumerate(ranking.get("items") or [])
        if isinstance(item, Mapping) and "step" in item
    }


def _identity_aggregates(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(item["step"]): item
        for item in (report.get("identity_ranking") or {}).get("items") or []
        if isinstance(item, Mapping) and "step" in item
    }


def _shortlist(report: Mapping[str, Any]) -> str:
    shortlist = report.get("automatic_shortlist") or {}
    status = str(shortlist.get("status", "unavailable"))
    if status != "available":
        reason = shortlist.get("reason") or "no reason recorded"
        return (
            '<section class="shortlist unavailable"><h2>Automatic evidence shortlist unavailable</h2>'
            f"<p>{_text(reason)}</p></section>"
        )
    items = shortlist.get("items") or []
    rows = []
    for rank, item in enumerate(items, 1):
        rows.append(
            "<li>"
            f"#{rank} · step {_text(item.get('step'))} · raw mean face cosine "
            f"{_number(item.get('mean_face_cosine_similarity'))} · mean clipping proxy "
            f"{_number(item.get('mean_clipping_fraction_proxy'))}"
            "</li>"
        )
    # An available shortlist that ranked only part of the run must say which
    # checkpoints never competed, otherwise a partial top three reads as if it
    # had beaten every checkpoint of the job.
    excluded = [item for item in shortlist.get("excluded") or [] if isinstance(item, Mapping)]
    partial = ""
    if excluded:
        entries = "".join(
            "<li>"
            f"step {_text(item.get('step'))} · {_text(item.get('reason') or 'no reason recorded')}"
            "</li>"
            for item in excluded
        )
        partial = (
            '<p class="caveat">Partial coverage: these checkpoints were not eligible '
            "for the ranking and did not compete.</p>"
            f"<ul class=\"shortlist-excluded\">{entries}</ul>"
        )
    return (
        '<section class="shortlist"><h2>Automatic evidence shortlist</h2>'
        '<p class="caveat">This evidence ordering does not select a model. A final checkpoint is not necessarily the highest-ranked checkpoint.</p>'
        f"<ol>{''.join(rows)}</ol>{partial}</section>"
    )


def _sample_card(
    sample: Mapping[str, Any], *, sample_root: Path, seen_names: set[str]
) -> str:
    path, name = _resolve_sample(sample_root, sample.get("path"))
    if name in seen_names:
        raise BackupError(f"gallery report references duplicate sample basename: {name}")
    seen_names.add(name)
    thumbnail, original = _image_data(path)
    identity = sample.get("identity") or {}
    metrics = sample.get("metrics") or {}
    pose = sample.get("pose_body_landmarks") or {}
    pose_status = str(pose.get("status", "unavailable"))
    if pose_status == "available" and isinstance(pose.get("values"), Mapping):
        pose_values = "".join(
            _metric(f"pose.{label}", value)
            for label, value in _flatten_values(pose["values"])
        )
    else:
        reason = pose.get("reason")
        pose_values = _metric(
            "pose status", pose_status + (f": {reason}" if reason else "")
        )
    prompt = sample.get("prompt")
    return f"""
      <article class="sample">
        <a class="preview" href="{original}" target="_blank" rel="noopener">
          <img src="{thumbnail}" alt="Sample {_text(name)}">
        </a>
        <div class="sample-data">
          <h4>Prompt {_text(sample.get('prompt_index'))} · seed {_text(sample.get('seed'))}</h4>
          <p class="prompt">{_text(prompt if prompt is not None else 'prompt unavailable')}</p>
          <dl>
            {_status_metric('raw face cosine', identity, 'cosine_similarity')}
            {_metric('clipping fraction proxy', metrics.get('clipping_fraction_proxy'))}
            {_metric('sharpness proxy', metrics.get('sharpness_laplacian_variance_proxy'))}
            {pose_values}
          </dl>
        </div>
      </article>
    """


def _checkpoint_card(
    checkpoint: Mapping[str, Any],
    *,
    aggregate: Mapping[str, Any] | None,
    aggregate_rank: int | None,
    sample_root: Path,
    seen_names: set[str],
) -> str:
    step = int(checkpoint.get("step", 0))
    samples = checkpoint.get("samples") or []
    clipping = [
        sample.get("metrics", {}).get("clipping_fraction_proxy")
        for sample in samples
        if sample.get("metrics", {}).get("clipping_fraction_proxy") is not None
    ]
    mean_clipping = (
        sum(float(value) for value in clipping) / len(clipping)
        if clipping
        else None
    )
    remote = checkpoint.get("remote_checkpoint") or {}
    association = checkpoint.get("remote_association") or {}
    stable = []
    if checkpoint.get("catalog_checkpoint_id"):
        stable.append(
            _metric("catalog checkpoint ID", checkpoint["catalog_checkpoint_id"])
        )
    if remote.get("commit_id"):
        stable.append(_metric("immutable Hub revision", remote["commit_id"]))
    heading = f"Step {step}"
    if checkpoint.get("final"):
        heading += " · final save"
    rank_label = (
        f"Identity aggregate rank #{aggregate_rank}"
        if aggregate_rank
        else "Identity aggregate rank unavailable"
    )
    common = aggregate.get("prompt_indices") if aggregate else None
    cards = "".join(
        _sample_card(sample, sample_root=sample_root, seen_names=seen_names)
        for sample in samples
    )
    return f"""
    <section class="checkpoint">
      <header>
        <div><h3>{_text(heading)}</h3><p>{_text(rank_label)}</p></div>
        <dl class="aggregate">
          {_metric('raw mean face cosine', aggregate.get('mean_face_cosine_similarity') if aggregate else None)}
          {_metric('common valid-face prompt indices', ', '.join(map(str, common)) if common else None)}
          {_metric('mean clipping fraction proxy', mean_clipping)}
          {_metric('sample run status', checkpoint.get('sample_run_status'))}
          {_metric('remote association', association.get('status'))}
          {''.join(stable)}
        </dl>
      </header>
      <div class="samples">{cards}</div>
    </section>
    """


def render_gallery(
    report_path: Path,
    *,
    sample_root: Path,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Render one evaluation report without inference or remote asset access."""
    report_path = report_path.resolve()
    sample_root = sample_root.resolve()
    if not report_path.is_file():
        raise BackupError(f"gallery evaluation report is missing: {report_path}")
    if not sample_root.is_dir():
        raise BackupError(f"gallery sample root is missing: {sample_root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise BackupError("gallery evaluation report must be a JSON object")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise BackupError("gallery evaluation report has no checkpoints")
    order = _checkpoint_order(report)
    aggregates = _identity_aggregates(report)
    ordered = sorted(
        checkpoints,
        key=lambda item: (
            order.get(int(item.get("step", 0)), len(order)),
            int(item.get("step", 0)),
        ),
    )
    label = _job_label(report, report_path)
    page_title = title or f"Evaluation gallery · {label}"
    seen_names: set[str] = set()
    checkpoint_html = "".join(
        _checkpoint_card(
            checkpoint,
            aggregate=aggregates.get(int(checkpoint.get("step", 0))),
            aggregate_rank=(
                order[int(checkpoint["step"])] + 1
                if int(checkpoint.get("step", 0)) in order
                else None
            ),
            sample_root=sample_root,
            seen_names=seen_names,
        )
        for checkpoint in ordered
    )
    identity = report.get("identity_ranking") or {}
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_text(page_title)}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#0c0e12; color:#eef1f5; }}
body {{ margin:0; padding:28px; }} main {{ max-width:1500px; margin:auto; }}
h1,h2,h3,h4,p {{ margin-top:0; }} .caveat {{ color:#aeb6c3; }}
.shortlist,.checkpoint {{ background:#151922; border:1px solid #2b3342; border-radius:14px; padding:18px; margin:18px 0; }}
.unavailable {{ border-color:#765d30; }} .checkpoint>header {{ display:flex; gap:28px; justify-content:space-between; flex-wrap:wrap; }}
.aggregate {{ min-width:360px; }} dl {{ display:grid; grid-template-columns:minmax(150px,auto) 1fr; gap:5px 12px; margin:0; }}
dt {{ color:#9ba8bb; }} dd {{ margin:0; overflow-wrap:anywhere; }} .samples {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px; margin-top:18px; }}
.sample {{ display:grid; grid-template-columns:minmax(140px,max-content) minmax(0,1fr); gap:14px; padding:12px; background:#0f131a; border-radius:10px; align-items:start; }}
.preview img {{ display:block; max-width:420px; width:100%; height:auto; border-radius:8px; }}
.prompt {{ min-height:2.5em; }}
@media(max-width:760px) {{ body {{ padding:12px; }} .sample {{ grid-template-columns:1fr; }} .preview img {{ width:auto; max-width:100%; max-height:520px; }} .aggregate {{ min-width:0; }} }}
</style></head><body><main>
<h1>{_text(page_title)}</h1>
<p class="caveat">Raw cosine is not a percentage. Image and pose metrics are evidence proxies, not quality or anatomy scores. Open any preview for its embedded original.</p>
{_shortlist(report)}
<section class="shortlist"><h2>Checkpoint identity ordering</h2><p>Status: {_text(identity.get('status', 'unavailable'))}. {_text(identity.get('reason') or identity.get('method') or '')}</p></section>
{checkpoint_html}
</main></body></html>"""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
