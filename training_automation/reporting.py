from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .backup import BackupError
from .state import atomic_write_json


PAGE_WIDTH = 1920
PAGE_MARGIN = 42
PAGE_GAP = 24
PAGE_COLUMNS = 3
PAGE_ROWS = 2
SAMPLES_PER_PAGE = PAGE_COLUMNS * PAGE_ROWS
CARD_WIDTH = (PAGE_WIDTH - 2 * PAGE_MARGIN - 2 * PAGE_GAP) // PAGE_COLUMNS
IMAGE_MAX = (CARD_WIDTH - 28, 400)
BODY_SIZE = 28
SMALL_SIZE = 24
TITLE_SIZE = 42
OVERVIEW_WIDTH = 1600

COLORS = {
    "background": "#0b0e13",
    "surface": "#141a24",
    "line": "#334055",
    "text": "#f2f4f7",
    "muted": "#aeb8c8",
    "accent": "#72ddc3",
    "warm": "#f0c875",
    "bar_track": "#293444",
}


@dataclass(frozen=True)
class CheckpointReport:
    entry_path: Path
    index_path: Path
    page_paths: tuple[Path, ...]


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        pass
    raise BackupError(f"evaluation report requires a scalable font at {size}px")


def _number(value: Any) -> str:
    if value is None:
        return "non disponibile"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if math.isfinite(number) else "non disponibile"


def _percent(value: Any) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return "non disponibile"
    return f"{number:.1f}%" if math.isfinite(number) else "non disponibile"


def _width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text or " ", font=font)
    return box[2] - box[0]


def wrap_pixels(
    draw: ImageDraw.ImageDraw,
    text: Any,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text).replace("\r", " ").replace("\n", " ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        fragment = ""
        for character in word:
            candidate = fragment + character
            if fragment and _width(draw, candidate, font) > max_width:
                lines.append(fragment)
                fragment = character
            else:
                fragment = candidate
        current = fragment
    if current:
        lines.append(current)
    return lines


def _line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return box[3] - box[1]


def _safe_sample(sample_root: Path, sample: Mapping[str, Any]) -> Path:
    name = Path(str(sample.get("path") or "").replace("\\", "/")).name
    root = sample_root.resolve()
    path = (root / name).resolve()
    if not name or path.parent != root or not path.is_file():
        raise BackupError(f"report sample is missing or escapes sample root: {name}")
    return path


def _pose_line(sample: Mapping[str, Any]) -> str:
    pose = sample.get("pose_body_landmarks") or {}
    status = str(pose.get("status", "unavailable"))
    if status != "available":
        labels = {
            "occluded": "coperta",
            "missing": "non rilevata",
            "ambiguous": "ambigua",
            "unavailable": "non disponibile",
            "degenerate": "non valida",
        }
        return f"Posa: {labels.get(status, status)}"
    ratios = ((pose.get("values") or {}).get("ratios") or {})
    shoulder = ratios.get("shoulder_to_hip_width")
    if shoulder is None:
        return "Posa: disponibile"
    return f"Posa: disponibile · spalle/fianchi {_number(shoulder)}"


def _metric_lines(sample: Mapping[str, Any]) -> list[str]:
    identity = sample.get("identity") or {}
    metrics = sample.get("metrics") or {}
    face = (
        _number(identity.get("cosine_similarity"))
        if identity.get("status") == "available"
        else str(identity.get("status", "unavailable"))
    )
    return [
        f"Prompt {sample.get('prompt_index')} · seed {sample.get('seed')}",
        f"Somiglianza volto: {face} · Pixel saturi: {_percent(metrics.get('clipping_fraction_proxy'))}",
        _pose_line(sample),
    ]


def _sample_card(sample: Mapping[str, Any], sample_root: Path) -> Image.Image:
    path = _safe_sample(sample_root, sample)
    try:
        with Image.open(path) as opened:
            opened.load()
            media = opened.convert("RGB")
    except Exception as exc:
        raise BackupError(f"report sample is not a readable image: {path.name}") from exc
    media.thumbnail(IMAGE_MAX)
    body = _font(BODY_SIZE)
    small = _font(SMALL_SIZE)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[tuple[str, ImageFont.ImageFont, str]] = []
    for index, value in enumerate(_metric_lines(sample)):
        color = COLORS["text"] if index == 0 else COLORS["accent"] if index == 1 else COLORS["warm"]
        font = body if index < 2 else small
        lines.extend((line, font, color) for line in wrap_pixels(
            probe, value, font=font, max_width=CARD_WIDTH - 28
        ))
    prompt_lines = wrap_pixels(
        probe,
        f"Testo: {sample.get('prompt') or 'non disponibile'}",
        font=small,
        max_width=CARD_WIDTH - 28,
    )[:2]
    lines.extend((line, small, COLORS["muted"]) for line in prompt_lines)
    text_height = sum(_line_height(font) + 7 for _, font, _ in lines)
    height = 14 + media.height + 16 + text_height + 14
    card = Image.new("RGB", (CARD_WIDTH, height), COLORS["surface"])
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, card.width - 1, card.height - 1), outline=COLORS["line"], width=2)
    card.paste(media, ((CARD_WIDTH - media.width) // 2, 14))
    y = 14 + media.height + 16
    for line, font, color in lines:
        draw.text((14, y), line, fill=color, font=font)
        y += _line_height(font) + 7
    return card


def render_checkpoint_pages(
    checkpoint: Mapping[str, Any],
    *,
    sample_root: Path,
    output_dir: Path,
    heading: str,
    aggregate: Mapping[str, Any],
) -> CheckpointReport:
    samples = sorted(
        checkpoint.get("samples") or [],
        key=lambda item: (int(item.get("prompt_index", -1)), int(item.get("seed", -1))),
    )
    if not samples:
        raise BackupError("checkpoint report has no samples")
    signatures = [(item.get("prompt_index"), item.get("seed")) for item in samples]
    if len(signatures) != len(set(signatures)):
        raise BackupError("checkpoint report contains duplicate prompt/seed samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    title_font = _font(TITLE_SIZE)
    body_font = _font(BODY_SIZE)
    page_paths = []
    for page_index, start in enumerate(range(0, len(samples), SAMPLES_PER_PAGE), 1):
        cards = [_sample_card(sample, sample_root) for sample in samples[start:start + SAMPLES_PER_PAGE]]
        row_heights = [
            max(card.height for card in cards[row:row + PAGE_COLUMNS])
            for row in range(0, len(cards), PAGE_COLUMNS)
        ]
        header_height = 150
        height = header_height + sum(row_heights) + (len(row_heights) + 1) * PAGE_GAP
        if height > 1600:
            raise BackupError("checkpoint report page exceeded its 1920x1600 layout bound")
        page = Image.new("RGB", (PAGE_WIDTH, height), COLORS["background"])
        draw = ImageDraw.Draw(page)
        draw.text((PAGE_MARGIN, 28), heading, fill=COLORS["text"], font=title_font)
        draw.text(
            (PAGE_MARGIN, 86),
            "Somiglianza volto media "
            f"{_number(aggregate.get('mean_face_cosine_similarity'))} · "
            f"Pixel saturi medi {_percent(aggregate.get('mean_clipping_fraction_proxy'))} · "
            f"pagina {page_index}/{math.ceil(len(samples) / SAMPLES_PER_PAGE)}",
            fill=COLORS["muted"],
            font=body_font,
        )
        y = header_height
        for row_index, row_height in enumerate(row_heights):
            for column, card in enumerate(
                cards[row_index * PAGE_COLUMNS:(row_index + 1) * PAGE_COLUMNS]
            ):
                x = PAGE_MARGIN + column * (CARD_WIDTH + PAGE_GAP)
                page.paste(card, (x, y))
            y += row_height + PAGE_GAP
        page_path = pages_dir / f"page-{page_index:02d}.png"
        page.save(page_path, format="PNG", optimize=True)
        page_paths.append(page_path)
    entry_path = output_dir / "contact-sheet.png"
    shutil.copyfile(page_paths[0], entry_path)
    index_path = output_dir / "report-index.json"
    atomic_write_json(index_path, {
        "schema_version": 2,
        "sample_count": len(samples),
        "samples_per_page": SAMPLES_PER_PAGE,
        "page_count": len(page_paths),
        "pages": [f"pages/{path.name}" for path in page_paths],
        "entry_preview": entry_path.name,
        "checkpoint_aggregate": dict(aggregate),
    })
    return CheckpointReport(entry_path, index_path, tuple(page_paths))


def _common_preview_prompt(shortlist: Sequence[Mapping[str, Any]]) -> int | None:
    sets = [set(item.get("prompt_indices") or []) for item in shortlist]
    common = set.intersection(*sets) if sets else set()
    return min(common) if common else None


def cosine_axis_ticks(left: int, right: int) -> tuple[tuple[str, int], ...]:
    if right <= left:
        raise ValueError("cosine axis requires positive width")
    return (("-1", left), ("0", left + (right - left) // 2), ("+1", right))


def render_top_comparison(
    *,
    report: Mapping[str, Any],
    sample_root: Path,
    shortlist: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
    output_path: Path,
    heading: str,
) -> Path:
    title = _font(TITLE_SIZE)
    body = _font(BODY_SIZE)
    small = _font(SMALL_SIZE)
    prompt_index = _common_preview_prompt(shortlist)
    cards = []
    for rank, aggregate in enumerate(shortlist, 1):
        checkpoint = checkpoints[int(aggregate["step"])]
        preview = next(
            (
                sample for sample in checkpoint.get("samples") or []
                if sample.get("prompt_index") == prompt_index
            ),
            (checkpoint.get("samples") or [None])[0],
        )
        if preview is None:
            raise BackupError("top comparison checkpoint has no preview sample")
        path = _safe_sample(sample_root, preview)
        with Image.open(path) as opened:
            opened.load()
            media = opened.convert("RGB")
        media.thumbnail((450, 390))
        cards.append((rank, aggregate, checkpoint, media))
    height = 780
    canvas = Image.new("RGB", (OVERVIEW_WIDTH, height), COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 30), heading, fill=COLORS["text"], font=title)
    draw.text(
        (42, 88),
        "Somiglianza volto · coseno da -1 a +1; più alto = più simile (non è una probabilità)",
        fill=COLORS["muted"],
        font=small,
    )
    if not cards:
        reason = (report.get("automatic_shortlist") or {}).get("reason") or "non disponibile"
        draw.text((42, 180), f"Top checkpoint non disponibili: {reason}", fill=COLORS["warm"], font=body)
    column_width = (OVERVIEW_WIDTH - 84 - 2 * 24) // 3
    for index, (rank, aggregate, checkpoint, media) in enumerate(cards):
        x = 42 + index * (column_width + 24)
        draw.text(
            (x, 145),
            f"TOP {rank} · step {checkpoint['step']}",
            fill=COLORS["text"],
            font=body,
        )
        canvas.paste(media, (x + (column_width - media.width) // 2, 195))
        y = 570
        cosine = float(aggregate["mean_face_cosine_similarity"])
        track_left, track_right = x, x + column_width
        draw.rectangle((track_left, y, track_right, y + 22), fill=COLORS["bar_track"])
        zero = track_left + column_width // 2
        value_x = track_left + round((max(-1.0, min(1.0, cosine)) + 1.0) * column_width / 2)
        draw.rectangle((min(zero, value_x), y, max(zero, value_x), y + 22), fill=COLORS["accent"])
        draw.line((zero, y - 5, zero, y + 27), fill=COLORS["text"], width=2)
        for label, tick_x in cosine_axis_ticks(track_left, track_right):
            label_width = _width(draw, label, small)
            draw.text((tick_x - label_width // 2, y + 30), label, fill=COLORS["muted"], font=small)
        draw.text((x, y + 64), f"Somiglianza volto: {_number(cosine)}", fill=COLORS["accent"], font=small)
        draw.text(
            (x, y + 98),
            f"Pixel saturi: {_percent(aggregate.get('mean_clipping_fraction_proxy'))} · "
            f"copertura: {len(aggregate.get('prompt_indices') or [])}/{len(checkpoint.get('samples') or [])}",
            fill=COLORS["muted"],
            font=small,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path
