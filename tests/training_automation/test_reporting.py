import json
from pathlib import Path

from PIL import Image

from training_automation.reporting import (
    BODY_SIZE,
    PAGE_WIDTH,
    SAMPLES_PER_PAGE,
    TITLE_SIZE,
    _metric_lines,
    cosine_axis_ticks,
    render_checkpoint_pages,
    render_top_comparison,
)


def _samples(root: Path, count: int = 23):
    root.mkdir(parents=True)
    values = []
    for index in range(count):
        name = f"sample-{index:02d}.png"
        Image.new("RGB", (240 + index, 360), (10 + index, 30, 50)).save(root / name)
        values.append({
            "path": f"/deleted/pod/{name}", "prompt_index": index,
            "prompt": f"articulated prompt {index}", "seed": 4200 + index,
            "metrics": {"clipping_fraction_proxy": index / 100},
            "identity": {"status": "available", "cosine_similarity": 0.7 + index / 1000},
            "pose_body_landmarks": {
                "status": "available",
                "values": {"ratios": {"shoulder_to_hip_width": 1.1}},
            },
        })
    return values


def test_paginated_report_is_image_first_bounded_and_keeps_all_samples(tmp_path):
    samples = _samples(tmp_path / "samples")
    rendered = render_checkpoint_pages(
        {"step": 100, "samples": samples}, sample_root=tmp_path / "samples",
        output_dir=tmp_path / "report", heading="0001-person · step 100",
        aggregate={"mean_face_cosine_similarity": 0.75,
                   "mean_clipping_fraction_proxy": 0.03,
                   "prompt_indices": list(range(20))},
    )
    index = json.loads(rendered.index_path.read_text())
    assert BODY_SIZE >= 28 and TITLE_SIZE >= 40
    assert SAMPLES_PER_PAGE <= 6
    assert index["sample_count"] == 23
    assert index["page_count"] == 4
    assert len(rendered.page_paths) == 4
    for page in rendered.page_paths:
        with Image.open(page) as image:
            image.load()
            assert image.width == PAGE_WIDTH
            assert image.height <= 1600
    assert rendered.entry_path.read_bytes() == rendered.page_paths[0].read_bytes()


def test_top_overview_compares_same_prompt_with_raw_cosine_axis(tmp_path):
    samples = _samples(tmp_path / "samples", count=3)
    checkpoints = {
        step: {"step": step, "samples": samples} for step in (100, 200, 300)
    }
    shortlist = [
        {"step": step, "mean_face_cosine_similarity": score,
         "mean_clipping_fraction_proxy": 0.01, "prompt_indices": [0, 1]}
        for step, score in ((200, 0.8), (100, 0.7), (300, 0.6))
    ]
    output = render_top_comparison(
        report={"automatic_shortlist": {"status": "available"}},
        sample_root=tmp_path / "samples", shortlist=shortlist,
        checkpoints=checkpoints, output_path=tmp_path / "overview.png",
        heading="Top checkpoint",
    )
    with Image.open(output) as image:
        image.load()
        assert image.width == 1600 and image.height <= 800
    assert cosine_axis_ticks(10, 110) == (("-1", 10), ("0", 60), ("+1", 110))


def test_visible_metric_labels_are_interpretable_italian_without_fake_scores():
    lines = _metric_lines({
        "prompt_index": 1, "seed": 42,
        "identity": {"status": "available", "cosine_similarity": 0.8123},
        "metrics": {"clipping_fraction_proxy": 0.025},
        "pose_body_landmarks": {"status": "occluded"},
    })
    assert lines == [
        "Prompt 1 · seed 42",
        "Somiglianza volto: 0.8123 · Pixel saturi: 2.5%",
        "Posa: coperta",
    ]
