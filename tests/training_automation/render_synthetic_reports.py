"""Generate reviewable synthetic report PNGs for the remote CI artifact."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from training_automation.reporting import render_checkpoint_pages, render_top_comparison


def main() -> None:
    root = Path(os.environ.get("TRAINING_REPORT_ARTIFACT_DIR", "synthetic-report-artifact"))
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    samples = []
    prompts = [
        "frontal waist-up daylight", "strict side profile", "rear over shoulder",
        "full-body contrapposto", "side-on seated torso turn", "cross-legged floor",
        "deep squat", "one-knee kneeling", "walking opposite arm swing",
        "lunge overhead", "table-supported lean", "stairs with one foot raised",
    ]
    for index, prompt in enumerate(prompts):
        name = f"synthetic-{index:02d}.png"
        Image.new("RGB", (520 + index * 7, 780), (25 + index * 8, 55, 90)).save(samples_root / name)
        samples.append({
            "path": str(samples_root / name), "prompt_index": index,
            "prompt": prompt, "seed": 4201 + index,
            "metrics": {"clipping_fraction_proxy": index / 200},
            "identity": {"status": "available", "cosine_similarity": 0.82 - index / 100},
            "pose_body_landmarks": {
                "status": "available" if index % 4 else "occluded",
                "values": {"ratios": {"shoulder_to_hip_width": 1.15}} if index % 4 else None,
            },
        })
    checkpoints = {}
    shortlist = []
    for rank, (step, score) in enumerate(((1200, 0.81), (1100, 0.79), (1000, 0.76)), 1):
        checkpoint = {"step": step, "samples": samples}
        checkpoints[step] = checkpoint
        aggregate = {
            "step": step, "mean_face_cosine_similarity": score,
            "mean_clipping_fraction_proxy": 0.02 + rank / 100,
            "prompt_indices": list(range(12)),
        }
        shortlist.append(aggregate)
        render_checkpoint_pages(
            checkpoint, sample_root=samples_root,
            output_dir=root / f"top-{rank}",
            heading=f"0001-synthetic · top-{rank} · step {step}", aggregate=aggregate,
        )
    render_top_comparison(
        report={"automatic_shortlist": {"status": "available"}},
        sample_root=samples_root, shortlist=shortlist, checkpoints=checkpoints,
        output_path=root / "overview.png", heading="0001-synthetic · top checkpoint",
    )


if __name__ == "__main__":
    main()
