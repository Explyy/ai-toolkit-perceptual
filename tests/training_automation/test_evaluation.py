import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from training_automation import evaluation


def setup_job(tmp_path: Path):
    job_id = "person-deadbeef0000"
    config = {
        "job": "extension",
        "config": {"name": job_id, "process": [{
            "train": {"steps": 200},
            "sample": {
                "seed": 10, "walk_seed": True,
                "samples": [{"prompt": "p0"}, {"prompt": "p1"}],
            },
        }]},
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / job_id
    samples = output / "samples"
    samples.mkdir(parents=True)
    (output / f"{job_id}_000000100.safetensors").write_bytes(b"one")
    (output / f"{job_id}.safetensors").write_bytes(b"final")
    for step in (100, 200):
        for index in (0, 1):
            pixels = np.full((12, 12, 3), 50 + step // 10 + index, dtype=np.uint8)
            Image.fromarray(pixels).save(samples / f"12345__{step:09d}_{index}.png")
    return config_path, output


def test_associates_only_checkpoint_steps_including_final_and_ranks(tmp_path):
    config_path, output = setup_job(tmp_path)
    report_path = evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    )
    report = json.loads(report_path.read_text())
    assert [item["step"] for item in report["checkpoints"]] == [100, 200]
    assert report["checkpoints"][1]["final"] is True
    assert report["ranking"]["status"] == "available"
    assert "heuristic" in report["ranking"]["method"]
    sample = report["checkpoints"][0]["samples"][0]
    assert sample["seed"] == 10
    assert sample["identity"]["status"] == "unavailable"
    assert sample["identity"]["cosine_similarity"] is None
    assert sample["pose_body_landmarks"]["status"] == "unavailable"


def test_incomplete_prompt_seed_set_disables_ranking(tmp_path):
    config_path, output = setup_job(tmp_path)
    (output / "samples" / "12345__000000200_1.png").unlink()
    report = json.loads(evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    ).read_text())
    assert report["ranking"]["status"] == "unavailable"
    assert "complete configured" in report["ranking"]["reason"]


class Faces:
    def __init__(self, count):
        self.count = count

    def embeddings(self, image):
        return [np.array([1.0, 0.0])] * self.count


def test_missing_and_ambiguous_faces_never_create_similarity():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    reference = {"status": "available", "embedding": np.array([1.0, 0.0]), "reason": None}
    missing = evaluation._identity_metric(image, Faces(0), reference)
    ambiguous = evaluation._identity_metric(image, Faces(2), reference)
    assert missing == {"status": "missing", "cosine_similarity": None, "reason": "no face detected"}
    assert ambiguous["status"] == "ambiguous" and ambiguous["cosine_similarity"] is None


def test_human_selection_persists_checkpoint_evidence(tmp_path):
    config_path, output = setup_job(tmp_path)
    report = evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    )
    selection = evaluation.persist_selection(report, 200, "manual review")
    content = json.loads(selection.read_text())
    assert content["selected_step"] == 200
    assert content["evidence"]["final"] is True

