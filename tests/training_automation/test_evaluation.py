import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from training_automation import cli, evaluation
from training_automation.backup import BackupError


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
    assert "not comparable" in report["ranking"]["reason"]


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


def test_archived_verified_checkpoint_is_evaluated_without_local_weight(tmp_path):
    config_path, output = setup_job(tmp_path)
    (output / f"person-deadbeef0000_000000100.safetensors").unlink()
    state = output / ".automation" / "backup-state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({
        "schema_version": 2,
        "destination": {
            "repo_id": "owner/private", "repo_type": "dataset",
            "remote_prefix": "training-backups",
        },
        "checkpoints": {
            "qualified": {
                "status": "backed_up", "verified": True, "step": 100,
                "job_id": "person-deadbeef0000",
                "catalog_checkpoint_id": "job--step-000000100--abc123",
                "commit_id": "immutable-revision", "cataloged": True,
            }
        },
    }), encoding="utf-8")
    report = json.loads(evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    ).read_text())
    archived = report["checkpoints"][0]
    assert archived["step"] == 100
    assert archived["checkpoint_paths"] == []
    assert archived["catalog_checkpoint_id"] == "job--step-000000100--abc123"
    assert archived["remote_checkpoint"]["commit_id"] == "immutable-revision"
    assert archived["remote_association"]["status"] == "unique"


def test_differing_remote_variants_at_same_step_are_explicitly_ambiguous(tmp_path):
    config_path, output = setup_job(tmp_path)
    state = output / ".automation" / "backup-state.json"
    state.parent.mkdir()
    receipt = {
        "status": "backed_up", "verified": True, "cataloged": True,
        "step": 100, "job_id": "person-deadbeef0000",
    }
    state.write_text(json.dumps({
        "schema_version": 2,
        "destination": {
            "repo_id": "owner/private", "repo_type": "dataset",
            "remote_prefix": "training-backups",
        },
        "checkpoints": {
            "zzz-sorted-last": {
                **receipt, "catalog_checkpoint_id": "checkpoint-first",
                "commit_id": "revision-first",
                "artifacts": [{"role": "weights", "sha256": "aaa"}],
            },
            "aaa-sorted-first": {
                **receipt, "catalog_checkpoint_id": "checkpoint-second",
                "commit_id": "revision-second",
                "artifacts": [{"role": "weights", "sha256": "bbb"}],
            },
        },
    }), encoding="utf-8")
    report_path = evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    )
    report = json.loads(report_path.read_text())
    checkpoint = report["checkpoints"][0]
    assert checkpoint["remote_association"]["status"] == "ambiguous"
    assert checkpoint["catalog_checkpoint_id"] is None
    assert checkpoint["remote_checkpoint"] is None
    assert {item["catalog_checkpoint_id"] for item in checkpoint["remote_checkpoint_candidates"]} == {
        "checkpoint-first", "checkpoint-second",
    }
    with pytest.raises(BackupError, match="multiple remote checkpoint variants"):
        cli.main([
            "select", str(report_path), "100", "--note", "manual",
            "--repo-id", "owner/private", "--repo-type", "dataset",
            "--model", "person",
        ])


def test_latest_complete_duplicate_sample_run_is_used_once(tmp_path):
    config_path, output = setup_job(tmp_path)
    samples = output / "samples"
    for index in (0, 1):
        pixels = np.full((12, 12, 3), 180 + index, dtype=np.uint8)
        Image.fromarray(pixels).save(samples / f"22345__{100:09d}_{index}.png")
    report = json.loads(evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output,
        reference_images=[], config={},
    ).read_text())
    checkpoint = report["checkpoints"][0]
    assert [Path(item["path"]).name for item in checkpoint["samples"]] == [
        "22345__000000100_0.png", "22345__000000100_1.png",
    ]
    assert checkpoint["sample_run_status"] == "complete"
    assert checkpoint["discarded_duplicate_or_partial_samples"] == 2


def test_identity_ranking_uses_common_valid_prompt_subset():
    checkpoints = [
        {"step": 1, "samples": [
            {"prompt_index": 0, "identity": {"status": "available", "cosine_similarity": 0.8}},
            {"prompt_index": 1, "identity": {"status": "available", "cosine_similarity": 0.7}},
            {"prompt_index": 2, "identity": {"status": "available", "cosine_similarity": 0.6}},
            {"prompt_index": 3, "identity": {"status": "missing", "cosine_similarity": None}},
        ]},
        {"step": 2, "samples": [
            {"prompt_index": 0, "identity": {"status": "available", "cosine_similarity": 0.7}},
            {"prompt_index": 1, "identity": {"status": "available", "cosine_similarity": 0.9}},
            {"prompt_index": 2, "identity": {"status": "available", "cosine_similarity": 0.8}},
            {"prompt_index": 3, "identity": {"status": "available", "cosine_similarity": 0.9}},
        ]},
    ]
    ranked, reason = evaluation.rank_identity(checkpoints)
    assert reason is None
    assert [item["step"] for item in ranked] == [2, 1]
    assert {tuple(item["prompt_indices"]) for item in ranked} == {(0, 1, 2)}


class ReferenceFaces:
    def embeddings(self, image):
        marker = int(image[0, 0, 0])
        if marker == 0:
            return []
        if marker == 2:
            return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        if marker == 3:
            return [np.array([0.0, 0.0])]
        return [np.array([1.0, 0.0])]


def test_reference_filter_averages_only_single_valid_faces_and_records_rejections(tmp_path):
    paths = []
    for index, marker in enumerate((1, 1, 1, 0, 2, 3)):
        path = tmp_path / f"reference-{index}.png"
        Image.fromarray(np.full((2, 2, 3), marker, dtype=np.uint8)).save(path)
        paths.append(path)
    result = evaluation._reference_identity(paths, ReferenceFaces(), {
        "single_face_only": True, "minimum_valid_count": 3,
        "minimum_valid_fraction": 0.5,
    })
    assert result["status"] == "available"
    assert result["filter"]["valid"] == 3
    assert result["filter"]["coverage_fraction"] == 0.5
    assert result["filter"]["excluded_counts"] == {"missing": 1, "multiple": 1, "invalid": 1}


def test_reference_filter_rejects_insufficient_coverage(tmp_path):
    paths = []
    for index, marker in enumerate((1, 1, 0, 0, 2)):
        path = tmp_path / f"reference-{index}.png"
        Image.fromarray(np.full((2, 2, 3), marker, dtype=np.uint8)).save(path)
        paths.append(path)
    result = evaluation._reference_identity(paths, ReferenceFaces(), {
        "single_face_only": True, "minimum_valid_count": 3,
        "minimum_valid_fraction": 0.5,
    })
    assert result["status"] == "unavailable"
    assert result["filter"]["valid"] == 2


def test_reference_filter_empty_set_reports_evidence():
    result = evaluation._reference_identity([], ReferenceFaces(), {
        "single_face_only": True, "minimum_valid_count": 3,
        "minimum_valid_fraction": 0.5,
    })
    assert result["status"] == "unavailable"
    assert result["filter"]["total"] == result["filter"]["valid"] == 0


def test_shortlist_ties_use_clipping_then_step():
    checkpoints = [
        {"step": step, "samples": [{"metrics": {"clipping_fraction_proxy": clipping}}]}
        for step, clipping in ((300, 0.2), (100, 0.1), (200, 0.1), (400, 0.0))
    ]
    identity = [
        {"step": step, "mean_face_cosine_similarity": 0.8, "prompt_indices": [0, 1, 2]}
        for step in (100, 200, 300, 400)
    ]
    assert [item["step"] for item in evaluation.shortlist_checkpoints(checkpoints, identity)] == [400, 100, 200]


@pytest.mark.parametrize(
    "checkpoint,ranking_error,expected",
    [
        ({"step": 100, "sample_run_status": "incomplete", "remote_association": {"status": "unique"}}, None, "incomplete"),
        ({"step": 100, "sample_run_status": "complete", "remote_association": {"status": "ambiguous"}}, None, "unique verified"),
        ({"step": 100, "sample_run_status": "complete", "remote_association": {"status": "unique"}}, "not comparable", "not comparable"),
    ],
)
def test_shortlist_requires_complete_comparable_uniquely_associated_evidence(
    checkpoint, ranking_error, expected,
):
    reason = evaluation.shortlist_unavailable_reason(
        [checkpoint], ranking_error=ranking_error, identity_ranking_error=None
    )
    assert expected in reason


def test_evaluate_cli_processes_existing_samples_without_training(tmp_path):
    config_path, output = setup_job(tmp_path)
    assert cli.main(["evaluate", str(config_path), str(output)]) == 0
    assert (output / ".automation" / "evaluation.json").is_file()


class MutatingPoseBackend:
    def __init__(self, *, fail=False):
        self.fail = fail

    def metrics(self, image):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if self.fail:
            raise RuntimeError("pose inference failed")
        return {"status": "missing", "values": None, "reason": "test"}


@pytest.mark.parametrize("initial", [None, "", "0"])
def test_evaluate_job_restores_exact_cuda_visibility(initial, tmp_path, monkeypatch):
    if initial is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial)
    config_path, output = setup_job(tmp_path)
    monkeypatch.setattr(
        evaluation, "_load_backend",
        lambda spec, options: MutatingPoseBackend() if spec == "mutating:pose" else None,
    )
    evaluation.evaluate_job(
        job_config_path=config_path, output_dir=output, reference_images=[],
        config={"landmark_backend": "mutating:pose"},
    )
    if initial is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == initial


@pytest.mark.parametrize("initial", [None, "0"])
def test_evaluate_job_restores_cuda_visibility_when_backend_raises(
    initial, tmp_path, monkeypatch,
):
    if initial is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial)
    config_path, output = setup_job(tmp_path)
    monkeypatch.setattr(
        evaluation, "_load_backend",
        lambda spec, options: MutatingPoseBackend(fail=True) if spec == "mutating:pose" else None,
    )
    with pytest.raises(RuntimeError, match="pose inference failed"):
        evaluation.evaluate_job(
            job_config_path=config_path, output_dir=output, reference_images=[],
            config={"landmark_backend": "mutating:pose"},
        )
    if initial is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == initial


def test_backend_preflight_restores_cuda_visibility_when_loading_raises(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def failing_load(spec, options):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        raise RuntimeError("backend load failed")

    monkeypatch.setattr(evaluation, "_load_backend", failing_load)
    with pytest.raises(RuntimeError, match="backend load failed"):
        evaluation.preflight_evaluation_backends({"face_backend": "mutating:face"})
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
