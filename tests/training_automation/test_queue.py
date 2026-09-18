import json
import fcntl
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from training_automation import cli, queue as queue_module
from training_automation.queue import QueueConfigurationError, QueueJob, TrainingQueue


def write_configs(tmp_path: Path) -> Path:
    trainer = {
        "job": "extension",
        "config": {
            "name": "template",
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": str(tmp_path / "output"),
                "datasets": [{"caption_ext": "txt", "resolution": [512]}],
                "model": {"arch": "flux2_klein_9b"},
                "train": {"steps": 10},
                "save": {"save_every": 5, "max_step_saves_to_keep": 2},
                "sample": {"sample_every": 5, "seed": 5, "walk_seed": True, "samples": [{"prompt": "one"}]},
            }],
        },
    }
    (tmp_path / "trainer.yaml").write_text(yaml.safe_dump(trainer), encoding="utf-8")
    (tmp_path / "dataset").mkdir()
    automation = {
        "schema_version": 1,
        "trainer_yaml": "trainer.yaml",
        "repo_root": str(tmp_path),
        "generated_dir": "generated",
        "state_path": "queue.json",
        "checkpoint_backup": {"enabled": False},
        "evaluation": {"enabled": False},
        "datasets": [{
            "name": "Person One", "folder": "dataset", "trigger_word": "TOK",
            "reference_images": [], "destination_kind": "loras",
        }],
    }
    path = tmp_path / "automation.yaml"
    path.write_text(yaml.safe_dump(automation), encoding="utf-8")
    return path


def test_materialization_is_stable_and_injects_dataset(tmp_path):
    config = write_configs(tmp_path)
    queue = TrainingQueue(config, run_command=lambda command, env: 0)
    first = queue.materialize()
    second = queue.materialize()
    assert first[0].job_id == second[0].job_id
    generated = yaml.safe_load(first[0].config_path.read_text())
    process = generated["config"]["process"][0]
    assert generated["config"]["name"] == first[0].job_id
    assert process["trigger_word"] == "TOK"
    assert process["datasets"][0]["folder_path"] == str((tmp_path / "dataset").resolve())


def test_job_identity_includes_effective_dataset_settings_and_revision(tmp_path):
    config = write_configs(tmp_path)
    first = TrainingQueue(config).materialize()[0].job_id
    document = yaml.safe_load(config.read_text())
    document["datasets"][0]["trainer_dataset"] = {"num_repeats": 7}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    second = TrainingQueue(config).materialize()[0].job_id
    document["datasets"][0]["dataset_revision"] = "captions-v2"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    third = TrainingQueue(config).materialize()[0].job_id
    assert len({first, second, third}) == 3


def test_duration_and_checkpoint_policy_preserve_identity_and_apply_after_digest(tmp_path):
    config = write_configs(tmp_path)
    first = TrainingQueue(config).materialize()[0]
    document = yaml.safe_load(config.read_text())
    document["checkpoint_policy"] = {"save_every": 5, "max_local_step_saves": 5}
    document["datasets"][0].update({
        "training_steps": 12,
        "training_accounting": {"loader_epochs": 6, "source_image_count": 2},
    })
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    second = TrainingQueue(config).materialize()[0]
    generated = yaml.safe_load(second.config_path.read_text())
    process = generated["config"]["process"][0]
    assert second.job_id == first.job_id
    assert process["train"]["steps"] == 12
    assert process["save"] == {"save_every": 5, "max_step_saves_to_keep": 5}
    assert generated["meta"]["training_automation_schedule"]["base_training_steps"] == 10


def test_duration_change_rejects_non_pending_existing_job(tmp_path):
    config = write_configs(tmp_path)
    queue = TrainingQueue(config)
    job = queue.materialize()[0]
    (tmp_path / "queue.json").write_text(json.dumps({
        "schema_version": 2,
        "jobs": {job.job_id: {
            "status": "completed", "training_status": "completed",
            "evaluation_status": "completed", "attempts": 1,
        }},
    }), encoding="utf-8")
    document = yaml.safe_load(config.read_text())
    document["datasets"][0]["training_steps"] = 12
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="only while training is pending"):
        TrainingQueue(config).materialize()


def test_unsharded_job_keeps_legacy_digest_and_completed_state_skip(tmp_path):
    config = write_configs(tmp_path)
    document = yaml.safe_load(config.read_text())
    raw = document["datasets"][0]
    trainer_bytes = (tmp_path / "trainer.yaml").read_bytes()
    legacy_identity = {
        "template_sha256": hashlib.sha256(trainer_bytes).hexdigest(),
        "folder": str((tmp_path / raw["folder"]).resolve()),
        "trigger_word": raw.get("trigger_word"),
        "reference_images": [],
        "name": raw.get("name"),
        "trainer_dataset": {},
        "dataset_revision": None,
    }
    digest = hashlib.sha256(
        json.dumps(
            legacy_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()[:12]
    expected_job_id = f"person-one-{digest}"
    calls = []
    queue = TrainingQueue(config, run_command=lambda command, env: calls.append(command) or 0)
    assert queue.materialize()[0].job_id == expected_job_id
    (tmp_path / "queue.json").write_text(json.dumps({
        "schema_version": 2,
        "jobs": {expected_job_id: {
            "status": "completed", "training_status": "completed",
            "evaluation_status": "completed", "attempts": 1,
        }},
    }), encoding="utf-8")
    state = queue.run()
    assert state["jobs"][expected_job_id]["status"] == "completed"
    assert calls == []


def test_sharded_queue_materializes_only_assignment_with_isolated_output(tmp_path):
    config = write_configs(tmp_path)
    document = yaml.safe_load(config.read_text())
    document.update({
        "shard_id": "a",
        "training_folder": str(tmp_path / "output" / "run-1" / "a"),
    })
    first = document["datasets"][0]
    first.update({"shard_id": "a", "expected_catalog_id": 1})
    document["datasets"].append({
        **first, "name": "Person Two", "shard_id": "b",
        "expected_catalog_id": 2,
    })
    document["checkpoint_backup"] = {
        "enabled": True, "repo_id": "owner/private", "repo_type": "dataset",
    }
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    jobs = TrainingQueue(config).materialize()
    assert len(jobs) == 1
    generated = yaml.safe_load(jobs[0].config_path.read_text())
    process = generated["config"]["process"][0]
    assert process["training_folder"] == str(tmp_path / "output" / "run-1" / "a")
    assert process["checkpoint_backup"]["catalog"]["expected_id"] == 1


def test_dry_run_does_not_launch_and_resume_skips_completed(tmp_path):
    config = write_configs(tmp_path)
    calls = []
    queue = TrainingQueue(config, run_command=lambda command, env: calls.append(command) or 0)
    dry = queue.run(dry_run=True)
    assert dry["dry_run"] is True and calls == []

    state = queue.run()
    assert len(calls) == 1
    job_id = next(iter(state["jobs"]))
    assert state["jobs"][job_id]["status"] == "completed"
    queue.run()
    assert len(calls) == 1


def test_interrupted_running_job_returns_to_pending_and_runs(tmp_path):
    config = write_configs(tmp_path)
    queue = TrainingQueue(config, run_command=lambda command, env: 0)
    job_id = queue.materialize()[0].job_id
    (tmp_path / "queue.json").write_text(json.dumps({
        "schema_version": 1, "jobs": {job_id: {"status": "running", "attempts": 1}}
    }), encoding="utf-8")
    state = queue.run()
    assert state["jobs"][job_id]["status"] == "completed"
    assert state["jobs"][job_id]["attempts"] == 2
    assert state["jobs"][job_id]["interrupted"] is True


def test_evaluation_failure_retries_without_relaunching_training(tmp_path, monkeypatch):
    config = write_configs(tmp_path)
    document = yaml.safe_load(config.read_text())
    document["evaluation"] = {"enabled": True}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    launches = []
    evaluations = []

    def evaluate_once_then_succeed(**kwargs):
        evaluations.append(kwargs["job_config_path"])
        if len(evaluations) == 1:
            raise RuntimeError("evaluation interrupted")
        report = tmp_path / "evaluation.json"
        report.write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setattr(queue_module, "evaluate_job", evaluate_once_then_succeed)
    queue = TrainingQueue(config, run_command=lambda command, env: launches.append(command) or 0)
    failed = queue.run()
    entry = next(iter(failed["jobs"].values()))
    assert entry["training_status"] == "completed"
    assert entry["evaluation_status"] == "failed"
    completed = queue.run()
    entry = next(iter(completed["jobs"].values()))
    assert entry["status"] == "completed"
    assert len(launches) == 1
    assert len(evaluations) == 2


@pytest.mark.parametrize("initial", [None, "0"])
def test_two_sequential_jobs_inherit_original_gpu_visibility_after_mutating_evaluation(
    initial, tmp_path, monkeypatch,
):
    if initial is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial)
    config = write_configs(tmp_path)
    document = yaml.safe_load(config.read_text())
    document["evaluation"] = {"enabled": True}
    document["datasets"].append({**document["datasets"][0], "name": "Person Two"})
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    trainer_visibility = []

    def launch(command, env):
        trainer_visibility.append(("CUDA_VISIBLE_DEVICES" in env, env.get("CUDA_VISIBLE_DEVICES")))
        return 0

    def mutating_evaluation(**kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        report = tmp_path / f"evaluation-{len(trainer_visibility)}.json"
        report.write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setattr(queue_module, "evaluate_job", mutating_evaluation)
    state = TrainingQueue(config, run_command=launch).run()
    expected = (initial is not None, initial)
    assert trainer_visibility == [expected, expected]
    assert {entry["status"] for entry in state["jobs"].values()} == {"completed"}
    if initial is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == initial


def test_queue_restores_gpu_visibility_when_evaluation_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    config = write_configs(tmp_path)
    document = yaml.safe_load(config.read_text())
    document["evaluation"] = {"enabled": True}
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    def failing_evaluation(**kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(queue_module, "evaluate_job", failing_evaluation)
    state = TrainingQueue(config, run_command=lambda command, env: 0).run()
    assert next(iter(state["jobs"].values()))["evaluation_status"] == "failed"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_queue_lock_refuses_second_launcher(tmp_path):
    config = write_configs(tmp_path)
    queue = TrainingQueue(config, run_command=lambda command, env: 0)
    lock_path = tmp_path / "queue.json.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            queue.run()


def test_cli_returns_nonzero_for_failed_job(tmp_path, monkeypatch):
    class FailedQueue:
        def __init__(self, config):
            pass

        def run(self, *, dry_run=False):
            return {"jobs": {"job": {"status": "failed"}}}

    monkeypatch.setattr(cli, "TrainingQueue", FailedQueue)
    assert cli.main(["run", str(tmp_path / "automation.yaml")]) == 1


def refinement_process(*, sample_every, save_every, steps=1200):
    return {
        "train": {"steps": steps},
        "save": {"save_every": save_every},
        "sample": {
            "sample_every": sample_every,
            "samples": [{"prompt": "single adult subject [trigger]"}],
        },
    }


def refinement_job(tmp_path):
    return QueueJob(
        job_id="subject-0-job",
        config_path=tmp_path / "subject-0-job.yaml",
        output_root=tmp_path / "output",
        reference_images=(),
    )


def test_refinement_preflight_refuses_a_save_cadence_out_of_step_with_sampling(tmp_path):
    """The pre-flight is the only place this costs nothing to catch.

    The archive gate reaches the same verdict, but only after the whole paid
    run is over. A refinement that samples every 100 steps and saves every 200
    has to be refused before the first GPU hour.
    """
    with pytest.raises(QueueConfigurationError, match="must be the same cadence"):
        TrainingQueue._assert_phase_evidence_reachable(
            refinement_job(tmp_path),
            refinement_process(sample_every=100, save_every=200),
            {"base_training_steps": 600},
        )


def test_refinement_preflight_refuses_an_unsupported_cadence(tmp_path):
    """A self-consistent pair is still refused when the cadence is not declared."""
    with pytest.raises(QueueConfigurationError, match="must be the same cadence"):
        TrainingQueue._assert_phase_evidence_reachable(
            refinement_job(tmp_path),
            refinement_process(sample_every=150, save_every=150),
            {"base_training_steps": 600},
        )


def test_refinement_preflight_accepts_the_shipped_cadence(tmp_path):
    """The control above must fail for the cadence, not for the fixture."""
    TrainingQueue._assert_phase_evidence_reachable(
        refinement_job(tmp_path),
        refinement_process(sample_every=100, save_every=100),
        {"base_training_steps": 600},
    )
