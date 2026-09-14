import json
import fcntl
from pathlib import Path

import pytest
import yaml

from training_automation import cli, queue as queue_module
from training_automation.queue import TrainingQueue


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
                "sample": {"seed": 5, "walk_seed": True, "samples": [{"prompt": "one"}]},
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
