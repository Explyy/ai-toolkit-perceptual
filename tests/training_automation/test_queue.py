import json
from pathlib import Path

import yaml

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

