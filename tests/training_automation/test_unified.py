import importlib.util
import json
from pathlib import Path

import yaml
from PIL import Image

from training_automation.queue import TrainingQueue
from training_automation.state import atomic_write_json
from training_automation.unified import (
    resolve_dataset_root,
    resolve_loras_root,
    run_unified_workflow,
)


FAKE_SPEC = importlib.util.spec_from_file_location(
    "unified_fakes", Path(__file__).with_name("fakes.py")
)
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient


class FakeQueue:
    def __init__(self, path):
        self.inner = TrainingQueue(path)
        self.state_path = self.inner.state_path

    def materialize(self):
        return self.inner.materialize()

    def run(self):
        jobs = self.materialize()
        state = {"schema_version": 2, "jobs": {}}
        for job in jobs:
            output = job.output_root / job.job_id
            (output / ".automation").mkdir(parents=True, exist_ok=True)
            (output / "samples").mkdir(parents=True, exist_ok=True)
            (output / ".automation" / "evaluation.json").write_text("{}", encoding="utf-8")
            Image.new("RGB", (8, 8), (1, 2, 3)).save(output / "samples" / "sample.png")
            state["jobs"][job.job_id] = {
                "status": "completed", "training_status": "completed",
                "evaluation_status": "completed",
            }
        atomic_write_json(self.state_path, state)
        return state


class FakeArchive:
    def __init__(self, **kwargs):
        pass

    def publish(self, files, completion):
        assert files and completion["status"] == "completed"
        return {"status": "completed", "completion_revision": "archive-revision"}


def _dataset(root: Path, name: str):
    folder = root / name
    folder.mkdir(parents=True)
    Image.new("RGB", (640, 960), (10, 20, 30)).save(folder / "one.png")
    (folder / "one.txt").write_text("Owhx subject", encoding="utf-8")


def _config(tmp_path: Path, worker_id: int) -> Path:
    path = tmp_path / f"unified-{worker_id}.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "dataset_root": str(tmp_path / "datasets"),
        "loras_root": str(tmp_path / "ComfyUI" / "models" / "loras"),
        "work_root": str(tmp_path / "automation" / f"worker-{worker_id}"),
        "hub": {"repo_id": "owner/private", "repo_type": "dataset", "token_env": "HF_TOKEN"},
        "worker": {"id": worker_id, "count": 2},
        "sync": {"catalog_prefix": "training-backups", "results_prefix": "training-results"},
        "discovery": {
            "quiet_seconds": 0, "target_exposures": 126,
            "ledger_path": "training-automation/workflow-ledger.json",
        },
        "queue": {
            "trainer_yaml": str(Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b-v2.yaml"),
            "repo_root": str(Path(__file__).parents[2]),
            "output_root": str(tmp_path / "output"), "trigger_word": "Owhx",
        },
        "checkpoint_policy": {"save_every": 100, "max_local_step_saves": 5},
        "evaluation": {"enabled": True},
    }, sort_keys=False), encoding="utf-8")
    return path


def test_resolves_only_agreed_gui_roots_and_bounded_comfyui_discovery(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    assert resolve_dataset_root(
        {"dataset_root": str(datasets)}, {"DATASETS_FOLDER": str(datasets)}
    ) == datasets.resolve()
    other = tmp_path / "other"
    other.mkdir()
    try:
        resolve_dataset_root(
            {"dataset_root": str(datasets)}, {"DATASETS_FOLDER": str(other)}
        )
    except Exception as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("divergent GUI dataset root was accepted")
    loras = tmp_path / "storage" / "coMFyUi" / "models" / "loras"
    loras.mkdir(parents=True)
    assert resolve_loras_root({"storage_root": str(tmp_path / "storage")}, {}) == loras.resolve()


def test_two_workers_finish_ten_disjoint_jobs_and_repeated_startup_is_idle(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for index in range(10):
        _dataset(datasets, f"Person_{index:02d}_Surname")
    client = FakeHubClient()
    client.revision = "a" * 40
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots = {client.revision: dict(client.remote)}
    published = []

    def publisher(**kwargs):
        published.append(kwargs["job_id"])
        return ({
            "status": "available", "revision": client.revision,
            "remote_root": f"training-results/{kwargs['run_id']}/{kwargs['job_id']}",
        }, [])

    common = dict(
        env={"HF_TOKEN": "secret"}, client=client,
        queue_factory=FakeQueue, sleep=lambda _: None, clock=lambda: 100,
        startup_sync=lambda **kwargs: [], publisher=publisher,
        latest_sync=lambda **kwargs: {"status": "completed"},
        archive_factory=FakeArchive,
    )
    first = run_unified_workflow(_config(tmp_path, 0), **common)
    second = run_unified_workflow(_config(tmp_path, 1), **common)
    assert first["status"] == second["status"] == "completed"
    assert len(published) == 10 and len(set(published)) == 10
    idle = run_unified_workflow(_config(tmp_path, 0), dry_run=True, **common)
    assert idle["status"] == "idle" and idle["pending"] == []
    ledger = json.loads(client.remote["training-automation/workflow-ledger.json"])
    assert len(ledger["datasets"]) == 10
    assert {item["status"] for item in ledger["datasets"].values()} == {"completed"}


def test_changed_completed_dataset_is_held_not_retrained(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.revision = "b" * 40
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots = {client.revision: dict(client.remote)}
    common = dict(
        env={"HF_TOKEN": "secret"}, client=client,
        queue_factory=FakeQueue, sleep=lambda _: None, clock=lambda: 100,
        startup_sync=lambda **kwargs: [],
        publisher=lambda **kwargs: ({"status": "available", "revision": client.revision}, []),
        latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
    )
    completed = run_unified_workflow(_config(tmp_path, 0), **common)
    assert completed["status"] == "completed"
    (datasets / "Ada_Lovelace" / "one.txt").write_text("changed", encoding="utf-8")
    held = run_unified_workflow(_config(tmp_path, 0), dry_run=True, **common)
    assert held["status"] == "held"
    assert held["held"][0]["folder"] == "Ada_Lovelace"
    ledger = json.loads(client.remote["training-automation/workflow-ledger.json"])
    assert ledger["datasets"]["Ada_Lovelace"]["status"] == "changed"
