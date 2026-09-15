import importlib.util
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
import pytest
from PIL import Image

from training_automation.queue import TrainingQueue
from training_automation.state import atomic_write_json, read_json
from training_automation.unified import (
    _repo,
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


def _config(tmp_path: Path, worker_id: int, worker_count: int = 2) -> Path:
    database = tmp_path / "aitk_db.db"
    if not database.exists():
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)"
        )
        connection.commit()
        connection.close()
    path = tmp_path / f"unified-{worker_id}.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "dataset_root": str(tmp_path / "datasets"),
        "loras_root": str(tmp_path / "ComfyUI" / "models" / "loras"),
        "work_root": str(tmp_path / "automation"),
        "gui_database": str(database),
        "initialize_gui_dataset_root": True,
        "hub": {"repo_id": "owner/private", "repo_type": "dataset", "token_env": "HF_TOKEN"},
        "worker": {"id": worker_id, "count": worker_count},
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
        {"dataset_root": str(datasets), "queue": {"repo_root": str(tmp_path)}},
        {"DATASETS_FOLDER": str(datasets)},
    ) == datasets.resolve()
    other = tmp_path / "other"
    other.mkdir()
    try:
        resolve_dataset_root(
            {"dataset_root": str(datasets), "queue": {"repo_root": str(tmp_path)}},
            {"DATASETS_FOLDER": str(other)},
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
    assert first["status"] in {"completed", "idle"}
    assert second["status"] in {"completed", "idle"}
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
    completed = run_unified_workflow(_config(tmp_path, 0, 1), **common)
    assert completed["status"] == "completed"
    (datasets / "Ada_Lovelace" / "one.txt").write_text("changed", encoding="utf-8")
    held = run_unified_workflow(_config(tmp_path, 0, 1), dry_run=True, **common)
    assert held["status"] == "held"
    assert held["held"][0]["folder"] == "Ada_Lovelace"
    ledger = json.loads(client.remote["training-automation/workflow-ledger.json"])
    assert ledger["datasets"]["Ada_Lovelace"]["status"] == "changed"


def test_gui_database_bridge_initializes_only_absent_setting_and_reads_token(tmp_path):
    datasets = tmp_path / "shared" / "datasets"
    datasets.mkdir(parents=True)
    native = tmp_path / "toolkit" / "datasets"
    native.mkdir(parents=True)
    database = tmp_path / "toolkit" / "aitk_db.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)"
    )
    connection.execute("INSERT INTO Settings(key, value) VALUES('HF_TOKEN', 'db-token')")
    connection.commit()
    connection.close()
    config = {
        "dataset_root": str(datasets),
        "gui_database": str(database),
        "initialize_gui_dataset_root": True,
        "queue": {"repo_root": str(tmp_path / "toolkit")},
    }

    assert resolve_dataset_root(config, {}) == datasets.resolve()
    assert _repo({**config, "hub": {"repo_id": "owner/private", "repo_type": "dataset"}}, {}) == (
        "owner/private", "dataset", "db-token"
    )
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT value FROM Settings WHERE key='DATASETS_FOLDER'"
    ).fetchone() == (str(datasets.resolve()),)
    connection.close()

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(Exception, match="differ"):
        resolve_dataset_root({**config, "dataset_root": str(other)}, {})
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT value FROM Settings WHERE key='DATASETS_FOLDER'"
    ).fetchone() == (str(datasets.resolve()),)
    connection.close()
    schema = (Path(__file__).parents[2] / "ui/prisma/schema.prisma").read_text()
    assert "model Settings" in schema and "key   String @unique" in schema


def test_gui_bridge_refuses_redirect_when_native_dataset_root_has_unseen_data(tmp_path):
    desired = tmp_path / "shared"
    desired.mkdir()
    toolkit = tmp_path / "toolkit"
    native = toolkit / "datasets"
    native.mkdir(parents=True)
    (native / "unseen.txt").write_text("existing", encoding="utf-8")
    database = toolkit / "aitk_db.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)"
    )
    connection.commit()
    connection.close()
    with pytest.raises(Exception, match="native dataset root contains data"):
        resolve_dataset_root({
            "dataset_root": str(desired), "gui_database": str(database),
            "initialize_gui_dataset_root": True,
            "queue": {"repo_root": str(toolkit)},
        }, {})


def test_gui_bridge_can_seed_exact_settings_table_on_clean_first_boot(tmp_path):
    desired = tmp_path / "shared" / "datasets"
    desired.mkdir(parents=True)
    toolkit = tmp_path / "toolkit"
    (toolkit / "datasets").mkdir(parents=True)
    database = toolkit / "aitk_db.db"
    assert resolve_dataset_root({
        "dataset_root": str(desired), "gui_database": str(database),
        "initialize_gui_dataset_root": True,
        "queue": {"repo_root": str(toolkit)},
    }, {}) == desired.resolve()
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT key, value FROM Settings"
    ).fetchall() == [("DATASETS_FOLDER", str(desired.resolve()))]
    connection.close()


def test_instance_environment_overrides_shared_worker_defaults(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    config = _config(tmp_path, 0, 1)
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)
    result = run_unified_workflow(
        config,
        env={"HF_TOKEN": "secret", "TRAINING_WORKER_ID": "1", "TRAINING_WORKER_COUNT": "2"},
        client=client, dry_run=True, sleep=lambda _: None, clock=lambda: 100,
        startup_sync=lambda **kwargs: [],
    )
    assert result["worker_id"] == 1 and result["worker_count"] == 2


class CountingQueue(FakeQueue):
    training_calls = []

    def run(self):
        jobs = self.materialize()
        prior = read_json(self.state_path, {"jobs": {}})
        for job in jobs:
            if (prior.get("jobs", {}).get(job.job_id) or {}).get("status") != "completed":
                self.training_calls.append(job.job_id)
        return super().run()


def test_completed_training_resumes_export_without_retraining_when_candidate_set_changes(tmp_path):
    CountingQueue.training_calls = []
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)
    attempts = {"count": 0}

    def publisher(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("synthetic export interruption")
        return {"status": "available", "revision": client.revision}, []

    common = dict(
        env={"HF_TOKEN": "secret"}, client=client, queue_factory=CountingQueue,
        sleep=lambda _: None, clock=lambda: 100, startup_sync=lambda **kwargs: [],
        publisher=publisher, latest_sync=lambda **kwargs: {"status": "completed"},
        archive_factory=FakeArchive,
    )
    with pytest.raises(RuntimeError, match="export interruption"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)
    assert len(CountingQueue.training_calls) == 1
    _dataset(datasets, "Bea_Hopper")

    result = run_unified_workflow(_config(tmp_path, 0, 1), **common)
    assert result["status"] == "completed"
    assert len(CountingQueue.training_calls) == 2
    assert len({*CountingQueue.training_calls}) == 2
    ledger = json.loads(client.remote["training-automation/workflow-ledger.json"])
    assert ledger["datasets"]["Ada_Lovelace"]["status"] == "completed"
    assert ledger["datasets"]["Bea_Hopper"]["status"] == "completed"


def test_prior_dataset_is_completed_before_later_dataset_failure(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    _dataset(datasets, "Bea_Hopper")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)

    class FailSecondQueue(FakeQueue):
        def run(self):
            dataset = yaml.safe_load(self.inner.config_path.read_text())["datasets"][0]
            if dataset["name"] == "Bea_Hopper":
                job = self.materialize()[0]
                state = {"schema_version": 2, "jobs": {job.job_id: {
                    "status": "failed", "training_status": "failed",
                    "evaluation_status": "pending",
                }}}
                atomic_write_json(self.state_path, state)
                return state
            return super().run()

    with pytest.raises(Exception, match="Bea_Hopper"):
        run_unified_workflow(
            _config(tmp_path, 0, 1), env={"HF_TOKEN": "secret"}, client=client,
            queue_factory=FailSecondQueue, sleep=lambda _: None, clock=lambda: 100,
            startup_sync=lambda **kwargs: [],
            publisher=lambda **kwargs: ({"status": "available", "revision": client.revision}, []),
            latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
        )
    ledger = json.loads(client.remote["training-automation/workflow-ledger.json"])
    assert ledger["datasets"]["Ada_Lovelace"]["status"] == "completed"
    assert ledger["datasets"]["Bea_Hopper"]["status"] == "incomplete"


def test_missing_local_execution_state_for_dispatched_dataset_fails_closed(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)
    common = dict(
        env={"HF_TOKEN": "secret"}, client=client, queue_factory=FakeQueue,
        sleep=lambda _: None, clock=lambda: 100, startup_sync=lambda **kwargs: [],
        publisher=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("export stopped")),
        latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
    )
    with pytest.raises(RuntimeError, match="export stopped"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)
    state_path = next((tmp_path / "automation").rglob("execution-state.json"))
    state_path.unlink()
    with pytest.raises(Exception, match="durable execution state is missing"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)


def test_missing_queue_state_after_completed_training_refuses_retraining(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)

    class CountingQueue(FakeQueue):
        calls = 0

        def run(self):
            type(self).calls += 1
            return super().run()

    common = dict(
        env={"HF_TOKEN": "secret"}, client=client, queue_factory=CountingQueue,
        sleep=lambda _: None, clock=lambda: 100, startup_sync=lambda **kwargs: [],
        publisher=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("export stopped")),
        latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
    )
    with pytest.raises(RuntimeError, match="export stopped"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)
    queue_state = next((tmp_path / "automation").rglob("queue-state.json"))
    queue_state.unlink()
    with pytest.raises(Exception, match="queue state is missing"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)
    assert CountingQueue.calls == 1


def test_interrupted_training_is_not_automatically_retried(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)

    class InterruptedQueue(FakeQueue):
        calls = 0

        def run(self):
            type(self).calls += 1
            job = self.materialize()[0]
            atomic_write_json(self.state_path, {
                "schema_version": 2,
                "jobs": {job.job_id: {
                    "status": "training",
                    "training_status": "running",
                    "evaluation_status": "pending",
                    "attempts": 1,
                }},
            })
            raise RuntimeError("simulated process termination")

    common = dict(
        env={"HF_TOKEN": "secret"}, client=client,
        sleep=lambda _: None, clock=lambda: 100, startup_sync=lambda **kwargs: [],
        publisher=lambda **kwargs: ({"status": "available", "revision": client.revision}, []),
        latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
    )
    with pytest.raises(RuntimeError, match="simulated process termination"):
        run_unified_workflow(
            _config(tmp_path, 0, 1), queue_factory=InterruptedQueue, **common
        )

    class RejectQueue(FakeQueue):
        calls = 0

        def run(self):
            type(self).calls += 1
            raise AssertionError("interrupted training must not be dispatched again")

    with pytest.raises(Exception, match="outcome is uncertain"):
        run_unified_workflow(
            _config(tmp_path, 0, 1), queue_factory=RejectQueue, **common
        )
    assert InterruptedQueue.calls == 1
    assert RejectQueue.calls == 0


def test_completed_queue_resumes_export_without_invoking_trainer(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _dataset(datasets, "Ada_Lovelace")
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)

    class CountingQueue(FakeQueue):
        calls = 0

        def run(self):
            type(self).calls += 1
            return super().run()

    exports = []

    def publisher(**kwargs):
        exports.append(kwargs["job_id"])
        if len(exports) == 1:
            raise RuntimeError("export stopped")
        return {"status": "available", "revision": client.revision}, []

    common = dict(
        env={"HF_TOKEN": "secret"}, client=client, queue_factory=CountingQueue,
        sleep=lambda _: None, clock=lambda: 100, startup_sync=lambda **kwargs: [],
        publisher=publisher,
        latest_sync=lambda **kwargs: {"status": "completed"}, archive_factory=FakeArchive,
    )
    with pytest.raises(RuntimeError, match="export stopped"):
        run_unified_workflow(_config(tmp_path, 0, 1), **common)
    result = run_unified_workflow(_config(tmp_path, 0, 1), **common)
    assert result["status"] == "completed"
    assert CountingQueue.calls == 1
    assert len(exports) == 2


def test_two_workers_with_same_work_root_use_disjoint_mutable_staging(tmp_path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for index in range(10):
        _dataset(datasets, f"Person_{index:02d}_Surname")

    class ThreadSafeHub(FakeHubClient):
        def __init__(self):
            super().__init__()
            self.guard = threading.RLock()
            self.local_artifacts = []

        def commit_files(self, *args, **kwargs):
            with self.guard:
                artifacts = args[2]
                self.local_artifacts.extend(
                    (item.remote_path, item.local_path) for item in artifacts
                )
                return super().commit_files(*args, **kwargs)

        def read_remote_file(self, *args, **kwargs):
            with self.guard:
                return super().read_remote_file(*args, **kwargs)

        def path_metadata(self, *args, **kwargs):
            with self.guard:
                return super().path_metadata(*args, **kwargs)

    client = ThreadSafeHub()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2, "models": []
    }).encode()
    client.snapshots[client.revision] = dict(client.remote)
    published = []
    published_guard = threading.Lock()

    def publisher(**kwargs):
        with published_guard:
            published.append(kwargs["job_id"])
        return {"status": "available", "revision": client.revision}, []

    config = _config(tmp_path, 0, 1)
    def run_worker(worker_id):
        return run_unified_workflow(
            config,
            env={"HF_TOKEN": "secret", "TRAINING_WORKER_ID": str(worker_id), "TRAINING_WORKER_COUNT": "2"},
            client=client, queue_factory=FakeQueue, sleep=lambda _: None,
            clock=lambda: 100, startup_sync=lambda **kwargs: [], publisher=publisher,
            latest_sync=lambda **kwargs: {"status": "completed"},
            archive_factory=FakeArchive,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_worker, (0, 1)))
    assert {item["worker_id"] for item in results} == {0, 1}
    assert len(published) == 10 and len(set(published)) == 10
    ledger_staging = {
        Path(local).parent.parent.name
        for remote, local in client.local_artifacts
        if remote == "training-automation/workflow-ledger.json"
    }
    catalog_staging = {
        part for remote, local in client.local_artifacts
        if remote == "training-backups/catalog.json"
        for part in Path(local).parts if part in {"worker-0", "worker-1"}
    }
    assert ledger_staging == {"worker-0", "worker-1"}
    assert catalog_staging == {"worker-0", "worker-1"}
