import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from training_automation.backup import BackupError
import training_automation.bootstrap as bootstrap
from training_automation.bootstrap import (
    NATIVE_KLEIN_FILENAME,
    run_parallel_bootstrap,
    validate_manifest,
)
from training_automation.queue import QueueJob
from training_automation.worker import run_worker


FAKE_SPEC = importlib.util.spec_from_file_location("parallel_bootstrap_fakes", Path(__file__).with_name("fakes.py"))
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient
MANIFEST_REVISION = "a" * 40
DATASET_REVISION = "b" * 40


@pytest.fixture(autouse=True)
def fake_backend_preflight(monkeypatch):
    monkeypatch.setattr(bootstrap, "preflight_evaluation_backends", lambda config: {})

    def publish_results(**kwargs):
        path = Path(kwargs["work_dir"]) / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"available"}', encoding="utf-8")
        return ({
            "status": "available",
            "exported_candidate_count": 3,
            "remote_root": f"training-results/{kwargs['run_id']}/{kwargs['job_id']}",
            "revision": "result-revision",
        }, [(path, f"results/{kwargs['job_id']}/index.json")])

    monkeypatch.setattr(bootstrap, "publish_ranked_results", publish_results)


class Pod:
    def __init__(self, *, valid=True):
        self.valid = valid
        self.deleted = []

    def instance(self, instance_id):
        return {
            "id": instance_id,
            "hashId": "hash-42" if self.valid else "wrong",
            "notes": "training-run:run-1;shard:a",
        }

    def delete(self, instance_id):
        self.deleted.append(instance_id)


class CompletedQueue:
    def __init__(
        self, path, *, reference_available=True,
        identity_ranking_status="available", write_sample=True,
        pose_backend_available=True,
    ):
        config = yaml.safe_load(Path(path).read_text())
        self.state_path = Path(config["state_path"])
        self.reference_available = reference_available
        self.identity_ranking_status = identity_ranking_status
        self.write_sample = write_sample
        self.config = config
        self.pose_configured = bool((config.get("evaluation") or {}).get("landmark_backend"))
        self.pose_backend_available = pose_backend_available
        self.jobs = []
        for index, dataset in enumerate(config["datasets"]):
            job_id = f"{dataset['name']}-job"
            config_path = Path(config["generated_dir"]) / f"{job_id}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            final_step = int(dataset.get("training_steps", 1200))
            policy = config.get("checkpoint_policy")
            config_path.write_text(yaml.safe_dump({
                "job": "extension",
                **({"meta": {"training_automation_schedule": {
                    "base_training_steps": 1200,
                    "resolved_training_steps": final_step,
                    "training_accounting": dataset.get("training_accounting"),
                    "checkpoint_policy": policy,
                }}} if dataset.get("training_steps") or policy else {}),
                "config": {
                    "name": job_id,
                    "process": [{
                        "train": {"steps": final_step},
                        "sample": {
                            "sample_every": 100,
                            "seed": 42,
                            "walk_seed": False,
                            "samples": [
                                {"prompt": f"clothed reference prompt {prompt_index}"}
                                for prompt_index in range(23)
                            ],
                        },
                    }],
                },
            }, sort_keys=False), encoding="utf-8")
            self.jobs.append(QueueJob(
                job_id=job_id, config_path=config_path,
                output_root=Path(config["training_folder"]), reference_images=(),
                training_steps=final_step,
                training_accounting=dataset.get("training_accounting"),
                checkpoint_policy=policy,
            ))

    def materialize(self):
        return self.jobs

    def run(self):
        state = {"schema_version": 2, "jobs": {}}
        for job in self.jobs:
            output = job.output_root / job.job_id
            automation = output / ".automation"
            samples = output / "samples"
            automation.mkdir(parents=True, exist_ok=True)
            samples.mkdir(parents=True, exist_ok=True)
            report_checkpoints = []
            if not self.write_sample:
                (samples / "not-a-sample-file").mkdir()
            trainer = yaml.safe_load(job.config_path.read_text())
            final_step = trainer["config"]["process"][0]["train"]["steps"]
            expected_steps = list(range(100, final_step, 100)) + [final_step]
            receipts = {}
            for step in expected_steps:
                checkpoint_samples = []
                for prompt_index in range(23):
                    sample_path = samples / f"1__{step:09d}_{prompt_index}.png"
                    if self.write_sample:
                        sample_path.write_bytes(b"sample")
                    checkpoint_samples.append({
                        "path": str(sample_path),
                        "prompt_index": prompt_index,
                        "seed": 42,
                        "identity": {
                            "status": (
                                "missing"
                                if self.identity_ranking_status == "unavailable"
                                and step == 100 and prompt_index == 0
                                else "available"
                            )
                        },
                        "pose_body_landmarks": {
                            "status": "available" if self.pose_configured else "unavailable",
                            "values": {} if self.pose_configured else None,
                        },
                    })
                report_checkpoints.append({
                    "step": step,
                    "final": step == final_step,
                    "sample_run_status": "complete",
                    "remote_association": {"status": "unique", "reason": None},
                    "samples": checkpoint_samples,
                })
                receipts[str(step)] = {
                    "status": "backed_up", "verified": True, "cataloged": True,
                    "final": step == final_step,
                }
            (automation / "backup-state.json").write_text(json.dumps({
                "schema_version": 2,
                "checkpoints": receipts,
            }), encoding="utf-8")
            reference = "available" if self.reference_available else "unavailable"
            (automation / "evaluation.json").write_text(json.dumps({
                "reference_identity_status": {"status": reference},
                "identity_ranking": {
                    "status": self.identity_ranking_status,
                    "reason": (
                        "valid identity subsets are not comparable across checkpoints"
                        if self.identity_ranking_status == "unavailable" else None
                    ),
                },
                "ranking": {"status": "available"},
                "pose_backend": {
                    "status": "available" if self.pose_configured and self.pose_backend_available else "unavailable"
                },
                "checkpoints": report_checkpoints,
                "reference_provenance": "training-set",
            }), encoding="utf-8")
            state["jobs"][job.job_id] = {
                "status": "completed", "training_status": "completed",
                "evaluation_status": "completed",
            }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        return state


def private_manifest():
    datasets = []
    files = {}
    for index in range(6):
        payload = f"image-{index}".encode()
        remote = f"datasets/subject-{index}/image.jpg"
        files[remote] = payload
        datasets.append({
            "id": f"subject-{index}", "shard_id": "a" if index < 3 else "b",
            "catalog_name": f"Private {index}", "expected_catalog_id": index + 1,
            "trigger_word": "TOKEN", "reference_paths": ["image.jpg"],
            "files": [{
                "remote_path": remote, "relative_path": "image.jpg",
                "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        })
    manifest = {
        "schema_version": 1, "run_id": "run-1",
        "recipe_id": "subject_likeness_masked_flux2_klein9b",
        "expected_jobs_per_shard": 3,
        "dataset_revision": DATASET_REVISION,
        "storage": {"minimum_free_bytes": 1},
        "model_sources": [
            {
                "kind": "base_model", "repo_id": "base/repo",
                "revision": "c" * 40, "local_path": "pinned/base",
                "artifact_path": "flux-2-klein-base-9b.safetensors",
                "allow_patterns": ["flux-2-klein-base-9b.safetensors"],
            },
            {
                "kind": "text_encoder", "repo_id": "text/repo",
                "revision": "e" * 40, "local_path": "pinned/text",
                "allow_patterns": ["*.json", "*.safetensors"],
            },
            {
                "kind": "vae", "repo_id": "vae/repo",
                "revision": "f" * 40, "local_path": "pinned/vae",
                "artifact_path": "ae.safetensors",
                "allow_patterns": ["ae.safetensors"],
            },
            {
                "kind": "depth_model", "repo_id": "depth/repo",
                "revision": "d" * 40, "local_path": "pinned/depth",
                "allow_patterns": ["**"],
            },
        ],
        "datasets": datasets,
        "evaluation": {
            "enabled": True, "require_identity_available": True,
            "reference_provenance": "training-set",
            "face_backend": "training_automation.backends:InsightFaceCPUBackend",
            "face_backend_options": {
                "model_dir": "/opt/training-automation-models/insightface/models/buffalo_l"
            },
        },
        "archive": {"remote_prefix": "training-runs"},
        "binding": {
            "remote_path": "bindings/run-1/{shard_id}.json",
            "wait_seconds": 0, "poll_seconds": 1,
        },
    }
    return manifest, files


def setup_client():
    manifest, files = private_manifest()
    client = FakeHubClient()
    client.snapshots[MANIFEST_REVISION] = {"private/manifest.json": json.dumps(manifest).encode()}
    client.snapshots[DATASET_REVISION] = files
    client.remote["bindings/run-1/a.json"] = json.dumps({
        "schema_version": 1, "run_id": "run-1", "shard_id": "a",
        "instance_id": 42, "instance_hash_id": "hash-42",
        "instance_notes": "training-run:run-1;shard:a",
    }).encode()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2,
        "models": [{
            "id": index + 1, "name": f"Private {index}",
            "folder": f"{index + 1:04d}-private-{index}",
            "base_arch": "flux2_klein_9b",
            "base_model": "black-forest-labs/FLUX.2-klein-base-9B",
            "trigger_word": "TOKEN", "destination_kind": "loras",
            "checkpoints": [], "selected_checkpoint_id": None, "selection": None,
        } for index in range(6)],
    }).encode()
    return client


def env(tmp_path):
    return {
        "HF_REPO_ID": "owner/private", "HF_REPO_TYPE": "dataset",
        "HF_MANIFEST_PATH": "private/manifest.json",
        "HF_MANIFEST_REVISION": MANIFEST_REVISION, "HF_TOKEN": "secret",
        "TRAINING_RUN_ID": "run-1", "TRAINING_SHARD_ID": "a",
        "TRAINING_STORAGE_ROOT": str(tmp_path / "storage"),
        "TRAINING_REPO_ROOT": str(tmp_path / "repo"),
        "SIMPLEPOD_API_TOKEN": "pod-secret",
    }


def snapshot_fetch(**kwargs):
    target = Path(kwargs["local_dir"])
    target.mkdir(parents=True, exist_ok=True)
    repo_files = {
        "base/repo": ["flux-2-klein-base-9b.safetensors"],
        "text/repo": ["config.json", "model.safetensors", "tokenizer.json"],
        "vae/repo": ["ae.safetensors"],
        "depth/repo": ["config.json", "model.safetensors"],
    }
    for relative in repo_files[kwargs["repo_id"]]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kwargs["revision"], encoding="utf-8")
    return str(target)


def test_parallel_bootstrap_archives_then_deletes_only_bound_instance(tmp_path):
    pod = Pod()
    client = setup_client()
    result = run_parallel_bootstrap(
        env=env(tmp_path), hub_client=client, pod_client=pod,
        queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
        recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
    )
    assert result["status"] == "delete-accepted"
    assert result["archive_completion_revision"]
    assert pod.deleted == [42]
    completion = json.loads(client.remote["training-runs/run-1/a/completion.json"])
    assert completion["completion"]["reference_provenance"] == "training-set"
    assert completion["completion"]["manifest_revision"] == MANIFEST_REVISION
    assert completion["completion"]["dataset_revision"] == DATASET_REVISION
    first_job = completion["completion"]["jobs"][0]
    assert first_job["automatic_result_export"]["status"] == "available"
    assert first_job["automatic_result_export"]["exported_candidate_count"] == 3
    sources = {
        source["kind"]: source for source in completion["completion"]["model_sources"]
    }
    assert sources["base_model"]["local_path"] == "pinned/base"
    assert sources["base_model"]["artifact_path"] == NATIVE_KLEIN_FILENAME
    assert sources["text_encoder"]["local_path"] == "pinned/text"
    assert "artifact_path" not in sources["text_encoder"]
    assert sources["vae"]["artifact_path"] == "ae.safetensors"
    evidence = {item["relative_path"] for item in completion["evidence"]}
    assert "deployment-manifest.json" in evidence
    assert "model-sources/base_model.json" in evidence
    assert "model-sources/depth_model.json" in evidence
    assert "model-sources/text_encoder.json" in evidence
    assert "model-sources/vae.json" in evidence
    assert "results/subject-0-job/index.json" in evidence
    trainer = yaml.safe_load(
        (tmp_path / "storage/automation/run-1/a/trainer.yaml").read_text(encoding="utf-8")
    )
    process = trainer["config"]["process"][0]
    assert process["logging"]["use_ui_logger"] is False
    assert "sqlite_db_path" not in process
    assert process["model"]["name_or_path"].endswith("/models/pinned/base")
    assert process["model"]["te_name_or_path"].endswith("/models/pinned/text")
    assert process["model"]["vae_path"].endswith("/models/pinned/vae/ae.safetensors")
    assert process["depth_consistency"]["model_id"].endswith("/models/pinned/depth")


def test_reference_identity_failure_is_durable_and_never_deletes(tmp_path):
    pod = Pod()
    with pytest.raises(BackupError, match="no successful reference identity"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=lambda path: CompletedQueue(path, reference_available=False),
            snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    state = json.loads((tmp_path / "storage/automation/run-1/a/bootstrap-state.json").read_text())
    assert state["status"] == "failed"
    assert state["delete_requested"] is False
    assert pod.deleted == []


def test_unavailable_generated_face_ranking_is_archived_without_fake_score(tmp_path):
    pod = Pod()
    client = setup_client()
    result = run_parallel_bootstrap(
        env=env(tmp_path), hub_client=client, pod_client=pod,
        queue_factory=lambda path: CompletedQueue(
            path, reference_available=True, identity_ranking_status="unavailable"
        ),
        snapshot_fetch=snapshot_fetch,
        recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
    )
    assert result["status"] == "delete-accepted"
    assert pod.deleted == [42]
    completion = json.loads(client.remote["training-runs/run-1/a/completion.json"])
    assert {
        item["identity_ranking_status"] for item in completion["completion"]["jobs"]
    } == {"unavailable"}
    report_paths = [
        path for path in client.remote
        if path.endswith("/evaluation.json") and "/evidence/jobs/" in path
    ]
    report = json.loads(client.remote[report_paths[0]])
    assert report["identity_ranking"]["status"] == "unavailable"
    assert report["checkpoints"][0]["samples"][0]["identity"]["status"] == "missing"


def test_directory_only_sample_output_never_passes_completion_gate(tmp_path):
    pod = Pod()
    with pytest.raises(BackupError, match="no sample files"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=lambda path: CompletedQueue(path, write_sample=False),
            snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    state = json.loads((tmp_path / "storage/automation/run-1/a/bootstrap-state.json").read_text())
    assert state["status"] == "failed"
    assert state["delete_requested"] is False
    assert pod.deleted == []


def test_wrong_instance_identity_stops_before_staging_or_delete(tmp_path):
    pod = Pod(valid=False)
    with pytest.raises(BackupError, match="hashId"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert pod.deleted == []
    assert not (tmp_path / "storage/datasets/run-1/a").exists()


def test_manifest_requires_exact_three_job_assignment_and_pinned_models():
    manifest, _ = private_manifest()
    assert len(validate_manifest(manifest, run_id="run-1", shard_id="a")) == 3
    manifest["datasets"][0]["shard_id"] = "b"
    with pytest.raises(BackupError, match="two disjoint three-job shards"):
        validate_manifest(manifest, run_id="run-1", shard_id="a")


def test_manifest_validates_adaptive_accounting_and_checkpoint_policy():
    manifest, _ = private_manifest()
    manifest["checkpoint_policy"] = {"save_every": 100, "max_local_step_saves": 5}
    manifest["datasets"][0].update({
        "training_steps": 1590,
        "training_accounting": {
            "source_image_count": 1, "loader_batches_per_epoch": 265,
            "loader_epochs": 6, "resolution_repeats": [16, 4, 1],
            "batch_size": 4, "original_image_exposures": 126,
            "partial_bucket_batches": "un-padded",
        },
    })
    assert len(validate_manifest(manifest, run_id="run-1", shard_id="a")) == 3
    manifest["datasets"][0]["training_accounting"]["loader_batches_per_epoch"] = 264
    with pytest.raises(BackupError, match="training_steps must equal"):
        validate_manifest(manifest, run_id="run-1", shard_id="a")


def test_nonround_final_schedule_and_pose_evidence_archive_before_delete(tmp_path):
    client = setup_client()
    manifest = json.loads(client.snapshots[MANIFEST_REVISION]["private/manifest.json"])
    manifest["checkpoint_policy"] = {"save_every": 100, "max_local_step_saves": 5}
    manifest["datasets"][0].update({
        "training_steps": 1590,
        "training_accounting": {
            "source_image_count": 1, "loader_batches_per_epoch": 265,
            "loader_epochs": 6, "resolution_repeats": [16, 4, 1],
            "batch_size": 4, "original_image_exposures": 126,
            "partial_bucket_batches": "un-padded",
        },
    })
    manifest["evaluation"].update({
        "landmark_backend": "training_automation.backends:UltralyticsPoseCPUBackend",
        "landmark_backend_options": {
            "model_path": "/opt/training-automation-models/ultralytics/yolo11n-pose.pt",
            "expected_sha256": "a" * 64,
        },
    })
    client.snapshots[MANIFEST_REVISION]["private/manifest.json"] = json.dumps(manifest).encode()
    pod = Pod()
    result = run_parallel_bootstrap(
        env=env(tmp_path), hub_client=client, pod_client=pod,
        queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
        recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
    )
    assert result["status"] == "delete-accepted"
    completion = json.loads(client.remote["training-runs/run-1/a/completion.json"])
    first = next(item for item in completion["completion"]["jobs"] if item["job_id"] == "subject-0-job")
    assert first["training_schedule"]["resolved_training_steps"] == 1590
    assert first["pose_backend_status"] == "available"
    assert pod.deleted == [42]


def test_configured_pose_backend_failure_never_deletes(tmp_path):
    client = setup_client()
    manifest = json.loads(client.snapshots[MANIFEST_REVISION]["private/manifest.json"])
    manifest["evaluation"].update({
        "landmark_backend": "training_automation.backends:UltralyticsPoseCPUBackend",
        "landmark_backend_options": {"model_path": "/pose.pt", "expected_sha256": "a" * 64},
    })
    client.snapshots[MANIFEST_REVISION]["private/manifest.json"] = json.dumps(manifest).encode()
    pod = Pod()
    with pytest.raises(BackupError, match="pose backend did not execute"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=client, pod_client=pod,
            queue_factory=lambda path: CompletedQueue(path, pose_backend_available=False),
            snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert pod.deleted == []


def test_completed_queue_restart_runs_archive_only_then_deletes(tmp_path, monkeypatch):
    class InterruptArchiveClient(FakeHubClient):
        fail_archive = True

        def commit_files(self, repo_id, repo_type, artifacts, message, parent_commit=None):
            if self.fail_archive and message.startswith("Archive"):
                raise RuntimeError("archive transport interrupted")
            return super().commit_files(
                repo_id, repo_type, artifacts, message, parent_commit
            )

    class OneAttemptArchive(bootstrap.EvidenceArchive):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, max_attempts=1, sleep=lambda _: None)

    client = InterruptArchiveClient()
    prepared = setup_client()
    client.remote = dict(prepared.remote)
    client.snapshots = {key: dict(value) for key, value in prepared.snapshots.items()}
    client.revision = prepared.revision
    pod = Pod()
    original_archive = bootstrap.EvidenceArchive
    monkeypatch.setattr(bootstrap, "EvidenceArchive", OneAttemptArchive)
    with pytest.raises(BackupError, match="evidence archive failed"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=client, pod_client=pod,
            queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    failed = json.loads(
        (tmp_path / "storage/automation/run-1/a/bootstrap-state.json").read_text()
    )
    assert failed["status"] == "failed" and failed["delete_requested"] is False
    assert pod.deleted == []

    class RecoveryQueue(CompletedQueue):
        run_calls = 0

        def run(self):
            type(self).run_calls += 1
            raise AssertionError("archive recovery must not invoke training queue run")

    client.fail_archive = False
    monkeypatch.setattr(bootstrap, "EvidenceArchive", original_archive)
    monkeypatch.setattr(
        bootstrap,
        "preflight_evaluation_backends",
        lambda config: (_ for _ in ()).throw(AssertionError("recovery must not load backends")),
    )
    result = run_parallel_bootstrap(
        env=env(tmp_path), hub_client=client, pod_client=pod,
        queue_factory=RecoveryQueue,
        snapshot_fetch=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not stage models")
        ),
        recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
    )
    assert result["status"] == "delete-accepted"
    assert RecoveryQueue.run_calls == 0
    assert pod.deleted == [42]


def test_supervisor_automatically_retries_only_completed_finalization(
    tmp_path, monkeypatch
):
    class InterruptArchiveClient(FakeHubClient):
        fail_archive = True

        def commit_files(self, repo_id, repo_type, artifacts, message, parent_commit=None):
            if self.fail_archive and message.startswith("Archive training evidence"):
                raise RuntimeError("transient archive interruption")
            return super().commit_files(
                repo_id, repo_type, artifacts, message, parent_commit=parent_commit
            )

    class OneAttemptArchive(bootstrap.EvidenceArchive):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, max_attempts=1, sleep=lambda _: None)

    class RecoveryQueue(CompletedQueue):
        def run(self):
            raise AssertionError("supervisor recovery must not invoke training")

    prepared = setup_client()
    client = InterruptArchiveClient()
    client.remote = dict(prepared.remote)
    client.snapshots = {
        key: dict(value) for key, value in prepared.snapshots.items()
    }
    client.revision = prepared.revision
    pod = Pod()
    attempts = []
    monkeypatch.setattr(bootstrap, "EvidenceArchive", OneAttemptArchive)

    class Process:
        def __init__(self, output, exit_code):
            self.stdout = io.StringIO(output)
            self.exit_code = exit_code

        def wait(self):
            return self.exit_code

    def popen(command, **kwargs):
        attempts.append(command)
        if len(attempts) == 2:
            client.fail_archive = False
        try:
            run_parallel_bootstrap(
                env=kwargs["env"],
                hub_client=client,
                pod_client=pod,
                queue_factory=CompletedQueue if len(attempts) == 1 else RecoveryQueue,
                snapshot_fetch=snapshot_fetch,
                recipe_path=(
                    Path(__file__).parents[2]
                    / "config/examples/klein_automation/"
                    "trainer-subject-likeness-masked-klein-9b.yaml"
                ),
            )
        except BackupError as exc:
            return Process(f"{type(exc).__name__}: {exc}\n", 1)
        return Process("archive verified and delete accepted\n", 0)

    held = []
    result = run_worker(
        env=env(tmp_path),
        popen=popen,
        hold=lambda: held.append(True),
        sleep=lambda _: None,
        console=io.StringIO(),
    )

    recovery = json.loads(
        (tmp_path / "storage/automation/run-1/a/worker-recovery-state.json")
        .read_text(encoding="utf-8")
    )
    assert result == 0
    assert len(attempts) == 2
    assert held == []
    assert recovery["status"] == "completed"
    assert pod.deleted == [42]
    completion = json.loads(client.remote["training-runs/run-1/a/completion.json"])
    assert "worker-recovery-state.json" in {
        item["relative_path"] for item in completion["evidence"]
    }


def test_restart_never_retries_an_uncertain_delete(tmp_path):
    state_path = tmp_path / "storage/automation/run-1/a/bootstrap-state.json"
    state_path.parent.mkdir(parents=True)
    original = {
        "schema_version": 1,
        "run_id": "run-1",
        "shard_id": "a",
        "status": "failed",
        "delete_requested": True,
        "error": "delete response was lost",
    }
    state_path.write_text(json.dumps(original), encoding="utf-8")
    pod = Pod()
    with pytest.raises(BackupError, match="uncertain outcome"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=lambda path: (_ for _ in ()).throw(
                AssertionError("uncertain deletion must stop before queue construction")
            ),
        )
    assert json.loads(state_path.read_text()) == original
    assert pod.deleted == []


def test_restart_never_treats_an_empty_persisted_state_as_a_fresh_run(tmp_path):
    state_path = tmp_path / "storage/automation/run-1/a/bootstrap-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    pod = Pod()

    with pytest.raises(BackupError, match="different run or shard"):
        run_parallel_bootstrap(
            env=env(tmp_path),
            hub_client=setup_client(),
            pod_client=pod,
            queue_factory=lambda path: (_ for _ in ()).throw(
                AssertionError("invalid persisted state must stop before queue construction")
            ),
        )

    assert state_path.read_text(encoding="utf-8") == "{}"
    assert pod.deleted == []


@pytest.mark.parametrize("field", ["id", "catalog_name", "expected_catalog_id"])
def test_manifest_rejects_global_identity_duplicates_across_shards(field):
    manifest, _ = private_manifest()
    manifest["datasets"][3][field] = manifest["datasets"][0][field]
    with pytest.raises(BackupError, match="globally unique"):
        validate_manifest(manifest, run_id="run-1", shard_id="a")


def test_manifest_rejects_unlaunched_third_shard():
    manifest, _ = private_manifest()
    manifest["datasets"][5]["shard_id"] = "c"
    with pytest.raises(BackupError, match="two disjoint three-job shards"):
        validate_manifest(manifest, run_id="run-1", shard_id="a")


def test_legacy_two_role_manifest_remains_valid():
    manifest, _ = private_manifest()
    manifest["model_sources"] = [
        source for source in manifest["model_sources"]
        if source["kind"] in {"base_model", "depth_model"}
    ]
    assert len(validate_manifest(manifest, run_id="run-1", shard_id="a")) == 3


def test_diffusers_only_base_fails_native_layout_before_trainer_invocation(tmp_path):
    client = setup_client()
    manifest = json.loads(client.snapshots[MANIFEST_REVISION]["private/manifest.json"])
    manifest["model_sources"] = [
        source for source in manifest["model_sources"]
        if source["kind"] in {"base_model", "depth_model"}
    ]
    base = next(source for source in manifest["model_sources"] if source["kind"] == "base_model")
    base.pop("artifact_path")
    base["allow_patterns"] = ["model_index.json", "transformer/*"]
    client.snapshots[MANIFEST_REVISION]["private/manifest.json"] = json.dumps(manifest).encode()
    trainer_calls = []

    def diffusers_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        if kwargs["repo_id"] == "base/repo":
            (target / "model_index.json").write_text("{}", encoding="utf-8")
            (target / "transformer").mkdir(exist_ok=True)
            (target / "transformer/model.safetensors").write_bytes(b"sharded")
        else:
            (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    pod = Pod()
    with pytest.raises(BackupError, match="native Klein transformer artifact is missing"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=client, pod_client=pod,
            queue_factory=lambda path: trainer_calls.append(path),
            snapshot_fetch=diffusers_snapshot,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert trainer_calls == []
    assert pod.deleted == []
    assert not (tmp_path / "storage/automation/run-1/a/queue-state.json").exists()


def test_manifest_rejects_unsafe_model_artifact_selector():
    manifest, _ = private_manifest()
    manifest["model_sources"][0]["artifact_path"] = "../outside.safetensors"
    with pytest.raises(BackupError, match="unsafe"):
        validate_manifest(manifest, run_id="run-1", shard_id="a")


def test_catalog_reservation_mismatch_stops_before_staging_or_delete(tmp_path):
    client = setup_client()
    catalog = json.loads(client.remote["training-backups/catalog.json"])
    catalog["models"][0]["id"] = 99
    client.remote["training-backups/catalog.json"] = json.dumps(catalog).encode()
    pod = Pod()
    with pytest.raises(BackupError, match="expected 1"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=client, pod_client=pod,
            queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert pod.deleted == []
    assert not (tmp_path / "storage/datasets/run-1/a").exists()


def test_disk_preflight_stops_before_staging_or_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bootstrap.shutil, "disk_usage",
        lambda _: SimpleNamespace(total=100, used=100, free=0),
    )
    pod = Pod()
    with pytest.raises(BackupError, match="storage preflight failed"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=CompletedQueue, snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert pod.deleted == []
    assert not (tmp_path / "storage/datasets/run-1/a").exists()


def test_evaluation_backend_preflight_failure_stops_before_training_or_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bootstrap, "preflight_evaluation_backends",
        lambda config: (_ for _ in ()).throw(ValueError("pose hash mismatch")),
    )
    trainer_calls = []
    pod = Pod()
    with pytest.raises(ValueError, match="pose hash mismatch"):
        run_parallel_bootstrap(
            env=env(tmp_path), hub_client=setup_client(), pod_client=pod,
            queue_factory=lambda path: trainer_calls.append(path), snapshot_fetch=snapshot_fetch,
            recipe_path=Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml",
        )
    assert trainer_calls == []
    assert pod.deleted == []


def test_shipped_manifest_assigns_two_disjoint_three_job_shards():
    path = Path(__file__).parents[2] / "config/examples/klein_automation/parallel-manifest.example.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    shard_a = validate_manifest(manifest, run_id=run_id, shard_id="a")
    shard_b = validate_manifest(manifest, run_id=run_id, shard_id="b")
    assert len(shard_a) == len(shard_b) == 3
    assert {item["id"] for item in shard_a}.isdisjoint(item["id"] for item in shard_b)
