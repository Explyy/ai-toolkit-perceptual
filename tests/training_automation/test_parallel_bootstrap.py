import hashlib
import importlib.util
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


FAKE_SPEC = importlib.util.spec_from_file_location("parallel_bootstrap_fakes", Path(__file__).with_name("fakes.py"))
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient
MANIFEST_REVISION = "a" * 40
DATASET_REVISION = "b" * 40


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
    ):
        config = yaml.safe_load(Path(path).read_text())
        self.state_path = Path(config["state_path"])
        self.reference_available = reference_available
        self.identity_ranking_status = identity_ranking_status
        self.write_sample = write_sample
        self.jobs = []
        for index, dataset in enumerate(config["datasets"]):
            job_id = f"{dataset['name']}-job"
            config_path = Path(config["generated_dir"]) / f"{job_id}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump({
                "job": "extension",
                "config": {
                    "name": job_id,
                    "process": [{
                        "train": {"steps": 1200},
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
            for step in range(100, 1201, 100):
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
                    })
                report_checkpoints.append({
                    "step": step,
                    "final": step == 1200,
                    "sample_run_status": "complete",
                    "samples": checkpoint_samples,
                })
            (automation / "backup-state.json").write_text(json.dumps({
                "schema_version": 2,
                "checkpoints": {"final": {
                    "status": "backed_up", "verified": True,
                    "cataloged": True, "final": True,
                }},
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


def test_shipped_manifest_assigns_two_disjoint_three_job_shards():
    path = Path(__file__).parents[2] / "config/examples/klein_automation/parallel-manifest.example.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    shard_a = validate_manifest(manifest, run_id=run_id, shard_id="a")
    shard_b = validate_manifest(manifest, run_id=run_id, shard_id="b")
    assert len(shard_a) == len(shard_b) == 3
    assert {item["id"] for item in shard_a}.isdisjoint(item["id"] for item in shard_b)
