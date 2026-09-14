import json
import importlib.util
from pathlib import Path

import pytest

from training_automation.backup import BackupConfigurationError, BackupError, CheckpointBackup
from training_automation.catalog import CatalogStore, restore_generation, restore_training
FAKE_SPEC = importlib.util.spec_from_file_location("training_automation_test_fakes", Path(__file__).with_name("fakes.py"))
FAKE_MODULE = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKE_MODULE)
FakeHubClient = FAKE_MODULE.FakeHubClient


def make_backup(tmp_path: Path, client: FakeHubClient, **kwargs) -> CheckpointBackup:
    return CheckpointBackup(
        repo_id=kwargs.pop("repo_id", "owner/private"),
        repo_type=kwargs.pop("repo_type", "dataset"),
        state_path=tmp_path / "backup.json",
        client=client,
        sleep=kwargs.pop("sleep", lambda _: None),
        catalog_metadata={
            "name": "character",
            "base_arch": "flux2_klein_9b",
            "base_model": "black-forest-labs/FLUX.2-klein-base-9B",
            "trigger_word": "TOK",
            "destination_kind": "loras",
        },
        **kwargs,
    )


def test_retries_interrupted_upload_then_verifies_and_catalogs(tmp_path):
    client = FakeHubClient()
    client.fail_backup_commits = 1
    checkpoint = tmp_path / "character_000000100.safetensors"
    optimizer = tmp_path / "optimizer.pt"
    checkpoint.write_bytes(b"weights")
    optimizer.write_bytes(b"optimizer")
    backup = make_backup(tmp_path, client, max_attempts=2)

    commit = backup.protect(
        job_id="job", checkpoint_id="step-000000100", step=100,
        paths=[checkpoint, optimizer], final=False,
    )

    state = json.loads((tmp_path / "backup.json").read_text())
    entry = next(iter(state["checkpoints"].values()))
    assert commit.startswith("r")
    assert entry["status"] == "backed_up"
    assert entry["verified"] is True and entry["cataloged"] is True
    assert entry["attempts"] == 2
    assert backup.can_delete(checkpoint)
    catalog = json.loads(client.remote["training-backups/catalog.json"])
    assert catalog["models"][0]["id"] == 1
    assert catalog["models"][0]["folder"] == "0001-character"
    assert catalog["models"][0]["checkpoints"][0]["revision"] == commit


def test_pending_upload_resumes_after_restart(tmp_path):
    client = FakeHubClient()
    client.fail_backup_commits = 1
    checkpoint = tmp_path / "job_000000001.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    backup = make_backup(tmp_path, client, max_attempts=1)
    with pytest.raises(BackupError):
        backup.protect(
            job_id="job", checkpoint_id="step-000000001", step=1,
            paths=[checkpoint], final=False,
        )
    assert next(iter(json.loads((tmp_path / "backup.json").read_text())["checkpoints"].values()))["status"] == "pending"

    restarted = make_backup(tmp_path, client, max_attempts=2)
    commits = restarted.resume_pending()
    assert len(commits) == 1
    assert next(iter(json.loads((tmp_path / "backup.json").read_text())["checkpoints"].values()))["status"] == "backed_up"


def test_failed_verification_never_marks_backed_up(tmp_path):
    client = FakeHubClient()
    client.corrupt_metadata = True
    checkpoint = tmp_path / "job_000000001.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    backup = make_backup(tmp_path, client, max_attempts=1)
    with pytest.raises(BackupError):
        backup.protect(
            job_id="job", checkpoint_id="step-000000001", step=1,
            paths=[checkpoint], final=False,
        )
    entry = next(iter(json.loads((tmp_path / "backup.json").read_text())["checkpoints"].values()))
    assert entry["status"] == "pending"
    assert not backup.can_delete(checkpoint)


def test_public_destination_fails_closed(tmp_path):
    backup = make_backup(tmp_path, FakeHubClient(private=False))
    with pytest.raises(BackupConfigurationError, match="non-private"):
        backup.validate_destination()


def test_repo_type_is_limited(tmp_path):
    with pytest.raises(BackupConfigurationError):
        make_backup(tmp_path, FakeHubClient(), repo_type="space")


def test_state_is_bound_to_destination_and_legacy_receipts_fail_closed(tmp_path):
    client = FakeHubClient()
    backup = make_backup(tmp_path, client)
    backup._save(backup._state())
    with pytest.raises(BackupConfigurationError, match="different repo_id"):
        make_backup(tmp_path, client, repo_id="owner/other")._state()

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"schema_version": 1, "checkpoints": {"old": {"status": "backed_up"}}}))
    with pytest.raises(BackupConfigurationError, match="legacy backup state"):
        CheckpointBackup(
            repo_id="owner/private", repo_type="dataset", state_path=legacy,
            client=client,
        )._state()


def test_same_step_new_bytes_create_new_immutable_checkpoint(tmp_path):
    client = FakeHubClient()
    checkpoint = tmp_path / "job_000000001.safetensors"
    checkpoint.write_bytes(b"first")
    backup = make_backup(tmp_path, client)
    first = backup.protect(
        job_id="job", checkpoint_id="step-000000001", step=1,
        paths=[checkpoint], final=False,
    )
    checkpoint.write_bytes(b"second")
    second = backup.protect(
        job_id="job", checkpoint_id="step-000000001", step=1,
        paths=[checkpoint], final=False,
    )
    assert first != second
    catalog = json.loads(client.remote["training-backups/catalog.json"])
    checkpoints = catalog["models"][0]["checkpoints"]
    assert len(checkpoints) == 2
    assert checkpoints[0]["checkpoint_id"] != checkpoints[1]["checkpoint_id"]


def test_backup_catalog_roundtrip_separates_generation_and_training_layout(tmp_path):
    client = FakeHubClient()
    checkpoint = tmp_path / "job_000000010.safetensors"
    optimizer = tmp_path / "optimizer.pt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"weights")
    optimizer.write_bytes(b"optimizer")
    config.write_bytes(b"config")
    backup = make_backup(tmp_path, client)
    backup.protect(
        job_id="job", checkpoint_id="step-000000010", step=10,
        paths=[checkpoint, optimizer, config], final=False,
    )
    store = CatalogStore(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog_path="training-backups/catalog.json", work_dir=tmp_path / "catalog-work",
    )
    catalog, _ = store.read()
    generation_root = tmp_path / "comfy-loras"
    generated = restore_generation(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog=catalog, identifier="0001", roots={"loras": generation_root},
        checkpoint_id=catalog["models"][0]["checkpoints"][0]["checkpoint_id"],
    )
    assert Path(generated[0]["path"]).parent == generation_root / "0001-character"
    resume_root = tmp_path / "resume-job-root"
    restored = restore_training(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog=catalog, identifier=1, target_root=resume_root,
        checkpoint_id=catalog["models"][0]["checkpoints"][0]["checkpoint_id"],
    )
    assert {Path(item["path"]).relative_to(resume_root).as_posix() for item in restored} == {
        "job_000000010.safetensors", "optimizer.pt", "config.yaml"
    }


def test_different_jobs_at_same_step_keep_distinct_catalog_checkpoints(tmp_path):
    client = FakeHubClient()
    first = tmp_path / "first_000000010.safetensors"
    second = tmp_path / "second_000000010.safetensors"
    first.write_bytes(b"same weights")
    second.write_bytes(b"same weights")
    backup = make_backup(tmp_path, client)
    backup.protect(
        job_id="first-job", checkpoint_id="step-000000010", step=10,
        paths=[first], final=False,
    )
    backup.protect(
        job_id="second-job", checkpoint_id="step-000000010", step=10,
        paths=[second], final=False,
    )
    catalog = json.loads(client.remote["training-backups/catalog.json"])
    checkpoint_ids = [item["checkpoint_id"] for item in catalog["models"][0]["checkpoints"]]
    assert len(checkpoint_ids) == 2
    assert checkpoint_ids[0].startswith("first-job--")
    assert checkpoint_ids[1].startswith("second-job--")
