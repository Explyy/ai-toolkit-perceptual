import json
import importlib.util
from pathlib import Path

import pytest

from training_automation.backup import BackupConfigurationError, BackupError, CheckpointBackup
FAKE_SPEC = importlib.util.spec_from_file_location("training_automation_test_fakes", Path(__file__).with_name("fakes.py"))
FAKE_MODULE = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKE_MODULE)
FakeHubClient = FAKE_MODULE.FakeHubClient


def make_backup(tmp_path: Path, client: FakeHubClient, **kwargs) -> CheckpointBackup:
    return CheckpointBackup(
        repo_id="owner/private",
        repo_type=kwargs.pop("repo_type", "dataset"),
        state_path=tmp_path / "backup.json",
        client=client,
        sleep=kwargs.pop("sleep", lambda _: None),
        catalog_metadata={
            "name": "character",
            "base_arch": "flux2_klein_9b",
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
    entry = state["checkpoints"]["job:step-000000100"]
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
    assert json.loads((tmp_path / "backup.json").read_text())["checkpoints"]["job:step-000000001"]["status"] == "pending"

    restarted = make_backup(tmp_path, client, max_attempts=2)
    commits = restarted.resume_pending()
    assert len(commits) == 1
    assert json.loads((tmp_path / "backup.json").read_text())["checkpoints"]["job:step-000000001"]["status"] == "backed_up"


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
    entry = json.loads((tmp_path / "backup.json").read_text())["checkpoints"]["job:step-000000001"]
    assert entry["status"] == "pending"
    assert not backup.can_delete(checkpoint)


def test_public_destination_fails_closed(tmp_path):
    backup = make_backup(tmp_path, FakeHubClient(private=False))
    with pytest.raises(BackupConfigurationError, match="non-private"):
        backup.validate_destination()


def test_repo_type_is_limited(tmp_path):
    with pytest.raises(BackupConfigurationError):
        make_backup(tmp_path, FakeHubClient(), repo_type="space")
