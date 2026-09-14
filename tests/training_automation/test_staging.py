import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from training_automation.backup import BackupError
from training_automation.staging import PinnedDatasetStager


FAKE_SPEC = importlib.util.spec_from_file_location("parallel_staging_fakes", Path(__file__).with_name("fakes.py"))
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient
REVISION = "a" * 40


def make_stager(tmp_path, client):
    return PinnedDatasetStager(
        client=client, repo_id="owner/private", repo_type="dataset",
        revision=REVISION, target_root=tmp_path / "datasets",
        state_path=tmp_path / "staging.json",
    )


def test_pinned_staging_is_verified_and_idempotent(tmp_path):
    client = FakeHubClient()
    payload = b"private image"
    client.snapshots[REVISION] = {"source/image.png": payload}
    dataset = {"id": "subject-1", "files": [{
        "remote_path": "source/image.png", "relative_path": "images/image.png",
        "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }]}
    first = make_stager(tmp_path, client).stage_dataset(dataset)
    second = make_stager(tmp_path, client).stage_dataset(dataset)
    assert first["files"][0]["install_status"] == "installed"
    assert second["files"][0]["install_status"] == "already-present"
    assert (tmp_path / "datasets/subject-1/images/image.png").read_bytes() == payload


def test_staging_failure_is_durable_and_paths_are_safe(tmp_path):
    client = FakeHubClient()
    client.snapshots[REVISION] = {"source/image.png": b"wrong"}
    dataset = {"id": "subject-1", "files": [{
        "remote_path": "source/image.png", "relative_path": "image.png",
        "size": 5, "sha256": hashlib.sha256(b"right").hexdigest(),
    }]}
    with pytest.raises(BackupError, match="verification failed"):
        make_stager(tmp_path, client).stage_dataset(dataset)
    state = json.loads((tmp_path / "staging.json").read_text())
    assert next(iter(state["files"].values()))["status"] == "failed"
    dataset["files"][0]["relative_path"] = "../escape"
    with pytest.raises(BackupError, match="unsafe"):
        make_stager(tmp_path, client).stage_dataset(dataset)
