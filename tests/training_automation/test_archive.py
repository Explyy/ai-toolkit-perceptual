import importlib.util
import json
from pathlib import Path

import pytest

from training_automation.backup import BackupError
from training_automation.archive import EvidenceArchive


FAKE_SPEC = importlib.util.spec_from_file_location("parallel_archive_fakes", Path(__file__).with_name("fakes.py"))
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient


class InterruptedArchiveClient(FakeHubClient):
    def __init__(self):
        super().__init__()
        self.failures = 1

    def commit_files(self, repo_id, repo_type, artifacts, message, parent_commit=None):
        if message.startswith("Archive") and self.failures:
            self.failures -= 1
            raise RuntimeError("interrupted evidence upload")
        return super().commit_files(repo_id, repo_type, artifacts, message, parent_commit)


def test_evidence_and_completion_manifest_are_verified_after_retry(tmp_path):
    client = InterruptedArchiveClient()
    report = tmp_path / "evaluation.json"
    sample = tmp_path / "sample.png"
    report.write_text("{}", encoding="utf-8")
    sample.write_bytes(b"pixels")
    archive = EvidenceArchive(
        client=client, repo_id="owner/private", repo_type="dataset",
        remote_prefix="training-runs", run_id="run-1", shard_id="a",
        state_path=tmp_path / "archive-state.json", sleep=lambda _: None,
    )
    state = archive.publish(
        [(report, "job/evaluation.json"), (sample, "job/samples/sample.png")],
        {"job_count": 1},
    )
    assert state["status"] == "completed"
    assert state["attempts"] == 2
    completion = json.loads(client.remote["training-runs/run-1/a/completion.json"])
    assert completion["status"] == "completed"
    assert completion["completion"]["job_count"] == 1
    assert len(completion["evidence"]) == 2


def test_archive_missing_remote_sha_rejects_same_size_download_mismatch(tmp_path):
    class MissingHashClient(FakeHubClient):
        def path_metadata(self, repo_id, repo_type, paths, revision):
            return {
                path: {"size": len(self.snapshots[revision][path]), "sha256": None}
                for path in paths
            }

        def download_file(self, repo_id, repo_type, path, revision, destination):
            payload = self.snapshots[revision][path]
            destination.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    report = tmp_path / "evaluation.json"
    report.write_text("{}", encoding="utf-8")
    archive = EvidenceArchive(
        client=MissingHashClient(), repo_id="owner/private", repo_type="dataset",
        remote_prefix="training-runs", run_id="run-1", shard_id="a",
        state_path=tmp_path / "archive-state.json", max_attempts=1,
    )
    with pytest.raises(BackupError, match="evidence archive failed") as caught:
        archive.publish([(report, "job/evaluation.json")], {"job_count": 1})
    assert "downloaded-byte hash verification" in str(caught.value.__cause__)
    state = json.loads((tmp_path / "archive-state.json").read_text())
    assert state["status"] == "failed"
    assert "downloaded-byte hash verification" in state["error"]
