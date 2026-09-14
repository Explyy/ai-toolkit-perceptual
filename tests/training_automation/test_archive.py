import importlib.util
import json
from pathlib import Path

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
