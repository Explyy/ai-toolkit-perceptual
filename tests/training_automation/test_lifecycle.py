import json

import pytest

from training_automation.backup import BackupError
from training_automation.lifecycle import InstanceBinding, verify_instance_identity, wait_for_binding


class Source:
    def __init__(self, document):
        self.document = document

    def read_remote_file(self, repo_id, repo_type, path):
        return json.dumps(self.document).encode(), "revision"


class Pod:
    def __init__(self, instance):
        self.value = instance

    def instance(self, instance_id):
        return self.value


def binding_document():
    return {
        "schema_version": 1, "run_id": "run-1", "shard_id": "a",
        "instance_id": 42, "instance_hash_id": "hash-42",
        "instance_notes": "training-run:run-1;shard:a",
    }


def test_exact_binding_and_instance_identity_are_required():
    binding = wait_for_binding(
        source=Source(binding_document()), repo_id="owner/private", repo_type="dataset",
        remote_path="bindings/run-1/a.json", run_id="run-1", shard_id="a",
        wait_seconds=0, poll_seconds=1,
    )
    instance = {"id": 42, "hashId": "hash-42", "notes": "training-run:run-1;shard:a"}
    assert verify_instance_identity(Pod(instance), binding) == instance
    with pytest.raises(BackupError, match="hashId"):
        verify_instance_identity(Pod({**instance, "hashId": "other"}), binding)


def test_wrong_shard_binding_is_refused():
    value = binding_document()
    value["shard_id"] = "b"
    with pytest.raises(BackupError, match="run_id or shard_id"):
        wait_for_binding(
            source=Source(value), repo_id="owner/private", repo_type="dataset",
            remote_path="bindings/run-1/a.json", run_id="run-1", shard_id="a",
            wait_seconds=0, poll_seconds=1,
        )


def test_binding_requires_positive_explicit_id():
    value = binding_document()
    value["instance_id"] = 0
    with pytest.raises(BackupError, match="positive numeric"):
        InstanceBinding.from_document(value)
