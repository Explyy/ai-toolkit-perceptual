import json
import urllib.error

import pytest

from training_automation.backup import BackupError
from training_automation.lifecycle import (
    InstanceBinding,
    InstanceIdentityError,
    PermanentApiError,
    SimplePodClient,
    delete_verified_instance,
    verify_instance_identity,
    wait_for_binding,
)


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


class FakeResponse:
    def __init__(self, status: int, payload: bytes = b""):
        self.status = status
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeApi:
    """Answer DELETE and GET from explicit queues; the last answer is sticky."""

    def __init__(self, *, delete, instance=(200,)):
        self.answers = {"DELETE": list(delete), "GET": list(instance)}
        self.calls = []

    def __call__(self, request, timeout=None):
        method = request.get_method()
        self.calls.append(method)
        queue = self.answers[method]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        if int(answer) >= 400:
            raise urllib.error.HTTPError(
                request.full_url, int(answer), "error", {}, None
            )
        return FakeResponse(int(answer))


def pod_client(api, sleeps):
    return SimplePodClient("token", opener=api, sleep=sleeps.append)


def test_delete_retries_a_transport_failure_and_ends_accepted():
    sleeps = []
    api = FakeApi(delete=[urllib.error.URLError("connection reset"), 204])

    assert pod_client(api, sleeps).delete(42) == "accepted"
    assert api.calls == ["DELETE", "GET", "DELETE"]
    assert sleeps == [5.0]


def test_delete_accepts_any_success_status_instead_of_only_204():
    sleeps = []
    api = FakeApi(delete=[200])

    assert pod_client(api, sleeps).delete(42) == "accepted"
    assert api.calls == ["DELETE"]
    assert sleeps == []


def test_delete_counts_a_missing_instance_as_already_deleted():
    sleeps = []
    api = FakeApi(delete=[404])

    assert pod_client(api, sleeps).delete(42) == "already-absent"
    assert sleeps == []


def test_unexpected_delete_answer_is_confirmed_against_the_instance_itself():
    sleeps = []
    api = FakeApi(delete=[500], instance=[404])

    assert pod_client(api, sleeps).delete(42) == "already-absent"
    assert api.calls == ["DELETE", "GET"]
    assert sleeps == []


def test_unconfirmed_delete_raises_only_after_a_bounded_budget():
    sleeps = []
    api = FakeApi(delete=[500], instance=[200])

    with pytest.raises(BackupError, match="unconfirmed after 4 attempts"):
        pod_client(api, sleeps).delete(42)
    assert api.calls.count("DELETE") == 4
    assert sleeps == [5.0, 15.0, 30.0]


def test_instance_verification_still_refuses_an_error_status():
    api = FakeApi(delete=[204], instance=[500])
    with pytest.raises(BackupError, match="failed with HTTP 500"):
        pod_client(api, []).instance(42)


def test_delete_verified_instance_records_the_request_before_issuing_it():
    events = []

    class Pod:
        def instance(self, instance_id):
            events.append(f"verify-{instance_id}")
            return {
                "id": instance_id, "hashId": "hash-42",
                "notes": "training-run:run-1;shard:a",
            }

        def delete(self, instance_id):
            events.append(f"delete-{instance_id}")
            return "accepted"

    binding = InstanceBinding.from_document(binding_document())
    outcome = delete_verified_instance(
        Pod(), binding, before_request=lambda: events.append("record")
    )
    assert outcome == "accepted"
    assert events == ["verify-42", "record", "delete-42"]


def test_refused_credentials_do_not_consume_the_retry_budget():
    sleeps = []
    api = FakeApi(delete=[403], instance=[200])

    with pytest.raises(PermanentApiError, match="refused with HTTP 403"):
        pod_client(api, sleeps).delete(42)
    # One delete, one confirmation read, and no waiting: a revoked token does
    # not heal inside the budget.
    assert api.calls == ["DELETE", "GET"]
    assert sleeps == []


def test_a_refusal_still_yields_to_an_instance_that_is_already_gone():
    sleeps = []
    api = FakeApi(delete=[401], instance=[404])

    assert pod_client(api, sleeps).delete(42) == "already-absent"
    assert sleeps == []


def test_identity_mismatch_is_reported_as_permanent():
    binding = InstanceBinding.from_document(binding_document())
    instance = {"id": 42, "hashId": "other", "notes": "training-run:run-1;shard:a"}

    with pytest.raises(InstanceIdentityError, match="hashId"):
        verify_instance_identity(Pod(instance), binding)
