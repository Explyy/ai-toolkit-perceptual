from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .backup import BackupError


SIMPLEPOD_API = "https://api.simplepod.ai"
# One transient answer must not strand a fully archived instance, and a delete
# loop must not bill forever either: the budget is these attempts and no more.
DELETE_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 15.0, 30.0)
# The exact success status of a live delete is not verified, so the class is
# accepted rather than one hard-coded value; these statuses mean the instance
# is not there to delete, which is the same terminal outcome.
ABSENT_STATUS_CODES = frozenset({404, 410})
DELETE_ACCEPTED = "accepted"
DELETE_ALREADY_ABSENT = "already-absent"
# Answers that will not become different answers by waiting: bad credentials,
# a refused method, a malformed request. Retrying these only spends money.
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 405, 422})


class PermanentApiError(BackupError):
    """The provider refused in a way that another attempt cannot change."""


class InstanceIdentityError(BackupError):
    """The instance answering is not the instance this process is bound to."""


class BindingSource(Protocol):
    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...


class SimplePodClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = SIMPLEPOD_API,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        if not token:
            raise BackupError("SIMPLEPOD_API_TOKEN is required for success-only self-delete")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def _send(self, method: str, path: str) -> tuple[int | None, bytes, str | None]:
        """Perform one call and report its outcome instead of raising.

        An HTTP answer is an answer, including an error status: the caller
        decides whether that status is terminal, retryable or proof that the
        instance is already gone. Only a transport failure has no status.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={"X-AUTH-TOKEN": self.token, "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=30) as response:
                return int(response.status), response.read(), None
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            except Exception:
                pass
            return int(exc.code), b"", None
        except urllib.error.URLError as exc:
            return None, b"", str(exc.reason)

    def _request(self, method: str, path: str) -> tuple[int, bytes]:
        status, payload, transport_error = self._send(method, path)
        if transport_error is not None:
            raise BackupError(f"SimplePod {method} {path} transport failure: {transport_error}")
        assert status is not None
        if status in PERMANENT_STATUS_CODES:
            raise PermanentApiError(f"SimplePod {method} {path} failed with HTTP {status}")
        if status >= 400:
            raise BackupError(f"SimplePod {method} {path} failed with HTTP {status}")
        return status, payload

    def instance(self, instance_id: int) -> dict[str, Any]:
        status, payload = self._request("GET", f"/instances/{instance_id}")
        if status != 200:
            raise BackupError(f"SimplePod instance verification returned HTTP {status}")
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise BackupError("SimplePod instance verification returned an invalid document")
        return document

    def instance_absent(self, instance_id: int) -> bool:
        """Ask the instance itself whether it still exists.

        The delete call's own answer is not trusted for this: an unexpected
        status or a dropped connection says nothing about the instance, while
        this read does.
        """
        status, _, transport_error = self._send("GET", f"/instances/{instance_id}")
        return transport_error is None and status in ABSENT_STATUS_CODES

    def delete(
        self,
        instance_id: int,
        *,
        backoff: Sequence[float] = DELETE_RETRY_BACKOFF_SECONDS,
    ) -> str:
        """Request deletion until it is accepted or the instance is observably gone.

        Any 2xx counts as accepted, because the exact success status of this
        endpoint is not verified and a single hard-coded value is what turned a
        transient answer into a permanently undeletable instance. Anything else
        is retried inside a bounded budget, and every failed attempt is followed
        by a direct read of the instance, so a delete that did take effect is
        never reported as uncertain. Only a budget that ends without either
        proof raises, and that remains the genuinely uncertain case.
        """
        schedule = [max(0.0, float(value)) for value in backoff]
        attempts: list[str] = []
        for index in range(len(schedule) + 1):
            status, _, transport_error = self._send("DELETE", f"/instances/{instance_id}")
            if transport_error is None and status is not None:
                if 200 <= status < 300:
                    return DELETE_ACCEPTED
                if status in ABSENT_STATUS_CODES:
                    return DELETE_ALREADY_ABSENT
                if status in PERMANENT_STATUS_CODES:
                    # A revoked token does not heal inside the budget, and the
                    # outer hold must not spend hours discovering that.
                    if self.instance_absent(instance_id):
                        return DELETE_ALREADY_ABSENT
                    raise PermanentApiError(
                        f"SimplePod delete of instance {instance_id} was refused with "
                        f"HTTP {status}"
                    )
                attempts.append(f"HTTP {status}")
            else:
                attempts.append(f"transport failure: {transport_error}")
            if self.instance_absent(instance_id):
                return DELETE_ALREADY_ABSENT
            if index < len(schedule):
                self._sleep(schedule[index])
        raise BackupError(
            f"SimplePod delete of instance {instance_id} is unconfirmed after "
            f"{len(schedule) + 1} attempts: {'; '.join(attempts)}"
        )


@dataclass(frozen=True)
class InstanceBinding:
    run_id: str
    shard_id: str
    instance_id: int
    instance_hash_id: str
    instance_notes: str

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "InstanceBinding":
        if value.get("schema_version") != 1:
            raise BackupError("unsupported instance binding schema")
        instance_id = int(value.get("instance_id", 0))
        if instance_id <= 0:
            raise BackupError("instance binding requires a positive numeric instance_id")
        fields = {
            "run_id": str(value.get("run_id", "")),
            "shard_id": str(value.get("shard_id", "")),
            "instance_hash_id": str(value.get("instance_hash_id", "")),
            "instance_notes": str(value.get("instance_notes", "")),
        }
        if any(not item for item in fields.values()):
            raise BackupError("instance binding is missing an identity field")
        return cls(instance_id=instance_id, **fields)


def wait_for_binding(
    *,
    source: BindingSource,
    repo_id: str,
    repo_type: str,
    remote_path: str,
    run_id: str,
    shard_id: str,
    wait_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> InstanceBinding:
    deadline = monotonic() + max(0.0, float(wait_seconds))
    while True:
        payload, _ = source.read_remote_file(repo_id, repo_type, remote_path)
        if payload is not None:
            binding = InstanceBinding.from_document(json.loads(payload))
            if binding.run_id != run_id or binding.shard_id != shard_id:
                raise BackupError("instance binding run_id or shard_id does not match this process")
            return binding
        if monotonic() >= deadline:
            raise BackupError(f"timed out waiting for exact instance binding at {remote_path}")
        sleep(max(0.1, float(poll_seconds)))


def delete_verified_instance(
    client: SimplePodClient,
    binding: InstanceBinding,
    *,
    before_request: Callable[[], None] | None = None,
) -> str:
    """Stop exactly the bound instance: verify identity, record, then request.

    This is the only definition of that order. The success route and the
    bounded diagnostic hold both go through it, so the hold can never delete an
    instance the success route would have refused to touch, and `before_request`
    is the last durable write of either path.
    """
    verify_instance_identity(client, binding)
    if before_request is not None:
        before_request()
    return str(client.delete(binding.instance_id) or DELETE_ACCEPTED)


def verify_instance_identity(
    client: SimplePodClient, binding: InstanceBinding
) -> dict[str, Any]:
    instance = client.instance(binding.instance_id)
    if int(instance.get("id", 0)) != binding.instance_id:
        raise InstanceIdentityError("SimplePod response instance id does not match the explicit binding")
    if str(instance.get("hashId", "")) != binding.instance_hash_id:
        raise InstanceIdentityError("SimplePod response hashId does not match the explicit binding")
    if str(instance.get("notes", "")) != binding.instance_notes:
        raise InstanceIdentityError("SimplePod response notes do not match the explicit binding")
    return instance
