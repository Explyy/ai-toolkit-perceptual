from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .backup import BackupError


SIMPLEPOD_API = "https://api.simplepod.ai"


class BindingSource(Protocol):
    def read_remote_file(
        self, repo_id: str, repo_type: str, path: str
    ) -> tuple[bytes | None, str]: ...


class SimplePodClient:
    def __init__(self, token: str, *, base_url: str = SIMPLEPOD_API):
        if not token:
            raise BackupError("SIMPLEPOD_API_TOKEN is required for success-only self-delete")
        self.token = token
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={"X-AUTH-TOKEN": self.token, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            exc.read()
            raise BackupError(f"SimplePod {method} {path} failed with HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise BackupError(f"SimplePod {method} {path} transport failure: {exc.reason}") from None

    def instance(self, instance_id: int) -> dict[str, Any]:
        status, payload = self._request("GET", f"/instances/{instance_id}")
        if status != 200:
            raise BackupError(f"SimplePod instance verification returned HTTP {status}")
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise BackupError("SimplePod instance verification returned an invalid document")
        return document

    def delete(self, instance_id: int) -> None:
        status, _ = self._request("DELETE", f"/instances/{instance_id}")
        if status != 204:
            raise BackupError(f"SimplePod delete returned HTTP {status}")


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


def verify_instance_identity(
    client: SimplePodClient, binding: InstanceBinding
) -> dict[str, Any]:
    instance = client.instance(binding.instance_id)
    if int(instance.get("id", 0)) != binding.instance_id:
        raise BackupError("SimplePod response instance id does not match the explicit binding")
    if str(instance.get("hashId", "")) != binding.instance_hash_id:
        raise BackupError("SimplePod response hashId does not match the explicit binding")
    if str(instance.get("notes", "")) != binding.instance_notes:
        raise BackupError("SimplePod response notes do not match the explicit binding")
    return instance
