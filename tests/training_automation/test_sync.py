import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from training_automation.backup import BackupError
from training_automation.sync import sync_latest_loras, sync_ranked_loras


FAKE_SPEC = importlib.util.spec_from_file_location(
    "sync_fakes", Path(__file__).with_name("fakes.py")
)
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient


REVISION = "a" * 40


class SyncFake(FakeHubClient):
    def __init__(self):
        super().__init__()
        self.revision = REVISION
        self.downloads = []

    def download_file(self, repo_id, repo_type, path, revision, destination):
        self.downloads.append(path)
        super().download_file(repo_id, repo_type, path, revision, destination)


def _client() -> tuple[SyncFake, str, bytes]:
    client = SyncFake()
    weight = b"verified-lora"
    weight_sha = hashlib.sha256(weight).hexdigest()
    folder = "0001-ada-lovelace"
    checkpoint_id = "step-000001200-final"
    result_root = f"training-results/run-one/{folder}"
    result_weight = f"{result_root}/top-1/weights/ada.safetensors"
    source_weight = f"training-backups/models/{folder}/checkpoints/{checkpoint_id}/weights/ada.safetensors"
    catalog = {
        "schema_version": 2,
        "models": [{
            "id": 1, "name": "Ada Lovelace", "folder": folder,
            "base_arch": "flux2_klein_9b", "base_model": "base",
            "trigger_word": "AdaToken", "destination_kind": "loras",
            "selected_checkpoint_id": None, "selection": None,
            "checkpoints": [{
                "checkpoint_id": checkpoint_id, "step": 1200, "final": True,
                "revision": REVISION,
                "weights": [{"remote_path": source_weight, "size": len(weight), "sha256": weight_sha}],
            }],
        }],
    }
    index = {
        "schema_version": 1, "run_id": "run-one",
        "model": {"id": 1, "folder": folder},
        "candidates": [{"rank": 1, "catalog_checkpoint_id": checkpoint_id}],
    }
    selection = {
        "schema_version": 1, "rank": 1, "run_id": "run-one",
        "model": {"id": 1}, "catalog_checkpoint_id": checkpoint_id,
        "source_weights": [{
            "source_remote_path": source_weight, "source_revision": REVISION,
            "destination_remote_path": result_weight,
            "size": len(weight), "sha256": weight_sha,
        }],
    }
    pointer = {
        "schema_version": 1, "status": "completed", "run_id": "run-one",
        "model": {"id": 1, "folder": folder},
        "index_path": f"{result_root}/index.json",
    }
    client.remote = {
        "training-backups/catalog.json": json.dumps(catalog).encode(),
        f"{result_root}/index.json": json.dumps(index).encode(),
        f"{result_root}/top-1/selection.json": json.dumps(selection).encode(),
        result_weight: weight,
        f"training-results/latest/{folder}.json": json.dumps(pointer).encode(),
    }
    client.snapshots = {REVISION: dict(client.remote)}
    return client, checkpoint_id, weight


def test_sync_uses_checkpoint_specific_no_clobber_path_and_is_idempotent(tmp_path):
    client, checkpoint_id, weight = _client()
    first = sync_latest_loras(
        client=client, repo_id="owner/private", repo_type="dataset",
        source_revision=REVISION, loras_root=tmp_path / "loras",
        work_dir=tmp_path / "work",
    )
    target = tmp_path / "loras" / "0001-ada-lovelace" / checkpoint_id / "ada.safetensors"
    assert target.read_bytes() == weight
    assert first["counts"]["downloaded"] == 1
    second = sync_latest_loras(
        client=client, repo_id="owner/private", repo_type="dataset",
        source_revision=REVISION, loras_root=tmp_path / "loras",
        work_dir=tmp_path / "work-2",
    )
    assert second["counts"]["reused"] == 1
    target.write_bytes(b"different")
    conflict = sync_latest_loras(
        client=client, repo_id="owner/private", repo_type="dataset",
        source_revision=REVISION, loras_root=tmp_path / "loras",
        work_dir=tmp_path / "work-3",
    )
    assert conflict["status"] == "conflicts"


def test_dry_run_downloads_bytes_when_metadata_hash_is_absent(tmp_path):
    client, _, _ = _client()
    original = client.path_metadata

    def metadata(repo_id, repo_type, paths, revision):
        result = original(repo_id, repo_type, paths, revision)
        for item in result.values():
            item.pop("sha256", None)
        return result

    client.path_metadata = metadata
    result = sync_ranked_loras(
        client=client, repo_id="owner/private", repo_type="dataset",
        source_revision=REVISION, run_id="run-one",
        loras_root=tmp_path / "loras", work_dir=tmp_path / "work",
        model_ids=(1,), dry_run=True,
    )
    assert result["counts"]["planned"] == 1
    assert any(path.endswith("ada.safetensors") for path in client.downloads)
    assert not list((tmp_path / "loras").rglob("*.safetensors"))


def test_latest_sync_skips_models_without_pointer_unless_explicitly_requested(tmp_path):
    client, _, _ = _client()
    client.remote.pop("training-results/latest/0001-ada-lovelace.json")
    client.snapshots[REVISION] = dict(client.remote)
    result = sync_latest_loras(
        client=client, repo_id="owner/private", repo_type="dataset",
        source_revision=REVISION, loras_root=tmp_path / "loras",
        work_dir=tmp_path / "work",
    )
    assert result["skipped"] == [{"model_id": 1, "reason": "latest pointer unavailable"}]
    with pytest.raises(BackupError, match="latest result pointer is missing"):
        sync_latest_loras(
            client=client, repo_id="owner/private", repo_type="dataset",
            source_revision=REVISION, loras_root=tmp_path / "loras",
            work_dir=tmp_path / "work-2", model_ids=(1,),
        )
