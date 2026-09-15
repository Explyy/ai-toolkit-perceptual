import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from training_automation.backup import BackupError, sha256_file
from training_automation.dataset_storage import (
    observe_remote_datasets,
    sync_remote_datasets,
    upload_dataset_folder,
)
from training_automation.discovery import scan_dataset_root


FAKE_SPEC = importlib.util.spec_from_file_location(
    "dataset_storage_fakes", Path(__file__).with_name("fakes.py")
)
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient

REVISION_A = "a" * 40
REVISION_B = "b" * 40
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"


def _folder(root: Path, name="Freya", count=2):
    folder = root / name
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (640, 960), (20 + index, 30, 40)).save(
            folder / f"image-{index}.png"
        )
        (folder / f"image-{index}.txt").write_text(
            "Owhx person", encoding="utf-8"
        )
    return folder


def _seed(client, local: Path, *, source_folder="Training_Def_Owhx_Freya"):
    snapshot = scan_dataset_root(local.parent)[0]
    files = []
    for name in snapshot.files:
        payload = (local / name).read_bytes()
        remote = f"datasets/{source_folder}/{name}"
        client.remote[remote] = payload
        files.append({
            "relative_path": name, "remote_path": remote,
            "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        })
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 2,
        "models": [{
            "id": 3, "name": "Freya", "folder": "0003-freya",
            "base_arch": "flux2_klein_9b", "base_model": BASE_MODEL,
            "trigger_word": "Owhx", "destination_kind": "loras",
            "checkpoints": [], "selected_checkpoint_id": None, "selection": None,
        }],
    }).encode()
    client.remote["training-automation/dataset-catalog.json"] = json.dumps({
        "schema_version": 1,
        "datasets": [{
            "id": 3, "name": "Freya", "canonical_folder": "0003-freya",
            "trigger_word": "Owhx", "fingerprint": snapshot.fingerprint,
            "image_count": snapshot.image_count,
            "sources": [{
                "remote_folder": f"datasets/{source_folder}",
                "revision": REVISION_A, "files": files,
            }],
        }],
    }).encode()
    client.revision = REVISION_B
    client.snapshots[REVISION_A] = dict(client.remote)
    client.remote["unrelated-worker-checkpoint.json"] = b"new head, same datasets"
    client.snapshots[REVISION_B] = dict(client.remote)
    return snapshot


def _observe(client, revision, work):
    return observe_remote_datasets(
        client=client, repo_id="owner/private", repo_type="dataset",
        revision=revision, remote_prefix="datasets", work_dir=work,
    )


def _sync(client, first, second, root, work):
    return sync_remote_datasets(
        client=client, repo_id="owner/private", repo_type="dataset",
        first=first, second=second, dataset_root=root, work_dir=work,
        catalog_path="training-automation/dataset-catalog.json",
        model_catalog_path="training-backups/catalog.json",
        base_arch="flux2_klein_9b", base_model=BASE_MODEL,
        minimum_free_bytes=0,
    )


def test_seeded_dataset_installs_canonical_folder_across_unrelated_head_change(tmp_path):
    source = _folder(tmp_path / "source")
    client = FakeHubClient()
    expected = _seed(client, source)
    first = _observe(client, REVISION_A, tmp_path / "observe-a")
    second = _observe(client, REVISION_B, tmp_path / "observe-b")
    assert first["datasets/Training_Def_Owhx_Freya"]["observation_fingerprint"] == second[
        "datasets/Training_Def_Owhx_Freya"
    ]["observation_fingerprint"]

    result = _sync(
        client, first, second, tmp_path / "datasets", tmp_path / "cache"
    )
    assert result["held"] == []
    assert result["installed"] == [{
        "remote_folder": "datasets/Training_Def_Owhx_Freya",
        "canonical_folder": "0003-freya", "dataset_id": 3,
        "status": "installed", "fingerprint": expected.fingerprint,
        "catalog_revision": REVISION_B,
    }]
    [installed] = scan_dataset_root(tmp_path / "datasets")
    assert installed.folder == "0003-freya"
    assert installed.name == "Freya"
    assert installed.fingerprint == expected.fingerprint

    repeated = _sync(
        client, second, second, tmp_path / "datasets", tmp_path / "cache"
    )
    assert repeated["installed"][0]["status"] == "already-present"


def test_identical_renamed_remote_folder_adds_alias_without_new_model(tmp_path):
    source = _folder(tmp_path / "source")
    client = FakeHubClient()
    _seed(client, source)
    for name in scan_dataset_root(source.parent)[0].files:
        client.remote[f"datasets/Freya Renamed/{name}"] = (source / name).read_bytes()
    client.snapshots[REVISION_B] = dict(client.remote)
    observed = _observe(client, REVISION_B, tmp_path / "observe")
    result = _sync(
        client, observed, observed, tmp_path / "datasets", tmp_path / "cache"
    )
    assert not result["held"]
    catalog = json.loads(client.remote["training-automation/dataset-catalog.json"])
    assert len(catalog["datasets"]) == 1
    assert {item["remote_folder"] for item in catalog["datasets"][0]["sources"]} == {
        "datasets/Training_Def_Owhx_Freya", "datasets/Freya Renamed",
    }
    models = json.loads(client.remote["training-backups/catalog.json"])["models"]
    assert [(item["id"], item["name"]) for item in models] == [(3, "Freya")]


def test_malformed_remote_folder_is_held_without_blocking_valid_dataset(tmp_path):
    source = _folder(tmp_path / "source")
    client = FakeHubClient()
    _seed(client, source)
    client.remote["datasets/Incomplete/person.png"] = b"not a complete pair"
    client.snapshots[REVISION_B] = dict(client.remote)
    observed = _observe(client, REVISION_B, tmp_path / "observe")
    assert observed["datasets/Incomplete"]["status"] == "held"
    result = _sync(
        client, observed, observed, tmp_path / "datasets", tmp_path / "cache"
    )
    assert result["installed"][0]["dataset_id"] == 3
    assert result["held"][0]["remote_folder"] == "datasets/Incomplete"


def test_corrupt_download_never_publishes_canonical_folder(tmp_path):
    source = _folder(tmp_path / "source")

    class CorruptClient(FakeHubClient):
        def download_file(self, repo_id, repo_type, path, revision, destination):
            super().download_file(repo_id, repo_type, path, revision, destination)
            if path.endswith("image-0.png"):
                payload = destination.read_bytes()
                destination.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    client = CorruptClient()
    _seed(client, source)
    observed = _observe(client, REVISION_B, tmp_path / "observe")
    result = _sync(
        client, observed, observed, tmp_path / "datasets", tmp_path / "cache"
    )
    assert result["installed"] == []
    assert "verification failed" in result["held"][0]["reason"]
    assert not (tmp_path / "datasets" / "0003-freya").exists()


def test_upload_commits_payload_then_manifest_and_refuses_existing_folder(tmp_path):
    folder = _folder(tmp_path / "source", name="José Persona")

    class ImmutableCommitClient(FakeHubClient):
        def __init__(self):
            super().__init__()
            self.revision = REVISION_A
            self.snapshots = {REVISION_A: {}}
            self.orders = []

        def commit_files(self, repo_id, repo_type, artifacts, message, parent_commit=None):
            self.orders.append([item.remote_path for item in artifacts])
            updated = dict(self.remote)
            for artifact in artifacts:
                updated[artifact.remote_path] = Path(artifact.local_path).read_bytes()
            self.remote = updated
            self.revision = REVISION_B
            self.snapshots[REVISION_B] = dict(updated)
            return REVISION_B

    client = ImmutableCommitClient()
    result = upload_dataset_folder(
        client=client, repo_id="owner/private", repo_type="dataset",
        folder=folder, remote_folder_name="José Persona",
        catalog_name="José Persona", trigger_word="Owhx",
        work_dir=tmp_path / "upload",
    )
    assert result["revision"] == REVISION_B
    assert client.orders[0][-1] == "datasets/José Persona/.training-automation.json"
    manifest = json.loads(client.remote[client.orders[0][-1]])
    assert manifest["catalog_name"] == "José Persona"
    assert len(manifest["files"]) == 4
    with pytest.raises(BackupError, match="refusing to overwrite"):
        upload_dataset_folder(
            client=client, repo_id="owner/private", repo_type="dataset",
            folder=folder, remote_folder_name="José Persona",
            work_dir=tmp_path / "again",
        )
