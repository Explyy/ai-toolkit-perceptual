import json
import importlib.util
from pathlib import Path

import pytest

from training_automation.backup import BackupError
from training_automation.catalog import CatalogStore, restore_generation
FAKE_SPEC = importlib.util.spec_from_file_location("training_automation_test_fakes", Path(__file__).with_name("fakes.py"))
FAKE_MODULE = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKE_MODULE)
FakeHubClient = FAKE_MODULE.FakeHubClient


def store(tmp_path, client):
    return CatalogStore(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog_path="training-backups/catalog.json", work_dir=tmp_path,
    )


def test_catalog_allocates_stable_numeric_id_and_retries_parent_conflict(tmp_path):
    client = FakeHubClient()
    client.catalog_conflicts = 1
    catalog = store(tmp_path, client)
    metadata = {"name": "Character Name", "base_arch": "flux2", "trigger_word": "TOK", "destination_kind": "loras"}
    first, _ = catalog.ensure_model(metadata)
    second, _ = catalog.ensure_model(metadata)
    assert first["id"] == second["id"] == 1
    assert first["folder"] == "0001-character-name"


def test_generation_restore_verifies_and_refuses_conflict(tmp_path):
    client = FakeHubClient()
    payload = b"model weights"
    client.remote["remote/model.safetensors"] = payload
    client.snapshots[client.revision] = dict(client.remote)
    catalog = {
        "schema_version": 1,
        "models": [{
            "id": 1, "name": "character", "folder": "0001-character",
            "destination_kind": "loras", "selected_checkpoint_id": "chosen",
            "checkpoints": [{
                "checkpoint_id": "chosen", "step": 100, "revision": "r0",
                "weights": [{
                    "remote_path": "remote/model.safetensors",
                    "relative_path": "0001-character/model.safetensors",
                    "size": len(payload),
                    "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                }],
            }],
        }],
    }
    root = tmp_path / "ComfyUI" / "models" / "loras"
    result = restore_generation(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog=catalog, identifier="1", roots={"loras": root},
    )
    target = root / "0001-character" / "model.safetensors"
    assert target.read_bytes() == payload
    assert result[0]["status"] == "installed"
    assert restore_generation(
        client=client, repo_id="owner/private", repo_type="dataset",
        catalog=catalog, identifier="character", roots={"loras": root},
    )[0]["status"] == "already-present"
    target.write_bytes(b"different")
    with pytest.raises(BackupError, match="refusing to overwrite"):
        restore_generation(
            client=client, repo_id="owner/private", repo_type="dataset",
            catalog=catalog, identifier="character", roots={"loras": root},
        )


def test_generation_restore_rejects_path_traversal(tmp_path):
    catalog = {
        "schema_version": 1,
        "models": [{
            "id": 1, "name": "bad", "folder": "0001-bad",
            "destination_kind": "loras", "selected_checkpoint_id": "x",
            "checkpoints": [{"checkpoint_id": "x", "revision": "r0", "weights": [{
                "remote_path": "remote/x", "relative_path": "../escape", "size": 1, "sha256": "x"
            }]}],
        }],
    }
    with pytest.raises(Exception, match="unsafe catalog relative path"):
        restore_generation(
            client=FakeHubClient(), repo_id="owner/private", repo_type="dataset",
            catalog=catalog, identifier=1, roots={"loras": tmp_path},
        )
