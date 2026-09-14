import json
import importlib.util
from pathlib import Path

import pytest

from training_automation.backup import BackupError
from training_automation.catalog import CatalogStore, restore_generation, restore_training
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
    metadata = {
        "name": "Character Name", "base_arch": "flux2",
        "base_model": "owner/exact-base", "trigger_word": "TOK",
        "destination_kind": "loras",
    }
    first, _ = catalog.ensure_model(metadata)
    second, _ = catalog.ensure_model(metadata)
    assert first["id"] == second["id"] == 1
    assert first["folder"] == "0001-character-name"


def test_schema_one_catalog_backfills_exact_base_model_without_renumbering(tmp_path):
    client = FakeHubClient()
    client.remote["training-backups/catalog.json"] = json.dumps({
        "schema_version": 1,
        "models": [{
            "id": 7, "name": "legacy", "folder": "0007-legacy",
            "base_arch": "flux2", "trigger_word": "TOK",
            "destination_kind": "loras", "checkpoints": [],
            "selected_checkpoint_id": None,
        }],
    }).encode()
    client.snapshots["r0"] = dict(client.remote)
    model, _ = store(tmp_path, client).ensure_model({
        "name": "legacy", "base_arch": "flux2",
        "base_model": "owner/exact-base", "trigger_word": "TOK",
        "destination_kind": "loras",
    })
    assert model["id"] == 7
    assert model["folder"] == "0007-legacy"
    saved = json.loads(client.remote["training-backups/catalog.json"])
    assert saved["schema_version"] == 2
    assert saved["models"][0]["base_model"] == "owner/exact-base"


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
        catalog=catalog, identifier="0001", roots={"loras": root},
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


def test_restore_does_not_clobber_file_created_during_download(tmp_path):
    payload = b"wanted"
    target = tmp_path / "loras" / "0001-character" / "model.safetensors"

    class RacingClient(FakeHubClient):
        def download_file(self, repo_id, repo_type, path, revision, destination):
            super().download_file(repo_id, repo_type, path, revision, destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"concurrent")

    client = RacingClient()
    client.remote["remote/model"] = payload
    client.snapshots["r0"] = dict(client.remote)
    catalog = {
        "schema_version": 2,
        "models": [{
            "id": 1, "name": "character", "folder": "0001-character",
            "destination_kind": "loras", "selected_checkpoint_id": "selected",
            "checkpoints": [{
                "checkpoint_id": "selected", "revision": "r0",
                "weights": [{
                    "remote_path": "remote/model",
                    "generation_relative_path": "0001-character/model.safetensors",
                    "size": len(payload),
                    "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                }],
            }],
        }],
    }
    with pytest.raises(BackupError, match="concurrently created"):
        restore_generation(
            client=client, repo_id="owner/private", repo_type="dataset",
            catalog=catalog, identifier=1, roots={"loras": tmp_path / "loras"},
        )
    assert target.read_bytes() == b"concurrent"


def test_imported_weight_does_not_claim_stateful_training_resume(tmp_path):
    client = FakeHubClient()
    client.remote["legacy/model.safetensors"] = b"legacy"
    client.snapshots["r0"] = dict(client.remote)
    catalog_store = store(tmp_path, client)
    catalog_store.import_existing(
        metadata={
            "name": "legacy", "base_arch": "flux2",
            "base_model": "owner/exact-base", "trigger_word": None,
            "destination_kind": "loras",
        },
        remote_path="legacy/model.safetensors", checkpoint_id="imported",
        step=0, final=True,
    )
    catalog, _ = catalog_store.read()
    with pytest.raises(BackupError, match="weights-only"):
        restore_training(
            client=client, repo_id="owner/private", repo_type="dataset",
            catalog=catalog, identifier="legacy", target_root=tmp_path / "resume",
        )


def test_selection_persists_immutable_revision_and_report_evidence(tmp_path):
    client = FakeHubClient()
    catalog_store = store(tmp_path, client)
    catalog_store.ensure_model({
        "name": "character", "base_arch": "flux2_klein_9b",
        "base_model": "owner/exact-base", "trigger_word": "TOK",
        "destination_kind": "loras",
    })
    catalog_store.add_checkpoint("character", {
        "checkpoint_id": "qualified-checkpoint", "step": 100,
        "final": False, "revision": "artifact-revision",
        "weights": [], "resume_artifacts": [],
    })
    catalog_store.select("0001", "qualified-checkpoint", evidence={
        "report_sha256": "abc", "selected_step": 100,
    })
    catalog, _ = catalog_store.read()
    selection = catalog["models"][0]["selection"]
    assert selection == {
        "checkpoint_id": "qualified-checkpoint",
        "checkpoint_revision": "artifact-revision",
        "evidence": {"report_sha256": "abc", "selected_step": 100},
    }
