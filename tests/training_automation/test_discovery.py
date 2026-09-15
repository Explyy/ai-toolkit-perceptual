import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from training_automation.backup import BackupError
from training_automation.discovery import (
    WorkflowLedgerStore,
    _bucket,
    assigned_worker,
    scan_dataset_root,
)

FAKE_SPEC = importlib.util.spec_from_file_location(
    "discovery_fakes", Path(__file__).with_name("fakes.py")
)
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient


def _dataset(root: Path, name: str, count: int = 2, size=(640, 960)) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", size, (20 + index, 40, 60)).save(folder / f"image-{index}.png")
        (folder / f"image-{index}.txt").write_text("Owhx subject", encoding="utf-8")
    return folder


def test_bucket_accounting_matches_native_oracle_and_unpadded_batches(tmp_path):
    oracle_path = Path(__file__).parents[2] / "toolkit" / "buckets.py"
    spec = importlib.util.spec_from_file_location("native_bucket_oracle", oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(oracle)
    for width, height in ((640, 960), (1000, 700), (333, 333), (2048, 512)):
        expected = oracle.get_bucket_for_image_size(
            width, height, resolution=768, divisibility=16
        )
        assert _bucket(width, height, 768, 16) == (expected["width"], expected["height"])

    _dataset(tmp_path, "Training_Def_Owhx_Freya", count=3)
    [snapshot] = scan_dataset_root(tmp_path, exposures=126)
    assert snapshot.image_count == 3
    assert snapshot.loader_epochs == 6
    assert snapshot.original_image_exposures == 126
    # Three identical images remain in three resolution buckets. Repeats are
    # unpadded: ceil(48/4) + ceil(12/4) + ceil(3/4) == 16 batches.
    assert snapshot.loader_batches_per_epoch == 16
    assert snapshot.training_steps == 96


def test_scanner_accepts_real_folder_grammar_metadata_and_rejects_symlinks(tmp_path):
    first = _dataset(tmp_path, "Ada Lovelace")
    (first / ".training-automation.json").write_text(json.dumps({
        "catalog_name": "Ada_Lovelace", "trigger_word": "AdaToken"
    }), encoding="utf-8")
    _dataset(tmp_path, "Training_Def_Owhx_Freya")
    snapshots = scan_dataset_root(tmp_path)
    assert {item.folder for item in snapshots} == {
        "Ada Lovelace", "Training_Def_Owhx_Freya"
    }
    ada = next(item for item in snapshots if item.folder == "Ada Lovelace")
    assert ada.name == "Ada_Lovelace" and ada.trigger_word == "AdaToken"
    bad = tmp_path / "Unsafe Folder"
    bad.mkdir()
    (bad / "escape.png").symlink_to(first / "image-0.png")
    (bad / "escape.txt").write_text("caption", encoding="utf-8")
    with pytest.raises(BackupError, match="symlink"):
        scan_dataset_root(tmp_path)


def test_remote_ledger_quiet_period_assignments_completion_and_changed_hold(tmp_path):
    _dataset(tmp_path / "datasets", "Person One")
    _dataset(tmp_path / "datasets", "Person Two")
    snapshots = scan_dataset_root(tmp_path / "datasets")
    client = FakeHubClient()
    store = WorkflowLedgerStore(
        client=client, repo_id="owner/private", repo_type="dataset",
        remote_path="training-automation/workflow-ledger.json",
        local_path=tmp_path / "ledger.json",
    )
    first, _ = store.reconcile(
        snapshots, worker_count=2, quiet_seconds=60, now=100
    )
    assert {item["status"] for item in first["datasets"].values()} == {"observing"}
    second, _ = store.reconcile(
        snapshots, worker_count=2, quiet_seconds=60, now=160
    )
    assert {item["status"] for item in second["datasets"].values()} == {"ready"}
    assert all(item["worker"] == assigned_worker(item["fingerprint"], 2, item["folder"])
               for item in second["datasets"].values())
    selected = snapshots[0]
    store.update_status(selected.folder, selected.fingerprint, "completed")
    same, _ = store.reconcile(snapshots, worker_count=2, quiet_seconds=60, now=220)
    assert same["datasets"][selected.folder]["status"] == "completed"
    caption = tmp_path / "datasets" / selected.folder / "image-0.txt"
    caption.write_text("changed caption", encoding="utf-8")
    changed = scan_dataset_root(tmp_path / "datasets")
    held, _ = store.reconcile(changed, worker_count=2, quiet_seconds=60, now=280)
    assert held["datasets"][selected.folder]["status"] == "changed"


def test_legacy_completed_identity_is_not_requeued(tmp_path):
    _dataset(tmp_path / "datasets", "Training_Def_Owhx_Freya")
    [snapshot] = scan_dataset_root(tmp_path / "datasets")
    client = FakeHubClient()
    store = WorkflowLedgerStore(
        client=client, repo_id="owner/private", repo_type="dataset",
        remote_path="training-automation/workflow-ledger.json",
        local_path=tmp_path / "ledger.json",
    )
    ledger, _ = store.reconcile(
        [snapshot], worker_count=1, quiet_seconds=0, now=1,
        legacy_completed={snapshot.folder: {
            "fingerprint": snapshot.fingerprint,
            "catalog_name": "Freya", "catalog_id": 3,
            "trigger_word": "Owhx", "run_id": "legacy-run",
            "source_dataset_revision": "a" * 40,
            "status": "completed", "legacy": True,
        }},
    )
    assert ledger["datasets"][snapshot.folder]["status"] == "completed"
    assert ledger["datasets"][snapshot.folder]["catalog_name"] == "Freya"
