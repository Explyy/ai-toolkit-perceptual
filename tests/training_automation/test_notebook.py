import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import training_automation.backup as backup
import training_automation.results as results
import training_automation.unified as unified


def _notebook():
    root = Path(__file__).parents[2]
    return json.loads((root / "notebooks/Klein_Unified_Training.ipynb").read_text())


def _code_cells(notebook):
    return [
        "".join(cell.get("source") or [])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]


def _execute(notebook, replacements=()):
    namespace = {}
    for source in _code_cells(notebook):
        for old, new in replacements:
            source = source.replace(old, new)
        exec(compile(source, "Klein_Unified_Training.ipynb", "exec"), namespace)
    return namespace


def test_unified_notebook_is_clean_pinned_and_uses_same_core():
    notebook = _notebook()
    code = "\n".join(_code_cells(notebook))
    assert re.search(r'SOURCE_REVISION = "[0-9a-f]{40}"', code)
    assert "run_unified_workflow" in code
    assert "sync_unified_loras" in code
    assert "refresh_archived_run_reports" in code
    assert "RUN_ID" in code and "HISTORICAL_MODEL_IDS" in code
    assert "'repo_root': TRAINER_ROOT" in code
    assert 'ACTION = "status"' in code
    assert 'RANKS = [1]' in code
    assert "/storage/ComfyUI/models/loras" in code
    assert "getpass.getpass" in code
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "secret" not in code.casefold()


def test_notebook_default_run_all_reads_status_without_clone_or_token(tmp_path, monkeypatch):
    status = {"schema_version": 1, "status": "idle", "attempts": 2}
    work = tmp_path / "work"
    work.mkdir()
    (work / "supervisor-state.json").write_text(json.dumps(status), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("default status action must not invoke a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    namespace = _execute(_notebook(), replacements=((
        'WORK_ROOT = "/storage/automation/unified"', f'WORK_ROOT = {str(work)!r}'
    ),))
    assert namespace["result"] == status


def test_notebook_sync_executes_pinned_setup_and_historical_fallback(tmp_path, monkeypatch):
    notebook = _notebook()
    expected_revision = re.search(
        r'SOURCE_REVISION = "([0-9a-f]{40})"', "\n".join(_code_cells(notebook))
    ).group(1)
    checkout = tmp_path / "checkout"
    work = tmp_path / "work"
    datasets = tmp_path / "datasets"
    loras = tmp_path / "ComfyUI" / "models" / "loras"
    datasets.mkdir()
    loras.mkdir(parents=True)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True)
        if "rev-parse" in command:
            return SimpleNamespace(stdout=expected_revision + "\n")
        return SimpleNamespace(stdout="")

    class Client:
        def __init__(self, token):
            assert token == "test-token"

    captured = {}
    def sync(config_path, **kwargs):
        captured["config"] = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text())))
        captured["kwargs"] = kwargs
        return [{"status": "completed", "files": [{"model_id": 1}]}]

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backup, "HuggingFaceBackupClient", Client)
    monkeypatch.setattr(unified, "sync_unified_loras", sync)
    replacements = (
        ('ACTION = "status"', 'ACTION = "sync"'),
        ('SOURCE_CHECKOUT = "/tmp/ai-toolkit-perceptual-pinned"', f'SOURCE_CHECKOUT = {str(checkout)!r}'),
        ('WORK_ROOT = "/storage/automation/unified"', f'WORK_ROOT = {str(work)!r}'),
        ('DATASET_ROOT = "/storage/datasets"', f'DATASET_ROOT = {str(datasets)!r}'),
        ('COMFYUI_ROOT = "/storage/ComfyUI"', f'COMFYUI_ROOT = {str(tmp_path / "ComfyUI")!r}'),
        ('LORA_ROOT = "/storage/ComfyUI/models/loras"', f'LORA_ROOT = {str(loras)!r}'),
    )
    namespace = _execute(notebook, replacements=replacements)

    assert namespace["result"][0]["status"] == "completed"
    assert any(command[:2] == ["git", "clone"] for command in commands)
    assert any("pip" in command for command in commands)
    assert captured["config"]["queue"]["repo_root"] == "/app/ai-toolkit"
    assert captured["config"]["sync"]["legacy_runs"] == [{
        "run_id": "e372619d-f4dc-4f4b-a2ba-b0df127e24fc",
        "model_ids": [1, 2, 3, 4, 5, 6],
    }]
    assert captured["kwargs"]["ranks"] == [1]


def test_notebook_refresh_uses_run_id_and_hub_archive_only(tmp_path, monkeypatch):
    notebook = _notebook()
    expected_revision = re.search(
        r'SOURCE_REVISION = "([0-9a-f]{40})"', "\n".join(_code_cells(notebook))
    ).group(1)
    checkout = tmp_path / "checkout"
    work = tmp_path / "work"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True)
        if "rev-parse" in command:
            return SimpleNamespace(stdout=expected_revision + "\n")
        return SimpleNamespace(stdout="")

    class Client:
        def __init__(self, token):
            pass

    captured = {}
    def refresh(**kwargs):
        captured.update(kwargs)
        return {"status": "completed", "model_weights_transferred": False}

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backup, "HuggingFaceBackupClient", Client)
    monkeypatch.setattr(results, "refresh_archived_run_reports", refresh)
    namespace = _execute(notebook, replacements=(
        ('ACTION = "status"', 'ACTION = "refresh"'),
        ('SOURCE_CHECKOUT = "/tmp/ai-toolkit-perceptual-pinned"', f'SOURCE_CHECKOUT = {str(checkout)!r}'),
        ('WORK_ROOT = "/storage/automation/unified"', f'WORK_ROOT = {str(work)!r}'),
    ))
    assert namespace["result"]["model_weights_transferred"] is False
    assert captured["run_id"] == "e372619d-f4dc-4f4b-a2ba-b0df127e24fc"
    assert captured["job_ids"] == []
