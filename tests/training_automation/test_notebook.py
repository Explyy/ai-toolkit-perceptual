import json
import re
from pathlib import Path


def test_unified_notebook_is_clean_pinned_and_uses_same_core():
    root = Path(__file__).parents[2]
    notebook = json.loads((root / "notebooks/Klein_Unified_Training.ipynb").read_text())
    code = "\n".join(
        "".join(cell.get("source") or [])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert re.search(r'SOURCE_REVISION = "[0-9a-f]{40}"', code)
    assert "run_unified_workflow" in code
    assert "sync_latest_loras" in code
    assert "publish_refreshed_report" in code
    assert 'ACTION = "status"' in code
    assert 'RANKS = [1]' in code
    assert "/storage/ComfyUI/models/loras" in code
    assert "getpass.getpass" in code
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "secret" not in code.casefold()
