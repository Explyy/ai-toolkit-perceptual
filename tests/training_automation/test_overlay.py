import importlib.util
from pathlib import Path

import pytest

OVERLAY_PATH = Path(__file__).parents[2] / "docker/automation/install_overlay.py"
SPEC = importlib.util.spec_from_file_location("install_training_overlay", OVERLAY_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
patch_trainer = MODULE.patch_trainer


PINNED_IMAGE_EXCERPT = '''
class BaseSDTrainProcess(BaseTrainProcess):
    def __init__(self):
        self.save_config = SaveConfig(**self.get_conf('save', {}))

    def clean_up_saves(self):
        if os.path.exists(self.save_root):
            for item in items_to_remove:
                print_acc(f"Removing old save: {item}")

    def save(self, step=None):
        # save learnable params as json if we have thim
        print_acc(f"Saved checkpoint to {file_path}")
        self.clean_up_saves()
        self.post_save_hook(file_path)
'''.lstrip()


def test_patches_pinned_image_anchors_without_replacing_trainer():
    patched = patch_trainer(PINNED_IMAGE_EXCERPT)
    assert "initialize_checkpoint_backup(self)" in patched
    assert "checkpoint_may_be_removed(self, item)" in patched
    save_start = patched.index("def save")
    assert patched.index("protect_training_checkpoint(self", save_start) < patched.index("self.clean_up_saves()", save_start)
    assert patch_trainer(patched) == patched


def test_rejects_unknown_or_partial_trainer():
    with pytest.raises(RuntimeError, match="anchor changed"):
        patch_trainer(PINNED_IMAGE_EXCERPT.replace("Saved checkpoint", "Checkpoint saved"))
    with pytest.raises(RuntimeError, match="incomplete"):
        patch_trainer(PINNED_IMAGE_EXCERPT + "\ninitialize_checkpoint_backup\n")


def test_current_checkout_is_already_a_complete_overlay():
    source = (Path(__file__).parents[2] / "jobs/process/BaseSDTrainProcess.py").read_text()
    assert patch_trainer(source) == source


def test_docker_overlay_keeps_gui_default_and_adds_opt_in_parallel_command():
    root = Path(__file__).parents[2]
    dockerfile = (root / "docker/automation/Dockerfile").read_text()
    command = (root / "docker/automation/run_parallel_training.sh").read_text()
    instructions = [line.strip() for line in dockerfile.splitlines() if not line.lstrip().startswith("#")]
    assert "COPY docker/automation/run_parallel_training.sh /run-parallel-training" in dockerfile
    assert not any(line.startswith("CMD ") for line in instructions)
    assert not any(line.startswith("ENTRYPOINT ") for line in instructions)
    assert "python -m training_automation.worker" in command
    assert "InsightFaceCPUBackend" in dockerfile
