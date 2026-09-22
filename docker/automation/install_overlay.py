from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_BLOCK = """from training_automation.trainer_hooks import (
    checkpoint_may_be_removed,
    initialize_checkpoint_backup,
    protect_training_checkpoint,
)

"""


def patch_trainer(source: str) -> str:
    """Apply narrow, checked anchors to either supported image trainer lineage."""
    if "initialize_checkpoint_backup" in source:
        required = ("checkpoint_may_be_removed", "protect_training_checkpoint", "checkpoint_path = file_path")
        if not all(item in source for item in required):
            raise RuntimeError("trainer contains an incomplete automation overlay")
        return source
    class_anchor = "class BaseSDTrainProcess(BaseTrainProcess):"
    if source.count(class_anchor) != 1:
        raise RuntimeError("unsupported trainer: BaseSDTrainProcess anchor changed")
    source = source.replace(class_anchor, IMPORT_BLOCK + class_anchor, 1)

    save_config = "        self.save_config = SaveConfig(**self.get_conf('save', {}))\n"
    if source.count(save_config) != 1:
        raise RuntimeError("unsupported trainer: save configuration anchor changed")
    source = source.replace(
        save_config,
        save_config + "        self._training_checkpoint_backup = initialize_checkpoint_backup(self)\n",
        1,
    )

    retention = "            for item in items_to_remove:\n                print_acc(f\"Removing old save: {item}\")"
    if source.count(retention) != 1:
        raise RuntimeError("unsupported trainer: retention anchor changed")
    source = source.replace(
        retention,
        "            for item in items_to_remove:\n"
        "                if not checkpoint_may_be_removed(self, item):\n"
        "                    print_acc(f\"Retaining unbacked save: {item}\")\n"
        "                    continue\n"
        "                print_acc(f\"Removing old save: {item}\")",
        1,
    )

    learnable = "        # save learnable params as json if we have thim\n"
    if source.count(learnable) != 1:
        raise RuntimeError("unsupported trainer: post-model-save anchor changed")
    source = source.replace(learnable, "        checkpoint_path = file_path\n\n" + learnable, 1)

    saved = "        print_acc(f\"Saved checkpoint to {file_path}\")"
    if source.count(saved) != 1:
        raise RuntimeError("unsupported trainer: completed-save anchor changed")
    source = source.replace(
        saved,
        "        print_acc(f\"Saved checkpoint to {checkpoint_path}\")",
        1,
    )
    cleanup = "        self.clean_up_saves()\n        self.post_save_hook(file_path)"
    if source.count(cleanup) != 1:
        raise RuntimeError("unsupported trainer: save cleanup anchor changed")
    source = source.replace(
        cleanup,
        "        protect_training_checkpoint(self, checkpoint_path, step)\n"
        "        self.clean_up_saves()\n"
        "        self.post_save_hook(file_path)",
        1,
    )
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    trainer = args.root / "jobs" / "process" / "BaseSDTrainProcess.py"
    original = trainer.read_text(encoding="utf-8")
    patched = patch_trainer(original)
    if patched != original:
        trainer.write_text(patched, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
