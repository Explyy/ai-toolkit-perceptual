from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .backup import BackupError


def _at(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def validate_parallel_recipe(recipe_path: Path, repo_root: Path) -> None:
    document = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    process = document["config"]["process"][0]
    expected = {
        ("train", "steps"): 1200,
        ("train", "batch_size"): 4,
        ("train", "optimizer"): "adamw8bit",
        ("train", "lr"): 0.00005,
        ("train", "weight_noise", "enabled"): True,
        ("train", "weight_noise", "mode"): "relative",
        ("train", "weight_noise", "sigma"): 0.0125,
        ("network", "type"): "lokr",
        ("network", "linear"): 32,
        ("datasets", "num_repeats"): [16, 4, 1],
        ("depth_consistency", "loss_weight"): 0.005,
        ("depth_consistency", "mask_source"): "subject",
        ("subject_mask", "enabled"): True,
        ("subject_mask", "background_loss_weight"): 0,
        ("subject_mask", "clothing_loss_weight"): 1,
        ("subject_mask", "body_loss_weight"): 1,
        ("sample", "sample_every"): 100,
        ("sample", "seed"): 42,
        ("sample", "walk_seed"): False,
    }
    for path, wanted in expected.items():
        if path[0] == "datasets":
            actual = process["datasets"][0][path[1]]
        else:
            actual = _at(process, *path)
        if actual != wanted:
            raise BackupError(f"parallel recipe drift at {'.'.join(path)}: {actual!r} != {wanted!r}")
    if len(process["sample"]["samples"]) != 23:
        raise BackupError("parallel recipe requires exactly 23 evaluation prompts")
    config_source = (repo_root / "toolkit" / "config_modules.py").read_text(encoding="utf-8")
    trainer_source = (
        repo_root / "extensions_built_in" / "sd_trainer" / "SDTrainer.py"
    ).read_text(encoding="utf-8")
    klein_source = (
        repo_root / "extensions_built_in" / "diffusion_models" / "flux2"
        / "flux2_klein_model.py"
    ).read_text(encoding="utf-8")
    flux2_source = (
        repo_root / "extensions_built_in" / "diffusion_models" / "flux2"
        / "flux2_model.py"
    ).read_text(encoding="utf-8")
    required_config = ["preprocess_dataset_raw_config", "isinstance(num_repeats, list)", "self.weight_noise"]
    required_trainer = [
        "from toolkit.subject_mask import cache_subject_masks",
        "from toolkit.depth_consistency import",
        "def _inject_weight_noise",
        "self.subject_mask_config",
        "self.depth_consistency_config",
    ]
    missing = [token for token in required_config if token not in config_source]
    missing += [token for token in required_trainer if token not in trainer_source]
    missing += [
        token for token in ["self.model_config.te_name_or_path", "flux2_klein_te_path"]
        if token not in klein_source
    ]
    missing += [
        token for token in [
            "flux-2-klein-base-9b.safetensors",
            "self.model_config.vae_path",
            "load_file(transformer_path",
        ]
        if token not in f"{klein_source}\n{flux2_source}"
    ]
    if missing:
        raise BackupError(f"pinned trainer image lacks selected recipe support: {missing}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()
    validate_parallel_recipe(args.recipe, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
