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


EVALUATION_COHORT = "subject-likeness-core-v3"
EVALUATION_PROMPT_COUNT = 6
# The fields a declared refinement phase is allowed to change. Pinning them to
# the values of the full run would make a refinement recipe fail the very
# validator that is supposed to protect it, so they are checked as a phase
# instead of compared to the base recipe.
REFINEMENT_FIELDS = (("train", "steps"), ("train", "lr"), ("datasets", "num_repeats"))
BASE_LEARNING_RATE = 0.00005
BASE_RESOLUTION_REPEATS = [16, 4, 1]


def _validate_cohort(document: dict[str, Any], *, prompt_count: int) -> None:
    """Both masked recipes ship the same fixed, seed-pinned evaluation cohort."""
    meta = document.get("meta") or {}
    cases = meta.get("evaluation_cases")
    if meta.get("evaluation_cohort") != EVALUATION_COHORT:
        raise BackupError(f"masked recipe requires the {EVALUATION_COHORT} evaluation cohort")
    if not isinstance(cases, list) or len(cases) != prompt_count:
        raise BackupError(f"masked recipe requires metadata for all {prompt_count} evaluation cases")
    ids = [item.get("case_id") for item in cases]
    seeds = [item.get("seed") for item in cases]
    prompts = document["config"]["process"][0]["sample"]["samples"]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise BackupError("masked evaluation case ids and seeds must be unique")
    if any(prompt.get("seed") != case.get("seed") for prompt, case in zip(prompts, cases)):
        raise BackupError("masked evaluation sample seeds differ from cohort metadata")
    if any("[trigger]" not in str(prompt.get("prompt", "")) for prompt in prompts):
        raise BackupError("every masked evaluation prompt must contain [trigger]")
    if any(
        not str(prompt.get("prompt", "")).startswith("single adult subject [trigger]")
        for prompt in prompts
    ):
        raise BackupError(
            "every masked evaluation prompt must open with 'single adult subject [trigger]'"
        )
    categories = [str(item.get("category") or "") for item in cases]
    for category, wanted in (("full-figure", 2), ("medium", 2), ("face", 2)):
        if categories.count(category) != wanted:
            raise BackupError(
                f"masked evaluation cohort requires exactly {wanted} {category} cases"
            )


def _validate_refinement_phase(process: dict[str, Any]) -> None:
    """Check the deviations a refinement recipe is allowed to carry.

    The refinement pass concentrates the same exposure budget on the highest
    resolution and the low-noise band; it is not a licence to retrain with a
    different budget or a larger step.
    """
    train = process["train"]
    dataset = process["datasets"][0]
    # A refinement pass must concentrate the budget on one band instead of the
    # uniform default, and there are exactly two ways to do it: the cubic
    # sampling of content_or_style: style, which favours the low-noise band, or
    # the centre-dense timestep array of timestep_type: sigmoid, which favours
    # the band where face geometry resolves. A recipe that does neither is a
    # plain rerun at a different resolution and is refused.
    band = (train.get("content_or_style"), train.get("timestep_type"))
    if band not in {("style", "linear"), ("balanced", "sigmoid")}:
        raise BackupError(
            "a refinement recipe must concentrate one noise band: either "
            "content_or_style: style with timestep_type: linear, or "
            "content_or_style: balanced with timestep_type: sigmoid; "
            f"got content_or_style={train.get('content_or_style')!r} "
            f"timestep_type={train.get('timestep_type')!r}"
        )
    lr = train.get("lr")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < BASE_LEARNING_RATE:
        raise BackupError(
            f"a refinement recipe requires a positive learning rate below {BASE_LEARNING_RATE}: {lr!r}"
        )
    repeats = dataset.get("num_repeats")
    resolution = dataset.get("resolution")
    if not isinstance(repeats, list) or sorted(repeats) != sorted(BASE_RESOLUTION_REPEATS):
        raise BackupError(
            f"a refinement recipe must reorder the {BASE_RESOLUTION_REPEATS} exposure budget, "
            f"not replace it: {repeats!r}"
        )
    if repeats == BASE_RESOLUTION_REPEATS:
        raise BackupError("a refinement recipe that keeps the base repeat order refines nothing")
    if not isinstance(resolution, list) or len(resolution) != len(repeats):
        raise BackupError(
            "a refinement recipe needs one num_repeats entry per configured resolution: "
            f"{repeats!r} over {resolution!r}"
        )
    if resolution != sorted(resolution):
        raise BackupError(f"a refinement recipe requires ascending resolutions: {resolution!r}")
    if repeats != sorted(repeats):
        # Comparing only the ends would accept [1, 16, 4], which concentrates the
        # budget at 768px. The weight has to grow with the resolution.
        raise BackupError(
            f"a refinement recipe must weight every higher resolution more than the one "
            f"below it, so num_repeats must ascend: {repeats!r}"
        )
    steps = train.get("steps")
    cadence = (process.get("sample") or {}).get("sample_every")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise BackupError(f"a refinement recipe requires a positive train.steps: {steps!r}")
    if not isinstance(cadence, int) or cadence <= 0 or steps % cadence:
        # The archive gate requires the configured samples at the final step, and
        # that verdict arrives only after the paid run.
        raise BackupError(
            f"a refinement recipe must end on its own {cadence!r}-step sampling cadence: {steps!r}"
        )


def _validate_recipe(
    recipe_path: Path, repo_root: Path, *, prompt_count: int, refinement: bool = False
) -> dict[str, Any]:
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
        ("train", "weight_noise", "bound_norm"): True,
        ("train", "diff_output_preservation"): True,
        ("train", "diff_output_preservation_multiplier"): 1,
        ("train", "diff_output_preservation_class"): "woman",
        ("train", "cache_text_embeddings"): False,
        ("network", "type"): "lokr",
        ("network", "linear"): 32,
        ("datasets", "num_repeats"): [16, 4, 1],
        ("save", "save_every"): 100,
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
    if refinement:
        for path in REFINEMENT_FIELDS:
            expected.pop(path, None)
        _validate_refinement_phase(process)
    for path, wanted in expected.items():
        if path[0] == "datasets":
            actual = process["datasets"][0][path[1]]
        else:
            actual = _at(process, *path)
        if actual != wanted:
            raise BackupError(f"parallel recipe drift at {'.'.join(path)}: {actual!r} != {wanted!r}")
    if len(process["sample"]["samples"]) != prompt_count:
        raise BackupError(f"recipe requires exactly {prompt_count} evaluation prompts")
    _validate_cohort(document, prompt_count=prompt_count)
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
    # bound_norm may live in the trainer or in the config module of the pinned
    # image; requiring it here turns an unsupported key into a build failure
    # instead of a silent or fatal surprise on the first paid GPU step.
    required_weight_noise = ["bound_norm"]
    required_trainer = [
        "from toolkit.subject_mask import cache_subject_masks",
        "from toolkit.depth_consistency import",
        "def _inject_weight_noise",
        "self.subject_mask_config",
        "self.depth_consistency_config",
        "diff_output_preservation requires a trigger_word",
    ]
    missing = [token for token in required_config if token not in config_source]
    missing += [token for token in required_trainer if token not in trainer_source]
    missing += [
        token for token in required_weight_noise
        if token not in f"{config_source}\n{trainer_source}"
    ]
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
    return document


def validate_parallel_recipe(recipe_path: Path, repo_root: Path) -> None:
    _validate_recipe(recipe_path, repo_root, prompt_count=EVALUATION_PROMPT_COUNT)


def validate_unified_recipe(recipe_path: Path, repo_root: Path) -> None:
    _validate_recipe(recipe_path, repo_root, prompt_count=EVALUATION_PROMPT_COUNT)


def validate_refinement_recipe(recipe_path: Path, repo_root: Path) -> None:
    """Validate the recipe of a declared refinement phase.

    Same cohort, same regularization, same exposure budget and same pinned image
    support as the full run; only the phase fields deviate.
    """
    _validate_recipe(
        recipe_path, repo_root, prompt_count=EVALUATION_PROMPT_COUNT, refinement=True
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", action="store_true")
    parser.add_argument("--refinement", action="store_true")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()
    if args.unified and args.refinement:
        parser.error("choose either --unified or --refinement")
    if args.refinement:
        validate_refinement_recipe(args.recipe, args.repo_root)
    elif args.unified:
        validate_unified_recipe(args.recipe, args.repo_root)
    else:
        validate_parallel_recipe(args.recipe, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
