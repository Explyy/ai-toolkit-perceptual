from pathlib import Path

import yaml

import pytest

from training_automation.backup import BackupError
from training_automation.recipe import validate_parallel_recipe, validate_unified_recipe


def pinned_image(tmp_path, *, omit=()):
    """A stub of the pinned trainer image that supports the selected recipe.

    ``omit`` drops a support token the image would normally carry, which is how
    an image that cannot run the recipe is expressed. recipe.py must fail the
    Docker build on it instead of letting the first paid GPU step find out.
    """
    def lines(*items):
        return "".join(f"{item}\n" for item in items if item not in omit)

    repo = tmp_path / "repo"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "extensions_built_in/sd_trainer").mkdir(parents=True)
    (repo / "extensions_built_in/diffusion_models/flux2").mkdir(parents=True)
    (repo / "toolkit/config_modules.py").write_text(
        lines(
            "def preprocess_dataset_raw_config(): pass",
            "if isinstance(num_repeats, list): pass",
            "self.weight_noise = value",
        ),
        encoding="utf-8",
    )
    (repo / "extensions_built_in/sd_trainer/SDTrainer.py").write_text(
        lines(
            "from toolkit.subject_mask import cache_subject_masks",
            "from toolkit.depth_consistency import compute_depth_consistency_loss",
            "def _inject_weight_noise(): pass",
            "self.subject_mask_config = value",
            "self.depth_consistency_config = value",
            "raise ValueError('diff_output_preservation requires a trigger_word')",
            "self.weight_noise.bound_norm",
        ),
        encoding="utf-8",
    )
    (repo / "extensions_built_in/diffusion_models/flux2/flux2_klein_model.py").write_text(
        lines(
            "self.model_config.te_name_or_path",
            "flux2_klein_te_path",
            "flux-2-klein-base-9b.safetensors",
        ),
        encoding="utf-8",
    )
    (repo / "extensions_built_in/diffusion_models/flux2/flux2_model.py").write_text(
        lines("self.model_config.vae_path", "load_file(transformer_path, device='cpu')"),
        encoding="utf-8",
    )
    return repo


def test_selected_masked_klein_recipe_contract_against_fixture(tmp_path):
    repo = pinned_image(tmp_path)
    recipe = Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml"
    validate_parallel_recipe(recipe, repo)

    unified = Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b-v2.yaml"
    validate_unified_recipe(unified, repo)
    document = yaml.safe_load(unified.read_text())
    process = document["config"]["process"][0]
    prompts = process["sample"]["samples"]
    cases = document["meta"]["evaluation_cases"]
    # subject-likeness-core-v3 is six cases: two full-figure, two medium, two
    # face. The twelve-case articulated-v2 cohort it replaced, and its
    # "deep squat" pose, no longer exist.
    assert len(prompts) == len(cases) == 6
    assert len({item["seed"] for item in prompts}) == 6
    assert [item["category"] for item in cases] == [
        "full-figure", "full-figure", "medium", "medium", "face", "face",
    ]
    assert all("single adult subject [trigger]" in item["prompt"] for item in prompts)
    assert any("full body contrapposto" in item["prompt"] for item in prompts)
    assert any("opposite left arm" in item["prompt"] for item in prompts)


@pytest.mark.parametrize(
    "token, missing",
    [
        ("self.weight_noise.bound_norm", "bound_norm"),
        (
            "raise ValueError('diff_output_preservation requires a trigger_word')",
            "diff_output_preservation requires a trigger_word",
        ),
    ],
)
def test_pinned_image_without_recipe_support_fails_the_build(tmp_path, token, missing):
    """An unsupported field must stop the build, not the first paid step.

    Bounded weight noise and the DOP trigger-word refusal are both silent when
    the image does not implement them: training starts and produces something
    other than the selected recipe. recipe.py runs in the Dockerfile precisely
    so that image never ships.
    """
    repo = pinned_image(tmp_path, omit=(token,))
    recipe = Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml"
    with pytest.raises(BackupError, match="pinned trainer image lacks selected recipe support") as caught:
        validate_parallel_recipe(recipe, repo)
    assert missing in str(caught.value)
