from pathlib import Path

from training_automation.recipe import validate_parallel_recipe


def test_selected_masked_klein_recipe_contract_against_fixture(tmp_path):
    repo = tmp_path / "repo"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "extensions_built_in/sd_trainer").mkdir(parents=True)
    (repo / "extensions_built_in/diffusion_models/flux2").mkdir(parents=True)
    (repo / "toolkit/config_modules.py").write_text(
        "def preprocess_dataset_raw_config(): pass\n"
        "if isinstance(num_repeats, list): pass\n"
        "self.weight_noise = value\n",
        encoding="utf-8",
    )
    (repo / "extensions_built_in/sd_trainer/SDTrainer.py").write_text(
        "from toolkit.subject_mask import cache_subject_masks\n"
        "from toolkit.depth_consistency import compute_depth_consistency_loss\n"
        "def _inject_weight_noise(): pass\n"
        "self.subject_mask_config = value\n"
        "self.depth_consistency_config = value\n",
        encoding="utf-8",
    )
    (repo / "extensions_built_in/diffusion_models/flux2/flux2_klein_model.py").write_text(
        "self.model_config.te_name_or_path\n"
        "flux2_klein_te_path\n"
        "flux-2-klein-base-9b.safetensors\n",
        encoding="utf-8",
    )
    (repo / "extensions_built_in/diffusion_models/flux2/flux2_model.py").write_text(
        "self.model_config.vae_path\nload_file(transformer_path, device='cpu')\n",
        encoding="utf-8",
    )
    recipe = Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml"
    validate_parallel_recipe(recipe, repo)
