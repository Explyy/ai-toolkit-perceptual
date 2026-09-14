from pathlib import Path

from training_automation.recipe import validate_parallel_recipe


def test_selected_masked_klein_recipe_contract_against_fixture(tmp_path):
    repo = tmp_path / "repo"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "jobs/process").mkdir(parents=True)
    (repo / "toolkit/config_modules.py").write_text(
        "def preprocess_dataset_raw_config(): pass\n"
        "if isinstance(num_repeats, list): pass\n"
        "self.weight_noise = value\n",
        encoding="utf-8",
    )
    (repo / "jobs/process/BaseSDTrainProcess.py").write_text(
        "weight_noise depth_consistency subject_mask\n", encoding="utf-8"
    )
    recipe = Path(__file__).parents[2] / "config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml"
    validate_parallel_recipe(recipe, repo)
