import csv

import numpy as np

from riskaware_eda.dataset import write_trajectories
from riskaware_eda.features import FeatureEncoder
from riskaware_eda.recipes import generate_recipes
from riskaware_eda.synthetic import generate_synthetic_trajectories


def test_dataset_contains_prefixes_and_final_labels(tmp_path):
    recipes = generate_recipes(3, 5, seed=2)
    trajectories = generate_synthetic_trajectories(
        recipes, circuit_count=2, seed=4
    )
    path = tmp_path / "trajectories.csv"
    report = write_trajectories(trajectories, path)
    assert report == {
        "trajectories": 6,
        "completed": 6,
        "failed": 0,
        "rows": 36,
    }

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row["step"]) for row in rows} == set(range(6))
    assert all(row["final_qor"] for row in rows)

    encoder = FeatureEncoder(max_recipe_length=5)
    vector = encoder.encode_row(rows[0])
    assert vector.shape == (len(encoder.feature_names),)
    assert np.isfinite(vector).all()
