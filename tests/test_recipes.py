import json

import pytest

from riskaware_eda.recipes import Recipe, generate_recipes, load_recipes, save_recipes


def test_recipe_generation_is_unique_and_deterministic(tmp_path):
    first = generate_recipes(40, 7, seed=11, max_consecutive=2)
    second = generate_recipes(40, 7, seed=11, max_consecutive=2)
    assert first == second
    assert len({recipe.operations for recipe in first}) == 40
    for recipe in first:
        for index in range(len(recipe.operations) - 2):
            assert len(set(recipe.operations[index : index + 3])) > 1

    path = tmp_path / "recipes.json"
    save_recipes(first, path, seed=11)
    assert load_recipes(path) == first
    assert json.loads(path.read_text())["schema_version"] == 1


def test_recipe_rejects_unknown_operator():
    with pytest.raises(ValueError, match="unknown ABC operators"):
        Recipe("bad", ("rewrite", "shell_injection"))
