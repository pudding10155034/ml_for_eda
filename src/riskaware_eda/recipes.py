from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


OPERATOR_COMMANDS: dict[str, str] = {
    "balance": "balance",
    "rewrite": "rewrite",
    "rewrite_z": "rewrite -z",
    "refactor": "refactor",
    "refactor_z": "refactor -z",
    "resub": "resub",
    "resub_z": "resub -z",
    "dc2": "dc2",
}

DEFAULT_OPERATORS: tuple[str, ...] = tuple(OPERATOR_COMMANDS)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ValueError("recipe_id cannot be empty")
        if not self.operations:
            raise ValueError("a recipe must contain at least one operation")
        unknown = sorted(set(self.operations) - set(OPERATOR_COMMANDS))
        if unknown:
            raise ValueError(f"unknown ABC operators: {', '.join(unknown)}")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.recipe_id, "operations": list(self.operations)}


def validate_operators(operators: Iterable[str]) -> tuple[str, ...]:
    result = tuple(operators)
    if not result:
        raise ValueError("operator set cannot be empty")
    unknown = sorted(set(result) - set(OPERATOR_COMMANDS))
    if unknown:
        raise ValueError(f"unknown ABC operators: {', '.join(unknown)}")
    return result


def generate_recipes(
    count: int,
    length: int,
    *,
    seed: int = 0,
    operators: Sequence[str] = DEFAULT_OPERATORS,
    max_consecutive: int = 2,
) -> list[Recipe]:
    """Generate deterministic, unique recipes without long repeated runs."""

    if count < 1 or length < 1:
        raise ValueError("count and length must be positive")
    if max_consecutive < 1:
        raise ValueError("max_consecutive must be positive")
    allowed = validate_operators(operators)
    theoretical = len(allowed) ** length
    if count > theoretical:
        raise ValueError("requested more recipes than the operator space contains")

    rng = random.Random(seed)
    recipes: list[Recipe] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = max(10_000, count * 200)
    while len(recipes) < count and attempts < max_attempts:
        attempts += 1
        sequence: list[str] = []
        for _ in range(length):
            choices = list(allowed)
            if len(sequence) >= max_consecutive:
                tail = sequence[-max_consecutive:]
                if len(set(tail)) == 1 and len(choices) > 1:
                    choices.remove(tail[0])
            sequence.append(rng.choice(choices))
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        recipes.append(Recipe(f"r{len(recipes):05d}", key))

    if len(recipes) != count:
        raise RuntimeError("could not generate enough unique recipes")
    return recipes


def save_recipes(
    recipes: Sequence[Recipe],
    path: str | Path,
    *,
    seed: int | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "operators": sorted({op for recipe in recipes for op in recipe.operations}),
        "recipes": [recipe.to_dict() for recipe in recipes],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def load_recipes(path: str | Path) -> list[Recipe]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["recipes"] if isinstance(payload, dict) else payload
    recipes = [
        Recipe(str(row.get("id", f"r{index:05d}")), tuple(row["operations"]))
        for index, row in enumerate(rows)
    ]
    ids = [recipe.recipe_id for recipe in recipes]
    if len(ids) != len(set(ids)):
        raise ValueError("recipe IDs must be unique")
    return recipes
