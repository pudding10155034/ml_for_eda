from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .dataset import read_dataset
from .model import RiskModel
from .qor import normalized_qor
from .recipes import Recipe
from .search import RiskAwareSearcher, SearchResult
from .types import NetworkStats, StopCallback, Trajectory, TrajectoryStep


def load_oracle_trajectories(
    dataset: str | Path,
    circuit_id: str,
) -> list[Trajectory]:
    rows = [
        row
        for row in read_dataset(dataset)
        if row["circuit_id"] == circuit_id and row["completed"] == "1"
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["recipe_id"], []).append(row)
    trajectories: list[Trajectory] = []
    for recipe_id, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["step"]))
        start = group[0]
        operations = tuple(filter(None, start["recipe"].split("|")))
        initial = NetworkStats(
            pis=int(start["pis"]),
            pos=int(start["pos"]),
            nodes=int(start["initial_nodes"]),
            depth=int(start["initial_depth"]),
        )
        trajectory = Trajectory(
            circuit_id=circuit_id,
            recipe_id=recipe_id,
            operations=operations,
            initial=initial,
            completed=True,
        )
        for row in group:
            step = int(row["step"])
            if step == 0:
                continue
            trajectory.steps.append(
                TrajectoryStep(
                    step=step,
                    action=row["action"],
                    stats=NetworkStats(
                        pis=initial.pis,
                        pos=initial.pos,
                        nodes=int(row["nodes"]),
                        depth=int(row["depth"]),
                    ),
                    runtime_s=float(row["runtime_s"]),
                    cumulative_runtime_s=float(row["cumulative_runtime_s"]),
                )
            )
        if len(trajectory.steps) == len(operations):
            trajectories.append(trajectory)
    if not trajectories:
        raise ValueError(f"no complete oracle trajectories for circuit '{circuit_id}'")
    return trajectories


def _replay(
    source: Trajectory,
    stop_callback: StopCallback | None,
) -> Trajectory:
    replay = Trajectory(
        circuit_id=source.circuit_id,
        recipe_id=source.recipe_id,
        operations=source.operations,
        initial=source.initial,
        setup_runtime_s=source.setup_runtime_s,
    )
    for step in source.steps:
        replay.steps.append(step)
        if stop_callback is not None:
            reason = stop_callback(replay)
            if reason:
                replay.stopped_early = True
                replay.stop_reason = reason
                return replay
    replay.completed = True
    return replay


@dataclass(frozen=True)
class BaselineResult:
    best_recipe_id: str
    best_qor: float
    total_runtime_s: float
    evaluations: int


@dataclass(frozen=True)
class SimulationResult:
    search: SearchResult
    random_baseline: BaselineResult
    oracle_best_recipe_id: str
    oracle_best_qor: float
    relative_gap_pct: float | None
    runtime_reduction_vs_random_pct: float

    def to_dict(self) -> dict[str, object]:
        return {
            "search": self.search.to_dict(),
            "random_baseline": asdict(self.random_baseline),
            "oracle_best_recipe_id": self.oracle_best_recipe_id,
            "oracle_best_qor": self.oracle_best_qor,
            "relative_gap_pct": self.relative_gap_pct,
            "runtime_reduction_vs_random_pct": self.runtime_reduction_vs_random_pct,
        }


def simulate_search(
    dataset: str | Path,
    circuit_id: str,
    model: RiskModel,
    *,
    budget: int,
    seed: int = 0,
    min_steps_before_stopping: int = 2,
) -> SimulationResult:
    oracle = load_oracle_trajectories(dataset, circuit_id)
    by_recipe = {trajectory.recipe_id: trajectory for trajectory in oracle}
    recipes = [
        Recipe(trajectory.recipe_id, trajectory.operations) for trajectory in oracle
    ]
    initial = oracle[0].initial
    searcher = RiskAwareSearcher(
        model,
        budget=budget,
        seed=seed,
        min_steps_before_stopping=min_steps_before_stopping,
    )

    def evaluator(recipe: Recipe, callback: StopCallback | None) -> Trajectory:
        return _replay(by_recipe[recipe.recipe_id], callback)

    search = searcher.run(
        circuit_id=circuit_id,
        initial=initial,
        recipes=recipes,
        evaluator=evaluator,
    )
    oracle_scores = {
        trajectory.recipe_id: normalized_qor(
            trajectory.current_stats, trajectory.initial
        )
        for trajectory in oracle
    }
    oracle_best_recipe = min(oracle_scores, key=oracle_scores.get)
    oracle_best_qor = oracle_scores[oracle_best_recipe]

    rng = random.Random(seed)
    random_order = list(oracle)
    rng.shuffle(random_order)
    random_selected = random_order[: min(budget, len(random_order))]
    random_best_trajectory = min(
        random_selected,
        key=lambda item: normalized_qor(item.current_stats, item.initial),
    )
    random_baseline = BaselineResult(
        best_recipe_id=random_best_trajectory.recipe_id,
        best_qor=normalized_qor(
            random_best_trajectory.current_stats, random_best_trajectory.initial
        ),
        total_runtime_s=sum(item.cumulative_runtime_s for item in random_selected),
        evaluations=len(random_selected),
    )
    relative_gap = (
        None
        if search.best_qor is None
        else 100.0 * (search.best_qor - oracle_best_qor) / max(abs(oracle_best_qor), 1e-12)
    )
    runtime_reduction = 100.0 * (
        1.0 - search.total_runtime_s / max(random_baseline.total_runtime_s, 1e-12)
    )
    return SimulationResult(
        search=search,
        random_baseline=random_baseline,
        oracle_best_recipe_id=oracle_best_recipe,
        oracle_best_qor=oracle_best_qor,
        relative_gap_pct=relative_gap,
        runtime_reduction_vs_random_pct=runtime_reduction,
    )
