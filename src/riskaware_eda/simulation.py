from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .dataset import read_dataset
from .model import Prediction, RiskModel
from .qor import normalized_qor
from .recipes import Recipe
from .search import RiskAwareSearcher, SearchResult
from .types import NetworkStats, StopCallback, Trajectory, TrajectoryStep


# A compact, named policy registry keeps ablations comparable while preserving
# the production defaults used by the formal experiment runner.
SEARCH_POLICIES: dict[str, dict[str, object]] = {
    "risk_aware": {
        "selection": "lcb",
        "safe_elimination": True,
        "early_stopping": True,
    },
    "random": {
        "selection": "random",
        "safe_elimination": False,
        "early_stopping": False,
    },
    "mean_greedy": {
        "selection": "mean",
        "safe_elimination": False,
        "early_stopping": False,
    },
    "lcb_only": {
        "selection": "lcb",
        "safe_elimination": False,
        "early_stopping": False,
    },
    "selection_only": {
        "selection": "lcb",
        "safe_elimination": True,
        "early_stopping": False,
    },
    "early_stop_only": {
        "selection": "random",
        "safe_elimination": False,
        "early_stopping": True,
    },
}


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


class OracleSimulator:
    """Prepared, reusable offline simulator for one model and one holdout.

    Start-state predictions are computed once. Prefix predictions are cached by
    ``(recipe_id, completed_steps)`` and reused across search seeds and budgets.
    The cache is intentionally scoped to this one oracle/model pair.
    """

    def __init__(
        self,
        oracle: Sequence[Trajectory],
        model: RiskModel,
    ) -> None:
        if not oracle:
            raise ValueError("oracle must contain at least one trajectory")
        circuit_ids = {trajectory.circuit_id for trajectory in oracle}
        if len(circuit_ids) != 1:
            raise ValueError(
                "oracle trajectories must belong to exactly one circuit"
            )
        recipe_ids = [trajectory.recipe_id for trajectory in oracle]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("oracle recipe IDs must be unique")

        self.oracle = tuple(oracle)
        self.model = model
        self.circuit_id = next(iter(circuit_ids))
        self.initial = self.oracle[0].initial
        self.by_recipe = {
            trajectory.recipe_id: trajectory for trajectory in self.oracle
        }
        self.recipes = tuple(
            Recipe(trajectory.recipe_id, trajectory.operations)
            for trajectory in self.oracle
        )
        start_states = [
            Trajectory(
                circuit_id=self.circuit_id,
                recipe_id=recipe.recipe_id,
                operations=recipe.operations,
                initial=self.initial,
            )
            for recipe in self.recipes
        ]
        self.start_predictions = dict(
            zip(
                recipe_ids,
                self.model.predict_trajectories(start_states),
                strict=True,
            )
        )
        self.oracle_scores = {
            trajectory.recipe_id: normalized_qor(
                trajectory.current_stats, trajectory.initial
            )
            for trajectory in self.oracle
        }
        self.oracle_best_recipe_id = min(
            self.oracle_scores, key=self.oracle_scores.get
        )
        self.oracle_best_qor = self.oracle_scores[self.oracle_best_recipe_id]
        self._prefix_predictions: dict[tuple[str, int], Prediction] = {}
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0

    def _predict_trajectory(self, trajectory: Trajectory) -> Prediction:
        key = (trajectory.recipe_id, len(trajectory.steps))
        prediction = self._prefix_predictions.get(key)
        if prediction is None:
            prediction = self.model.predict_trajectory(trajectory)
            self._prefix_predictions[key] = prediction
            self._prefix_cache_misses += 1
        else:
            self._prefix_cache_hits += 1
        return prediction

    def cache_info(self) -> dict[str, int]:
        return {
            "start_predictions": len(self.start_predictions),
            "prefix_entries": len(self._prefix_predictions),
            "prefix_hits": self._prefix_cache_hits,
            "prefix_misses": self._prefix_cache_misses,
        }

    def simulate_budgets(
        self,
        budgets: Sequence[int],
        *,
        seed: int = 0,
        min_steps_before_stopping: int = 2,
        policy: str = "risk_aware",
    ) -> dict[int, SimulationResult]:
        budget_values = tuple(dict.fromkeys(int(item) for item in budgets))
        if not budget_values:
            raise ValueError("at least one budget is required")
        if min(budget_values) < 1:
            raise ValueError("budgets must be positive")
        try:
            policy_options = SEARCH_POLICIES[policy]
        except KeyError as exc:
            raise ValueError(
                f"unknown search policy '{policy}'; "
                f"choose from {sorted(SEARCH_POLICIES)}"
            ) from exc

        def evaluator(
            recipe: Recipe,
            callback: StopCallback | None,
        ) -> Trajectory:
            return _replay(self.by_recipe[recipe.recipe_id], callback)

        searcher = RiskAwareSearcher(
            self.model,
            budget=max(budget_values),
            seed=seed,
            min_steps_before_stopping=min_steps_before_stopping,
            selection=str(policy_options["selection"]),
            safe_elimination=bool(policy_options["safe_elimination"]),
            early_stopping=bool(policy_options["early_stopping"]),
        )
        searches = searcher.run_budgets(
            circuit_id=self.circuit_id,
            initial=self.initial,
            recipes=self.recipes,
            evaluator=evaluator,
            budgets=budget_values,
            start_predictions=self.start_predictions,
            predict_trajectory=self._predict_trajectory,
        )

        rng = random.Random(seed)
        random_order = list(self.oracle)
        rng.shuffle(random_order)
        results: dict[int, SimulationResult] = {}
        for budget in budget_values:
            random_selected = random_order[: min(budget, len(random_order))]
            random_best_trajectory = min(
                random_selected,
                key=lambda item: self.oracle_scores[item.recipe_id],
            )
            random_baseline = BaselineResult(
                best_recipe_id=random_best_trajectory.recipe_id,
                best_qor=self.oracle_scores[random_best_trajectory.recipe_id],
                total_runtime_s=sum(
                    item.cumulative_runtime_s for item in random_selected
                ),
                evaluations=len(random_selected),
            )
            search = searches[budget]
            relative_gap = (
                None
                if search.best_qor is None
                else 100.0
                * (search.best_qor - self.oracle_best_qor)
                / max(abs(self.oracle_best_qor), 1e-12)
            )
            runtime_reduction = 100.0 * (
                1.0
                - search.total_runtime_s
                / max(random_baseline.total_runtime_s, 1e-12)
            )
            results[budget] = SimulationResult(
                search=search,
                random_baseline=random_baseline,
                oracle_best_recipe_id=self.oracle_best_recipe_id,
                oracle_best_qor=self.oracle_best_qor,
                relative_gap_pct=relative_gap,
                runtime_reduction_vs_random_pct=runtime_reduction,
            )
        return results


def simulate_search(
    dataset: str | Path,
    circuit_id: str,
    model: RiskModel,
    *,
    budget: int,
    seed: int = 0,
    min_steps_before_stopping: int = 2,
    policy: str = "risk_aware",
) -> SimulationResult:
    oracle = load_oracle_trajectories(dataset, circuit_id)
    return simulate_search_from_oracle(
        oracle,
        model,
        budget=budget,
        seed=seed,
        min_steps_before_stopping=min_steps_before_stopping,
        policy=policy,
    )


def simulate_search_from_oracle(
    oracle: Sequence[Trajectory],
    model: RiskModel,
    *,
    budget: int,
    seed: int = 0,
    min_steps_before_stopping: int = 2,
    policy: str = "risk_aware",
) -> SimulationResult:
    """Simulate search using an in-memory oracle trajectory set.

    This compatibility entry point prepares a one-shot ``OracleSimulator``.
    Experiment sweeps construct one simulator per holdout so its predictions
    can be reused across budgets and random seeds.
    """
    return OracleSimulator(oracle, model).simulate_budgets(
        (budget,),
        seed=seed,
        min_steps_before_stopping=min_steps_before_stopping,
        policy=policy,
    )[budget]
