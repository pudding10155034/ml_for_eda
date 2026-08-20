from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

from .model import Prediction, RiskModel
from .qor import QoRWeights, normalized_qor
from .recipes import Recipe
from .types import NetworkStats, StopCallback, Trajectory


@dataclass(frozen=True)
class EvaluationRecord:
    order: int
    recipe_id: str
    start_prediction: Prediction
    end_prediction: Prediction | None
    completed: bool
    stopped_early: bool
    steps_executed: int
    runtime_s: float
    observed_qor: float | None
    incumbent_qor: float | None
    stop_reason: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


@dataclass
class SearchResult:
    circuit_id: str
    budget: int
    risk_alpha: float
    evaluations: list[EvaluationRecord] = field(default_factory=list)
    best_recipe_id: str | None = None
    best_qor: float | None = None
    setup_runtime_s: float = 0.0
    candidates_eliminated: int = 0
    termination_reason: str = "budget_exhausted"

    @property
    def total_runtime_s(self) -> float:
        return self.setup_runtime_s + sum(item.runtime_s for item in self.evaluations)

    @property
    def completed_evaluations(self) -> int:
        return sum(item.completed for item in self.evaluations)

    @property
    def early_stops(self) -> int:
        return sum(item.stopped_early for item in self.evaluations)

    def to_dict(self) -> dict[str, object]:
        return {
            "circuit_id": self.circuit_id,
            "budget": self.budget,
            "risk_alpha": self.risk_alpha,
            "selected": len(self.evaluations),
            "completed_evaluations": self.completed_evaluations,
            "early_stops": self.early_stops,
            "candidates_eliminated": self.candidates_eliminated,
            "best_recipe_id": self.best_recipe_id,
            "best_qor": self.best_qor,
            "setup_runtime_s": self.setup_runtime_s,
            "total_runtime_s": self.total_runtime_s,
            "termination_reason": self.termination_reason,
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


Evaluator = Callable[[Recipe, StopCallback | None], Trajectory]


class RiskAwareSearcher:
    """Optimistic selection plus conformal safe-elimination and early stopping."""

    def __init__(
        self,
        model: RiskModel,
        *,
        budget: int,
        seed: int = 0,
        min_steps_before_stopping: int = 2,
        weights: QoRWeights = QoRWeights(),
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        if min_steps_before_stopping < 1:
            raise ValueError("min_steps_before_stopping must be positive")
        self.model = model
        self.budget = budget
        self.seed = seed
        self.min_steps_before_stopping = min_steps_before_stopping
        self.weights = weights

    def run(
        self,
        *,
        circuit_id: str,
        initial: NetworkStats,
        recipes: Sequence[Recipe],
        evaluator: Evaluator,
        setup_runtime_s: float = 0.0,
    ) -> SearchResult:
        if not recipes:
            raise ValueError("candidate recipes cannot be empty")
        recipe_ids = [recipe.recipe_id for recipe in recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("candidate recipe IDs must be unique")

        rng = random.Random(self.seed)
        remaining = {recipe.recipe_id: recipe for recipe in recipes}
        start_states = [
            Trajectory(
                circuit_id=circuit_id,
                recipe_id=recipe.recipe_id,
                operations=recipe.operations,
                initial=initial,
            )
            for recipe in recipes
        ]
        predictions = dict(
            zip(recipe_ids, self.model.predict_trajectories(start_states), strict=True)
        )
        result = SearchResult(
            circuit_id=circuit_id,
            budget=self.budget,
            risk_alpha=self.model.alpha,
            setup_runtime_s=setup_runtime_s,
        )

        while remaining and len(result.evaluations) < self.budget:
            if result.best_qor is not None:
                eliminated = [
                    recipe_id
                    for recipe_id in remaining
                    if predictions[recipe_id].lower > result.best_qor
                ]
                for recipe_id in eliminated:
                    remaining.pop(recipe_id)
                result.candidates_eliminated += len(eliminated)
                if not remaining:
                    result.termination_reason = "all_candidates_safely_eliminated"
                    break

            if result.best_qor is None:
                recipe_id = rng.choice(sorted(remaining))
            else:
                recipe_id = min(
                    remaining,
                    key=lambda item: (
                        predictions[item].lower,
                        predictions[item].mean,
                        item,
                    ),
                )
            recipe = remaining.pop(recipe_id)
            incumbent_before = result.best_qor

            def stop_callback(partial: Trajectory) -> str | None:
                if result.best_qor is None:
                    return None
                if len(partial.steps) < self.min_steps_before_stopping:
                    return None
                prediction = self.model.predict_trajectory(partial)
                if prediction.lower > result.best_qor:
                    return "conformal_lower_bound_above_incumbent"
                return None

            trajectory = evaluator(recipe, stop_callback)
            observed_qor = None
            if trajectory.completed:
                observed_qor = normalized_qor(
                    trajectory.current_stats, trajectory.initial, self.weights
                )
                if result.best_qor is None or observed_qor < result.best_qor:
                    result.best_qor = observed_qor
                    result.best_recipe_id = recipe.recipe_id
            end_prediction = (
                self.model.predict_trajectory(trajectory) if trajectory.steps else None
            )
            result.evaluations.append(
                EvaluationRecord(
                    order=len(result.evaluations) + 1,
                    recipe_id=recipe.recipe_id,
                    start_prediction=predictions[recipe.recipe_id],
                    end_prediction=end_prediction,
                    completed=trajectory.completed,
                    stopped_early=trajectory.stopped_early,
                    steps_executed=len(trajectory.steps),
                    runtime_s=trajectory.cumulative_runtime_s,
                    observed_qor=observed_qor,
                    incumbent_qor=result.best_qor,
                    stop_reason=trajectory.stop_reason,
                    error=trajectory.error,
                )
            )

        if not remaining and result.termination_reason == "budget_exhausted":
            result.termination_reason = "candidate_pool_exhausted"
        return result


def run_live_search(
    *,
    model: RiskModel,
    session: object,
    recipes: Sequence[Recipe],
    budget: int,
    seed: int = 0,
    min_steps_before_stopping: int = 2,
) -> SearchResult:
    """Run search against an ``ABCSession`` without coupling core logic to ABC."""

    initial = getattr(session, "initial_stats", None)
    if not isinstance(initial, NetworkStats):
        raise TypeError("session must expose initialized NetworkStats")
    circuit = getattr(session, "circuit", None)
    circuit_id = getattr(circuit, "stem", "circuit")
    setup_runtime_s = float(getattr(session, "setup_runtime_s", 0.0))
    searcher = RiskAwareSearcher(
        model,
        budget=budget,
        seed=seed,
        min_steps_before_stopping=min_steps_before_stopping,
    )
    return searcher.run(
        circuit_id=circuit_id,
        initial=initial,
        recipes=recipes,
        evaluator=session.run_recipe,
        setup_runtime_s=setup_runtime_s,
    )
