from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

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
TrajectoryPredictor = Callable[[Trajectory], Prediction]


def _snapshot_result(
    source: SearchResult,
    *,
    budget: int,
    termination_reason: str,
) -> SearchResult:
    """Copy the mutable search state at an exact budget boundary."""
    return SearchResult(
        circuit_id=source.circuit_id,
        budget=budget,
        risk_alpha=source.risk_alpha,
        evaluations=list(source.evaluations),
        best_recipe_id=source.best_recipe_id,
        best_qor=source.best_qor,
        setup_runtime_s=source.setup_runtime_s,
        candidates_eliminated=source.candidates_eliminated,
        termination_reason=termination_reason,
    )


class RiskAwareSearcher:
    """Configurable budgeted search with optional risk controls.

    The default policy is the production risk-aware method: lower-confidence
    bound selection, conformal safe elimination, and conformal early stopping.
    The explicit switches are intentionally small so baseline and ablation
    runs share exactly the same evaluator and accounting code.
    """

    def __init__(
        self,
        model: RiskModel,
        *,
        budget: int,
        seed: int = 0,
        min_steps_before_stopping: int = 2,
        weights: QoRWeights = QoRWeights(),
        selection: str = "lcb",
        safe_elimination: bool = True,
        early_stopping: bool = True,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        if min_steps_before_stopping < 1:
            raise ValueError("min_steps_before_stopping must be positive")
        if selection not in {"lcb", "mean", "random"}:
            raise ValueError(
                "selection must be one of 'lcb', 'mean', or 'random'"
            )
        self.model = model
        self.budget = budget
        self.seed = seed
        self.min_steps_before_stopping = min_steps_before_stopping
        self.weights = weights
        self.selection = selection
        self.safe_elimination = bool(safe_elimination)
        self.early_stopping = bool(early_stopping)

    def run(
        self,
        *,
        circuit_id: str,
        initial: NetworkStats,
        recipes: Sequence[Recipe],
        evaluator: Evaluator,
        setup_runtime_s: float = 0.0,
        start_predictions: Mapping[str, Prediction] | None = None,
        predict_trajectory: TrajectoryPredictor | None = None,
    ) -> SearchResult:
        return self.run_budgets(
            circuit_id=circuit_id,
            initial=initial,
            recipes=recipes,
            evaluator=evaluator,
            budgets=(self.budget,),
            setup_runtime_s=setup_runtime_s,
            start_predictions=start_predictions,
            predict_trajectory=predict_trajectory,
        )[self.budget]

    def run_budgets(
        self,
        *,
        circuit_id: str,
        initial: NetworkStats,
        recipes: Sequence[Recipe],
        evaluator: Evaluator,
        budgets: Sequence[int],
        setup_runtime_s: float = 0.0,
        start_predictions: Mapping[str, Prediction] | None = None,
        predict_trajectory: TrajectoryPredictor | None = None,
    ) -> dict[int, SearchResult]:
        """Run once and snapshot results at multiple exact budget boundaries.

        Elimination is evaluated at the start of the next search iteration. A
        snapshot is consequently taken immediately after the matching
        evaluation and before a larger budget is allowed to eliminate more
        candidates. This preserves the behavior of independent budget runs.
        """
        if not recipes:
            raise ValueError("candidate recipes cannot be empty")
        requested_budgets = tuple(sorted(set(int(item) for item in budgets)))
        if not requested_budgets:
            raise ValueError("at least one budget is required")
        if requested_budgets[0] < 1:
            raise ValueError("budgets must be positive")
        if requested_budgets[-1] > self.budget:
            raise ValueError("requested budget exceeds searcher budget")
        recipe_ids = [recipe.recipe_id for recipe in recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("candidate recipe IDs must be unique")

        rng = random.Random(self.seed)
        remaining = {recipe.recipe_id: recipe for recipe in recipes}
        if start_predictions is None:
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
                zip(
                    recipe_ids,
                    self.model.predict_trajectories(start_states),
                    strict=True,
                )
            )
        else:
            missing = [item for item in recipe_ids if item not in start_predictions]
            if missing:
                raise ValueError(
                    f"start predictions missing recipe IDs: {missing[:3]}"
                )
            predictions = {item: start_predictions[item] for item in recipe_ids}
        predictor = predict_trajectory or self.model.predict_trajectory
        result = SearchResult(
            circuit_id=circuit_id,
            budget=requested_budgets[-1],
            risk_alpha=self.model.alpha,
            setup_runtime_s=setup_runtime_s,
        )
        snapshots: dict[int, SearchResult] = {}
        requested_set = set(requested_budgets)

        while remaining and len(result.evaluations) < requested_budgets[-1]:
            if self.safe_elimination and result.best_qor is not None:
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

            if self.selection == "random" or result.best_qor is None:
                recipe_id = rng.choice(sorted(remaining))
            elif self.selection == "mean":
                recipe_id = min(
                    remaining,
                    key=lambda item: (predictions[item].mean, item),
                )
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

            def stop_callback(partial: Trajectory) -> str | None:
                if not self.early_stopping:
                    return None
                if result.best_qor is None:
                    return None
                if len(partial.steps) < self.min_steps_before_stopping:
                    return None
                prediction = predictor(partial)
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
                predictor(trajectory) if trajectory.steps else None
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

            selected = len(result.evaluations)
            if selected in requested_set:
                reason = (
                    "candidate_pool_exhausted"
                    if not remaining
                    else "budget_exhausted"
                )
                snapshots[selected] = _snapshot_result(
                    result,
                    budget=selected,
                    termination_reason=reason,
                )

        if not remaining and result.termination_reason == "budget_exhausted":
            result.termination_reason = "candidate_pool_exhausted"
        for budget in requested_budgets:
            if budget not in snapshots:
                snapshots[budget] = _snapshot_result(
                    result,
                    budget=budget,
                    termination_reason=result.termination_reason,
                )
        return {budget: snapshots[budget] for budget in requested_budgets}


def run_live_search(
    *,
    model: RiskModel,
    session: object,
    recipes: Sequence[Recipe],
    budget: int,
    seed: int = 0,
    min_steps_before_stopping: int = 2,
    selection: str = "lcb",
    safe_elimination: bool = True,
    early_stopping: bool = True,
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
        selection=selection,
        safe_elimination=safe_elimination,
        early_stopping=early_stopping,
    )
    return searcher.run(
        circuit_id=circuit_id,
        initial=initial,
        recipes=recipes,
        evaluator=session.run_recipe,
        setup_runtime_s=setup_runtime_s,
    )
