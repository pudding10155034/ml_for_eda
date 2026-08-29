from collections import Counter

from riskaware_eda.dataset import write_trajectories
from riskaware_eda.model import Prediction, RiskModel, train_risk_model
from riskaware_eda.recipes import generate_recipes
from riskaware_eda.simulation import (
    OracleSimulator,
    SimulationResult,
    load_oracle_trajectories,
    simulate_search,
    simulate_search_from_oracle,
)
from riskaware_eda.synthetic import generate_synthetic_trajectories
from riskaware_eda.types import NetworkStats, Trajectory, TrajectoryStep


class _RuleModel:
    def __init__(self, *, start_lower: float, prefix_lower: float) -> None:
        self.alpha = 0.1
        self.start_lower = start_lower
        self.prefix_lower = prefix_lower
        self.batch_calls = 0
        self.predicted_states: Counter[tuple[object, ...]] = Counter()

    @staticmethod
    def _key(trajectory: Trajectory) -> tuple[object, ...]:
        stats = trajectory.current_stats
        return (
            trajectory.circuit_id,
            trajectory.recipe_id,
            len(trajectory.steps),
            stats.nodes,
            stats.depth,
        )

    def _prediction(self, trajectory: Trajectory) -> Prediction:
        self.predicted_states[self._key(trajectory)] += 1
        lower = self.start_lower if not trajectory.steps else self.prefix_lower
        return Prediction(
            mean=lower + 0.25,
            lower=lower,
            upper=lower + 0.5,
            scale=0.1,
        )

    def predict_trajectories(
        self, trajectories: list[Trajectory]
    ) -> list[Prediction]:
        self.batch_calls += 1
        return [self._prediction(trajectory) for trajectory in trajectories]

    def predict_trajectory(self, trajectory: Trajectory) -> Prediction:
        return self._prediction(trajectory)


def _oracle(recipe_count: int = 4, length: int = 3) -> list[Trajectory]:
    initial = NetworkStats(pis=8, pos=4, nodes=100, depth=20)
    trajectories = []
    for recipe_index in range(recipe_count):
        operations = tuple("balance" for _ in range(length))
        trajectory = Trajectory(
            circuit_id="heldout",
            recipe_id=f"r{recipe_index:05d}",
            operations=operations,
            initial=initial,
            completed=True,
        )
        cumulative = 0.0
        for step, action in enumerate(operations, start=1):
            cumulative += 0.01 * (recipe_index + 1)
            trajectory.steps.append(
                TrajectoryStep(
                    step=step,
                    action=action,
                    stats=NetworkStats(
                        pis=initial.pis,
                        pos=initial.pos,
                        nodes=initial.nodes - step * (recipe_index + 2),
                        depth=initial.depth - step,
                    ),
                    runtime_s=0.01 * (recipe_index + 1),
                    cumulative_runtime_s=cumulative,
                )
            )
        trajectories.append(trajectory)
    return trajectories


def _assert_matches_independent(
    oracle: list[Trajectory],
    *,
    budgets: tuple[int, ...],
    start_lower: float,
    prefix_lower: float,
) -> dict[int, SimulationResult]:
    simulator = OracleSimulator(
        oracle,
        _RuleModel(start_lower=start_lower, prefix_lower=prefix_lower),
    )
    combined = simulator.simulate_budgets(budgets, seed=0)
    for budget, result in combined.items():
        independent = simulate_search_from_oracle(
            oracle,
            _RuleModel(start_lower=start_lower, prefix_lower=prefix_lower),
            budget=budget,
            seed=0,
        )
        assert result.to_dict() == independent.to_dict()
    return combined


def test_cross_circuit_training_and_budgeted_simulation(tmp_path):
    recipes = generate_recipes(36, 6, seed=8)
    trajectories = generate_synthetic_trajectories(
        recipes, circuit_count=8, seed=9
    )
    dataset = tmp_path / "data.csv"
    write_trajectories(trajectories, dataset)

    target = "syn07"
    model, report = train_risk_model(
        dataset,
        excluded_circuits=[target],
        alpha=0.10,
        n_estimators=40,
        seed=3,
    )
    assert target not in report.train_circuits
    assert target not in report.calibration_circuits
    assert report.simultaneous_trajectory_coverage >= 0.85
    assert report.train_rows > report.calibration_rows > 0

    artifact = tmp_path / "model.joblib"
    model.save(artifact)
    restored = RiskModel.load(artifact)
    assert restored.qhat == model.qhat
    assert restored.encoder.feature_names == model.encoder.feature_names

    result = simulate_search(
        dataset,
        target,
        restored,
        budget=10,
        seed=4,
    )
    payload = result.to_dict()
    assert payload["search"]["selected"] <= 10
    assert payload["search"]["completed_evaluations"] >= 1
    assert payload["search"]["best_qor"] is not None
    assert payload["oracle_best_qor"] > 0
    assert payload["random_baseline"]["evaluations"] == 10


def test_multi_budget_preserves_safe_elimination_boundary():
    combined = _assert_matches_independent(
        _oracle(),
        budgets=(2, 1),
        start_lower=2.0,
        prefix_lower=2.0,
    )
    small = combined[1].search
    large = combined[2].search
    assert len(small.evaluations) == 1
    assert small.candidates_eliminated == 0
    assert small.termination_reason == "budget_exhausted"
    assert len(large.evaluations) == 1
    assert large.candidates_eliminated == 3
    assert large.termination_reason == "all_candidates_safely_eliminated"


def test_multi_budget_preserves_candidate_pool_exhaustion():
    combined = _assert_matches_independent(
        _oracle(),
        budgets=(4, 2),
        start_lower=-2.0,
        prefix_lower=-2.0,
    )
    assert combined[2].search.termination_reason == "budget_exhausted"
    assert combined[4].search.termination_reason == "candidate_pool_exhausted"
    assert combined[2].search.evaluations is not combined[4].search.evaluations
    assert len(combined[2].search.evaluations) == 2
    assert len(combined[4].search.evaluations) == 4


def test_multi_budget_preserves_early_stopping():
    combined = _assert_matches_independent(
        _oracle(),
        budgets=(2, 4),
        start_lower=-2.0,
        prefix_lower=2.0,
    )
    assert combined[4].search.completed_evaluations == 1
    assert combined[4].search.early_stops == 3
    assert [
        item.steps_executed for item in combined[4].search.evaluations
    ] == [3, 2, 2, 2]


def test_oracle_simulator_reuses_holdout_predictions():
    model = _RuleModel(start_lower=-2.0, prefix_lower=-2.0)
    simulator = OracleSimulator(_oracle(), model)
    simulator.simulate_budgets((2, 4), seed=0)
    simulator.simulate_budgets((2, 4), seed=1)

    info = simulator.cache_info()
    assert model.batch_calls == 1
    assert info["start_predictions"] == 4
    assert info["prefix_hits"] > 0
    assert info["prefix_entries"] == info["prefix_misses"]
    assert set(model.predicted_states.values()) == {1}


def test_real_model_multi_budget_matches_independent_runs(tmp_path):
    recipes = generate_recipes(24, 5, seed=18)
    trajectories = generate_synthetic_trajectories(
        recipes, circuit_count=6, seed=19
    )
    dataset = tmp_path / "multi-budget.csv"
    write_trajectories(trajectories, dataset)
    target = "syn05"
    model, _ = train_risk_model(
        dataset,
        excluded_circuits=[target],
        alpha=0.1,
        n_estimators=20,
        seed=20,
    )
    oracle = load_oracle_trajectories(dataset, target)
    combined = OracleSimulator(oracle, model).simulate_budgets(
        (3, 6, 10), seed=21
    )
    for budget, result in combined.items():
        independent = simulate_search_from_oracle(
            oracle,
            model,
            budget=budget,
            seed=21,
        )
        assert result.to_dict() == independent.to_dict()
