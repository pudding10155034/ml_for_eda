import json

import pytest

from riskaware_eda.model import Prediction
from riskaware_eda.search import RiskAwareSearcher
from riskaware_eda.simulation import OracleSimulator, SEARCH_POLICIES
from riskaware_eda.types import NetworkStats, Trajectory, TrajectoryStep

from scripts.analyze_results import analyze, bootstrap_ci, validate_grid


class _ConstantModel:
    alpha = 0.1

    def predict_trajectories(self, trajectories):
        return [self.predict_trajectory(item) for item in trajectories]

    def predict_trajectory(self, trajectory):
        return Prediction(mean=1.0, lower=0.0, upper=2.0, scale=0.1)


def _oracle():
    initial = NetworkStats(pis=2, pos=1, nodes=20, depth=5)
    result = []
    for index in range(4):
        trajectory = Trajectory(
            circuit_id="heldout",
            recipe_id=f"r{index:05d}",
            operations=("balance", "rewrite"),
            initial=initial,
            completed=True,
        )
        trajectory.steps.extend(
            [
                TrajectoryStep(
                    step=1,
                    action="balance",
                    stats=NetworkStats(pis=2, pos=1, nodes=18 - index, depth=4),
                    runtime_s=0.01,
                    cumulative_runtime_s=0.01,
                ),
                TrajectoryStep(
                    step=2,
                    action="rewrite",
                    stats=NetworkStats(pis=2, pos=1, nodes=16 - index, depth=3),
                    runtime_s=0.01,
                    cumulative_runtime_s=0.02,
                ),
            ]
        )
        result.append(trajectory)
    return result


def test_bootstrap_is_deterministic_and_grid_validation_detects_missing():
    assert bootstrap_ci([1.0, 2.0, 3.0], seed=5, reps=100) == bootstrap_ci(
        [1.0, 2.0, 3.0], seed=5, reps=100
    )
    rows = [
        {
            "method": "risk_aware",
            "recipe_seed": 0,
            "holdout_circuit": "c",
            "budget": 1,
            "search_seed": 0,
        }
    ]
    with pytest.raises(ValueError, match="incomplete"):
        validate_grid(
            rows,
            methods=["risk_aware"],
            recipe_seeds=[0],
            circuits=["c"],
            budgets=[1, 2],
            search_seeds=[0],
        )


def test_analyzer_writes_machine_report_and_svg(tmp_path):
    source = tmp_path / "simulations" / "seed_00000" / "c"
    source.mkdir(parents=True)
    payload = {
        "recipe_seed": 0,
        "holdout_circuit": "c",
        "budget": 1,
        "search_seed": 0,
        "result": {
            "search": {
                "circuit_id": "c",
                "budget": 1,
                "selected": 1,
                "completed_evaluations": 1,
                "early_stops": 0,
                "candidates_eliminated": 0,
                "best_recipe_id": "r0",
                "best_qor": 1.1,
                "total_runtime_s": 0.5,
                "termination_reason": "budget_exhausted",
            },
            "random_baseline": {
                "best_recipe_id": "r1",
                "best_qor": 1.2,
                "total_runtime_s": 0.6,
            },
            "oracle_best_recipe_id": "r0",
            "oracle_best_qor": 1.0,
            "relative_gap_pct": 10.0,
            "runtime_reduction_vs_random_pct": 16.666,
        },
    }
    (source / "cell.json").write_text(json.dumps(payload), encoding="utf-8")
    report = analyze(
        simulation_root=source.parent.parent.parent,
        output_dir=tmp_path / "analysis",
        bootstrap_reps=10,
    )
    assert report["validation"]["complete"]
    assert (tmp_path / "analysis" / "report.json").is_file()
    assert (tmp_path / "analysis" / "report.md").is_file()
    assert (tmp_path / "analysis" / "validation.md").is_file()
    assert (tmp_path / "analysis" / "figures" / "gap_by_budget.svg").is_file()


def test_policy_registry_and_simulator_baselines():
    assert set(SEARCH_POLICIES) == {
        "risk_aware",
        "random",
        "mean_greedy",
        "lcb_only",
        "selection_only",
        "early_stop_only",
    }
    with pytest.raises(ValueError, match="selection"):
        RiskAwareSearcher(_ConstantModel(), budget=2, selection="invalid")
    simulator = OracleSimulator(_oracle(), _ConstantModel())
    for policy in SEARCH_POLICIES:
        result = simulator.simulate_budgets((2,), seed=3, policy=policy)[2]
        assert len(result.search.evaluations) <= 2
        assert result.search.best_qor is not None
