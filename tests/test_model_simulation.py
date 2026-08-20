from riskaware_eda.dataset import write_trajectories
from riskaware_eda.model import RiskModel, train_risk_model
from riskaware_eda.recipes import generate_recipes
from riskaware_eda.simulation import simulate_search
from riskaware_eda.synthetic import generate_synthetic_trajectories


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
