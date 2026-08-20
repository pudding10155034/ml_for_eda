import csv
import json
import threading

from riskaware_eda.experiment import ExperimentRunner, load_experiment_config
from riskaware_eda.types import NetworkStats, Trajectory, TrajectoryStep


class _FakeSession:
    def __init__(self, circuit, factory):
        self.circuit = circuit
        self.factory = factory

    def __enter__(self):
        with self.factory.lock:
            self.factory.sessions += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run_recipe(self, recipe):
        circuit_index = int(self.circuit.stem.removeprefix("c"))
        recipe_index = int(recipe.recipe_id.removeprefix("r"))
        initial = NetworkStats(
            pis=8 + circuit_index,
            pos=4,
            nodes=140 + 7 * circuit_index,
            depth=30 + circuit_index,
        )
        trajectory = Trajectory(
            circuit_id=self.circuit.stem,
            recipe_id=recipe.recipe_id,
            operations=recipe.operations,
            initial=initial,
            setup_runtime_s=0.0005,
        )
        cumulative = 0.0
        recipe_gain = 1 + ((recipe_index * 3 + circuit_index) % 7)
        for step, action in enumerate(recipe.operations, start=1):
            runtime = 0.0005 * (1 + recipe_index % 3)
            cumulative += runtime
            trajectory.steps.append(
                TrajectoryStep(
                    step=step,
                    action=action,
                    stats=NetworkStats(
                        pis=initial.pis,
                        pos=initial.pos,
                        nodes=max(1, initial.nodes - step * recipe_gain),
                        depth=max(
                            1,
                            initial.depth
                            - (step // 2)
                            - ((recipe_index + circuit_index) % 3),
                        ),
                    ),
                    runtime_s=runtime,
                    cumulative_runtime_s=cumulative,
                )
            )
        trajectory.completed = True
        return trajectory


class _FakeRunner:
    def __init__(self, factory):
        self.factory = factory

    def session(self, circuit):
        return _FakeSession(circuit, self.factory)


class _FakeRunnerFactory:
    def __init__(self):
        self.sessions = 0
        self.lock = threading.Lock()

    def __call__(self, binary, *, timeout_s, keep_workdir):
        assert timeout_s > 0
        assert keep_workdir is False
        return _FakeRunner(self)


def _write_config(tmp_path):
    (tmp_path / "fake-abc").write_bytes(b"fake-binary")
    circuits = []
    for index in range(4):
        circuit = tmp_path / f"c{index}.aig"
        circuit.write_bytes(b"fake-aiger")
        circuits.append(circuit.name)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "unit-pilot",
                "project_root": ".",
                "output_dir": "out",
                "abc": "fake-abc",
                "circuits": circuits,
                "recipes": {
                    "count": 12,
                    "length": 4,
                    "seeds": [3],
                    "max_consecutive": 2,
                },
                "collection": {"jobs": 2, "timeout_s": 10},
                "training": {
                    "alpha": 0.2,
                    "trees": 16,
                    "min_samples_leaf": 1,
                    "calibration_fraction": 0.34,
                    "model_jobs": 1,
                },
                "evaluation": {
                    "budgets": [4, 8],
                    "search_seeds": [0, 1],
                    "min_stop_step": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_experiment_runner_completes_and_resumes(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    factory = _FakeRunnerFactory()

    first = ExperimentRunner(
        config,
        runner_factory=factory,
    ).run()
    assert first["collection"]["pending"] == 4
    assert first["collection"]["completed"] == 4
    assert first["training"]["trained"] == 4
    assert first["evaluation"]["completed"] == 16
    assert factory.sessions == 4

    with (config.output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert {row["holdout_circuit"] for row in rows} == {
        "c0",
        "c1",
        "c2",
        "c3",
    }

    resumed = ExperimentRunner(
        config,
        resume=True,
        runner_factory=factory,
    ).run()
    assert resumed["collection"]["pending"] == 0
    assert resumed["collection"]["skipped"] == 4
    assert resumed["training"] == {
        "trained": 0,
        "skipped": 4,
        "elapsed_wall_s": 0.0,
    }
    assert resumed["evaluation"]["completed"] == 0
    assert resumed["evaluation"]["skipped"] == 16
    assert factory.sessions == 4

    config.circuits[0].write_bytes(b"updated-aiger")
    invalidated = ExperimentRunner(
        config,
        resume=True,
        runner_factory=factory,
    ).run()
    assert invalidated["collection"]["pending"] == 1
    assert invalidated["collection"]["skipped"] == 3
    assert invalidated["training"]["trained"] == 4
    assert invalidated["evaluation"]["completed"] == 16
    assert factory.sessions == 5


def test_experiment_dry_run_does_not_create_output(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    result = ExperimentRunner(config).run(dry_run=True)
    assert result["dry_run"] is True
    assert result["plan"]["estimated_operator_steps"] == 192
    assert not config.output_dir.exists()
