import csv
import json
import threading

import pytest

from riskaware_eda.experiment import (
    ExperimentRunner,
    _ExperimentLock,
    load_experiment_config,
)
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


class _RecordingSession(_FakeSession):
    def run_recipe(self, recipe):
        key = (self.circuit.stem, recipe.recipe_id)
        with self.factory.lock:
            self.factory.calls.append(key)
            should_fail = self.factory.fail_recipe == key
            if should_fail:
                self.factory.fail_recipe = None
        if should_fail:
            raise RuntimeError("injected interruption")
        return super().run_recipe(recipe)


class _RecordingRunner(_FakeRunner):
    def session(self, circuit):
        return _RecordingSession(circuit, self.factory)


class _RecordingRunnerFactory(_FakeRunnerFactory):
    def __init__(self, fail_recipe=None):
        super().__init__()
        self.fail_recipe = fail_recipe
        self.calls = []

    def __call__(self, binary, *, timeout_s, keep_workdir):
        assert timeout_s > 0
        assert keep_workdir is False
        return _RecordingRunner(self)


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


def test_evaluation_resume_repairs_only_missing_budget_artifacts(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    factory = _FakeRunnerFactory()
    ExperimentRunner(config, runner_factory=factory).run()

    simulation_root = config.output_dir / "simulations" / "seed_00003"
    legacy = simulation_root / "c1" / "budget_0004_search_00000.json"
    legacy_payload = json.loads(legacy.read_text(encoding="utf-8"))
    legacy_payload.pop("simulation_artifact_schema_version")
    legacy_payload.pop("oracle")
    legacy_payload.pop("oracle_signature")
    legacy.write_text(
        json.dumps(legacy_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    paths = sorted(simulation_root.rglob("*.json"))
    assert len(paths) == 16
    low = simulation_root / "c0" / "budget_0004_search_00000.json"
    high = simulation_root / "c0" / "budget_0008_search_00000.json"
    expected = {
        path: json.loads(path.read_text(encoding="utf-8"))["result"]
        for path in (low, high)
    }
    untouched = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths
        if path not in {low, high}
    }
    low.unlink()
    high.write_text('{"torn":', encoding="utf-8")

    resumed = ExperimentRunner(
        config,
        resume=True,
        runner_factory=factory,
    ).run(phase="evaluate")

    assert resumed["evaluation"]["completed"] == 2
    assert resumed["evaluation"]["skipped"] == 14
    for path, result in expected.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["result"] == result
        assert payload["simulation_artifact_schema_version"] == 2
        assert payload["oracle"].endswith("/shards/seed_00003/c0.csv")
        assert payload["oracle_signature"]
    for path, (content, modified) in untouched.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == modified
    with (config.output_dir / "results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 16
    assert factory.sessions == 4


def test_experiment_dry_run_does_not_create_output(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    result = ExperimentRunner(config).run(dry_run=True)
    assert result["dry_run"] is True
    assert result["plan"]["estimated_operator_steps"] == 192
    assert not config.output_dir.exists()


def _interrupt_after_four_recipes(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    factory = _RecordingRunnerFactory(("c0", "r00004"))
    with pytest.raises(RuntimeError, match="collection shard"):
        ExperimentRunner(config, runner_factory=factory).run(phase="collect")
    progress = (
        config.output_dir
        / "checkpoints"
        / "collect"
        / "seed_00003"
        / "c0.progress.json"
    )
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["completed_recipes"] == 4
    return config, factory, progress


def _assert_complete_shard(config, circuit_id="c0"):
    shard = config.output_dir / "shards" / "seed_00003" / f"{circuit_id}.csv"
    with shard.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 * 5
    for recipe_index in range(12):
        recipe_id = f"r{recipe_index:05d}"
        recipe_rows = [row for row in rows if row["recipe_id"] == recipe_id]
        assert [int(row["step"]) for row in recipe_rows] == [0, 1, 2, 3, 4]
    assert all(None not in row for row in rows)


def test_recipe_level_resume_recovers_legacy_partial_and_torn_tail(tmp_path):
    config, factory, progress = _interrupt_after_four_recipes(tmp_path)
    partial = (
        config.output_dir / "shards" / "seed_00003" / "c0.csv.partial"
    )
    legacy = partial.with_name("c0.csv.tmp")
    partial.replace(legacy)
    progress.unlink()
    with legacy.open("a", encoding="utf-8", newline="") as handle:
        handle.write('"torn trailing record')

    factory.calls.clear()
    result = ExperimentRunner(
        config,
        resume=True,
        runner_factory=factory,
    ).run(phase="collect")

    c0_calls = [recipe_id for circuit, recipe_id in factory.calls if circuit == "c0"]
    assert c0_calls == [f"r{index:05d}" for index in range(4, 12)]
    assert result["collection"]["pending"] == 1
    checkpoint = json.loads(
        (
            config.output_dir
            / "checkpoints"
            / "collect"
            / "seed_00003"
            / "c0.done.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["resumed_trajectories"] == 4
    assert checkpoint["new_trajectories"] == 8
    assert not legacy.exists()
    assert not partial.exists()
    _assert_complete_shard(config)


@pytest.mark.parametrize("dependency", ["circuit", "recipes", "abc"])
def test_partial_resume_invalidates_changed_dependencies(tmp_path, dependency):
    config, factory, _ = _interrupt_after_four_recipes(tmp_path)
    if dependency == "circuit":
        config.circuits[0].write_bytes(b"changed-circuit")
    elif dependency == "abc":
        config.abc.write_bytes(b"changed-abc-binary")
    else:
        recipe_path = config.output_dir / "recipes" / "seed_00003.json"
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        current = payload["recipes"][0]["operations"][0]
        payload["recipes"][0]["operations"][0] = (
            "dc2" if current != "dc2" else "balance"
        )
        recipe_path.write_text(json.dumps(payload), encoding="utf-8")

    factory.calls.clear()
    ExperimentRunner(
        config,
        resume=True,
        runner_factory=factory,
    ).run(phase="collect")
    c0_calls = [recipe_id for circuit, recipe_id in factory.calls if circuit == "c0"]
    assert c0_calls == [f"r{index:05d}" for index in range(12)]
    checkpoint = json.loads(
        (
            config.output_dir
            / "checkpoints"
            / "collect"
            / "seed_00003"
            / "c0.done.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["resumed_trajectories"] == 0
    _assert_complete_shard(config)


def test_single_run_lock_rejects_second_runner_and_is_reusable(tmp_path):
    config = load_experiment_config(_write_config(tmp_path))
    factory = _FakeRunnerFactory()
    lock_path = config.output_dir / "run.lock"
    with _ExperimentLock(lock_path):
        with pytest.raises(RuntimeError, match="another experiment runner"):
            ExperimentRunner(
                config,
                runner_factory=factory,
            ).run(phase="collect")
        assert factory.sessions == 0
        assert not (config.output_dir / "manifest.json").exists()

    ExperimentRunner(config, runner_factory=factory).run(phase="collect")
    assert factory.sessions == 4
