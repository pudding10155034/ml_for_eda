#!/usr/bin/env python3
"""Benchmark exact evaluation optimizations without modifying artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from riskaware_eda.experiment import ExperimentConfig, load_experiment_config
from riskaware_eda.model import RiskModel
from riskaware_eda.simulation import (
    OracleSimulator,
    load_oracle_trajectories,
    simulate_search_from_oracle,
)


def _key(circuit: str, budget: int, search_seed: int) -> str:
    return f"{circuit}:{budget}:{search_seed}"


def _load_holdout(
    root: Path,
    recipe_seed: int,
    circuit: str,
) -> tuple[RiskModel, list[Any]]:
    seed_tag = f"seed_{recipe_seed:05d}"
    model = RiskModel.load(
        root / "models" / seed_tag / f"holdout_{circuit}.joblib"
    )
    shard = root / "shards" / seed_tag / f"{circuit}.csv"
    oracle = load_oracle_trajectories(
        shard if shard.is_file() else root / "datasets" / f"{seed_tag}.csv",
        circuit,
    )
    return model, oracle


def _run_mode(
    mode: str,
    *,
    config: ExperimentConfig,
    root: Path,
    recipe_seed: int,
    circuits: tuple[str, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
    outputs: dict[str, dict[str, object]] = {}
    load_wall_s = prepare_wall_s = sweep_wall_s = 0.0
    cache: dict[str, dict[str, int]] = {}
    total_started = time.perf_counter()

    for circuit in circuits:
        started = time.perf_counter()
        model, oracle = _load_holdout(root, recipe_seed, circuit)
        load_wall_s += time.perf_counter() - started

        if mode == "independent":
            started = time.perf_counter()
            for budget in config.evaluation.budgets:
                for search_seed in config.evaluation.search_seeds:
                    result = simulate_search_from_oracle(
                        oracle,
                        model,
                        budget=budget,
                        seed=search_seed,
                        min_steps_before_stopping=(
                            config.evaluation.min_stop_step
                        ),
                    )
                    outputs[_key(circuit, budget, search_seed)] = result.to_dict()
            sweep_wall_s += time.perf_counter() - started
            del model, oracle
            continue

        started = time.perf_counter()
        simulator = OracleSimulator(oracle, model)
        prepare_wall_s += time.perf_counter() - started
        started = time.perf_counter()
        if mode == "cached":
            for budget in config.evaluation.budgets:
                for search_seed in config.evaluation.search_seeds:
                    result = simulator.simulate_budgets(
                        (budget,),
                        seed=search_seed,
                        min_steps_before_stopping=(
                            config.evaluation.min_stop_step
                        ),
                    )[budget]
                    outputs[_key(circuit, budget, search_seed)] = result.to_dict()
        elif mode == "fused":
            for search_seed in config.evaluation.search_seeds:
                results = simulator.simulate_budgets(
                    config.evaluation.budgets,
                    seed=search_seed,
                    min_steps_before_stopping=config.evaluation.min_stop_step,
                )
                for budget, result in results.items():
                    outputs[_key(circuit, budget, search_seed)] = result.to_dict()
        else:
            raise ValueError(f"unknown benchmark mode: {mode}")
        sweep_wall_s += time.perf_counter() - started
        cache[circuit] = simulator.cache_info()
        del simulator, model, oracle

    return outputs, {
        "load_wall_s": load_wall_s,
        "prepare_wall_s": prepare_wall_s,
        "sweep_wall_s": sweep_wall_s,
        "total_wall_s": time.perf_counter() - total_started,
        "cache": cache,
    }


def _digest(outputs: dict[str, dict[str, object]]) -> str:
    serialized = json.dumps(
        outputs,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _load_saved_results(
    root: Path,
    recipe_seed: int,
    circuits: tuple[str, ...],
    budgets: tuple[int, ...],
    search_seeds: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    saved: dict[str, dict[str, object]] = {}
    seed_tag = f"seed_{recipe_seed:05d}"
    for circuit in circuits:
        for budget in budgets:
            for search_seed in search_seeds:
                path = (
                    root
                    / "simulations"
                    / seed_tag
                    / circuit
                    / f"budget_{budget:04d}_search_{search_seed:05d}.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                saved[_key(circuit, budget, search_seed)] = payload["result"]
    return saved


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--experiment-dir",
        help="artifact directory override; defaults to config output_dir",
    )
    parser.add_argument("--recipe-seed", type=int)
    parser.add_argument("--circuit", action="append", dest="circuits")
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    root = (
        Path(args.experiment_dir).expanduser().resolve()
        if args.experiment_dir
        else config.output_dir
    )
    recipe_seed = (
        args.recipe_seed
        if args.recipe_seed is not None
        else config.recipes.seeds[0]
    )
    configured_circuits = tuple(path.stem for path in config.circuits)
    circuits = tuple(args.circuits or configured_circuits)
    unknown = set(circuits) - set(configured_circuits)
    if unknown:
        parser.error(f"unknown circuit IDs: {sorted(unknown)}")

    modes = ("independent", "cached", "fused")
    samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    reference: dict[str, dict[str, object]] | None = None
    for repeat in range(args.repeats):
        order = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
        for mode in order:
            outputs, timing = _run_mode(
                mode,
                config=config,
                root=root,
                recipe_seed=recipe_seed,
                circuits=circuits,
            )
            if reference is None:
                reference = outputs
            elif outputs != reference:
                raise RuntimeError(
                    f"{mode} output differs from the exact reference result"
                )
            timing["digest"] = _digest(outputs)
            samples[mode].append(timing)

    assert reference is not None
    artifact_match: bool | None = None
    if args.verify_artifacts:
        artifact_match = _load_saved_results(
            root,
            recipe_seed,
            circuits,
            config.evaluation.budgets,
            config.evaluation.search_seeds,
        ) == reference
        if not artifact_match:
            raise RuntimeError("benchmark output differs from saved artifacts")

    medians = {
        mode: statistics.median(item["total_wall_s"] for item in values)
        for mode, values in samples.items()
    }
    payload = {
        "config": str(config.source),
        "experiment_dir": str(root),
        "recipe_seed": recipe_seed,
        "circuits": list(circuits),
        "tasks": len(reference),
        "repeats": args.repeats,
        "exact_digest": _digest(reference),
        "artifact_match": artifact_match,
        "median_total_wall_s": medians,
        "speedup_vs_independent": {
            mode: medians["independent"] / medians[mode]
            for mode in ("cached", "fused")
        },
        "samples": samples,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
