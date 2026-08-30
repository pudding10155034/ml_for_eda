#!/usr/bin/env python3
"""Run resumable baseline and ablation sweeps against saved oracle shards.

The runner reuses the exact models, recipes, circuits, and replay evaluator
from a completed experiment.  Only the search policy changes, so a policy
comparison is paired at the recipe-seed/circuit/budget/search-seed cell.  Each
cell is committed atomically and can be resumed safely after interruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from riskaware_eda.experiment import _ExperimentLock, load_experiment_config
from riskaware_eda.model import RiskModel
from riskaware_eda.simulation import SEARCH_POLICIES, OracleSimulator, load_oracle_trajectories


ABLATION_SCHEMA_VERSION = 1
DEFAULT_METHODS = tuple(SEARCH_POLICIES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed_tag(seed: int) -> str:
    return f"seed_{seed:05d}"


def _path_from(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _split(value: str | None, cast: Any = str) -> list[Any] | None:
    if value is None or not value.strip():
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _load_settings(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.settings is not None:
        settings_path = Path(args.settings).expanduser().resolve()
        settings = _read_json(settings_path)
        base_raw = settings.get("base_config")
        if not base_raw:
            raise ValueError("ablation settings require 'base_config'")
        base_config = _path_from(settings_path.parent, str(base_raw))
    else:
        settings_path = None
        settings = {}
        if args.config is None:
            raise ValueError("provide --settings or --config")
        base_config = Path(args.config).expanduser().resolve()
    config = load_experiment_config(base_config)
    source_root = _path_from(
        config.project_root,
        args.experiment_dir or settings.get("experiment_dir", config.output_dir),
    )
    output_root = _path_from(
        config.project_root,
        args.output_dir or settings.get("output_dir", f"artifacts/ablations/{config.name}"),
    )
    values: dict[str, Any] = {
        "settings_source": str(settings_path) if settings_path else None,
        "base_config": str(base_config),
        "experiment_dir": str(source_root),
        "output_dir": str(output_root),
        "methods": (
            [item.strip() for item in args.methods.split(",") if item.strip()]
            if args.methods
            else settings.get("methods", list(DEFAULT_METHODS))
        ),
        "recipe_seeds": _split(args.recipe_seeds, int) or settings.get("recipe_seeds", list(config.recipes.seeds)),
        "circuits": _split(args.circuits) or settings.get("circuits", [path.stem for path in config.circuits]),
        "budgets": _split(args.budgets, int) or settings.get("budgets", list(config.evaluation.budgets)),
        "search_seeds": _split(args.search_seeds, int) or settings.get("search_seeds", list(config.evaluation.search_seeds)),
        "min_stop_step": int(settings.get("min_stop_step", config.evaluation.min_stop_step)),
    }
    values["methods"] = [str(item) for item in values["methods"]]
    values["recipe_seeds"] = sorted({int(item) for item in values["recipe_seeds"]})
    values["circuits"] = [str(item) for item in values["circuits"]]
    values["budgets"] = sorted({int(item) for item in values["budgets"]})
    values["search_seeds"] = sorted({int(item) for item in values["search_seeds"]})
    if not values["methods"] or any(item not in SEARCH_POLICIES for item in values["methods"]):
        unknown = sorted(set(values["methods"]) - set(SEARCH_POLICIES))
        raise ValueError(f"unknown ablation method(s): {unknown}; choose from {sorted(SEARCH_POLICIES)}")
    if not values["recipe_seeds"] or not values["circuits"] or not values["budgets"] or not values["search_seeds"]:
        raise ValueError("recipe seeds, circuits, budgets, and search seeds cannot be empty")
    if min(values["budgets"]) < 1 or max(values["budgets"]) > config.recipes.count:
        raise ValueError("ablation budgets must be within the configured recipe count")
    if values["min_stop_step"] < 1:
        raise ValueError("min_stop_step must be positive")
    configured_circuits = {path.stem: path for path in config.circuits}
    missing_circuits = sorted(set(values["circuits"]) - set(configured_circuits))
    if missing_circuits:
        raise ValueError(f"circuits not found in base config: {missing_circuits}")
    values["circuit_paths"] = {item: str(configured_circuits[item]) for item in values["circuits"]}
    values["config_fingerprint"] = config.fingerprint
    values["settings_fingerprint"] = _sha256_json({key: value for key, value in values.items() if key not in {"circuit_paths"}})
    return config, values


def _artifact_path(root: Path, method: str, seed: int, circuit: str, budget: int, search_seed: int) -> Path:
    return root / method / "simulations" / _seed_tag(seed) / circuit / f"budget_{budget:04d}_search_{search_seed:05d}.json"


def _artifact_complete(path: Path, expected: Mapping[str, Any], source: Mapping[str, Mapping[str, int]]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        result = payload.get("result", {})
        search = result.get("search", {})
        return bool(
            payload.get("ablation_artifact_schema_version") == ABLATION_SCHEMA_VERSION
            and payload.get("settings_fingerprint") == expected["settings_fingerprint"]
            and payload.get("method") == expected["method"]
            and payload.get("recipe_seed") == expected["recipe_seed"]
            and payload.get("holdout_circuit") == expected["holdout_circuit"]
            and payload.get("budget") == expected["budget"]
            and payload.get("search_seed") == expected["search_seed"]
            and payload.get("model_signature") == source["model"]
            and payload.get("oracle_signature") == source["oracle"]
            and isinstance(search, Mapping)
            and "best_qor" in search
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _row(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    search = result["search"]
    random_baseline = result["random_baseline"]
    return {
        "method": payload["method"],
        "recipe_seed": payload["recipe_seed"],
        "holdout_circuit": payload["holdout_circuit"],
        "budget": payload["budget"],
        "search_seed": payload["search_seed"],
        "best_recipe_id": search.get("best_recipe_id"),
        "oracle_best_recipe_id": result.get("oracle_best_recipe_id"),
        "best_qor": search.get("best_qor"),
        "oracle_best_qor": result.get("oracle_best_qor"),
        "relative_gap_pct": result.get("relative_gap_pct"),
        "total_runtime_s": search.get("total_runtime_s"),
        "random_best_qor": random_baseline.get("best_qor"),
        "random_runtime_s": random_baseline.get("total_runtime_s"),
        "runtime_reduction_vs_random_pct": result.get("runtime_reduction_vs_random_pct"),
        "selected": search.get("selected"),
        "completed_evaluations": search.get("completed_evaluations"),
        "early_stops": search.get("early_stops"),
        "candidates_eliminated": search.get("candidates_eliminated"),
        "termination_reason": search.get("termination_reason"),
    }


def _write_aggregate(root: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    flattened: list[Path] = []
    for method in settings["methods"]:
        flattened.extend(sorted((root / method / "simulations").rglob("*.json")))
    rows: list[dict[str, Any]] = []
    for path in flattened:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("ablation_artifact_schema_version") == ABLATION_SCHEMA_VERSION:
            rows.append(_row(payload))
    rows.sort(key=lambda item: (item["method"], item["recipe_seed"], item["holdout_circuit"], item["budget"], item["search_seed"]))
    results_path = root / "results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["method", "recipe_seed", "holdout_circuit", "budget", "search_seed"]
    temporary = results_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_path)
    by_method_budget: dict[str, Any] = {}
    for method in settings["methods"]:
        for budget in settings["budgets"]:
            selected = [item for item in rows if item["method"] == method and int(item["budget"]) == budget]
            gaps = [float(item["relative_gap_pct"]) for item in selected if item.get("relative_gap_pct") is not None]
            reductions = [float(item["runtime_reduction_vs_random_pct"]) for item in selected]
            by_method_budget[f"{method}|{budget}"] = {
                "runs": len(selected),
                "mean_relative_gap_pct": sum(gaps) / len(gaps) if gaps else None,
                "mean_runtime_reduction_vs_random_pct": sum(reductions) / len(reductions) if reductions else None,
            }
    summary = {
        "ablation_schema_version": ABLATION_SCHEMA_VERSION,
        "settings_fingerprint": settings["settings_fingerprint"],
        "runs": len(rows),
        "expected_runs": len(settings["methods"]) * len(settings["recipe_seeds"]) * len(settings["circuits"]) * len(settings["budgets"]) * len(settings["search_seeds"]),
        "results": str(results_path),
        "by_method_budget": by_method_budget,
        "finished_at": _utc_now(),
    }
    _atomic_json(summary, root / "summary.json")
    return {"results": str(results_path), "summary": str(root / "summary.json"), "runs": len(rows)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, settings = _load_settings(args)
    source_root = Path(settings["experiment_dir"])
    output_root = Path(settings["output_dir"])
    if not source_root.is_dir():
        raise FileNotFoundError(f"source experiment directory not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("settings_fingerprint") != settings["settings_fingerprint"]:
            raise ValueError("ablation output belongs to a different settings fingerprint")
        if not args.resume and not args.dry_run:
            raise ValueError("ablation output already exists; pass --resume to continue")
    elif any(item.name != "run.lock" for item in output_root.iterdir()) and not args.dry_run:
        raise ValueError("non-empty ablation output has no compatible manifest")
    expected_runs = len(settings["methods"]) * len(settings["recipe_seeds"]) * len(settings["circuits"]) * len(settings["budgets"]) * len(settings["search_seeds"])
    plan = {
        "source_experiment": str(source_root),
        "output_dir": str(output_root),
        "methods": settings["methods"],
        "recipe_seeds": settings["recipe_seeds"],
        "circuits": settings["circuits"],
        "budgets": settings["budgets"],
        "search_seeds": settings["search_seeds"],
        "expected_runs": expected_runs,
        "settings_fingerprint": settings["settings_fingerprint"],
    }
    if args.dry_run:
        return {"dry_run": True, "plan": plan}
    manifest = {
        "ablation_schema_version": ABLATION_SCHEMA_VERSION,
        "status": "running",
        "runner_pid": os.getpid(),
        "updated_at": _utc_now(),
        "config_fingerprint": config.fingerprint,
        "settings_fingerprint": settings["settings_fingerprint"],
        "settings": {key: value for key, value in settings.items() if key != "circuit_paths"},
        "plan": plan,
        "software": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    if manifest_path.is_file():
        manifest["created_at"] = _read_json(manifest_path).get("created_at", _utc_now())
    else:
        manifest["created_at"] = _utc_now()
    _atomic_json(manifest, manifest_path)
    completed = skipped = 0
    started = time.perf_counter()
    try:
        with _ExperimentLock(output_root / "run.lock"):
            for seed in settings["recipe_seeds"]:
                recipe_path = source_root / "recipes" / f"{_seed_tag(seed)}.json"
                if not recipe_path.is_file():
                    raise FileNotFoundError(f"recipe artifact not found: {recipe_path}")
                for circuit_id in settings["circuits"]:
                    model_path = source_root / "models" / _seed_tag(seed) / f"holdout_{circuit_id}.joblib"
                    oracle_path = source_root / "shards" / _seed_tag(seed) / f"{circuit_id}.csv"
                    if not model_path.is_file() or not oracle_path.is_file():
                        raise FileNotFoundError(f"model/oracle artifacts missing for seed={seed}, circuit={circuit_id}")
                    model_signature = _file_signature(model_path)
                    oracle_signature = _file_signature(oracle_path)
                    oracle = load_oracle_trajectories(oracle_path, circuit_id)
                    model = RiskModel.load(model_path)
                    simulator = OracleSimulator(oracle, model)
                    source_signatures = {"model": model_signature, "oracle": oracle_signature}
                    for method in settings["methods"]:
                        for search_seed in settings["search_seeds"]:
                            pending = []
                            for budget in settings["budgets"]:
                                destination = _artifact_path(output_root, method, seed, circuit_id, budget, search_seed)
                                expected = {
                                    "settings_fingerprint": settings["settings_fingerprint"],
                                    "method": method,
                                    "recipe_seed": seed,
                                    "holdout_circuit": circuit_id,
                                    "budget": budget,
                                    "search_seed": search_seed,
                                }
                                if args.resume and _artifact_complete(destination, expected, source_signatures):
                                    skipped += 1
                                else:
                                    pending.append((budget, destination, expected))
                            if not pending:
                                continue
                            print(f"ablation: seed={seed} circuit={circuit_id} method={method} search_seed={search_seed} budgets={','.join(str(item[0]) for item in pending)}", flush=True)
                            results = simulator.simulate_budgets(
                                [item[0] for item in pending],
                                seed=search_seed,
                                min_steps_before_stopping=settings["min_stop_step"],
                                policy=method,
                            )
                            for budget, destination, expected in pending:
                                _atomic_json(
                                    {
                                        "ablation_artifact_schema_version": ABLATION_SCHEMA_VERSION,
                                        **expected,
                                        "model": str(model_path),
                                        "model_signature": model_signature,
                                        "oracle": str(oracle_path),
                                        "oracle_signature": oracle_signature,
                                        "result": results[budget].to_dict(),
                                        "finished_at": _utc_now(),
                                    },
                                    destination,
                                )
                                completed += 1
                    del simulator, model, oracle
        aggregate = _write_aggregate(output_root, settings)
        manifest["status"] = "complete"
        manifest["updated_at"] = _utc_now()
        manifest["timings"] = {"wall_s": time.perf_counter() - started}
        manifest["aggregate"] = aggregate
        _atomic_json(manifest, manifest_path)
        return {"completed": completed, "skipped": skipped, **aggregate}
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["updated_at"] = _utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(manifest, manifest_path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, help="ablation JSON settings file")
    parser.add_argument("--config", type=Path, help="base experiment config")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", help="comma-separated methods")
    parser.add_argument("--recipe-seeds", help="comma-separated recipe seeds")
    parser.add_argument("--circuits", help="comma-separated circuit IDs")
    parser.add_argument("--budgets", help="comma-separated budgets")
    parser.add_argument("--search-seeds", help="comma-separated search seeds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
