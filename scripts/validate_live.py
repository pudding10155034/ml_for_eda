#!/usr/bin/env python3
"""Validate the risk-aware search against live ABC for a small, resumable set.

The tool reuses a completed experiment's recipe and model artifacts, executes
only the requested live cells, and writes a side-by-side oracle comparison.
It is deliberately separate from the formal offline experiment so live
measurements cannot overwrite replay results.
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

from riskaware_eda.abc_runner import ABCRunner
from riskaware_eda.experiment import _ExperimentLock, load_experiment_config
from riskaware_eda.model import RiskModel
from riskaware_eda.qor import normalized_qor
from riskaware_eda.recipes import load_recipes
from riskaware_eda.search import run_live_search
from riskaware_eda.simulation import SEARCH_POLICIES, load_oracle_trajectories


LIVE_SCHEMA_VERSION = 2
_LEGACY_LIVE_SCHEMA_VERSION = 1
_DEFAULT_METHOD = "risk_aware"


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


def _seed_tag(seed: int) -> str:
    return f"seed_{seed:05d}"


def _path_from(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _split(value: str | None, cast: Any = str) -> list[Any] | None:
    if value is None or not value.strip():
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _coerce_methods(value: Any) -> list[str]:
    """Normalize a CLI/config method list and reject ambiguous values."""

    if isinstance(value, str):
        methods = _split(value) or []
    elif isinstance(value, Sequence):
        methods = [str(item).strip() for item in value]
    else:
        raise ValueError("methods must be a comma-separated string or list")
    methods = [item for item in methods if item]
    if not methods:
        raise ValueError("at least one live-validation method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("live-validation methods must be unique")
    unknown = sorted(set(methods) - set(SEARCH_POLICIES))
    if unknown:
        raise ValueError(
            f"unknown live-validation methods: {unknown}; "
            f"choose from {sorted(SEARCH_POLICIES)}"
        )
    return methods


def _settings_fingerprint(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _legacy_settings_fingerprint(values: Mapping[str, Any]) -> str:
    """Reproduce schema-1 fingerprints for safe default-run migration."""

    legacy = {
        key: value
        for key, value in values.items()
        if key not in {"methods", "repeats", "settings_fingerprint"}
    }
    return _settings_fingerprint(legacy)


def _settings_compatible(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    if previous.get("settings_fingerprint") == current.get("settings_fingerprint"):
        return True
    if previous.get("live_validation_schema_version") != _LEGACY_LIVE_SCHEMA_VERSION:
        return False
    # Schema 1 had exactly one implicit risk-aware run. Allow that output to
    # resume and be upgraded in place, but never reinterpret it as a multi-policy
    # or repeated experiment.
    if current.get("methods") != [_DEFAULT_METHOD] or current.get("repeats") != 1:
        return False
    previous_settings = previous.get("settings")
    if not isinstance(previous_settings, Mapping):
        return False
    previous_core = {
        key: value
        for key, value in previous_settings.items()
        if key not in {"methods", "repeats", "settings_fingerprint"}
    }
    current_core = {
        key: value
        for key, value in current.items()
        if key not in {"methods", "repeats", "settings_fingerprint"}
    }
    return (
        previous_core == current_core
        and previous.get("settings_fingerprint")
        == _settings_fingerprint(previous_core)
    )


def _load_settings(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.settings is not None:
        settings_path = Path(args.settings).expanduser().resolve()
        settings = _read_json(settings_path)
        base_raw = settings.get("base_config")
        if not base_raw:
            raise ValueError("live settings require 'base_config'")
        base_config = _path_from(settings_path.parent, str(base_raw))
    else:
        settings_path = None
        settings = {}
        if args.config is None:
            raise ValueError("provide --settings or --config")
        base_config = Path(args.config).expanduser().resolve()
    config = load_experiment_config(base_config)
    source_root = _path_from(config.project_root, args.experiment_dir or settings.get("experiment_dir", config.output_dir))
    output_root = _path_from(config.project_root, args.output_dir or settings.get("output_dir", f"artifacts/live_validation/{config.name}"))
    circuits = _split(args.circuits) or settings.get("circuits", [path.stem for path in config.circuits])
    budgets = _split(args.budgets, int) or settings.get("budgets", list(config.evaluation.budgets))
    search_seeds = _split(args.search_seeds, int) or settings.get("search_seeds", list(config.evaluation.search_seeds))
    methods = _coerce_methods(
        args.methods
        if args.methods is not None
        else settings.get("methods", [_DEFAULT_METHOD])
    )
    repeats = int(
        args.repeats if args.repeats is not None else settings.get("repeats", 1)
    )
    recipe_seed = int(args.recipe_seed if args.recipe_seed is not None else settings.get("recipe_seed", config.recipes.seeds[0]))
    values: dict[str, Any] = {
        "settings_source": str(settings_path) if settings_path else None,
        "base_config": str(base_config),
        "experiment_dir": str(source_root),
        "output_dir": str(output_root),
        "recipe_seed": recipe_seed,
        "circuits": [str(item) for item in circuits],
        "budgets": sorted({int(item) for item in budgets}),
        "search_seeds": sorted({int(item) for item in search_seeds}),
        "methods": methods,
        "repeats": repeats,
        "min_stop_step": int(settings.get("min_stop_step", config.evaluation.min_stop_step)),
        "timeout_s": float(args.timeout_s if args.timeout_s is not None else settings.get("timeout_s", config.collection.timeout_s)),
        "keep_workdir": bool(settings.get("keep_workdir", False)),
        "config_fingerprint": config.fingerprint,
    }
    if values["recipe_seed"] not in config.recipes.seeds:
        raise ValueError(f"recipe seed {values['recipe_seed']} is not in the base config")
    configured_circuits = {path.stem: path for path in config.circuits}
    missing = sorted(set(values["circuits"]) - set(configured_circuits))
    if missing:
        raise ValueError(f"circuits not found in base config: {missing}")
    if not values["budgets"] or min(values["budgets"]) < 1 or max(values["budgets"]) > config.recipes.count:
        raise ValueError("live budgets must be within the configured recipe count")
    if (
        not values["search_seeds"]
        or values["repeats"] < 1
        or values["min_stop_step"] < 1
        or values["timeout_s"] <= 0
    ):
        raise ValueError(
            "search seeds, repeats, min_stop_step, and timeout_s must be valid"
        )
    values["settings_fingerprint"] = _settings_fingerprint(values)
    return config, values


def _artifact_path(
    root: Path,
    seed: int,
    circuit: str,
    budget: int,
    search_seed: int,
    *,
    method: str = _DEFAULT_METHOD,
    repeat: int = 0,
    legacy_layout: bool = False,
) -> Path:
    destination = root / _seed_tag(seed) / circuit
    if not (legacy_layout and method == _DEFAULT_METHOD and repeat == 0):
        destination = destination / f"method_{method}" / f"repeat_{repeat:03d}"
    return destination / f"budget_{budget:04d}_search_{search_seed:05d}.json"


def _artifact_complete(
    path: Path,
    expected: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, int]],
    *,
    legacy_settings_fingerprint: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        search = payload.get("search")
        schema = payload.get("live_validation_schema_version")
        payload_method = payload.get("method", _DEFAULT_METHOD)
        payload_repeat = int(payload.get("repeat", 0))
        fingerprints = {expected["settings_fingerprint"]}
        if (
            legacy_settings_fingerprint
            and expected["method"] == _DEFAULT_METHOD
            and expected["repeat"] == 0
        ):
            fingerprints.add(legacy_settings_fingerprint)
        return bool(
            schema in {LIVE_SCHEMA_VERSION, _LEGACY_LIVE_SCHEMA_VERSION}
            and payload.get("settings_fingerprint") in fingerprints
            and payload.get("recipe_seed") == expected["recipe_seed"]
            and payload.get("holdout_circuit") == expected["holdout_circuit"]
            and payload.get("budget") == expected["budget"]
            and payload.get("search_seed") == expected["search_seed"]
            and payload_method == expected["method"]
            and payload_repeat == expected["repeat"]
            and payload.get("model_signature") == signatures["model"]
            and payload.get("recipe_signature") == signatures["recipe"]
            and payload.get("circuit_signature") == signatures["circuit"]
            and isinstance(search, Mapping)
            and "best_qor" in search
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _row(payload: Mapping[str, Any]) -> dict[str, Any]:
    search = payload["search"]
    return {
        "recipe_seed": payload["recipe_seed"],
        "holdout_circuit": payload["holdout_circuit"],
        "method": payload.get("method", _DEFAULT_METHOD),
        "repeat": int(payload.get("repeat", 0)),
        "budget": payload["budget"],
        "search_seed": payload["search_seed"],
        "best_qor": search.get("best_qor"),
        "oracle_best_qor": payload.get("oracle_best_qor"),
        "relative_gap_pct": payload.get("relative_gap_pct"),
        "live_wall_s": payload.get("live_wall_s"),
        "selected": search.get("selected"),
        "completed_evaluations": search.get("completed_evaluations"),
        "early_stops": search.get("early_stops"),
        "best_recipe_id": search.get("best_recipe_id"),
        "oracle_best_recipe_id": payload.get("oracle_best_recipe_id"),
        "termination_reason": search.get("termination_reason"),
    }


def _aggregate(root: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    paths = sorted(root.rglob("budget_*_search_*.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("live_validation_schema_version") in {
            LIVE_SCHEMA_VERSION,
            _LEGACY_LIVE_SCHEMA_VERSION,
        }:
            rows.append(_row(payload))
    rows.sort(
        key=lambda item: (
            item["recipe_seed"],
            item["holdout_circuit"],
            item["method"],
            item["repeat"],
            item["budget"],
            item["search_seed"],
        )
    )
    results_path = root / "results.csv"
    fields = list(rows[0]) if rows else ["recipe_seed", "holdout_circuit", "budget", "search_seed"]
    temporary = results_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_path)
    summary = {
        "live_validation_schema_version": LIVE_SCHEMA_VERSION,
        "settings_fingerprint": settings["settings_fingerprint"],
        "runs": len(rows),
        "expected_runs": (
            len(settings["circuits"])
            * len(settings["budgets"])
            * len(settings["search_seeds"])
            * len(settings["methods"])
            * int(settings["repeats"])
        ),
        "methods": list(settings["methods"]),
        "repeats": int(settings["repeats"]),
        "complete": len(rows)
        == (
            len(settings["circuits"])
            * len(settings["budgets"])
            * len(settings["search_seeds"])
            * len(settings["methods"])
            * int(settings["repeats"])
        ),
        "results": str(results_path),
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
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    legacy_settings_fingerprint = _legacy_settings_fingerprint(settings)
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if not _settings_compatible(previous, settings):
            raise ValueError("live-validation output belongs to a different settings fingerprint")
        if not args.resume and not args.dry_run:
            raise ValueError("live-validation output already exists; pass --resume to continue")
    elif output_root.exists() and any(
        item.name != "run.lock" for item in output_root.iterdir()
    ) and not args.dry_run:
        raise ValueError("non-empty live-validation output has no compatible manifest")
    plan = {
        "source_experiment": str(source_root),
        "output_dir": str(output_root),
        "recipe_seed": settings["recipe_seed"],
        "circuits": settings["circuits"],
        "budgets": settings["budgets"],
        "search_seeds": settings["search_seeds"],
        "methods": settings["methods"],
        "repeats": settings["repeats"],
        "expected_runs": (
            len(settings["circuits"])
            * len(settings["budgets"])
            * len(settings["search_seeds"])
            * len(settings["methods"])
            * int(settings["repeats"])
        ),
        "settings_fingerprint": settings["settings_fingerprint"],
    }
    if args.dry_run:
        return {"dry_run": True, "plan": plan}
    manifest = {
        "live_validation_schema_version": LIVE_SCHEMA_VERSION,
        "status": "running",
        "runner_pid": os.getpid(),
        "updated_at": _utc_now(),
        "settings_fingerprint": settings["settings_fingerprint"],
        "settings": settings,
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
        recipe_path = source_root / "recipes" / f"{_seed_tag(settings['recipe_seed'])}.json"
        recipes = load_recipes(recipe_path)
        runner = ABCRunner(config.abc, timeout_s=settings["timeout_s"], keep_workdir=settings["keep_workdir"])
        with _ExperimentLock(output_root / "run.lock"):
            for circuit_id in settings["circuits"]:
                circuit_path = next(path for path in config.circuits if path.stem == circuit_id)
                model_path = source_root / "models" / _seed_tag(settings["recipe_seed"]) / f"holdout_{circuit_id}.joblib"
                shard_path = source_root / "shards" / _seed_tag(settings["recipe_seed"]) / f"{circuit_id}.csv"
                if not model_path.is_file() or not shard_path.is_file():
                    raise FileNotFoundError(f"model/oracle artifacts missing for circuit={circuit_id}")
                signatures = {
                    "model": _file_signature(model_path),
                    "recipe": _file_signature(recipe_path),
                    "circuit": _file_signature(circuit_path),
                }
                oracle = load_oracle_trajectories(shard_path, circuit_id)
                oracle_scores = {
                    trajectory.recipe_id: normalized_qor(trajectory.current_stats, trajectory.initial)
                    for trajectory in oracle
                }
                oracle_best_recipe_id = min(oracle_scores, key=oracle_scores.get)
                oracle_best_qor = oracle_scores[oracle_best_recipe_id]
                model = RiskModel.load(model_path)
                legacy_layout = (
                    settings["methods"] == [_DEFAULT_METHOD]
                    and int(settings["repeats"]) == 1
                )
                for method in settings["methods"]:
                    policy = SEARCH_POLICIES[method]
                    for repeat in range(int(settings["repeats"])):
                        for search_seed in settings["search_seeds"]:
                            for budget in settings["budgets"]:
                                destination = _artifact_path(
                                    output_root,
                                    settings["recipe_seed"],
                                    circuit_id,
                                    budget,
                                    search_seed,
                                    method=method,
                                    repeat=repeat,
                                    legacy_layout=legacy_layout,
                                )
                                expected = {
                                    "settings_fingerprint": settings["settings_fingerprint"],
                                    "recipe_seed": settings["recipe_seed"],
                                    "holdout_circuit": circuit_id,
                                    "method": method,
                                    "repeat": repeat,
                                    "budget": budget,
                                    "search_seed": search_seed,
                                }
                                if args.resume and _artifact_complete(
                                    destination,
                                    expected,
                                    signatures,
                                    legacy_settings_fingerprint=legacy_settings_fingerprint,
                                ):
                                    skipped += 1
                                    continue
                                started_cell = time.perf_counter()
                                print(
                                    "live validation: "
                                    f"circuit={circuit_id} method={method} "
                                    f"repeat={repeat} budget={budget} "
                                    f"search_seed={search_seed}",
                                    flush=True,
                                )
                                with runner.session(circuit_path) as session:
                                    search = run_live_search(
                                        model=model,
                                        session=session,
                                        recipes=recipes,
                                        budget=budget,
                                        seed=search_seed,
                                        min_steps_before_stopping=settings["min_stop_step"],
                                        selection=str(policy["selection"]),
                                        safe_elimination=bool(policy["safe_elimination"]),
                                        early_stopping=bool(policy["early_stopping"]),
                                    )
                                relative_gap = (
                                    None
                                    if search.best_qor is None
                                    else 100.0
                                    * (search.best_qor - oracle_best_qor)
                                    / max(abs(oracle_best_qor), 1e-12)
                                )
                                _atomic_json(
                                    {
                                        "live_validation_schema_version": LIVE_SCHEMA_VERSION,
                                        **expected,
                                        "model": str(model_path),
                                        "model_signature": signatures["model"],
                                        "recipe": str(recipe_path),
                                        "recipe_signature": signatures["recipe"],
                                        "circuit": str(circuit_path),
                                        "circuit_signature": signatures["circuit"],
                                        "oracle": str(shard_path),
                                        "oracle_best_recipe_id": oracle_best_recipe_id,
                                        "oracle_best_qor": oracle_best_qor,
                                        "relative_gap_pct": relative_gap,
                                        "live_wall_s": time.perf_counter() - started_cell,
                                        "search": search.to_dict(),
                                        "finished_at": _utc_now(),
                                    },
                                    destination,
                                )
                                completed += 1
        aggregate = _aggregate(output_root, settings)
        manifest["status"] = "complete"
        manifest["updated_at"] = _utc_now()
        manifest["aggregate"] = aggregate
        manifest["timings"] = {"wall_s": time.perf_counter() - started}
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
    parser.add_argument("--settings", type=Path, help="live-validation JSON settings file")
    parser.add_argument("--config", type=Path, help="base experiment config")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recipe-seed", type=int)
    parser.add_argument("--circuits")
    parser.add_argument("--budgets")
    parser.add_argument("--search-seeds")
    parser.add_argument(
        "--methods",
        help="comma-separated search policies (default: risk_aware)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="repeat each method/circuit/budget/search-seed cell",
    )
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
