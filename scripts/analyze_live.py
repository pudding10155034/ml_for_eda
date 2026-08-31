#!/usr/bin/env python3
"""Audit and summarize resumable live-ABC validation results.

Unlike ``analyze_results.py`` (which targets offline oracle replay), this
report keeps the repeat dimension and summarizes measured ABC wall time.  It
can therefore show both policy differences and the noise of repeating the
same live cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.analyze_results import summarize_values
except ModuleNotFoundError:  # Running this file directly from the scripts directory.
    from analyze_results import summarize_values


DEFAULT_METHOD = "risk_aware"
REQUIRED_FIELDS = {
    "recipe_seed",
    "holdout_circuit",
    "budget",
    "search_seed",
    "relative_gap_pct",
    "live_wall_s",
}


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
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _optional_float(value: str | None) -> float | None:
    if value in {None, "", "None", "null"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"live results CSV not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(
                f"live results missing required columns: {sorted(missing)}"
            )
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(reader, start=2):
            try:
                row: dict[str, Any] = {
                    "recipe_seed": int(raw["recipe_seed"]),
                    "holdout_circuit": str(raw["holdout_circuit"]),
                    "method": str(raw.get("method") or DEFAULT_METHOD),
                    "repeat": int(raw.get("repeat") or 0),
                    "budget": int(raw["budget"]),
                    "search_seed": int(raw["search_seed"]),
                    "relative_gap_pct": _optional_float(
                        raw.get("relative_gap_pct")
                    ),
                    "live_wall_s": float(raw["live_wall_s"]),
                    "selected": int(raw["selected"])
                    if raw.get("selected") not in {None, ""}
                    else None,
                    "completed_evaluations": int(raw["completed_evaluations"])
                    if raw.get("completed_evaluations") not in {None, ""}
                    else None,
                    "early_stops": int(raw["early_stops"])
                    if raw.get("early_stops") not in {None, ""}
                    else None,
                    "best_recipe_id": raw.get("best_recipe_id") or None,
                    "oracle_best_recipe_id": raw.get("oracle_best_recipe_id")
                    or None,
                    "termination_reason": raw.get("termination_reason") or None,
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid live result row {index}: {exc}") from exc
            if row["repeat"] < 0 or row["budget"] < 1:
                raise ValueError(f"invalid repeat/budget in live result row {index}")
            if not math.isfinite(row["live_wall_s"]) or row["live_wall_s"] < 0:
                raise ValueError(f"invalid live_wall_s in live result row {index}")
            rows.append(row)
    if not rows:
        raise ValueError(f"live results CSV is empty: {source}")
    return rows


def _settings_axes(settings_path: str | Path | None) -> dict[str, Any] | None:
    if settings_path is None:
        return None
    payload = _read_json(Path(settings_path).expanduser().resolve())
    methods = payload.get("methods", [DEFAULT_METHOD])
    if isinstance(methods, str):
        methods = [item.strip() for item in methods.split(",") if item.strip()]
    if not isinstance(methods, list) or not methods:
        raise ValueError("live settings methods must be a non-empty list")
    circuits = payload.get("circuits")
    budgets = payload.get("budgets")
    search_seeds = payload.get("search_seeds")
    if not all(isinstance(item, list) and item for item in (circuits, budgets, search_seeds)):
        raise ValueError(
            "live settings must include non-empty circuits, budgets, and search_seeds"
        )
    return {
        "methods": [str(item) for item in methods],
        "recipe_seeds": [int(payload.get("recipe_seed", 0))],
        "circuits": [str(item) for item in circuits],
        "budgets": [int(item) for item in budgets],
        "search_seeds": [int(item) for item in search_seeds],
        "repeats": int(payload.get("repeats", 1)),
    }


def validate_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    axes: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    observed = [
        (
            str(row["method"]),
            int(row["recipe_seed"]),
            str(row["holdout_circuit"]),
            int(row["repeat"]),
            int(row["budget"]),
            int(row["search_seed"]),
        )
        for row in rows
    ]
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for key in observed:
        counts[key] += 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    inferred = {
        "methods": sorted({key[0] for key in observed}),
        "recipe_seeds": sorted({key[1] for key in observed}),
        "circuits": sorted({key[2] for key in observed}),
        "repeats": max(key[3] for key in observed) + 1,
        "budgets": sorted({key[4] for key in observed}),
        "search_seeds": sorted({key[5] for key in observed}),
    }
    expected_axes = dict(inferred)
    if axes is not None:
        expected_axes.update(
            {
                "methods": list(axes["methods"]),
                "recipe_seeds": [int(item) for item in axes["recipe_seeds"]],
                "circuits": list(axes["circuits"]),
                "repeats": int(axes["repeats"]),
                "budgets": [int(item) for item in axes["budgets"]],
                "search_seeds": [int(item) for item in axes["search_seeds"]],
            }
        )
    if expected_axes["repeats"] < 1:
        raise ValueError("repeats must be positive")
    expected = {
        (method, recipe_seed, circuit, repeat, budget, search_seed)
        for method in expected_axes["methods"]
        for recipe_seed in expected_axes["recipe_seeds"]
        for circuit in expected_axes["circuits"]
        for repeat in range(expected_axes["repeats"])
        for budget in expected_axes["budgets"]
        for search_seed in expected_axes["search_seeds"]
    }
    actual = set(observed)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected) if axes is not None else []
    result = {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "missing_rows": missing,
        "unexpected_rows": unexpected,
        "duplicate_rows": len(duplicates),
        "complete": not missing and not unexpected and not duplicates,
        "axes": expected_axes,
    }
    if require_complete and not result["complete"]:
        raise ValueError(
            "incomplete live validation grid: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={len(duplicates)}"
        )
    return result


def _group(rows: Iterable[Mapping[str, Any]], *fields: str) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
    bootstrap_reps: int,
) -> dict[str, Any]:
    return summarize_values(
        (row.get(field) for row in rows), seed=seed, reps=bootstrap_reps
    )


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation: Mapping[str, Any],
    source: Path,
    output_dir: Path,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    by_method_budget: dict[str, Any] = {}
    for index, (key, group) in enumerate(
        sorted(_group(rows, "method", "budget").items(), key=str)
    ):
        method, budget = key
        by_method_budget[f"{method}|{budget}"] = {
            "dimensions": {"method": method, "budget": int(budget)},
            "runs": len(group),
            "metrics": {
                metric: _metric_summary(
                    group,
                    metric,
                    seed=seed + index * 101 + metric_index,
                    bootstrap_reps=bootstrap_reps,
                )
                for metric_index, metric in enumerate(
                    ("relative_gap_pct", "live_wall_s", "early_stops")
                )
            },
        }

    repeat_cells = _group(
        rows,
        "method",
        "recipe_seed",
        "holdout_circuit",
        "budget",
        "search_seed",
    )
    repeat_rows: list[dict[str, Any]] = []
    for key, group in sorted(repeat_cells.items(), key=str):
        timings = [float(row["live_wall_s"]) for row in group]
        if len(timings) < 2:
            continue
        mean = sum(timings) / len(timings)
        std = math.sqrt(
            sum((value - mean) ** 2 for value in timings) / (len(timings) - 1)
        )
        repeat_rows.append(
            {
                "method": key[0],
                "recipe_seed": key[1],
                "holdout_circuit": key[2],
                "budget": key[3],
                "search_seed": key[4],
                "repeats": len(timings),
                "mean_wall_s": mean,
                "std_wall_s": std,
                "cv_pct": 100.0 * std / mean if mean > 0 else None,
            }
        )
    noise_by_method: dict[str, Any] = {}
    for index, (method_key, group) in enumerate(
        sorted(_group(repeat_rows, "method").items(), key=str)
    ):
        method = method_key[0]
        noise_by_method[method] = {
            "cells": len(group),
            "metrics": {
                "cv_pct": summarize_values(
                    (row["cv_pct"] for row in group),
                    seed=seed + 1000 + index,
                    reps=bootstrap_reps,
                ),
                "std_wall_s": summarize_values(
                    (row["std_wall_s"] for row in group),
                    seed=seed + 1100 + index,
                    reps=bootstrap_reps,
                ),
            },
        }

    budgets = sorted({int(row["budget"]) for row in rows})
    winners: dict[str, Any] = {}
    for budget in budgets:
        groups = {
            method: group
            for (method, group_budget), group in _group(
                rows, "method", "budget"
            ).items()
            if int(group_budget) == budget
        }
        gap_means = {
            method: _metric_summary(
                group,
                "relative_gap_pct",
                seed=seed + 2000 + index,
                bootstrap_reps=bootstrap_reps,
            ).get("mean")
            for index, (method, group) in enumerate(sorted(groups.items()))
        }
        wall_means = {
            method: _metric_summary(
                group,
                "live_wall_s",
                seed=seed + 2100 + index,
                bootstrap_reps=bootstrap_reps,
            ).get("mean")
            for index, (method, group) in enumerate(sorted(groups.items()))
        }
        winners[str(budget)] = {
            "lowest_gap_method": min(
                (item for item in gap_means.items() if item[1] is not None),
                key=lambda item: (item[1], item[0]),
                default=(None, None),
            )[0],
            "fastest_method": min(
                (item for item in wall_means.items() if item[1] is not None),
                key=lambda item: (item[1], item[0]),
                default=(None, None),
            )[0],
            "gap_means": gap_means,
            "wall_means": wall_means,
        }

    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "source": str(source),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "scope": {
            "rows": len(rows),
            "methods": validation["axes"]["methods"],
            "budgets": validation["axes"]["budgets"],
            "repeats": validation["axes"]["repeats"],
        },
        "validation": dict(validation),
        "by_method_budget": by_method_budget,
        "repeat_noise_by_method": noise_by_method,
        "repeat_cells": repeat_rows,
        "winners_by_budget": winners,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Live ABC validation report",
        "",
        f"- Grid: {validation['observed_rows']}/{validation['expected_rows']} cells; complete={validation['complete']}.",
        f"- Methods: {', '.join(report['scope']['methods'])}.",
        f"- Repeats per cell: {report['scope']['repeats']}.",
        "",
        "## Policy comparison",
        "",
        "| Method | Budget | Mean gap (%) | Gap CI95 | Mean wall (s) | Wall CI95 |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for key, group in sorted(report["by_method_budget"].items()):
        gap = group["metrics"]["relative_gap_pct"]
        wall = group["metrics"]["live_wall_s"]
        gap_ci = gap.get("ci95")
        wall_ci = wall.get("ci95")
        gap_interval = "n/a" if not gap_ci else f"[{_fmt(gap_ci[0], 3)}, {_fmt(gap_ci[1], 3)}]"
        wall_interval = "n/a" if not wall_ci else f"[{_fmt(wall_ci[0], 3)}, {_fmt(wall_ci[1], 3)}]"
        method, budget = key.split("|", 1)
        lines.append(
            f"| {method} | {budget} | {_fmt(gap.get('mean'))} | {gap_interval} | {_fmt(wall.get('mean'))} | {wall_interval} |"
        )
    lines.extend(["", "## Repeat timing noise", ""])
    if not report["repeat_noise_by_method"]:
        lines.append("No repeated cells were available; run with `repeats >= 2` to estimate noise.")
    else:
        lines.extend(
            [
                "| Method | Cells | Mean CV (%) | Mean timing std (s) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method, item in sorted(report["repeat_noise_by_method"].items()):
            cv = item["metrics"]["cv_pct"].get("mean")
            std = item["metrics"]["std_wall_s"].get("mean")
            lines.append(f"| {method} | {item['cells']} | {_fmt(cv)} | {_fmt(std)} |")
    lines.extend(["", "## Descriptive winners", ""])
    for budget, item in sorted(report["winners_by_budget"].items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"- Budget {budget}: lowest mean gap = `{item['lowest_gap_method'] or 'n/a'}`, "
            f"fastest mean wall time = `{item['fastest_method'] or 'n/a'}`."
        )
    lines.extend(
        [
            "",
            "These are descriptive live measurements on the selected circuits and seeds; they are not a significance test or a guarantee under hardware/process changes.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    results_path: str | Path,
    output_dir: str | Path,
    settings_path: str | Path | None = None,
    bootstrap_reps: int = 1000,
    seed: int = 20260831,
    require_complete: bool = True,
) -> dict[str, Any]:
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be non-negative")
    source = Path(results_path).expanduser().resolve()
    rows = load_rows(source)
    validation = validate_grid(
        rows,
        axes=_settings_axes(settings_path),
        require_complete=require_complete,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(
        rows,
        validation=validation,
        source=source,
        output_dir=output,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    _atomic_json(report, output / "report.json")
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(
        results_path=args.results,
        output_dir=args.output_dir,
        settings_path=args.settings,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
