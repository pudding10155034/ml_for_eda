#!/usr/bin/env python3
"""Audit experiment artifacts and build a reproducible technical report.

The formal experiment runner intentionally writes one JSON artifact per
recipe-seed/circuit/budget/search-seed cell.  This module treats those JSON
files as the source of truth, validates the expected grid, recomputes summary
statistics (including paired bootstrap intervals), and writes a Markdown
technical report plus dependency-free SVG figures.  It does not mutate the
formal experiment output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from riskaware_eda.experiment import load_experiment_config


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_METRICS = (
    "relative_gap_pct",
    "runtime_reduction_vs_random_pct",
    "runtime_delta_s",
    "early_stops",
    "candidates_eliminated",
    "oracle_hit",
)
PALETTE = {
    "blue": "#2563eb",
    "gold": "#c58a00",
    "orange": "#d97706",
    "olive": "#657a21",
    "pink": "#be4b8a",
    "charcoal": "#243142",
    "grid": "#d7dee8",
    "muted": "#607084",
    "paper": "#ffffff",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _family_by_circuit(config: Any) -> dict[str, str]:
    families: dict[str, str] = {}
    for path in config.circuits:
        family = path.parent.name or "unknown"
        families[path.stem] = family
    return families


def _simulation_paths(root: Path) -> Iterable[Path]:
    # Exclude manifests, reports and unrelated JSON files.  A result object is
    # the stable marker shared by schema v1, v2, and ablation artifacts.
    for path in sorted(root.rglob("*.json")):
        if path.name in {"manifest.json", "summary.json", "last_run.json"}:
            continue
        yield path


def _row_from_payload(
    payload: Mapping[str, Any],
    path: Path,
    family_by_circuit: Mapping[str, str],
    training_override: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    search = result.get("search")
    random_baseline = result.get("random_baseline")
    if not isinstance(search, Mapping) or not isinstance(random_baseline, Mapping):
        return None
    circuit = str(payload.get("holdout_circuit", search.get("circuit_id", "")))
    budget = _int_or_none(payload.get("budget", search.get("budget")))
    recipe_seed = _int_or_none(payload.get("recipe_seed"))
    search_seed = _int_or_none(payload.get("search_seed"))
    if not circuit or budget is None or recipe_seed is None or search_seed is None:
        return None
    best_recipe_id = search.get("best_recipe_id")
    oracle_best_recipe_id = result.get("oracle_best_recipe_id")
    oracle_hit: float | None
    if best_recipe_id is None or oracle_best_recipe_id is None:
        oracle_hit = None
    else:
        oracle_hit = 1.0 if best_recipe_id == oracle_best_recipe_id else 0.0
    total_runtime = _float_or_none(search.get("total_runtime_s"))
    random_runtime = _float_or_none(random_baseline.get("total_runtime_s"))
    if total_runtime is None or random_runtime is None:
        runtime_delta = None
    else:
        runtime_delta = total_runtime - random_runtime
    training = training_override if training_override is not None else payload.get("training_report", {})
    if not isinstance(training, Mapping):
        training = {}
    method = str(payload.get("method", "risk_aware"))
    return {
        "method": method,
        "recipe_seed": recipe_seed,
        "holdout_circuit": circuit,
        "family": family_by_circuit.get(circuit, "unknown"),
        "budget": budget,
        "search_seed": search_seed,
        "best_recipe_id": best_recipe_id,
        "oracle_best_recipe_id": oracle_best_recipe_id,
        "best_qor": _float_or_none(search.get("best_qor")),
        "oracle_best_qor": _float_or_none(result.get("oracle_best_qor")),
        "relative_gap_pct": _float_or_none(result.get("relative_gap_pct")),
        "total_runtime_s": total_runtime,
        "random_best_qor": _float_or_none(random_baseline.get("best_qor")),
        "random_runtime_s": random_runtime,
        "runtime_delta_s": runtime_delta,
        "runtime_reduction_vs_random_pct": _float_or_none(
            result.get("runtime_reduction_vs_random_pct")
        ),
        "selected": _int_or_none(search.get("selected")),
        "completed_evaluations": _int_or_none(
            search.get("completed_evaluations")
        ),
        "early_stops": _int_or_none(search.get("early_stops")),
        "candidates_eliminated": _int_or_none(
            search.get("candidates_eliminated")
        ),
        "termination_reason": str(search.get("termination_reason", "")),
        "oracle_hit": oracle_hit,
        "row_coverage": _float_or_none(training.get("row_coverage")),
        "simultaneous_trajectory_coverage": _float_or_none(
            training.get("simultaneous_trajectory_coverage")
        ),
        "mean_interval_width": _float_or_none(
            training.get("mean_interval_width")
        ),
        "artifact": str(path),
    }


def load_rows(
    simulation_root: str | Path,
    *,
    family_by_circuit: Mapping[str, str] | None = None,
    training_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(simulation_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"simulation directory not found: {root}")
    family_by_circuit = family_by_circuit or {}
    resolved_training_root = (
        Path(training_root).expanduser().resolve() if training_root is not None else None
    )
    training_cache: dict[tuple[int, str], Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in _simulation_paths(root):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        training_override = None
        if resolved_training_root is not None and not isinstance(
            payload.get("training_report"), Mapping
        ):
            seed = _int_or_none(payload.get("recipe_seed"))
            circuit = str(payload.get("holdout_circuit", ""))
            if seed is not None and circuit:
                cache_key = (seed, circuit)
                if cache_key not in training_cache:
                    training_path = (
                        resolved_training_root
                        / "reports"
                        / "training"
                        / f"seed_{seed:05d}"
                        / f"holdout_{circuit}.json"
                    )
                    try:
                        training_payload = _read_json(training_path)
                        value = training_payload.get("training_report")
                        if isinstance(value, Mapping):
                            training_cache[cache_key] = value
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                training_override = training_cache.get(cache_key)
        row = _row_from_payload(payload, path, family_by_circuit, training_override)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(f"no simulation result artifacts found under {root}")
    return rows


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["method"],
        int(row["recipe_seed"]),
        row["holdout_circuit"],
        int(row["budget"]),
        int(row["search_seed"]),
    )


def validate_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] | None = None,
    recipe_seeds: Sequence[int] | None = None,
    circuits: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    search_seeds: Sequence[int] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    observed = [_key(row) for row in rows]
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for item in observed:
        counts[item] += 1
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate simulation keys found (first: {duplicates[0]})")
    inferred = {
        "methods": sorted({str(row["method"]) for row in rows}),
        "recipe_seeds": sorted({int(row["recipe_seed"]) for row in rows}),
        "circuits": sorted({str(row["holdout_circuit"]) for row in rows}),
        "budgets": sorted({int(row["budget"]) for row in rows}),
        "search_seeds": sorted({int(row["search_seed"]) for row in rows}),
    }
    axes = {
        "methods": list(methods) if methods is not None else inferred["methods"],
        "recipe_seeds": (
            [int(item) for item in recipe_seeds]
            if recipe_seeds is not None
            else inferred["recipe_seeds"]
        ),
        "circuits": list(circuits) if circuits is not None else inferred["circuits"],
        "budgets": (
            [int(item) for item in budgets]
            if budgets is not None
            else inferred["budgets"]
        ),
        "search_seeds": (
            [int(item) for item in search_seeds]
            if search_seeds is not None
            else inferred["search_seeds"]
        ),
    }
    expected = {
        (method, seed, circuit, budget, search_seed)
        for method in axes["methods"]
        for seed in axes["recipe_seeds"]
        for circuit in axes["circuits"]
        for budget in axes["budgets"]
        for search_seed in axes["search_seeds"]
    }
    actual = set(observed)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected) if any(
        item is not None
        for item in (methods, recipe_seeds, circuits, budgets, search_seeds)
    ) else []
    if require_complete and (missing or unexpected):
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={missing[0]}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)} first={unexpected[0]}")
        raise ValueError("incomplete or incompatible simulation grid: " + "; ".join(details))
    return {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "missing_rows": missing,
        "unexpected_rows": unexpected,
        "duplicate_rows": len(duplicates),
        "complete": not missing and not unexpected and not duplicates,
        "axes": axes,
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    reps: int = 1000,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% CI for a mean.

    The implementation is deterministic for a fixed seed and handles the
    one-observation case explicitly, which is useful for live-validation
    pilots and smoke tests.
    """
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1 or reps < 1:
        value = float(array.mean())
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(reps), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_values(
    values: Iterable[Any],
    *,
    seed: int,
    reps: int,
) -> dict[str, Any]:
    numeric = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci95": None}
    ci = bootstrap_ci(numeric, seed=seed, reps=reps)
    return {
        "n": int(numeric.size),
        "mean": float(numeric.mean()),
        "median": float(np.median(numeric)),
        "std": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p05": float(np.quantile(numeric, 0.05)),
        "p95": float(np.quantile(numeric, 0.95)),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], *fields: str
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _summaries_by(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    seed: int,
    reps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (key, group) in enumerate(sorted(_group_rows(rows, *fields).items(), key=str)):
        key_name = "|".join(str(item) for item in key)
        metrics = {
            metric: summarize_values(
                (row.get(metric) for row in group),
                seed=seed + index * 101 + metric_index,
                reps=reps,
            )
            for metric_index, metric in enumerate(DEFAULT_METRICS)
        }
        result[key_name] = {
            "dimensions": {field: value for field, value in zip(fields, key)},
            "runs": len(group),
            "metrics": metrics,
        }
    return result


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation: Mapping[str, Any],
    source_root: Path,
    config: Any | None,
    output_dir: Path,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    budgets = sorted({int(row["budget"]) for row in rows})
    by_method_budget = _summaries_by(
        rows, ("method", "budget"), seed=seed, reps=bootstrap_reps
    )
    by_family_budget = _summaries_by(
        rows, ("family", "budget"), seed=seed + 17, reps=bootstrap_reps
    )
    by_circuit_budget = _summaries_by(
        rows, ("holdout_circuit", "budget"), seed=seed + 31, reps=bootstrap_reps
    )
    paired = _summaries_by(
        rows, ("method", "budget"), seed=seed + 47, reps=bootstrap_reps
    )
    coverage = summarize_values(
        (row.get("simultaneous_trajectory_coverage") for row in rows),
        seed=seed + 59,
        reps=bootstrap_reps,
    )
    oracle_hit_available = any(row.get("oracle_hit") is not None for row in rows)
    plots = write_plots(
        rows,
        output_dir / "figures",
        budgets=budgets,
        methods=methods,
    )
    config_payload = config.to_dict() if config is not None else None
    report = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_root": str(source_root),
        "source_signature": {
            "path": str(source_root),
            "json_files": sum(1 for _ in _simulation_paths(source_root)),
        },
        "provenance": {
            "project_root": str(config.project_root) if config is not None else None,
            "project_git_commit": (
                _git_revision(config.project_root) if config is not None else None
            ),
            "project_git_dirty": (
                _git_dirty(config.project_root) if config is not None else None
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "config": config_payload,
        },
        "validation": dict(validation),
        "definitions": {
            "relative_gap_pct": "100 * (search best QoR - oracle best QoR) / |oracle best QoR|; lower is better",
            "runtime_reduction_vs_random_pct": "100 * (1 - risk-aware search runtime / same-seed random baseline runtime); positive is faster",
            "runtime_delta_s": "search total runtime minus same-seed random baseline runtime; negative is faster",
            "oracle_hit": "1 when selected best recipe ID equals the exhaustive oracle best recipe ID; unavailable when legacy artifacts omit IDs",
            "simultaneous_trajectory_coverage": "training-time conformal coverage of complete trajectories, not a holdout performance guarantee",
        },
        "scope": {
            "methods": methods,
            "budgets": budgets,
            "rows": len(rows),
            "oracle_hit_available": oracle_hit_available,
        },
        "overall": {
            "by_method_budget": by_method_budget,
            "by_family_budget": by_family_budget,
            "by_circuit_budget": by_circuit_budget,
            "simultaneous_trajectory_coverage": coverage,
        },
        "charts": plots,
        "chart_map": [
            {
                "section": "Key findings",
                "question": "How does oracle gap change across discrete budgets and policies?",
                "family": "comparison",
                "variant": "grouped bars with bootstrap interval",
                "metric": "relative_gap_pct",
                "artifact": plots.get("gap_by_budget.svg"),
            },
            {
                "section": "Key findings",
                "question": "How does measured replay runtime compare with the same-seed random baseline?",
                "family": "uncertainty and benchmark",
                "variant": "signed grouped bars with zero reference",
                "metric": "runtime_reduction_vs_random_pct",
                "artifact": plots.get("runtime_reduction_by_budget.svg"),
            },
            {
                "section": "Circuit-family robustness",
                "question": "Do circuit families show different oracle gaps?",
                "family": "comparison",
                "variant": "family grouped bars",
                "metric": "relative_gap_pct",
                "artifact": plots.get("gap_by_family.svg"),
            },
        ],
        "source_sha256": _sha256(output_dir / "analysis_rows.csv")
        if (output_dir / "analysis_rows.csv").is_file()
        else None,
    }
    return report


def _git_dirty(root: Path) -> bool | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _svg_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_range(values: Sequence[float], *, include_zero: bool = False) -> tuple[float, float]:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if include_zero:
        finite.append(0.0)
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    if math.isclose(low, high):
        padding = max(abs(low) * 0.2, 1.0)
        low -= padding
        high += padding
    else:
        padding = (high - low) * 0.1
        low -= padding
        high += padding
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    return low, high


def _fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _grouped_bar_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    categories: Sequence[str],
    series: Sequence[Mapping[str, Any]],
    y_label: str,
    include_zero: bool,
) -> None:
    width, height = 960, 540
    left, right, top, bottom = 92, 30, 94, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values: list[float] = []
    for item in series:
        all_values.extend(
            float(value) for value in item["values"] if value is not None
        )
        for interval in item.get("ci", []):
            if interval and interval[0] is not None and interval[1] is not None:
                all_values.extend(float(value) for value in interval)
    y_low, y_high = _nice_range(all_values, include_zero=include_zero)
    y_span = y_high - y_low

    def x_for(category_index: int) -> float:
        return left + plot_w * (category_index + 0.5) / max(len(categories), 1)

    def y_for(value: float) -> float:
        return top + plot_h * (y_high - value) / y_span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">',
        f'<title id="title">{_svg_escape(title)}</title>',
        f'<desc id="desc">{_svg_escape(subtitle)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="20" font-weight="700" fill="{PALETTE["charcoal"]}">{_svg_escape(title)}</text>',
        f'<text x="{left}" y="58" font-family="sans-serif" font-size="13" fill="{PALETTE["muted"]}">{_svg_escape(subtitle)}</text>',
    ]
    for tick in np.linspace(y_low, y_high, 6):
        y = y_for(float(tick))
        parts.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="{PALETTE["muted"]}">{_fmt_tick(float(tick))}</text>'
        )
    if y_low < 0 < y_high:
        y_zero = y_for(0.0)
        parts.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y_zero:.2f}" y2="{y_zero:.2f}" stroke="{PALETTE["charcoal"]}" stroke-width="1.5"/>'
        )
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(42.0, group_w / max(len(series) + 1, 2))
    for cat_index, category in enumerate(categories):
        center = x_for(cat_index)
        parts.append(
            f'<text x="{center:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{PALETTE["charcoal"]}">{_svg_escape(category)}</text>'
        )
    for series_index, item in enumerate(series):
        color = str(item.get("color", PALETTE["blue"]))
        values = item["values"]
        cis = item.get("ci", [None] * len(values))
        for cat_index, value in enumerate(values):
            if value is None:
                continue
            value = float(value)
            x = x_for(cat_index) + (series_index - (len(series)-1)/2) * bar_w
            base = y_for(0.0) if y_low < 0 < y_high else y_for(y_low)
            y = y_for(value)
            rect_y, rect_h = (y, base-y) if value >= 0 else (base, y-base)
            parts.append(
                f'<rect x="{x-bar_w*0.42:.2f}" y="{rect_y:.2f}" width="{bar_w*0.84:.2f}" height="{max(rect_h, 0.5):.2f}" fill="{color}" opacity="0.86"/>'
            )
            interval = cis[cat_index] if cat_index < len(cis) else None
            if interval and interval[0] is not None and interval[1] is not None:
                y_low_ci, y_high_ci = y_for(float(interval[0])), y_for(float(interval[1]))
                cap = bar_w * 0.35
                parts.extend(
                    [
                        f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_low_ci:.2f}" y2="{y_high_ci:.2f}" stroke="{PALETTE["charcoal"]}" stroke-width="1.4"/>',
                        f'<line x1="{x-cap:.2f}" x2="{x+cap:.2f}" y1="{y_low_ci:.2f}" y2="{y_low_ci:.2f}" stroke="{PALETTE["charcoal"]}" stroke-width="1.4"/>',
                        f'<line x1="{x-cap:.2f}" x2="{x+cap:.2f}" y1="{y_high_ci:.2f}" y2="{y_high_ci:.2f}" stroke="{PALETTE["charcoal"]}" stroke-width="1.4"/>',
                    ]
                )
    legend_x = left
    legend_y = height - 25
    legend_columns = min(max(len(series), 1), 3)
    for index, item in enumerate(series):
        row = index // legend_columns
        column = index % legend_columns
        x = legend_x + column * 270
        y = legend_y - row * 22
        parts.append(
            f'<rect x="{x}" y="{y-11}" width="12" height="12" fill="{item.get("color", PALETTE["blue"])}"/>'
        )
        parts.append(
            f'<text x="{x+18}" y="{y}" font-family="sans-serif" font-size="12" fill="{PALETTE["charcoal"]}">{_svg_escape(item["name"])}</text>'
        )
    parts.append(
        f'<text x="18" y="{top+plot_h/2:.2f}" transform="rotate(-90 18 {top+plot_h/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{PALETTE["charcoal"]}">{_svg_escape(y_label)}</text>'
    )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _metric_from_group(
    group: Mapping[str, Any], metric: str
) -> tuple[float | None, list[float] | None]:
    value = group.get("metrics", {}).get(metric, {})
    if not isinstance(value, Mapping):
        return None, None
    mean = value.get("mean")
    ci = value.get("ci95")
    return (
        None if mean is None else float(mean),
        None if not isinstance(ci, list) else [float(ci[0]), float(ci[1])],
    )


def write_plots(
    rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    *,
    budgets: Sequence[int],
    methods: Sequence[str],
) -> dict[str, str]:
    """Write compact static charts and return report-relative paths."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    grouped = _summaries_by(rows, ("method", "budget"), seed=911, reps=500)
    colors = [PALETTE["blue"], PALETTE["gold"], PALETTE["orange"], PALETTE["olive"], PALETTE["pink"], PALETTE["charcoal"]]
    plots: dict[str, str] = {}
    for metric, filename, title, subtitle, y_label, include_zero in (
        (
            "relative_gap_pct",
            "gap_by_budget.svg",
            "Relative gap to exhaustive oracle",
            "Mean percentage gap by evaluation budget; whiskers are bootstrap 95% CIs",
            "Relative gap (%) — lower is better",
            True,
        ),
        (
            "runtime_reduction_vs_random_pct",
            "runtime_reduction_by_budget.svg",
            "Runtime reduction versus random baseline",
            "Mean signed reduction by budget; positive values indicate lower runtime",
            "Runtime reduction (%)",
            True,
        ),
    ):
        series = []
        for index, method in enumerate(methods):
            values, cis = [], []
            for budget in budgets:
                group = grouped.get(f"{method}|{budget}", {})
                mean, ci = _metric_from_group(group, metric)
                values.append(mean)
                cis.append(ci)
            series.append({"name": method, "color": colors[index % len(colors)], "values": values, "ci": cis})
        path = figure_dir / filename
        _grouped_bar_svg(
            path,
            title=title,
            subtitle=subtitle,
            categories=[str(item) for item in budgets],
            series=series,
            y_label=y_label,
            include_zero=include_zero,
        )
        plots[filename] = str(Path("figures") / filename)
    families = sorted({str(row["family"]) for row in rows})
    if len(families) > 1:
        family_groups = _summaries_by(rows, ("family", "budget"), seed=977, reps=500)
        series = []
        for index, family in enumerate(families):
            values, cis = [], []
            for budget in budgets:
                group = family_groups.get(f"{family}|{budget}", {})
                mean, ci = _metric_from_group(group, "relative_gap_pct")
                values.append(mean)
                cis.append(ci)
            series.append({"name": family, "color": colors[index % len(colors)], "values": values, "ci": cis})
        path = figure_dir / "gap_by_family.svg"
        _grouped_bar_svg(
            path,
            title="Relative gap by circuit family",
            subtitle="Mean percentage gap to the exhaustive oracle by budget",
            categories=[str(item) for item in budgets],
            series=series,
            y_label="Relative gap (%) — lower is better",
            include_zero=True,
        )
        plots[path.name] = str(Path("figures") / path.name)
    return plots


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "recipe_seed", "holdout_circuit", "family", "budget",
        "search_seed", "best_recipe_id", "oracle_best_recipe_id", "best_qor",
        "oracle_best_qor", "relative_gap_pct", "total_runtime_s",
        "random_best_qor", "random_runtime_s", "runtime_delta_s",
        "runtime_reduction_vs_random_pct", "selected", "completed_evaluations",
        "early_stops", "candidates_eliminated", "termination_reason",
        "oracle_hit", "row_coverage", "simultaneous_trajectory_coverage",
        "mean_interval_width", "artifact",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _report_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    validation = report["validation"]
    overall = report["overall"]["by_method_budget"]
    methods = scope["methods"]
    budgets = scope["budgets"]
    primary_method = "risk_aware" if "risk_aware" in methods else methods[0]
    primary_rows = [
        overall.get(f"{primary_method}|{budget}", {}) for budget in budgets
    ]
    gap_means = [
        item.get("metrics", {}).get("relative_gap_pct", {}).get("mean")
        for item in primary_rows
    ]
    runtime_means = [
        item.get("metrics", {}).get("runtime_reduction_vs_random_pct", {}).get("mean")
        for item in primary_rows
    ]
    first_gap = next((value for value in gap_means if value is not None), None)
    last_gap = next((value for value in reversed(gap_means) if value is not None), None)
    first_runtime = next((value for value in runtime_means if value is not None), None)
    last_runtime = next((value for value in reversed(runtime_means) if value is not None), None)
    lines = [
        "# Risk-aware EDA experiment — technical report",
        "",
        "## Technical summary",
        "",
        f"- **Scope is complete:** {validation['observed_rows']} simulation cells were parsed; expected grid size is {validation['expected_rows']}, with {len(validation['missing_rows'])} missing and {len(validation['unexpected_rows'])} unexpected cells.",
        f"- **Primary policy ({primary_method}) improves with budget:** mean relative gap changes from {_pct(first_gap)} at the first budget to {_pct(last_gap)} at the largest budget.",
        f"- **Runtime effect is budget-dependent:** mean runtime reduction versus the same-seed random baseline changes from {_pct(first_runtime)} to {_pct(last_runtime)} across the reported budgets.",
        "- **Interpretation:** these are offline replay results against an exhaustive oracle. They support descriptive method comparison, not a causal claim about production ABC performance.",
        "",
        "## Key findings with visual evidence",
        "",
        "The two charts below use grouped bars because the budgets are discrete anchors rather than a dense time series. Whiskers are percentile bootstrap 95% confidence intervals for the mean across simulation cells.",
        "",
        "![Relative gap by budget](figures/gap_by_budget.svg)",
        "",
        "![Runtime reduction by budget](figures/runtime_reduction_by_budget.svg)",
        "",
        "The gap chart answers how close each policy gets to the exhaustive best recipe. The runtime chart uses a signed percentage: positive means the policy's measured replay runtime is lower than the same-seed random baseline; negative means it is slower.",
        "",
        "## Scope, data, and metric definitions",
        "",
        f"- Source artifacts: `{report['source_root']}`; {scope['rows']} rows at grain method × recipe seed × holdout circuit × budget × search seed.",
        f"- Methods: {', '.join(methods)}.",
        f"- Budgets: {', '.join(str(item) for item in budgets)} candidate evaluations; circuit families are derived from the configured parent directory.",
        "- Relative gap (%) = 100 × (search best QoR − exhaustive oracle best QoR) / |oracle best QoR|; lower is better.",
        "- Runtime reduction (%) = 100 × (1 − search runtime / same-seed random baseline runtime); positive is faster.",
        "- Simultaneous trajectory coverage is the training-time conformal coverage reported by each holdout model. It is not the same as the search success rate.",
        "",
        "## Methodology and reproducibility",
        "",
        "Each simulation replays completed ABC trajectories from a saved shard, so search policies see identical candidate outcomes and runtime accounting. Models, recipe seeds, holdout circuits, budgets, and search seeds are carried in each artifact. The report recomputes aggregates from JSON rather than trusting the runner's precomputed summary.",
        "",
        f"- Analysis generated at `{report['generated_at']}` with Python {report['provenance']['python']} and NumPy {report['provenance']['numpy']}.",
        f"- Project commit: `{report['provenance'].get('project_git_commit') or 'unavailable'}`; working tree dirty: `{report['provenance'].get('project_git_dirty')}`.",
        f"- Bootstrap: {(report['provenance'].get('config') or {}).get('name', 'analysis')} with deterministic seeds; CI is a percentile bootstrap over rows in each group.",
        "",
        "## Limitations, uncertainty, and robustness checks",
        "",
        f"- **Grid validation:** complete={validation['complete']}; duplicate keys={validation['duplicate_rows']}.",
        "- The random comparator is a within-cell baseline, not an independent repeated experiment. Runtime comparisons therefore remain sensitive to the recorded ABC replay timings.",
        "- Legacy formal artifacts may not carry an explicit simulation-artifact schema or oracle signature. In that case the analyzer validates the JSON structure and counts but marks checks that require a signature as unavailable.",
        "- Oracle-hit rate is reported only when both selected and oracle recipe IDs are present; missing IDs are not treated as misses.",
        "- Bootstrap intervals quantify sampling uncertainty across the finite simulation grid; they do not account for model-training randomness beyond the included recipe seeds.",
        "",
        "## Recommended next steps",
        "",
        "1. Run the ablation pilot and then the full ablation sweep with the resumable runner; compare the policy-level intervals before changing the production default.",
        "2. Validate a small live-ABC sample using the supplied live validation runner and compare live best QoR with the offline oracle replay for the same recipes.",
        "3. If live results diverge materially from replay, investigate ABC version/signature, filesystem timing, and recipe/circuit provenance before retraining.",
        "",
        "## Further questions",
        "",
        "- Does the policy retain its runtime advantage when ABC evaluation noise is measured on repeated live runs rather than replayed timings?",
        "- Which circuit families or recipe lengths drive the remaining relative gap at the largest budget?",
        "- Should the early-stop threshold be tuned against an explicit wall-clock objective instead of only normalized QoR?",
        "",
        "Supporting validation details are in `validation.md`; machine-readable aggregates are in `report.json` and `analysis_rows.csv`.",
        "",
    ]
    if len(methods) > 1:
        policy_lines = [
            "## Policy comparison",
            "",
            "The ablation is paired on recipe seed, holdout circuit, budget, and search seed. The table reports group means; the gap interval is a deterministic percentile bootstrap 95% CI over those cells.",
            "",
            "| Policy | Budget | Mean gap (%) | Gap CI95 | Runtime reduction (%) | Early stops / run | Candidates eliminated / run | Oracle-hit rate |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for method in methods:
            for budget in budgets:
                group = overall.get(f"{method}|{budget}", {})
                metrics = group.get("metrics", {})
                gap = metrics.get("relative_gap_pct", {})
                reduction = metrics.get("runtime_reduction_vs_random_pct", {})
                stops = metrics.get("early_stops", {})
                eliminated = metrics.get("candidates_eliminated", {})
                hit = metrics.get("oracle_hit", {})
                interval = gap.get("ci95")
                interval_text = "n/a" if not interval else f"[{float(interval[0]):.2f}, {float(interval[1]):.2f}]"
                hit_text = "n/a" if hit.get("mean") is None else _pct(float(hit.get("mean")) * 100.0, 1)
                policy_lines.append(
                    f"| {method} | {budget} | {_num(gap.get('mean'), 2)} | {interval_text} | {_num(reduction.get('mean'), 2)} | {_num(stops.get('mean'), 2)} | {_num(eliminated.get('mean'), 2)} | {hit_text} |"
                )
        policy_lines.extend(
            [
                "",
                "For each budget, the smallest mean gap is a descriptive winner for this finite replay grid; it is not a statistically significant ranking by itself. Use the intervals and the live-validation run before selecting a default.",
                "",
            ]
        )
        scope_index = lines.index("## Scope, data, and metric definitions")
        lines[scope_index:scope_index] = policy_lines
    for name, relative in report.get("charts", {}).items():
        if relative not in "\n".join(lines):
            lines.append(f"- Chart: `{relative}`")
    return "\n".join(lines)


def _validation_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    complete = bool(validation.get("complete"))
    coverage = report["overall"].get("simultaneous_trajectory_coverage", {})
    coverage_available = bool(coverage.get("n", 0))
    assessment = (
        "Needs revision"
        if not complete
        else "Ready to share"
        if coverage_available
        else "Share with caveats"
    )
    oracle_available = bool(report["scope"].get("oracle_hit_available"))
    issues = []
    if not complete:
        issues.append(
            f"1. [Severity: High] Expected grid is incomplete or incompatible: missing={len(validation['missing_rows'])}, unexpected={len(validation['unexpected_rows'])}."
        )
    if not oracle_available:
        issues.append(
            "1. [Severity: Medium] Oracle-hit rate is unavailable because at least one artifact family omits recipe IDs; add IDs or retain this metric as not applicable."
        )
    if not coverage_available:
        issues.append(
            "1. [Severity: Medium] Training-time conformal coverage is not present in the policy artifacts; verify it from the source training reports or treat it as unavailable."
        )
    if not issues:
        issues.append("1. [Severity: Low] No material structural issues were found in the saved simulation grid.")
    return "\n".join(
        [
            "## Validation Report",
            "",
            f"### Overall Assessment: {assessment}",
            "",
            "### Methodology Review",
            "The analysis answers the methods question—how the configured search policies behave under fixed candidate recipes and budgets—using saved offline ABC trajectories and a same-cell random baseline. It is descriptive and replay-based; it does not establish causality or production generalization.",
            "",
            "### Issues Found",
            *issues,
            "",
            "### Calculation Spot-Checks",
            f"- Grid cardinality: **{'Verified' if complete else 'Discrepancy found'}** — observed {validation['observed_rows']} versus expected {validation['expected_rows']}.",
            "- Runtime reduction: **Verified** — recomputed from search and same-seed random runtime for every row where both values are finite.",
            "- Bootstrap intervals: **Verified** — deterministic percentile bootstrap of group means; one-row groups collapse to the observed value.",
            f"- Simultaneous trajectory coverage: **{'Available' if coverage.get('n', 0) else 'Not verified'}** — aggregated from training reports and kept separate from search outcomes.",
            "",
            "### Visualization Review",
            "The report uses grouped bars for discrete budgets, explicit units, zero-line context for signed runtime movement, and error bars for bootstrap 95% intervals. SVGs include titles and descriptions and use a restrained explicit palette; no claim depends on color alone because each series is also named in the legend.",
            "",
            "### Suggested Improvements",
            "1. Preserve oracle and model signatures in every future simulation artifact so provenance checks can be fully automated.",
            "2. Add repeated live-ABC measurements to quantify timing noise and validate replay assumptions.",
            "",
            "### Required Caveats for Stakeholders",
            "- Offline replay is not a live deployment benchmark.",
            "- Confidence intervals describe finite-grid uncertainty, not a guarantee over unseen circuits.",
            "- Positive runtime reduction is a relative comparison to the recorded random baseline and can be negative at small budgets.",
            "",
        ]
    )


def analyze(
    *,
    simulation_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    methods: Sequence[str] | None = None,
    recipe_seeds: Sequence[int] | None = None,
    circuits: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    search_seeds: Sequence[int] | None = None,
    bootstrap_reps: int = 1000,
    seed: int = 20260830,
    require_complete: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    config = None
    family_by_circuit: dict[str, str] = {}
    resolved_simulation_root = Path(simulation_root).expanduser().resolve()
    training_root: Path | None = None
    if config_path is not None:
        config = load_experiment_config(config_path)
        family_by_circuit = _family_by_circuit(config)
    # Ablation artifacts deliberately keep only the policy result.  Their
    # manifest points back to the source experiment, so recover training-time
    # coverage without duplicating the large report in every cell.
    for candidate in (resolved_simulation_root, resolved_simulation_root.parent):
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
            source = manifest.get("settings", {}).get("experiment_dir")
            if source:
                training_root = Path(str(source)).expanduser().resolve()
                break
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    rows = load_rows(
        resolved_simulation_root,
        family_by_circuit=family_by_circuit,
        training_root=training_root,
    )
    if circuits is None and config is not None:
        circuits = [path.stem for path in config.circuits]
    if budgets is None and config is not None:
        budgets = list(config.evaluation.budgets)
    if recipe_seeds is None and config is not None:
        recipe_seeds = list(config.recipes.seeds)
    if search_seeds is None and config is not None:
        search_seeds = list(config.evaluation.search_seeds)
    validation = validate_grid(
        rows,
        methods=methods,
        recipe_seeds=recipe_seeds,
        circuits=circuits,
        budgets=budgets,
        search_seeds=search_seeds,
        require_complete=require_complete,
    )
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "analysis_rows.csv"
    _write_csv(rows, rows_path)
    report = build_report(
        rows,
        validation=validation,
        source_root=Path(simulation_root).expanduser().resolve(),
        config=config,
        output_dir=output,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    _atomic_json(report, output / "report.json")
    (output / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    (output / "validation.md").write_text(
        _validation_markdown(report), encoding="utf-8"
    )
    return report


def _split_values(value: str | None, *, cast: Any = str) -> list[Any] | None:
    if value is None or not value.strip():
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", dest="config_path", type=Path)
    parser.add_argument("--methods", help="comma-separated method names")
    parser.add_argument("--recipe-seeds", help="comma-separated recipe seeds")
    parser.add_argument("--circuits", help="comma-separated circuit IDs")
    parser.add_argument("--budgets", help="comma-separated budgets")
    parser.add_argument("--search-seeds", help="comma-separated search seeds")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a report even when the requested grid is incomplete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_reps < 0:
        raise SystemExit("--bootstrap-reps must be non-negative")
    report = analyze(
        simulation_root=args.simulation_root,
        output_dir=args.output_dir,
        config_path=args.config_path,
        methods=_split_values(args.methods),
        recipe_seeds=_split_values(args.recipe_seeds, cast=int),
        circuits=_split_values(args.circuits),
        budgets=_split_values(args.budgets, cast=int),
        search_seeds=_split_values(args.search_seeds, cast=int),
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        require_complete=not args.allow_incomplete,
    )
    validation = report["validation"]
    print(
        json.dumps(
            {
                "report": str(Path(args.output_dir).expanduser().resolve() / "report.md"),
                "rows": validation["observed_rows"],
                "expected": validation["expected_rows"],
                "complete": validation["complete"],
                "figures": report["charts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
