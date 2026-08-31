import csv

import pytest

from scripts.analyze_live import analyze, validate_grid


def _write_results(path):
    fields = [
        "recipe_seed",
        "holdout_circuit",
        "method",
        "repeat",
        "budget",
        "search_seed",
        "relative_gap_pct",
        "live_wall_s",
        "selected",
        "completed_evaluations",
        "early_stops",
        "best_recipe_id",
        "oracle_best_recipe_id",
        "termination_reason",
    ]
    rows = []
    for method, offset in (("risk_aware", 0.0), ("random", 1.0)):
        for repeat in range(2):
            rows.append(
                {
                    "recipe_seed": 7,
                    "holdout_circuit": "adder",
                    "method": method,
                    "repeat": repeat,
                    "budget": 2,
                    "search_seed": 0,
                    "relative_gap_pct": offset + repeat,
                    "live_wall_s": 0.5 + offset * 0.1 + repeat * 0.05,
                    "selected": 2,
                    "completed_evaluations": 2,
                    "early_stops": 0,
                    "best_recipe_id": "r0",
                    "oracle_best_recipe_id": "r0",
                    "termination_reason": "budget_exhausted",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_live_analyzer_reports_policy_and_repeat_noise(tmp_path):
    results = tmp_path / "results.csv"
    _write_results(results)
    settings = tmp_path / "settings.json"
    settings.write_text(
        '{"recipe_seed": 7, "circuits": ["adder"], "budgets": [2], '
        '"search_seeds": [0], "methods": ["risk_aware", "random"], '
        '"repeats": 2}',
        encoding="utf-8",
    )
    report = analyze(
        results_path=results,
        output_dir=tmp_path / "analysis",
        settings_path=settings,
        bootstrap_reps=20,
    )
    assert report["validation"]["complete"] is True
    assert report["scope"]["repeats"] == 2
    assert set(report["by_method_budget"]) == {"random|2", "risk_aware|2"}
    assert set(report["repeat_noise_by_method"]) == {"random", "risk_aware"}
    assert report["winners_by_budget"]["2"]["lowest_gap_method"] == "risk_aware"
    assert (tmp_path / "analysis" / "report.json").is_file()
    assert (tmp_path / "analysis" / "report.md").is_file()


def test_live_grid_detects_missing_repeat():
    rows = [
        {
            "method": "risk_aware",
            "recipe_seed": 7,
            "holdout_circuit": "adder",
            "repeat": 0,
            "budget": 2,
            "search_seed": 0,
        }
    ]
    with pytest.raises(ValueError, match="incomplete live validation grid"):
        validate_grid(
            rows,
            axes={
                "methods": ["risk_aware"],
                "recipe_seeds": [7],
                "circuits": ["adder"],
                "repeats": 2,
                "budgets": [2],
                "search_seeds": [0],
            },
        )


def test_live_analyzer_accepts_legacy_csv_without_method_columns(tmp_path):
    results = tmp_path / "legacy.csv"
    fields = [
        "recipe_seed",
        "holdout_circuit",
        "budget",
        "search_seed",
        "relative_gap_pct",
        "live_wall_s",
    ]
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "recipe_seed": 7,
                "holdout_circuit": "adder",
                "budget": 2,
                "search_seed": 0,
                "relative_gap_pct": 0.0,
                "live_wall_s": 0.5,
            }
        )
    report = analyze(
        results_path=results,
        output_dir=tmp_path / "analysis",
        bootstrap_reps=0,
    )
    assert report["validation"]["complete"] is True
    assert "risk_aware|2" in report["by_method_budget"]
