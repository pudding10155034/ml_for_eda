import csv
import json

import pytest

from scripts.validate_live import (
    LIVE_SCHEMA_VERSION,
    _aggregate,
    _artifact_complete,
    _artifact_path,
    _load_settings,
    _settings_compatible,
    build_parser,
    run,
)


def _write_base_config(tmp_path):
    (tmp_path / "fake-abc").write_bytes(b"abc")
    circuits = []
    for index in range(3):
        path = tmp_path / f"c{index}.aig"
        path.write_bytes(b"aiger")
        circuits.append(path.name)
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "live-test",
                "project_root": ".",
                "abc": "fake-abc",
                "circuits": circuits,
                "recipes": {"count": 4, "length": 2, "seeds": [7]},
                "collection": {"jobs": 1, "timeout_s": 10},
                "training": {
                    "alpha": 0.1,
                    "trees": 10,
                    "min_samples_leaf": 1,
                    "calibration_fraction": 0.25,
                    "model_jobs": 1,
                },
                "evaluation": {
                    "budgets": [2],
                    "search_seeds": [0],
                    "min_stop_step": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return base


def test_live_settings_accept_methods_and_repeats(tmp_path):
    base = _write_base_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "base_config": base.name,
                "experiment_dir": "source",
                "output_dir": "output",
                "recipe_seed": 7,
                "circuits": ["c0"],
                "budgets": [2],
                "search_seeds": [0],
                "methods": ["random", "risk_aware"],
                "repeats": 3,
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--settings", str(settings_path)])
    config, values = _load_settings(args)

    assert config.name == "live-test"
    assert values["methods"] == ["random", "risk_aware"]
    assert values["repeats"] == 3
    assert values["recipe_seeds"] == [7]
    assert len(values["settings_fingerprint"]) == 64


def test_live_settings_accept_multiple_recipe_seeds_from_cli(tmp_path):
    base = _write_base_config(tmp_path)
    # Add a second configured seed to the base experiment.
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["recipes"]["seeds"] = [3, 7]
    base.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "source").mkdir()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"base_config": base.name, "experiment_dir": "source"}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--settings", str(settings_path), "--recipe-seeds", "7,3"]
    )
    _, values = _load_settings(args)
    assert values["recipe_seeds"] == [3, 7]
    assert values["recipe_seed"] is None


def test_live_dry_run_does_not_create_output_directory(tmp_path):
    base = _write_base_config(tmp_path)
    (tmp_path / "source").mkdir()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"base_config": base.name, "experiment_dir": "source", "output_dir": "output"}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--settings", str(settings_path), "--dry-run"])
    result = run(args)
    assert result["dry_run"] is True
    assert not (tmp_path / "output").exists()


def test_live_settings_reject_unknown_method(tmp_path):
    base = _write_base_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"base_config": base.name, "methods": ["bogus"]}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--settings", str(settings_path)])
    with pytest.raises(ValueError, match="unknown live-validation methods"):
        _load_settings(args)


def test_artifact_layout_and_legacy_schema_resume(tmp_path):
    legacy = _artifact_path(
        tmp_path, 7, "c0", 2, 0, method="risk_aware", repeat=0, legacy_layout=True
    )
    nested = _artifact_path(
        tmp_path, 7, "c0", 2, 0, method="random", repeat=2, legacy_layout=False
    )
    assert legacy.parts[-2:] == ("c0", "budget_0002_search_00000.json")
    assert nested.parts[-3:] == (
        "method_random",
        "repeat_002",
        "budget_0002_search_00000.json",
    )

    signatures = {
        "model": {"size": 1, "mtime_ns": 2},
        "recipe": {"size": 3, "mtime_ns": 4},
        "circuit": {"size": 5, "mtime_ns": 6},
    }
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        json.dumps(
            {
                "live_validation_schema_version": 1,
                "settings_fingerprint": "legacy",
                "recipe_seed": 7,
                "holdout_circuit": "c0",
                "budget": 2,
                "search_seed": 0,
                "model_signature": signatures["model"],
                "recipe_signature": signatures["recipe"],
                "circuit_signature": signatures["circuit"],
                "search": {"best_qor": 1.0},
            }
        ),
        encoding="utf-8",
    )
    assert _artifact_complete(
        legacy,
        {
            "settings_fingerprint": "new",
            "recipe_seed": 7,
            "holdout_circuit": "c0",
            "method": "risk_aware",
            "repeat": 0,
            "budget": 2,
            "search_seed": 0,
        },
        signatures,
        legacy_settings_fingerprint="legacy",
    )


def test_aggregate_counts_method_and_repeat(tmp_path):
    settings = {
        "settings_fingerprint": "settings",
        "recipe_seed": 7,
        "recipe_seeds": [7],
        "circuits": ["c0"],
        "budgets": [2],
        "search_seeds": [0],
        "methods": ["risk_aware", "random"],
        "repeats": 2,
    }
    for method in settings["methods"]:
        for repeat in range(settings["repeats"]):
            path = _artifact_path(
                tmp_path,
                7,
                "c0",
                2,
                0,
                method=method,
                repeat=repeat,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "live_validation_schema_version": LIVE_SCHEMA_VERSION,
                        "recipe_seed": 7,
                        "holdout_circuit": "c0",
                        "method": method,
                        "repeat": repeat,
                        "budget": 2,
                        "search_seed": 0,
                        "search": {
                            "best_qor": 1.0,
                            "selected": 2,
                            "completed_evaluations": 2,
                        },
                        "relative_gap_pct": 0.0,
                        "live_wall_s": 0.1 + repeat,
                    }
                ),
                encoding="utf-8",
            )

    aggregate = _aggregate(tmp_path, settings)
    assert aggregate["runs"] == 4
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_runs"] == 4
    assert summary["complete"] is True
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["method"], int(row["repeat"])) for row in rows} == {
        ("random", 0),
        ("random", 1),
        ("risk_aware", 0),
        ("risk_aware", 1),
    }


def test_schema_one_default_manifest_is_compatible():
    previous = {
        "live_validation_schema_version": 1,
        "settings": {
            "a": 1,
            "recipe_seed": 0,
            "settings_fingerprint": "ignored",
        },
        "settings_fingerprint": "8a4c3e1b34f423a6b2b3a9f7d22d2f7b1d7343c9d7393f4e5ed8f3a0de9b9b37",
    }
    current = {
        "a": 1,
        "recipe_seed": 0,
        "methods": ["risk_aware"],
        "repeats": 1,
        "settings_fingerprint": "different",
    }
    # Use the implementation's fingerprint to avoid depending on a literal hash.
    from scripts.validate_live import _settings_fingerprint

    previous["settings_fingerprint"] = _settings_fingerprint(
        {"a": 1, "recipe_seed": 0}
    )
    assert _settings_compatible(previous, current)


def test_schema_two_single_seed_manifest_migrates():
    from scripts.validate_live import _settings_fingerprint

    previous_settings = {
        "a": 1,
        "recipe_seed": 7,
        "methods": ["risk_aware"],
        "repeats": 1,
    }
    previous = {
        "live_validation_schema_version": 2,
        "settings": previous_settings,
        "settings_fingerprint": _settings_fingerprint(previous_settings),
    }
    current = {
        "a": 1,
        "recipe_seed": 7,
        "recipe_seeds": [7],
        "methods": ["risk_aware"],
        "repeats": 1,
        "settings_fingerprint": "different",
    }
    assert _settings_compatible(previous, current)
