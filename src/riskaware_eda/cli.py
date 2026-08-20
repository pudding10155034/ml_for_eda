from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .abc_runner import ABCRunner, iter_circuit_files
from .dataset import collect_abc_trajectories, write_trajectories
from .model import RiskModel, train_risk_model
from .recipes import DEFAULT_OPERATORS, generate_recipes, load_recipes, save_recipes
from .search import run_live_search
from .simulation import simulate_search
from .synthetic import generate_synthetic_trajectories


def _write_json(payload: Any, destination: str | Path | None = None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if destination is None:
        print(rendered)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    temporary.replace(path)


def _split_operators(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _command_recipes(args: argparse.Namespace) -> dict[str, object]:
    operators = _split_operators(args.operators)
    recipes = generate_recipes(
        args.count,
        args.length,
        seed=args.seed,
        operators=operators,
        max_consecutive=args.max_consecutive,
    )
    save_recipes(recipes, args.output, seed=args.seed)
    return {
        "output": str(Path(args.output).resolve()),
        "count": len(recipes),
        "length": args.length,
        "operators": list(operators),
        "seed": args.seed,
    }


def _command_synthetic(args: argparse.Namespace) -> dict[str, object]:
    recipes = load_recipes(args.recipes)
    trajectories = generate_synthetic_trajectories(
        recipes,
        circuit_count=args.circuits,
        seed=args.seed,
    )
    report = write_trajectories(trajectories, args.output)
    return {
        "output": str(Path(args.output).resolve()),
        "synthetic_circuits": args.circuits,
        **report,
    }


def _command_collect(args: argparse.Namespace) -> dict[str, object]:
    circuits = list(iter_circuit_files(args.circuits))
    if not circuits:
        raise ValueError("no supported circuit files were found")
    recipes = load_recipes(args.recipes)
    runner = ABCRunner(
        args.abc,
        timeout_s=args.timeout,
        keep_workdir=args.keep_workdirs,
    )
    trajectories = collect_abc_trajectories(
        circuits,
        recipes,
        runner,
        jobs=args.jobs,
    )
    report = write_trajectories(trajectories, args.output)
    return {
        "output": str(Path(args.output).resolve()),
        "circuits": [str(path) for path in circuits],
        "recipes": len(recipes),
        **report,
    }


def _command_train(args: argparse.Namespace) -> dict[str, object]:
    calibration = args.calibration_circuit or None
    model, report = train_risk_model(
        args.dataset,
        excluded_circuits=args.exclude_circuit,
        calibration_circuits=calibration,
        calibration_fraction=args.calibration_fraction,
        alpha=args.alpha,
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    model.save(args.output)
    payload = report.to_dict()
    payload["model"] = str(Path(args.output).resolve())
    if args.report:
        _write_json(payload, args.report)
    return payload


def _command_simulate(args: argparse.Namespace) -> dict[str, object]:
    model = RiskModel.load(args.model)
    result = simulate_search(
        args.dataset,
        args.circuit,
        model,
        budget=args.budget,
        seed=args.seed,
        min_steps_before_stopping=args.min_stop_step,
    )
    payload = result.to_dict()
    if args.output:
        _write_json(payload, args.output)
    return payload


def _command_search(args: argparse.Namespace) -> dict[str, object]:
    model = RiskModel.load(args.model)
    recipes = load_recipes(args.recipes)
    runner = ABCRunner(
        args.abc,
        timeout_s=args.timeout,
        keep_workdir=args.keep_workdir,
    )
    with runner.session(args.circuit) as session:
        result = run_live_search(
            model=model,
            session=session,
            recipes=recipes,
            budget=args.budget,
            seed=args.seed,
            min_steps_before_stopping=args.min_stop_step,
        )
    payload = result.to_dict()
    if args.output:
        _write_json(payload, args.output)
    return payload


def _command_demo(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recipes_path = output / "recipes.json"
    dataset_path = output / "synthetic_trajectories.csv"
    model_path = output / "risk_model.joblib"
    report_path = output / "training_report.json"
    simulation_path = output / "simulation.json"

    recipes = generate_recipes(
        args.recipes,
        args.length,
        seed=args.seed,
    )
    save_recipes(recipes, recipes_path, seed=args.seed)
    trajectories = generate_synthetic_trajectories(
        recipes,
        circuit_count=args.circuits,
        seed=args.seed,
    )
    dataset_report = write_trajectories(trajectories, dataset_path)
    target = f"syn{args.circuits - 1:02d}"
    model, training_report = train_risk_model(
        dataset_path,
        excluded_circuits=[target],
        calibration_fraction=0.25,
        alpha=args.alpha,
        n_estimators=args.trees,
        seed=args.seed,
    )
    model.save(model_path)
    _write_json(training_report.to_dict(), report_path)
    simulation = simulate_search(
        dataset_path,
        target,
        model,
        budget=min(args.budget, args.recipes),
        seed=args.seed,
    )
    _write_json(simulation.to_dict(), simulation_path)
    return {
        "output": str(output.resolve()),
        "target_circuit": target,
        "dataset": dataset_report,
        "training": training_report.to_dict(),
        "simulation": simulation.to_dict(),
        "artifacts": {
            "recipes": str(recipes_path.resolve()),
            "dataset": str(dataset_path.resolve()),
            "model": str(model_path.resolve()),
            "training_report": str(report_path.resolve()),
            "simulation": str(simulation_path.resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskaware-eda",
        description="Risk-aware budgeted search for Berkeley ABC synthesis recipes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipes = subparsers.add_parser("recipes", help="generate candidate recipes")
    recipes.add_argument("--output", required=True)
    recipes.add_argument("--count", type=int, default=500)
    recipes.add_argument("--length", type=int, default=10)
    recipes.add_argument("--seed", type=int, default=0)
    recipes.add_argument("--max-consecutive", type=int, default=2)
    recipes.add_argument("--operators", default=",".join(DEFAULT_OPERATORS))
    recipes.set_defaults(handler=_command_recipes)

    synthetic = subparsers.add_parser(
        "synthetic-data", help="create a small deterministic development dataset"
    )
    synthetic.add_argument("--recipes", required=True)
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--circuits", type=int, default=12)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.set_defaults(handler=_command_synthetic)

    collect = subparsers.add_parser("collect", help="run recipes on real ABC circuits")
    collect.add_argument("--abc", default="abc")
    collect.add_argument("--circuits", nargs="+", required=True)
    collect.add_argument("--recipes", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--jobs", type=int, default=1)
    collect.add_argument("--timeout", type=float, default=120.0)
    collect.add_argument("--keep-workdirs", action="store_true")
    collect.set_defaults(handler=_command_collect)

    train = subparsers.add_parser("train", help="train and calibrate the surrogate")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--report")
    train.add_argument("--exclude-circuit", action="append", default=[])
    train.add_argument("--calibration-circuit", action="append", default=[])
    train.add_argument("--calibration-fraction", type=float, default=0.25)
    train.add_argument("--alpha", type=float, default=0.01)
    train.add_argument("--trees", type=int, default=200)
    train.add_argument("--min-samples-leaf", type=int, default=2)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(handler=_command_train)

    simulate = subparsers.add_parser(
        "simulate", help="replay a held-out circuit as a zero-cost oracle"
    )
    simulate.add_argument("--dataset", required=True)
    simulate.add_argument("--model", required=True)
    simulate.add_argument("--circuit", required=True)
    simulate.add_argument("--budget", type=int, default=20)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--min-stop-step", type=int, default=2)
    simulate.add_argument("--output")
    simulate.set_defaults(handler=_command_simulate)

    search = subparsers.add_parser("search", help="run live budgeted search with ABC")
    search.add_argument("--abc", default="abc")
    search.add_argument("--circuit", required=True)
    search.add_argument("--recipes", required=True)
    search.add_argument("--model", required=True)
    search.add_argument("--budget", type=int, default=20)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--min-stop-step", type=int, default=2)
    search.add_argument("--timeout", type=float, default=120.0)
    search.add_argument("--keep-workdir", action="store_true")
    search.add_argument("--output")
    search.set_defaults(handler=_command_search)

    demo = subparsers.add_parser("demo", help="run the full pipeline synthetically")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--circuits", type=int, default=10)
    demo.add_argument("--recipes", type=int, default=80)
    demo.add_argument("--length", type=int, default=8)
    demo.add_argument("--budget", type=int, default=15)
    demo.add_argument("--alpha", type=float, default=0.01)
    demo.add_argument("--trees", type=int, default=120)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(handler=_command_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    _write_json(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
