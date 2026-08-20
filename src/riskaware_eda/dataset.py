from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .abc_runner import ABCRunner
from .qor import QoRWeights, normalized_qor
from .recipes import Recipe
from .types import NetworkStats, Trajectory


DATASET_FIELDS = (
    "circuit_id",
    "recipe_id",
    "recipe",
    "recipe_length",
    "step",
    "action",
    "completed",
    "pis",
    "pos",
    "initial_nodes",
    "initial_depth",
    "previous_nodes",
    "previous_depth",
    "nodes",
    "depth",
    "node_ratio",
    "depth_ratio",
    "qor",
    "runtime_s",
    "cumulative_runtime_s",
    "final_nodes",
    "final_depth",
    "final_qor",
    "error",
)


def trajectory_rows(
    trajectory: Trajectory,
    weights: QoRWeights = QoRWeights(),
) -> Iterator[dict[str, object]]:
    final_stats = trajectory.current_stats if trajectory.completed else None
    final_qor = (
        normalized_qor(final_stats, trajectory.initial, weights)
        if final_stats is not None
        else ""
    )
    states: list[tuple[int, str, NetworkStats, NetworkStats, float, float]] = [
        (0, "__start__", trajectory.initial, trajectory.initial, 0.0, 0.0)
    ]
    previous = trajectory.initial
    for item in trajectory.steps:
        states.append(
            (
                item.step,
                item.action,
                item.stats,
                previous,
                item.runtime_s,
                item.cumulative_runtime_s,
            )
        )
        previous = item.stats

    for step, action, stats, prior, runtime, cumulative in states:
        yield {
            "circuit_id": trajectory.circuit_id,
            "recipe_id": trajectory.recipe_id,
            "recipe": "|".join(trajectory.operations),
            "recipe_length": len(trajectory.operations),
            "step": step,
            "action": action,
            "completed": int(trajectory.completed),
            "pis": trajectory.initial.pis,
            "pos": trajectory.initial.pos,
            "initial_nodes": trajectory.initial.nodes,
            "initial_depth": trajectory.initial.depth,
            "previous_nodes": prior.nodes,
            "previous_depth": prior.depth,
            "nodes": stats.nodes,
            "depth": stats.depth,
            "node_ratio": stats.nodes / max(trajectory.initial.nodes, 1),
            "depth_ratio": stats.depth / max(trajectory.initial.depth, 1),
            "qor": normalized_qor(stats, trajectory.initial, weights),
            "runtime_s": runtime,
            "cumulative_runtime_s": cumulative,
            "final_nodes": final_stats.nodes if final_stats else "",
            "final_depth": final_stats.depth if final_stats else "",
            "final_qor": final_qor,
            "error": trajectory.error or "",
        }


def write_trajectories(
    trajectories: Iterable[Trajectory],
    destination: str | Path,
    *,
    weights: QoRWeights = QoRWeights(),
) -> dict[str, int]:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    trajectory_count = row_count = completed_count = failed_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for trajectory in trajectories:
            trajectory_count += 1
            completed_count += int(trajectory.completed)
            failed_count += int(trajectory.error is not None)
            for row in trajectory_rows(trajectory, weights):
                writer.writerow(row)
                row_count += 1
    temporary.replace(path)
    return {
        "trajectories": trajectory_count,
        "completed": completed_count,
        "failed": failed_count,
        "rows": row_count,
    }


def read_dataset(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_abc_trajectories(
    circuits: Sequence[Path],
    recipes: Sequence[Recipe],
    runner: ABCRunner,
    *,
    jobs: int = 1,
) -> list[Trajectory]:
    if jobs < 1:
        raise ValueError("jobs must be positive")

    def collect_one(circuit: Path) -> list[Trajectory]:
        with runner.session(circuit) as session:
            return [session.run_recipe(recipe) for recipe in recipes]

    if jobs == 1:
        return [item for circuit in circuits for item in collect_one(circuit)]

    by_circuit: dict[str, list[Trajectory]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(collect_one, circuit): circuit for circuit in circuits}
        for future in as_completed(futures):
            circuit = futures[future]
            by_circuit[str(circuit)] = future.result()
    return [item for circuit in circuits for item in by_circuit[str(circuit)]]
