from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .recipes import Recipe
from .types import NetworkStats, Trajectory, TrajectoryStep


_NODE_EFFECT = {
    "balance": 0.003,
    "rewrite": -0.035,
    "rewrite_z": -0.045,
    "refactor": -0.028,
    "refactor_z": -0.038,
    "resub": -0.032,
    "resub_z": -0.041,
    "dc2": -0.052,
}
_DEPTH_EFFECT = {
    "balance": -0.075,
    "rewrite": -0.012,
    "rewrite_z": 0.006,
    "refactor": -0.018,
    "refactor_z": 0.004,
    "resub": -0.008,
    "resub_z": 0.008,
    "dc2": -0.025,
}
_RUNTIME_FACTOR = {
    "balance": 0.7,
    "rewrite": 1.0,
    "rewrite_z": 1.25,
    "refactor": 1.15,
    "refactor_z": 1.35,
    "resub": 1.4,
    "resub_z": 1.65,
    "dc2": 1.9,
}


@dataclass(frozen=True)
class SyntheticCircuit:
    circuit_id: str
    initial: NetworkStats
    node_sensitivity: float
    depth_sensitivity: float
    family_bias: float


def _stable_noise(key: str, scale: float) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return (unit * 2.0 - 1.0) * scale


def make_synthetic_circuits(count: int, seed: int = 0) -> list[SyntheticCircuit]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    circuits: list[SyntheticCircuit] = []
    for index in range(count):
        nodes = int(math.exp(rng.uniform(math.log(1_500), math.log(90_000))))
        depth = rng.randint(20, 900)
        circuits.append(
            SyntheticCircuit(
                circuit_id=f"syn{index:02d}",
                initial=NetworkStats(
                    pis=rng.randint(8, 1_200),
                    pos=rng.randint(1, 400),
                    nodes=nodes,
                    depth=depth,
                ),
                node_sensitivity=rng.uniform(0.75, 1.25),
                depth_sensitivity=rng.uniform(0.72, 1.28),
                family_bias=(-1.0, 0.0, 1.0)[index % 3],
            )
        )
    return circuits


def run_synthetic_recipe(circuit: SyntheticCircuit, recipe: Recipe) -> Trajectory:
    trajectory = Trajectory(
        circuit_id=circuit.circuit_id,
        recipe_id=recipe.recipe_id,
        operations=recipe.operations,
        initial=circuit.initial,
        setup_runtime_s=0.0005,
    )
    stats = circuit.initial
    counts: dict[str, int] = {}
    cumulative = 0.0
    previous_action: str | None = None
    for step, action in enumerate(recipe.operations, start=1):
        counts[action] = counts.get(action, 0) + 1
        saturation = 1.0 / (1.0 + 0.18 * (counts[action] - 1))
        family_node = 0.004 * circuit.family_bias if action in {"dc2", "resub_z"} else 0.0
        family_depth = -0.004 * circuit.family_bias if action == "balance" else 0.0
        synergy = -0.012 if previous_action == "balance" and action.startswith("rewrite") else 0.0
        node_change = (
            _NODE_EFFECT[action] * circuit.node_sensitivity * saturation
            + family_node
            + synergy
            + _stable_noise(f"n:{circuit.circuit_id}:{recipe.recipe_id}:{step}", 0.008)
        )
        depth_change = (
            _DEPTH_EFFECT[action] * circuit.depth_sensitivity * saturation
            + family_depth
            + _stable_noise(f"d:{circuit.circuit_id}:{recipe.recipe_id}:{step}", 0.012)
        )
        nodes = max(circuit.initial.pis + circuit.initial.pos, int(round(stats.nodes * (1 + node_change))))
        depth = max(1, int(round(stats.depth * (1 + depth_change))))
        runtime = 0.0005 + _RUNTIME_FACTOR[action] * max(stats.nodes, 100) / 5_000_000
        cumulative += runtime
        stats = NetworkStats(
            pis=stats.pis,
            pos=stats.pos,
            nodes=nodes,
            depth=depth,
        )
        trajectory.steps.append(
            TrajectoryStep(
                step=step,
                action=action,
                stats=stats,
                runtime_s=runtime,
                cumulative_runtime_s=cumulative,
            )
        )
        previous_action = action
    trajectory.completed = True
    return trajectory


def generate_synthetic_trajectories(
    recipes: Sequence[Recipe],
    *,
    circuit_count: int = 12,
    seed: int = 0,
) -> list[Trajectory]:
    circuits = make_synthetic_circuits(circuit_count, seed)
    return [run_synthetic_recipe(circuit, recipe) for circuit in circuits for recipe in recipes]
