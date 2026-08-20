from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .recipes import DEFAULT_OPERATORS
from .types import NetworkStats, Trajectory


@dataclass(frozen=True)
class FeatureEncoder:
    """Fixed, circuit-agnostic encoding for a recipe prefix and current AIG state."""

    operators: tuple[str, ...] = DEFAULT_OPERATORS
    max_recipe_length: int = 20

    def __post_init__(self) -> None:
        if self.max_recipe_length < 1:
            raise ValueError("max_recipe_length must be positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [
            "log_pis",
            "log_pos",
            "log_initial_nodes",
            "log_initial_depth",
            "log_current_nodes",
            "log_current_depth",
            "node_ratio",
            "depth_ratio",
            "node_improvement",
            "depth_improvement",
            "delta_node_ratio",
            "delta_depth_ratio",
            "step_fraction",
            "remaining_fraction",
            "recipe_length_fraction",
        ]
        for operator in self.operators:
            names.extend(
                (
                    f"total_{operator}",
                    f"prefix_{operator}",
                    f"remaining_{operator}",
                    f"last_{operator}",
                )
            )
        names.append("last_start")
        for position in range(self.max_recipe_length):
            for operator in self.operators:
                names.append(f"pos_{position}_{operator}")
        return tuple(names)

    def encode(
        self,
        *,
        initial: NetworkStats,
        current: NetworkStats,
        previous: NetworkStats | None,
        operations: Sequence[str],
        step: int,
        last_action: str | None,
    ) -> np.ndarray:
        length = len(operations)
        if length < 1:
            raise ValueError("operations cannot be empty")
        if step < 0 or step > length:
            raise ValueError("step is outside the recipe")
        previous = previous or current
        initial_nodes = max(initial.nodes, 1)
        initial_depth = max(initial.depth, 1)
        scale = max(length, 1)
        values: list[float] = [
            math.log1p(initial.pis),
            math.log1p(initial.pos),
            math.log1p(initial.nodes),
            math.log1p(initial.depth),
            math.log1p(current.nodes),
            math.log1p(current.depth),
            current.nodes / initial_nodes,
            current.depth / initial_depth,
            (initial.nodes - current.nodes) / initial_nodes,
            (initial.depth - current.depth) / initial_depth,
            (current.nodes - previous.nodes) / initial_nodes,
            (current.depth - previous.depth) / initial_depth,
            step / scale,
            (length - step) / scale,
            min(length, self.max_recipe_length) / self.max_recipe_length,
        ]
        prefix = operations[:step]
        remaining = operations[step:]
        for operator in self.operators:
            values.extend(
                (
                    operations.count(operator) / scale,
                    prefix.count(operator) / scale,
                    remaining.count(operator) / scale,
                    float(last_action == operator),
                )
            )
        values.append(float(last_action is None))
        for position in range(self.max_recipe_length):
            operation = operations[position] if position < length else None
            values.extend(float(operation == operator) for operator in self.operators)
        return np.asarray(values, dtype=np.float64)

    def encode_trajectory(self, trajectory: Trajectory) -> np.ndarray:
        if trajectory.steps:
            current = trajectory.steps[-1].stats
            previous = (
                trajectory.steps[-2].stats
                if len(trajectory.steps) >= 2
                else trajectory.initial
            )
            step = trajectory.steps[-1].step
            last_action = trajectory.steps[-1].action
        else:
            current = previous = trajectory.initial
            step = 0
            last_action = None
        return self.encode(
            initial=trajectory.initial,
            current=current,
            previous=previous,
            operations=trajectory.operations,
            step=step,
            last_action=last_action,
        )

    def encode_row(self, row: Mapping[str, str]) -> np.ndarray:
        operations = tuple(filter(None, row["recipe"].split("|")))
        initial = NetworkStats(
            pis=int(row["pis"]),
            pos=int(row["pos"]),
            nodes=int(row["initial_nodes"]),
            depth=int(row["initial_depth"]),
        )
        current = NetworkStats(
            pis=initial.pis,
            pos=initial.pos,
            nodes=int(row["nodes"]),
            depth=int(row["depth"]),
        )
        previous = NetworkStats(
            pis=initial.pis,
            pos=initial.pos,
            nodes=int(row["previous_nodes"]),
            depth=int(row["previous_depth"]),
        )
        action = row.get("action") or None
        if action == "__start__":
            action = None
        return self.encode(
            initial=initial,
            current=current,
            previous=previous,
            operations=operations,
            step=int(row["step"]),
            last_action=action,
        )
