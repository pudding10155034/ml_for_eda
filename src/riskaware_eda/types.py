from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class NetworkStats:
    """Compact AIG statistics emitted by ABC ``print_stats``."""

    pis: int
    pos: int
    nodes: int
    depth: int
    latches: int = 0

    def __post_init__(self) -> None:
        for name in ("pis", "pos", "nodes", "depth", "latches"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TrajectoryStep:
    """One observed state after executing an ABC operator."""

    step: int
    action: str
    stats: NetworkStats
    runtime_s: float
    cumulative_runtime_s: float

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("trajectory steps are one-indexed")
        if self.runtime_s < 0 or self.cumulative_runtime_s < 0:
            raise ValueError("runtime must be non-negative")


@dataclass
class Trajectory:
    """A complete or early-stopped synthesis-recipe execution."""

    circuit_id: str
    recipe_id: str
    operations: tuple[str, ...]
    initial: NetworkStats
    steps: list[TrajectoryStep] = field(default_factory=list)
    setup_runtime_s: float = 0.0
    completed: bool = False
    stopped_early: bool = False
    stop_reason: str | None = None
    error: str | None = None

    @property
    def current_stats(self) -> NetworkStats:
        return self.steps[-1].stats if self.steps else self.initial

    @property
    def cumulative_runtime_s(self) -> float:
        return self.steps[-1].cumulative_runtime_s if self.steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StopCallback = Callable[[Trajectory], str | None]
TrajectoryRunner = Callable[[str, Sequence[str], StopCallback | None], Trajectory]
