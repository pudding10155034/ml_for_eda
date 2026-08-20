from __future__ import annotations

from dataclasses import dataclass

from .types import NetworkStats


@dataclass(frozen=True)
class QoRWeights:
    """Weights for a normalized node/depth minimization objective."""

    nodes: float = 0.5
    depth: float = 0.5

    def __post_init__(self) -> None:
        if self.nodes < 0 or self.depth < 0:
            raise ValueError("QoR weights must be non-negative")
        if self.nodes + self.depth <= 0:
            raise ValueError("at least one QoR weight must be positive")

    @property
    def normalized(self) -> tuple[float, float]:
        total = self.nodes + self.depth
        return self.nodes / total, self.depth / total


def normalized_qor(
    stats: NetworkStats,
    initial: NetworkStats,
    weights: QoRWeights = QoRWeights(),
) -> float:
    """Return a scalar loss; lower is better and the initial network is 1.0."""

    node_weight, depth_weight = weights.normalized
    node_ratio = stats.nodes / max(initial.nodes, 1)
    depth_ratio = stats.depth / max(initial.depth, 1)
    return node_weight * node_ratio + depth_weight * depth_ratio
