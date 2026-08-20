from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from .dataset import read_dataset
from .features import FeatureEncoder
from .types import Trajectory


@dataclass(frozen=True)
class Prediction:
    mean: float
    lower: float
    upper: float
    scale: float


@dataclass(frozen=True)
class TrainingReport:
    train_circuits: tuple[str, ...]
    calibration_circuits: tuple[str, ...]
    excluded_circuits: tuple[str, ...]
    train_rows: int
    calibration_rows: int
    calibration_trajectories: int
    alpha: float
    qhat: float
    scale_floor: float
    row_coverage: float
    simultaneous_trajectory_coverage: float
    mean_interval_width: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RiskModel:
    """Random-forest surrogate with trajectory-level split-conformal intervals.

    Calibration scores are maximized over every prefix of a trajectory. The
    resulting interval is therefore designed for repeated early-stopping checks,
    rather than only one fixed prediction point.
    """

    SERIALIZATION_VERSION = 1

    def __init__(
        self,
        forest: RandomForestRegressor,
        encoder: FeatureEncoder,
        *,
        qhat: float,
        scale_floor: float,
        alpha: float,
        train_circuits: Sequence[str],
        calibration_circuits: Sequence[str],
    ) -> None:
        self.forest = forest
        self.encoder = encoder
        self.qhat = float(qhat)
        self.scale_floor = float(scale_floor)
        self.alpha = float(alpha)
        self.train_circuits = tuple(train_circuits)
        self.calibration_circuits = tuple(calibration_circuits)

    def _predict_matrix(
        self, matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        tree_predictions = np.asarray(
            [tree.predict(matrix) for tree in self.forest.estimators_],
            dtype=np.float64,
        )
        means = tree_predictions.mean(axis=0)
        ddof = 1 if tree_predictions.shape[0] > 1 else 0
        scales = np.maximum(tree_predictions.std(axis=0, ddof=ddof), self.scale_floor)
        radii = self.qhat * scales
        return means, scales, means - radii, means + radii

    def predict_vectors(self, matrix: np.ndarray) -> list[Prediction]:
        means, scales, lowers, uppers = self._predict_matrix(matrix)
        return [
            Prediction(float(mean), float(lower), float(upper), float(scale))
            for mean, lower, upper, scale in zip(means, lowers, uppers, scales)
        ]

    def predict_trajectory(self, trajectory: Trajectory) -> Prediction:
        return self.predict_vectors(self.encoder.encode_trajectory(trajectory))[0]

    def predict_trajectories(
        self, trajectories: Sequence[Trajectory]
    ) -> list[Prediction]:
        """Predict several trajectory prefixes in one forest traversal."""
        if not trajectories:
            return []
        matrix = np.vstack(
            [self.encoder.encode_trajectory(trajectory) for trajectory in trajectories]
        )
        return self.predict_vectors(matrix)

    def save(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(
            {
                "serialization_version": self.SERIALIZATION_VERSION,
                "sklearn_version": sklearn.__version__,
                "model": self,
            },
            temporary,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, source: str | Path) -> "RiskModel":
        payload = joblib.load(Path(source))
        if payload.get("serialization_version") != cls.SERIALIZATION_VERSION:
            raise ValueError("unsupported model serialization version")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("artifact does not contain a RiskModel")
        return model


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        raise ValueError("cannot calibrate without scores")
    rank = math.ceil((scores.size + 1) * (1.0 - alpha))
    index = min(max(rank - 1, 0), scores.size - 1)
    return float(np.sort(scores)[index])


def _complete_prefix_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    result = []
    for row in rows:
        if row.get("completed") != "1" or not row.get("final_qor"):
            continue
        if int(row["step"]) >= int(row["recipe_length"]):
            continue
        result.append(row)
    return result


def train_risk_model(
    dataset: str | Path,
    *,
    excluded_circuits: Sequence[str] = (),
    calibration_circuits: Sequence[str] | None = None,
    calibration_fraction: float = 0.25,
    alpha: float = 0.01,
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
    seed: int = 0,
    n_jobs: int = -1,
) -> tuple[RiskModel, TrainingReport]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")
    if n_estimators < 10:
        raise ValueError("n_estimators must be at least 10")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    rows = _complete_prefix_rows(read_dataset(dataset))
    excluded = set(excluded_circuits)
    rows = [row for row in rows if row["circuit_id"] not in excluded]
    circuits = sorted({row["circuit_id"] for row in rows})
    if len(circuits) < 2:
        raise ValueError("training requires at least two non-excluded circuits")

    if calibration_circuits is None:
        rng = np.random.default_rng(seed)
        shuffled = list(circuits)
        rng.shuffle(shuffled)
        calibration_count = min(
            len(circuits) - 1,
            max(1, int(round(len(circuits) * calibration_fraction))),
        )
        calibration_set = set(shuffled[:calibration_count])
    else:
        calibration_set = set(calibration_circuits)
        missing = calibration_set - set(circuits)
        if missing:
            raise ValueError(f"calibration circuits not found: {sorted(missing)}")
        if calibration_set == set(circuits):
            raise ValueError("calibration split leaves no training circuits")

    train_rows = [row for row in rows if row["circuit_id"] not in calibration_set]
    calibration_rows_data = [row for row in rows if row["circuit_id"] in calibration_set]
    max_recipe_length = max(int(row["recipe_length"]) for row in rows)
    encoder = FeatureEncoder(max_recipe_length=max_recipe_length)
    x_train = np.vstack([encoder.encode_row(row) for row in train_rows])
    y_train = np.asarray([float(row["final_qor"]) for row in train_rows])

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=0.8,
        n_jobs=n_jobs,
        random_state=seed,
    )
    forest.fit(x_train, y_train)

    x_calibration = np.vstack([encoder.encode_row(row) for row in calibration_rows_data])
    y_calibration = np.asarray(
        [float(row["final_qor"]) for row in calibration_rows_data]
    )
    tree_predictions = np.asarray(
        [tree.predict(x_calibration) for tree in forest.estimators_],
        dtype=np.float64,
    )
    means = tree_predictions.mean(axis=0)
    raw_scales = tree_predictions.std(axis=0, ddof=1)
    positive_scales = raw_scales[raw_scales > 0]
    scale_floor = float(np.quantile(positive_scales, 0.1)) if positive_scales.size else 1e-3
    scale_floor = max(scale_floor, 1e-4)
    scales = np.maximum(raw_scales, scale_floor)
    normalized_errors = np.abs(y_calibration - means) / scales

    trajectory_maxima: dict[tuple[str, str], float] = {}
    for row, score in zip(calibration_rows_data, normalized_errors):
        key = (row["circuit_id"], row["recipe_id"])
        trajectory_maxima[key] = max(trajectory_maxima.get(key, 0.0), float(score))
    max_scores = np.asarray(list(trajectory_maxima.values()), dtype=np.float64)
    qhat = _finite_sample_quantile(max_scores, alpha)
    radii = qhat * scales
    covered = np.abs(y_calibration - means) <= radii

    trajectory_covered: dict[tuple[str, str], bool] = {}
    widths_by_row: list[float] = []
    for row, is_covered, radius in zip(calibration_rows_data, covered, radii):
        key = (row["circuit_id"], row["recipe_id"])
        trajectory_covered[key] = trajectory_covered.get(key, True) and bool(is_covered)
        widths_by_row.append(float(2 * radius))

    train_circuits = tuple(sorted(set(circuits) - calibration_set))
    calibration_circuit_tuple = tuple(sorted(calibration_set))
    model = RiskModel(
        forest,
        encoder,
        qhat=qhat,
        scale_floor=scale_floor,
        alpha=alpha,
        train_circuits=train_circuits,
        calibration_circuits=calibration_circuit_tuple,
    )
    report = TrainingReport(
        train_circuits=train_circuits,
        calibration_circuits=calibration_circuit_tuple,
        excluded_circuits=tuple(sorted(excluded)),
        train_rows=len(train_rows),
        calibration_rows=len(calibration_rows_data),
        calibration_trajectories=len(trajectory_maxima),
        alpha=alpha,
        qhat=qhat,
        scale_floor=scale_floor,
        row_coverage=float(np.mean(covered)),
        simultaneous_trajectory_coverage=float(np.mean(list(trajectory_covered.values()))),
        mean_interval_width=float(np.mean(widths_by_row)),
    )
    return model, report
