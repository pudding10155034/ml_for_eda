from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .abc_runner import ABCRunner, iter_circuit_files
from .dataset import DATASET_FIELDS, trajectory_rows
from .model import RiskModel, train_risk_model
from .recipes import Recipe, generate_recipes, load_recipes, save_recipes
from .simulation import OracleSimulator, load_oracle_trajectories


RUNNER_SCHEMA_VERSION = 1
COLLECTION_PROGRESS_SCHEMA_VERSION = 1
SIMULATION_ARTIFACT_SCHEMA_VERSION = 2
PROGRESS_LOG_INTERVAL = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a JSON object")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _number_tuple(value: Any, label: str, *, positive: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    result = tuple(int(item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} values must be unique")
    if positive and any(item < 1 for item in result):
        raise ValueError(f"{label} values must be positive")
    if not positive and any(item < 0 for item in result):
        raise ValueError(f"{label} values must be non-negative")
    return result


@dataclass(frozen=True)
class RecipePlan:
    count: int
    length: int
    seeds: tuple[int, ...]
    max_consecutive: int = 2


@dataclass(frozen=True)
class CollectionPlan:
    jobs: int = 1
    timeout_s: float = 120.0


@dataclass(frozen=True)
class TrainingPlan:
    alpha: float = 0.01
    trees: int = 200
    min_samples_leaf: int = 2
    calibration_fraction: float = 0.25
    model_jobs: int = 1


@dataclass(frozen=True)
class EvaluationPlan:
    budgets: tuple[int, ...]
    search_seeds: tuple[int, ...]
    min_stop_step: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    project_root: Path
    name: str
    output_dir: Path
    abc: Path
    circuits: tuple[Path, ...]
    recipes: RecipePlan
    collection: CollectionPlan
    training: TrainingPlan
    evaluation: EvaluationPlan
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "name": self.name,
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "abc": str(self.abc),
            "circuits": [str(path) for path in self.circuits],
            "recipes": asdict(self.recipes),
            "collection": asdict(self.collection),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "fingerprint": self.fingerprint,
        }


def load_experiment_config(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    jobs: int | None = None,
) -> ExperimentConfig:
    path = Path(source).expanduser().resolve()
    payload = _read_json(path)
    schema_version = int(payload.get("schema_version", 1))
    if schema_version != RUNNER_SCHEMA_VERSION:
        raise ValueError(f"unsupported experiment schema version: {schema_version}")

    name = str(payload.get("name", path.stem)).strip()
    if not name:
        raise ValueError("experiment name cannot be empty")
    project_root_raw = Path(str(payload.get("project_root", ".."))).expanduser()
    project_root = (
        project_root_raw
        if project_root_raw.is_absolute()
        else path.parent / project_root_raw
    ).resolve()

    configured_output = Path(str(payload.get("output_dir", f"artifacts/experiments/{name}")))
    if output_dir is not None:
        configured_output = Path(output_dir).expanduser()
        resolved_output = (
            configured_output
            if configured_output.is_absolute()
            else Path.cwd() / configured_output
        ).resolve()
    else:
        resolved_output = (
            configured_output
            if configured_output.is_absolute()
            else project_root / configured_output
        ).resolve()

    abc_raw = Path(str(payload.get("abc", ".tools/abc"))).expanduser()
    abc = (abc_raw if abc_raw.is_absolute() else project_root / abc_raw).resolve()
    if not abc.is_file():
        raise FileNotFoundError(f"ABC binary not found: {abc}")
    circuit_values = payload.get("circuits")
    if not isinstance(circuit_values, list) or not circuit_values:
        raise ValueError("circuits must be a non-empty JSON array")
    circuit_inputs = []
    for value in circuit_values:
        circuit_path = Path(str(value)).expanduser()
        circuit_inputs.append(
            circuit_path if circuit_path.is_absolute() else project_root / circuit_path
        )
    circuits = tuple(iter_circuit_files(circuit_inputs))
    if not circuits:
        raise ValueError("no supported circuit files were found")
    if len(circuits) < 3:
        raise ValueError("at least three circuits are required for LOCO training")
    circuit_ids = [circuit.stem for circuit in circuits]
    if len(circuit_ids) != len(set(circuit_ids)):
        raise ValueError("circuit file stems must be unique")

    recipe_payload = _mapping(payload, "recipes")
    recipe_plan = RecipePlan(
        count=_positive_int(recipe_payload.get("count", 500), "recipes.count"),
        length=_positive_int(recipe_payload.get("length", 10), "recipes.length"),
        seeds=_number_tuple(recipe_payload.get("seeds", [0]), "recipes.seeds", positive=False),
        max_consecutive=_positive_int(
            recipe_payload.get("max_consecutive", 2), "recipes.max_consecutive"
        ),
    )

    collection_payload = _mapping(payload, "collection")
    collection_plan = CollectionPlan(
        jobs=_positive_int(
            jobs if jobs is not None else collection_payload.get("jobs", 1),
            "collection.jobs",
        ),
        timeout_s=float(collection_payload.get("timeout_s", 120.0)),
    )
    if collection_plan.timeout_s <= 0:
        raise ValueError("collection.timeout_s must be positive")

    training_payload = _mapping(payload, "training")
    training_plan = TrainingPlan(
        alpha=float(training_payload.get("alpha", 0.01)),
        trees=_positive_int(training_payload.get("trees", 200), "training.trees"),
        min_samples_leaf=_positive_int(
            training_payload.get("min_samples_leaf", 2),
            "training.min_samples_leaf",
        ),
        calibration_fraction=float(
            training_payload.get("calibration_fraction", 0.25)
        ),
        model_jobs=int(training_payload.get("model_jobs", 1)),
    )
    if not 0 < training_plan.alpha < 1:
        raise ValueError("training.alpha must be between zero and one")
    if training_plan.trees < 10:
        raise ValueError("training.trees must be at least 10")
    if not 0 < training_plan.calibration_fraction < 1:
        raise ValueError("training.calibration_fraction must be between zero and one")
    if training_plan.model_jobs == 0:
        raise ValueError("training.model_jobs cannot be zero")

    evaluation_payload = _mapping(payload, "evaluation")
    evaluation_plan = EvaluationPlan(
        budgets=_number_tuple(
            evaluation_payload.get("budgets", [10, 20, 50]),
            "evaluation.budgets",
            positive=True,
        ),
        search_seeds=_number_tuple(
            evaluation_payload.get("search_seeds", [0]),
            "evaluation.search_seeds",
            positive=False,
        ),
        min_stop_step=_positive_int(
            evaluation_payload.get("min_stop_step", 2),
            "evaluation.min_stop_step",
        ),
    )
    if max(evaluation_plan.budgets) > recipe_plan.count:
        raise ValueError("evaluation budgets cannot exceed recipes.count")

    fingerprint_payload = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "name": name,
        "abc": str(abc),
        "circuits": [str(item) for item in circuits],
        "recipes": asdict(recipe_plan),
        "collection": asdict(collection_plan),
        "training": asdict(training_plan),
        "evaluation": asdict(evaluation_plan),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ExperimentConfig(
        source=path,
        project_root=project_root,
        name=name,
        output_dir=resolved_output,
        abc=abc,
        circuits=circuits,
        recipes=recipe_plan,
        collection=collection_plan,
        training=training_plan,
        evaluation=evaluation_plan,
        fingerprint=fingerprint,
    )


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty(path: Path) -> bool | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _file_signature(path: Path) -> dict[str, int]:
    stats = path.stat()
    return {"size": stats.st_size, "mtime_ns": stats.st_mtime_ns}


def _signature_matches(
    payload: Mapping[str, Any], key: str, path: Path
) -> bool:
    try:
        return payload.get(key) == _file_signature(path)
    except OSError:
        return False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("riskaware-eda", "numpy", "scipy", "scikit-learn", "joblib"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _active_legacy_runner_pids(config_source: Path) -> list[int]:
    """Find pre-lock runner processes using the same config on Linux/WSL."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    active: list[int] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == os.getpid():
            continue
        try:
            arguments = [
                item.decode(errors="replace")
                for item in (process_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if (
                "experiment" not in arguments
                or "--config" not in arguments
                or not any("riskaware-eda" in item for item in arguments)
            ):
                continue
            raw_config = Path(arguments[arguments.index("--config") + 1])
            if not raw_config.is_absolute():
                raw_config = Path(os.readlink(process_dir / "cwd")) / raw_config
            if raw_config.resolve() == config_source.resolve():
                active.append(pid)
        except (IndexError, OSError, UnicodeError):
            continue
    return sorted(active)


@dataclass(frozen=True)
class _RecoveredCSV:
    rows: tuple[dict[str, str], ...]
    recipe_ids: tuple[str, ...]


def _valid_recipe_rows(
    rows: Sequence[Mapping[str, str]],
    recipe: Recipe,
    circuit_id: str,
) -> bool:
    expected_steps = list(range(len(recipe.operations) + 1))
    try:
        steps = [int(row["step"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(rows) == len(expected_steps)
        and steps == expected_steps
        and all(
            set(row) == set(DATASET_FIELDS)
            and all(value is not None for value in row.values())
            for row in rows
        )
        and all(row.get("circuit_id") == circuit_id for row in rows)
        and all(row.get("recipe_id") == recipe.recipe_id for row in rows)
        and all(
            row.get("recipe") == "|".join(recipe.operations) for row in rows
        )
        and all(
            row.get("recipe_length") == str(len(recipe.operations)) for row in rows
        )
        and all(row.get("completed") == "1" for row in rows)
        and all(not row.get("error") for row in rows)
        and rows[0].get("action") == "__start__"
        and [row.get("action") for row in rows[1:]] == list(recipe.operations)
        and all(row.get("final_qor") not in {None, ""} for row in rows)
    )


def _recover_csv_prefix(
    source: Path,
    recipes: Sequence[Recipe],
    circuit_id: str,
) -> _RecoveredCSV:
    """Return the longest valid, ordered trajectory prefix in a partial shard."""
    if not source.is_file() or not recipes:
        return _RecoveredCSV((), ())
    recovered_rows: list[dict[str, str]] = []
    recovered_ids: list[str] = []
    recipe_index = 0
    current_rows: list[dict[str, str]] = []

    def accept_current() -> bool:
        nonlocal recipe_index, current_rows
        if not current_rows or recipe_index >= len(recipes):
            return False
        recipe = recipes[recipe_index]
        if not _valid_recipe_rows(current_rows, recipe, circuit_id):
            return False
        recovered_rows.extend(current_rows)
        recovered_ids.append(recipe.recipe_id)
        recipe_index += 1
        current_rows = []
        return True

    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != DATASET_FIELDS:
                return _RecoveredCSV((), ())
            for row in reader:
                if recipe_index >= len(recipes):
                    break
                expected_id = recipes[recipe_index].recipe_id
                recipe_id = row.get("recipe_id")
                if not current_rows:
                    if recipe_id != expected_id:
                        break
                    current_rows.append(row)
                elif recipe_id == expected_id:
                    current_rows.append(row)
                else:
                    if not accept_current():
                        break
                    if recipe_index >= len(recipes):
                        break
                    if recipe_id != recipes[recipe_index].recipe_id:
                        break
                    current_rows.append(row)
            if current_rows:
                accept_current()
    except (OSError, UnicodeDecodeError, csv.Error):
        if current_rows:
            accept_current()
    return _RecoveredCSV(tuple(recovered_rows), tuple(recovered_ids))


class _ExperimentLock:
    """Process-scoped advisory lock for one experiment output directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "_ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner details unavailable"
            handle.close()
            raise RuntimeError(
                f"another experiment runner is active for {self.path.parent}: {owner}"
            ) from exc
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": _utc_now(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        return False


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        resume: bool = False,
        runner_factory: Callable[..., Any] = ABCRunner,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.resume = resume
        self.runner_factory = runner_factory
        self.progress = progress or (lambda _: None)
        self.manifest_path = config.output_dir / "manifest.json"
        self.lock_path = config.output_dir / "run.lock"

    def plan(self) -> dict[str, Any]:
        recipe_seed_count = len(self.config.recipes.seeds)
        circuit_count = len(self.config.circuits)
        collection_tasks = recipe_seed_count * circuit_count
        evaluation_tasks = (
            collection_tasks
            * len(self.config.evaluation.budgets)
            * len(self.config.evaluation.search_seeds)
        )
        return {
            "name": self.config.name,
            "output_dir": str(self.config.output_dir),
            "fingerprint": self.config.fingerprint,
            "circuits": circuit_count,
            "recipe_seeds": recipe_seed_count,
            "recipes_per_seed": self.config.recipes.count,
            "recipe_length": self.config.recipes.length,
            "collection_tasks": collection_tasks,
            "estimated_operator_steps": (
                collection_tasks
                * self.config.recipes.count
                * self.config.recipes.length
            ),
            "training_tasks": collection_tasks,
            "evaluation_tasks": evaluation_tasks,
            "collection_jobs": self.config.collection.jobs,
            "model_jobs": self.config.training.model_jobs,
        }

    def _manifest(self, *, status: str) -> dict[str, Any]:
        return {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "fingerprint": self.config.fingerprint,
            "status": status,
            "runner_pid": os.getpid(),
            "updated_at": _utc_now(),
            "config_source": str(self.config.source),
            "config": self.config.to_dict(),
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
                "project_git_commit": _git_revision(self.config.project_root),
                "project_git_dirty": _git_dirty(self.config.project_root),
                "abc_git_commit": _git_revision(
                    self.config.project_root / ".tools" / "abc-src"
                ),
                "abc_git_dirty": _git_dirty(
                    self.config.project_root / ".tools" / "abc-src"
                ),
                "benchmark_git_commit": _git_revision(
                    self.config.project_root / "data" / "epfl"
                ),
                "benchmark_git_dirty": _git_dirty(
                    self.config.project_root / "data" / "epfl"
                ),
            },
            "plan": self.plan(),
        }

    def _prepare_output(self) -> None:
        output = self.config.output_dir
        if self.manifest_path.is_file():
            manifest = _read_json(self.manifest_path)
            if manifest.get("fingerprint") != self.config.fingerprint:
                raise ValueError(
                    "output directory belongs to a different experiment configuration"
                )
            if not self.resume:
                raise ValueError("output already exists; pass --resume to continue it")
        elif output.exists() and any(
            item != self.lock_path for item in output.iterdir()
        ):
            raise ValueError("non-empty output directory has no compatible manifest")
        else:
            output.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(status="running")
        if self.manifest_path.is_file():
            previous = _read_json(self.manifest_path)
            manifest["created_at"] = previous.get("created_at", _utc_now())
        else:
            manifest["created_at"] = _utc_now()
        _atomic_json(manifest, self.manifest_path)

    def _legacy_runner_pids(self) -> list[int]:
        if not self.manifest_path.is_file():
            return []
        try:
            manifest = _read_json(self.manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        if manifest.get("status") != "running" or manifest.get("runner_pid"):
            return []
        return _active_legacy_runner_pids(self.config.source)

    def _finish_manifest(self, status: str, error: str | None = None) -> None:
        manifest = _read_json(self.manifest_path)
        manifest["status"] = status
        manifest["updated_at"] = _utc_now()
        if error is not None:
            manifest["error"] = error
        else:
            manifest.pop("error", None)
        _atomic_json(manifest, self.manifest_path)

    @staticmethod
    def _seed_tag(seed: int) -> str:
        return f"seed_{seed:05d}"

    def _recipe_path(self, seed: int) -> Path:
        return self.config.output_dir / "recipes" / f"{self._seed_tag(seed)}.json"

    def _shard_path(self, seed: int, circuit: Path) -> Path:
        return (
            self.config.output_dir
            / "shards"
            / self._seed_tag(seed)
            / f"{circuit.stem}.csv"
        )

    def _collection_checkpoint(self, seed: int, circuit: Path) -> Path:
        return (
            self.config.output_dir
            / "checkpoints"
            / "collect"
            / self._seed_tag(seed)
            / f"{circuit.stem}.done.json"
        )

    def _collection_failure(self, seed: int, circuit: Path) -> Path:
        return self._collection_checkpoint(seed, circuit).with_name(
            f"{circuit.stem}.failed.json"
        )

    def _collection_progress(self, seed: int, circuit: Path) -> Path:
        return self._collection_checkpoint(seed, circuit).with_name(
            f"{circuit.stem}.progress.json"
        )

    def _partial_shard_path(self, seed: int, circuit: Path) -> Path:
        return self._shard_path(seed, circuit).with_suffix(".csv.partial")

    def _legacy_partial_shard_path(self, seed: int, circuit: Path) -> Path:
        return self._shard_path(seed, circuit).with_suffix(".csv.tmp")

    def _collection_dependencies(self, seed: int, circuit: Path) -> dict[str, Any]:
        return {
            "fingerprint": self.config.fingerprint,
            "recipe_seed": seed,
            "circuit_id": circuit.stem,
            "circuit_path": str(circuit),
            "circuit_signature": _file_signature(circuit),
            "recipe_signature": _file_signature(self._recipe_path(seed)),
            "abc_signature": _file_signature(self.config.abc),
        }

    @staticmethod
    def _progress_matches_dependencies(
        payload: Mapping[str, Any], dependencies: Mapping[str, Any]
    ) -> bool:
        return all(payload.get(key) == value for key, value in dependencies.items())

    def _write_collection_progress(
        self,
        seed: int,
        circuit: Path,
        partial: Path,
        *,
        completed: int,
        rows: int,
        total: int,
        dependencies: Mapping[str, Any],
        recovery_source: str | None,
    ) -> None:
        _atomic_json(
            {
                "progress_schema_version": COLLECTION_PROGRESS_SCHEMA_VERSION,
                **dependencies,
                "partial": str(partial),
                "partial_signature": _file_signature(partial),
                "completed_recipes": completed,
                "total_recipes": total,
                "rows": rows,
                "last_recipe_id": (
                    None if completed == 0 else f"r{completed - 1:05d}"
                ),
                "recovery_source": recovery_source,
                "runner_pid": os.getpid(),
                "heartbeat_at": _utc_now(),
            },
            self._collection_progress(seed, circuit),
        )

    def _prepare_partial_shard(
        self,
        seed: int,
        circuit: Path,
        recipes: Sequence[Recipe],
        dependencies: Mapping[str, Any],
    ) -> tuple[Path, _RecoveredCSV, str | None]:
        partial = self._partial_shard_path(seed, circuit)
        legacy = self._legacy_partial_shard_path(seed, circuit)
        shard = self._shard_path(seed, circuit)
        checkpoint = self._collection_checkpoint(seed, circuit)
        progress_path = self._collection_progress(seed, circuit)
        candidates: list[Path] = []
        progress_exists = progress_path.is_file()
        progress_compatible = False
        if progress_exists:
            try:
                progress_compatible = self._progress_matches_dependencies(
                    _read_json(progress_path), dependencies
                )
            except (OSError, ValueError, json.JSONDecodeError):
                progress_compatible = False

        if self.resume and (progress_compatible or not progress_exists):
            candidates.extend(path for path in (partial, legacy) if path.is_file())
            if shard.is_file() and (not checkpoint.is_file() or progress_compatible):
                candidates.append(shard)
        elif self.resume and progress_exists and not progress_compatible:
            self.progress(
                f"collection invalidated stale partial: seed={seed} "
                f"circuit={circuit.stem}"
            )

        recovered = _RecoveredCSV((), ())
        recovery_source: str | None = None
        for candidate in candidates:
            candidate_recovery = _recover_csv_prefix(
                candidate, recipes, circuit.stem
            )
            if len(candidate_recovery.recipe_ids) > len(recovered.recipe_ids):
                recovered = candidate_recovery
                recovery_source = str(candidate)

        partial.parent.mkdir(parents=True, exist_ok=True)
        temporary = partial.with_suffix(partial.suffix + ".recovering")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
            writer.writeheader()
            writer.writerows(recovered.rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(partial)
        self._write_collection_progress(
            seed,
            circuit,
            partial,
            completed=len(recovered.recipe_ids),
            rows=len(recovered.rows),
            total=len(recipes),
            dependencies=dependencies,
            recovery_source=recovery_source,
        )
        return partial, recovered, recovery_source

    def _dataset_path(self, seed: int) -> Path:
        return self.config.output_dir / "datasets" / f"{self._seed_tag(seed)}.csv"

    def _dataset_checkpoint(self, seed: int) -> Path:
        return (
            self.config.output_dir
            / "checkpoints"
            / "datasets"
            / f"{self._seed_tag(seed)}.done.json"
        )

    def _dataset_complete(self, seed: int) -> bool:
        destination = self._dataset_path(seed)
        checkpoint = self._dataset_checkpoint(seed)
        if not destination.is_file() or not checkpoint.is_file():
            return False
        try:
            payload = _read_json(checkpoint)
            source_signatures = self._source_signatures(seed)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            payload.get("fingerprint") == self.config.fingerprint
            and payload.get("sources") == source_signatures
            and _signature_matches(payload, "dataset_signature", destination)
            and all(
                self._collection_complete(seed, circuit)
                for circuit in self.config.circuits
            )
        )

    def _model_path(self, seed: int, circuit: Path) -> Path:
        return (
            self.config.output_dir
            / "models"
            / self._seed_tag(seed)
            / f"holdout_{circuit.stem}.joblib"
        )

    def _training_report_path(self, seed: int, circuit: Path) -> Path:
        return (
            self.config.output_dir
            / "reports"
            / "training"
            / self._seed_tag(seed)
            / f"holdout_{circuit.stem}.json"
        )

    def _simulation_path(
        self, seed: int, circuit: Path, budget: int, search_seed: int
    ) -> Path:
        return (
            self.config.output_dir
            / "simulations"
            / self._seed_tag(seed)
            / circuit.stem
            / f"budget_{budget:04d}_search_{search_seed:05d}.json"
        )

    def _ensure_recipes(self) -> dict[int, list[Recipe]]:
        by_seed: dict[int, list[Recipe]] = {}
        for seed in self.config.recipes.seeds:
            path = self._recipe_path(seed)
            if path.is_file():
                payload = _read_json(path)
                if payload.get("seed") != seed:
                    raise ValueError(f"recipe seed mismatch: {path}")
                recipes = load_recipes(path)
            else:
                recipes = generate_recipes(
                    self.config.recipes.count,
                    self.config.recipes.length,
                    seed=seed,
                    max_consecutive=self.config.recipes.max_consecutive,
                )
                save_recipes(recipes, path, seed=seed)
                self.progress(f"generated {len(recipes)} recipes for seed {seed}")
            if len(recipes) != self.config.recipes.count:
                raise ValueError(f"recipe count mismatch: {path}")
            if any(
                len(recipe.operations) != self.config.recipes.length
                for recipe in recipes
            ):
                raise ValueError(f"recipe length mismatch: {path}")
            by_seed[seed] = recipes
        return by_seed

    def _collection_complete(self, seed: int, circuit: Path) -> bool:
        checkpoint = self._collection_checkpoint(seed, circuit)
        shard = self._shard_path(seed, circuit)
        recipe_path = self._recipe_path(seed)
        if not checkpoint.is_file() or not shard.is_file():
            return False
        try:
            payload = _read_json(checkpoint)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        report = payload.get("report", {})
        return bool(
            payload.get("fingerprint") == self.config.fingerprint
            and payload.get("recipe_seed") == seed
            and payload.get("circuit_path") == str(circuit)
            and report.get("trajectories") == self.config.recipes.count
            and report.get("completed") == self.config.recipes.count
            and report.get("failed") == 0
            and _signature_matches(payload, "shard_signature", shard)
            and _signature_matches(payload, "circuit_signature", circuit)
            and _signature_matches(payload, "recipe_signature", recipe_path)
            and _signature_matches(payload, "abc_signature", self.config.abc)
        )

    def _collect_one(
        self, seed: int, circuit: Path, recipes: Sequence[Recipe]
    ) -> dict[str, Any]:
        shard = self._shard_path(seed, circuit)
        checkpoint = self._collection_checkpoint(seed, circuit)
        failure = self._collection_failure(seed, circuit)
        progress_path = self._collection_progress(seed, circuit)
        legacy_partial = self._legacy_partial_shard_path(seed, circuit)
        started = time.perf_counter()
        completed_count = 0
        row_count = 0
        try:
            dependencies = self._collection_dependencies(seed, circuit)
            partial, recovered, recovery_source = self._prepare_partial_shard(
                seed, circuit, recipes, dependencies
            )
            completed_count = len(recovered.recipe_ids)
            row_count = len(recovered.rows)
            if completed_count:
                self.progress(
                    f"collection resume: seed={seed} circuit={circuit.stem} "
                    f"{completed_count}/{len(recipes)} recipes"
                )

            remaining = recipes[completed_count:]
            if remaining:
                runner = self.runner_factory(
                    self.config.abc,
                    timeout_s=self.config.collection.timeout_s,
                    keep_workdir=False,
                )
                with runner.session(circuit) as session, partial.open(
                    "a", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
                    for recipe in remaining:
                        trajectory = session.run_recipe(recipe)
                        if (
                            not trajectory.completed
                            or trajectory.error is not None
                            or trajectory.circuit_id != circuit.stem
                            or trajectory.recipe_id != recipe.recipe_id
                            or trajectory.operations != recipe.operations
                            or len(trajectory.steps) != len(recipe.operations)
                        ):
                            detail = trajectory.error or "incomplete trajectory"
                            raise RuntimeError(
                                f"recipe {recipe.recipe_id} failed: {detail}"
                            )
                        rows = list(trajectory_rows(trajectory))
                        writer.writerows(rows)
                        handle.flush()
                        os.fsync(handle.fileno())
                        completed_count += 1
                        row_count += len(rows)
                        self._write_collection_progress(
                            seed,
                            circuit,
                            partial,
                            completed=completed_count,
                            rows=row_count,
                            total=len(recipes),
                            dependencies=dependencies,
                            recovery_source=recovery_source,
                        )
                        if (
                            completed_count % PROGRESS_LOG_INTERVAL == 0
                            or completed_count == len(recipes)
                        ):
                            self.progress(
                                f"collection progress: seed={seed} "
                                f"circuit={circuit.stem} "
                                f"{completed_count}/{len(recipes)} recipes"
                            )

            partial.replace(shard)
            report = {
                "trajectories": len(recipes),
                "completed": len(recipes),
                "failed": 0,
                "rows": row_count,
            }
            elapsed = time.perf_counter() - started
            metadata: dict[str, Any] = {
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                **dependencies,
                "shard": str(shard),
                "shard_signature": _file_signature(shard),
                "resumed_trajectories": len(recovered.recipe_ids),
                "new_trajectories": len(recipes) - len(recovered.recipe_ids),
                "recovery_source": recovery_source,
                "elapsed_wall_s": elapsed,
                "finished_at": _utc_now(),
                "report": report,
            }
            _atomic_json(metadata, checkpoint)
            progress_path.unlink(missing_ok=True)
            legacy_partial.unlink(missing_ok=True)
            failure.unlink(missing_ok=True)
            return {"ok": True, **metadata}
        except Exception as exc:
            metadata = {
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "fingerprint": self.config.fingerprint,
                "recipe_seed": seed,
                "circuit_id": circuit.stem,
                "circuit_path": str(circuit),
                "completed_recipes": completed_count,
                "total_recipes": len(recipes),
                "elapsed_wall_s": time.perf_counter() - started,
                "finished_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            _atomic_json(metadata, failure)
            return {"ok": False, **metadata}

    def _source_signatures(self, seed: int) -> list[dict[str, Any]]:
        signatures = []
        for circuit in self.config.circuits:
            shard = self._shard_path(seed, circuit)
            signatures.append(
                {
                    "path": str(shard),
                    **_file_signature(shard),
                }
            )
        return signatures

    def _merge_dataset(self, seed: int) -> dict[str, Any]:
        destination = self._dataset_path(seed)
        checkpoint = self._dataset_checkpoint(seed)
        source_signatures = self._source_signatures(seed)
        if self.resume and self._dataset_complete(seed):
            return {"skipped": True, **_read_json(checkpoint)}

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        expected_header: bytes | None = None
        with temporary.open("wb") as output:
            for circuit in self.config.circuits:
                shard = self._shard_path(seed, circuit)
                with shard.open("rb") as source:
                    header = source.readline()
                    if expected_header is None:
                        expected_header = header
                        output.write(header)
                    elif header != expected_header:
                        raise ValueError(f"CSV header mismatch: {shard}")
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(destination)
        rows = 0
        trajectories = completed = failed = 0
        for circuit in self.config.circuits:
            payload = _read_json(self._collection_checkpoint(seed, circuit))
            report = payload["report"]
            rows += int(report["rows"])
            trajectories += int(report["trajectories"])
            completed += int(report["completed"])
            failed += int(report["failed"])
        metadata = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "fingerprint": self.config.fingerprint,
            "recipe_seed": seed,
            "dataset": str(destination),
            "dataset_signature": _file_signature(destination),
            "sources": source_signatures,
            "report": {
                "rows": rows,
                "trajectories": trajectories,
                "completed": completed,
                "failed": failed,
            },
            "finished_at": _utc_now(),
        }
        _atomic_json(metadata, checkpoint)
        return {"skipped": False, **metadata}

    def collect(self, recipes_by_seed: Mapping[int, Sequence[Recipe]]) -> dict[str, Any]:
        pending: list[tuple[int, Path, Sequence[Recipe]]] = []
        skipped = 0
        for seed, recipes in recipes_by_seed.items():
            for circuit in self.config.circuits:
                if self.resume and self._collection_complete(seed, circuit):
                    skipped += 1
                else:
                    pending.append((seed, circuit, recipes))
        self.progress(
            f"collection: {len(pending)} pending shard(s), {skipped} resumed"
        )
        results: list[dict[str, Any]] = []
        if pending:
            workers = min(self.config.collection.jobs, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._collect_one, seed, circuit, recipes): (
                        seed,
                        circuit,
                    )
                    for seed, circuit, recipes in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "done" if result["ok"] else "failed"
                    self.progress(
                        f"collection {status}: seed={result['recipe_seed']} "
                        f"circuit={result['circuit_id']}"
                    )
        failures = [result for result in results if not result["ok"]]
        if failures:
            details = "; ".join(
                f"{item['circuit_id']}: {item.get('error', 'trajectory failure')}"
                for item in failures[:3]
            )
            raise RuntimeError(f"{len(failures)} collection shard(s) failed: {details}")

        datasets = []
        for seed in self.config.recipes.seeds:
            missing = [
                circuit.stem
                for circuit in self.config.circuits
                if not self._collection_complete(seed, circuit)
            ]
            if missing:
                raise RuntimeError(f"missing completed shards for seed {seed}: {missing}")
            merged = self._merge_dataset(seed)
            datasets.append(merged)
            self.progress(f"dataset ready: seed={seed}")
        return {
            "pending": len(pending),
            "completed": len(results),
            "skipped": skipped,
            "resumed_recipes": sum(
                int(result.get("resumed_trajectories", 0)) for result in results
            ),
            "new_recipes": sum(
                int(result.get("new_trajectories", 0)) for result in results
            ),
            "datasets": datasets,
        }

    def _training_complete(self, seed: int, circuit: Path) -> bool:
        report_path = self._training_report_path(seed, circuit)
        model_path = self._model_path(seed, circuit)
        dataset_path = self._dataset_path(seed)
        if not report_path.is_file() or not model_path.is_file() or not dataset_path.is_file():
            return False
        try:
            payload = _read_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            payload.get("fingerprint") == self.config.fingerprint
            and payload.get("recipe_seed") == seed
            and payload.get("holdout_circuit") == circuit.stem
            and payload.get("dataset_signature") == _file_signature(dataset_path)
            and _signature_matches(payload, "model_signature", model_path)
        )

    def train(self) -> dict[str, Any]:
        trained = skipped = 0
        elapsed_total = 0.0
        for seed in self.config.recipes.seeds:
            dataset = self._dataset_path(seed)
            if not self._dataset_complete(seed):
                raise RuntimeError(
                    f"merged dataset is missing or stale for seed={seed}; "
                    "run the collect phase first"
                )
            for circuit in self.config.circuits:
                if self.resume and self._training_complete(seed, circuit):
                    skipped += 1
                    continue
                model_path = self._model_path(seed, circuit)
                report_path = self._training_report_path(seed, circuit)
                failure_path = report_path.with_suffix(".failed.json")
                started = time.perf_counter()
                self.progress(
                    f"training: seed={seed} holdout={circuit.stem}"
                )
                try:
                    model, report = train_risk_model(
                        dataset,
                        excluded_circuits=[circuit.stem],
                        calibration_fraction=self.config.training.calibration_fraction,
                        alpha=self.config.training.alpha,
                        n_estimators=self.config.training.trees,
                        min_samples_leaf=self.config.training.min_samples_leaf,
                        seed=seed,
                        n_jobs=self.config.training.model_jobs,
                    )
                    model.save(model_path)
                    elapsed = time.perf_counter() - started
                    payload = {
                        "runner_schema_version": RUNNER_SCHEMA_VERSION,
                        "fingerprint": self.config.fingerprint,
                        "recipe_seed": seed,
                        "holdout_circuit": circuit.stem,
                        "dataset": str(dataset),
                        "dataset_signature": _file_signature(dataset),
                        "model": str(model_path),
                        "model_signature": _file_signature(model_path),
                        "elapsed_wall_s": elapsed,
                        "finished_at": _utc_now(),
                        "training_report": report.to_dict(),
                    }
                    _atomic_json(payload, report_path)
                    failure_path.unlink(missing_ok=True)
                    trained += 1
                    elapsed_total += elapsed
                except Exception as exc:
                    _atomic_json(
                        {
                            "runner_schema_version": RUNNER_SCHEMA_VERSION,
                            "fingerprint": self.config.fingerprint,
                            "recipe_seed": seed,
                            "holdout_circuit": circuit.stem,
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_at": _utc_now(),
                        },
                        failure_path,
                    )
                    raise
        return {
            "trained": trained,
            "skipped": skipped,
            "elapsed_wall_s": elapsed_total,
        }

    def _simulation_complete(
        self, seed: int, circuit: Path, budget: int, search_seed: int
    ) -> bool:
        path = self._simulation_path(seed, circuit, budget, search_seed)
        model_path = self._model_path(seed, circuit)
        dataset_path = self._dataset_path(seed)
        oracle_path = self._shard_path(seed, circuit)
        if not path.is_file() or not model_path.is_file() or not dataset_path.is_file():
            return False
        try:
            payload = _read_json(path)
            artifact_schema = int(
                payload.get("simulation_artifact_schema_version", 1)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            artifact_schema in {1, SIMULATION_ARTIFACT_SCHEMA_VERSION}
            and payload.get("fingerprint") == self.config.fingerprint
            and payload.get("recipe_seed") == seed
            and payload.get("holdout_circuit") == circuit.stem
            and payload.get("budget") == budget
            and payload.get("search_seed") == search_seed
            and payload.get("dataset_signature") == _file_signature(dataset_path)
            and payload.get("model_signature") == _file_signature(model_path)
            and (
                artifact_schema == 1
                or (
                    payload.get("oracle") == str(oracle_path)
                    and payload.get("oracle_signature")
                    == _file_signature(oracle_path)
                )
            )
        )

    def evaluate(self) -> dict[str, Any]:
        completed = skipped = 0
        for seed in self.config.recipes.seeds:
            dataset = self._dataset_path(seed)
            if not self._dataset_complete(seed):
                raise RuntimeError(
                    f"merged dataset is missing or stale for seed={seed}; "
                    "run the collect phase first"
                )
            for circuit in self.config.circuits:
                pending_by_search_seed = {
                    search_seed: tuple(
                        budget
                        for budget in self.config.evaluation.budgets
                        if not (
                            self.resume
                            and self._simulation_complete(
                                seed, circuit, budget, search_seed
                            )
                        )
                    )
                    for search_seed in self.config.evaluation.search_seeds
                }
                pending = sum(
                    len(budgets) for budgets in pending_by_search_seed.values()
                )
                skipped += (
                    len(self.config.evaluation.budgets)
                    * len(self.config.evaluation.search_seeds)
                    - pending
                )
                if not pending:
                    continue
                model_path = self._model_path(seed, circuit)
                report_path = self._training_report_path(seed, circuit)
                if not self._training_complete(seed, circuit):
                    raise RuntimeError(
                        f"training artifacts are missing or stale for seed={seed}, "
                        f"holdout={circuit.stem}; run the train phase first"
                    )
                model = RiskModel.load(model_path)
                oracle_path = self._shard_path(seed, circuit)
                oracle = load_oracle_trajectories(oracle_path, circuit.stem)
                simulator = OracleSimulator(oracle, model)
                training_payload = _read_json(report_path)["training_report"]
                dataset_signature = _file_signature(dataset)
                model_signature = _file_signature(model_path)
                oracle_signature = _file_signature(oracle_path)
                for search_seed, pending_budgets in pending_by_search_seed.items():
                    if not pending_budgets:
                        continue
                    self.progress(
                        f"simulation: seed={seed} holdout={circuit.stem} "
                        f"budgets={','.join(str(item) for item in pending_budgets)} "
                        f"search_seed={search_seed}"
                    )
                    results = simulator.simulate_budgets(
                        pending_budgets,
                        seed=search_seed,
                        min_steps_before_stopping=(
                            self.config.evaluation.min_stop_step
                        ),
                    )
                    for budget in pending_budgets:
                        destination = self._simulation_path(
                            seed, circuit, budget, search_seed
                        )
                        _atomic_json(
                            {
                                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                                "simulation_artifact_schema_version": (
                                    SIMULATION_ARTIFACT_SCHEMA_VERSION
                                ),
                                "fingerprint": self.config.fingerprint,
                                "recipe_seed": seed,
                                "holdout_circuit": circuit.stem,
                                "budget": budget,
                                "search_seed": search_seed,
                                "dataset_signature": dataset_signature,
                                "model_signature": model_signature,
                                "oracle": str(oracle_path),
                                "oracle_signature": oracle_signature,
                                "training_report": training_payload,
                                "result": results[budget].to_dict(),
                                "finished_at": _utc_now(),
                            },
                            destination,
                        )
                        completed += 1
                del simulator, model, oracle
        aggregate = self._aggregate_results()
        return {"completed": completed, "skipped": skipped, **aggregate}

    def _aggregate_results(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for seed in self.config.recipes.seeds:
            for circuit in self.config.circuits:
                for budget in self.config.evaluation.budgets:
                    for search_seed in self.config.evaluation.search_seeds:
                        path = self._simulation_path(
                            seed, circuit, budget, search_seed
                        )
                        if not path.is_file():
                            raise FileNotFoundError(f"simulation output not found: {path}")
                        payload = _read_json(path)
                        result = payload["result"]
                        search = result["search"]
                        random_baseline = result["random_baseline"]
                        training = payload["training_report"]
                        rows.append(
                            {
                                "recipe_seed": seed,
                                "holdout_circuit": circuit.stem,
                                "budget": budget,
                                "search_seed": search_seed,
                                "best_qor": search["best_qor"],
                                "oracle_best_qor": result["oracle_best_qor"],
                                "relative_gap_pct": result["relative_gap_pct"],
                                "total_runtime_s": search["total_runtime_s"],
                                "random_best_qor": random_baseline["best_qor"],
                                "random_runtime_s": random_baseline["total_runtime_s"],
                                "runtime_reduction_vs_random_pct": result[
                                    "runtime_reduction_vs_random_pct"
                                ],
                                "selected": search["selected"],
                                "completed_evaluations": search[
                                    "completed_evaluations"
                                ],
                                "early_stops": search["early_stops"],
                                "candidates_eliminated": search[
                                    "candidates_eliminated"
                                ],
                                "termination_reason": search["termination_reason"],
                                "row_coverage": training["row_coverage"],
                                "simultaneous_trajectory_coverage": training[
                                    "simultaneous_trajectory_coverage"
                                ],
                                "mean_interval_width": training[
                                    "mean_interval_width"
                                ],
                            }
                        )

        results_path = self.config.output_dir / "results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = results_path.with_suffix(results_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(results_path)

        summaries: dict[str, Any] = {}
        for budget in self.config.evaluation.budgets:
            selected = [row for row in rows if row["budget"] == budget]
            gaps = [
                float(row["relative_gap_pct"])
                for row in selected
                if row["relative_gap_pct"] is not None
            ]
            total_selected = sum(int(row["selected"]) for row in selected)
            summaries[str(budget)] = {
                "runs": len(selected),
                "mean_relative_gap_pct": statistics.fmean(gaps) if gaps else None,
                "mean_runtime_reduction_vs_random_pct": statistics.fmean(
                    float(row["runtime_reduction_vs_random_pct"])
                    for row in selected
                ),
                "early_stop_rate": (
                    sum(int(row["early_stops"]) for row in selected)
                    / max(total_selected, 1)
                ),
                "mean_simultaneous_trajectory_coverage": statistics.fmean(
                    float(row["simultaneous_trajectory_coverage"])
                    for row in selected
                ),
            }
        summary = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "fingerprint": self.config.fingerprint,
            "runs": len(rows),
            "results": str(results_path),
            "by_budget": summaries,
            "finished_at": _utc_now(),
        }
        summary_path = self.config.output_dir / "summary.json"
        _atomic_json(summary, summary_path)
        return {"results": str(results_path), "summary": str(summary_path)}

    def run(self, *, phase: str = "all", dry_run: bool = False) -> dict[str, Any]:
        if phase not in {"collect", "train", "evaluate", "all"}:
            raise ValueError(f"unknown experiment phase: {phase}")
        if dry_run:
            return {"dry_run": True, "phase": phase, "plan": self.plan()}

        legacy_pids = self._legacy_runner_pids()
        if legacy_pids:
            raise RuntimeError(
                "a legacy experiment runner is still active for this config "
                f"(PID(s): {', '.join(str(pid) for pid in legacy_pids)}); "
                "wait for it to finish or stop it before resuming"
            )

        with _ExperimentLock(self.lock_path):
            return self._run_locked(phase)

    def _run_locked(self, phase: str) -> dict[str, Any]:
        self._prepare_output()
        run_started = time.perf_counter()
        payload: dict[str, Any] = {
            "dry_run": False,
            "phase": phase,
            "plan": self.plan(),
            "timings": {},
        }
        try:
            recipes_by_seed = self._ensure_recipes()
            if phase in {"collect", "all"}:
                phase_started = time.perf_counter()
                payload["collection"] = self.collect(recipes_by_seed)
                payload["timings"]["collection_wall_s"] = (
                    time.perf_counter() - phase_started
                )
            if phase in {"train", "all"}:
                phase_started = time.perf_counter()
                payload["training"] = self.train()
                payload["timings"]["training_wall_s"] = (
                    time.perf_counter() - phase_started
                )
            if phase in {"evaluate", "all"}:
                phase_started = time.perf_counter()
                payload["evaluation"] = self.evaluate()
                payload["timings"]["evaluation_wall_s"] = (
                    time.perf_counter() - phase_started
                )
            payload["timings"]["total_wall_s"] = time.perf_counter() - run_started
            _atomic_json(payload, self.config.output_dir / "last_run.json")
            self._finish_manifest("complete")
            return payload
        except BaseException as exc:
            self._finish_manifest("failed", f"{type(exc).__name__}: {exc}")
            raise
