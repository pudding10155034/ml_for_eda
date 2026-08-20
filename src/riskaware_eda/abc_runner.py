from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .recipes import OPERATOR_COMMANDS, Recipe
from .types import NetworkStats, StopCallback, Trajectory, TrajectoryStep


class ABCExecutionError(RuntimeError):
    """Raised when ABC fails or does not emit parseable statistics."""


_IO_RE = re.compile(r"i/o\s*=\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_PI_RE = re.compile(r"\bpi\s*=\s*(\d+)", re.IGNORECASE)
_PO_RE = re.compile(r"\bpo\s*=\s*(\d+)", re.IGNORECASE)
_LATCH_RE = re.compile(r"\b(?:lat|latch(?:es)?)\s*=\s*(\d+)", re.IGNORECASE)
_NODE_RES = (
    re.compile(r"\band\s*=\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bnd\s*=\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bnodes?\s*=\s*(\d+)", re.IGNORECASE),
)
_DEPTH_RES = (
    re.compile(r"\blev\s*=\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bdepth\s*=\s*(\d+)", re.IGNORECASE),
)


def _last_integer(line: str, patterns: tuple[re.Pattern[str], ...]) -> int | None:
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def parse_print_stats(output: str) -> NetworkStats:
    """Parse the last ABC ``print_stats`` line in a process output stream."""

    for line in reversed(output.splitlines()):
        io_match = _IO_RE.search(line)
        if io_match:
            pis, pos = int(io_match.group(1)), int(io_match.group(2))
        else:
            pi_match, po_match = _PI_RE.search(line), _PO_RE.search(line)
            if not pi_match or not po_match:
                continue
            pis, pos = int(pi_match.group(1)), int(po_match.group(1))
        nodes = _last_integer(line, _NODE_RES)
        depth = _last_integer(line, _DEPTH_RES)
        if nodes is None or depth is None:
            continue
        latch_match = _LATCH_RE.search(line)
        return NetworkStats(
            pis=pis,
            pos=pos,
            nodes=nodes,
            depth=depth,
            latches=int(latch_match.group(1)) if latch_match else 0,
        )
    tail = output[-1_500:].strip()
    raise ABCExecutionError(f"ABC did not emit parseable print_stats output:\n{tail}")


def _resolve_binary(binary: str | Path) -> str:
    candidate = str(binary)
    located = shutil.which(candidate)
    if located:
        return str(Path(located).resolve())
    path = Path(candidate).expanduser()
    if path.is_file():
        return str(path.resolve())
    raise FileNotFoundError(
        f"ABC binary '{candidate}' was not found. Run scripts/setup_abc.sh or pass --abc."
    )


class ABCSession:
    """One initialized circuit that can evaluate many recipes efficiently."""

    def __init__(
        self,
        binary: str,
        circuit: Path,
        *,
        timeout_s: float,
        keep_workdir: bool,
    ) -> None:
        self.binary = binary
        self.circuit = circuit
        self.timeout_s = timeout_s
        self.keep_workdir = keep_workdir
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.workdir: Path | None = None
        self.initial_stats: NetworkStats | None = None
        self.setup_runtime_s = 0.0

    def __enter__(self) -> "ABCSession":
        if not self.circuit.is_file():
            raise FileNotFoundError(f"circuit not found: {self.circuit}")
        if self.keep_workdir:
            self.workdir = Path(tempfile.mkdtemp(prefix="riskaware_abc_"))
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix="riskaware_abc_")
            self.workdir = Path(self._temporary.name)

        suffix = self.circuit.suffix.lower() or ".aig"
        local_input = self.workdir / f"input{suffix}"
        shutil.copy2(self.circuit, local_input)
        started = time.perf_counter()
        output = self._run(
            f"read {local_input.name}; strash; print_stats; write_aiger baseline.aig"
        )
        self.setup_runtime_s = time.perf_counter() - started
        self.initial_stats = parse_print_stats(output)
        baseline = self.workdir / "baseline.aig"
        if not baseline.is_file():
            raise ABCExecutionError("ABC initialization did not write baseline.aig")
        return self

    def __exit__(self, *_: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()

    def _run(self, command: str) -> str:
        assert self.workdir is not None
        try:
            completed = subprocess.run(
                [self.binary, "-c", command],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ABCExecutionError(
                f"ABC command exceeded the {self.timeout_s:g}s timeout"
            ) from exc
        output = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0:
            raise ABCExecutionError(
                f"ABC exited with status {completed.returncode}:\n{output[-1_500:]}"
            )
        lowered = output.lower()
        if "cannot open" in lowered or "unknown command" in lowered:
            raise ABCExecutionError(output[-1_500:])
        return output

    def run_recipe(
        self,
        recipe: Recipe,
        stop_callback: StopCallback | None = None,
    ) -> Trajectory:
        if self.workdir is None or self.initial_stats is None:
            raise RuntimeError("ABCSession must be used as a context manager")

        trajectory = Trajectory(
            circuit_id=self.circuit.stem,
            recipe_id=recipe.recipe_id,
            operations=recipe.operations,
            initial=self.initial_stats,
            setup_runtime_s=self.setup_runtime_s,
        )
        current = self.workdir / f"{recipe.recipe_id}_state_0.aig"
        shutil.copy2(self.workdir / "baseline.aig", current)
        cumulative = 0.0

        for index, operator in enumerate(recipe.operations, start=1):
            command = OPERATOR_COMMANDS[operator]
            next_state = self.workdir / f"{recipe.recipe_id}_state_{index}.aig"
            started = time.perf_counter()
            try:
                output = self._run(
                    f"read_aiger {current.name}; {command}; print_stats; "
                    f"write_aiger {next_state.name}"
                )
                stats = parse_print_stats(output)
            except ABCExecutionError as exc:
                trajectory.error = str(exc)
                trajectory.stop_reason = "abc_error"
                return trajectory
            runtime = time.perf_counter() - started
            cumulative += runtime
            trajectory.steps.append(
                TrajectoryStep(
                    step=index,
                    action=operator,
                    stats=stats,
                    runtime_s=runtime,
                    cumulative_runtime_s=cumulative,
                )
            )
            current = next_state
            if stop_callback is not None:
                reason = stop_callback(trajectory)
                if reason:
                    trajectory.stopped_early = True
                    trajectory.stop_reason = reason
                    return trajectory

        trajectory.completed = True
        return trajectory


class ABCRunner:
    def __init__(
        self,
        binary: str | Path = "abc",
        *,
        timeout_s: float = 120.0,
        keep_workdir: bool = False,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.binary = _resolve_binary(binary)
        self.timeout_s = timeout_s
        self.keep_workdir = keep_workdir

    def session(self, circuit: str | Path) -> ABCSession:
        return ABCSession(
            self.binary,
            Path(circuit).expanduser().resolve(),
            timeout_s=self.timeout_s,
            keep_workdir=self.keep_workdir,
        )


def iter_circuit_files(paths: list[str | Path]) -> Iterator[Path]:
    """Expand files/directories into supported circuit files."""

    supported = {".aig", ".aiger", ".blif", ".bench", ".v", ".verilog"}
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in supported:
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate
