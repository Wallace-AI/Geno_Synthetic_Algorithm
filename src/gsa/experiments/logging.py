"""Three-tier logging: run records (Parquet), generation records (Parquet partitioned),
sparse improvement snapshots (JSONL with cap)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict


class RunRecord(BaseModel):
    """One row of the run-level ledger. Immutable after construction.

    Optional-typed fields (`rho`, `n_families`, `noise_mode`, `error_message`)
    are required-with-`None`-allowed: the caller must declare a value each time
    so that a forgotten field does not silently default away.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    git_commit: str
    config_hash: str
    env_hash: str
    hardware_fingerprint: str
    algorithm: str
    variant: str
    benchmark: str
    dim: int
    rho: float | None
    n_families: int | None
    noise_mode: str | None
    seed_master: int
    evaluation_budget: int
    final_best_observed: float
    final_best_true: float
    auc_convergence: float
    evaluations_to_target: float  # inf if never hit
    target_hit: bool
    total_evaluations: int
    total_invalid_offspring: int
    total_repairs: int
    wall_clock_seconds: float
    peak_memory_mb: float
    status: str  # completed | failed | timeout
    error_message: str | None


class RunLogger:
    """Buffered writer for run-level records.

    Appends to existing Parquet via read-modify-write (acceptable since this
    is run-level — at most ~13k rows for the full battery)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[RunRecord] = []

    def write(self, record: RunRecord) -> None:
        self._buffer.append(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        new_df = pd.DataFrame([r.model_dump() for r in self._buffer])
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(self.path, index=False)
        self._buffer.clear()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # If flush() raises while an exception is propagating, the original is
        # lost — acceptable for fail-loud research code (we want both signals).
        self.flush()


class GenerationRecord(BaseModel):
    """One row per generation per run. Immutable after construction."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    generation: int
    eval_count_at_gen: int
    best_so_far_observed: float
    best_so_far_true: float
    current_best_observed: float
    mean_fitness: float
    std_fitness: float
    diversity_Z: float | None
    diversity_R: float | None
    diversity_B: float | None
    diversity_C: float | None
    diversity_Cx: float | None
    diversity_E: float | None
    operator_success_Z: float | None
    operator_success_R: float | None
    operator_success_B: float | None
    operator_success_C: float | None
    operator_success_Cx: float | None
    operator_success_E: float | None
    n_invalid_in_gen: int
    n_repaired_in_gen: int


class GenerationLogger:
    """Per-generation Parquet logger, partitioned by (benchmark, algorithm).

    Path layout: `<root>/<benchmark>/<algorithm>.parquet`. Each shard grows
    independently as runs complete; reads can scope to a single shard.
    """

    def __init__(self, root: Path, benchmark: str, algorithm: str) -> None:
        self.path = Path(root) / benchmark / f"{algorithm}.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[GenerationRecord] = []

    def write(self, record: GenerationRecord) -> None:
        self._buffer.append(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        new_df = pd.DataFrame([r.model_dump() for r in self._buffer])
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(self.path, index=False)
        self._buffer.clear()

    def __enter__(self) -> "GenerationLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.flush()


class SnapshotLogger:
    """Sparse JSONL writer with reservoir sampling cap.

    Single-writer per file; `flush()` truncates and rewrites. Up to `cap_count`
    snapshots OR `cap_bytes` total — Vitter Algorithm R against `cap_count`,
    with the byte cap enforced as a hard ceiling at every accept site.

    Semantics:
      - A payload whose serialized size exceeds `cap_bytes` is dropped
        entirely (counted in `n_dropped_oversize`, NOT in `n_seen`) — the
        reservoir stays unbiased over events that could be retained.
      - Phase 1 (buffer below `cap_count` AND would still fit `cap_bytes`):
        append.
      - Phase 2 (buffer at `cap_count` OR Phase 1 byte-check failed):
        accept with probability `cap_count / n_seen`. If accepted, swap the
        new line for a uniformly chosen existing entry — but only if the
        post-swap buffer still satisfies `cap_bytes`. A swap that would
        violate the cap is silently treated as "don't swap" (Vitter R is
        unbiased under this; the trade-off is a small bias toward smaller
        payloads when bytes binds, bounded by the cap_count vs cap_bytes
        ratio).
    """

    def __init__(
        self,
        path: Path,
        cap_count: int = 50,
        cap_bytes: int = 10_000_000,
        rng_seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_count = cap_count
        self.cap_bytes = cap_bytes
        self._buffer: list[str] = []  # serialized JSON lines
        self._buffer_bytes = 0
        self.n_seen = 0
        self.n_dropped_oversize = 0
        self._rng = random.Random(rng_seed)

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload)
        line_size = len(line) + 1  # +1 for trailing newline at flush time

        # Hard guard: a single line bigger than cap_bytes can never fit;
        # drop without counting in n_seen so reservoir uniformity over
        # retainable events is preserved.
        if line_size > self.cap_bytes:
            self.n_dropped_oversize += 1
            return

        self.n_seen += 1

        # Phase 1: fill the reservoir while both caps allow.
        if (
            len(self._buffer) < self.cap_count
            and self._buffer_bytes + line_size <= self.cap_bytes
        ):
            self._buffer.append(line)
            self._buffer_bytes += line_size
            return

        # Phase 2: Vitter Algorithm R against cap_count.
        if self._rng.random() < (self.cap_count / self.n_seen):
            idx = self._rng.randint(0, len(self._buffer) - 1)
            old = self._buffer[idx]
            old_size = len(old) + 1
            # Only accept the swap if the byte cap would hold afterwards.
            if self._buffer_bytes - old_size + line_size <= self.cap_bytes:
                self._buffer[idx] = line
                self._buffer_bytes += line_size - old_size
            # else: silently drop — equivalent to Vitter R rolling "don't swap"
            # for this draw. Small bias toward smaller payloads, bounded.

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.path.open("w") as f:
            for line in self._buffer:
                f.write(line + "\n")

    def __enter__(self) -> "SnapshotLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.flush()
