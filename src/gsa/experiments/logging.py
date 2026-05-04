"""Three-tier logging: run records (Parquet), generation records (Parquet partitioned),
sparse improvement snapshots (JSONL with cap)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: str
    git_commit: str
    config_hash: str
    env_hash: str
    hardware_fingerprint: str
    algorithm: str
    variant: str
    benchmark: str
    dim: int
    rho: Optional[float]
    n_families: Optional[int]
    noise_mode: Optional[str]
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
    error_message: Optional[str]


class RunLogger:
    """Buffered writer for run-level records.

    Appends to existing Parquet via read-modify-write (acceptable since this
    is run-level — at most ~13k rows for the full battery)."""

    def __init__(self, path: Path):
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()
