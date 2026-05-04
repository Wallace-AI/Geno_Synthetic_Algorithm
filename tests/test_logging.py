import math

import pandas as pd
import pytest
from pydantic import ValidationError

from gsa.experiments.logging import RunLogger, RunRecord


def make_record(run_id: str = "r1", seed: int = 0) -> RunRecord:
    return RunRecord(
        run_id=run_id, git_commit="abc", config_hash="xyz", env_hash="e" * 64,
        hardware_fingerprint="hw", algorithm="GSA_FULL_ENSEMBLE", variant="full",
        benchmark="typed_additive", dim=20, rho=None, n_families=6,
        noise_mode=None, seed_master=seed, evaluation_budget=10000,
        final_best_observed=0.1, final_best_true=0.1, auc_convergence=0.5,
        evaluations_to_target=100, target_hit=True,
        total_evaluations=10000, total_invalid_offspring=0, total_repairs=0,
        wall_clock_seconds=1.5, peak_memory_mb=42.0,
        status="completed", error_message=None,
    )


def test_run_logger_writes_parquet(tmp_path):
    log = RunLogger(tmp_path / "runs.parquet")
    log.write(make_record("r1", 0))
    log.write(make_record("r2", 1))
    log.flush()
    df = pd.read_parquet(tmp_path / "runs.parquet")
    assert len(df) == 2
    assert set(df["run_id"]) == {"r1", "r2"}


def test_run_logger_appends_existing_file(tmp_path):
    log1 = RunLogger(tmp_path / "runs.parquet")
    log1.write(make_record("r1", 0))
    log1.flush()

    log2 = RunLogger(tmp_path / "runs.parquet")
    log2.write(make_record("r2", 1))
    log2.flush()

    df = pd.read_parquet(tmp_path / "runs.parquet")
    assert len(df) == 2


def test_run_logger_context_manager_auto_flushes(tmp_path):
    """Exiting the context manager must flush buffered records to disk."""
    path = tmp_path / "runs.parquet"
    with RunLogger(path) as log:
        log.write(make_record("r1", 0))
    df = pd.read_parquet(path)
    assert len(df) == 1


def test_run_logger_empty_flush_is_idempotent(tmp_path):
    """Flushing with no buffered records (and again after a real flush) must
    not error and must not create or truncate the Parquet file."""
    path = tmp_path / "runs.parquet"
    log = RunLogger(path)
    log.flush()  # empty buffer, file does not exist
    assert not path.exists()
    log.write(make_record("r1", 0))
    log.flush()
    log.flush()  # double-flush after real write
    df = pd.read_parquet(path)
    assert len(df) == 1


def test_run_logger_round_trips_inf_evaluations_to_target(tmp_path):
    """`evaluations_to_target` is `inf` when the target was never hit;
    Parquet must round-trip that value losslessly."""
    rec = make_record("r1", 0).model_copy(
        update={"evaluations_to_target": float("inf"), "target_hit": False}
    )
    log = RunLogger(tmp_path / "runs.parquet")
    log.write(rec)
    log.flush()
    df = pd.read_parquet(tmp_path / "runs.parquet")
    assert math.isinf(df["evaluations_to_target"].iloc[0])


def test_run_record_is_frozen():
    """A logged record is part of the audit trail and must not mutate."""
    rec = make_record("r1", 0)
    with pytest.raises(ValidationError):
        rec.run_id = "r2"  # type: ignore[misc]
