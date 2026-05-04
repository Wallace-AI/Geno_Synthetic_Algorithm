import tempfile
from pathlib import Path

import pandas as pd

from gsa.experiments.logging import RunRecord, RunLogger


def make_record(run_id="r1", seed=0) -> RunRecord:
    return RunRecord(
        run_id=run_id, git_commit="abc", config_hash="xyz", env_hash="e"*64,
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
