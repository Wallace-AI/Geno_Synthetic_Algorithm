import json
import math

import pandas as pd
import pytest
from pydantic import ValidationError

from gsa.experiments.logging import (
    GenerationLogger,
    GenerationRecord,
    RunLogger,
    RunRecord,
    SnapshotLogger,
)


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


def make_gen(run_id: str = "r1", gen: int = 0) -> GenerationRecord:
    return GenerationRecord(
        run_id=run_id, generation=gen, eval_count_at_gen=gen * 50,
        best_so_far_observed=1.0 / (gen + 1),
        best_so_far_true=1.0 / (gen + 1),
        current_best_observed=1.0 / (gen + 1),
        mean_fitness=2.0, std_fitness=0.5,
        diversity_Z=0.1, diversity_R=0.2, diversity_B=0.3,
        diversity_C=0.4, diversity_Cx=0.5, diversity_E=0.6,
        operator_success_Z=0.5, operator_success_R=0.5, operator_success_B=0.5,
        operator_success_C=0.5, operator_success_Cx=0.5, operator_success_E=0.5,
        n_invalid_in_gen=0, n_repaired_in_gen=0,
    )


def test_generation_logger_partitions_by_benchmark_algorithm(tmp_path):
    log = GenerationLogger(tmp_path, benchmark="typed_additive",
                           algorithm="GSA_FULL_ENSEMBLE")
    for g in range(5):
        log.write(make_gen("r1", g))
    log.flush()
    expected = tmp_path / "typed_additive" / "GSA_FULL_ENSEMBLE.parquet"
    assert expected.exists()
    df = pd.read_parquet(expected)
    assert len(df) == 5


def test_generation_record_is_frozen():
    """Logged generation rows must be immutable, like RunRecord."""
    rec = make_gen("r1", 0)
    with pytest.raises(ValidationError):
        rec.generation = 99  # type: ignore[misc]


def test_snapshot_logger_caps_at_50(tmp_path):
    log = SnapshotLogger(tmp_path / "r1.jsonl", cap_count=50, cap_bytes=10_000_000)
    for i in range(100):
        log.write({"event": "improvement", "best": 1.0 / (i + 1), "i": i})
    log.flush()
    lines = (tmp_path / "r1.jsonl").read_text().strip().splitlines()
    # Reservoir sampling keeps exactly cap_count after the cap is hit
    assert len(lines) == 50
    parsed = [json.loads(line) for line in lines]
    indices = sorted(p["i"] for p in parsed)
    # The first 50 events fill the reservoir; afterwards, sampling is
    # probabilistic — assert size only, not specific indices
    assert all(0 <= i < 100 for i in indices)


def test_snapshot_logger_caps_at_bytes(tmp_path):
    log = SnapshotLogger(tmp_path / "r1.jsonl", cap_count=10_000, cap_bytes=200)
    big_payload = {"x": "y" * 100}
    for _ in range(50):
        log.write(big_payload)
    log.flush()
    size = (tmp_path / "r1.jsonl").stat().st_size
    assert size <= 250  # one line of slack


def test_snapshot_logger_byte_cap_holds_with_heterogeneous_sizes(tmp_path):
    """Mixed payload sizes must never push the file above cap_bytes.

    The prior implementation only checked the byte cap on Phase-1 fills, so
    a Phase-2 swap could replace a small entry with a larger one and exceed
    the cap. With explicit per-swap byte enforcement, the file size is
    bounded at cap_bytes regardless of write order.
    """
    path = tmp_path / "r1.jsonl"
    log = SnapshotLogger(path, cap_count=10, cap_bytes=300, rng_seed=0)
    # Alternate small (~25-byte) and large (~210-byte) payloads.
    for i in range(200):
        if i % 2 == 0:
            log.write({"i": i})
        else:
            log.write({"i": i, "blob": "z" * 200})
    log.flush()
    size = path.stat().st_size
    assert size <= 300, f"file size {size} exceeded cap_bytes=300"


def test_snapshot_logger_drops_oversized_single_payload(tmp_path):
    """A single payload larger than cap_bytes is dropped entirely; it is
    not entered in n_seen so reservoir uniformity over retainable events
    is preserved."""
    path = tmp_path / "r1.jsonl"
    log = SnapshotLogger(path, cap_count=50, cap_bytes=100, rng_seed=0)
    log.write({"blob": "z" * 500})  # serialized > 100 bytes
    log.write({"i": 1})  # small, fits
    log.flush()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"i": 1}
    assert log.n_dropped_oversize == 1
    assert log.n_seen == 1  # the dropped event was NOT counted in n_seen
