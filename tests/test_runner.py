import pandas as pd

from gsa.experiments.runner import RunSpec, run_one, run_many


def make_spec(seed=0, tmp=None):
    return RunSpec(
        algorithm="GSA_FULL_ENSEMBLE",
        benchmark="typed_additive",
        benchmark_kwargs={"dim": 8, "families": ("R", "B")},
        algorithm_kwargs={},
        seed=seed,
        budget=500,
        output_dir=str(tmp) if tmp else "results/raw/test",
    )


def test_run_one_produces_run_record(tmp_path):
    spec = make_spec(0, tmp_path)
    rec = run_one(spec)
    assert rec.status in ("completed", "failed")
    assert rec.algorithm == "GSA_FULL_ENSEMBLE"
    assert rec.seed_master == 0


def test_run_many_writes_parquet_with_all_seeds(tmp_path):
    specs = [make_spec(s, tmp_path) for s in range(3)]
    run_many(specs, parallel=False)
    df = pd.read_parquet(tmp_path / "runs.parquet")
    assert len(df) == 3


def test_run_one_handles_failure_gracefully(tmp_path):
    """If the algorithm raises, the row should be status=failed, not crash."""
    spec = make_spec(0, tmp_path)
    spec.benchmark = "no_such_benchmark"
    rec = run_one(spec)
    assert rec.status == "failed"
    assert rec.error_message
