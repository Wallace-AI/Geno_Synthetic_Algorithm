import numpy as np
import pandas as pd

from gsa.analysis.aggregate import per_cell_summary, rank_table
from gsa.analysis.statistics import (
    wilcoxon_paired, mann_whitney_u, vargha_delaney_a12, holm_correct,
    pairwise_wilcoxon_holm,
)


def _fake_runs():
    rows = []
    rng = np.random.default_rng(0)
    for algo, mu in [("GSA", 0.1), ("EA", 0.5), ("RANDOM", 1.5)]:
        for seed in range(10):
            rows.append({
                "algorithm": algo, "benchmark": "typed_additive", "dim": 10,
                "seed_master": seed, "status": "completed",
                "final_best_true": float(mu + rng.normal(scale=0.05)),
                "target_hit": False,
            })
    return pd.DataFrame(rows)


def test_per_cell_summary_returns_one_row_per_cell():
    df = _fake_runs()
    summary = per_cell_summary(df)
    assert len(summary) == 3
    assert "median" in summary.columns
    assert "iqr" in summary.columns


def test_rank_table_orders_by_median():
    df = _fake_runs()
    rt = rank_table(df)
    gsa_row = rt[rt["algorithm"] == "GSA"].iloc[0]
    random_row = rt[rt["algorithm"] == "RANDOM"].iloc[0]
    assert gsa_row["rank"] == 1
    assert random_row["rank"] == 3


def test_wilcoxon_paired_detects_difference():
    a = np.array([0.1, 0.2, 0.15, 0.18, 0.12])
    b = np.array([0.5, 0.6, 0.55, 0.58, 0.52])
    _, p = wilcoxon_paired(a, b)
    assert p < 0.1


def test_mann_whitney_u_returns_p():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([1.0, 2.0, 3.0])
    _, p = mann_whitney_u(a, b)
    assert 0 <= p <= 1


def test_a12_extremes():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    assert vargha_delaney_a12(a, b) == 1.0
    assert vargha_delaney_a12(b, a) == 0.0


def test_holm_correct_increases_p_values():
    raw = [0.001, 0.01, 0.04]
    adj = holm_correct(raw)
    assert all(a >= r for a, r in zip(adj, raw))
    assert adj == sorted(adj)


def test_pairwise_wilcoxon_vs_base():
    df = _fake_runs()
    res = pairwise_wilcoxon_holm(df, "GSA")
    # GSA was significantly better than both EA and RANDOM
    assert len(res) == 2
    for _, row in res.iterrows():
        assert row["A12"] >= 0.5  # GSA dominates
