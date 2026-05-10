"""Non-parametric statistical tests: Wilcoxon, Mann-Whitney, Friedman, A12."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test. Returns (statistic, p_value).

    Falls back to (0, 1.0) when all paired differences are zero (degenerate)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b, zero_method="zsplit")
    return float(res.statistic), float(res.pvalue)


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def friedman(values_per_algo: dict[str, np.ndarray]) -> tuple[float, float]:
    """Friedman test across algorithms. Each value array must have the same length
    (paired by, e.g., per-seed final fitness on the same problem)."""
    arrs = list(values_per_algo.values())
    res = stats.friedmanchisquare(*arrs)
    return float(res.statistic), float(res.pvalue)


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """A12 effect size: P(a < b) + 0.5 * P(a == b). Lower a is better, so
    A12 > 0.5 means `a` tends to win."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return 0.5
    less = float(np.sum(a[:, None] < b[None, :]))
    equal = float(np.sum(a[:, None] == b[None, :]))
    return (less + 0.5 * equal) / (n_a * n_b)


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    out = [None] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adjusted = (n - rank) * p_values[i]
        running_max = max(running_max, adjusted)
        out[i] = min(1.0, running_max)
    return out


def pairwise_wilcoxon_holm(df: pd.DataFrame, base_algo: str, *,
                           value_col: str = "final_best_true",
                           group_cols: list[str] = None) -> pd.DataFrame:
    """Compare every algorithm against `base_algo` via paired Wilcoxon, with
    Holm correction across the family of comparisons.

    Pairing: by `group_cols` + seed_master; defaults to per-(benchmark, dim, seed).
    Returns long-format dataframe with columns: algorithm, p_raw, p_holm, A12.
    """
    if group_cols is None:
        group_cols = ["benchmark", "dim"]
    completed = df[df["status"] == "completed"].copy()
    base = completed[completed["algorithm"] == base_algo]
    if base.empty:
        return pd.DataFrame(columns=["algorithm", "p_raw", "p_holm", "A12"])

    others = [a for a in completed["algorithm"].unique() if a != base_algo]
    pairs = []
    for algo in others:
        b = completed[completed["algorithm"] == algo]
        merged = base.merge(b, on=group_cols + ["seed_master"],
                            suffixes=("_base", "_other"))
        if merged.empty:
            continue
        a_vals = merged[f"{value_col}_base"].values
        b_vals = merged[f"{value_col}_other"].values
        try:
            _, p = wilcoxon_paired(a_vals, b_vals)
        except ValueError:
            p = 1.0
        a12 = vargha_delaney_a12(a_vals, b_vals)
        pairs.append((algo, p, a12))
    if not pairs:
        return pd.DataFrame(columns=["algorithm", "p_raw", "p_holm", "A12"])
    df_out = pd.DataFrame(pairs, columns=["algorithm", "p_raw", "A12"])
    df_out["p_holm"] = holm_correct(df_out["p_raw"].tolist())
    return df_out[["algorithm", "p_raw", "p_holm", "A12"]]
