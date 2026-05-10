"""Aggregate raw run records into per-cell summary statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def per_cell_summary(df: pd.DataFrame,
                     group_cols: list[str] = None) -> pd.DataFrame:
    """One row per (algorithm, benchmark, dim, [extras]) cell with median/IQR.

    `group_cols` defaults to ['algorithm', 'benchmark', 'dim'] but can be
    extended with ['rho'], ['n_families'], ['noise_mode'] when those vary.
    """
    if group_cols is None:
        group_cols = ["algorithm", "benchmark", "dim"]
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=group_cols + [
            "n_seeds", "median", "iqr", "mean", "std", "min", "max",
            "target_hit_rate"])

    grp = completed.groupby(group_cols)["final_best_true"]
    summary = grp.agg(
        n_seeds="count",
        median="median",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()
    summary["q25"] = grp.quantile(0.25).reset_index(drop=True)
    summary["q75"] = grp.quantile(0.75).reset_index(drop=True)
    summary["iqr"] = summary["q75"] - summary["q25"]
    target_grp = completed.groupby(group_cols)["target_hit"].mean()
    summary["target_hit_rate"] = target_grp.reset_index(drop=True)
    return summary


def rank_table(df: pd.DataFrame, group_cols: list[str] = None) -> pd.DataFrame:
    """Per-(benchmark, dim) ranking of algorithms by median final_best_true.

    Lower is better — rank 1 = best."""
    if group_cols is None:
        group_cols = ["benchmark", "dim"]
    summary = per_cell_summary(df, group_cols + ["algorithm"])
    if summary.empty:
        return summary
    summary["rank"] = summary.groupby(group_cols)["median"].rank(method="min")
    return summary[group_cols + ["algorithm", "median", "rank"]]
