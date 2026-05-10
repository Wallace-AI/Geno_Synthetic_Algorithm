"""Generate paper-ready tables (markdown + LaTeX)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gsa.analysis.aggregate import per_cell_summary, rank_table
from gsa.analysis.statistics import pairwise_wilcoxon_holm


def _format_num(x: float) -> str:
    if pd.isna(x):
        return "-"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def median_iqr_table(df: pd.DataFrame, out_path_md: Path,
                     extra_group_cols: list[str] = None) -> pd.DataFrame:
    """Median ± IQR table per (algorithm, benchmark, dim) cell.

    Writes markdown to `out_path_md` and returns the underlying dataframe."""
    group_cols = ["benchmark", "dim"]
    if extra_group_cols:
        group_cols = group_cols + [c for c in extra_group_cols
                                    if c in df.columns and df[c].notna().any()]
    summary = per_cell_summary(df, group_cols + ["algorithm"])
    if summary.empty:
        out_path_md.parent.mkdir(parents=True, exist_ok=True)
        out_path_md.write_text("(no completed runs)\n")
        return summary

    summary["display"] = summary.apply(
        lambda r: f"{_format_num(r['median'])} ({_format_num(r['iqr'])})", axis=1
    )
    pivot = summary.pivot_table(index=group_cols, columns="algorithm",
                                 values="display", aggfunc="first")
    out_path_md.parent.mkdir(parents=True, exist_ok=True)
    out_path_md.write_text(
        "# Median (IQR) of final_best_true per cell\n\n"
        + pivot.to_markdown()
    )
    return summary


def stat_test_table(df: pd.DataFrame, base_algo: str, out_path_md: Path) -> pd.DataFrame:
    """Pairwise Wilcoxon + Holm + A12 vs. base algorithm."""
    res = pairwise_wilcoxon_holm(df, base_algo)
    out_path_md.parent.mkdir(parents=True, exist_ok=True)
    if res.empty:
        out_path_md.write_text("(no comparable pairs)\n")
        return res
    res_fmt = res.copy()
    res_fmt["p_raw"] = res_fmt["p_raw"].apply(lambda x: f"{x:.3e}")
    res_fmt["p_holm"] = res_fmt["p_holm"].apply(lambda x: f"{x:.3e}")
    res_fmt["A12"] = res_fmt["A12"].apply(lambda x: f"{x:.3f}")
    out_path_md.write_text(
        f"# Pairwise comparison vs. {base_algo}\n\n"
        f"A12 > 0.5 means {base_algo} wins on a typical seed.\n\n"
        + res_fmt.to_markdown(index=False)
    )
    return res


def rank_summary_table(df: pd.DataFrame, out_path_md: Path) -> pd.DataFrame:
    rt = rank_table(df)
    out_path_md.parent.mkdir(parents=True, exist_ok=True)
    if rt.empty:
        out_path_md.write_text("(no completed runs)\n")
        return rt
    out_path_md.write_text(
        "# Per-(benchmark, dim) ranks (1 = best)\n\n"
        + rt.pivot_table(index=["benchmark", "dim"], columns="algorithm",
                          values="rank", aggfunc="first").to_markdown()
    )
    return rt
