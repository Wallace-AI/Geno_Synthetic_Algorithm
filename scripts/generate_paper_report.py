"""Generate paper-ready tables, plots, and markdown report from runs.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from gsa.analysis.aggregate import per_cell_summary, rank_table
from gsa.analysis.statistics import pairwise_wilcoxon_holm
from gsa.analysis.plots import (
    headline_rank_vs_n_families, per_benchmark_boxplots, epistasis_heatmap,
)
from gsa.analysis.tables import (
    median_iqr_table, stat_test_table, rank_summary_table,
)


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body.rstrip()}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/raw/paper/runs.parquet")
    parser.add_argument("--out-dir", type=str, default="results/reports/paper")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} run records")

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    median_iqr_table(df, tables_dir / "median_iqr_per_cell.md")
    median_iqr_table(df, tables_dir / "median_iqr_typed_epistatic.md",
                     extra_group_cols=["rho"])
    median_iqr_table(df, tables_dir / "median_iqr_typed_mix.md",
                     extra_group_cols=["n_families"])
    median_iqr_table(df, tables_dir / "median_iqr_typed_noisy.md",
                     extra_group_cols=["noise_mode"])
    rank_summary_table(df, tables_dir / "rank_summary.md")
    stat_test_table(df, base_algo="GSA_FULL_ENSEMBLE",
                     out_path_md=tables_dir / "wilcoxon_vs_gsa_full.md")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    headline_rank_vs_n_families(df, figures_dir / "headline_rank_vs_n_families.png")
    per_benchmark_boxplots(df, figures_dir)
    epistasis_heatmap(df, figures_dir / "epistasis_heatmap.png")

    # ------------------------------------------------------------------
    # Top-level markdown report
    # ------------------------------------------------------------------
    completed = df[df["status"] == "completed"].copy()
    failed = df[df["status"] == "failed"]

    failure_lines = []
    if not failed.empty:
        failure_lines.append("Expected failures (flattened baselines cannot encode Cx/E families):")
        f_summary = failed.groupby(["algorithm", "benchmark"]).size().reset_index(name="count")
        failure_lines.append(f_summary.to_markdown(index=False))

    median_per_algo = (
        completed.groupby(["benchmark", "algorithm"])["final_best_true"]
        .median()
        .unstack(fill_value=float("nan"))
    )
    overall_rank = (
        median_per_algo.rank(axis=1, method="min").mean(axis=0).sort_values()
    )

    report_lines = [
        "# GSA Experiment Report",
        "",
        f"- Input: `{in_path}`",
        f"- Total runs: {len(df)} ({len(completed)} completed, {len(failed)} failed)",
        f"- Total wall-clock: {df['wall_clock_seconds'].sum():.1f}s",
        "",
        _section("Summary: average rank per algorithm",
                 "Lower rank = better. Ranks computed per (benchmark, dim) on median final fitness, then averaged across cells.\n\n" +
                 overall_rank.to_frame("avg_rank").to_markdown()),
        _section("Median final fitness (lower = better)",
                 median_per_algo.to_markdown()),
        _section("Failures (architectural, not algorithm bugs)",
                 "\n".join(failure_lines) if failure_lines else "(none)"),
        _section("Tables",
                 "- `tables/median_iqr_per_cell.md` — main table\n"
                 "- `tables/median_iqr_typed_epistatic.md` — epistasis sweep\n"
                 "- `tables/median_iqr_typed_mix.md` — n_families sweep\n"
                 "- `tables/median_iqr_typed_noisy.md` — noise modes\n"
                 "- `tables/rank_summary.md`\n"
                 "- `tables/wilcoxon_vs_gsa_full.md`"),
        _section("Figures",
                 "- `figures/headline_rank_vs_n_families.png`\n"
                 "- `figures/epistasis_heatmap.png`\n"
                 "- `figures/box_<benchmark>_d<D>.png`"),
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines))
    print(f"Report written to {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
