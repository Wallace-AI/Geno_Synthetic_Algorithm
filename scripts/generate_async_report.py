"""Generate the async-vs-sync sub-report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsa.analysis.statistics import wilcoxon_paired, vargha_delaney_a12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/raw/async/runs.parquet")
    parser.add_argument("--out-dir", type=str, default="results/reports/async")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    completed = df[df["status"] == "completed"].copy()

    # Median table
    pivot_fit = (
        completed.groupby(["benchmark", "dim", "evaluation_budget", "algorithm"])
        ["final_best_true"].median().unstack()
    )
    pivot_wall = (
        completed.groupby(["benchmark", "algorithm"])
        ["wall_clock_seconds"].median().unstack()
    )

    # Per-pair Wilcoxon: ASYNC vs SYNC counterpart, paired by seed within
    # (benchmark, dim, budget).
    pairs = [("GSA_FULL_ENSEMBLE", "GSA_ASYNC"),
             ("GSA_DIRECT", "GSA_ASYNC_DIRECT")]
    rows = []
    for sync_algo, async_algo in pairs:
        for (bench, dim, bud), grp in completed.groupby(
                ["benchmark", "dim", "evaluation_budget"]):
            sync = grp[grp["algorithm"] == sync_algo].sort_values("seed_master")
            asyn = grp[grp["algorithm"] == async_algo].sort_values("seed_master")
            if len(sync) == 0 or len(asyn) == 0:
                continue
            sync_v = sync["final_best_true"].values
            asyn_v = asyn["final_best_true"].values
            try:
                _, p = wilcoxon_paired(sync_v, asyn_v)
            except Exception:
                p = 1.0
            a12 = vargha_delaney_a12(sync_v, asyn_v)
            rows.append({
                "benchmark": bench, "dim": dim, "budget": bud,
                "sync": sync_algo, "async": async_algo,
                "median_sync": float(np.median(sync_v)),
                "median_async": float(np.median(asyn_v)),
                "p_value": p, "A12_sync_vs_async": a12,
            })
    test_df = pd.DataFrame(rows)
    test_df.to_csv(out_dir / "wilcoxon_async_vs_sync.csv", index=False)

    # Plot: per-cell median bars, sync vs async paired
    fig, axes = plt.subplots(1, len(test_df.groupby(["benchmark", "dim", "budget"])),
                              figsize=(max(8, 3 * len(test_df) // 2), 4),
                              squeeze=False)
    cells = list(test_df.groupby(["benchmark", "dim", "budget"]))
    for ax, ((bench, dim, bud), grp) in zip(axes[0], cells):
        labels = [f"{r['sync']}\nvs\n{r['async']}" for _, r in grp.iterrows()]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, grp["median_sync"].values, width=0.35,
               label="sync", color="#1f77b4")
        ax.bar(x + 0.18, grp["median_async"].values, width=0.35,
               label="async", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{bench} D={dim}\nbudget={bud}", fontsize=9)
        ax.set_ylabel("median final fitness")
        ax.legend(fontsize=8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "async_vs_sync_medians.png", dpi=120)
    plt.close(fig)

    # Markdown report
    out = ["# Asynchronous vs Synchronous GSA",
           "",
           f"- Input: `{args.input}`",
           f"- Total runs: {len(df)} ({len(completed)} completed)",
           "",
           "## Median final fitness (lower = better)",
           "",
           pivot_fit.to_markdown(),
           "",
           "## Median wall-clock seconds per run",
           "",
           pivot_wall.to_markdown(),
           "",
           "## Paired Wilcoxon: SYNC vs ASYNC (A12 > 0.5 means sync wins on a typical seed)",
           "",
           test_df.to_markdown(index=False),
           "",
           "## Figure",
           "",
           "- `figures/async_vs_sync_medians.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Async report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
