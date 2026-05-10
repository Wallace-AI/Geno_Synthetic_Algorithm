"""Generate the larger-budget sub-report.

Tests whether typed-operator advantage compounds at larger budgets.
"""
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
    parser.add_argument("--input", type=str,
                        default="results/raw/budgets/runs.parquet")
    parser.add_argument("--out-dir", type=str,
                        default="results/reports/budgets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    completed = df[df["status"] == "completed"].copy()

    # Median final fitness per (benchmark, budget, algorithm)
    pivot = (completed.groupby(
        ["benchmark", "evaluation_budget", "algorithm"])
        ["final_best_true"].median().unstack())

    # Per-cell rank, then average across cells
    rank = pivot.rank(axis=1, method="average")
    mean_rank = rank.mean(axis=0).sort_values()

    # Plot: median final fitness vs budget, one panel per benchmark
    benches = sorted(completed["benchmark"].unique())
    fig, axes = plt.subplots(1, len(benches),
                              figsize=(5 * len(benches), 4),
                              squeeze=False)
    for ax, bench in zip(axes[0], benches):
        sub = completed[completed["benchmark"] == bench]
        for algo, grp in sub.groupby("algorithm"):
            agg = (grp.groupby("evaluation_budget")
                   ["final_best_true"].median().sort_index())
            ax.plot(agg.index, agg.values, marker="o", label=algo)
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("median final fitness")
        ax.set_title(bench)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "budget_sweep.png", dpi=120)
    plt.close(fig)

    # Paired Wilcoxon at each budget: GSA_ELITE_CONTEXT vs FLATTENED_DE
    rows = []
    pairs = [
        ("GSA_ELITE_CONTEXT", "FLATTENED_DE"),
        ("GSA_ELITE_CONTEXT", "FLATTENED_EA"),
        ("GSA_FULL_ENSEMBLE", "FLATTENED_DE"),
        ("GSA_DIRECT", "FLATTENED_DE"),
    ]
    for a_algo, b_algo in pairs:
        for bud, grp in completed.groupby("evaluation_budget"):
            a_v = []
            b_v = []
            for bench in grp["benchmark"].unique():
                sub = grp[grp["benchmark"] == bench]
                for seed, sg in sub.groupby("seed_master"):
                    a_row = sg[sg["algorithm"] == a_algo]
                    b_row = sg[sg["algorithm"] == b_algo]
                    if len(a_row) == 0 or len(b_row) == 0:
                        continue
                    a_v.append(float(a_row["final_best_true"].iloc[0]))
                    b_v.append(float(b_row["final_best_true"].iloc[0]))
            if not a_v:
                continue
            try:
                _, p = wilcoxon_paired(np.array(a_v), np.array(b_v))
            except Exception:
                p = 1.0
            a12 = vargha_delaney_a12(np.array(a_v), np.array(b_v))
            rows.append({
                "budget": int(bud), "a": a_algo, "b": b_algo,
                "median_a": float(np.median(a_v)),
                "median_b": float(np.median(b_v)),
                "p_value": p, "A12_a_vs_b": a12,
            })
    test_df = pd.DataFrame(rows)
    test_df.to_csv(out_dir / "wilcoxon_by_budget.csv", index=False)

    out = ["# Larger-budget sub-report",
           "",
           "Hypothesis: at larger budgets, GSA's per-iteration ensemble cost "
           "amortises and typed operators should overtake flattened baselines.",
           "",
           f"- Input: `{args.input}`",
           f"- Total runs: {len(df)} ({len(completed)} completed)",
           "",
           "## Median final fitness per (benchmark, budget, algorithm)",
           "",
           pivot.to_markdown(),
           "",
           "## Mean rank across (benchmark, budget) cells",
           "",
           mean_rank.to_frame("mean_rank").to_markdown(),
           "",
           "## Paired Wilcoxon: A12 < 0.5 means A wins (lower fitness) "
           "on a typical pair",
           "",
           "Pairings are by (benchmark, seed) at each budget level.",
           "",
           test_df.to_markdown(index=False),
           "",
           "## Figure",
           "",
           "- `figures/budget_sweep.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Budget report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
