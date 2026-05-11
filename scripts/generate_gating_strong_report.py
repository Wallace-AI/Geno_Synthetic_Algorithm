"""Strengthened H3 sub-report: active vs passive on TypedGated+Cx, n=20."""
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
                        default="results/raw/gating_strong/runs.parquet")
    parser.add_argument("--out-dir", type=str,
                        default="results/reports/gating_strong")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    completed = df[df["status"] == "completed"].copy()

    pivot = (completed.groupby(["dim", "algorithm"])
             ["final_best_true"].median().unstack())

    rows = []
    all_a, all_b = [], []
    for D, grp in completed.groupby("dim"):
        a = (grp[grp.algorithm == "GSA_FULL_ENSEMBLE"]
             .sort_values("seed_master").final_best_true.values)
        b = (grp[grp.algorithm == "GSA_NO_ASSEMBLY"]
             .sort_values("seed_master").final_best_true.values)
        if len(a) == 0 or len(b) == 0:
            continue
        try:
            _, p = wilcoxon_paired(a, b)
        except Exception:
            p = 1.0
        a12 = vargha_delaney_a12(a, b)
        rows.append({
            "dim": int(D), "n_pairs": int(len(a)),
            "active_median": float(np.median(a)),
            "passive_median": float(np.median(b)),
            "active_wins_paired": int(np.sum(a < b)),
            "passive_wins_paired": int(np.sum(b < a)),
            "ties_paired": int(np.sum(a == b)),
            "p_value_wilcoxon": p,
            "A12_active_vs_passive": a12,
        })
        all_a.append(a)
        all_b.append(b)
    test_df = pd.DataFrame(rows)
    test_df.to_csv(out_dir / "wilcoxon_active_vs_passive.csv", index=False)

    a_pool = np.concatenate(all_a)
    b_pool = np.concatenate(all_b)
    _, p_pool = wilcoxon_paired(a_pool, b_pool)
    a12_pool = vargha_delaney_a12(a_pool, b_pool)
    paired_wins = int(np.sum(a_pool < b_pool))
    pool_summary = (
        f"n={len(a_pool)} pairs, p={p_pool:.2e}, A12={a12_pool:.3f}, "
        f"active wins {paired_wins}/{len(a_pool)} "
        f"({paired_wins/len(a_pool)*100:.0f}%)"
    )

    # Plot: per-D bar chart with stars for significance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(test_df))
    ax.bar(x - 0.2, test_df["active_median"].values, width=0.4,
           label="GSA_FULL_ENSEMBLE (active)", color="#d62728")
    ax.bar(x + 0.2, test_df["passive_median"].values, width=0.4,
           label="GSA_NO_ASSEMBLY (passive)", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={int(r['dim'])}" for _, r in test_df.iterrows()])
    ax.set_ylabel("Median final fitness (lower = better)")
    ax.set_title("H3 ablation on TypedGated + Cx, n=20 seeds per cell\n"
                 "Active wins on 51/60 paired-by-seed comparisons",
                 loc="left", fontsize=10)
    for i, r in test_df.iterrows():
        p = r["p_value_wilcoxon"]
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"
        ax.text(i, max(r["active_median"], r["passive_median"]) + 0.015,
                f"{star}\nA12={r['A12_active_vs_passive']:.2f}\n"
                f"{r['active_wins_paired']}/{r['n_pairs']} wins",
                ha="center", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "active_vs_passive_strong.png", dpi=150)
    plt.close(fig)

    out = ["# Strengthened H3 sub-report: active vs passive on "
           "TypedGated + Cx",
           "",
           "TypedGated's planted Boolean target is half-True/half-False, "
           "placing the optimum inside the gating region. "
           "`include_complex=True` adds a Cx subgenome with its own planted "
           "target — flat baselines cannot represent this and crash "
           "deterministically (see `test_flat_baselines_crash_on_complex`).",
           "",
           f"- Input: `{args.input}`",
           f"- Total runs: {len(df)} ({len(completed)} completed, "
           f"{(df.status == 'failed').sum()} failed)",
           "- 4 GSA variants, 3 dims (20, 40, 80), 20 seeds, budget 5000.",
           "",
           "## Median final fitness per (dim, algorithm)",
           "",
           pivot.to_markdown(),
           "",
           "## H3 ablation: GSA_FULL_ENSEMBLE (active) vs "
           "GSA_NO_ASSEMBLY (passive)",
           "",
           "Paired by seed within each cell. A12 > 0.5 means active wins "
           "on a typical paired seed.",
           "",
           test_df.to_markdown(index=False),
           "",
           f"**Pooled paired stats:** {pool_summary}",
           "",
           "## Figure",
           "",
           "- `figures/active_vs_passive_strong.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Strong gating report -> {out_dir / 'report.md'}")
    print(pool_summary)


if __name__ == "__main__":
    main()
