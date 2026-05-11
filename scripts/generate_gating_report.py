"""H3 sub-report: ActiveAssembly vs PassiveAssembly on TypedGated."""
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
                        default="results/raw/gating/runs.parquet")
    parser.add_argument("--out-dir", type=str,
                        default="results/reports/gating")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    completed = df[df["status"] == "completed"].copy()

    pivot = (completed.groupby(
        ["dim", "evaluation_budget", "algorithm"])
        ["final_best_true"].median().unstack())

    # Paired Wilcoxon: active (FULL_ENSEMBLE) vs passive (NO_ASSEMBLY)
    rows = []
    for (dim, bud), grp in completed.groupby(["dim", "evaluation_budget"]):
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
            "dim": int(dim), "budget": int(bud),
            "active_median": float(np.median(a)),
            "passive_median": float(np.median(b)),
            "p_value": p, "A12_active_vs_passive": a12,
        })
    test_df = pd.DataFrame(rows)
    test_df.to_csv(out_dir / "wilcoxon_active_vs_passive.csv", index=False)

    # Plot: paired bars of active vs passive medians per cell
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cells = [(r["dim"], r["budget"]) for _, r in test_df.iterrows()]
    x = np.arange(len(cells))
    ax.bar(x - 0.2, test_df["active_median"].values, width=0.4,
           label="GSA_FULL_ENSEMBLE (active)", color="#d62728")
    ax.bar(x + 0.2, test_df["passive_median"].values, width=0.4,
           label="GSA_NO_ASSEMBLY (passive)", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}\nbudget={b}" for d, b in cells], fontsize=9)
    ax.set_ylabel("Median final fitness (lower = better)")
    ax.set_title("H3 ablation: Active vs Passive assembly on TypedGated\n"
                 "annotated with A12 (>0.5 means active wins on typical seed)",
                 loc="left", fontsize=10)
    for i, r in test_df.iterrows():
        ax.text(i, max(r["active_median"], r["passive_median"]) + 0.02,
                f"A12={r['A12_active_vs_passive']:.2f}",
                ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "active_vs_passive.png", dpi=150)
    plt.close(fig)

    out = ["# H3 sub-report: ActiveAssembly vs PassiveAssembly on TypedGated",
           "",
           "TypedGated's optimum sits inside the gating region "
           "(half the planted Boolean target bits are False). "
           "Under active assembly the R values at gated-off positions are "
           "masked to 0 and need not be optimised; under passive assembly "
           "they feed straight through to fitness and must be driven to zero.",
           "",
           f"- Input: `{args.input}`",
           f"- Total runs: {len(df)} ({len(completed)} completed)",
           "",
           "## Median final fitness per (dim, budget, algorithm)",
           "",
           pivot.to_markdown(),
           "",
           "## Paired Wilcoxon: active vs passive",
           "",
           "Paired by seed within each (dim, budget) cell. "
           "A12 > 0.5 means active wins on a typical seed.",
           "",
           test_df.to_markdown(index=False),
           "",
           "## Figure",
           "",
           "- `figures/active_vs_passive.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Gating report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
