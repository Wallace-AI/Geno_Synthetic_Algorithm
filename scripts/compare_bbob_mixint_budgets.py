"""Compare BBOB-MixInt results across two budgets (e.g. 5k vs 100k).

Reports mean rank per algorithm at each budget side by side, and per-cell
median final fitness so we can see whether the FLATTENED_DE vs GSA gap
narrows asymptotically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsa.analysis.statistics import wilcoxon_paired, vargha_delaney_a12


FUNCTIONS = (1, 8, 15, 21)
INSTANCES = (1, 2, 3)
DIMS = (10,)
ALGORITHMS = (
    "FLATTENED_DE", "FLATTENED_EA", "MIXED_VARIABLE_GA",
    "GSA_FULL_ENSEMBLE", "GSA_ELITE_CONTEXT", "GSA_DIRECT",
)
FN_NAMES = {1: "f01_sphere", 8: "f08_rosenbrock",
            15: "f15_rastrigin", 21: "f21_gallagher"}


def _config_hash(algorithm: str, function: int, instance: int, dim: int,
                 budget: int) -> str:
    payload = json.dumps({
        "algorithm": algorithm,
        "benchmark": "coco_mixint",
        "benchmark_kwargs": {"function": function, "instance": instance,
                             "dim": dim},
        "algorithm_kwargs": {},
        "budget": budget,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load(path: str, budget: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["status"] == "completed"].copy()
    lookup = []
    for fn, inst, dim, algo in product(FUNCTIONS, INSTANCES, DIMS, ALGORITHMS):
        h = _config_hash(algo, fn, inst, dim, budget)
        lookup.append({"config_hash": h, "function": fn, "instance": inst,
                       "fn_name": FN_NAMES[fn]})
    lookup_df = pd.DataFrame(lookup)
    df = df.merge(lookup_df, on="config_hash", how="left")
    df["budget"] = budget
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-low", type=str,
                        default="results/raw/bbob_mixint/runs.parquet")
    parser.add_argument("--budget-low", type=int, default=5000)
    parser.add_argument("--input-high", type=str,
                        default="results/raw/bbob_mixint_100k/runs.parquet")
    parser.add_argument("--budget-high", type=int, default=100000)
    parser.add_argument("--out-dir", type=str,
                        default="results/reports/bbob_mixint_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df_low = _load(args.input_low, args.budget_low)
    df_high = _load(args.input_high, args.budget_high)
    df = pd.concat([df_low, df_high], ignore_index=True)

    # Mean rank per (budget, algorithm) across cells
    pivot = (df.groupby(["budget", "fn_name", "instance", "algorithm"])
             ["final_best_observed"].median().unstack("algorithm"))
    rank = pivot.rank(axis=1, method="average")
    mean_rank = (rank.reset_index().groupby("budget").mean(numeric_only=True))

    # Wilcoxon at each budget: GSA_FULL_ENSEMBLE / GSA_ELITE_CONTEXT /
    # GSA_DIRECT vs FLATTENED_DE, paired by (function, instance, seed).
    rows = []
    for budget, grp in df.groupby("budget"):
        for gsa_algo in ["GSA_FULL_ENSEMBLE", "GSA_ELITE_CONTEXT", "GSA_DIRECT"]:
            for base_algo in ["FLATTENED_DE", "FLATTENED_EA"]:
                gsa_v, base_v = [], []
                for (fn, inst), sub in grp.groupby(["function", "instance"]):
                    for seed, sg in sub.groupby("seed_master"):
                        a = sg[sg["algorithm"] == gsa_algo]
                        b = sg[sg["algorithm"] == base_algo]
                        if len(a) == 0 or len(b) == 0:
                            continue
                        gsa_v.append(float(a["final_best_observed"].iloc[0]))
                        base_v.append(float(b["final_best_observed"].iloc[0]))
                if not gsa_v:
                    continue
                try:
                    _, p = wilcoxon_paired(np.array(gsa_v), np.array(base_v))
                except Exception:
                    p = 1.0
                a12 = vargha_delaney_a12(np.array(gsa_v), np.array(base_v))
                rows.append({
                    "budget": int(budget),
                    "gsa": gsa_algo, "baseline": base_algo,
                    "median_gsa": float(np.median(gsa_v)),
                    "median_baseline": float(np.median(base_v)),
                    "p_value": p, "A12_gsa_vs_baseline": a12,
                })
    test_df = pd.DataFrame(rows)
    test_df.to_csv(out_dir / "wilcoxon_by_budget.csv", index=False)

    # Plot mean rank shift between the two budgets
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mean_rank_t = mean_rank.T
    x = np.arange(len(mean_rank_t.index))
    width = 0.4
    ax.bar(x - width/2, mean_rank_t[args.budget_low].values, width,
           label=f"{args.budget_low} evals", color="#1f77b4")
    ax.bar(x + width/2, mean_rank_t[args.budget_high].values, width,
           label=f"{args.budget_high} evals", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_rank_t.index, rotation=20, fontsize=8)
    ax.set_ylabel("Mean rank (lower = better)")
    ax.set_title(f"BBOB-MixInt mean rank: {args.budget_low} vs "
                 f"{args.budget_high} evaluations")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "rank_shift.png", dpi=120)
    plt.close(fig)

    # Markdown
    out = [f"# BBOB-MixInt budget comparison: {args.budget_low} vs "
           f"{args.budget_high}",
           "",
           f"- Low-budget input:  `{args.input_low}` ({len(df_low)} runs)",
           f"- High-budget input: `{args.input_high}` ({len(df_high)} runs)",
           "",
           "## Mean rank per algorithm at each budget",
           "",
           "Lower rank = better. Averaged across 12 cells (4 functions x "
           "3 instances).",
           "",
           mean_rank.T.to_markdown(),
           "",
           "## Median final fitness per (function, instance, budget)",
           "",
           pivot.to_markdown(),
           "",
           "## Paired Wilcoxon: GSA vs flattened baselines at each budget",
           "",
           "A12 < 0.5 means GSA wins (lower fitness) on a typical paired seed.",
           "",
           test_df.to_markdown(index=False),
           "",
           "## Figure",
           "",
           "- `figures/rank_shift.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Comparison report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
