"""Paper-grade headline figure for the BBOB-MixInt budget crossover.

Line chart: mean rank (across 12 cells) vs evaluation budget on log x-axis,
one line per algorithm. The visual story is the rank crossover between
budgets: FLATTENED_EA rises (gets worse), GSA_DIRECT descends (gets better),
and at 100k they have switched places.
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


FUNCTIONS = (1, 8, 15, 21)
INSTANCES = (1, 2, 3)
DIMS = (10,)
ALGORITHMS = (
    "FLATTENED_DE", "FLATTENED_EA", "MIXED_VARIABLE_GA",
    "GSA_FULL_ENSEMBLE", "GSA_ELITE_CONTEXT", "GSA_DIRECT",
)
FN_NAMES = {1: "f01_sphere", 8: "f08_rosenbrock",
            15: "f15_rastrigin", 21: "f21_gallagher"}

# Algorithm display order, colours, and line styles chosen so the visual
# carries semantic content (GSA family in one hue, flattened in another).
DISPLAY = {
    "FLATTENED_DE":      ("Flattened DE",      "#1f77b4", "-",  "o"),
    "FLATTENED_EA":      ("Flattened EA",      "#1f77b4", "--", "s"),
    "MIXED_VARIABLE_GA": ("Mixed-Variable GA", "#1f77b4", ":",  "^"),
    "GSA_DIRECT":        ("GSA Direct",        "#d62728", "-",  "o"),
    "GSA_ELITE_CONTEXT": ("GSA Elite Context", "#d62728", "--", "s"),
    "GSA_FULL_ENSEMBLE": ("GSA Full Ensemble", "#d62728", ":",  "^"),
}


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
                        default="results/reports/bbob_mixint_compare/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_low = _load(args.input_low, args.budget_low)
    df_high = _load(args.input_high, args.budget_high)
    df = pd.concat([df_low, df_high], ignore_index=True)

    # Per-cell median, then rank within (budget, cell)
    cell_med = (df.groupby(["budget", "fn_name", "instance", "algorithm"])
                ["final_best_observed"].median().unstack("algorithm"))
    rank = cell_med.rank(axis=1, method="average")
    mean_rank = (rank.reset_index()
                 .groupby("budget")[list(ALGORITHMS)].mean())

    budgets = sorted(mean_rank.index)
    x = np.array(budgets, dtype=float)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for algo in ALGORITHMS:
        label, color, ls, marker = DISPLAY[algo]
        y = mean_rank[algo].values
        ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                markersize=7, linewidth=1.8, label=label)
        # Right-side label
        ax.annotate(label, xy=(x[-1], y[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b):,}" for b in x])
    ax.set_xlim(x[0] * 0.85, x[-1] * 2.2)
    ax.invert_yaxis()  # lower rank = better; plot best at the top
    ax.set_xlabel("Evaluation budget per run")
    ax.set_ylabel("Mean rank across 12 cells   (lower = better)")
    ax.set_title("BBOB-MixInt: budget crossover\n"
                 "GSA Direct rises to second rank; "
                 "Flattened EA stagnates",
                 loc="left")
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="lower left", frameon=False, ncol=2)

    fig.tight_layout()
    out_path = out_dir / "budget_crossover_headline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Crossover figure -> {out_path}")


if __name__ == "__main__":
    main()
