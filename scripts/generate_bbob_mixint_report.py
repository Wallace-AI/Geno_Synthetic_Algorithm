"""Generate the BBOB-MixInt sub-report."""
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

# Mirror the script's grid so we can recover (function, instance, dim) per
# config_hash. Must match scripts/run_bbob_mixint_experiments.py exactly.
FUNCTIONS = (1, 8, 15, 21)
INSTANCES = (1, 2, 3)
DIMS = (10,)
ALGORITHMS = (
    "FLATTENED_DE", "FLATTENED_EA", "MIXED_VARIABLE_GA",
    "GSA_FULL_ENSEMBLE", "GSA_ELITE_CONTEXT", "GSA_DIRECT",
)
FN_NAMES = {1: "f01_sphere", 8: "f08_rosenbrock",
            15: "f15_rastrigin", 21: "f21_gallagher"}
BUDGET = 5000


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="results/raw/bbob_mixint/runs.parquet")
    parser.add_argument("--out-dir", type=str,
                        default="results/reports/bbob_mixint")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="Evaluation budget used (for config_hash recovery).")
    args = parser.parse_args()
    budget = args.budget

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    completed = df[df["status"] == "completed"].copy()

    # Recover (function, instance, dim) by mapping config_hash -> grid.
    lookup = []
    for fn, inst, dim, algo in product(FUNCTIONS, INSTANCES, DIMS, ALGORITHMS):
        h = _config_hash(algo, fn, inst, dim, budget)
        lookup.append({"config_hash": h, "function": fn, "instance": inst,
                       "fn_name": FN_NAMES[fn]})
    lookup_df = pd.DataFrame(lookup)
    completed = completed.merge(lookup_df, on="config_hash", how="left")
    completed["cell_idx"] = (
        completed["function"].astype(str) + "_i" + completed["instance"].astype(str)
    )

    # Per (cell, algorithm) median final fitness.
    pivot = (completed.groupby(["fn_name", "instance", "algorithm"])
             ["final_best_observed"].median().unstack())

    # Aggregate across cells: median of medians, plus rank-sum.
    rank_per_cell = pivot.rank(axis=1, method="average")
    mean_rank = rank_per_cell.mean(axis=0).sort_values()

    # Pairwise Wilcoxon between GSA_FULL_ENSEMBLE and each baseline,
    # paired by (cell, seed).
    rows = []
    sync_algo = "GSA_FULL_ENSEMBLE"
    other_algos = [a for a in completed["algorithm"].unique() if a != sync_algo]
    for other in other_algos:
        sync_v = []
        other_v = []
        for cell, grp in completed.groupby("cell_idx"):
            for seed, sub in grp.groupby("seed_master"):
                a_row = sub[sub["algorithm"] == sync_algo]
                b_row = sub[sub["algorithm"] == other]
                if len(a_row) == 0 or len(b_row) == 0:
                    continue
                sync_v.append(float(a_row["final_best_observed"].iloc[0]))
                other_v.append(float(b_row["final_best_observed"].iloc[0]))
        if not sync_v:
            continue
        try:
            _, p = wilcoxon_paired(np.array(sync_v), np.array(other_v))
        except Exception:
            p = 1.0
        a12 = vargha_delaney_a12(np.array(sync_v), np.array(other_v))
        rows.append({
            "reference": sync_algo, "vs": other,
            "median_ref": float(np.median(sync_v)),
            "median_vs": float(np.median(other_v)),
            "p_value": p, "A12_ref_vs_vs": a12,
        })
    test_df = pd.DataFrame(rows).sort_values("median_vs")
    test_df.to_csv(out_dir / "wilcoxon_gsa_vs_baselines.csv", index=False)

    # Plot mean rank
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_rank.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_xlabel("Mean rank across cells (lower = better)")
    ax.set_title("BBOB-MixInt: 4 functions x 3 instances x 5 seeds, dim=10, budget=5000")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "bbob_mixint_mean_rank.png", dpi=120)
    plt.close(fig)

    out = ["# BBOB-MixInt sub-report (COCO bbob-mixint suite)",
           "",
           "Functions: f1 (Sphere), f8 (Rosenbrock), f15 (Rastrigin), "
           "f21 (Gallagher Gauss 101).",
           f"Instances: 1, 2, 3. Dim: 10 (8 integer + 2 real). Budget: {budget}.",
           "Seeds: 5.",
           "",
           f"- Input: `{args.input}`",
           f"- Total runs: {len(df)} ({len(completed)} completed)",
           "",
           "## Mean rank by algorithm",
           "",
           "Lower rank = better. Each cell (function x instance) is one "
           "ranking; we average across cells.",
           "",
           mean_rank.to_frame("mean_rank").to_markdown(),
           "",
           "## Median final fitness per cell",
           "",
           pivot.to_markdown(),
           "",
           "## Paired Wilcoxon: GSA_FULL_ENSEMBLE vs each baseline",
           "",
           "Pairings are by (function, instance, seed). A12 > 0.5 means "
           "GSA wins on a typical pair.",
           "",
           test_df.to_markdown(index=False),
           "",
           "## Figure",
           "",
           "- `figures/bbob_mixint_mean_rank.png`",
           ""]
    (out_dir / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"BBOB-MixInt report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
