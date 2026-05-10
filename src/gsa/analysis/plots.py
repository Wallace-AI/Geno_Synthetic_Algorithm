"""Plot generation: convergence-style summaries, headline rank-vs-n_families."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in process pools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def headline_rank_vs_n_families(df: pd.DataFrame, out_path: Path) -> None:
    """Rank-vs-n_families plot for Typed-Mix benchmark.

    Each algorithm gets a curve over n_families ∈ {1..6}, showing its median
    rank across seeds at each x. Lower is better."""
    sub = df[(df["status"] == "completed") &
              (df["benchmark"] == "typed_mix")].copy()
    if sub.empty or "n_families" not in sub.columns:
        return
    # Per (n_families, seed): rank algorithms by final_best_true (lower=better)
    sub["rank"] = sub.groupby(["n_families", "dim", "seed_master"])[
        "final_best_true"].rank(method="min")
    median_rank = sub.groupby(["n_families", "algorithm"])["rank"].median().unstack()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in median_rank.columns:
        ax.plot(median_rank.index, median_rank[algo], marker="o", label=algo)
    ax.set_xlabel("n_families")
    ax.set_ylabel("median rank (lower = better)")
    ax.set_title("Algorithm rank vs. number of typed gene families")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=120)
    plt.close(fig)


def per_benchmark_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """One boxplot per (benchmark, dim) cell of final_best_true across seeds."""
    sub = df[df["status"] == "completed"].copy()
    if sub.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for (bench, dim), grp in sub.groupby(["benchmark", "dim"]):
        algos = sorted(grp["algorithm"].unique())
        data = [grp[grp["algorithm"] == a]["final_best_true"].values for a in algos]
        fig, ax = plt.subplots(figsize=(max(6, len(algos) * 0.7), 4))
        ax.boxplot(data, labels=algos, showfliers=True)
        ax.set_title(f"{bench} D={dim}: final best (lower = better)")
        ax.set_ylabel("final_best_true")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"box_{bench}_d{dim}.png", dpi=120)
        plt.close(fig)


def epistasis_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """For Typed Epistatic: median final fitness per (algorithm, rho) cell."""
    sub = df[(df["status"] == "completed") &
              (df["benchmark"] == "typed_epistatic")].copy()
    if sub.empty or sub["rho"].isna().all():
        return
    pivot = sub.groupby(["algorithm", "rho"])["final_best_true"].median().unstack()
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(pivot.index))))
    log_pivot = np.log10(pivot.values + 1e-9)
    im = ax.imshow(log_pivot, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("rho (epistasis)")
    ax.set_title("Typed Epistatic: log10 median fitness")
    fig.colorbar(im, ax=ax, label="log10(f)")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=120)
    plt.close(fig)
