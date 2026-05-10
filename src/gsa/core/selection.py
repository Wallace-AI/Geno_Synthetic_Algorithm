"""Selection methods.

Tournament-3 is the default. Diversity-regularized selection blends fitness rank
and diversity rank by α (α=1.0 → pure fitness; α=0.7 → GSA_FULL_ENSEMBLE default;
α=0.0 → pure diversity)."""
from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    """0-based ranks; ties broken by index order."""
    order = np.argsort(values, kind="stable")
    if not ascending:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def tournament_select(fitness: np.ndarray, *, k: int = 3,
                      rng: np.random.Generator,
                      minimize: bool = True) -> int:
    """Pick k random indices, return the one with best fitness."""
    n = len(fitness)
    candidates = rng.choice(n, size=min(k, n), replace=False)
    if minimize:
        return int(candidates[np.argmin(fitness[candidates])])
    return int(candidates[np.argmax(fitness[candidates])])


def diversity_regularized_select(fitness: np.ndarray, diversity: np.ndarray,
                                 *, k: int = 3,
                                 rng: np.random.Generator,
                                 alpha: float = 0.7,
                                 minimize: bool = True) -> int:
    """Score = α·fitness_rank + (1-α)·(reverse diversity rank).

    Lower combined-score is better. fitness_rank is 0 for best fitness;
    diversity_rank is 0 for least diverse, so we use (n-1)-div_rank to make
    "high diversity → low score (good)"."""
    fit_ranks = _ranks(fitness, ascending=minimize)
    div_ranks = _ranks(-diversity, ascending=True)  # higher diversity -> lower rank
    n = len(fitness)
    combined = alpha * fit_ranks + (1.0 - alpha) * div_ranks
    candidates = rng.choice(n, size=min(k, n), replace=False)
    return int(candidates[np.argmin(combined[candidates])])
