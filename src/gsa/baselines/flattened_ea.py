"""Generational GA on flattened float vector.

Tournament-3 selection, single-point crossover, Gaussian mutation σ=0.1·range."""
from __future__ import annotations

import numpy as np

from gsa.baselines.decoder import (
    flatten_specs, decode_to_bundle, random_flat,
)
from gsa.baselines.random_search import BaselineResult
from gsa.experiments.seed_control import derive_run_seeds


def _tournament(fit: np.ndarray, k: int, rng: np.random.Generator) -> int:
    n = len(fit)
    idx = rng.choice(n, size=min(k, n), replace=False)
    return int(idx[np.argmin(fit[idx])])


def flattened_ea(problem, master_seed: int, *,
                 pop_size: int = 50,
                 xover_p: float = 0.9, mut_sigma_frac: float = 0.1,
                 tournament_k: int = 3) -> BaselineResult:
    seeds = derive_run_seeds(master_seed)
    rng_init = np.random.default_rng(seeds.seed_init)
    rng_op = np.random.default_rng(seeds.seed_operators)
    rng_sel = np.random.default_rng(seeds.seed_selection)
    layout = flatten_specs(problem.specs)
    D = layout.total_dim

    scale = np.zeros(D)
    for fam, sl in layout.slices.items():
        spec = problem.specs[fam]
        if hasattr(spec, "lo") and hasattr(spec, "hi"):
            scale[sl] = mut_sigma_frac * (spec.hi - spec.lo).astype(float)
        else:
            scale[sl] = mut_sigma_frac * 2.0

    pop = np.array([random_flat(problem.specs, layout, rng_init)
                    for _ in range(pop_size)])
    fit = np.full(pop_size, np.inf)
    best_f = float("inf")
    best_bundle = None
    for i in range(pop_size):
        if not problem.budget.has(1):
            break
        bundle = decode_to_bundle(pop[i], problem.specs, layout)
        f = problem.evaluate(bundle)
        fit[i] = f
        if f < best_f:
            best_f, best_bundle = f, bundle

    mut_p = 1.0 / max(1, D)
    while problem.budget.has(1):
        new_pop = pop.copy()
        new_fit = fit.copy()
        for i in range(pop_size):
            if not problem.budget.has(1):
                break
            a = _tournament(fit, tournament_k, rng_sel)
            b = _tournament(fit, tournament_k, rng_sel)
            if rng_op.random() < xover_p:
                cut = int(rng_op.integers(1, D))
                child = np.concatenate([pop[a, :cut], pop[b, cut:]])
            else:
                child = pop[a].copy()
            mask = rng_op.random(D) < mut_p
            child = child + mask * rng_op.normal(scale=scale)
            child_bundle = decode_to_bundle(child, problem.specs, layout)
            f_child = problem.evaluate(child_bundle)
            new_pop[i] = child
            new_fit[i] = f_child
            if f_child < best_f:
                best_f, best_bundle = f_child, child_bundle
        pop, fit = new_pop, new_fit

    return BaselineResult(best_f, best_bundle, problem.budget.consumed)
