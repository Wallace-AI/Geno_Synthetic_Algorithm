"""DE/rand/1/bin on flattened float vector with decoder."""
from __future__ import annotations

import numpy as np

from gsa.baselines.decoder import (
    flatten_specs, decode_to_bundle, random_flat,
)
from gsa.baselines.random_search import BaselineResult
from gsa.experiments.seed_control import derive_run_seeds


def flattened_de(problem, master_seed: int, *, F: float = 0.5, CR: float = 0.9,
                 pop_size: int = 50) -> BaselineResult:
    seeds = derive_run_seeds(master_seed)
    rng_init = np.random.default_rng(seeds.seed_init)
    rng_op = np.random.default_rng(seeds.seed_operators)
    layout = flatten_specs(problem.specs)
    D = layout.total_dim

    pop = np.array([random_flat(problem.specs, layout, rng_init)
                    for _ in range(pop_size)])
    fit = np.zeros(pop_size)
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

    while problem.budget.has(1):
        for i in range(pop_size):
            if not problem.budget.has(1):
                break
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = rng_op.choice(idxs, size=3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            mask = rng_op.random(D) < CR
            j_rand = rng_op.integers(0, D)
            mask[j_rand] = True
            child = np.where(mask, v, pop[i])
            child_bundle = decode_to_bundle(child, problem.specs, layout)
            f_child = problem.evaluate(child_bundle)
            if f_child < fit[i]:
                pop[i] = child
                fit[i] = f_child
                if f_child < best_f:
                    best_f, best_bundle = f_child, child_bundle

    return BaselineResult(best_f, best_bundle, problem.budget.consumed)
