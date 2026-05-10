"""Cooperative coevolution with RANDOM (not type-based) decomposition.

Per spec §4.2: tests "any decomposition" against GSA's "type-decomposition
specifically"."""
from __future__ import annotations

import numpy as np

from gsa.baselines.decoder import (
    flatten_specs, decode_to_bundle, random_flat,
)
from gsa.baselines.random_search import BaselineResult
from gsa.experiments.seed_control import derive_run_seeds


def _make_random_groups(total_dim: int, n_groups: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Random partitioning of indices into n_groups roughly equal subsets."""
    perm = rng.permutation(total_dim)
    return [np.sort(g) for g in np.array_split(perm, n_groups)]


def cooperative_coevolution(problem, master_seed: int, *,
                             n_subgroups: int = 5,
                             pop_size: int = 20,
                             F: float = 0.5, CR: float = 0.9) -> BaselineResult:
    seeds = derive_run_seeds(master_seed)
    rng_init = np.random.default_rng(seeds.seed_init)
    rng_op = np.random.default_rng(seeds.seed_operators)
    layout = flatten_specs(problem.specs)
    D = layout.total_dim
    groups = _make_random_groups(D, n_subgroups, rng_init)

    subpops = [
        np.array([random_flat(problem.specs, layout, rng_init)[g]
                  for _ in range(pop_size)])
        for g in groups
    ]
    elite_flat = np.zeros(D)
    for g, sub in zip(groups, subpops):
        elite_flat[g] = sub[0]

    fit_subs = [np.full(pop_size, np.inf) for _ in groups]
    best_f = float("inf")
    best_bundle = None

    for s_idx, (g, sub) in enumerate(zip(groups, subpops)):
        for i in range(pop_size):
            if not problem.budget.has(1):
                break
            test = elite_flat.copy()
            test[g] = sub[i]
            bundle = decode_to_bundle(test, problem.specs, layout)
            f = problem.evaluate(bundle)
            fit_subs[s_idx][i] = f
            if f < best_f:
                best_f, best_bundle = f, bundle
                elite_flat = test

    while problem.budget.has(1):
        for s_idx, (g, sub) in enumerate(zip(groups, subpops)):
            if not problem.budget.has(1):
                break
            for i in range(pop_size):
                if not problem.budget.has(1):
                    break
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2, r3 = rng_op.choice(idxs, size=3, replace=False)
                v = sub[r1] + F * (sub[r2] - sub[r3])
                mask = rng_op.random(len(g)) < CR
                j_rand = rng_op.integers(0, len(g))
                mask[j_rand] = True
                child = np.where(mask, v, sub[i])
                test = elite_flat.copy()
                test[g] = child
                bundle = decode_to_bundle(test, problem.specs, layout)
                f_child = problem.evaluate(bundle)
                if f_child < fit_subs[s_idx][i]:
                    sub[i] = child
                    fit_subs[s_idx][i] = f_child
                    if f_child < best_f:
                        best_f, best_bundle = f_child, bundle
                        elite_flat = test

    return BaselineResult(best_f, best_bundle, problem.budget.consumed)
