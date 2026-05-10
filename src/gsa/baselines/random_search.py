"""Random search baselines: typed and flattened."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gsa.baselines.decoder import flatten_specs, decode_to_bundle, random_flat
from gsa.core.genome import TypedBundle
from gsa.core.populations import sample_initial_subgenome
from gsa.experiments.seed_control import derive_run_seeds


@dataclass
class BaselineResult:
    best_fitness: float
    best_bundle: Optional[TypedBundle]
    total_evaluations: int


def random_flattened_search(problem, master_seed: int) -> BaselineResult:
    seeds = derive_run_seeds(master_seed)
    rng = np.random.default_rng(seeds.seed_init)
    layout = flatten_specs(problem.specs)
    best_f = float("inf")
    best_bundle: Optional[TypedBundle] = None
    while problem.budget.has(1):
        flat = random_flat(problem.specs, layout, rng)
        bundle = decode_to_bundle(flat, problem.specs, layout)
        f = problem.evaluate(bundle)
        if f < best_f:
            best_f = f
            best_bundle = bundle
    return BaselineResult(best_f, best_bundle, problem.budget.consumed)


def random_typed_search(problem, master_seed: int) -> BaselineResult:
    """Typed random: sample each family from its admissible distribution."""
    seeds = derive_run_seeds(master_seed)
    rng = np.random.default_rng(seeds.seed_init)
    best_f = float("inf")
    best_bundle: Optional[TypedBundle] = None
    while problem.budget.has(1):
        bundle = TypedBundle({
            fam: sample_initial_subgenome(spec, rng)
            for fam, spec in problem.specs.items()
        })
        f = problem.evaluate(bundle)
        if f < best_f:
            best_f = f
            best_bundle = bundle
    return BaselineResult(best_f, best_bundle, problem.budget.consumed)
