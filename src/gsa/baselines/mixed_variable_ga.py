"""Mixed-Variable GA via pymoo when available; local fallback otherwise."""
from __future__ import annotations

import numpy as np

from gsa.baselines.random_search import BaselineResult
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.operators import (
    integer_operator, real_operator_de, boolean_operator,
    categorical_operator, complex_operator, embedding_operator,
    OperatorContext,
)
from gsa.core.populations import sample_initial_subgenome
from gsa.core.types import GeneFamily
from gsa.experiments.seed_control import derive_run_seeds


try:
    import pymoo  # noqa
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False


_OP_DISPATCH = {
    GeneFamily.Z: integer_operator,
    GeneFamily.R: real_operator_de,
    GeneFamily.B: boolean_operator,
    GeneFamily.C: categorical_operator,
    GeneFamily.Cx: complex_operator,
    GeneFamily.E: embedding_operator,
}


def _local_mixed_var_ga(problem, master_seed: int, pop_size: int) -> BaselineResult:
    seeds = derive_run_seeds(master_seed)
    rng_init = np.random.default_rng(seeds.seed_init)
    rng_op = np.random.default_rng(seeds.seed_operators)
    rng_sel = np.random.default_rng(seeds.seed_selection)

    pop: list[TypedBundle] = []
    fit = np.full(pop_size, np.inf)
    for _ in range(pop_size):
        if not problem.budget.has(1):
            break
        bundle = TypedBundle({
            fam: sample_initial_subgenome(spec, rng_init)
            for fam, spec in problem.specs.items()
        })
        pop.append(bundle)
    for i, bundle in enumerate(pop):
        if not problem.budget.has(1):
            break
        fit[i] = problem.evaluate(bundle)

    best_f = float("inf")
    best_bundle = None
    for i in range(len(pop)):
        if fit[i] < best_f:
            best_f, best_bundle = fit[i], pop[i]

    while problem.budget.has(1):
        for i in range(len(pop)):
            if not problem.budget.has(1):
                break
            idx = rng_sel.choice(len(pop), size=min(3, len(pop)), replace=False)
            parent = pop[int(idx[np.argmin(fit[idx])])]

            child_subs = {}
            for fam, sg in parent.subgenomes.items():
                ctx = None
                if fam == GeneFamily.R and len(pop) >= 3:
                    donor_idx = rng_op.choice(len(pop), size=3, replace=False)
                    ctx = OperatorContext(donors=[pop[j].subgenomes[fam].values
                                                  for j in donor_idx])
                op = _OP_DISPATCH[fam]
                child_vals = op(sg.values, sg.spec, rng=rng_op, ctx=ctx)
                child_subs[fam] = TypedSubgenome(fam, child_vals, sg.spec)
            child = TypedBundle(child_subs)
            f_child = problem.evaluate(child)
            j_worst = int(np.argmax(fit))
            if f_child < fit[j_worst]:
                pop[j_worst] = child
                fit[j_worst] = f_child
                if f_child < best_f:
                    best_f, best_bundle = f_child, child

    return BaselineResult(best_f, best_bundle, problem.budget.consumed)


def _pymoo_mixed_var_ga(problem, master_seed: int, pop_size: int) -> BaselineResult:
    """pymoo-based wrapper. Currently delegates to local fallback for
    deterministic ordering across pymoo versions."""
    return _local_mixed_var_ga(problem, master_seed, pop_size)


def mixed_variable_ga(problem, master_seed: int, *,
                      pop_size: int = 50) -> BaselineResult:
    if _PYMOO_AVAILABLE:
        return _pymoo_mixed_var_ga(problem, master_seed, pop_size)
    return _local_mixed_var_ga(problem, master_seed, pop_size)
