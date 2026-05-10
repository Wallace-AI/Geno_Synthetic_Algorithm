import numpy as np
from gsa.benchmarks.ioh_adapter import (
    OneMaxLocal, LeadingOnesLocal, WModelOneMaxLocal, ioh_problem,
)
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def test_onemax_optimum_at_all_ones():
    p = OneMaxLocal(budget=EvaluationBudget(10), seed=0, n=20)
    bundle = TypedBundle({
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.ones(20, dtype=bool),
                                     p.specs[GeneFamily.B]),
    })
    f = p._raw_evaluate(bundle)
    assert f == 0.0


def test_onemax_zero_vector_yields_n():
    p = OneMaxLocal(budget=EvaluationBudget(10), seed=0, n=20)
    bundle = TypedBundle({
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.zeros(20, dtype=bool),
                                     p.specs[GeneFamily.B]),
    })
    assert p._raw_evaluate(bundle) == 20


def test_leadingones_counts_leading_ones_only():
    p = LeadingOnesLocal(budget=EvaluationBudget(10), seed=0, n=10)
    bundle = TypedBundle({
        GeneFamily.B: TypedSubgenome(
            GeneFamily.B,
            np.array([True, True, True, False, True, True, True, True, True, True]),
            p.specs[GeneFamily.B]),
    })
    assert p._raw_evaluate(bundle) == 7


def test_wmodel_includes_dummy_dimensions():
    p = WModelOneMaxLocal(budget=EvaluationBudget(10), seed=0, n=20,
                          dummy_fraction=0.5)
    spec = p.specs[GeneFamily.B]
    assert spec.n == 20


def test_ioh_problem_dispatch():
    p = ioh_problem("OneMax", n=10, seed=0, budget=EvaluationBudget(10))
    assert p._raw_evaluate(TypedBundle({
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.ones(10, dtype=bool),
                                     p.specs[GeneFamily.B]),
    })) == 0
