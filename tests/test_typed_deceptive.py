import numpy as np
from gsa.benchmarks.typed_deceptive import TypedDeceptive
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def test_planted_optimum_yields_zero():
    p = TypedDeceptive(budget=EvaluationBudget(100), seed=0, dim=12)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p.target[f].copy(), p.specs[f])
        for f in p.specs
    })
    f = p._raw_evaluate(bundle)
    assert f < 1e-6


def test_all_zeros_boolean_yields_high_fitness():
    """All-zeros within each 4-bit trap is the deceptive local attractor."""
    p = TypedDeceptive(budget=EvaluationBudget(100), seed=0, dim=12)
    bspec = p.specs[GeneFamily.B]
    z_bool = np.zeros(bspec.n, dtype=bool)
    bundle = TypedBundle({
        f: TypedSubgenome(f,
                          z_bool if f == GeneFamily.B else p.target[f].copy(),
                          p.specs[f])
        for f in p.specs
    })
    f = p._raw_evaluate(bundle)
    assert f > 0.5


def test_only_includes_zrbc():
    p = TypedDeceptive(budget=EvaluationBudget(100), seed=0, dim=12)
    assert set(p.specs.keys()).issubset({GeneFamily.Z, GeneFamily.R,
                                         GeneFamily.B, GeneFamily.C})
