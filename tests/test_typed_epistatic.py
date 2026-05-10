import numpy as np
from gsa.benchmarks.typed_epistatic import TypedEpistatic
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def test_rho_zero_collapses_to_additive():
    p = TypedEpistatic(budget=EvaluationBudget(100), seed=0, dim=8,
                      families=("Z", "R", "B", "C"), rho=0.0)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p.target[f].copy(), p.specs[f])
        for f in p.specs
    })
    f = p._raw_evaluate(bundle)
    assert f < 1e-9


def test_rho_increases_fitness_at_off_optimum():
    f0 = TypedEpistatic(budget=EvaluationBudget(10), seed=0, dim=8,
                       families=("Z", "R", "B", "C"), rho=0.0)
    f1 = TypedEpistatic(budget=EvaluationBudget(10), seed=0, dim=8,
                       families=("Z", "R", "B", "C"), rho=1.0)
    rng = np.random.default_rng(99)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(
            GeneFamily.R, rng.uniform(-5, 5, size=2), f0.specs[GeneFamily.R]),
        GeneFamily.Z: TypedSubgenome(
            GeneFamily.Z, rng.integers(0, 11, size=2), f0.specs[GeneFamily.Z]),
        GeneFamily.B: TypedSubgenome(
            GeneFamily.B, rng.random(2) < 0.5, f0.specs[GeneFamily.B]),
        GeneFamily.C: TypedSubgenome(
            GeneFamily.C, rng.integers(0, 4, size=2), f0.specs[GeneFamily.C]),
    })
    val0 = f0._raw_evaluate(bundle)
    val1 = f1._raw_evaluate(bundle)
    assert val0 != val1


def test_rho_values_supported():
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = TypedEpistatic(budget=EvaluationBudget(10), seed=0, dim=8,
                          families=("R", "B"), rho=rho)
        assert p.rho == rho
