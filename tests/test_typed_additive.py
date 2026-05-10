import numpy as np
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def test_planted_optimum_yields_zero():
    p = TypedAdditive(budget=EvaluationBudget(1000), seed=42, dim=12,
                      families=("Z", "R", "B", "C", "Cx", "E"))
    target_bundle = TypedBundle({
        f: TypedSubgenome(f, p.target[f].copy(), p.specs[f])
        for f in p.specs
    })
    f = p._raw_evaluate(target_bundle)
    assert f < 1e-9


def test_random_bundle_has_positive_fitness():
    p = TypedAdditive(budget=EvaluationBudget(1000), seed=0, dim=12,
                      families=("R", "B"))
    # Use a different seed than the problem's so we don't accidentally
    # reproduce the planted target via aligned RNG calls.
    rng = np.random.default_rng(99)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R, rng.uniform(-5, 5, size=6),
                                     p.specs[GeneFamily.R]),
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     rng.random(6) < 0.5,
                                     p.specs[GeneFamily.B]),
    })
    f = p._raw_evaluate(bundle)
    assert f > 0.0


def test_evaluate_consumes_budget():
    budget = EvaluationBudget(10)
    p = TypedAdditive(budget=budget, seed=0, dim=4, families=("R",))
    rng = np.random.default_rng(0)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R,
                                     rng.uniform(-1, 1, size=4),
                                     p.specs[GeneFamily.R]),
    })
    p.evaluate(bundle)
    assert budget.consumed == 1


def test_one_family_only():
    p = TypedAdditive(budget=EvaluationBudget(100), seed=0, dim=8, families=("R",))
    assert set(p.specs.keys()) == {GeneFamily.R}
    assert p.specs[GeneFamily.R].n == 8
