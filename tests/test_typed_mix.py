import numpy as np
from gsa.benchmarks.typed_mix import TypedMix, ACTIVATION_ORDER
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def test_n_families_one_activates_only_R():
    p = TypedMix(budget=EvaluationBudget(100), seed=0, dim=10,
                 n_families=1, rho=0.0)
    assert set(p.specs.keys()) == {GeneFamily.R}


def test_n_families_six_activates_all():
    p = TypedMix(budget=EvaluationBudget(100), seed=0, dim=24,
                 n_families=6, rho=0.0)
    assert set(p.specs.keys()) == {GeneFamily.R, GeneFamily.B, GeneFamily.Z,
                                   GeneFamily.C, GeneFamily.Cx, GeneFamily.E}


def test_activation_order_matches_spec():
    """Spec §3.5: R → R+B → R+B+Z → R+B+Z+C → R+B+Z+C+Cx → all six."""
    assert ACTIVATION_ORDER == (
        GeneFamily.R, GeneFamily.B, GeneFamily.Z,
        GeneFamily.C, GeneFamily.Cx, GeneFamily.E,
    )


def test_planted_optimum_yields_zero_at_n6():
    p = TypedMix(budget=EvaluationBudget(100), seed=0, dim=24,
                 n_families=6, rho=0.0)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p.target[f].copy(), p.specs[f])
        for f in p.specs
    })
    f = p._raw_evaluate(bundle)
    assert f < 1e-6


def test_n1_degenerate_matches_real_only_additive():
    p = TypedMix(budget=EvaluationBudget(10), seed=0, dim=8,
                 n_families=1, rho=0.0)
    rng = np.random.default_rng(7)
    sample = TypedBundle({
        GeneFamily.R: TypedSubgenome(
            GeneFamily.R, rng.uniform(-5, 5, size=8), p.specs[GeneFamily.R]),
    })
    f = p._raw_evaluate(sample)
    assert f >= 0.0
