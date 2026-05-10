import numpy as np
from gsa.core.credit import (
    DirectCredit, EliteCredit, EnsembleCredit, MarginalCredit,
    EvaluatedBundle,
)
from gsa.core.genome import TypedSubgenome, TypedBundle
from gsa.core.types import GeneFamily, RealSpec, BooleanSpec


def make_bundle(r_vals=(0.5, 0.5)):
    rspec = RealSpec(n=2, lo=np.zeros(2), hi=np.ones(2))
    bspec = BooleanSpec(n=2)
    return TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R, np.array(r_vals), rspec),
        GeneFamily.B: TypedSubgenome(GeneFamily.B, np.array([True, True]), bspec),
    })


def test_direct_credit_assigns_full_fitness_to_each():
    cm = DirectCredit()
    eb = EvaluatedBundle(bundle=make_bundle(), fitness=2.5)
    credits = cm.assign(eb, partner_pool=None, problem=None, rng=None)
    assert credits[GeneFamily.R] == 2.5
    assert credits[GeneFamily.B] == 2.5


def test_elite_credit_uses_elite_partners(monkeypatch):
    """Elite credit pairs the subgenome with elite reps from other families.

    We mock the problem.evaluate so we control returned fitness."""
    cm = EliteCredit()
    eb = EvaluatedBundle(bundle=make_bundle((0.0, 0.0)), fitness=10.0)
    elite_R = TypedSubgenome(GeneFamily.R,
                             np.array([0.0, 0.0]),
                             eb.bundle.get(GeneFamily.R).spec)
    elite_B = TypedSubgenome(GeneFamily.B,
                             np.array([True, True]),
                             eb.bundle.get(GeneFamily.B).spec)
    elites = {GeneFamily.R: elite_R, GeneFamily.B: elite_B}

    class FakeProblem:
        def evaluate(self, bundle):
            return 1.5

    p = FakeProblem()
    credits = cm.assign(eb, partner_pool=elites, problem=p, rng=None)
    assert credits[GeneFamily.R] == 1.5
    assert credits[GeneFamily.B] == 1.5


def test_ensemble_credit_averages_K_partners():
    cm = EnsembleCredit(K=5)
    eb = EvaluatedBundle(bundle=make_bundle(), fitness=1.0)
    rspec = eb.bundle.get(GeneFamily.R).spec
    bspec = eb.bundle.get(GeneFamily.B).spec
    R_pool = [TypedSubgenome(GeneFamily.R,
                             np.array([0.5, 0.5]),
                             rspec) for _ in range(10)]
    B_pool = [TypedSubgenome(GeneFamily.B,
                             np.array([True, True]),
                             bspec) for _ in range(10)]
    pools = {GeneFamily.R: R_pool, GeneFamily.B: B_pool}
    rng = np.random.default_rng(0)

    seen = []

    class FakeProblem:
        def evaluate(self, bundle):
            seen.append(1)
            return 2.0

    p = FakeProblem()
    credits = cm.assign(eb, partner_pool=pools, problem=p, rng=rng)
    # 2 families × 5 partner contexts = 10 evaluations
    assert len(seen) == 10
    assert credits[GeneFamily.R] == 2.0
    assert credits[GeneFamily.B] == 2.0


def test_marginal_credit_uses_neutral_default():
    cm = MarginalCredit()
    eb = EvaluatedBundle(bundle=make_bundle((1.0, 1.0)), fitness=4.0)

    class FakeProblem:
        # Returns 4.0 for full bundle; 1.0 if a family is replaced by neutral
        def evaluate(self, bundle):
            R_vals = bundle.get(GeneFamily.R).values
            B_vals = bundle.get(GeneFamily.B).values
            if (R_vals == 0).all() or (~B_vals).all():
                return 1.0
            return 4.0

    credits = cm.assign(eb, partner_pool=None, problem=FakeProblem(), rng=None)
    # Marginal contribution = 4.0 (with) - 1.0 (neutral) = 3.0
    assert credits[GeneFamily.R] == 3.0
    assert credits[GeneFamily.B] == 3.0
