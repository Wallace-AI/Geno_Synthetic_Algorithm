import numpy as np
from gsa.benchmarks.typed_noisy import TypedNoisy
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.experiments.budget import EvaluationBudget


def test_noise_changes_observed_fitness_but_not_true():
    p = TypedNoisy(budget=EvaluationBudget(100), seed=0, dim=8,
                   noise_mode="gaussian", sigma=0.5)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p.target[f].copy(), p.specs[f])
        for f in p.specs
    })
    obs = [p.evaluate(bundle) for _ in range(20)]
    assert np.std(obs) > 0
    assert p.true_evaluate(bundle) < 1e-9


def test_heavy_tailed_more_extreme_than_gaussian():
    p_g = TypedNoisy(budget=EvaluationBudget(1000), seed=0, dim=4,
                     noise_mode="gaussian", sigma=1.0)
    p_h = TypedNoisy(budget=EvaluationBudget(1000), seed=0, dim=4,
                     noise_mode="heavy_tailed", sigma=1.0, df=3)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p_g.target[f].copy(), p_g.specs[f])
        for f in p_g.specs
    })
    obs_g = np.array([p_g.evaluate(bundle) for _ in range(200)])
    obs_h = np.array([p_h.evaluate(bundle) for _ in range(200)])
    assert np.max(np.abs(obs_h)) > np.max(np.abs(obs_g))


def test_noise_seeded_reproducibly():
    p1 = TypedNoisy(budget=EvaluationBudget(10), seed=42, dim=4,
                    noise_mode="gaussian", sigma=0.5)
    p2 = TypedNoisy(budget=EvaluationBudget(10), seed=42, dim=4,
                    noise_mode="gaussian", sigma=0.5)
    bundle = TypedBundle({
        f: TypedSubgenome(f, p1.target[f].copy(), p1.specs[f])
        for f in p1.specs
    })
    a = p1.evaluate(bundle)
    b = p2.evaluate(bundle)
    assert a == b
