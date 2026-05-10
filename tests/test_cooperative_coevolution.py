from gsa.baselines.cooperative_coevolution import cooperative_coevolution
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def test_coop_coev_decomposes_by_random_grouping():
    budget = EvaluationBudget(2000)
    p = TypedAdditive(budget=budget, seed=0, dim=12,
                      families=("R", "B", "Z", "C"))
    result = cooperative_coevolution(p, master_seed=42,
                                     n_subgroups=4, pop_size=20)
    assert result.best_fitness < 5.0
    assert result.total_evaluations <= 2000


def test_coop_coev_grouping_is_random_not_by_type():
    """Critical for Approach R: tests "any decomposition" vs.
    "type-decomposition specifically." Verify grouping is random."""
    from gsa.baselines.cooperative_coevolution import _make_random_groups
    import numpy as np
    rng = np.random.default_rng(0)
    groups = _make_random_groups(total_dim=20, n_groups=4, rng=rng)
    flat = sum((list(g) for g in groups), [])
    assert sorted(flat) == list(range(20))
    for g in groups:
        assert not (np.array_equal(np.sort(g), np.arange(g[0], g[0] + len(g))))
