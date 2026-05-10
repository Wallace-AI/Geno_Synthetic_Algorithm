"""P5 gate: OneMax matches published optimum within 5% at D=50."""
from gsa.benchmarks.ioh_adapter import OneMaxLocal
from gsa.baselines.flattened_ea import flattened_ea
from gsa.experiments.budget import EvaluationBudget


def test_flattened_ea_solves_onemax_d50():
    p = OneMaxLocal(budget=EvaluationBudget(10 * 50 ** 2), seed=0, n=50)
    result = flattened_ea(p, master_seed=42, pop_size=50)
    # Within 5% of optimum (= 0)
    assert result.best_fitness <= 0.05 * 50  # at most 2.5 mismatched bits
