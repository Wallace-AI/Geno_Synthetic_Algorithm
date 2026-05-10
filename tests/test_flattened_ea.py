from gsa.baselines.flattened_ea import flattened_ea
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def test_flattened_ea_improves_over_initial_random():
    budget = EvaluationBudget(2000)
    p = TypedAdditive(budget=budget, seed=0, dim=8, families=("R", "B"))
    result = flattened_ea(p, master_seed=42, pop_size=30)
    assert result.best_fitness < 1.0
    assert result.total_evaluations <= 2000
