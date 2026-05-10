from gsa.baselines.flattened_de import flattened_de
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def test_flattened_de_beats_random_on_small_real_problem():
    budget = EvaluationBudget(2000)
    p = TypedAdditive(budget=budget, seed=0, dim=8, families=("R",))
    result = flattened_de(p, master_seed=42, F=0.5, CR=0.9, pop_size=30)
    assert result.best_fitness < 0.1
    assert result.total_evaluations <= 2000


def test_flattened_de_handles_mixed_types_via_decoder():
    budget = EvaluationBudget(2000)
    p = TypedAdditive(budget=budget, seed=0, dim=12,
                       families=("Z", "R", "B", "C"))
    result = flattened_de(p, master_seed=42, pop_size=30)
    assert result.total_evaluations <= 2000
