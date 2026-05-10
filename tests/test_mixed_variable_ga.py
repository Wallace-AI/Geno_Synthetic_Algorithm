from gsa.baselines.mixed_variable_ga import mixed_variable_ga
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def test_mixed_var_ga_runs_without_error():
    budget = EvaluationBudget(1000)
    p = TypedAdditive(budget=budget, seed=0, dim=10,
                      families=("R", "B", "Z", "C"))
    result = mixed_variable_ga(p, master_seed=42, pop_size=30)
    assert result.total_evaluations <= 1000
    assert result.best_fitness >= 0


def test_mixed_var_ga_falls_back_when_pymoo_unavailable(monkeypatch):
    """Force the local fallback path."""
    import gsa.baselines.mixed_variable_ga as mvg
    monkeypatch.setattr(mvg, "_PYMOO_AVAILABLE", False)
    budget = EvaluationBudget(500)
    p = TypedAdditive(budget=budget, seed=0, dim=8, families=("R", "B"))
    result = mixed_variable_ga(p, master_seed=42, pop_size=20)
    assert result.total_evaluations <= 500
