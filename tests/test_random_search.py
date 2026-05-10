import numpy as np
from gsa.baselines.random_search import random_flattened_search, random_typed_search
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def test_random_flattened_returns_best_seen():
    budget = EvaluationBudget(total=200)
    p = TypedAdditive(budget=budget, seed=0, dim=10, families=("R", "B"))
    result = random_flattened_search(p, master_seed=42)
    assert result.best_fitness > 0
    assert result.total_evaluations <= 200


def test_random_typed_works_with_complex_and_embedding():
    budget = EvaluationBudget(total=100)
    p = TypedAdditive(budget=budget, seed=0, dim=12,
                      families=("R", "B", "Cx", "E"))
    result = random_typed_search(p, master_seed=42)
    assert result.total_evaluations <= 100
    assert result.best_fitness >= 0


def test_random_search_reproducible():
    budget1 = EvaluationBudget(50)
    budget2 = EvaluationBudget(50)
    p1 = TypedAdditive(budget=budget1, seed=0, dim=8, families=("R",))
    p2 = TypedAdditive(budget=budget2, seed=0, dim=8, families=("R",))
    r1 = random_flattened_search(p1, master_seed=42)
    r2 = random_flattened_search(p2, master_seed=42)
    assert r1.best_fitness == r2.best_fitness
