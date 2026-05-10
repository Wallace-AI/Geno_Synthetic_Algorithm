"""P4 gate: each baseline beats RANDOM_FLATTENED on Typed Additive D=20 by ≥1 OOM."""
from gsa.baselines.random_search import random_flattened_search
from gsa.baselines.flattened_de import flattened_de
from gsa.baselines.flattened_ea import flattened_ea
from gsa.baselines.mixed_variable_ga import mixed_variable_ga
from gsa.baselines.cooperative_coevolution import cooperative_coevolution
from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.experiments.budget import EvaluationBudget


def _problem():
    return TypedAdditive(budget=EvaluationBudget(5000),
                          seed=0, dim=20, families=("R", "B"))


def test_random_baseline_floor():
    p = _problem()
    r = random_flattened_search(p, master_seed=42)
    assert r.best_fitness > 0.0


def test_flattened_de_beats_random_floor():
    f_random = random_flattened_search(_problem(), master_seed=42).best_fitness
    f_de = flattened_de(_problem(), master_seed=42).best_fitness
    assert f_de < 0.1 * f_random, \
        f"DE ({f_de:.3f}) must beat Random ({f_random:.3f}) by >=1 OOM"


def test_flattened_ea_beats_random_floor():
    f_random = random_flattened_search(_problem(), master_seed=42).best_fitness
    f_ea = flattened_ea(_problem(), master_seed=42).best_fitness
    assert f_ea < 0.1 * f_random


def test_mixed_var_ga_beats_random_floor():
    f_random = random_flattened_search(_problem(), master_seed=42).best_fitness
    f_mv = mixed_variable_ga(_problem(), master_seed=42).best_fitness
    assert f_mv < 0.1 * f_random


def test_coop_coev_beats_random_floor():
    f_random = random_flattened_search(_problem(), master_seed=42).best_fitness
    f_cc = cooperative_coevolution(_problem(), master_seed=42).best_fitness
    assert f_cc < 0.1 * f_random
