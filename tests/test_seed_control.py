import numpy as np
from gsa.experiments.seed_control import RunSeeds, derive_run_seeds


def test_derive_run_seeds_deterministic():
    a = derive_run_seeds(master_seed=42)
    b = derive_run_seeds(master_seed=42)
    assert a.seed_init == b.seed_init
    assert a.seed_problem == b.seed_problem
    assert a.seed_operators == b.seed_operators
    assert a.seed_selection == b.seed_selection
    assert a.seed_noise == b.seed_noise


def test_derive_run_seeds_different_for_different_master():
    a = derive_run_seeds(master_seed=42)
    b = derive_run_seeds(master_seed=43)
    assert a.seed_init != b.seed_init


def test_derive_run_seeds_subseeds_distinct():
    seeds = derive_run_seeds(master_seed=42)
    s = {seeds.seed_init, seeds.seed_problem, seeds.seed_operators,
         seeds.seed_selection, seeds.seed_noise}
    assert len(s) == 5  # all distinct


def test_run_seeds_produces_reproducible_rng():
    s = derive_run_seeds(master_seed=42)
    rng1 = np.random.default_rng(s.seed_operators)
    rng2 = np.random.default_rng(s.seed_operators)
    assert np.array_equal(rng1.random(100), rng2.random(100))
