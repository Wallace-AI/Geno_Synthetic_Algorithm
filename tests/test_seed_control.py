import numpy as np
import pytest
from pydantic import ValidationError

from gsa.experiments.seed_control import RunSeeds, derive_run_seeds


def test_derive_run_seeds_deterministic():
    a = derive_run_seeds(master_seed=42)
    b = derive_run_seeds(master_seed=42)
    assert a == b


def test_derive_run_seeds_all_subseeds_change_with_master():
    """Every sub-seed must depend on the master, not just one of them."""
    a = derive_run_seeds(master_seed=42)
    b = derive_run_seeds(master_seed=43)
    assert a.seed_init != b.seed_init
    assert a.seed_problem != b.seed_problem
    assert a.seed_operators != b.seed_operators
    assert a.seed_selection != b.seed_selection
    assert a.seed_noise != b.seed_noise


def test_derive_run_seeds_subseeds_distinct():
    seeds = derive_run_seeds(master_seed=42)
    s = {seeds.seed_init, seeds.seed_problem, seeds.seed_operators,
         seeds.seed_selection, seeds.seed_noise}
    assert len(s) == 5


def test_run_seeds_produces_reproducible_rng():
    s = derive_run_seeds(master_seed=42)
    rng1 = np.random.default_rng(s.seed_operators)
    rng2 = np.random.default_rng(s.seed_operators)
    assert np.array_equal(rng1.random(100), rng2.random(100))


def test_run_seeds_is_frozen():
    """RunSeeds must be immutable to prevent accidental mutation mid-run."""
    s = derive_run_seeds(master_seed=42)
    with pytest.raises(ValidationError):
        s.seed_init = 999  # type: ignore[misc]


def test_derive_run_seeds_golden_values_master_42():
    """Regression guard: pin exact seed values for master_seed=42.

    The literal integers below were captured from the reference implementation
    (numpy SeedSequence.spawn → 128-bit little-endian extraction). If this
    test breaks, the seed-derivation algorithm changed and pre-registered
    experiments using master_seed=42 will no longer reproduce.
    """
    s = derive_run_seeds(master_seed=42)
    assert s.master_seed == 42
    assert s.seed_init     == 89243099840137918447686405965974980260
    assert s.seed_problem  == 142136486468327698964923369428397472954
    assert s.seed_operators == 198713065984535447990110872336773596653
    assert s.seed_selection == 315732897500224043183049612165647419589
    assert s.seed_noise    == 25548743054925076440679625865156830294
