"""Reproducibility gate: same master_seed -> byte-identical key outputs.

This file is the canonical home for the reproducibility contract. It will
expand in Task 2.20 to assert byte-identical Tier-1 logs from end-to-end
GSA runs. For now, it asserts the foundation: seeds are deterministic and
the RNGs derived from those seeds produce byte-identical streams.
"""
import numpy as np

from gsa.experiments.seed_control import derive_run_seeds


def test_seeds_deterministic_across_runs():
    a = derive_run_seeds(master_seed=42)
    b = derive_run_seeds(master_seed=42)
    assert a.model_dump() == b.model_dump()


def test_seeded_rng_outputs_byte_identical():
    seeds = derive_run_seeds(master_seed=42)
    rng_a = np.random.default_rng(seeds.seed_operators)
    rng_b = np.random.default_rng(seeds.seed_operators)

    a = rng_a.normal(size=(10, 10))
    b = rng_b.normal(size=(10, 10))

    assert (a == b).all()
    assert a.tobytes() == b.tobytes()


def test_seeds_for_different_masters_diverge():
    a = derive_run_seeds(master_seed=1)
    b = derive_run_seeds(master_seed=2)
    rng_a = np.random.default_rng(a.seed_operators)
    rng_b = np.random.default_rng(b.seed_operators)
    assert not np.array_equal(rng_a.normal(size=100), rng_b.normal(size=100))
