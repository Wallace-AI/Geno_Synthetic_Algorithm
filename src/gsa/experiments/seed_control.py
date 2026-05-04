"""Deterministic seed derivation for runs.

A single `master_seed` deterministically derives sub-seeds for each source of
randomness in a run. This guarantees reproducibility: same master_seed →
byte-identical run output.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class RunSeeds(BaseModel):
    master_seed: int
    seed_init: int
    seed_problem: int
    seed_operators: int
    seed_selection: int
    seed_noise: int


def derive_run_seeds(master_seed: int) -> RunSeeds:
    """Spawn 5 sub-seeds from master via numpy SeedSequence.

    The five sub-seeds correspond to:
        - seed_init: initial population sampling
        - seed_problem: benchmark instance construction (planted optima)
        - seed_operators: variation operators
        - seed_selection: tournament/sampling
        - seed_noise: observation-noise wrapper
    """
    ss = np.random.SeedSequence(master_seed)
    init, problem, operators, selection, noise = ss.spawn(5)
    return RunSeeds(
        master_seed=master_seed,
        seed_init=int(init.generate_state(1)[0]),
        seed_problem=int(problem.generate_state(1)[0]),
        seed_operators=int(operators.generate_state(1)[0]),
        seed_selection=int(selection.generate_state(1)[0]),
        seed_noise=int(noise.generate_state(1)[0]),
    )
