"""Deterministic seed derivation for runs.

A single `master_seed` deterministically derives sub-seeds for each source of
randomness in a run. This guarantees reproducibility: same master_seed →
byte-identical run output.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


def _entropy_128(ss: np.random.SeedSequence) -> int:
    """Extract 128 bits of entropy from a spawned SeedSequence.

    SeedSequence carries 128 bits of internal state. `generate_state(1)`
    returns one uint32 and discards 96 bits — risking sub-seed collisions.
    `generate_state(4)` returns four uint32 = 128 bits, which we pack into
    one Python int (np.random.default_rng accepts arbitrary-size ints).
    """
    return int.from_bytes(ss.generate_state(4).tobytes(), "little")


class RunSeeds(BaseModel):
    model_config = ConfigDict(frozen=True)

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

    Each sub-seed preserves the full 128 bits of entropy from the spawned
    SeedSequence, matching the resolution of np.random.default_rng's
    accepted seed range.
    """
    ss = np.random.SeedSequence(master_seed)
    init, problem, operators, selection, noise = ss.spawn(5)
    return RunSeeds(
        master_seed=master_seed,
        seed_init=_entropy_128(init),
        seed_problem=_entropy_128(problem),
        seed_operators=_entropy_128(operators),
        seed_selection=_entropy_128(selection),
        seed_noise=_entropy_128(noise),
    )
