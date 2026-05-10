"""Typed Deceptive benchmark per spec §3.3.

- Boolean: 4-bit deceptive traps. All-zeros within a trap = local attractor.
- Categorical: routes to one of K=3 sublandscapes; only category 0 is the
  global basin; categories 1, 2 are local easy basins.
- Real: shifted Rastrigin within the chosen sublandscape.

Families: Z, R, B, C. Excluded: Cx, E."""
from __future__ import annotations

import numpy as np

from gsa.benchmarks.base import Problem
from gsa.core.genome import TypedBundle
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec, TypeSpec,
)


def _trap_score(bits: np.ndarray, trap_size: int = 4) -> float:
    """Per-trap score: 0 when all-ones; trap_size at all-zeros."""
    n_traps = len(bits) // trap_size
    score = 0.0
    for t in range(n_traps):
        chunk = bits[t * trap_size:(t + 1) * trap_size].astype(int)
        u = int(chunk.sum())
        if u == trap_size:
            score += 0.0
        else:
            score += (trap_size - u) + 0.5 * (u == 0)
    return float(score)


def _shifted_rastrigin(x: np.ndarray, shift: np.ndarray) -> float:
    A = 10.0
    z = x - shift
    return float(A * len(z) + np.sum(z ** 2 - A * np.cos(2 * np.pi * z)))


class TypedDeceptive(Problem):
    def __init__(self, budget, seed: int, dim: int = 20,
                 trap_size: int = 4, n_sublandscapes: int = 3,
                 real_lo: float = -5.0, real_hi: float = 5.0):
        self.dim = dim
        self.trap_size = trap_size
        self.n_sublandscapes = n_sublandscapes
        self.real_lo, self.real_hi = real_lo, real_hi
        super().__init__(budget=budget, seed=seed)

    def _setup(self) -> None:
        rng = np.random.default_rng(self.seed)
        each = self.dim // 4
        rem = self.dim - each * 4
        n_R = each + rem
        n_Z = n_B = n_C = each
        # Round B up to multiple of trap_size
        n_B = max(self.trap_size,
                  ((n_B + self.trap_size - 1) // self.trap_size) * self.trap_size)

        self._specs: dict[GeneFamily, TypeSpec] = {
            GeneFamily.Z: IntegerSpec(n=n_Z, lo=np.zeros(n_Z, dtype=int),
                                       hi=np.full(n_Z, 5)),
            GeneFamily.R: RealSpec(n=n_R, lo=np.full(n_R, self.real_lo),
                                    hi=np.full(n_R, self.real_hi)),
            GeneFamily.B: BooleanSpec(n=n_B),
            GeneFamily.C: CategoricalSpec(n=n_C,
                                           n_categories=[self.n_sublandscapes] * n_C),
        }
        self.target: dict[GeneFamily, np.ndarray] = {
            GeneFamily.Z: rng.integers(0, 6, size=n_Z),
            GeneFamily.R: rng.uniform(-2, 2, size=n_R),
            GeneFamily.B: np.ones(n_B, dtype=bool),
            GeneFamily.C: np.zeros(n_C, dtype=np.int64),
        }
        sub_shifts = rng.uniform(-3, 3, size=(self.n_sublandscapes, n_R))
        sub_shifts[0] = self.target[GeneFamily.R]
        self._sub_shifts = sub_shifts

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        total = 0.0
        Z = bundle.subgenomes.get(GeneFamily.Z)
        if Z is not None:
            total += 0.1 * float(np.sum(np.abs(Z.values - self.target[GeneFamily.Z])))

        B = bundle.subgenomes.get(GeneFamily.B)
        if B is not None:
            total += _trap_score(B.values, trap_size=self.trap_size)

        C = bundle.subgenomes.get(GeneFamily.C)
        R = bundle.subgenomes.get(GeneFamily.R)
        if R is not None:
            if C is not None and len(C.values) > 0:
                cat = int(C.values[0])
            else:
                cat = 0
            cat_penalty = 0.0 if cat == 0 else float(2 + 5 * cat)
            shift = self._sub_shifts[cat]
            total += 0.05 * _shifted_rastrigin(R.values, shift) + cat_penalty
            if C is not None and len(C.values) > 1:
                total += float(np.sum(C.values[1:] != 0))

        return total

    def true_optimum(self) -> float:
        return 0.0
