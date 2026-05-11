"""TypedGated benchmark: a fitness landscape whose optimum requires
Boolean gating to be in effect.

Design (testing H3):
  - target_B has a deliberate mix of True/False bits (half-and-half by
    default). The optimum sits **inside** the gating region: at positions
    where target_B[i] = False, the algorithm must not contribute an active
    Real coord.
  - target_R is sampled freely; it is the "intended" value at positions
    where target_B[i] = True.
  - Fitness reads the assembled phenotype's `R_effective`, which is
    Boolean-gated R under ActiveAssembly and raw R under PassiveAssembly.

Penalty contract (lower = better, f=0 at optimum):
  - B mismatch:   sum_i I(B[i] != target_B[i]) / D       in [0, 1]
  - R at on:      sum_{i: target_B[i]=True}  (R_eff[i] - target_R[i])^2
  - R at off:     sum_{i: target_B[i]=False} (R_eff[i])^2  (should be 0)
  - Z (optional): sum_j |Z[j] - target_Z[j]| / max_range  in [0, 1]

Behaviour under each assembly mode at the planted optimum (B = target_B,
R[active] = target_R[active]):
  - ActiveAssembly:   R_effective[inactive] = 0 (masked)  → penalty = 0.
                      R[inactive] is *free*, fewer effective dims to optimise.
  - PassiveAssembly:  R_effective[inactive] = R[inactive] (raw)
                      → the algorithm must additionally drive R[inactive] -> 0.

The expected H3 result is that GSA_FULL_ENSEMBLE (active) converges faster
than GSA_NO_ASSEMBLY (passive) on this benchmark, because active assembly
shrinks the effective search space at the optimum.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gsa.benchmarks.base import Problem
from gsa.core.genome import TypedBundle
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, TypeSpec,
)


class TypedGated(Problem):
    needs_phenotype = True

    def __init__(self, budget, seed: int, dim: int = 20,
                 active_fraction: float = 0.5,
                 include_integer: bool = True,
                 real_lo: float = -5.0, real_hi: float = 5.0,
                 integer_lo: int = 0, integer_hi: int = 10):
        if not (0.0 < active_fraction < 1.0):
            raise ValueError("active_fraction must be strictly between 0 and 1")
        self.dim = int(dim)
        self.active_fraction = float(active_fraction)
        self.include_integer = bool(include_integer)
        self.real_lo, self.real_hi = float(real_lo), float(real_hi)
        self.integer_lo, self.integer_hi = int(integer_lo), int(integer_hi)
        super().__init__(budget=budget, seed=seed)

    def _setup(self) -> None:
        rng = np.random.default_rng(self.seed)
        D = self.dim
        self._specs: dict[GeneFamily, TypeSpec] = {
            GeneFamily.R: RealSpec(n=D,
                                   lo=np.full(D, self.real_lo),
                                   hi=np.full(D, self.real_hi)),
            GeneFamily.B: BooleanSpec(n=D),
        }
        if self.include_integer:
            self._specs[GeneFamily.Z] = IntegerSpec(
                n=D,
                lo=np.full(D, self.integer_lo, dtype=np.int64),
                hi=np.full(D, self.integer_hi, dtype=np.int64),
            )

        # Planted target. Half-True / half-False by default, shuffled.
        n_active = int(round(self.active_fraction * D))
        n_active = max(1, min(D - 1, n_active))
        target_B = np.zeros(D, dtype=bool)
        target_B[:n_active] = True
        rng.shuffle(target_B)
        self._target_B = target_B

        # target_R sampled uniformly in [lo, hi]; meaningful only at
        # target_B[i] = True. We still sample at every i (harmless, since
        # the off-positions use 0 as the target for raw R under passive).
        self._target_R = rng.uniform(self.real_lo, self.real_hi, size=D)

        # Integer target — adds a typed dimension that any flattened-DE
        # baseline must also handle via rounding.
        if self.include_integer:
            self._target_Z = rng.integers(self.integer_lo,
                                          self.integer_hi + 1, size=D)
            self._integer_range = max(1.0,
                float(self.integer_hi - self.integer_lo))
        else:
            self._target_Z = None
            self._integer_range = 1.0

        # Per-component weights so each component caps at 1 at worst-case.
        # We weight them roughly equally — the gating dynamics are what
        # we care about, not absolute fitness magnitude.
        self._w_B = 1.0 / D
        # Worst-case (R - target_R)^2 over [lo, hi]^2 bounded by (hi-lo)^2;
        # weight by 1/(D * (hi-lo)^2) keeps the R term in [0, 1].
        self._w_R = 1.0 / (D * (self.real_hi - self.real_lo) ** 2)
        self._w_Z = 1.0 / (D * self._integer_range) if self.include_integer else 0.0

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        # Fallback path (used by true_evaluate and tests): assemble
        # actively so the "natural" semantics of the benchmark hold.
        from gsa.core.assembly import ActiveAssembly
        pheno, _ = ActiveAssembly().assemble(bundle)
        return self._raw_evaluate_pheno(bundle, pheno)

    def _raw_evaluate_pheno(self, bundle: TypedBundle, phenotype: Any) -> float:
        # Boolean penalty: Hamming distance to target_B, normalised.
        B = bundle.subgenomes[GeneFamily.B].values.astype(bool)
        b_pen = float(np.sum(B != self._target_B)) * self._w_B

        # R penalty: effective R is gated under ActiveAssembly, raw under
        # PassiveAssembly. The fitness *contract* says the optimum requires
        # R_eff[i] = target_R[i] at on-positions and R_eff[i] = 0 at
        # off-positions. Active assembly satisfies the off-position
        # condition automatically (masks to 0); passive must drive raw R -> 0.
        r_eff = np.asarray(phenotype.features["R_effective"], dtype=float)
        target_eff = np.where(self._target_B, self._target_R, 0.0)
        r_pen = float(np.sum((r_eff - target_eff) ** 2)) * self._w_R

        # Integer penalty (always raw, no gating semantics).
        z_pen = 0.0
        if self.include_integer:
            Z = bundle.subgenomes[GeneFamily.Z].values.astype(np.int64)
            z_pen = float(np.sum(np.abs(Z - self._target_Z))) * self._w_Z

        return b_pen + r_pen + z_pen

    def true_optimum(self) -> float:
        return 0.0
