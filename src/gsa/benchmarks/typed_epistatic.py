"""Typed Epistatic benchmark per spec §3.2.

f(x; ρ) = (1-ρ) * f_additive(x) + ρ * f_interaction(x).

Interaction mechanics:
  1. Boolean → Real gating: b_j gates r_j contribution; inactive coords penalized.
  2. Integer → Real subfunction: z_j ∈ {0,1,2,3} indexes sphere/ellipsoid/Rosenbrock/shifted.
  3. Categorical → Real optimum shift.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.core.genome import TypedBundle
from gsa.core.types import GeneFamily


class TypedEpistatic(TypedAdditive):
    def __init__(self, budget, seed: int, dim: int = 20,
                 families: Sequence[str] = ("Z", "R", "B", "C"),
                 rho: float = 0.5,
                 penalty_inactive: float = 0.5,
                 **kwargs):
        if rho < 0 or rho > 1:
            raise ValueError("rho must be in [0, 1]")
        self.rho = rho
        self.penalty_inactive = penalty_inactive
        super().__init__(budget=budget, seed=seed, dim=dim,
                         families=families, **kwargs)
        # Override targets for B/C/Z so the planted optimum yields f=0 even
        # when ρ>0. Categorical target=0 hits the zero-shift subspace; integer
        # target=0 selects the sphere subfunction; Boolean target=all-True
        # passes the gating check.
        import numpy as _np
        if GeneFamily.B in self._specs:
            self.target[GeneFamily.B] = _np.ones(self._specs[GeneFamily.B].n,
                                                  dtype=bool)
        if GeneFamily.C in self._specs:
            self.target[GeneFamily.C] = _np.zeros(self._specs[GeneFamily.C].n,
                                                   dtype=_np.int64)
        if GeneFamily.Z in self._specs:
            self.target[GeneFamily.Z] = _np.zeros(self._specs[GeneFamily.Z].n,
                                                   dtype=_np.int64)

    def _interaction(self, bundle: TypedBundle) -> float:
        """Boolean→Real gating + Integer→Real subfunction selection +
        Categorical→Real shift."""
        R = bundle.subgenomes.get(GeneFamily.R)
        B = bundle.subgenomes.get(GeneFamily.B)
        Z = bundle.subgenomes.get(GeneFamily.Z)
        C = bundle.subgenomes.get(GeneFamily.C)

        if R is None:
            return 0.0
        r_target = self.target[GeneFamily.R]
        r_vals = R.values.copy()

        # Categorical shift. Pin shifts[0]=0 so planted target_C=0 gives no
        # shift and keeps the planted optimum at f=0.
        if C is not None:
            cats = C.values
            rng = np.random.default_rng(self.seed + 7)
            shifts = rng.uniform(-0.5, 0.5, size=(self.n_categories,))
            shifts[0] = 0.0
            n = min(len(cats), len(r_target))
            r_target = r_target.copy()
            r_target[:n] = r_target[:n] + shifts[cats[:n]]

        # Integer subfunction selection
        if Z is not None:
            zs = Z.values % 4
            n = min(len(zs), len(r_vals))
            sub_costs = np.zeros(n)
            for j in range(n):
                d = r_vals[j] - r_target[j]
                if zs[j] == 0:
                    sub_costs[j] = d ** 2
                elif zs[j] == 1:
                    sub_costs[j] = (j + 1) * d ** 2
                elif zs[j] == 2:
                    sub_costs[j] = d ** 2 + 0.1 * abs(d) ** 1.5
                else:
                    sub_costs[j] = d ** 2  # shifted variant: keep zero at d=0
                                            # so planted optimum stays at f=0
        else:
            n = len(r_vals)
            sub_costs = (r_vals - r_target[:n]) ** 2

        # Boolean gating
        if B is not None:
            gates = B.values.astype(bool)
            target_gates = self.target[GeneFamily.B].astype(bool)
            m = min(len(gates), n)
            cost = np.zeros(n)
            cost[:m] = np.where(
                gates[:m] == target_gates[:m],
                sub_costs[:m],
                sub_costs[:m] + self.penalty_inactive,
            )
            if n > m:
                cost[m:] = sub_costs[m:]
        else:
            cost = sub_costs

        return float(np.sum(cost) * self._weights.get(GeneFamily.R, 1.0))

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        f_add = super()._raw_evaluate(bundle)
        f_int = self._interaction(bundle)
        return (1 - self.rho) * f_add + self.rho * f_int

    def true_optimum(self) -> float:
        return 0.0
