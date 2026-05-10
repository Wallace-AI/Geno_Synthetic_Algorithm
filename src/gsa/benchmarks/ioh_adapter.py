"""IOH Boolean adapter with local fallbacks per spec §3.6."""
from __future__ import annotations

import numpy as np

from gsa.benchmarks.base import Problem
from gsa.core.genome import TypedBundle
from gsa.core.types import BooleanSpec, GeneFamily, TypeSpec


try:
    import ioh  # noqa
    _IOH_AVAILABLE = True
except ImportError:
    _IOH_AVAILABLE = False


class OneMaxLocal(Problem):
    def __init__(self, budget, seed: int, n: int = 50):
        self.n = n
        super().__init__(budget=budget, seed=seed)

    def _setup(self):
        self._specs = {GeneFamily.B: BooleanSpec(n=self.n)}

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        bits = bundle.subgenomes[GeneFamily.B].values
        return float(self.n - int(bits.sum()))

    def true_optimum(self) -> float:
        return 0.0


class LeadingOnesLocal(Problem):
    def __init__(self, budget, seed: int, n: int = 50):
        self.n = n
        super().__init__(budget=budget, seed=seed)

    def _setup(self):
        self._specs = {GeneFamily.B: BooleanSpec(n=self.n)}

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        bits = bundle.subgenomes[GeneFamily.B].values.astype(bool)
        leading = 0
        for b in bits:
            if b:
                leading += 1
            else:
                break
        return float(self.n - leading)

    def true_optimum(self) -> float:
        return 0.0


class WModelOneMaxLocal(Problem):
    """W-model OneMax with dummy-variable layer."""

    def __init__(self, budget, seed: int, n: int = 50,
                 dummy_fraction: float = 0.5,
                 epistasis: bool = False):
        self.n = n
        self.dummy_fraction = dummy_fraction
        self.epistasis = epistasis
        super().__init__(budget=budget, seed=seed)

    def _setup(self):
        rng = np.random.default_rng(self.seed)
        self._specs = {GeneFamily.B: BooleanSpec(n=self.n)}
        n_dummy = int(self.n * self.dummy_fraction)
        idx = rng.permutation(self.n)
        self._dummy_idx = idx[:n_dummy]
        self._effective_idx = idx[n_dummy:]

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        bits = bundle.subgenomes[GeneFamily.B].values.astype(bool)
        eff = bits[self._effective_idx]
        score = int(eff.sum())
        if self.epistasis:
            penalty = 0
            for i in range(0, len(eff) - 1, 2):
                if eff[i] != eff[i + 1]:
                    penalty += 1
            score -= penalty
        return float(len(eff) - score)

    def true_optimum(self) -> float:
        return 0.0


def ioh_problem(name: str, *, n: int, seed: int, budget, **kwargs) -> Problem:
    """Dispatch to ioh-wrapped or local-fallback implementation.

    Currently routes to local fallbacks even when ioh is installed for
    deterministic semantics per spec §5.6."""
    name_lower = name.lower()
    if name_lower == "onemax":
        return OneMaxLocal(budget=budget, seed=seed, n=n)
    if name_lower == "leadingones":
        return LeadingOnesLocal(budget=budget, seed=seed, n=n)
    if name_lower == "wmodel_onemax":
        return WModelOneMaxLocal(budget=budget, seed=seed, n=n, **kwargs)
    if name_lower == "wmodel_onemax_epistasis":
        return WModelOneMaxLocal(budget=budget, seed=seed, n=n,
                                 epistasis=True, **kwargs)
    raise ValueError(f"unknown IOH problem: {name}")
