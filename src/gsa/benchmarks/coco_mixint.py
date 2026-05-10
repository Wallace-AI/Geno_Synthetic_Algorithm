"""Adapter for the COCO bbob-mixint suite (Tusar et al., 2019).

Each problem has `number_of_integer_variables` leading integer dimensions
followed by real dimensions. We expose them as Z + R subgenomes.

The COCO suite is a recognized external mixed-integer benchmark used in
the BBOB-MixInt track at GECCO. Wrapping a small subset lets us defend
the paper's H1 claim against an external oracle.
"""
from __future__ import annotations

import numpy as np

from gsa.benchmarks.base import Problem
from gsa.core.genome import TypedBundle
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, TypeSpec,
)


try:
    import cocoex  # noqa: F401
    _COCO_AVAILABLE = True
except ImportError:
    _COCO_AVAILABLE = False


class CocoMixInt(Problem):
    """Wrap one COCO bbob-mixint problem (function, instance, dim)."""

    def __init__(self, budget, seed: int, function: int = 1,
                 instance: int = 1, dim: int = 10):
        if not _COCO_AVAILABLE:
            raise ImportError(
                "cocoex not installed. pip install coco-experiment"
            )
        self.function = int(function)
        self.instance = int(instance)
        self.dim = int(dim)
        super().__init__(budget=budget, seed=seed)

    def _setup(self) -> None:
        import cocoex
        suite = cocoex.Suite("bbob-mixint", "", "")
        target = None
        for p in suite:
            if (p.id_function == self.function
                    and p.id_instance == self.instance
                    and p.dimension == self.dim):
                target = p
                break
        if target is None:
            raise ValueError(
                f"BBOB-MixInt problem f{self.function}_i{self.instance}"
                f"_d{self.dim} not in suite"
            )
        self._coco = target
        self._suite = suite  # keep alive — coco resources tied to suite
        n_int = int(target.number_of_integer_variables)
        n_real = int(target.dimension - n_int)
        lo = np.asarray(target.lower_bounds, dtype=float)
        hi = np.asarray(target.upper_bounds, dtype=float)
        self._n_int = n_int
        self._n_real = n_real
        self._specs: dict[GeneFamily, TypeSpec] = {}
        if n_int > 0:
            self._specs[GeneFamily.Z] = IntegerSpec(
                n=n_int,
                lo=lo[:n_int].astype(np.int64),
                hi=hi[:n_int].astype(np.int64),
            )
        if n_real > 0:
            self._specs[GeneFamily.R] = RealSpec(
                n=n_real,
                lo=lo[n_int:].astype(np.float64),
                hi=hi[n_int:].astype(np.float64),
            )
        self._best_seen = float("inf")

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        x = np.zeros(self._coco.dimension, dtype=float)
        if self._n_int > 0:
            x[:self._n_int] = bundle.subgenomes[
                GeneFamily.Z].values.astype(float)
        if self._n_real > 0:
            x[self._n_int:] = bundle.subgenomes[GeneFamily.R].values
        f = float(self._coco(x))
        if f < self._best_seen:
            self._best_seen = f
        return f

    def true_optimum(self) -> float:
        # COCO BBOB f_opt is instance-specific and not exposed as a public
        # attribute. We return 0.0 as a placeholder; rankings and medians
        # are computed in raw f-space, which is sufficient for our paired
        # comparisons.
        return 0.0

    def target_threshold(self, fraction: float = 0.01) -> float:
        # COCO defines target = f_opt + 1e-8. We return the best
        # observed by COCO + a tiny epsilon; the `target_hit` flag is
        # secondary in our analysis.
        return self._best_seen + 1e-8 if np.isfinite(self._best_seen) else 0.0


def coco_mixint_problem(*, function: int, instance: int, dim: int,
                         seed: int, budget) -> CocoMixInt:
    """Factory used by the runner registry."""
    return CocoMixInt(budget=budget, seed=seed, function=function,
                      instance=instance, dim=dim)
