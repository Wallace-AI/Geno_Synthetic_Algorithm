"""Typed Additive benchmark per spec §3.1.

Per-family components, each weighted to unit max contribution. Planted optima
are sampled at construction time from `seed`. f(x*) = 0 exactly.

Dimension allocation: equal split of D among active families; remainder to R."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gsa.benchmarks.base import Problem
from gsa.core.genome import TypedBundle
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec, TypeSpec,
)


_FAMILY_MAP = {
    "Z": GeneFamily.Z, "R": GeneFamily.R, "B": GeneFamily.B,
    "C": GeneFamily.C, "Cx": GeneFamily.Cx, "E": GeneFamily.E,
}


def _allocate_dims(dim: int, families: Sequence[GeneFamily]) -> dict[GeneFamily, int]:
    """Equal split with remainder to R if present, else first family."""
    n = len(families)
    base = dim // n
    rem = dim - base * n
    out = {f: base for f in families}
    receiver = GeneFamily.R if GeneFamily.R in out else families[0]
    out[receiver] += rem
    return out


class TypedAdditive(Problem):
    def __init__(self, budget, seed: int, dim: int = 20,
                 families: Sequence[str] = ("Z", "R", "B", "C", "Cx", "E"),
                 embedding_dim_k: int = 8,
                 integer_lo: int = 0, integer_hi: int = 10,
                 real_lo: float = -5.0, real_hi: float = 5.0,
                 n_categories: int = 4,
                 cx_r_min: float = 0.5, cx_r_max: float = 2.0):
        self.dim = dim
        self.families = tuple(_FAMILY_MAP[f] for f in families)
        self.embedding_dim_k = embedding_dim_k
        self.integer_lo, self.integer_hi = integer_lo, integer_hi
        self.real_lo, self.real_hi = real_lo, real_hi
        self.n_categories = n_categories
        self.cx_r_min, self.cx_r_max = cx_r_min, cx_r_max
        super().__init__(budget=budget, seed=seed)

    def _setup(self) -> None:
        rng = np.random.default_rng(self.seed)
        alloc = _allocate_dims(self.dim, self.families)
        self._specs: dict[GeneFamily, TypeSpec] = {}
        self.target: dict[GeneFamily, np.ndarray] = {}

        for fam in self.families:
            n = alloc[fam]
            if fam == GeneFamily.Z:
                lo = np.full(n, self.integer_lo)
                hi = np.full(n, self.integer_hi)
                self._specs[fam] = IntegerSpec(n=n, lo=lo, hi=hi)
                self.target[fam] = rng.integers(lo, hi + 1)
            elif fam == GeneFamily.R:
                lo = np.full(n, self.real_lo)
                hi = np.full(n, self.real_hi)
                self._specs[fam] = RealSpec(n=n, lo=lo, hi=hi)
                self.target[fam] = rng.uniform(lo, hi)
            elif fam == GeneFamily.B:
                self._specs[fam] = BooleanSpec(n=n)
                self.target[fam] = rng.random(n) < 0.5
            elif fam == GeneFamily.C:
                self._specs[fam] = CategoricalSpec(
                    n=n, n_categories=[self.n_categories] * n
                )
                self.target[fam] = rng.integers(0, self.n_categories, size=n)
            elif fam == GeneFamily.Cx:
                self._specs[fam] = ComplexSpec(n=n, r_min=self.cx_r_min,
                                                r_max=self.cx_r_max)
                r = rng.uniform(self.cx_r_min, self.cx_r_max, size=n)
                phi = rng.uniform(-np.pi, np.pi, size=n)
                self.target[fam] = r * np.exp(1j * phi)
            elif fam == GeneFamily.E:
                self._specs[fam] = EmbeddingSpec(n=n, dim=self.embedding_dim_k)
                v = rng.normal(size=(n, self.embedding_dim_k))
                v = v / np.linalg.norm(v, axis=-1, keepdims=True)
                self.target[fam] = v

        # Per-family weights so each family's max contribution is unity.
        self._weights = self._compute_weights()

    def _compute_weights(self) -> dict[GeneFamily, float]:
        w = {}
        for fam, spec in self._specs.items():
            if fam == GeneFamily.Z:
                max_contrib = float(np.sum(np.maximum(
                    self.target[fam] - spec.lo,
                    spec.hi - self.target[fam],
                )))
            elif fam == GeneFamily.R:
                max_contrib = float(np.sum(np.maximum(
                    (self.target[fam] - spec.lo) ** 2,
                    (spec.hi - self.target[fam]) ** 2,
                )))
            elif fam == GeneFamily.B:
                max_contrib = float(spec.n)
            elif fam == GeneFamily.C:
                max_contrib = float(spec.n)
            elif fam == GeneFamily.Cx:
                max_contrib = float(spec.n) * (4 * spec.r_max ** 2)
            elif fam == GeneFamily.E:
                max_contrib = float(spec.n) * 2.0  # 1 - cos in [0, 2]
            w[fam] = 1.0 / max(max_contrib, 1e-12)
        return w

    @property
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        return self._specs

    def _component(self, fam: GeneFamily, vals: np.ndarray) -> float:
        t = self.target[fam]
        if fam == GeneFamily.Z:
            return float(np.sum(np.abs(vals - t)))
        if fam == GeneFamily.R:
            return float(np.sum((vals - t) ** 2))
        if fam == GeneFamily.B:
            return float(np.sum(vals.astype(bool) ^ t.astype(bool)))
        if fam == GeneFamily.C:
            return float(np.sum(vals != t))
        if fam == GeneFamily.Cx:
            return float(np.sum(np.abs(vals - t) ** 2))
        if fam == GeneFamily.E:
            return float(np.sum(1.0 - np.sum(vals * t, axis=-1)))
        raise ValueError(fam)

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        total = 0.0
        for fam, sg in bundle.subgenomes.items():
            if fam not in self._specs:
                continue
            total += self._weights[fam] * self._component(fam, sg.values)
        return total

    def true_optimum(self) -> float:
        return 0.0
