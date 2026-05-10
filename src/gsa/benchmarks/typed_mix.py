"""Typed-Mix Gradient benchmark per spec §3.5 — the headline figure source.

Two sweep axes:
    n_families ∈ {1..6}, ρ ∈ {0.0, 0.5}.

Activation order: R → R+B → R+B+Z → R+B+Z+C → R+B+Z+C+Cx → all six.

Built atop TypedEpistatic so ρ controls additive vs. interaction blend."""
from __future__ import annotations

from gsa.benchmarks.typed_epistatic import TypedEpistatic
from gsa.core.types import GeneFamily


ACTIVATION_ORDER = (
    GeneFamily.R, GeneFamily.B, GeneFamily.Z,
    GeneFamily.C, GeneFamily.Cx, GeneFamily.E,
)
_FAM_LETTER = {GeneFamily.R: "R", GeneFamily.B: "B", GeneFamily.Z: "Z",
               GeneFamily.C: "C", GeneFamily.Cx: "Cx", GeneFamily.E: "E"}


class TypedMix(TypedEpistatic):
    def __init__(self, budget, seed: int, dim: int = 20,
                 n_families: int = 6, rho: float = 0.0, **kwargs):
        if not 1 <= n_families <= 6:
            raise ValueError("n_families must be in 1..6")
        active = ACTIVATION_ORDER[:n_families]
        family_letters = tuple(_FAM_LETTER[f] for f in active)
        self.n_families = n_families
        super().__init__(budget=budget, seed=seed, dim=dim,
                         families=family_letters, rho=rho, **kwargs)
