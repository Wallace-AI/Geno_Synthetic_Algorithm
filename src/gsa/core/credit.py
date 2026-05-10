"""Four credit-assignment modes per spec §2.4.

Common interface:
    assign(eb: EvaluatedBundle, partner_pool, problem, rng) -> dict[family, credit]

Budget accounting: the Problem ABC counts each evaluate() call as 1 budget unit.
- DirectCredit: 0 extra evaluations beyond the assembled fitness already in eb.
- EliteCredit: 1 extra per family (paired with elites of other families).
- EnsembleCredit: K per family.
- MarginalCredit: 1 extra per family (replacement with neutral default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec, TypeSpec,
)


@dataclass
class EvaluatedBundle:
    bundle: TypedBundle
    fitness: float


def _neutral_subgenome(spec: TypeSpec, family: GeneFamily) -> TypedSubgenome:
    """Family-specific neutral default per spec §2.4 marginal credit."""
    if isinstance(spec, IntegerSpec):
        vals = np.clip(np.zeros(spec.n, dtype=np.int64), spec.lo, spec.hi)
        return TypedSubgenome(family, vals, spec)
    if isinstance(spec, RealSpec):
        vals = np.zeros(spec.n)
        return TypedSubgenome(family, vals, spec)
    if isinstance(spec, BooleanSpec):
        return TypedSubgenome(family, np.zeros(spec.n, dtype=bool), spec)
    if isinstance(spec, CategoricalSpec):
        return TypedSubgenome(family, np.zeros(spec.n, dtype=np.int64), spec)
    if isinstance(spec, ComplexSpec):
        vals = np.ones(spec.n, dtype=np.complex128) * (spec.r_min + spec.r_max) / 2.0
        return TypedSubgenome(family, vals, spec)
    if isinstance(spec, EmbeddingSpec):
        v = np.zeros((spec.n, spec.dim))
        v[:, 0] = 1.0  # canonical first-axis unit vector as "mean direction"
        return TypedSubgenome(family, v, spec)
    raise TypeError(f"unknown spec: {type(spec)}")


class CreditAssigner:
    """Common interface.

    `target_family`, if set, restricts credit computation to that single
    family — used by the optimizer's per-family inner loop to avoid scoring
    credits that won't be consumed (saves K-1× extra evaluations per call
    on multi-family bundles).
    """

    def assign(self, eb: EvaluatedBundle, partner_pool: Any,
               problem: Any, rng: Optional[np.random.Generator],
               target_family: Optional[GeneFamily] = None,
               ) -> dict[GeneFamily, float]:
        raise NotImplementedError


def _families_to_score(eb: EvaluatedBundle,
                       target_family: Optional[GeneFamily]) -> list[GeneFamily]:
    if target_family is not None:
        return [target_family]
    return list(eb.bundle.subgenomes.keys())


@dataclass
class DirectCredit(CreditAssigner):
    """Every participating subgenome receives the assembled fitness."""

    def assign(self, eb, partner_pool, problem, rng, target_family=None):
        return {fam: eb.fitness for fam in _families_to_score(eb, target_family)}


@dataclass
class EliteCredit(CreditAssigner):
    """Pair the subgenome with elite reps from every other family."""

    def assign(self, eb, partner_pool, problem, rng, target_family=None):
        out = {}
        for fam in _families_to_score(eb, target_family):
            sg = eb.bundle.subgenomes[fam]
            other_fams = [f for f in eb.bundle.subgenomes if f != fam]
            if not other_fams:
                out[fam] = float(eb.fitness)
                continue
            partners = TypedBundle({
                f: (sg if f == fam else partner_pool[f])
                for f in eb.bundle.subgenomes
            })
            out[fam] = problem.evaluate(partners)
        return out


@dataclass
class EnsembleCredit(CreditAssigner):
    K: int = 5

    def assign(self, eb, partner_pool, problem, rng, target_family=None):
        out = {}
        for fam in _families_to_score(eb, target_family):
            sg = eb.bundle.subgenomes[fam]
            other_fams = [f for f in eb.bundle.subgenomes if f != fam]
            if not other_fams:
                out[fam] = float(eb.fitness)
                continue
            assert rng is not None
            scores = []
            for _ in range(self.K):
                comb = {fam: sg}
                for other_fam in other_fams:
                    pool = partner_pool[other_fam]
                    j = int(rng.integers(0, len(pool)))
                    comb[other_fam] = pool[j]
                scores.append(problem.evaluate(TypedBundle(comb)))
            out[fam] = float(np.mean(scores))
        return out


@dataclass
class MarginalCredit(CreditAssigner):
    """Marginal contribution: f(bundle) - f(bundle with subgenome -> neutral)."""

    def assign(self, eb, partner_pool, problem, rng, target_family=None):
        out = {}
        for fam in _families_to_score(eb, target_family):
            sg = eb.bundle.subgenomes[fam]
            neutral = _neutral_subgenome(sg.spec, fam)
            replaced = TypedBundle({
                f: (neutral if f == fam else other_sg)
                for f, other_sg in eb.bundle.subgenomes.items()
            })
            f_neutral = problem.evaluate(replaced)
            out[fam] = float(eb.fitness - f_neutral)
        return out
