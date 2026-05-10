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
    """Common interface."""

    def assign(self, eb: EvaluatedBundle, partner_pool: Any,
               problem: Any, rng: Optional[np.random.Generator]) -> dict[GeneFamily, float]:
        raise NotImplementedError


@dataclass
class DirectCredit(CreditAssigner):
    """Every participating subgenome receives the assembled fitness."""

    def assign(self, eb, partner_pool, problem, rng):
        return {fam: eb.fitness for fam in eb.bundle.subgenomes}


@dataclass
class EliteCredit(CreditAssigner):
    """Pair the subgenome with elite reps from every other family."""

    def assign(self, eb, partner_pool, problem, rng):
        # partner_pool is dict[family, TypedSubgenome] of elites
        out = {}
        for fam, sg in eb.bundle.subgenomes.items():
            other_fams = [f for f in eb.bundle.subgenomes if f != fam]
            if not other_fams:
                # Single-family bundle: elite-pairing is degenerate, fall back
                # to assembled fitness without burning an extra evaluation.
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

    def assign(self, eb, partner_pool, problem, rng):
        # partner_pool is dict[family, list[TypedSubgenome]] sampling pool
        out = {}
        for fam, sg in eb.bundle.subgenomes.items():
            other_fams = [f for f in eb.bundle.subgenomes if f != fam]
            if not other_fams:
                # K identical evals on a single-family bundle = waste; reuse
                # the assembled fitness already in eb.
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

    def assign(self, eb, partner_pool, problem, rng):
        out = {}
        for fam, sg in eb.bundle.subgenomes.items():
            neutral = _neutral_subgenome(sg.spec, fam)
            replaced = TypedBundle({
                f: (neutral if f == fam else other_sg)
                for f, other_sg in eb.bundle.subgenomes.items()
            })
            f_neutral = problem.evaluate(replaced)
            out[fam] = float(eb.fitness - f_neutral)
        return out
