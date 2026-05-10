"""Shared flattened-encoding utilities for baselines that operate on a
single float vector (FLATTENED_DE, FLATTENED_EA, GSA_GENERIC_OPERATORS).

Encoding policy:
  Z: stored as float; on decode round and clip to [lo, hi]
  R: stored as float; on decode clip to [lo, hi]
  B: stored as float; on decode threshold at 0.5
  C: stored as float; on decode round and clip to [0, n_cat-1]
  Cx: NOT supported (out of scope for flattened baselines per spec)
  E: NOT supported (out of scope for flattened baselines per spec)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec, TypeSpec,
)


@dataclass
class FlatLayout:
    slices: dict[GeneFamily, slice]
    total_dim: int


def flatten_specs(specs: dict[GeneFamily, TypeSpec]) -> FlatLayout:
    """Compute slice offsets for each family in the flat vector."""
    slices = {}
    offset = 0
    for fam in (GeneFamily.R, GeneFamily.Z, GeneFamily.B, GeneFamily.C,
                GeneFamily.Cx, GeneFamily.E):
        if fam not in specs:
            continue
        if fam in (GeneFamily.Cx, GeneFamily.E):
            raise ValueError(
                f"Family {fam} not supported in flattened encoding "
                f"(spec §4.2 baselines are limited to Z/R/B/C)"
            )
        n = specs[fam].n
        slices[fam] = slice(offset, offset + n)
        offset += n
    return FlatLayout(slices=slices, total_dim=offset)


def encode_bundle(bundle: TypedBundle, layout: FlatLayout) -> np.ndarray:
    flat = np.zeros(layout.total_dim)
    for fam, sl in layout.slices.items():
        sg = bundle.subgenomes[fam]
        flat[sl] = sg.values.astype(float)
    return flat


def decode_to_bundle(flat: np.ndarray, specs: dict[GeneFamily, TypeSpec],
                     layout: FlatLayout) -> TypedBundle:
    sub = {}
    for fam, sl in layout.slices.items():
        spec = specs[fam]
        chunk = flat[sl]
        if isinstance(spec, RealSpec):
            v = np.clip(chunk, spec.lo, spec.hi)
            sub[fam] = TypedSubgenome(fam, v.astype(np.float64), spec)
        elif isinstance(spec, IntegerSpec):
            v = np.clip(np.round(chunk), spec.lo, spec.hi).astype(np.int64)
            sub[fam] = TypedSubgenome(fam, v, spec)
        elif isinstance(spec, BooleanSpec):
            v = (chunk > 0.5)
            sub[fam] = TypedSubgenome(fam, v, spec)
        elif isinstance(spec, CategoricalSpec):
            nc = np.array(spec.n_categories)
            v = np.clip(np.round(chunk), 0, nc - 1).astype(np.int64)
            sub[fam] = TypedSubgenome(fam, v, spec)
        else:
            raise TypeError(f"unhandled spec: {type(spec)}")
    return TypedBundle(sub)


def random_flat(specs: dict[GeneFamily, TypeSpec], layout: FlatLayout,
                rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform random flat vector matching the layout."""
    flat = np.zeros(layout.total_dim)
    for fam, sl in layout.slices.items():
        spec = specs[fam]
        if isinstance(spec, RealSpec):
            flat[sl] = rng.uniform(spec.lo, spec.hi)
        elif isinstance(spec, IntegerSpec):
            flat[sl] = rng.uniform(spec.lo - 0.5, spec.hi + 0.5)
        elif isinstance(spec, BooleanSpec):
            flat[sl] = rng.random(spec.n)
        elif isinstance(spec, CategoricalSpec):
            for j in range(spec.n):
                flat[sl][j] = rng.uniform(-0.5, spec.n_categories[j] - 0.5)
    return flat
