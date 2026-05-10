"""Type-native variation operators, one per gene family.

Each operator signature:
    operator(parent: np.ndarray, spec: TypeSpec, *, rng: np.random.Generator,
             ctx: OperatorContext | None, **hyperparams) -> np.ndarray

OperatorContext supplies cross-population information needed by some operators
(e.g., DE needs three donor vectors)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gsa.core.types import (
    IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec,
)


@dataclass
class OperatorContext:
    """Information operators may need beyond the parent.

    For DE: three donor vectors. For mask recombination: a partner. For
    fitness-weighted categorical sampling: per-category fitness scores.
    """
    donors: list[np.ndarray] = field(default_factory=list)
    partner: Optional[np.ndarray] = None
    category_weights: Optional[list[np.ndarray]] = None


def integer_operator(parent: np.ndarray, spec: IntegerSpec, *,
                     rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                     scale_frac: float = 0.1) -> np.ndarray:
    """Bounded integer random walk with discrete-Laplace step."""
    ranges = np.maximum(1, spec.hi - spec.lo)
    scales = np.maximum(1.0, scale_frac * ranges).astype(float)
    # Discrete Laplace: difference of two geometric distributions
    p = 1.0 - np.exp(-1.0 / scales)
    n = len(parent)
    g1 = rng.geometric(p, size=n) - 1
    g2 = rng.geometric(p, size=n) - 1
    delta = g1 - g2
    child = parent + delta
    child = np.clip(child, spec.lo, spec.hi).astype(np.int64)
    return child


def real_operator_de(parent: np.ndarray, spec: RealSpec, *,
                     rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                     F: float = 0.5, CR: float = 0.9) -> np.ndarray:
    """DE/rand/1/bin: v = x_r1 + F*(x_r2 - x_r3); binomial xover with parent."""
    ranges = spec.hi - spec.lo
    if ctx is None or len(ctx.donors) < 3:
        # Fallback: Gaussian mutation when no DE donors available
        sigma = 0.1 * ranges
        v = parent + rng.normal(scale=sigma)
    else:
        r1, r2, r3 = ctx.donors[:3]
        v = r1 + F * (r2 - r3)
    # Binomial crossover with parent
    mask = rng.random(parent.shape) < CR
    j_rand = rng.integers(0, len(parent))
    mask[j_rand] = True
    child = np.where(mask, v, parent)
    # Bound reflection
    below = child < spec.lo
    above = child > spec.hi
    child = np.where(below, 2 * spec.lo - child, child)
    child = np.where(above, 2 * spec.hi - child, child)
    child = np.clip(child, spec.lo, spec.hi)
    return child.astype(np.float64)


def boolean_operator(parent: np.ndarray, spec: BooleanSpec, *,
                     rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                     p: Optional[float] = None) -> np.ndarray:
    """Bit-flip with optional mask recombination from ctx.partner."""
    if p is None:
        p = 1.0 / max(1, spec.n)
    flip = rng.random(parent.shape) < p
    child = parent.astype(bool) ^ flip
    if ctx is not None and ctx.partner is not None:
        recomb_mask = rng.random(parent.shape) < 0.5
        child = np.where(recomb_mask, ctx.partner.astype(bool), child)
    return child.astype(bool)


def categorical_operator(parent: np.ndarray, spec: CategoricalSpec, *,
                         rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                         p: Optional[float] = None) -> np.ndarray:
    """Per-coord replacement from admissible set."""
    if p is None:
        p = 1.0 / max(1, spec.n)
    child = parent.astype(np.int64).copy()
    for j in range(spec.n):
        if rng.random() < p:
            child[j] = rng.integers(0, spec.n_categories[j])
    return child


def complex_operator(parent: np.ndarray, spec: ComplexSpec, *,
                     rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                     sigma_r: float = 0.1, sigma_phi: float = np.pi / 8) -> np.ndarray:
    """Magnitude-phase mutation."""
    r = np.abs(parent)
    phi = np.angle(parent)
    eta_r = rng.normal(scale=sigma_r * (spec.r_max - spec.r_min), size=parent.shape)
    eta_phi = rng.normal(scale=sigma_phi, size=parent.shape)
    r_new = np.clip(r + eta_r, spec.r_min, spec.r_max)
    phi_new = np.arctan2(np.sin(phi + eta_phi), np.cos(phi + eta_phi))
    return r_new * np.exp(1j * phi_new)


def embedding_operator(parent: np.ndarray, spec: EmbeddingSpec, *,
                       rng: np.random.Generator, ctx: Optional[OperatorContext] = None,
                       sigma: float = 0.1, tau: float = 0.7,
                       max_attempts: int = 5) -> np.ndarray:
    """Norm-preserving Gaussian + cosine-distance rejection."""
    parent = parent / (np.linalg.norm(parent, axis=-1, keepdims=True) + 1e-12)
    for _ in range(max_attempts):
        z = rng.normal(size=parent.shape)
        v = parent + sigma * z
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
        cos = np.sum(parent * v, axis=-1)
        if (cos >= tau).all():
            return v
    return v  # accept last attempt even if rejected


def generic_operator(parent: np.ndarray, spec, *,
                     rng: np.random.Generator,
                     ctx: Optional[OperatorContext] = None,
                     sigma_frac: float = 0.1) -> np.ndarray:
    """Generic Gaussian mutation followed by family-appropriate decoder.

    Used by GSA_GENERIC_OPERATORS variant. Treats EVERY family as if it were
    real, mutates with Gaussian, then decodes back. This is intentionally
    representation-naive (the brief's flattened-baseline operator)."""
    if isinstance(spec, RealSpec):
        ranges = spec.hi - spec.lo
        v = parent + rng.normal(scale=sigma_frac * ranges)
        return np.clip(v, spec.lo, spec.hi).astype(np.float64)
    if isinstance(spec, IntegerSpec):
        ranges = spec.hi - spec.lo
        v = parent.astype(float) + rng.normal(scale=sigma_frac * ranges)
        return np.clip(np.round(v), spec.lo, spec.hi).astype(np.int64)
    if isinstance(spec, BooleanSpec):
        v = parent.astype(float) + rng.normal(scale=sigma_frac, size=parent.shape)
        return (v > 0.5).astype(bool)
    if isinstance(spec, CategoricalSpec):
        v = parent.astype(float) + rng.normal(scale=sigma_frac, size=parent.shape)
        nc = np.array(spec.n_categories)
        return np.clip(np.round(v), 0, nc - 1).astype(np.int64)
    if isinstance(spec, ComplexSpec):
        re = parent.real + rng.normal(scale=sigma_frac, size=parent.shape)
        im = parent.imag + rng.normal(scale=sigma_frac, size=parent.shape)
        out = re + 1j * im
        # Renormalize magnitudes into bounds
        r = np.abs(out)
        scale = np.where((r > 0) & (r > spec.r_max), spec.r_max / r, 1.0)
        scale = np.where(r < spec.r_min, spec.r_min / np.maximum(r, 1e-12), scale)
        return out * scale
    if isinstance(spec, EmbeddingSpec):
        v = parent + rng.normal(scale=sigma_frac, size=parent.shape)
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    raise TypeError(f"unknown spec type: {type(spec)}")
