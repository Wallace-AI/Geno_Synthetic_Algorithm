"""Per-family distance functions and population diversity."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from gsa.core.types import GeneFamily


def distance_Z(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def distance_R(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def distance_B(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a != b))


def distance_C(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return 1.0 - float(np.mean(a == b))


def distance_Cx(a: np.ndarray, b: np.ndarray,
                weight_r: float = 0.5, weight_phi: float = 0.5) -> float:
    """Weighted L2 over (Δr, Δφ).

    Δr is magnitude difference; Δφ is the wrapped-phase difference normalized
    by π so it is on the same scale as Δr."""
    dr = np.abs(a) - np.abs(b)
    phi_a = np.angle(a)
    phi_b = np.angle(b)
    dphi = np.arctan2(np.sin(phi_a - phi_b), np.cos(phi_a - phi_b)) / np.pi
    return float(np.sqrt(weight_r * np.sum(dr ** 2) + weight_phi * np.sum(dphi ** 2)))


def distance_E(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row (1 - cosine similarity).

    a, b shape: (n, dim) with unit-norm rows."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    cos = np.sum(a * b, axis=-1)
    return float(np.mean(1.0 - cos))


_DISPATCH = {
    GeneFamily.Z: distance_Z,
    GeneFamily.R: distance_R,
    GeneFamily.B: distance_B,
    GeneFamily.C: distance_C,
    GeneFamily.Cx: distance_Cx,
    GeneFamily.E: distance_E,
}


def population_diversity(pop: list[np.ndarray], family: GeneFamily) -> float:
    """Mean pairwise distance across the population."""
    if len(pop) < 2:
        return 0.0
    fn = _DISPATCH[family]
    pairs = list(combinations(pop, 2))
    return float(np.mean([fn(a, b) for a, b in pairs]))
