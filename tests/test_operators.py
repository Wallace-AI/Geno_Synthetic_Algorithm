import numpy as np
import pytest

from gsa.core.operators import (
    integer_operator, real_operator_de, boolean_operator,
    categorical_operator, complex_operator, embedding_operator,
    OperatorContext,
)
from gsa.core.types import (
    IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec,
)


def make_rng(seed=0):
    return np.random.default_rng(seed)


# Integer operator
def test_integer_operator_respects_bounds():
    spec = IntegerSpec(n=3, lo=np.array([0, 0, 0]), hi=np.array([5, 5, 5]))
    rng = make_rng(0)
    parent = np.array([3, 3, 3])
    for _ in range(20):
        child = integer_operator(parent, spec, rng=rng)
        assert (child >= spec.lo).all()
        assert (child <= spec.hi).all()
        assert child.dtype.kind == "i"


def test_integer_operator_mutates_some_coords():
    spec = IntegerSpec(n=10, lo=np.zeros(10, dtype=int), hi=np.ones(10, dtype=int) * 100)
    rng = make_rng(0)
    parent = np.full(10, 50)
    different = 0
    for _ in range(50):
        child = integer_operator(parent, spec, rng=rng)
        if not np.array_equal(child, parent):
            different += 1
    assert different > 0


# Real operator (DE/rand/1/bin)
def test_real_operator_de_uses_three_donors():
    spec = RealSpec(n=2, lo=-np.ones(2) * 5, hi=np.ones(2) * 5)
    rng = make_rng(0)
    target = np.zeros(2)
    donors = [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    ctx = OperatorContext(donors=donors)
    child = real_operator_de(target, spec, rng=rng, ctx=ctx, F=0.5, CR=0.9)
    assert child.shape == (2,)
    assert (child >= spec.lo).all()
    assert (child <= spec.hi).all()


def test_real_operator_de_falls_back_with_few_donors():
    spec = RealSpec(n=2, lo=-np.ones(2), hi=np.ones(2))
    rng = make_rng(0)
    target = np.zeros(2)
    ctx = OperatorContext(donors=[])  # no donors -> must fallback to Gaussian
    child = real_operator_de(target, spec, rng=rng, ctx=ctx, F=0.5, CR=0.9)
    assert child.shape == (2,)


# Boolean operator
def test_boolean_operator_flips_with_prob():
    spec = BooleanSpec(n=20)
    rng = make_rng(0)
    parent = np.zeros(20, dtype=bool)
    flipped_total = 0
    for _ in range(100):
        child = boolean_operator(parent, spec, rng=rng, p=0.5)
        flipped_total += int(child.sum())
    # With p=0.5 over 100 trials × 20 bits, expect ~1000 flips ± few sigma
    assert 800 < flipped_total < 1200


def test_boolean_operator_no_flip_when_p_zero():
    spec = BooleanSpec(n=10)
    rng = make_rng(0)
    parent = np.zeros(10, dtype=bool)
    child = boolean_operator(parent, spec, rng=rng, p=0.0)
    assert not child.any()


# Categorical operator
def test_categorical_operator_replaces_in_admissible_set():
    spec = CategoricalSpec(n=3, n_categories=[3, 4, 2])
    rng = make_rng(0)
    parent = np.array([0, 0, 0])
    for _ in range(50):
        child = categorical_operator(parent, spec, rng=rng, p=0.5)
        assert (child[0] >= 0) and (child[0] < 3)
        assert (child[1] >= 0) and (child[1] < 4)
        assert (child[2] >= 0) and (child[2] < 2)


# Complex operator
def test_complex_operator_respects_magnitude_bounds():
    spec = ComplexSpec(n=3, r_min=0.5, r_max=2.0)
    rng = make_rng(0)
    parent = np.array([1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j])
    for _ in range(30):
        child = complex_operator(parent, spec, rng=rng,
                                 sigma_r=0.1, sigma_phi=np.pi / 8)
        mags = np.abs(child)
        assert (mags >= spec.r_min - 1e-9).all()
        assert (mags <= spec.r_max + 1e-9).all()


# Embedding operator
def test_embedding_operator_preserves_unit_norm():
    spec = EmbeddingSpec(n=2, dim=8)
    rng = make_rng(0)
    parent = rng.normal(size=(2, 8))
    parent /= np.linalg.norm(parent, axis=-1, keepdims=True)
    for _ in range(30):
        child = embedding_operator(parent, spec, rng=rng, sigma=0.1, tau=0.7)
        norms = np.linalg.norm(child, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)


def test_embedding_operator_cosine_rejection_keeps_neighborhood():
    spec = EmbeddingSpec(n=1, dim=8)
    rng = make_rng(0)
    parent = np.array([[1.0] + [0.0] * 7])
    parent /= np.linalg.norm(parent, axis=-1, keepdims=True)
    for _ in range(30):
        child = embedding_operator(parent, spec, rng=rng, sigma=0.05, tau=0.95)
        cos = float(np.sum(parent[0] * child[0]))
        # tau=0.95 means cosine to parent should be high
        assert cos >= 0.85
