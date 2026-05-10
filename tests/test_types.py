import numpy as np
import pytest
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec, TypeSpec,
)


def test_gene_family_enum_has_six_members():
    assert {f.name for f in GeneFamily} == {"Z", "R", "B", "C", "Cx", "E"}


def test_integer_spec_validates_bounds():
    s = IntegerSpec(n=3, lo=np.array([0, 0, 0]), hi=np.array([10, 10, 10]))
    assert s.family == GeneFamily.Z
    assert s.n == 3
    with pytest.raises(ValueError):
        IntegerSpec(n=3, lo=np.array([5, 5, 5]), hi=np.array([1, 1, 1]))  # lo > hi


def test_real_spec_validates_bounds():
    s = RealSpec(n=2, lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]))
    assert s.family == GeneFamily.R


def test_boolean_spec():
    s = BooleanSpec(n=5)
    assert s.family == GeneFamily.B


def test_categorical_spec_admissible_sets():
    s = CategoricalSpec(n=2, n_categories=[3, 4])
    assert s.family == GeneFamily.C
    assert s.n_categories == [3, 4]


def test_complex_spec():
    s = ComplexSpec(n=2, r_min=0.1, r_max=2.0)
    assert s.family == GeneFamily.Cx


def test_embedding_spec():
    s = EmbeddingSpec(n=3, dim=8)
    assert s.family == GeneFamily.E
    assert s.dim == 8
