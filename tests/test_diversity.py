import numpy as np
from gsa.core.diversity import (
    distance_Z, distance_R, distance_B, distance_C, distance_Cx, distance_E,
    population_diversity,
)
from gsa.core.types import GeneFamily


def test_distance_Z_l1():
    assert distance_Z(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0
    assert distance_Z(np.array([1, 0, 0]), np.array([0, 0, 0])) == 1
    assert distance_Z(np.array([3, 4]), np.array([0, 0])) == 7


def test_distance_R_l2():
    assert distance_R(np.array([0.0]), np.array([0.0])) == 0
    assert abs(distance_R(np.array([3.0, 4.0]), np.array([0.0, 0.0])) - 5.0) < 1e-9


def test_distance_B_hamming():
    assert distance_B(np.array([True, False, True]),
                      np.array([True, False, True])) == 0
    assert distance_B(np.array([True, False]),
                      np.array([False, True])) == 2


def test_distance_C_one_minus_match():
    assert distance_C(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])) == 0
    assert abs(distance_C(np.array([0, 1, 2, 3]),
                           np.array([1, 1, 2, 3])) - 0.25) < 1e-9


def test_distance_Cx_weighted():
    a = np.array([1+0j, 0+1j])
    b = np.array([1+0j, 0+1j])
    assert distance_Cx(a, b) == 0
    assert distance_Cx(np.array([0+0j]), np.array([1+0j])) > 0


def test_distance_E_one_minus_cosine():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(distance_E(a, a)) < 1e-9
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    # both rows orthogonal -> cosine 0 -> distance 1 each -> mean 1
    assert abs(distance_E(a, b) - 1.0) < 1e-9


def test_population_diversity_mean_pairwise():
    # 3 individuals, real, mean pairwise L2
    pop = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
    div = population_diversity(pop, GeneFamily.R)
    # pairwise: 1, 2, 1 -> mean = 4/3
    assert abs(div - 4 / 3) < 1e-9
