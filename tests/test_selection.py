import numpy as np

from gsa.core.selection import tournament_select, diversity_regularized_select


def test_tournament_picks_best_in_random_subset():
    rng = np.random.default_rng(0)
    fitness = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    # Run many tournaments; index 1 (lowest fitness) should win majority
    wins = np.zeros(5, dtype=int)
    for _ in range(500):
        idx = tournament_select(fitness, k=3, rng=rng, minimize=True)
        wins[idx] += 1
    assert wins[1] > wins[0]
    assert wins[1] > wins[2]


def test_diversity_regularized_blends_fitness_and_diversity():
    rng = np.random.default_rng(0)
    fitness = np.array([1.0, 1.0, 1.0])
    diversity = np.array([0.0, 1.0, 0.5])
    # alpha=0.0 -> diversity dominates; index 1 (most diverse) wins
    wins_div = np.zeros(3, dtype=int)
    for _ in range(500):
        idx = diversity_regularized_select(fitness, diversity, k=3,
                                            rng=rng, alpha=0.0, minimize=True)
        wins_div[idx] += 1
    assert wins_div[1] > wins_div[2]


def test_alpha_one_collapses_to_tournament():
    rng = np.random.default_rng(0)
    fitness = np.array([5.0, 1.0, 4.0])
    diversity = np.array([1.0, 0.0, 0.5])
    wins = np.zeros(3, dtype=int)
    for _ in range(500):
        idx = diversity_regularized_select(fitness, diversity, k=3,
                                            rng=rng, alpha=1.0, minimize=True)
        wins[idx] += 1
    assert wins[1] == 500  # always picks lowest-fitness regardless of diversity
