import numpy as np
from gsa.core.populations import TypedPopulation, sample_initial_subgenome
from gsa.core.types import GeneFamily, RealSpec, BooleanSpec


def test_sample_initial_real_in_bounds():
    rng = np.random.default_rng(0)
    spec = RealSpec(n=4, lo=-np.ones(4), hi=np.ones(4))
    sg = sample_initial_subgenome(spec, rng=rng)
    assert sg.family == GeneFamily.R
    assert sg.values.shape == (4,)
    assert (sg.values >= spec.lo).all()
    assert (sg.values <= spec.hi).all()


def test_sample_initial_boolean_random():
    rng = np.random.default_rng(0)
    spec = BooleanSpec(n=10)
    sg = sample_initial_subgenome(spec, rng=rng)
    assert sg.values.dtype == bool
    assert sg.values.shape == (10,)


def test_typed_population_initial_size():
    rng = np.random.default_rng(0)
    spec = RealSpec(n=2, lo=np.zeros(2), hi=np.ones(2))
    pop = TypedPopulation(spec=spec, size=20, rng=rng)
    pop.sample_initial()
    assert len(pop.individuals) == 20
    for sg in pop.individuals:
        assert sg.values.shape == (2,)


def test_typed_population_diversity_metric():
    rng = np.random.default_rng(0)
    spec = RealSpec(n=2, lo=-np.ones(2), hi=np.ones(2))
    pop = TypedPopulation(spec=spec, size=10, rng=rng)
    pop.sample_initial()
    div = pop.diversity()
    assert div >= 0.0
