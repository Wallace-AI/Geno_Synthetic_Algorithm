"""Per-family typed population: stores individuals, fitness, supports diversity calc."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gsa.core.diversity import population_diversity
from gsa.core.genome import TypedSubgenome
from gsa.core.types import (
    GeneFamily, IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
    ComplexSpec, EmbeddingSpec, TypeSpec,
)


def sample_initial_subgenome(spec: TypeSpec, rng: np.random.Generator) -> TypedSubgenome:
    """Uniform random sample from the spec's admissible domain."""
    if isinstance(spec, IntegerSpec):
        vals = rng.integers(spec.lo, spec.hi + 1)
        return TypedSubgenome(GeneFamily.Z, vals.astype(np.int64), spec)
    if isinstance(spec, RealSpec):
        vals = rng.uniform(spec.lo, spec.hi)
        return TypedSubgenome(GeneFamily.R, vals.astype(np.float64), spec)
    if isinstance(spec, BooleanSpec):
        vals = rng.random(spec.n) < 0.5
        return TypedSubgenome(GeneFamily.B, vals, spec)
    if isinstance(spec, CategoricalSpec):
        vals = np.array([rng.integers(0, k) for k in spec.n_categories],
                        dtype=np.int64)
        return TypedSubgenome(GeneFamily.C, vals, spec)
    if isinstance(spec, ComplexSpec):
        r = rng.uniform(spec.r_min, spec.r_max, size=spec.n)
        phi = rng.uniform(-np.pi, np.pi, size=spec.n)
        vals = r * np.exp(1j * phi)
        return TypedSubgenome(GeneFamily.Cx, vals, spec)
    if isinstance(spec, EmbeddingSpec):
        v = rng.normal(size=(spec.n, spec.dim))
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
        return TypedSubgenome(GeneFamily.E, v, spec)
    raise TypeError(f"Unknown spec: {type(spec)}")


@dataclass
class TypedPopulation:
    spec: TypeSpec
    size: int
    rng: np.random.Generator
    individuals: list[TypedSubgenome] = field(default_factory=list)
    fitness: Optional[np.ndarray] = None  # shape (size,)

    def sample_initial(self) -> None:
        self.individuals = [
            sample_initial_subgenome(self.spec, self.rng) for _ in range(self.size)
        ]
        self.fitness = np.full(self.size, np.inf)

    def values(self) -> list[np.ndarray]:
        return [sg.values for sg in self.individuals]

    def diversity(self) -> float:
        return population_diversity(self.values(), self.spec.family)

    def best_index(self) -> int:
        if self.fitness is None:
            return 0
        return int(np.argmin(self.fitness))

    def best(self) -> TypedSubgenome:
        return self.individuals[self.best_index()]

    def replace(self, idx: int, new: TypedSubgenome, fitness: float) -> None:
        self.individuals[idx] = new
        if self.fitness is None:
            self.fitness = np.full(self.size, np.inf)
        self.fitness[idx] = fitness
