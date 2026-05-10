"""Problem ABC for all benchmarks.

Each problem exposes:
    .specs: dict[GeneFamily, TypeSpec]
    .evaluate(bundle) -> float (consumes 1 budget unit)
    .true_optimum() -> float (the planted optimum value)
    .target_threshold() -> float (the ε-target for "hit" reporting)
    .true_evaluate(bundle) -> float (zero-noise evaluation; for noisy benchmarks)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from gsa.core.genome import TypedBundle
from gsa.core.types import GeneFamily, TypeSpec
from gsa.experiments.budget import EvaluationBudget


class Problem(ABC):
    def __init__(self, budget: EvaluationBudget, seed: int):
        self.budget = budget
        self.seed = seed
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Construct planted optimum and per-family TypeSpecs from self.seed."""

    @property
    @abstractmethod
    def specs(self) -> dict[GeneFamily, TypeSpec]:
        ...

    @abstractmethod
    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        """Compute fitness without consuming budget. Override in subclasses."""

    def evaluate(self, bundle: TypedBundle) -> float:
        """Consumes 1 budget unit per call."""
        self.budget.consume(1)
        return self._raw_evaluate(bundle)

    def true_evaluate(self, bundle: TypedBundle) -> float:
        """Zero-noise evaluation. For non-noisy problems == _raw_evaluate."""
        return self._raw_evaluate(bundle)

    @abstractmethod
    def true_optimum(self) -> float:
        """The planted optimum value (0.0 for our minimization additives)."""

    def target_threshold(self, fraction: float = 0.01) -> float:
        """ε-target: within `fraction` of optimum gap from `true_optimum()`."""
        return self.true_optimum() + fraction
