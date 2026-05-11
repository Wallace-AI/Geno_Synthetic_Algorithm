"""Problem ABC for all benchmarks.

Each problem exposes:
    .specs: dict[GeneFamily, TypeSpec]
    .evaluate(bundle) -> float (consumes 1 budget unit)
    .true_optimum() -> float (the planted optimum value)
    .target_threshold() -> float (the ε-target for "hit" reporting)
    .true_evaluate(bundle) -> float (zero-noise evaluation; for noisy benchmarks)

Assembly-aware benchmarks (needs_phenotype=True) override
`_raw_evaluate_pheno(bundle, phenotype)` to read the assembled features
(in particular `phenotype.features["R_effective"]`, which is gated under
ActiveAssembly and raw under PassiveAssembly). The optimizer injects its
assembler via `problem.set_assembler(...)`; baselines leave the default
in place (ActiveAssembly), which means they see the same fitness signal
as `GSA_FULL_ENSEMBLE` on assembly-aware benchmarks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from gsa.core.genome import TypedBundle
from gsa.core.types import GeneFamily, TypeSpec
from gsa.experiments.budget import EvaluationBudget


class Problem(ABC):
    # Subclasses that need to read assembled phenotype features set this
    # to True. When False (default), assembly is skipped per evaluate() to
    # avoid the small numpy overhead for the majority of benchmarks that
    # do not consult the assembler's output.
    needs_phenotype: bool = False

    def __init__(self, budget: EvaluationBudget, seed: int,
                 assembler: Optional[Any] = None):
        self.budget = budget
        self.seed = seed
        self._assembler = assembler  # late-bound default to avoid import cycle
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

    def _raw_evaluate_pheno(self, bundle: TypedBundle, phenotype: Any) -> float:
        """Phenotype-aware variant. Default falls back to bundle-only path."""
        return self._raw_evaluate(bundle)

    @property
    def assembler(self) -> Any:
        if self._assembler is None:
            from gsa.core.assembly import ActiveAssembly
            self._assembler = ActiveAssembly()
        return self._assembler

    def set_assembler(self, assembler: Any) -> None:
        """Used by the optimizer to inject its assembly mode."""
        self._assembler = assembler

    def evaluate(self, bundle: TypedBundle) -> float:
        """Consumes 1 budget unit per call."""
        self.budget.consume(1)
        if self.needs_phenotype:
            pheno, _ = self.assembler.assemble(bundle)
            return self._raw_evaluate_pheno(bundle, pheno)
        return self._raw_evaluate(bundle)

    def true_evaluate(self, bundle: TypedBundle) -> float:
        """Zero-noise evaluation. For non-noisy problems == _raw_evaluate."""
        if self.needs_phenotype:
            pheno, _ = self.assembler.assemble(bundle)
            return self._raw_evaluate_pheno(bundle, pheno)
        return self._raw_evaluate(bundle)

    @abstractmethod
    def true_optimum(self) -> float:
        """The planted optimum value (0.0 for our minimization additives)."""

    def target_threshold(self, fraction: float = 0.01) -> float:
        """ε-target: within `fraction` of optimum gap from `true_optimum()`."""
        return self.true_optimum() + fraction
