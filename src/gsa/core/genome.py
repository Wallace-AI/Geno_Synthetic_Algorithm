"""Containers for typed subgenomes, bundles, and assembled phenotypes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from gsa.core.types import GeneFamily, TypeSpec


@dataclass
class TypedSubgenome:
    family: GeneFamily
    values: np.ndarray
    spec: TypeSpec


@dataclass
class TypedBundle:
    """A complete candidate's typed subgenomes, keyed by family."""
    subgenomes: dict[GeneFamily, TypedSubgenome] = field(default_factory=dict)

    @property
    def total_coords(self) -> int:
        return sum(sg.spec.n for sg in self.subgenomes.values())

    def families(self) -> list[GeneFamily]:
        return list(self.subgenomes.keys())

    def get(self, family: GeneFamily) -> Optional[TypedSubgenome]:
        return self.subgenomes.get(family)


@dataclass
class AssemblyDiagnostics:
    valid: bool
    repair_count: int
    invalid_reason: Optional[str]
    n_active_genes: int


@dataclass
class Phenotype:
    features: dict[str, Any]  # numerical features ready for fitness fn
    bundle: TypedBundle       # back-reference to the source bundle
    diagnostics: AssemblyDiagnostics
