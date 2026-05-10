"""Gene family types and per-family specifications."""
from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class GeneFamily(str, Enum):
    Z = "Z"   # Integer
    R = "R"   # Real
    B = "B"   # Boolean
    C = "C"   # Categorical
    Cx = "Cx" # Complex
    E = "E"   # Embedding


class _BaseSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    n: int  # number of gene coordinates


class IntegerSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.Z
    lo: np.ndarray  # shape (n,)
    hi: np.ndarray

    @model_validator(mode="after")
    def _check_bounds(self):
        if (self.lo > self.hi).any():
            raise ValueError("IntegerSpec: lo must be <= hi elementwise")
        return self


class RealSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.R
    lo: np.ndarray
    hi: np.ndarray

    @model_validator(mode="after")
    def _check_bounds(self):
        if (self.lo > self.hi).any():
            raise ValueError("RealSpec: lo must be <= hi elementwise")
        return self


class BooleanSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.B


class CategoricalSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.C
    n_categories: list[int]  # length n; admissible set size per coord

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.n_categories) != self.n:
            raise ValueError("CategoricalSpec: n_categories length != n")
        return self


class ComplexSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.Cx
    r_min: float = 0.0
    r_max: float = 1.0


class EmbeddingSpec(_BaseSpec):
    family: GeneFamily = GeneFamily.E
    dim: int = 8


TypeSpec = Union[IntegerSpec, RealSpec, BooleanSpec, CategoricalSpec,
                 ComplexSpec, EmbeddingSpec]
