"""Typed Noisy benchmark per spec §3.4.

Wrapper over TypedEpistatic at ρ=0.5. Two noise modes: Gaussian additive,
heavy-tailed (Student's t, df=3). Noise is seeded for reproducibility."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gsa.benchmarks.typed_epistatic import TypedEpistatic
from gsa.core.genome import TypedBundle


class TypedNoisy(TypedEpistatic):
    def __init__(self, budget, seed: int, dim: int = 20,
                 families: Sequence[str] = ("Z", "R", "B", "C"),
                 noise_mode: str = "gaussian",
                 sigma: float = 0.1,
                 df: int = 3,
                 rho: float = 0.5,
                 **kwargs):
        if noise_mode not in ("gaussian", "heavy_tailed"):
            raise ValueError(f"unknown noise_mode: {noise_mode}")
        self.noise_mode = noise_mode
        self.sigma = sigma
        self.df = df
        super().__init__(budget=budget, seed=seed, dim=dim, families=families,
                         rho=rho, **kwargs)
        self._noise_rng = np.random.default_rng(seed + 12345)

    def _draw_noise(self) -> float:
        if self.noise_mode == "gaussian":
            return float(self._noise_rng.normal(scale=self.sigma))
        return float(self._noise_rng.standard_t(self.df) * self.sigma)

    def _raw_evaluate(self, bundle: TypedBundle) -> float:
        f_true = super()._raw_evaluate(bundle)
        return f_true + self._draw_noise()

    def true_evaluate(self, bundle: TypedBundle) -> float:
        return super()._raw_evaluate(bundle)
