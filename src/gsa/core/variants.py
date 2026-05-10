"""GSA variant configurations per spec §4.1."""
from __future__ import annotations

from gsa.core.optimizer import GSAConfig
from gsa.core.types import GeneFamily


# Asynchronous schedule: structural genes (B, C, Cx, E) update less frequently
# than the fast continuous-coefficient gene (R), and integers update at an
# intermediate rate. The choice mirrors the spec §4.5 motivation: structural
# changes have larger phenotypic impact and benefit from giving the faster
# subpopulations time to settle between perturbations.
_ASYNC_PERIODS = {
    GeneFamily.R: 1,
    GeneFamily.Z: 2,
    GeneFamily.B: 4,
    GeneFamily.C: 4,
    GeneFamily.Cx: 4,
    GeneFamily.E: 4,
}


GSA_VARIANTS: dict[str, dict] = {
    "GSA_FULL_ENSEMBLE": dict(credit_mode="ensemble", K=5,
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=0.7),
    "GSA_DIRECT":         dict(credit_mode="direct",
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=0.7),
    "GSA_ELITE_CONTEXT":  dict(credit_mode="elite",
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=0.7),
    "GSA_NO_DIVERSITY":   dict(credit_mode="ensemble", K=5,
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=1.0),
    "GSA_GENERIC_OPERATORS": dict(credit_mode="ensemble", K=5,
                                   operator_mode="generic",
                                   assembly_mode="active", diversity_alpha=0.7),
    "GSA_NO_ASSEMBLY":    dict(credit_mode="ensemble", K=5,
                               operator_mode="type_native",
                               assembly_mode="passive", diversity_alpha=0.7),
    "GSA_ASYNC":          dict(credit_mode="ensemble", K=5,
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=0.7,
                               family_update_periods=_ASYNC_PERIODS),
    "GSA_ASYNC_DIRECT":   dict(credit_mode="direct",
                               operator_mode="type_native",
                               assembly_mode="active", diversity_alpha=0.7,
                               family_update_periods=_ASYNC_PERIODS),
}


def build_config(variant: str, **overrides) -> GSAConfig:
    if variant not in GSA_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base = GSA_VARIANTS[variant].copy()
    base.update(overrides)
    return GSAConfig(**base)
