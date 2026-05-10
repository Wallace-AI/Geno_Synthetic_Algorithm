"""GSA variant configurations per spec §4.1."""
from __future__ import annotations

from gsa.core.optimizer import GSAConfig


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
}


def build_config(variant: str, **overrides) -> GSAConfig:
    if variant not in GSA_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base = GSA_VARIANTS[variant].copy()
    base.update(overrides)
    return GSAConfig(**base)
