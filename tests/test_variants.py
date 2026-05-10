from gsa.core.variants import GSA_VARIANTS, build_config
from gsa.core.types import GeneFamily


def test_eight_variants_present():
    assert set(GSA_VARIANTS.keys()) == {
        "GSA_FULL_ENSEMBLE", "GSA_DIRECT", "GSA_ELITE_CONTEXT",
        "GSA_NO_DIVERSITY", "GSA_GENERIC_OPERATORS", "GSA_NO_ASSEMBLY",
        "GSA_ASYNC", "GSA_ASYNC_DIRECT",
    }


def test_full_ensemble_config():
    cfg = build_config("GSA_FULL_ENSEMBLE")
    assert cfg.credit_mode == "ensemble"
    assert cfg.operator_mode == "type_native"
    assert cfg.assembly_mode == "active"
    assert cfg.diversity_alpha == 0.7
    assert cfg.K == 5


def test_no_diversity_config():
    cfg = build_config("GSA_NO_DIVERSITY")
    assert cfg.diversity_alpha == 1.0


def test_generic_operators_config():
    cfg = build_config("GSA_GENERIC_OPERATORS")
    assert cfg.operator_mode == "generic"


def test_no_assembly_config():
    cfg = build_config("GSA_NO_ASSEMBLY")
    assert cfg.assembly_mode == "passive"


def test_direct_config():
    cfg = build_config("GSA_DIRECT")
    assert cfg.credit_mode == "direct"


def test_elite_config():
    cfg = build_config("GSA_ELITE_CONTEXT")
    assert cfg.credit_mode == "elite"


def test_async_config_has_periods():
    cfg = build_config("GSA_ASYNC")
    assert cfg.family_update_periods is not None
    assert cfg.family_update_periods[GeneFamily.R] == 1
    assert cfg.family_update_periods[GeneFamily.B] == 4
    # Synchronous variants should have no periods set
    sync_cfg = build_config("GSA_FULL_ENSEMBLE")
    assert sync_cfg.family_update_periods is None


def test_async_direct_config():
    cfg = build_config("GSA_ASYNC_DIRECT")
    assert cfg.credit_mode == "direct"
    assert cfg.family_update_periods is not None
