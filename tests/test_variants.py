from gsa.core.variants import GSA_VARIANTS, build_config


def test_six_variants_present():
    assert set(GSA_VARIANTS.keys()) == {
        "GSA_FULL_ENSEMBLE", "GSA_DIRECT", "GSA_ELITE_CONTEXT",
        "GSA_NO_DIVERSITY", "GSA_GENERIC_OPERATORS", "GSA_NO_ASSEMBLY",
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
