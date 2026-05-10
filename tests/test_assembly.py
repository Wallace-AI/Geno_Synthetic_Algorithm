import numpy as np
from gsa.core.assembly import ActiveAssembly, PassiveAssembly
from gsa.core.genome import TypedSubgenome, TypedBundle
from gsa.core.types import (
    GeneFamily, RealSpec, BooleanSpec, IntegerSpec, CategoricalSpec,
)


def make_bundle():
    rspec = RealSpec(n=3, lo=np.zeros(3), hi=np.ones(3))
    bspec = BooleanSpec(n=3)  # gates each R coord
    return TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R, np.array([0.5, 0.5, 0.5]), rspec),
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.array([True, False, True]), bspec),
    })


def test_active_assembly_gates_R_by_B():
    asm = ActiveAssembly(boolean_gates_real=True)
    bundle = make_bundle()
    pheno, diag = asm.assemble(bundle)
    # Index 1's R is masked off -> phenotype value should be 0 there
    R_eff = pheno.features["R_effective"]
    assert R_eff[0] == 0.5
    assert R_eff[1] == 0.0  # gated off
    assert R_eff[2] == 0.5
    assert diag.valid is True
    assert diag.n_active_genes == 5  # 2 active R + 3 B (B genes themselves count)


def test_active_assembly_zero_active_when_all_gated():
    asm = ActiveAssembly(boolean_gates_real=True)
    rspec = RealSpec(n=2, lo=np.zeros(2), hi=np.ones(2))
    bspec = BooleanSpec(n=2)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R, np.array([0.5, 0.5]), rspec),
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.array([False, False]), bspec),
    })
    pheno, diag = asm.assemble(bundle)
    R_eff = pheno.features["R_effective"]
    assert (R_eff == 0).all()


def test_passive_assembly_does_not_gate():
    asm = PassiveAssembly()
    bundle = make_bundle()
    pheno, diag = asm.assemble(bundle)
    R_eff = pheno.features["R_effective"]
    # Boolean does not gate -> R values unchanged
    assert np.allclose(R_eff, np.array([0.5, 0.5, 0.5]))


def test_active_assembly_categorical_routing():
    """Categorical genes route to a sublandscape index."""
    cspec = CategoricalSpec(n=1, n_categories=[3])
    bundle = TypedBundle({
        GeneFamily.C: TypedSubgenome(GeneFamily.C, np.array([2]), cspec),
    })
    asm = ActiveAssembly(categorical_routes=True)
    pheno, diag = asm.assemble(bundle)
    assert pheno.features["C_route"] == 2
