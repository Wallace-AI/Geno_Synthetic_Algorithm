import numpy as np
from gsa.core.genome import (
    TypedSubgenome, TypedBundle, Phenotype, AssemblyDiagnostics,
)
from gsa.core.types import (
    GeneFamily, RealSpec, BooleanSpec, IntegerSpec,
)


def test_typed_subgenome_holds_array_and_spec():
    spec = RealSpec(n=3, lo=np.zeros(3), hi=np.ones(3))
    sg = TypedSubgenome(family=GeneFamily.R, values=np.array([0.1, 0.2, 0.3]), spec=spec)
    assert sg.family == GeneFamily.R
    assert sg.values.shape == (3,)


def test_typed_bundle_collects_subgenomes():
    rspec = RealSpec(n=2, lo=np.zeros(2), hi=np.ones(2))
    bspec = BooleanSpec(n=3)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(family=GeneFamily.R,
                                     values=np.array([0.5, 0.5]),
                                     spec=rspec),
        GeneFamily.B: TypedSubgenome(family=GeneFamily.B,
                                     values=np.array([True, False, True]),
                                     spec=bspec),
    })
    assert set(bundle.subgenomes.keys()) == {GeneFamily.R, GeneFamily.B}
    assert bundle.total_coords == 5


def test_phenotype_dataclass():
    rspec = RealSpec(n=1, lo=np.zeros(1), hi=np.ones(1))
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(family=GeneFamily.R,
                                     values=np.array([0.5]), spec=rspec)
    })
    diag = AssemblyDiagnostics(valid=True, repair_count=0,
                                invalid_reason=None, n_active_genes=1)
    pheno = Phenotype(features={"x": np.array([0.5])}, bundle=bundle,
                      diagnostics=diag)
    assert pheno.diagnostics.valid is True
