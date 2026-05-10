import numpy as np
from gsa.baselines.decoder import (
    flatten_specs, encode_bundle, decode_to_bundle, FlatLayout,
)
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import (
    GeneFamily, RealSpec, IntegerSpec, BooleanSpec, CategoricalSpec,
)


def make_specs():
    return {
        GeneFamily.R: RealSpec(n=2, lo=-np.ones(2) * 5, hi=np.ones(2) * 5),
        GeneFamily.Z: IntegerSpec(n=2, lo=np.zeros(2, dtype=int),
                                   hi=np.full(2, 10)),
        GeneFamily.B: BooleanSpec(n=3),
        GeneFamily.C: CategoricalSpec(n=2, n_categories=[3, 4]),
    }


def test_flatten_layout_records_offsets():
    layout = flatten_specs(make_specs())
    assert layout.total_dim == 2 + 2 + 3 + 2  # 9
    assert layout.slices[GeneFamily.R].stop - layout.slices[GeneFamily.R].start == 2


def test_round_trip_preserves_values():
    specs = make_specs()
    layout = flatten_specs(specs)
    bundle = TypedBundle({
        GeneFamily.R: TypedSubgenome(GeneFamily.R,
                                     np.array([1.0, -2.0]), specs[GeneFamily.R]),
        GeneFamily.Z: TypedSubgenome(GeneFamily.Z,
                                     np.array([3, 7]), specs[GeneFamily.Z]),
        GeneFamily.B: TypedSubgenome(GeneFamily.B,
                                     np.array([True, False, True]),
                                     specs[GeneFamily.B]),
        GeneFamily.C: TypedSubgenome(GeneFamily.C,
                                     np.array([1, 2]), specs[GeneFamily.C]),
    })
    flat = encode_bundle(bundle, layout)
    assert flat.shape == (9,)
    bundle2 = decode_to_bundle(flat, specs, layout)
    assert np.array_equal(bundle2.subgenomes[GeneFamily.R].values,
                          bundle.subgenomes[GeneFamily.R].values)
    assert np.array_equal(bundle2.subgenomes[GeneFamily.Z].values,
                          bundle.subgenomes[GeneFamily.Z].values)
    assert np.array_equal(bundle2.subgenomes[GeneFamily.B].values,
                          bundle.subgenomes[GeneFamily.B].values)


def test_decoder_clips_and_rounds():
    specs = make_specs()
    layout = flatten_specs(specs)
    flat = np.array([10.0, -10.0,
                     7.4, -1.0,
                     0.6, 0.4, 0.7,
                     2.7, 5.3])
    bundle = decode_to_bundle(flat, specs, layout)
    assert (bundle.subgenomes[GeneFamily.R].values >= -5).all()
    assert (bundle.subgenomes[GeneFamily.R].values <= 5).all()
    z = bundle.subgenomes[GeneFamily.Z].values
    assert z[0] == 7
    assert z[1] == 0
    b = bundle.subgenomes[GeneFamily.B].values
    assert b[0] == True and b[1] == False and b[2] == True
    c = bundle.subgenomes[GeneFamily.C].values
    assert 0 <= c[0] < 3
    assert 0 <= c[1] < 4
