import numpy as np
import pytest

from gsa.benchmarks.typed_gated import TypedGated
from gsa.core.assembly import ActiveAssembly, PassiveAssembly
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


def _planted_optimum_bundle(p: TypedGated, off_R_value: float = 0.0
                            ) -> TypedBundle:
    """Bundle that hits the optimum under ActiveAssembly. Under
    PassiveAssembly the R values at off-positions still matter and must
    equal 0 for fitness to be 0."""
    B = TypedSubgenome(GeneFamily.B,
                       p._target_B.copy(),
                       p.specs[GeneFamily.B])
    R_vals = np.where(p._target_B, p._target_R, off_R_value)
    R = TypedSubgenome(GeneFamily.R,
                       R_vals.astype(np.float64),
                       p.specs[GeneFamily.R])
    sub = {GeneFamily.R: R, GeneFamily.B: B}
    if p.include_integer:
        Z = TypedSubgenome(GeneFamily.Z,
                           p._target_Z.copy(),
                           p.specs[GeneFamily.Z])
        sub[GeneFamily.Z] = Z
    return TypedBundle(sub)


def test_specs_have_required_families():
    p = TypedGated(EvaluationBudget(10), seed=0, dim=12, include_integer=False)
    assert GeneFamily.R in p.specs
    assert GeneFamily.B in p.specs
    assert GeneFamily.Z not in p.specs


def test_planted_target_mix_is_not_all_true():
    """Half-and-half target_B ensures the optimum sits inside the gating
    region — the precondition H3 needs."""
    p = TypedGated(EvaluationBudget(10), seed=0, dim=20, include_integer=False)
    assert 0 < p._target_B.sum() < p.dim


def test_active_optimum_is_zero():
    p = TypedGated(EvaluationBudget(10), seed=0, dim=16, include_integer=False)
    p.set_assembler(ActiveAssembly())
    bundle = _planted_optimum_bundle(p, off_R_value=1234.5)  # garbage
    f = p.evaluate(bundle)
    assert f == pytest.approx(0.0, abs=1e-12), (
        "Active assembly should mask off-position R to 0 regardless of "
        "what value the bundle holds there."
    )


def test_passive_optimum_requires_zeroed_off_positions():
    p = TypedGated(EvaluationBudget(10), seed=0, dim=16, include_integer=False)
    p.set_assembler(PassiveAssembly())

    # Off-positions hold garbage: passive should NOT score zero.
    bad = _planted_optimum_bundle(p, off_R_value=1.5)
    f_bad = p.evaluate(bad)
    assert f_bad > 0.0

    # Off-positions zeroed: passive matches active.
    good = _planted_optimum_bundle(p, off_R_value=0.0)
    f_good = p.evaluate(good)
    assert f_good == pytest.approx(0.0, abs=1e-12)


def test_runner_dispatch():
    """End-to-end: a tiny GSA run on typed_gated completes."""
    from gsa.experiments.runner import RunSpec, run_one
    spec = RunSpec(
        algorithm="GSA_FULL_ENSEMBLE",
        benchmark="typed_gated",
        benchmark_kwargs={"dim": 12, "include_integer": False},
        seed=0, budget=200,
        output_dir="results/raw/_test_gated",
    )
    rec = run_one(spec)
    assert rec.status == "completed"
    assert np.isfinite(rec.final_best_observed)


def test_include_complex_adds_cx_family():
    p = TypedGated(EvaluationBudget(10), seed=0, dim=8,
                   include_integer=False, include_complex=True)
    assert GeneFamily.Cx in p.specs
    assert p.specs[GeneFamily.Cx].n == 8


def test_complex_planted_optimum_is_zero_under_active():
    """At B=target_B, R[active]=target_R[active], Cx=target_Cx, fitness=0
    under ActiveAssembly regardless of R at gated-off positions."""
    p = TypedGated(EvaluationBudget(10), seed=1, dim=10,
                   include_integer=False, include_complex=True)
    p.set_assembler(ActiveAssembly())
    B = TypedSubgenome(GeneFamily.B, p._target_B.copy(),
                       p.specs[GeneFamily.B])
    R_vals = np.where(p._target_B, p._target_R, 42.0)  # garbage at off
    R = TypedSubgenome(GeneFamily.R, R_vals.astype(np.float64),
                       p.specs[GeneFamily.R])
    Cx = TypedSubgenome(GeneFamily.Cx, p._target_Cx.copy(),
                        p.specs[GeneFamily.Cx])
    bundle = TypedBundle({GeneFamily.R: R, GeneFamily.B: B,
                          GeneFamily.Cx: Cx})
    f = p.evaluate(bundle)
    assert f == pytest.approx(0.0, abs=1e-12)


def test_flat_baselines_crash_on_complex():
    """When include_complex=True, flattened decoders cannot represent Cx,
    so the run records as failed — the architectural-reach mechanism."""
    from gsa.experiments.runner import RunSpec, run_one
    spec = RunSpec(
        algorithm="FLATTENED_DE",
        benchmark="typed_gated",
        benchmark_kwargs={"dim": 10, "include_integer": False,
                          "include_complex": True},
        seed=0, budget=200,
        output_dir="results/raw/_test_gated_cx",
    )
    rec = run_one(spec)
    assert rec.status == "failed"
