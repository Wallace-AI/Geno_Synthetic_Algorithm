import importlib.util

import numpy as np
import pytest

from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.types import GeneFamily
from gsa.experiments.budget import EvaluationBudget


_COCO_AVAILABLE = importlib.util.find_spec("cocoex") is not None
pytestmark = pytest.mark.skipif(
    not _COCO_AVAILABLE, reason="cocoex (coco-experiment) not installed"
)


def _make(seed=0, function=1, instance=1, dim=10, total_budget=10):
    from gsa.benchmarks.coco_mixint import CocoMixInt
    return CocoMixInt(
        budget=EvaluationBudget(total_budget),
        seed=seed, function=function, instance=instance, dim=dim,
    )


def test_coco_mixint_specs_split_integer_real():
    p = _make(dim=10)
    assert GeneFamily.Z in p.specs
    assert GeneFamily.R in p.specs
    z = p.specs[GeneFamily.Z]
    r = p.specs[GeneFamily.R]
    # BBOB-MixInt specifies 80/20 integer/real split: dim=10 -> 8 + 2.
    assert z.n + r.n == 10
    assert z.n == 8
    assert r.n == 2


def test_coco_mixint_evaluate_consumes_budget():
    p = _make(total_budget=5)
    z_spec = p.specs[GeneFamily.Z]
    r_spec = p.specs[GeneFamily.R]
    bundle = TypedBundle({
        GeneFamily.Z: TypedSubgenome(GeneFamily.Z,
                                     np.zeros(z_spec.n, dtype=np.int64),
                                     z_spec),
        GeneFamily.R: TypedSubgenome(GeneFamily.R,
                                     np.zeros(r_spec.n, dtype=np.float64),
                                     r_spec),
    })
    f = p.evaluate(bundle)
    assert np.isfinite(f)
    assert p.budget.consumed == 1


def test_coco_mixint_unknown_problem_raises():
    with pytest.raises(ValueError):
        _make(function=999, instance=1, dim=10)


def test_coco_mixint_runs_through_runner():
    """End-to-end: a tiny baseline run on coco_mixint completes."""
    from gsa.experiments.runner import RunSpec, run_one
    spec = RunSpec(
        algorithm="FLATTENED_DE",
        benchmark="coco_mixint",
        benchmark_kwargs={"function": 1, "instance": 1, "dim": 10},
        seed=0, budget=200,
        output_dir="results/raw/_test_coco",
    )
    rec = run_one(spec)
    assert rec.status == "completed"
    assert np.isfinite(rec.final_best_observed)
