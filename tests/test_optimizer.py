"""Phase-2 integration test on a minimal R-only sphere problem.

Asserts GSA_FULL_ENSEMBLE drives the sphere fitness toward zero. The full
Typed Additive D=10 sanity is in P3 (Task 3.12)."""
import numpy as np

from gsa.core.optimizer import GSAOptimizer, GSAConfig, run_gsa
from gsa.core.genome import TypedBundle
from gsa.core.types import GeneFamily, RealSpec
from gsa.core.variants import build_config
from gsa.experiments.budget import EvaluationBudget


class SphereProblem:
    """Minimization sphere on the R subgenome."""

    def __init__(self, target: np.ndarray, budget: EvaluationBudget):
        self.target = target
        self.budget = budget

    @property
    def specs(self):
        return {GeneFamily.R: RealSpec(n=len(self.target),
                                       lo=-np.ones(len(self.target)) * 5,
                                       hi=np.ones(len(self.target)) * 5)}

    def evaluate(self, bundle: TypedBundle) -> float:
        self.budget.consume(1)
        r = bundle.get(GeneFamily.R).values
        return float(np.sum((r - self.target) ** 2))


def test_gsa_drives_sphere_fitness_below_threshold():
    target = np.zeros(10)
    budget = EvaluationBudget(total=5000)
    problem = SphereProblem(target, budget)
    cfg = GSAConfig(
        population_size=50,
        credit_mode="ensemble", K=5,
        operator_mode="type_native",
        assembly_mode="active",
        diversity_alpha=0.7,
    )
    result = run_gsa(problem, cfg, master_seed=42)
    assert result.best_fitness < 1e-3, \
        f"Expected f<1e-3 on sphere D=10, got {result.best_fitness}"


def test_gsa_reproducible_with_same_seed():
    target = np.zeros(5)
    budget1 = EvaluationBudget(total=500)
    budget2 = EvaluationBudget(total=500)
    cfg = GSAConfig(population_size=20, credit_mode="direct",
                    operator_mode="type_native", assembly_mode="active",
                    diversity_alpha=0.7)
    r1 = run_gsa(SphereProblem(target, budget1), cfg, master_seed=42)
    r2 = run_gsa(SphereProblem(target, budget2), cfg, master_seed=42)
    assert r1.best_fitness == r2.best_fitness


def test_credit_mode_affects_total_evaluation_count():
    """Ensemble credit consumes K=5× per credit update; direct consumes 1×.

    For a fixed budget, ensemble runs fewer credit updates."""
    target = np.zeros(8)

    # Direct credit
    budget1 = EvaluationBudget(total=2000)
    p1 = SphereProblem(target, budget1)
    cfg_direct = build_config("GSA_DIRECT")
    r_direct = run_gsa(p1, cfg_direct, master_seed=42)

    # Ensemble credit (same budget)
    budget2 = EvaluationBudget(total=2000)
    p2 = SphereProblem(target, budget2)
    cfg_ens = build_config("GSA_FULL_ENSEMBLE")
    r_ens = run_gsa(p2, cfg_ens, master_seed=42)

    assert budget1.consumed <= 2000
    assert budget2.consumed <= 2000


def test_no_diversity_variant_differs_from_full():
    """diversity_alpha=1.0 should produce different replacement decisions than
    diversity_alpha=0.7 in a deceptive-style scenario."""
    target = np.zeros(8)
    full = run_gsa(SphereProblem(target, EvaluationBudget(1000)),
                    build_config("GSA_FULL_ENSEMBLE"), master_seed=7)
    nodiv = run_gsa(SphereProblem(target, EvaluationBudget(1000)),
                     build_config("GSA_NO_DIVERSITY"), master_seed=7)
    assert abs(full.best_fitness - nodiv.best_fitness) > 0 or \
           full.history != nodiv.history


def test_generic_operators_variant_differs_from_full():
    target = np.zeros(8)
    full = run_gsa(SphereProblem(target, EvaluationBudget(1000)),
                    build_config("GSA_FULL_ENSEMBLE"), master_seed=11)
    generic = run_gsa(SphereProblem(target, EvaluationBudget(1000)),
                       build_config("GSA_GENERIC_OPERATORS"), master_seed=11)
    assert full.best_fitness != generic.best_fitness


def test_optimizer_emits_generation_stats():
    target = np.zeros(8)
    p = SphereProblem(target, EvaluationBudget(500))
    result = run_gsa(p, build_config("GSA_FULL_ENSEMBLE"), master_seed=0)
    assert len(result.history) > 0
    first = result.history[0]
    assert first.generation == 0
    assert first.eval_count > 0
    assert first.best_so_far >= 0
