"""GSA main optimizer loop.

Dispatches operators per family, manages typed populations, applies credit
assignment, performs selection, and tracks generation statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gsa.core.assembly import ActiveAssembly, PassiveAssembly
from gsa.core.credit import (
    DirectCredit, EliteCredit, EnsembleCredit, MarginalCredit, EvaluatedBundle,
)
from gsa.core.diversity import _DISPATCH as _DIV_DISPATCH
from gsa.core.genome import TypedBundle, TypedSubgenome
from gsa.core.operators import (
    integer_operator, real_operator_de, boolean_operator,
    categorical_operator, complex_operator, embedding_operator,
    generic_operator, OperatorContext,
)
from gsa.core.populations import TypedPopulation
from gsa.core.selection import diversity_regularized_select, tournament_select
from gsa.core.types import GeneFamily
from gsa.experiments.seed_control import derive_run_seeds


_TYPE_NATIVE_DISPATCH = {
    GeneFamily.Z: integer_operator,
    GeneFamily.R: real_operator_de,
    GeneFamily.B: boolean_operator,
    GeneFamily.C: categorical_operator,
    GeneFamily.Cx: complex_operator,
    GeneFamily.E: embedding_operator,
}


@dataclass
class GSAConfig:
    population_size: int = 50
    credit_mode: str = "ensemble"        # direct | elite | ensemble | marginal
    K: int = 5                            # ensemble partner count
    operator_mode: str = "type_native"    # type_native | generic
    assembly_mode: str = "active"         # active | passive
    diversity_alpha: float = 0.7
    selection_k: int = 3
    F: float = 0.5
    CR: float = 0.9
    # Asynchronous evolution: per-family update period in outer generations.
    # `None` = every family updates every generation (synchronous).
    # `{R: 1, B: 4, Z: 2, C: 4, Cx: 4, E: 4}` = R updates each gen, B every
    # 4 gens, etc. Families absent from the dict default to period 1.
    family_update_periods: Optional[dict] = None


@dataclass
class GenerationStats:
    generation: int
    eval_count: int
    best_so_far: float
    mean_fitness: float
    std_fitness: float
    diversity_per_family: dict
    operator_success_per_family: dict
    n_invalid: int
    n_repaired: int


@dataclass
class GSAResult:
    best_fitness: float
    best_bundle: Optional[TypedBundle]
    history: list = field(default_factory=list)
    total_evaluations: int = 0
    total_invalid: int = 0
    total_repairs: int = 0


def _make_credit_assigner(cfg: GSAConfig):
    if cfg.credit_mode == "direct":
        return DirectCredit()
    if cfg.credit_mode == "elite":
        return EliteCredit()
    if cfg.credit_mode == "ensemble":
        return EnsembleCredit(K=cfg.K)
    if cfg.credit_mode == "marginal":
        return MarginalCredit()
    raise ValueError(f"Unknown credit_mode: {cfg.credit_mode}")


def _make_assembly(cfg: GSAConfig):
    return ActiveAssembly() if cfg.assembly_mode == "active" else PassiveAssembly()


def _operator_for(family, mode: str):
    if mode == "generic":
        return generic_operator
    return _TYPE_NATIVE_DISPATCH[family]


def _apply_operator(parent: TypedSubgenome, family: GeneFamily,
                    population: TypedPopulation, *,
                    rng: np.random.Generator, cfg: GSAConfig) -> TypedSubgenome:
    """Run operator on parent, building OperatorContext from population."""
    op = _operator_for(family, cfg.operator_mode)
    ctx = None
    if cfg.operator_mode == "type_native" and family == GeneFamily.R:
        # DE/best/1: r1 = population best, r2/r3 = random distinct others
        if population.size >= 3:
            fit = population.fitness
            if fit is not None and not np.all(np.isinf(fit)):
                best_i = int(np.argmin(fit))
            else:
                best_i = int(rng.integers(0, population.size))
            others = [i for i in range(population.size) if i != best_i]
            two = list(rng.choice(others, size=2, replace=False))
            donors = [population.individuals[best_i].values,
                      population.individuals[two[0]].values,
                      population.individuals[two[1]].values]
            ctx = OperatorContext(donors=donors)
        new_vals = real_operator_de(parent.values, parent.spec,
                                    rng=rng, ctx=ctx, F=cfg.F, CR=cfg.CR)
    else:
        new_vals = op(parent.values, parent.spec, rng=rng, ctx=ctx)
    return TypedSubgenome(family, new_vals, parent.spec)


class GSAOptimizer:
    def __init__(self, problem, cfg: GSAConfig, master_seed: int):
        self.problem = problem
        self.cfg = cfg
        self.seeds = derive_run_seeds(master_seed)
        self.rng_init = np.random.default_rng(self.seeds.seed_init)
        self.rng_op = np.random.default_rng(self.seeds.seed_operators)
        self.rng_sel = np.random.default_rng(self.seeds.seed_selection)
        self.assembler = _make_assembly(cfg)
        self.credit = _make_credit_assigner(cfg)

        self.populations: dict[GeneFamily, TypedPopulation] = {}
        self.best_fitness = float("inf")
        self.best_bundle: Optional[TypedBundle] = None
        self.history: list[GenerationStats] = []
        self.total_invalid = 0
        self.total_repairs = 0

    def _initialize(self):
        for family, spec in self.problem.specs.items():
            pop = TypedPopulation(spec=spec, size=self.cfg.population_size,
                                  rng=self.rng_init)
            pop.sample_initial()
            self.populations[family] = pop
        # Evaluate initial bundles by index pairing across families
        n = self.cfg.population_size
        for i in range(n):
            if not self.problem.budget.has(1):
                return
            bundle = TypedBundle({
                fam: self.populations[fam].individuals[i]
                for fam in self.populations
            })
            pheno, diag = self.assembler.assemble(bundle)
            f = self.problem.evaluate(bundle)
            for fam in self.populations:
                self.populations[fam].fitness[i] = f
            if not diag.valid:
                self.total_invalid += 1
            self.total_repairs += diag.repair_count
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_bundle = bundle

    def _elite_partners(self) -> dict[GeneFamily, TypedSubgenome]:
        return {
            fam: pop.best() for fam, pop in self.populations.items()
        }

    def _ensemble_pool(self) -> dict[GeneFamily, list[TypedSubgenome]]:
        """Top-half of each population as the partner sampling pool."""
        out = {}
        for fam, pop in self.populations.items():
            order = np.argsort(pop.fitness)
            top_half = order[: max(1, pop.size // 2)]
            out[fam] = [pop.individuals[i] for i in top_half]
        return out

    def _diversity_per_family(self) -> dict[GeneFamily, np.ndarray]:
        """For each family, an array of per-individual diversity scores
        (mean distance from individual to the rest of the population)."""
        out = {}
        for fam, pop in self.populations.items():
            fn = _DIV_DISPATCH[fam]
            n = pop.size
            scores = np.zeros(n)
            for i in range(n):
                d = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    d += fn(pop.individuals[i].values, pop.individuals[j].values)
                scores[i] = d / max(1, n - 1)
            out[fam] = scores
        return out

    def _families_active_this_gen(self, gen: int) -> list:
        periods = self.cfg.family_update_periods or {}
        active = []
        for fam in self.populations:
            period = max(1, int(periods.get(fam, 1)))
            if gen % period == 0:
                active.append(fam)
        return active

    def _step(self, gen: int = 0):
        elite = self._elite_partners()
        ens_pool = self._ensemble_pool()
        diversity_scores = self._diversity_per_family()
        op_success = {fam: 0 for fam in self.populations}
        op_attempts = {fam: 0 for fam in self.populations}
        n_invalid = 0
        n_repaired = 0

        active_families = self._families_active_this_gen(gen)
        for fam in active_families:
            pop = self.populations[fam]
            for i in range(pop.size):
                if not self.problem.budget.has(1):
                    return op_success, op_attempts, n_invalid, n_repaired
                # Selection: pick parent index per cfg.diversity_alpha
                if self.cfg.diversity_alpha < 1.0:
                    parent_idx = diversity_regularized_select(
                        pop.fitness, diversity_scores[fam],
                        k=self.cfg.selection_k, rng=self.rng_sel,
                        alpha=self.cfg.diversity_alpha)
                else:
                    parent_idx = tournament_select(
                        pop.fitness, k=self.cfg.selection_k, rng=self.rng_sel)
                parent = pop.individuals[parent_idx]
                child = _apply_operator(parent, fam, pop,
                                        rng=self.rng_op, cfg=self.cfg)

                # Build a candidate bundle pairing child with current i-th
                # individuals of every other family.
                bundle = TypedBundle({
                    other_fam: (child if other_fam == fam
                                else self.populations[other_fam].individuals[i])
                    for other_fam in self.populations
                })
                pheno, diag = self.assembler.assemble(bundle)
                if not diag.valid:
                    n_invalid += 1
                n_repaired += diag.repair_count

                # Assembled fitness — counts 1 budget unit
                f_assembled = self.problem.evaluate(bundle)
                eb = EvaluatedBundle(bundle=bundle, fitness=f_assembled)

                # Credit: dict[family, credit_value]. Costs extra evaluations
                # for elite/ensemble/marginal modes.
                if isinstance(self.credit, EliteCredit):
                    pool_arg = elite
                elif isinstance(self.credit, EnsembleCredit):
                    pool_arg = ens_pool
                else:
                    pool_arg = None
                try:
                    credits = self.credit.assign(eb, partner_pool=pool_arg,
                                                 problem=self.problem,
                                                 rng=self.rng_op,
                                                 target_family=fam)
                except Exception:
                    # Budget exhausted mid-credit-update: fall back to f_assembled
                    credits = {fam: f_assembled}

                # Replacement: replace parent if THIS family's credit is better
                op_attempts[fam] += 1
                credit_for_fam = credits.get(fam, f_assembled)
                if credit_for_fam < pop.fitness[parent_idx]:
                    pop.replace(parent_idx, child, credit_for_fam)
                    op_success[fam] += 1

                if f_assembled < self.best_fitness:
                    self.best_fitness = f_assembled
                    self.best_bundle = bundle
        return op_success, op_attempts, n_invalid, n_repaired

    def run(self, max_generations: int = 10**9) -> GSAResult:
        self._initialize()
        gen = 0
        while gen < max_generations and self.problem.budget.has(1):
            try:
                step_out = self._step(gen)
            except Exception:
                break  # budget exhausted or other terminal issue
            if step_out is None:
                break
            op_success, op_attempts, n_invalid, n_repaired = step_out
            self.total_invalid += n_invalid
            self.total_repairs += n_repaired
            self.history.append(GenerationStats(
                generation=gen,
                eval_count=self.problem.budget.consumed,
                best_so_far=self.best_fitness,
                mean_fitness=float(np.mean(np.concatenate(
                    [pop.fitness for pop in self.populations.values()]))),
                std_fitness=float(np.std(np.concatenate(
                    [pop.fitness for pop in self.populations.values()]))),
                diversity_per_family={fam: pop.diversity()
                                       for fam, pop in self.populations.items()},
                operator_success_per_family={fam: op_success[fam] / max(1, op_attempts[fam])
                                              for fam in self.populations},
                n_invalid=n_invalid, n_repaired=n_repaired,
            ))
            gen += 1
        return GSAResult(
            best_fitness=self.best_fitness,
            best_bundle=self.best_bundle,
            history=self.history,
            total_evaluations=self.problem.budget.consumed,
            total_invalid=self.total_invalid,
            total_repairs=self.total_repairs,
        )


def run_gsa(problem, cfg: GSAConfig, master_seed: int) -> GSAResult:
    return GSAOptimizer(problem, cfg, master_seed).run()
