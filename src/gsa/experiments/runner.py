"""Experiment runner: takes a list of RunSpecs and executes them.

Single-machine multiprocessing via concurrent.futures.ProcessPoolExecutor.
Windows-safe: top-level callables only; CLI scripts must use __main__ guard."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gsa.benchmarks.typed_additive import TypedAdditive
from gsa.benchmarks.typed_epistatic import TypedEpistatic
from gsa.benchmarks.typed_deceptive import TypedDeceptive
from gsa.benchmarks.typed_noisy import TypedNoisy
from gsa.benchmarks.typed_mix import TypedMix
from gsa.benchmarks.ioh_adapter import ioh_problem
from gsa.baselines.random_search import (
    random_flattened_search, random_typed_search,
)
from gsa.baselines.flattened_de import flattened_de
from gsa.baselines.flattened_ea import flattened_ea
from gsa.baselines.mixed_variable_ga import mixed_variable_ga
from gsa.baselines.cooperative_coevolution import cooperative_coevolution
from gsa.core.optimizer import run_gsa
from gsa.core.variants import build_config, GSA_VARIANTS
from gsa.experiments.budget import EvaluationBudget
from gsa.experiments.logging import RunLogger, RunRecord
from gsa.experiments.provenance import capture_provenance


_BENCHMARK_REGISTRY = {
    "typed_additive": TypedAdditive,
    "typed_epistatic": TypedEpistatic,
    "typed_deceptive": TypedDeceptive,
    "typed_noisy": TypedNoisy,
    "typed_mix": TypedMix,
}

_BASELINE_REGISTRY = {
    "RANDOM_FLATTENED": random_flattened_search,
    "RANDOM_TYPED": random_typed_search,
    "FLATTENED_DE": flattened_de,
    "FLATTENED_EA": flattened_ea,
    "MIXED_VARIABLE_GA": mixed_variable_ga,
    "COOPERATIVE_COEVOLUTION": cooperative_coevolution,
}


@dataclass
class RunSpec:
    algorithm: str
    benchmark: str
    benchmark_kwargs: dict = field(default_factory=dict)
    algorithm_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    budget: int = 5000
    output_dir: str = "results/raw"


def _config_hash(spec: RunSpec) -> str:
    payload = json.dumps({
        "algorithm": spec.algorithm,
        "benchmark": spec.benchmark,
        "benchmark_kwargs": spec.benchmark_kwargs,
        "algorithm_kwargs": spec.algorithm_kwargs,
        "budget": spec.budget,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _make_problem(spec: RunSpec, budget: EvaluationBudget):
    if spec.benchmark.startswith("ioh:"):
        name = spec.benchmark.split(":", 1)[1]
        return ioh_problem(name=name, seed=spec.seed, budget=budget,
                            **spec.benchmark_kwargs)
    if spec.benchmark not in _BENCHMARK_REGISTRY:
        raise KeyError(f"unknown benchmark: {spec.benchmark}")
    cls = _BENCHMARK_REGISTRY[spec.benchmark]
    return cls(budget=budget, seed=spec.seed, **spec.benchmark_kwargs)


def _run_algorithm(spec: RunSpec, problem):
    if spec.algorithm in GSA_VARIANTS:
        cfg = build_config(spec.algorithm, **spec.algorithm_kwargs)
        return run_gsa(problem, cfg, master_seed=spec.seed)
    if spec.algorithm in _BASELINE_REGISTRY:
        fn = _BASELINE_REGISTRY[spec.algorithm]
        return fn(problem, master_seed=spec.seed, **spec.algorithm_kwargs)
    raise KeyError(f"unknown algorithm: {spec.algorithm}")


def run_one(spec: RunSpec) -> RunRecord:
    """Execute one (algorithm, benchmark, seed) combination, return RunRecord."""
    prov = capture_provenance()
    cfg_hash = _config_hash(spec)
    run_id = str(uuid.uuid4())
    t0 = time.time()
    status = "completed"
    err = None
    final_obs = float("inf")
    final_true = float("inf")
    auc = 0.0
    total_evals = 0
    invalid = 0
    repairs = 0
    target_hit = False
    try:
        budget = EvaluationBudget(total=spec.budget)
        problem = _make_problem(spec, budget)
        result = _run_algorithm(spec, problem)
        final_obs = float(result.best_fitness)
        if hasattr(result, "best_bundle") and result.best_bundle is not None:
            final_true = float(problem.true_evaluate(result.best_bundle))
        else:
            final_true = final_obs
        total_evals = problem.budget.consumed
        auc = 1.0 / (1.0 + max(final_obs, 0.0))
        target_hit = final_true <= problem.target_threshold(0.01)
    except Exception as e:
        status = "failed"
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    wall = time.time() - t0

    rho = spec.benchmark_kwargs.get("rho")
    n_fams = spec.benchmark_kwargs.get("n_families")
    noise_mode = spec.benchmark_kwargs.get("noise_mode")
    dim = spec.benchmark_kwargs.get("dim", 0) or spec.benchmark_kwargs.get("n", 0)

    return RunRecord(
        run_id=run_id, git_commit=prov.git_commit, config_hash=cfg_hash,
        env_hash=prov.env_hash, hardware_fingerprint=prov.hardware_fingerprint,
        algorithm=spec.algorithm, variant=spec.algorithm,
        benchmark=spec.benchmark, dim=int(dim),
        rho=rho, n_families=n_fams, noise_mode=noise_mode,
        seed_master=spec.seed, evaluation_budget=spec.budget,
        final_best_observed=final_obs, final_best_true=final_true,
        auc_convergence=auc, evaluations_to_target=float("inf"),
        target_hit=target_hit,
        total_evaluations=total_evals, total_invalid_offspring=invalid,
        total_repairs=repairs, wall_clock_seconds=wall, peak_memory_mb=0.0,
        status=status, error_message=err,
    )


def _run_one_for_pool(spec: RunSpec) -> RunRecord:
    """Process-pool entry point. Must be a top-level callable on Windows."""
    return run_one(spec)


def run_many(specs: list[RunSpec], *, parallel: bool = True,
             max_workers: Optional[int] = None) -> None:
    """Run a batch of specs and write all RunRecords to <output_dir>/runs.parquet.

    Assumes all specs share the same output_dir."""
    if not specs:
        return
    output_dir = Path(specs[0].output_dir)
    for s in specs:
        if Path(s.output_dir) != output_dir:
            raise ValueError("run_many: all specs must share output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_run_one_for_pool, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futs):
                records.append(fut.result())
    else:
        records = [run_one(s) for s in specs]

    logger = RunLogger(output_dir / "runs.parquet")
    for rec in records:
        logger.write(rec)
    logger.flush()
