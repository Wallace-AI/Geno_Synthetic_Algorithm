"""Run the async-vs-sync experiment matrix.

Compares GSA_ASYNC and GSA_ASYNC_DIRECT against their synchronous counterparts
on multi-family benchmarks, where the structural-gene update period creates
the largest budget difference. Single-family problems (OneMax) are excluded
because async has no purchase when only one family exists.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsa.experiments.runner import RunSpec, run_many


def build_specs(n_seeds: int, output_dir: str) -> list[RunSpec]:
    specs: list[RunSpec] = []

    sync_async_pairs = [
        ("GSA_FULL_ENSEMBLE", "GSA_ASYNC"),
        ("GSA_DIRECT", "GSA_ASYNC_DIRECT"),
    ]
    flat_algos = ["GSA_FULL_ENSEMBLE", "GSA_ASYNC",
                  "GSA_DIRECT", "GSA_ASYNC_DIRECT"]

    # Cell A: Typed Additive D=20, 4 families. Pure separability — async
    # should not help much, but it should not hurt much either.
    for algo, seed in product(flat_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_additive",
            benchmark_kwargs={"dim": 20, "families": ("R", "B", "Z", "C")},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell B: Typed Epistatic D=20 with rho=0.5. Cross-type interaction
    # via Boolean gating + Integer subfunction — the regime where keeping
    # structural genes (B, Z) stable while R refines should help most.
    for algo, seed in product(flat_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_epistatic",
            benchmark_kwargs={"dim": 20,
                              "families": ("R", "B", "Z", "C"),
                              "rho": 0.5},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell C: Typed-Mix at n_families = 6 (all six). Async is the only
    # GSA variant whose total per-outer-gen cost on six families is small
    # enough to do more than a handful of generations at the standard budget.
    for algo, seed in product(flat_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_mix",
            benchmark_kwargs={"dim": 24, "n_families": 6, "rho": 0.0},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell D: Larger budget on Typed Epistatic. Async should compound
    # its budget savings as the budget grows.
    for algo, seed in product(flat_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_epistatic",
            benchmark_kwargs={"dim": 20,
                              "families": ("R", "B", "Z", "C"),
                              "rho": 0.5},
            seed=seed, budget=15000, output_dir=output_dir,
        ))

    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/async")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir)
    print(f"Built {len(specs)} async-vs-sync specs (seeds={args.seeds})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
