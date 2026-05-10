"""Larger-budget sweep: tests whether typed-operator advantage compounds.

Hypothesis:
  At budget 5k, GSA_*'s K=5 ensemble overhead burns budget faster than the
  typed-operator advantage compounds (a finding from the paper matrix). At
  larger budgets, the per-iteration cost amortises and GSA's typed
  operators should overtake flattened baselines. We sweep budgets across
  {5k, 15k, 30k} on three multi-family benchmarks.

Cells:
  - typed_additive    D=20 R/B/Z/C       budgets {5k, 15k, 30k}
  - typed_epistatic   D=20 R/B/Z/C rho=0.5  budgets {5k, 15k, 30k}
  - typed_mix         D=24 n_families=4  budgets {5k, 15k, 30k}

Algorithms: 3 GSA variants + 2 flattened baselines.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsa.experiments.runner import RunSpec, run_many


ALGORITHMS = (
    "GSA_FULL_ENSEMBLE",
    "GSA_ELITE_CONTEXT",
    "GSA_DIRECT",
    "FLATTENED_DE",
    "FLATTENED_EA",
)

BUDGETS = (5000, 15000, 30000)


def build_specs(n_seeds: int, output_dir: str) -> list[RunSpec]:
    specs: list[RunSpec] = []

    # Typed additive — separable, smooth. At small budgets flattened wins.
    for algo, bud, seed in product(ALGORITHMS, BUDGETS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_additive",
            benchmark_kwargs={"dim": 20, "families": ("R", "B", "Z", "C")},
            seed=seed, budget=bud, output_dir=output_dir,
        ))

    # Typed epistatic rho=0.5 — cross-family interaction. The regime where
    # type-native operators should help most as budget grows.
    for algo, bud, seed in product(ALGORITHMS, BUDGETS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_epistatic",
            benchmark_kwargs={"dim": 20,
                              "families": ("R", "B", "Z", "C"),
                              "rho": 0.5},
            seed=seed, budget=bud, output_dir=output_dir,
        ))

    # Typed mix n_families=4 — same families as the others for direct
    # comparison.
    for algo, bud, seed in product(ALGORITHMS, BUDGETS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_mix",
            benchmark_kwargs={"dim": 24, "n_families": 4, "rho": 0.0},
            seed=seed, budget=bud, output_dir=output_dir,
        ))

    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/budgets")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir)
    print(f"Built {len(specs)} budget-sweep specs (seeds={args.seeds})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
