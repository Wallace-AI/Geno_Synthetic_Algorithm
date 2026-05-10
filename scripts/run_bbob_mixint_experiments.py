"""Run a small subset of the COCO bbob-mixint suite as an external sanity check.

Functions covered (chosen for diversity of landscape):
  f1  : Sphere (unimodal, separable)
  f8  : Rosenbrock (unimodal, non-separable, valley)
  f15 : Rastrigin (highly multimodal, regular structure)
  f21 : Gallagher Gauss 101 Peaks (multimodal, irregular)

Three instances per function provide instance variance. Dim=10 keeps the
runner fast while still mixing 8 integer + 2 real variables (the BBOB-MixInt
80/20 split).
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsa.experiments.runner import RunSpec, run_many


FUNCTIONS = (1, 8, 15, 21)
INSTANCES = (1, 2, 3)
DIMS = (10,)
ALGORITHMS = (
    "FLATTENED_DE",
    "FLATTENED_EA",
    "MIXED_VARIABLE_GA",
    "GSA_FULL_ENSEMBLE",
    "GSA_ELITE_CONTEXT",
    "GSA_DIRECT",
)


def build_specs(n_seeds: int, output_dir: str, budget: int) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for fn, inst, dim, algo, seed in product(
            FUNCTIONS, INSTANCES, DIMS, ALGORITHMS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo,
            benchmark="coco_mixint",
            benchmark_kwargs={"function": fn, "instance": inst, "dim": dim},
            seed=seed, budget=budget, output_dir=output_dir,
        ))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/bbob_mixint")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir, args.budget)
    print(f"Built {len(specs)} BBOB-MixInt specs (seeds={args.seeds}, "
          f"budget={args.budget})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
