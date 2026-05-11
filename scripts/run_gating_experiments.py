"""H3 ablation: ActiveAssembly vs PassiveAssembly on TypedGated.

TypedGated's optimum sits inside the gating region (half the planted
Boolean target bits are False). Under active assembly, R values at
gated-off positions are masked to 0 and need not be optimised. Under
passive assembly, those same R values feed straight through to fitness
and must be driven to zero — strictly more search effort.

Algorithms:
  - GSA_FULL_ENSEMBLE  (active, ensemble credit)   — H3 PRO
  - GSA_NO_ASSEMBLY    (passive, ensemble credit)  — H3 CON, direct pair
  - GSA_ELITE_CONTEXT  (active, elite credit)      — strong-GSA reference
  - FLATTENED_DE       — flattened baseline (sees same gated fitness)
  - FLATTENED_EA       — flattened baseline
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
    "GSA_NO_ASSEMBLY",
    "GSA_ELITE_CONTEXT",
    "FLATTENED_DE",
    "FLATTENED_EA",
)
DIMS = (20, 40)
BUDGETS = (5000, 15000)


def build_specs(n_seeds: int, output_dir: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for algo, dim, bud, seed in product(
            ALGORITHMS, DIMS, BUDGETS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_gated",
            benchmark_kwargs={"dim": dim, "active_fraction": 0.5,
                              "include_integer": True},
            seed=seed, budget=bud, output_dir=output_dir,
        ))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/gating")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir)
    print(f"Built {len(specs)} gating specs (seeds={args.seeds})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
