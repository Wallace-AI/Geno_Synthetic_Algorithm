"""Strengthened H3 ablation on TypedGated with Cx family.

Differences from `run_gating_experiments.py`:
- 20 seeds per cell (up from 5) for statistical power.
- Drops the 15k-budget cells where the effect previously washed out.
- Adds D=80 (the larger-D variant where passive has more inactive
  coordinates to drive to zero, sharpening the contrast).
- include_complex=True: planted Cx target makes flat baselines crash
  deterministically, so the section is no longer dominated by
  "flat-DE wins again".
- Flattened baselines excluded; the architectural-reach mechanism is
  documented in §8.3 and tested by `test_flat_baselines_crash_on_complex`.
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
    "GSA_FULL_ENSEMBLE",   # active, ensemble  — H3 pro
    "GSA_NO_ASSEMBLY",     # passive, ensemble — H3 con (direct pair)
    "GSA_ELITE_CONTEXT",   # active, elite credit — strong-GSA reference
    "GSA_DIRECT",          # active, direct credit — fastest GSA reference
)
DIMS = (20, 40, 80)
BUDGET = 5000


def build_specs(n_seeds: int, output_dir: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for algo, dim, seed in product(ALGORITHMS, DIMS, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_gated",
            benchmark_kwargs={"dim": dim, "active_fraction": 0.5,
                              "include_integer": True,
                              "include_complex": True},
            seed=seed, budget=BUDGET, output_dir=output_dir,
        ))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/gating_strong")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir)
    print(f"Built {len(specs)} strengthened gating specs (seeds={args.seeds})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
