"""Run the focused experiment matrix that backs the GSA paper.

Generates `results/raw/paper/runs.parquet`. Designed to fit in ~10 min on
8 cores. Cells chosen to test each of H1–H6:
  H1 representation       — Typed Additive D=20 (all 6 GSA + 4 baselines)
  H2 operator-validity    — Same matrix, GSA_GENERIC_OPERATORS as ablation
  H3 assembly             — GSA_NO_ASSEMBLY ablation on Typed Deceptive
  H4 credit-assignment    — Typed Epistatic, rho ∈ {0, 0.5}
  H5 parallelism          — wall-clock recorded; analysis happens offline
  H6 noise robustness     — Typed Noisy, gaussian + heavy-tailed
  Headline figure         — Typed-Mix Gradient n_families ∈ {1..6}
  Boolean (OneMax)        — H1 again, B-only family
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import product

# Make src/ importable when run as a plain script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsa.experiments.runner import RunSpec, run_many


N_SEEDS_DEFAULT = 5


def build_specs(n_seeds: int, output_dir: str) -> list[RunSpec]:
    specs: list[RunSpec] = []

    # Cell 1: Typed Additive D=20, families R/B/Z/C
    full_algos = ["GSA_FULL_ENSEMBLE", "GSA_DIRECT", "GSA_ELITE_CONTEXT",
                  "GSA_NO_DIVERSITY", "GSA_GENERIC_OPERATORS", "GSA_NO_ASSEMBLY",
                  "RANDOM_FLATTENED", "FLATTENED_DE", "FLATTENED_EA",
                  "MIXED_VARIABLE_GA", "COOPERATIVE_COEVOLUTION"]
    for algo, seed in product(full_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_additive",
            benchmark_kwargs={"dim": 20, "families": ("R", "B", "Z", "C")},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell 2: Typed Epistatic D=20, rho ∈ {0.0, 0.5}
    epi_algos = ["GSA_FULL_ENSEMBLE", "GSA_DIRECT", "GSA_ELITE_CONTEXT",
                 "FLATTENED_EA", "RANDOM_FLATTENED"]
    for algo, rho, seed in product(epi_algos, [0.0, 0.5], range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_epistatic",
            benchmark_kwargs={"dim": 20,
                              "families": ("R", "B", "Z", "C"),
                              "rho": rho},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell 3: Typed Deceptive D=20
    dec_algos = ["GSA_FULL_ENSEMBLE", "GSA_NO_DIVERSITY", "GSA_NO_ASSEMBLY",
                 "FLATTENED_EA", "RANDOM_FLATTENED"]
    for algo, seed in product(dec_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_deceptive",
            benchmark_kwargs={"dim": 20},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell 4: Typed Noisy D=20, two noise modes
    noisy_algos = ["GSA_FULL_ENSEMBLE", "GSA_DIRECT", "GSA_NO_ASSEMBLY",
                   "FLATTENED_EA", "RANDOM_FLATTENED"]
    for algo, noise, seed in product(noisy_algos,
                                       ["gaussian", "heavy_tailed"],
                                       range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_noisy",
            benchmark_kwargs={"dim": 20,
                              "families": ("R", "B", "Z", "C"),
                              "noise_mode": noise, "sigma": 0.2,
                              "rho": 0.5},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell 5: Typed-Mix Gradient (the headline figure)
    mix_algos = ["GSA_FULL_ENSEMBLE", "FLATTENED_EA", "RANDOM_FLATTENED"]
    for algo, n_fams, seed in product(mix_algos, range(1, 7), range(n_seeds)):
        # Need at least n_fams in dim allocation: each family gets dim//n_fams
        # coords; minimum 1. Embedding family also needs >=1 coord. Keep dim=24.
        specs.append(RunSpec(
            algorithm=algo, benchmark="typed_mix",
            benchmark_kwargs={"dim": 24, "n_families": n_fams, "rho": 0.0},
            seed=seed, budget=5000, output_dir=output_dir,
        ))

    # Cell 6: OneMax D=50 (Boolean only)
    boolean_algos = ["GSA_FULL_ENSEMBLE", "FLATTENED_EA", "RANDOM_FLATTENED"]
    for algo, seed in product(boolean_algos, range(n_seeds)):
        specs.append(RunSpec(
            algorithm=algo, benchmark="ioh:OneMax",
            benchmark_kwargs={"n": 50},
            seed=seed, budget=15000, output_dir=output_dir,
        ))

    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="results/raw/paper")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially (debug mode).")
    args = parser.parse_args()

    specs = build_specs(args.seeds, args.output_dir)
    print(f"Built {len(specs)} specs (seeds={args.seeds})")
    print(f"Output: {args.output_dir}")
    run_many(specs, parallel=not args.sequential, max_workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
