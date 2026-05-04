"""Phase-1 smoke test: verifies infrastructure (seeds, logging, provenance)
works end-to-end. Does NOT yet exercise GSA logic — that is wired in Task 2.20.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gsa.experiments.logging import RunLogger, RunRecord
from gsa.experiments.provenance import capture_provenance
from gsa.experiments.seed_control import derive_run_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    prov = capture_provenance()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(out_dir / "runs.parquet")
    for seed in range(cfg["n_seeds"]):
        seeds = derive_run_seeds(master_seed=seed)
        rec = RunRecord(
            run_id=f"smoke-{seed}", git_commit=prov.git_commit,
            config_hash="smoke", env_hash=prov.env_hash,
            hardware_fingerprint=prov.hardware_fingerprint,
            algorithm="SMOKE", variant="smoke", benchmark="smoke",
            dim=cfg["dim"], rho=None, n_families=None, noise_mode=None,
            seed_master=seeds.master_seed, evaluation_budget=cfg["budget"],
            final_best_observed=0.0, final_best_true=0.0, auc_convergence=0.0,
            evaluations_to_target=0, target_hit=True,
            total_evaluations=0, total_invalid_offspring=0, total_repairs=0,
            wall_clock_seconds=0.0, peak_memory_mb=0.0,
            status="completed", error_message=None,
        )
        logger.write(rec)
    logger.flush()
    print(f"Smoke OK: wrote {cfg['n_seeds']} records to {out_dir / 'runs.parquet'}")


if __name__ == "__main__":
    main()
