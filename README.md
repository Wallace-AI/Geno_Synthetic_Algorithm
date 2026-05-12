# Geno-Synthetic Algorithm — experiment battery

Reference implementation and reproducible experiment battery for the
**Geno-Synthetic Algorithm (GSA)**, a type-factored coevolutionary
optimization method that evolves heterogeneous genotypes
(integer, real, Boolean, categorical, complex, embedding) in separate
typed subpopulations using type-native operators, then assembles them
into phenotypes for joint fitness evaluation.

This repository contains the code, raw run logs, and analysis reports
used to produce the empirical results in the GSA paper.

## What's here

- **Core GSA implementation** — typed subpopulations, type-native operators,
  active/passive phenotype assembly, three credit-assignment schemes
  (direct, elite, ensemble), and an asynchronous variant.
- **Eight registered GSA variants** for ablation:
  `GSA_FULL_ENSEMBLE`, `GSA_DIRECT`, `GSA_ELITE_CONTEXT`, `GSA_NO_DIVERSITY`,
  `GSA_GENERIC_OPERATORS`, `GSA_NO_ASSEMBLY`, `GSA_ASYNC`, `GSA_ASYNC_DIRECT`.
- **Flattened baselines** — DE, EA, and Mixed-Variable GA over a single
  decoded representation, for like-for-like comparison.
- **Benchmarks** — typed synthetic suite (additive, epistatic, deceptive,
  noisy, mix), TypedGated (active-assembly stress test), an IOH Boolean
  bridge, and a COCO BBOB-MixInt adapter (`bbob-mixint`).
- **Experiment battery** — programmatic run matrices in `scripts/`,
  parquet logs in `results/raw/`, and rendered reports + figures in
  `results/reports/`.
- **Analysis** — Wilcoxon paired tests, Vargha–Delaney Â12, Friedman with
  Holm correction, rank summaries, and paper-grade plots.

## Quickstart

```bash
python -m venv .venv
# Windows
.venv/Scripts/activate
# Unix
# source .venv/bin/activate

pip install -e .[test]

# Sanity check
python -c "import gsa; print(gsa.__version__)"  # -> 0.1.0
pytest                                           # 164 tests, all green
```

Optional extras for the BBOB-MixInt suite and the IOH Boolean bridge:

```bash
pip install coco-experiment ioh
```

## Reproducing the paper results

Each script writes raw runs to `results/raw/<matrix>/runs.parquet` and a
matching report script renders markdown + figures into
`results/reports/<matrix>/`. All seeds are deterministic.

| Matrix | Run | Report |
| --- | --- | --- |
| Paper headline (6 cells × 5 seeds) | `python scripts/run_paper_experiments.py` | `python scripts/generate_paper_report.py` |
| Async vs sync sub-matrix (80 runs) | `python scripts/run_async_experiments.py` | `python scripts/generate_async_report.py` |
| Larger-budget sweep (225 runs, {5k, 15k, 30k}) | `python scripts/run_budget_experiments.py` | `python scripts/generate_budget_report.py` |
| BBOB-MixInt @ 5k (360 runs) | `python scripts/run_bbob_mixint_experiments.py --budget 5000 --output-dir results/raw/bbob_mixint` | `python scripts/generate_bbob_mixint_report.py --input results/raw/bbob_mixint/runs.parquet --out-dir results/reports/bbob_mixint` |
| BBOB-MixInt @ 100k (360 runs) | `python scripts/run_bbob_mixint_experiments.py --budget 100000 --output-dir results/raw/bbob_mixint_100k` | `python scripts/generate_bbob_mixint_report.py --input results/raw/bbob_mixint_100k/runs.parquet --out-dir results/reports/bbob_mixint_100k --budget 100000` |
| BBOB-MixInt 5k-vs-100k comparison | — | `python scripts/compare_bbob_mixint_budgets.py` + `python scripts/generate_crossover_figure.py` |
| TypedGated H3 ablation, strengthened (240 runs) | `python scripts/run_gating_strengthened.py` | `python scripts/generate_gating_strong_report.py` |

Pre-rendered reports for each matrix are committed under
`results/reports/` so the figures can be browsed on GitHub without
re-running anything.

## Headline findings (calibrated)

- **Architectural reach** is GSA's clearest win: it is the only method
  that runs on Cx/E gene families; flattened baselines crash by encoder
  `ValueError`, deterministically.
- **Active assembly matters when the optimum sits inside the gating
  region** (TypedGated + Cx, n=20, D ∈ {20, 40, 80}): active wins on
  51/60 paired-by-seed comparisons (85%), pooled Wilcoxon
  *p* = 2.9 × 10⁻⁷.
- **Type-native operators matter**: `GSA_GENERIC_OPERATORS` is
  consistently the worst variant.
- **Credit assignment**: `GSA_ELITE_CONTEXT` significantly outperforms
  `GSA_FULL_ENSEMBLE` (*p* = 5.4 × 10⁻⁷, Â12 = 0.0). Default to elite
  credit, not ensemble.
- **Asymptotic crossover on an external suite (BBOB-MixInt)**: at 5k
  evals FLATTENED_DE wins (mean rank 2.17); at 100k evals GSA_DIRECT is
  statistically indistinguishable from FLATTENED_DE (Â12 = 0.499,
  *p* = 0.61) and FLATTENED_EA collapses from rank 2.67 to 4.25.
- **Honest negative result**: at small budgets on smooth multi-family
  synthetic problems, flattened DE wins — the per-iteration K-ensemble
  cost burns budget faster than the typed-operator advantage compounds.

See `results/reports/<matrix>/report.md` for full per-cell statistics.

## Layout

```
src/gsa/
  core/         # typed subgenomes, operators, assemblers, credit schemes
  baselines/    # flattened DE / EA / Mixed-Variable GA
  benchmarks/   # typed synthetic suite, TypedGated, COCO BBOB-MixInt, IOH
  experiments/  # run specification, runner, evaluation budget
  analysis/     # statistics, aggregation, plots, tables
configs/        # experiment YAML configs
scripts/        # run_*.py and generate_*.py entry points
tests/          # 164 pytest tests
results/
  raw/<matrix>/runs.parquet
  reports/<matrix>/{report.md, figures/, tables/, *.csv}
docs/           # design specs, planning notes
```

## Citing

If you use this code, please cite the GSA paper. A `CITATION.cff` will
be added once the arXiv preprint is posted.

## License

MIT — see [LICENSE](LICENSE).
