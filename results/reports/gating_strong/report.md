# Strengthened H3 sub-report: active vs passive on TypedGated + Cx

TypedGated's planted Boolean target is half-True/half-False, placing the optimum inside the gating region. `include_complex=True` adds a Cx subgenome with its own planted target — flat baselines cannot represent this and crash deterministically (see `test_flat_baselines_crash_on_complex`).

- Input: `results/raw/gating_strong/runs.parquet`
- Total runs: 240 (240 completed, 0 failed)
- 4 GSA variants, 3 dims (20, 40, 80), 20 seeds, budget 5000.

## Median final fitness per (dim, algorithm)

|   dim |   GSA_DIRECT |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |   GSA_NO_ASSEMBLY |
|------:|-------------:|--------------------:|--------------------:|------------------:|
|    20 |     0.577172 |            0.410255 |            0.717271 |          0.732856 |
|    40 |     0.779135 |            0.627584 |            0.84409  |          0.875505 |
|    80 |     0.930425 |            0.823706 |            0.956539 |          0.983254 |

## H3 ablation: GSA_FULL_ENSEMBLE (active) vs GSA_NO_ASSEMBLY (passive)

Paired by seed within each cell. A12 > 0.5 means active wins on a typical paired seed.

|   dim |   n_pairs |   active_median |   passive_median |   active_wins_paired |   passive_wins_paired |   ties_paired |   p_value_wilcoxon |   A12_active_vs_passive |
|------:|----------:|----------------:|-----------------:|---------------------:|----------------------:|--------------:|-------------------:|------------------------:|
|    20 |        20 |        0.717271 |         0.732856 |                   16 |                     4 |             0 |        0.0192337   |                  0.6025 |
|    40 |        20 |        0.84409  |         0.875505 |                   18 |                     2 |             0 |        0.000394821 |                  0.73   |
|    80 |        20 |        0.956539 |         0.983254 |                   17 |                     3 |             0 |        0.000394821 |                  0.68   |

**Pooled paired stats:** n=60 pairs, p=2.88e-07, A12=0.562, active wins 51/60 (85%)

## Figure

- `figures/active_vs_passive_strong.png`
