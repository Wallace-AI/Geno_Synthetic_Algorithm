# H3 sub-report: ActiveAssembly vs PassiveAssembly on TypedGated

TypedGated's optimum sits inside the gating region (half the planted Boolean target bits are False). Under active assembly the R values at gated-off positions are masked to 0 and need not be optimised; under passive assembly they feed straight through to fitness and must be driven to zero.

- Input: `results/raw/gating/runs.parquet`
- Total runs: 100 (100 completed)

## Median final fitness per (dim, budget, algorithm)

|             |   FLATTENED_DE |   FLATTENED_EA |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |   GSA_NO_ASSEMBLY |
|:------------|---------------:|---------------:|--------------------:|--------------------:|------------------:|
| (20, 5000)  |     0.122333   |    0.0129677   |            0.222133 |            0.461718 |          0.485143 |
| (20, 15000) |     0.00586867 |    2.97214e-06 |            0.090177 |            0.330285 |          0.327636 |
| (40, 5000)  |     0.216525   |    0.151137    |            0.392509 |            0.620572 |          0.654198 |
| (40, 15000) |     0.0838784  |    0.00523446  |            0.200578 |            0.540703 |          0.530193 |

## Paired Wilcoxon: active vs passive

Paired by seed within each (dim, budget) cell. A12 > 0.5 means active wins on a typical seed.

|   dim |   budget |   active_median |   passive_median |   p_value |   A12_active_vs_passive |
|------:|---------:|----------------:|-----------------:|----------:|------------------------:|
|    20 |     5000 |        0.461718 |         0.485143 |    0.125  |                    0.84 |
|    20 |    15000 |        0.330285 |         0.327636 |    0.625  |                    0.44 |
|    40 |     5000 |        0.620572 |         0.654198 |    0.1875 |                    0.76 |
|    40 |    15000 |        0.540703 |         0.530193 |    1      |                    0.44 |

## Figure

- `figures/active_vs_passive.png`
