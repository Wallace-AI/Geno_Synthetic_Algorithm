# Larger-budget sub-report

Hypothesis: at larger budgets, GSA's per-iteration ensemble cost amortises and typed operators should overtake flattened baselines.

- Input: `results/raw/budgets/runs.parquet`
- Total runs: 225 (225 completed)

## Median final fitness per (benchmark, budget, algorithm)

|                            |   FLATTENED_DE |   FLATTENED_EA |   GSA_DIRECT |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |
|:---------------------------|---------------:|---------------:|-------------:|--------------------:|--------------------:|
| ('typed_additive', 5000)   |    0.0268486   |    0.200005    |  0.313476    |         0.225646    |           0.665433  |
| ('typed_additive', 15000)  |    2.57811e-07 |    0.2         |  0.105882    |         1.27386e-06 |           0.363709  |
| ('typed_additive', 30000)  |    3.45461e-26 |    0.2         |  0.0453197   |         2.15767e-12 |           0.0990171 |
| ('typed_epistatic', 5000)  |    9.00359e-05 |    8.92783e-06 |  0.230263    |         0.0906949   |           0.473841  |
| ('typed_epistatic', 15000) |    1.53998e-15 |    3.82457e-07 |  0.0772126   |         2.07823e-07 |           0.205815  |
| ('typed_epistatic', 30000) |    0           |    3.08969e-08 |  0.000623065 |         1.28601e-13 |           0.0712244 |
| ('typed_mix', 5000)        |    0.00106896  |    1.15941e-05 |  0.597647    |         0.195429    |           0.750495  |
| ('typed_mix', 15000)       |    6.40496e-11 |    3.91306e-07 |  0.320493    |         1.15375e-06 |           0.556294  |
| ('typed_mix', 30000)       |    2.8523e-27  |    3.44534e-08 |  0.0905433   |         7.07863e-12 |           0.25095   |

## Mean rank across (benchmark, budget) cells

| algorithm         |   mean_rank |
|:------------------|------------:|
| FLATTENED_DE      |     1.22222 |
| GSA_ELITE_CONTEXT |     2.44444 |
| FLATTENED_EA      |     2.66667 |
| GSA_DIRECT        |     3.77778 |
| GSA_FULL_ENSEMBLE |     4.88889 |

## Paired Wilcoxon: A12 < 0.5 means A wins (lower fitness) on a typical pair

Pairings are by (benchmark, seed) at each budget level.

|   budget | a                 | b            |    median_a |    median_b |     p_value |   A12_a_vs_b |
|---------:|:------------------|:-------------|------------:|------------:|------------:|-------------:|
|     5000 | GSA_ELITE_CONTEXT | FLATTENED_DE | 0.162754    | 0.00086409  | 6.10352e-05 |    0.0222222 |
|    15000 | GSA_ELITE_CONTEXT | FLATTENED_DE | 4.23804e-07 | 1.3004e-09  | 0.229309    |    0.226667  |
|    30000 | GSA_ELITE_CONTEXT | FLATTENED_DE | 4.53316e-12 | 4.42926e-29 | 6.10352e-05 |    0         |
|     5000 | GSA_ELITE_CONTEXT | FLATTENED_EA | 0.162754    | 2.05219e-05 | 0.0255737   |    0.208889  |
|    15000 | GSA_ELITE_CONTEXT | FLATTENED_EA | 4.23804e-07 | 3.91306e-07 | 0.561401    |    0.613333  |
|    30000 | GSA_ELITE_CONTEXT | FLATTENED_EA | 4.53316e-12 | 3.44534e-08 | 6.10352e-05 |    1         |
|     5000 | GSA_FULL_ENSEMBLE | FLATTENED_DE | 0.658933    | 0.00086409  | 6.10352e-05 |    0         |
|    15000 | GSA_FULL_ENSEMBLE | FLATTENED_DE | 0.320526    | 1.3004e-09  | 6.10352e-05 |    0         |
|    30000 | GSA_FULL_ENSEMBLE | FLATTENED_DE | 0.127892    | 4.42926e-29 | 6.10352e-05 |    0         |
|     5000 | GSA_DIRECT        | FLATTENED_DE | 0.313476    | 0.00086409  | 6.10352e-05 |    0         |
|    15000 | GSA_DIRECT        | FLATTENED_DE | 0.105882    | 1.3004e-09  | 6.10352e-05 |    0         |
|    30000 | GSA_DIRECT        | FLATTENED_DE | 0.0287103   | 4.42926e-29 | 6.10352e-05 |    0         |

## Figure

- `figures/budget_sweep.png`
