# Asynchronous vs Synchronous GSA

- Input: `results/raw/async/runs.parquet`
- Total runs: 80 (80 completed)

## Median final fitness (lower = better)

|                                |   GSA_ASYNC |   GSA_ASYNC_DIRECT |   GSA_DIRECT |   GSA_FULL_ENSEMBLE |
|:-------------------------------|------------:|-------------------:|-------------:|--------------------:|
| ('typed_additive', 20, 5000)   |    0.744381 |           0.531112 |    0.313476  |            0.665433 |
| ('typed_epistatic', 20, 5000)  |    0.459405 |           0.295487 |    0.230263  |            0.473841 |
| ('typed_epistatic', 20, 15000) |    0.321803 |           0.114681 |    0.0772126 |            0.205815 |
| ('typed_mix', 24, 5000)        |    1.37555  |           1.08077  |    0.989187  |            1.38941  |

## Median wall-clock seconds per run

| benchmark       |   GSA_ASYNC |   GSA_ASYNC_DIRECT |   GSA_DIRECT |   GSA_FULL_ENSEMBLE |
|:----------------|------------:|-------------------:|-------------:|--------------------:|
| typed_additive  |     0.75829 |            3.53323 |      1.99818 |            0.507768 |
| typed_epistatic |     1.87585 |            5.64316 |      4.56807 |            1.41805  |
| typed_mix       |     1.74466 |            8.98495 |      4.36691 |            1.18737  |

## Paired Wilcoxon: SYNC vs ASYNC (A12 > 0.5 means sync wins on a typical seed)

| benchmark       |   dim |   budget | sync              | async            |   median_sync |   median_async |   p_value |   A12_sync_vs_async |
|:----------------|------:|---------:|:------------------|:-----------------|--------------:|---------------:|----------:|--------------------:|
| typed_additive  |    20 |     5000 | GSA_FULL_ENSEMBLE | GSA_ASYNC        |     0.665433  |       0.744381 |    0.25   |                0.64 |
| typed_epistatic |    20 |     5000 | GSA_FULL_ENSEMBLE | GSA_ASYNC        |     0.473841  |       0.459405 |    0.625  |                0.52 |
| typed_epistatic |    20 |    15000 | GSA_FULL_ENSEMBLE | GSA_ASYNC        |     0.205815  |       0.321803 |    0.0625 |                0.88 |
| typed_mix       |    24 |     5000 | GSA_FULL_ENSEMBLE | GSA_ASYNC        |     1.38941   |       1.37555  |    0.625  |                0.44 |
| typed_additive  |    20 |     5000 | GSA_DIRECT        | GSA_ASYNC_DIRECT |     0.313476  |       0.531112 |    0.0625 |                0.88 |
| typed_epistatic |    20 |     5000 | GSA_DIRECT        | GSA_ASYNC_DIRECT |     0.230263  |       0.295487 |    0.3125 |                0.76 |
| typed_epistatic |    20 |    15000 | GSA_DIRECT        | GSA_ASYNC_DIRECT |     0.0772126 |       0.114681 |    0.3125 |                0.72 |
| typed_mix       |    24 |     5000 | GSA_DIRECT        | GSA_ASYNC_DIRECT |     0.989187  |       1.08077  |    0.0625 |                0.84 |

## Figure

- `figures/async_vs_sync_medians.png`
