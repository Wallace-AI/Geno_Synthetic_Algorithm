# BBOB-MixInt sub-report (COCO bbob-mixint suite)

Functions: f1 (Sphere), f8 (Rosenbrock), f15 (Rastrigin), f21 (Gallagher Gauss 101).
Instances: 1, 2, 3. Dim: 10 (8 integer + 2 real). Budget: 5000.
Seeds: 5.

- Input: `results/raw/bbob_mixint/runs.parquet`
- Total runs: 360 (360 completed)

## Mean rank by algorithm

Lower rank = better. Each cell (function x instance) is one ranking; we average across cells.

| algorithm         |   mean_rank |
|:------------------|------------:|
| FLATTENED_DE      |     2.16667 |
| FLATTENED_EA      |     2.66667 |
| MIXED_VARIABLE_GA |     2.83333 |
| GSA_DIRECT        |     3.25    |
| GSA_ELITE_CONTEXT |     4.08333 |
| GSA_FULL_ENSEMBLE |     6       |

## Median final fitness per cell

|                       |   FLATTENED_DE |   FLATTENED_EA |   GSA_DIRECT |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |   MIXED_VARIABLE_GA |
|:----------------------|---------------:|---------------:|-------------:|--------------------:|--------------------:|--------------------:|
| ('f01_sphere', 1)     |       79.48    |      79.48     |     79.4875  |            79.703   |            81.8271  |           79.5126   |
| ('f01_sphere', 2)     |      394.48    |     394.48     |    394.482   |           394.481   |           399.229   |          394.485    |
| ('f01_sphere', 3)     |     -247.11    |    -247.11     |   -247.109   |          -247.11    |          -244.763   |         -247.099    |
| ('f08_rosenbrock', 1) |        1.49153 |       1.49741  |      1.78434 |             2.67032 |             7.72901 |            1.76453  |
| ('f08_rosenbrock', 2) |       -8.21495 |      -9.99808  |     -9.13899 |            -8.95341 |            -3.076   |           -9.68305  |
| ('f08_rosenbrock', 3) |        1.24087 |       1.27763  |      1.21718 |             1.83571 |            13.9872  |            1.22232  |
| ('f15_rastrigin', 1)  |      100.606   |     106.238    |    105.78    |           104.892   |           109.585   |          104.696    |
| ('f15_rastrigin', 2)  |       11.3267  |      11.6635   |     12.7101  |            12.981   |            14.209   |           12.0626   |
| ('f15_rastrigin', 3)  |       -0.48054 |      -0.598791 |      1.54885 |             1.10582 |             4.23893 |           -0.723611 |
| ('f21_gallagher', 1)  |       40.78    |      54.4517   |     40.8834  |            56.6976  |            58.9412  |           55.9788   |
| ('f21_gallagher', 2)  |       -1.11274 |       7.08227  |     -1.14364 |            -0.64344 |            10.0221  |           -1.58204  |
| ('f21_gallagher', 3)  |     -364.389   |    -370.84     |   -370.768   |          -370.22    |          -362.52    |         -370.829    |

## Paired Wilcoxon: GSA_FULL_ENSEMBLE vs each baseline

Pairings are by (function, instance, seed). A12 > 0.5 means GSA wins on a typical pair.

| reference         | vs                |   median_ref |   median_vs |     p_value |   A12_ref_vs_vs |
|:------------------|:------------------|-------------:|------------:|------------:|----------------:|
| GSA_FULL_ENSEMBLE | FLATTENED_DE      |      13.3285 |     1.4915  | 2.5648e-11  |        0.403611 |
| GSA_FULL_ENSEMBLE | MIXED_VARIABLE_GA |      13.3285 |     1.74941 | 1.62956e-11 |        0.403056 |
| GSA_FULL_ENSEMBLE | GSA_DIRECT        |      13.3285 |     1.78434 | 1.62956e-11 |        0.413889 |
| GSA_FULL_ENSEMBLE | GSA_ELITE_CONTEXT |      13.3285 |     2.65684 | 3.39779e-10 |        0.413333 |
| GSA_FULL_ENSEMBLE | FLATTENED_EA      |      13.3285 |     3.02476 | 1.88908e-09 |        0.415556 |

## Figure

- `figures/bbob_mixint_mean_rank.png`
