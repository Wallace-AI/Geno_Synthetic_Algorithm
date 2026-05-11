# BBOB-MixInt sub-report (COCO bbob-mixint suite)

Functions: f1 (Sphere), f8 (Rosenbrock), f15 (Rastrigin), f21 (Gallagher Gauss 101).
Instances: 1, 2, 3. Dim: 10 (8 integer + 2 real). Budget: 100000.
Seeds: 5.

- Input: `results/raw/bbob_mixint_100k/runs.parquet`
- Total runs: 360 (360 completed)

## Mean rank by algorithm

Lower rank = better. Each cell (function x instance) is one ranking; we average across cells.

| algorithm         |   mean_rank |
|:------------------|------------:|
| FLATTENED_DE      |     2.45833 |
| GSA_DIRECT        |     2.75    |
| MIXED_VARIABLE_GA |     2.91667 |
| GSA_ELITE_CONTEXT |     3.375   |
| FLATTENED_EA      |     4.25    |
| GSA_FULL_ENSEMBLE |     5.25    |

## Median final fitness per cell

|                       |   FLATTENED_DE |   FLATTENED_EA |   GSA_DIRECT |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |   MIXED_VARIABLE_GA |
|:----------------------|---------------:|---------------:|-------------:|--------------------:|--------------------:|--------------------:|
| ('f01_sphere', 1)     |        79.48   |      79.48     |    79.48     |           79.48     |            79.4916  |           79.48     |
| ('f01_sphere', 2)     |       394.48   |     394.48     |   394.48     |          394.48     |           394.48    |          394.48     |
| ('f01_sphere', 3)     |      -247.11   |    -247.11     |  -247.11     |         -247.11     |          -247.11    |         -247.11     |
| ('f08_rosenbrock', 1) |         1.4915 |       1.49158  |     1.7203   |            1.7203   |             1.77483 |            1.4915   |
| ('f08_rosenbrock', 2) |        -9.7086 |      -9.99993  |    -9.7712   |           -9.7712   |            -9.77119 |           -9.97025  |
| ('f08_rosenbrock', 3) |         0.9862 |       1.22825  |     1.215    |            1.22325  |             1.25023 |            0.986262 |
| ('f15_rastrigin', 1)  |       100      |     105.875    |   104.503    |          104.299    |           104.153   |          100        |
| ('f15_rastrigin', 2)  |        11.1771 |      11.1787   |    10.842    |           11.7996   |            12.1159  |            7.0049   |
| ('f15_rastrigin', 3)  |        -4.822  |      -0.716747 |    -0.916256 |           -0.404774 |            -0.12536 |           -4.822    |
| ('f21_gallagher', 1)  |        40.78   |      50.4627   |    40.78     |           48.9408   |            42.1728  |           55.9033   |
| ('f21_gallagher', 2)  |        -1.6    |       6.19264  |    -1.6      |           -1.6      |            -1.57408 |           -1.6      |
| ('f21_gallagher', 3)  |      -365.832  |    -370.84     |  -370.84     |         -370.84     |          -370.666   |         -370.84     |

## Paired Wilcoxon: GSA_FULL_ENSEMBLE vs each baseline

Pairings are by (function, instance, seed). A12 > 0.5 means GSA wins on a typical pair.

| reference         | vs                |   median_ref |   median_vs |     p_value |   A12_ref_vs_vs |
|:------------------|:------------------|-------------:|------------:|------------:|----------------:|
| GSA_FULL_ENSEMBLE | FLATTENED_DE      |      1.58489 |     1.4915  | 0.00822198  |        0.473889 |
| GSA_FULL_ENSEMBLE | MIXED_VARIABLE_GA |      1.58489 |     1.4915  | 2.46262e-05 |        0.465278 |
| GSA_FULL_ENSEMBLE | GSA_ELITE_CONTEXT |      1.58489 |     1.6059  | 0.055618    |        0.481944 |
| GSA_FULL_ENSEMBLE | FLATTENED_EA      |      1.58489 |     1.63568 | 0.111811    |        0.491944 |
| GSA_FULL_ENSEMBLE | GSA_DIRECT        |      1.58489 |     1.7203  | 0.0015095   |        0.475278 |

## Figure

- `figures/bbob_mixint_mean_rank.png`
