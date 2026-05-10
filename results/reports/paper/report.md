# GSA Experiment Report

- Input: `results\raw\paper\runs.parquet`
- Total runs: 285 (265 completed, 20 failed)
- Total wall-clock: 221.1s


## Summary: average rank per algorithm

Lower rank = better. Ranks computed per (benchmark, dim) on median final fitness, then averaged across cells.

| algorithm               |   avg_rank |
|:------------------------|-----------:|
| COOPERATIVE_COEVOLUTION |    1       |
| FLATTENED_EA            |    1.5     |
| FLATTENED_DE            |    2       |
| MIXED_VARIABLE_GA       |    3       |
| GSA_ELITE_CONTEXT       |    3.5     |
| GSA_FULL_ENSEMBLE       |    3.66667 |
| GSA_DIRECT              |    4.33333 |
| GSA_NO_DIVERSITY        |    4.5     |
| RANDOM_FLATTENED        |    4.66667 |
| GSA_NO_ASSEMBLY         |    4.66667 |
| GSA_GENERIC_OPERATORS   |   11       |


## Median final fitness (lower = better)

| benchmark       |   COOPERATIVE_COEVOLUTION |   FLATTENED_DE |   FLATTENED_EA |   GSA_DIRECT |   GSA_ELITE_CONTEXT |   GSA_FULL_ENSEMBLE |   GSA_GENERIC_OPERATORS |   GSA_NO_ASSEMBLY |   GSA_NO_DIVERSITY |   MIXED_VARIABLE_GA |   RANDOM_FLATTENED |
|:----------------|--------------------------:|---------------:|---------------:|-------------:|--------------------:|--------------------:|------------------------:|------------------:|-------------------:|--------------------:|-------------------:|
| ioh:OneMax      |             nan           |    nan         |    0           |   nan        |         nan         |            0        |              nan        |        nan        |         nan        |          nan        |          10        |
| typed_additive  |               1.39252e-06 |      0.0268486 |    0.200005    |     0.313476 |           0.225646  |            0.665433 |                0.784076 |          0.665433 |           0.611262 |            0.040181 |           0.645719 |
| typed_deceptive |             nan           |    nan         |    1.00806     |   nan        |         nan         |            5.45492  |              nan        |          5.45492  |           3.99728  |          nan        |           6.09676  |
| typed_epistatic |             nan           |    nan         |    4.20819e-06 |     0.289297 |           0.0957155 |            0.588239 |              nan        |        nan        |         nan        |          nan        |           0.593336 |
| typed_mix       |             nan           |    nan         |    6.76742e-05 |   nan        |         nan         |            0.555207 |              nan        |        nan        |         nan        |          nan        |           0.40902  |
| typed_noisy     |             nan           |    nan         |    0.158348    |     0.811893 |         nan         |            0.785715 |              nan        |          0.785715 |         nan        |          nan        |           1.0042   |


## Failures (architectural, not algorithm bugs)

Expected failures (flattened baselines cannot encode Cx/E families):
| algorithm        | benchmark   |   count |
|:-----------------|:------------|--------:|
| FLATTENED_EA     | typed_mix   |      10 |
| RANDOM_FLATTENED | typed_mix   |      10 |


## Tables

- `tables/median_iqr_per_cell.md` — main table
- `tables/median_iqr_typed_epistatic.md` — epistasis sweep
- `tables/median_iqr_typed_mix.md` — n_families sweep
- `tables/median_iqr_typed_noisy.md` — noise modes
- `tables/rank_summary.md`
- `tables/wilcoxon_vs_gsa_full.md`


## Figures

- `figures/headline_rank_vs_n_families.png`
- `figures/epistasis_heatmap.png`
- `figures/box_<benchmark>_d<D>.png`
