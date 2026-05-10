# Pairwise comparison vs. GSA_FULL_ENSEMBLE

A12 > 0.5 means GSA_FULL_ENSEMBLE wins on a typical seed.

| algorithm               |     p_raw |    p_holm |   A12 |
|:------------------------|----------:|----------:|------:|
| GSA_DIRECT              | 0.003907  | 0.03126   | 0.335 |
| GSA_ELITE_CONTEXT       | 5.96e-08  | 5.364e-07 | 0     |
| GSA_NO_DIVERSITY        | 0.375     | 1         | 0.42  |
| GSA_GENERIC_OPERATORS   | 0.125     | 0.5       | 0.82  |
| GSA_NO_ASSEMBLY         | 1         | 1         | 0.5   |
| RANDOM_FLATTENED        | 0.8127    | 1         | 0.505 |
| FLATTENED_DE            | 0.0625    | 0.4375    | 0     |
| FLATTENED_EA            | 5.285e-27 | 5.285e-26 | 0.172 |
| MIXED_VARIABLE_GA       | 0.0625    | 0.4375    | 0     |
| COOPERATIVE_COEVOLUTION | 0.0625    | 0.4375    | 0     |