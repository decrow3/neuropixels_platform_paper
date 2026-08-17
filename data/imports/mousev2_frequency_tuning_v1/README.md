# MouseV2 trial-derived SF x TF tuning

Each unit is fit with a joint Poisson count model using log-Gaussian SF and
TF terms plus an orientation-periodic von Mises term over all 100 conditions.
The empirical 5 x 5 surface remains an orientation-marginal diagnostic.
Tuning is tested from presentation-level
spike counts with a joint 25-cell omnibus F test and separate marginal SF
and TF F tests. Reliability is the correlation between balanced alternating
repeat halves of the 25-cell surface, reported with Spearman-Brown correction.

Support requires dataset-wide BH-FDR q <= 0.05 for the joint and
axis-specific tuning tests, BH-FDR significance for positive reliability,
corrected split-half reliability >= 0.3, parametric
pseudo-R2 >= 0.1, identified widths, and a peak not pinned
to the one-octave extrapolation bound. Peaks outside the tested range are
retained and explicitly flagged.
Preferences that fail this contract are stored as missing and must not be mapped.

| Session | Units | Supported SF | Supported TF |
| --- | ---: | ---: | ---: |
| site2 | 2,732 | 276 (10.1%) | 134 (4.9%) |
| site3 | 2,251 | 371 (16.5%) | 200 (8.9%) |
| site4 | 2,202 | 248 (11.3%) | 190 (8.6%) |
| site5 | 2,996 | 459 (15.3%) | 284 (9.5%) |
| site6 | 2,353 | 248 (10.5%) | 172 (7.3%) |
| site7 | 2,512 | 319 (12.7%) | 214 (8.5%) |
| site8 | 2,925 | 283 (9.7%) | 166 (5.7%) |
| site9 | 2,403 | 430 (17.9%) | 276 (11.5%) |

Files:

- `frequency_tuning_support.csv`: one row per unit, tests, q-values, and gated preferences.
- `site*/frequency_tuning_surface.csv.gz`: one row per unit and SF x TF cell.
- `run_manifest.json`: inputs, thresholds, code, and output hashes.
