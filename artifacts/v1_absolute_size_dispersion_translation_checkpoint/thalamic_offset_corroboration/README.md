# Simultaneous thalamic corroboration of the V1/screen offset

## Test

The previously estimated V1 offsets were held fixed. LGd and LP offsets were then
estimated independently from released thalamic RF centers and three-dimensional
CCF coordinates. For each held-out animal, the thalamic CCF-to-RF geometry was
learned from all other animals with session fixed effects; the held-out session's
robust RF-minus-anatomy residual provided its candidate common screen offset.

LGd and LP were fitted separately. Affine versus quadratic geometry was selected
using median leave-one-animal-out centered block R-squared, without reference to
V1 agreement. A cell-count-weighted LGd/LP estimate was also evaluated.

The primary population required an on-screen RF. A sensitivity population also
required `p_value_rf < 0.01`.

## First limitation: thalamic localization is weak

The released thalamic RF centers do not support precise per-session translations:

- all on-screen units: median held-out centered R-squared was -0.024 for LGd and
  -0.007 for LP; median split-half offset discrepancies were 14.1 and 13.1 degrees;
- p<0.01 units: median held-out centered R-squared improved to 0.092 for LGd and
  0.041 for LP, but median split-half discrepancies remained 14.7 and 12.4 degrees.

Some individual components were more repeatable: all-unit LGd elevation had
split-half Pearson r=0.772, and p<0.01 LGd azimuth had r=0.598. Therefore the
analysis is not uniformly devoid of thalamic signal, but LP and the complementary
LGd axes remain noisy.

## V1 agreement

The combined thalamic offset did not corroborate V1:

| Selection | Axis | Sessions | Pearson r | p | LOO R2 |
|---|---|---:|---:|---:|---:|
| on-screen | azimuth | 40 | -0.115 | 0.481 | -0.091 |
| on-screen | elevation | 40 | 0.077 | 0.635 | -0.084 |
| p<0.01 on-screen | azimuth | 38 | -0.025 | 0.882 | -0.092 |
| p<0.01 on-screen | elevation | 38 | 0.097 | 0.563 | -0.079 |

All four leave-one-session-out predictions were worse than a constant in squared
error. Separate LGd and LP estimates were also inconsistent. Notably, even the
relatively repeatable all-unit LGd elevation component had essentially zero V1
association (r=-0.068), while p<0.01 LGd azimuth had only r=0.191.

## Concrete cases

- `798911424` is a local success: its p<0.01 LP estimate (-0.6, +7.0 degrees)
  closely matches V1 (+1.1, +6.2 degrees), with positive held-out thalamic
  progression R-squared (0.134).
- `760345702` fails despite positive thalamic progression R-squared (0.105): V1
  is (+4.3, -10.5 degrees), whereas combined thalamus is (-17.7, +15.2 degrees).
- `771990200` also points oppositely, and its thalamic progression itself is not
  supported out of sample.

Thus the `798911424` agreement is not a general cross-animal phenomenon.

## Interpretation

The released LGd/LP measurements cannot currently be used as an absolute
registration anchor. The result also cautions against interpreting the extremely
repeatable V1 RF-to-CCF residual as pure eye/screen translation: if a large common
screen shift dominated it, some concordance with independently sampled thalamic
RFs would be expected. The V1 offset probably contains substantial animal-specific
cortical retinotopic geometry, while the thalamic estimate contains additional
sampling and RF-fit noise.

The cleanest remaining version of this test would refit thalamic RFs from raw
Gabor responses using the improved RF model and then repeat the frozen analysis.
That is materially more expensive because current improved-fit caches contain
cortical, not thalamic, populations.

## Outputs

- `Figure_v1_thalamic_offset_corroboration.png`: full population comparison.
- `Figure_concrete_thalamic_cases.png`: three preregistered concrete cases.
- `thalamic_geometry_model_audit.csv`: independent geometry/model audit.
- `thalamic_offset_split_half.csv`: thalamic repeatability.
- `v1_thalamic_corroboration_stats.csv`: correlation and held-out prediction.
- `concrete_cases.csv`: case-level offsets.
- nucleus- and selection-specific offset/comparison CSV files.
