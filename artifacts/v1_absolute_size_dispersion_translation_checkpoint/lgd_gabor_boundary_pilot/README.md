# Raw LGd Gabor boundary-map checkpoint

## Purpose

This is the concrete-case checkpoint before fitting a 33-session LGd anatomical
map. It tests whether raw Gabor responses contain held-out spatial information
for LGd units whose RF centers are censored by the stimulus boundary.

Three sessions were selected before raw fitting:

- `755434585`: strong previous held-out LGd anatomical progression;
- `760345702`: typical anatomical span;
- `754829445`: broad two-probe coverage but failed previous atlas prediction.

All LGd units with CCF coordinates were included. Selection did not use Allen's
released RF significance or `on_screen_rf` flag.

## Method

For every unit, spike counts were extracted for all 3,645 Gabor presentations.
Repeats of each position/orientation condition were deterministically divided
into train and test sets. An analytic-aperture Gaussian RF was fitted on training
presentations, with its center allowed to extend 20 degrees beyond the sampled
9x9 grid. Held-out spatial gain is the test-set Poisson-deviance improvement over
an orientation-only model.

An off-screen fitted center is treated as usable boundary evidence only when its
held-out spatial gain is positive. Fits with zero/negative gain are retained as
controls but must not enter the anatomical map as point centers.

## Results

| Session | LGd units | Positive held-out spatial gain | Positive boundary centers | Positive units absent from released clean set |
|---|---:|---:|---:|---:|
| 755434585 | 93 | 19 | 1 | 10 |
| 760345702 | 78 | 20 | 5 | 14 |
| 754829445 | 170 | 43 | 16 | 29 |

Across the three sessions, 82/341 units carried held-out spatial information;
22 had fitted centers beyond the sampled grid. Fifty-three of the 82 useful raw
fits were not in the corresponding significant/on-screen released-RF set. Twenty
of the 22 boundary fits were likewise unavailable to the previous clean-center
analysis.

The concrete raw maps show the intended distinction:

- strong interior units have localized peaks and finite held-out gains;
- strong boundary units show rising response flanks at a grid edge and retain
  positive held-out gain despite an extrapolated center;
- nonresponsive controls can run to a parameter corner with zero gain and are
  therefore excluded from map-center evidence.

The multi-probe failure case `754829445` contains the most boundary information:
16 held-out-positive off-screen centers. Its two tracks also show visibly
different retinotopic ranges, making it particularly important for identifying
transverse LGd geometry.

## Interpretation

The raw boundary likelihood is viable and recovers information discarded by
`on_screen_rf`. It is not an independent biological variable—the same Gabor
responses determine RF location—but it is a materially better observation model
for constructing the LGd map.

This checkpoint does not yet establish a cross-session LGd map. The next stage is
to extract and fit all 33 LGd sessions, then fit a smooth CCF-to-RF field with
session translations. Model smoothness must be chosen from held-out response
prediction and within-session RF differences; raw-fit centers with nonpositive
held-out gain must not be treated as coordinates.

## Outputs

- one directory per session containing compact raw counts, unit fits, selected
  examples, and `Figure_LGd_raw_boundary_maps.png`;
- `session_summary.csv`;
- `selected_examples.csv`.
