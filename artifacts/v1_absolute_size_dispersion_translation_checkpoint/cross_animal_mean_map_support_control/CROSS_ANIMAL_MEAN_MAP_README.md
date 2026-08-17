# Leave-one-animal-out V1 mean-map support control

## Question

Does the exact anatomical support of a V1 recording inflate local RF covariance
once the expected mean retinotopic progression is learned independently of the
target animal?

## Identification

Each of the 54 animals contributes one session and almost always one V1 probe, so
penetration medians alone cannot distinguish a CCF map from an animal-specific RF
translation. Each trajectory was therefore divided into six physical blocks and
block median CCF and RF coordinates were calculated. The population CCF-to-RF
geometry was fitted after centering both predictors and RF coordinates within
session. This removes an arbitrary RF translation per animal while retaining
within-trajectory gradients. Every target specimen was excluded from fitting.

Affine and quadratic AP/ML CCF models were fitted with robust, session-balanced
regression. The quadratic model is displayed; the affine model is a sensitivity
analysis. No target-animal RF value contributes to its predicted gradient.
Translations shown in the first figure column are added only to overlay predicted
and observed block locations. They do not affect covariance.

## Concrete cases

| Session | Selection role | Quadratic gradient R2 vs constant | Median sampling fraction |
|---|---|---:|---:|
| 760345702 | Previous covariance-trace success | 0.58 | 3.7% |
| 719161530 | Previous typical case | 0.29 | 2.1% |
| 835479236 | Previous failure / strongest CCF association | 0.06 | 0.5% |

Affine results are similar: gradient R2 values are 0.62, 0.19, and 0.07, and
sampling fractions are 4.4%, 3.1%, and 0.7%.

## Initial interpretation

The independent population geometry predicts meaningful held-out block gradients
in the prior success and typical cases. The expected covariance caused by their
exact anatomical support is nevertheless only a few percent of raw covariance.

The failure case contains a separated low-depth block whose median azimuth is
about -10 degrees, whereas the remaining blocks are about 35-52 degrees. Its CCF
displacement is far too small for the learned population map to predict that RF
jump. Consequently, the external geometry does not validate the target gradient
and does not classify the large observed dispersion as a trivial sampling effect.
This may reflect an RF outlier population, an unmodeled anatomical/area boundary,
or an animal-specific map distortion; this checkpoint does not distinguish them.

The earlier result that removed all shank-predictable covariance remains
superseded. Smooth anatomy-linked dispersion is not a nuisance. In these three
cases, the narrower independently estimated support correction is small.

## Limits

- The common geometry is identified from within-probe directional gradients, not
  independent two-dimensional maps within each animal.
- AP/ML CCF is treated as tangential cortex; surface curvature and depth-dependent
  projection are not yet modeled.
- Six block medians reduce single-cell scatter but do not eliminate it.
- Three preregistered cases are not a population estimate.

## Outputs

- `Figure_v1_cross_animal_mean_map_support_cases.png`
- `Figure_v1_cross_animal_model_comparison.png`
- `heldout_session_model_audit.csv`
- `heldout_block_predictions.csv`
- `heldout_unit_support_decomposition.csv.gz`
- `heldout_model_coefficients.csv`
- `analysis_metadata.json`

Reproduce from the repository root with:

```bash
python -m scripts.check_v1_cross_animal_mean_map_support --overwrite
```
