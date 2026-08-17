# REJECTED correction — Allen BO 1.1 RF-center affine diagnostic

**Status: invalid as a coordinate correction.** RF centers were used both as the
coordinates to transform and as the area-consensus registration landmarks. Holding
SF/TF out of transform fitting does not remove that circular retinotopic assumption.
This output is retained only as a documented failure mode and must not generate
aligned primary maps.

One global affine transform per session was estimated only from RF evidence.
Area-specific RF centers were matched to leave-one-session-out area consensus
centers with square-root unit-count weights. Ridge strength was selected by
held-out-area RF prediction, without consulting SF or TF. The selected lambda
was 0.1.

The unchanged transform was then applied to every V1/HVA unit. Agreement was
evaluated against leave-one-session-out SF/TF templates with weights based on
local unit density and effective reference-session evidence.

| Group | Preference | Sessions | Median r raw | Median r affine | Median Δr (95% bootstrap CI) | Median RMSE raw | Median RMSE affine | Median ΔRMSE (octaves) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HVA pooled | SF | 31 | 0.126 | 0.161 | -0.008 (-0.178, +0.163) | 0.262 | 0.228 | -0.058 |
| HVA pooled | TF | 31 | 0.440 | 0.498 | +0.093 (-0.029, +0.262) | 0.338 | 0.262 | -0.071 |
| V1 | SF | 31 | -0.067 | -0.206 | -0.070 (-0.194, +0.037) | 0.271 | 0.237 | -0.014 |
| V1 | TF | 31 | 0.663 | 0.633 | -0.053 (-0.187, +0.029) | 0.157 | 0.141 | -0.019 |

## Result

Weighted RMSE decreases after affine alignment for all four group-by-frequency
combinations, but spatial-pattern correlation does not improve consistently:
both V1 correlations decline, pooled-HVA SF is essentially unchanged, and only
pooled-HVA TF shows a positive median correlation change. The RF-selected transforms
also compress the visual field strongly. Therefore the RMSE reduction is not accepted
as evidence that a free global affine transform recovers a common tuning map.

## Interpretation boundary

This is a sensitivity analysis, not gaze calibration. A fitted affine transform
can absorb true retinotopic targeting differences as well as screen/eye geometry.
The independent tuning evaluation limits circularity, but area-center consensus
and affine regularization remain modeling assumptions. Improvement should be
accepted only if it is consistent across SF and TF, V1 and HVAs, correlation and
RMSE, without extreme determinants or singular values.

Across sessions, affine determinants ranged -0.377–0.816; singular values ranged 0.009–1.421. 5 session(s) reverse orientation.
A physically constrained translation/rotation/scale sensitivity is required before
using an alignment in the primary maps.
