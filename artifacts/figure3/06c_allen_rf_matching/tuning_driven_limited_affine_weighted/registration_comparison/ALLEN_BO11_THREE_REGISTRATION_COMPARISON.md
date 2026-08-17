# Allen BO 1.1 three-registration comparison

The middle row applies one translation per session: the median center of that
session's supported V1 RF units is moved to the cross-session median V1 RF center.
The identical translation is applied to V1 and HVA SF/TF maps. There is no rotation,
scale, shear, or tuning-driven fitting in this middle-row registration.

This matches the center-registration concept, but is distinct from the rejected
all-area RF-consensus affine diagnostic, which used multiple area centers and allowed
scale, shear, rotation, and reflection.

| Registration | Group | Map | Median paired Δr versus raw |
| --- | --- | --- | ---: |
| v1_rf_center_translation | HVA pooled | SF | -0.035 |
| v1_rf_center_translation | HVA pooled | TF | -0.028 |
| v1_rf_center_translation | V1 | SF | +0.017 |
| v1_rf_center_translation | V1 | TF | +0.021 |
| tuning_fitted_affine | HVA pooled | SF | +0.333 |
| tuning_fitted_affine | HVA pooled | TF | +0.044 |
| tuning_fitted_affine | V1 | SF | +0.080 |
| tuning_fitted_affine | V1 | TF | +0.014 |
