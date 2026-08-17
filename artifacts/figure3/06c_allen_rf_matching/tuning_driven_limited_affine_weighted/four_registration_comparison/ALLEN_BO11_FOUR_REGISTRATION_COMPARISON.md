# Allen BO 1.1 four-registration comparison

The middle row applies one translation per session: the median center of that
session's supported V1 RF units is moved to the cross-session median V1 RF center.
The identical translation is applied to V1 and HVA SF/TF maps. There is no rotation,
scale, shear, or tuning-driven fitting in this middle-row registration.

This matches the center-registration concept, but is distinct from the rejected
all-area RF-consensus affine diagnostic, which used multiple area centers and allowed
scale, shear, rotation, and reflection.

The fourth row uses the **ccf_to_v1_rf_translation** model on the 23/31 sessions with reconstructed V1 CCF coordinates.
For each session, a robust session-balanced CCF→RF model was learned from V1 units in all other sessions.
The median held-out V1 prediction residual defines one translation shared by V1 and simultaneous HVA maps.
The comparison restricts every row to the same CCF-available sessions. RF size, HVA units, SF, and TF were not used to fit or select row 4.

| Registration | Group | Map | Median paired Δr versus raw |
| --- | --- | --- | ---: |
| v1_rf_center_translation | HVA pooled | SF | -0.003 |
| v1_rf_center_translation | HVA pooled | TF | -0.039 |
| v1_rf_center_translation | V1 | SF | +0.022 |
| v1_rf_center_translation | V1 | TF | +0.057 |
| tuning_fitted_affine | HVA pooled | SF | +0.221 |
| tuning_fitted_affine | HVA pooled | TF | +0.024 |
| tuning_fitted_affine | V1 | SF | +0.113 |
| tuning_fitted_affine | V1 | TF | +0.085 |
| ccf_to_v1_rf_translation | HVA pooled | SF | -0.019 |
| ccf_to_v1_rf_translation | HVA pooled | TF | -0.057 |
| ccf_to_v1_rf_translation | V1 | SF | -0.087 |
| ccf_to_v1_rf_translation | V1 | TF | +0.033 |
