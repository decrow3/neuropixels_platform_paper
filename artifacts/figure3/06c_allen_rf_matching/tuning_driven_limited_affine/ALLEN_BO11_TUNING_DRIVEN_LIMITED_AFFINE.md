# Allen BO 1.1 tuning-driven limited-affine stacking pilot

## Status: deliberately exploratory and in-sample

One orientation-preserving affine transform per session was optimized jointly on
V1 SF, V1 TF, pooled-HVA SF, and pooled-HVA TF maps. The four maps receive equal
objective weight; grid comparisons are weighted by local unit-density evidence.
This directly uses the outcomes whose agreement is reported and is therefore a
visibility/feasibility pilot, not validation or an estimated gaze correction.

Transform bounds: translation ±15°, rotation ±12°, independent scales 0.85–1.15,
and shear ±0.12. Positive scales prohibit reflection.

| Group | Preference | Sessions | Median r raw | Median r aligned | Median paired Δr | Median RMSE raw | Median RMSE aligned | Median paired ΔRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HVA pooled | SF | 31 | 0.117 | 0.361 | +0.360 | 0.273 | 0.270 | -0.016 |
| HVA pooled | TF | 31 | 0.405 | 0.536 | +0.086 | 0.331 | 0.333 | -0.008 |
| V1 | SF | 31 | 0.144 | 0.521 | +0.178 | 0.266 | 0.267 | -0.004 |
| V1 | TF | 31 | 0.651 | 0.555 | -0.081 | 0.150 | 0.157 | +0.010 |

## Boundary behavior

- `translation_azimuth_deg`: 5/31 sessions within 1% of a bound.
- `translation_elevation_deg`: 6/31 sessions within 1% of a bound.
- `rotation_deg`: 6/31 sessions within 1% of a bound.
- `log_scale_azimuth`: 6/31 sessions within 1% of a bound.
- `log_scale_elevation`: 2/31 sessions within 1% of a bound.
- `shear`: 1/31 sessions within 1% of a bound.

Large apparent gains or frequent boundary solutions indicate registration
flexibility rather than biological validation. The aligned templates must not
replace unaligned maps without an independent landmark or held-out replication.

`Figure_allen_bo11_tuning_driven_stacked_templates.png` shows the raw and
tuning-fitted aggregate maps on identical per-panel color scales.
