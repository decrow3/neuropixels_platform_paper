# Iteration 2 — provisional measured retinotopy

- RF unit mapping: 20,374 / 20,374 units.
- Probe centers: 32 session × probe estimates.
- Position population: 4,807 Pilot-QC units.
- B>C>A>E is strictly descending in RF azimuth in 3/8 sessions and descending allowing ties in 5/8.
- Median absolute Pilot-QC versus default-QC center shift: 0.0 deg x, 0.0 deg y.
- Median per-probe grid-edge fraction: 0.46.
- Gaze correction: none; no gaze-corrected all-session export is currently available.

The measured centers do not support treating B>C>A>E as a universal one-dimensional order. The two-dimensional coordinates should be retained, while the categorical probe view remains a sensitivity analysis.

These raw grid-argmax peaks do not provide final RF significance or area filters and must not be used for `p_value_rf`/`area_rf` selection.

Response-property checkpoint: `mod_idx_dg`, `dark flashes`, population `common_qc`. TTFS is raw relative to NWB start_time.

## Descriptive response-coordinate associations

The correlations below use 32 session × probe observations and are descriptive only; they do not account for the matched probes within sessions.

| Metric | Coordinate | n | Spearman rho |
| --- | --- | ---: | ---: |
| time_to_first_spike_fl | rf_center_x_deg | 32 | 0.025 |
| time_to_first_spike_fl | rf_center_y_deg | 32 | 0.245 |
| mod_idx_dg | rf_center_x_deg | 32 | 0.243 |
| mod_idx_dg | rf_center_y_deg | 32 | 0.155 |
| timescale_ac | rf_center_x_deg | 32 | -0.157 |
| timescale_ac | rf_center_y_deg | 32 | -0.241 |
