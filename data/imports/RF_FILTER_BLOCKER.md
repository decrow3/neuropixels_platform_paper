# RF filter export required for Figure 3

_Audit frozen 2026-08-05_

## Decision

Iteration 4B (`published_like`) and Iteration 4C (`rf_area`) must not run for
MouseV2 yet. The closest current all-session RF import contains peak coordinates
and a diagnostic quality flag, but it does not contain a validated receptive-field
significance p-value or RF area in degrees squared. Substituting that diagnostic
flag would silently change the released Figure 3 population definition.

The grating-rate part of the released population filter is available here:
`firing_rate_dg` is the preferred-condition mean spike count divided by the
validated nominal 1.0-second analysis duration. The remaining blockers are RF
significance and RF area.

## Evidence from the neighboring repository

- The only batch `outputs/.../rf_metrics.csv` found in PilotAnalysis is for
  subject 810531. Its columns are peak diagnostics (`peak_x`, `peak_y`,
  `peak_value`, `compactness`, `peak_to_noise`, and `total_spikes`), not the
  required RF significance and area fields.
- `PilotAnalysis/compute_stimulus_metrics.py` explicitly writes
  `p_value_rf = NaN`.
- Its current `area_rf` is the number of grid cells above a fixed fraction of
  the unit's maximum response multiplied by grid-cell area. That exploratory
  definition has not been benchmarked as Allen-equivalent.
- The versioned all-session import in this repository,
  `pilot_rf_peaks_v1`, maps all 20,374 units but uses unsmoothed per-unit spike
  count argmax coordinates with no gaze correction. Its stricter Pilot RF
  diagnostic QC retains 4,807 units; this is not a published-like filter.

## Work required in PilotAnalysis

1. Implement and test an RF significance statistic that respects the Gabor
   trial structure, preferably a trial-label permutation test across position
   and orientation.
2. Define RF area in degrees squared, including response smoothing/modeling,
   thresholding, edge handling, and invalid-fit behavior.
3. Benchmark significance and area on compatible Allen data or against the
   original Allen implementation; document any unavoidable accommodation.
4. Decide whether the paper-facing RF uses raw or gaze-corrected stimulus
   coordinates, and export that choice explicitly.
5. Export all eight sessions with one row per unit and at least:
   `subject_id`, `local_unit_id`, `probe`, `rf_center_x_deg`,
   `rf_center_y_deg`, `area_rf_deg2`, `p_value_rf`, `has_significant_rf`,
   `rf_method`, and `gaze_correction`.
6. Write a provenance sidecar containing the PilotAnalysis commit, NWB identity,
   parameters, schema version, and creation time.

## Work required in this repository after export

1. Import by `subject_id + local_unit_id`, map to the stable offset `unit_id`,
   and reject duplicates or unmapped rows.
2. Validate p-value range, nonnegative area, units, method/gaze fields, probe
   labels, all-session coverage, and missingness by session and probe.
3. Regenerate the population flow before applying the mask.
4. Run `04b_published_like` using exactly `p_value_rf < 0.01`,
   `area_rf < 2500`, `snr > 1`, and `firing_rate_dg > 0.1`.
5. Run `04c_rf_area` only after the area-method benchmark is accepted.
6. Review session balance and sensitivity against `04a_common_qc` before
   selecting a primary population or updating the claim.

## Acceptance condition

The blocker is resolved only when a versioned, validated all-session export can
compute the named MouseV2 `published_like` mask without weakening, renaming, or
imputing any required criterion.
