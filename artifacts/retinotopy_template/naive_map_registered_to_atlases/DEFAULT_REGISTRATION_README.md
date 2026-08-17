# Default registration (locked)

This is the fallback registration between recorded RF/CCF data and the Zhuang et al. (2017)
population retinotopic map, as of the per-session offset + iterative span-match calibration.
See `reports/ATLAS_ANCHORED_TRANSLATION_REGISTRATION_NOTES.md` for the full narrative; this
file just pins down the exact locked numbers and file pointers.

## Fixed anatomical registration (shared across all sessions)

CCF is already a common anatomical frame, so this part is a one-time coordinate-frame
calibration, not a per-session fit. Values from `translation_rotation_fit_manifest.json`:

- rotation: -8.1 deg
- translation: (+41.1, -2.4) px, on top of the V1-anchor placement
- scale: 104.6 px/mm (Zhuang Figure 3 scale bar), reflection: left-right mirrored (locked)
- V1 anchor: session data's own V1 median CCF position <-> Zhuang seed pixel (200, 240)

## Span-matched Zhuang field (converged, per `iteratively_span_match_and_fit_per_session_offsets.py`)

Zhuang's raw field is compressed relative to the real data on both axes; gain is a linear
stretch anchored at the V1 seed pixel value (`rescaled = anchor + gain * (raw - anchor)`).
The gain must be computed against the SAME per-session-registered cells it will be used to
register (circular), so it was solved iteratively to convergence (3 iterations):

- azimuth gain: 1.159
- elevation gain: 1.416

File: `artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa/interpolated_fields_and_field_sign_domain_patched_span_matched.npz`
(keys `azimuth_span_matched_deg`, `elevation_span_matched_deg`)

## Per-session RF offset

Only the RF offset (gaze/eye-position) varies per session -- a robust (Huber) location fit per
session, capped in deviation-magnitude from the pooled offset at the 85th percentile of the
observed spread (direction preserved, magnitude clipped) so no single session can dominate.
Pooled offset: azimuth +39.5 deg, elevation -5.1 deg. 7/45 sessions capped; all sessions had
sufficient support (>=20 valid cells).

File: `per_session_rf_offset/per_session_rf_offset.csv` (+ `per_session_offset_manifest.json`)

## Reference figure

`Figure_default_registration_all_cells_over_zhuang.png` -- all-session pooled cells, each
corrected by its own session's final offset, over the converged span-matched Zhuang background.

## Known limitation / open next step

This registration treats the retinotopic MAP SHAPE as identical across sessions (only a
uniform per-session offset varies). The user has flagged that real per-mouse heterogeneity in
the map over CCF coordinates should be allowed next -- i.e. something beyond a rigid per-
session translation. That extension is not yet implemented; this document describes the
rigid-offset baseline it should be compared against.
