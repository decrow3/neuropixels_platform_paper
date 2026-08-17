# Allen BO 1.1 V1/HVA eye-tracking availability

The cohort is the 31 Brain Observatory 1.1 sessions represented in `artifacts/allen_trusted_interior_rf_distributions/trusted_interior_units.csv`.

Allen's public warehouse `EcephysSession.fail_eye_tracking` field was queried on 2026-08-15. AllenSDK implements `get_sessions(has_eye_tracking=True)` as `fail_eye_tracking == false` and implements `has_eye_tracking=False` as `fail_eye_tracking == true`.

- 25/31 sessions (80.6%) have valid eye tracking.
- 6/31 sessions (19.4%) are marked as failed eye tracking.
- Failed session IDs: 715093703, 719161530, 721123822, 732592105, 737581020, 739448407.

Local NWB cross-checks agree with the warehouse metadata: session 746083955 contains raw and filtered gaze mapping, while session 737581020 contains rig geometry but no raw gaze mapping.

Warehouse query endpoint: `http://api.brain-map.org/api/v2/data/query.json`, model `EcephysSession`, filtered by the 31 session IDs and `fail_eye_tracking`.
