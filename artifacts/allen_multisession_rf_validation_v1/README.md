# Allen multisession RF validation v1

This bundle audits four Allen Brain Observatory 1.1 sessions through the complete RF workflow: native-NWB ingestion, original AllenSDK reproduction, alternative point/aperture geometry, rotated fits, population gaze correction, synthetic eye-trace recovery, and cross-session registration readiness.

## Main outcome

- The original Allen RF code runs successfully on all four native sessions when the sessions are processed sequentially.
- Allen's released center/area metrics are reproduced exactly for roughly 70–80% of QC-selected units, leaving a historical-pipeline mismatch that should remain explicit.
- Analytic circular-aperture fits reduce trusted interior RF areas by about 23–31% in V1 and 12–15% in HVAs relative to point-center fits.
- Rotation gives small held-out predictive gains, mainly in HVAs, without materially changing population RF sizes.
- Calibrated population gaze correction does not consistently sharpen RFs: three sessions select zero gain and the fourth has a very small gain without area reduction.
- Synthetic population decoding partially recovers a shared eye trace only at large neuron counts; it is a secondary research direction, not a default correction.
- RF size surfaces contain potentially useful area-specific registration information, but four-session consistency is not yet adequate for RF size to serve as a sole registration anchor.

## Audit order

1. `00_inventory/` — session availability, storage, and selected cohorts
2. `01_ingestion/` — native NWB extraction and eye-trace coverage
3. `02_allen_baseline/` — original AllenSDK reproduction
4. `03_geometry/` — point versus aperture and axis-aligned versus rotated fits
5. `04_gaze/` — calibrated population gaze validation with held-out neurons and controls
6. `05_synthetic/` — synthetic shared eye-trace recovery
7. `06_cross_session/` — session-level geometry comparisons
8. `07_registration_readiness/` — V1/HVA and area-specific size surfaces
9. `08_validation/` — automated scorecard and validation tables

Open `report.html` for the portable, chart-based audit report. `chart_map.csv` maps each report chart to its query and source tables. `artifact.json` is the report source specification.

## Important caveats

- The released Allen table is not regenerated exactly for 20–30% of selected units.
- Cross-session rotation was evaluated on a fixed held-out subset rather than every unit.
- Registration-readiness conclusions cover four sessions and must be expanded area by area.
- Portable-report QA is structural because Chromium is not installed in this environment; all source PNGs were separately checked for readability.
