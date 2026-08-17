# Iteration 6C — Allen achieved RF matching audit

## Status: targeting audit implemented; response adjustment pending

Allen used ISI-derived retinotopic maps to target a common V1-aligned region
in V1, LM, AL, AM, and PM. RL received a documented geometric-center
accommodation because its retinotopic center often lies near the RL–S1 boundary.
This audit tests achieved unit RF centers; it does not reinterpret intended
target coordinates as neural measurements.

Primary audit population: `published_like` (20,829 units with finite RF centers).
Common-support subset: 19,603 units from sessions containing a valid V1 center.
RF-center dispersion and individual RF area are reported separately.

## Paired HVA–V1 session-center distances

| Area | Session pairs | Median (deg) | IQR (deg) |
| --- | ---: | ---: | ---: |
| LM | 38 | 25.9 | 13.0–35.2 |
| RL | 46 | 31.7 | 24.0–39.3 |
| AL | 43 | 33.3 | 24.5–40.5 |
| PM | 33 | 33.7 | 22.2–43.2 |
| AM | 46 | 19.8 | 10.1–25.9 |

These distances summarize robust session × area centers in screen coordinates
relative to simultaneously recorded V1. They are achieved sampling offsets, not
errors in the ISI map and not estimates of individual RF size.

## Common-support interpretation

`rf_common_support_summary.csv` reports two diagnostics after centering every
session on its V1 median: inclusion in the full V1 convex hull and in a robust
axis-aligned V1 box. The robust-box result is deliberately conservative and is
the figure's displayed diagnostic; neither rule is yet a matching estimator.

## Claim gate

The audit outputs are sufficient to specify the RF-adjusted response model and
matching/weighting strategy. They do not yet establish that the hierarchy metrics
survive RF adjustment. The next implementation must fit the predeclared
session-aware models and report balance and discarded support.

## Outputs

- `rf_population_flow.csv`: nested population counts by cohort and area.
- `rf_probe_summary.csv`: achieved centers and dispersion per probe.
- `rf_session_area_summary.csv`: robust combined session × area summaries.
- `rf_paired_hva_v1_offsets.csv`: paired signed offsets and distances.
- `rf_paired_offset_summary.csv`: cohort × area offset summaries.
- `rf_population_sensitivity.csv`: pooled paired offsets across all declared populations.
- `rf_unit_common_support.csv`: session-centered unit coordinates and support flags.
- `rf_common_support_summary.csv`: cohort × area support fractions.
- `Figure_allen_rf_matching.png`: targeting-audit diagnostic.
- `run_manifest.json`: input checksum and audit parameters.
