# Iteration 6C — RF-adjusted Allen response analysis

## Status: first response-adjustment checkpoint implemented

Area contrasts are identified within Allen sessions using session fixed effects
and session-clustered uncertainty. The primary RF-adjusted view is restricted to
the conservative session-centered V1 support box and uses flexible azimuth and
elevation terms. A separate same-session nearest-neighbor sensitivity matches each
HVA unit to a V1 unit in two-dimensional RF-center space with replacement.

## Area coefficients before and after RF adjustment

| Outcome | Area | Unadjusted | RF-adjusted |
| --- | --- | ---: | ---: |
| TTFS (ms) | LM | +3.427 | +1.131 |
| TTFS (ms) | RL | +5.934 | +1.941 |
| TTFS (ms) | AL | +4.063 | +0.433 |
| TTFS (ms) | PM | +10.568 | +7.771 |
| TTFS (ms) | AM | +8.965 | +6.475 |
| log10 modulation index | LM | -0.233 | -0.228 |
| log10 modulation index | RL | -0.257 | -0.296 |
| log10 modulation index | AL | -0.295 | -0.261 |
| log10 modulation index | PM | -0.345 | -0.303 |
| log10 modulation index | AM | -0.361 | -0.352 |
| log10 F1/F0 | LM | -0.026 | -0.001 |
| log10 F1/F0 | RL | -0.074 | -0.034 |
| log10 F1/F0 | AL | -0.058 | -0.034 |
| log10 F1/F0 | PM | -0.045 | -0.020 |
| log10 F1/F0 | AM | -0.074 | -0.018 |
| response timescale (ms) | LM | -0.270 | -3.285 |
| response timescale (ms) | RL | +4.834 | -2.179 |
| response timescale (ms) | AL | +0.334 | -4.843 |
| response timescale (ms) | PM | +7.594 | +3.603 |
| response timescale (ms) | AM | +7.292 | +1.769 |
| RF area (deg²) | LM | +125.423 | +137.825 |
| RF area (deg²) | RL | +101.246 | +182.564 |
| RF area (deg²) | AL | +151.029 | +314.686 |
| RF area (deg²) | PM | +201.633 | +227.210 |
| RF area (deg²) | AM | +256.970 | +328.231 |

The coefficient change combines explicit RF adjustment with restriction to V1
common support. `model_area_effects.csv` also contains an unadjusted
common-support model, which separates those two changes.

## Matched sensitivity

| Outcome | Area | Sessions | Matched HVA−V1 | 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| log10 F1/F0 | AL | 34 | -0.111 | -0.154 to -0.068 |
| log10 F1/F0 | AM | 42 | -0.103 | -0.137 to -0.069 |
| log10 F1/F0 | LM | 28 | -0.046 | -0.101 to +0.006 |
| log10 F1/F0 | PM | 24 | -0.032 | -0.090 to +0.027 |
| log10 F1/F0 | RL | 26 | +0.009 | -0.058 to +0.073 |
| log10 modulation index | AL | 34 | -0.185 | -0.286 to -0.081 |
| log10 modulation index | AM | 42 | -0.387 | -0.448 to -0.328 |
| log10 modulation index | LM | 28 | -0.268 | -0.370 to -0.168 |
| log10 modulation index | PM | 24 | -0.261 | -0.391 to -0.135 |
| log10 modulation index | RL | 26 | -0.215 | -0.321 to -0.107 |
| RF area (deg²) | AL | 34 | +308.476 | +223.612 to +396.742 |
| RF area (deg²) | AM | 42 | +355.650 | +310.933 to +404.834 |
| RF area (deg²) | LM | 28 | +87.372 | +12.528 to +158.258 |
| RF area (deg²) | PM | 24 | +181.519 | +76.923 to +285.912 |
| RF area (deg²) | RL | 26 | +230.646 | +129.875 to +324.635 |
| response timescale (ms) | AL | 14 | -0.373 | -10.027 to +8.126 |
| response timescale (ms) | AM | 25 | +7.337 | +2.594 to +12.526 |
| response timescale (ms) | LM | 15 | -1.633 | -8.086 to +4.475 |
| response timescale (ms) | PM | 10 | +8.596 | +0.853 to +16.628 |
| response timescale (ms) | RL | 7 | +5.146 | -4.628 to +17.262 |
| TTFS (ms) | AL | 25 | +1.666 | -2.018 to +5.360 |
| TTFS (ms) | AM | 39 | +4.468 | +1.931 to +6.917 |
| TTFS (ms) | LM | 24 | +2.142 | -2.236 to +6.706 |
| TTFS (ms) | PM | 17 | +4.021 | +0.481 to +7.217 |
| TTFS (ms) | RL | 20 | +2.374 | -1.431 to +6.096 |

Mean absolute RF-coordinate SMD changes from 0.920 before matching to 0.202 after matching.
The 10° caliper discards a mean 16.4% of otherwise eligible HVA units across session × area × outcome cells.

### Caliper trade-off

| Caliper (deg) | Mean |SMD| after | Mean HVA discarded | Minimum session pairs |
| ---: | ---: | ---: | ---: |
| 5 | 0.101 | 45.0% | 3 |
| 7.5 | 0.158 | 29.3% | 6 |
| 10 | 0.225 | 17.2% | 7 |
| 15 | 0.375 | 7.4% | 13 |
Area × RF-position interactions have joint p < 0.05 for 3/5 outcomes; these are model checks, not a second primary test.

## Interpretation boundary

This checkpoint addresses achieved RF-center sampling within the Allen dataset.
It does not calibrate MouseV2 and Allen response levels, validate MouseV2 RF
area/significance, or make the cross-dataset claim pass. Matching is with
replacement and balance/discarded support must accompany every reported result.

## Outputs

- `model_coefficients.csv`: every fitted coefficient and clustered interval.
- `model_area_effects.csv`: HVA−V1 coefficients for each model.
- `model_fit_summary.csv`: formulas, sample sizes, fit summaries, and interaction tests.
- `matched_session_contrasts.csv`: paired session-level matched contrasts.
- `matched_balance.csv`: RF balance and match-distance diagnostics.
- `matched_area_summary.csv`: equal-session contrasts and bootstrap intervals.
- `matching_caliper_sensitivity.csv`: balance, attrition, and effect sensitivity across calipers.
- `hierarchy_associations.csv`: descriptive hierarchy slopes across five areas.
- `Figure_allen_rf_adjusted_response.png`: primary diagnostic figure.
- `run_manifest.json`: input/code/output checksums and parameters.
