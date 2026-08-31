# Robust within-V1 versus post-V1 spread comparison

_Generated 2026-08-24._

## Primary estimand

The primary question is whether stable HVA identity explains more session-level
variance than stable probe identity within V1. Omega-squared is computed from
session-by-group means. Whole session vectors are resampled so simultaneously
recorded groups remain together in every bootstrap draw.

| Metric | V1 probe ω² | 95% CI | HVA area ω² | 95% CI | Δω² HVA−V1 | 95% CI | P(Δ≤0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFS (ms) | 0.1605 | [+0.0822, +0.4787] | 0.2369 | [+0.1672, +0.3418] | +0.0765 | [-0.2421, +0.1918] | 0.474 |
| log10 F1/F0 | -0.0578 | [-0.0937, +0.2420] | 0.0051 | [-0.0089, +0.0600] | +0.0628 | [-0.2271, +0.1248] | 0.333 |
| Response timescale (ms) | 0.0015 | [-0.0838, +0.3553] | 0.1803 | [+0.1229, +0.2728] | +0.1789 | [-0.1632, +0.3027] | 0.165 |

All three direct identity-comparison intervals cross zero. The current data
therefore do not establish that HVA identity explains more variation than position
within V1.

## Secondary heterogeneity diagnostic

For each session, the diagnostic computes the sample SD across available group means.
MouseV2 sessions require all four probes; Allen sessions require at least
3 of the five cortical post-V1 areas. This measures within-session separation,
not stable named-group identity.

| Metric | V1 sessions | HVA sessions | Mean V1 SD | Mean HVA SD | Δ HVA−V1 | 95% CI | P(Δ≤0) | LOO Δ range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFS (ms) | 8 | 58 | 2.416 | 3.943 | +1.527 | [+0.7639, +2.228] | 0.000 | [+1.386, +1.746] |
| log10 F1/F0 | 8 | 58 | 0.02754 | 0.04899 | +0.02145 | [+0.01106, +0.03123] | 0.000 | [+0.01957, +0.02426] |
| Response timescale (ms) | 8 | 55 | 5.886 | 6.858 | +0.9718 | [-1.013, +2.871] | 0.164 | [+0.4379, +1.725] |

## Interpretation limits

- Probe letters are categorical within-V1 locations, not numerical hierarchy scores.
- Shared y-limits make raw panels visually comparable, but absolute MouseV2–Allen
  centers remain affected by protocol and population differences.
- Session-centering removes global session offsets; it does not match RF location,
  layer, depth, firing rate, stimulus support, or cell type.
- Allen sessions contain different HVA subsets. The primary minimum-area rule is
  explicit, and `n_groups` plus `groups_present` are preserved in the spread CSV.
- Unit-level KDEs are deliberately omitted: they weight sessions by retained unit count
  and are not the inferential estimand.

## Audit artifacts

- `Figure3_robust_session_group_means.csv`: every retained session × group mean, unit count, and centered value.
- `Figure3_robust_session_spreads.csv`: every retained session spread and observed group composition.
- `Figure3_robust_spread_comparison.png` and `Figure3_robust_spread_comparison.pdf`: rendered figure.
