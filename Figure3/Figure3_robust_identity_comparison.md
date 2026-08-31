# Robust within-V1 versus post-V1 spread comparison

_Generated 2026-08-26._

## Primary estimand

The cortical HVA set is LM, RL, AL, PM, and AM. The thalamic lateral
posterior nucleus (LP) is excluded from the cortical-HVA estimand.

The primary question is whether stable HVA identity explains more session-level
variance than stable probe identity within V1. Omega-squared is computed from
session-by-group means. Whole session vectors are resampled so simultaneously
recorded groups remain together in every bootstrap draw.

| Metric | V1 probe ω² | 95% CI | HVA area ω² | 95% CI | Δω² HVA−V1 | 95% CI | P(Δ≤0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFS (ms) | 0.0439 | [-0.0734, +0.3422] | 0.0759 | [+0.0019, +0.2669] | +0.0320 | [-0.2553, +0.2532] | 0.417 |
| log10 F1/F0 | -0.0688 | [-0.0950, +0.2043] | -0.0065 | [-0.0276, +0.1428] | +0.0622 | [-0.1818, +0.1861] | 0.285 |
| Response timescale (ms) | -0.0560 | [-0.0983, +0.4753] | 0.1339 | [+0.0759, +0.2421] | +0.1899 | [-0.3217, +0.2877] | 0.219 |

All three direct identity-comparison intervals cross zero. The current data
therefore do not establish that HVA identity explains more variation than position
within V1.

For response timescale specifically, the later full-cell sensitivity analysis
strengthens this claim gate: the primary HVA/V1 structured-variance ratio is
2.99x, but it falls to 2.47x for V1 neurons valid in all ten matched draws and
1.43x under a matched fitted-error <10 ms gate; both sensitivity difference
intervals include zero. See the
[interim conclusion](../artifacts/figure3/07_big_picture_concrete_first/timescale_population_sensitivity/TIMESCALE_INTERIM_CONCLUSION.md).

## Secondary heterogeneity diagnostic

For each session, the diagnostic computes the sample SD across available group means.
MouseV2 sessions require all four probes; Allen sessions require at least
3 of the five cortical post-V1 areas. This measures within-session separation,
not stable named-group identity.

| Metric | V1 sessions | HVA sessions | Mean V1 SD | Mean HVA SD | Δ HVA−V1 | 95% CI | P(Δ≤0) | LOO Δ range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFS (ms) | 8 | 12 | 2.741 | 5.05 | +2.309 | [+0.6061, +3.818] | 0.004 | [+2.012, +2.838] |
| log10 F1/F0 | 8 | 27 | 0.03155 | 0.04683 | +0.01528 | [+0.003591, +0.02701] | 0.004 | [+0.01235, +0.01747] |
| Response timescale (ms) | 6 | 48 | 5.099 | 7.642 | +2.543 | [+1.069, +4.018] | 0.000 | [+2.226, +3.055] |

## Hierarchy-score fits

Slopes use area-level centers. Confidence intervals resample session means within
each area. The post-V1-only fit is reported separately so a VISp-to-HVA step is not
mistaken for a graded trend among higher areas.

| Metric | Scope | Areas | Slope | 95% CI | Area-mean r |
|---|---|---:|---:|---:|---:|
| TTFS (ms) | VISp_plus_post_V1 | 6 | +7.321 | [+3.491, +11.27] | +0.902 |
| TTFS (ms) | post_V1_only | 5 | +7.869 | [+2.542, +13.38] | +0.850 |
| log10 F1/F0 | VISp_plus_post_V1 | 6 | -0.1075 | [-0.1378, -0.07749] | -0.819 |
| log10 F1/F0 | post_V1_only | 5 | -0.03635 | [-0.09002, +0.02046] | -0.729 |
| Response timescale (ms) | VISp_plus_post_V1 | 6 | +11.3 | [+7.866, +14.81] | +0.819 |
| Response timescale (ms) | post_V1_only | 5 | +12.91 | [+7.577, +18.17] | +0.763 |

The positive timescale slope is descriptive and is not the claim gate. It is a
fit through only five HVA area centers and its session bootstrap does not
propagate the later matched-draw-completeness and fit-eligibility sensitivities.
Those sensitivities reduce the full-cell point ratio from 2.99x to 2.47x and
1.43x, with both alternative HVA-minus-V1 difference intervals crossing zero.
Accordingly, the slope must not be cited as establishing a temporal hierarchy.

## Interpretation limits

- Probe letters are categorical within-V1 locations, not numerical hierarchy scores.
- F1/F0 uses common QC and matched 1-s, 15-trial, SF 0.04, contrast 0.8 support.
  Allen F1/F0 is restricted to the 28 complete Brain Observatory sessions; TTFS
  and timescale retain their existing source-session support.
- Session-centering removes global session offsets; it does not match RF location,
  layer, depth, firing rate, stimulus support, or cell type.
- Allen sessions contain different HVA subsets. The primary minimum-area rule is
  explicit, and `n_groups` plus `groups_present` are preserved in the spread CSV.
- Unit-level KDEs are deliberately omitted: they weight sessions by retained unit count
  and are not the inferential estimand.

## Audit artifacts

- `Figure3_robust_session_group_means.csv`: every retained session × group mean, unit count, and centered value.
- `Figure3_robust_session_spreads.csv`: every retained session spread and observed group composition.
- `Figure3_hierarchy_fit_stats.csv`: all-area and post-V1-only hierarchy slopes with session-bootstrap intervals.
- `Figure3_timescale_population_flow.csv`: gate-by-gate MouseV2 timescale-population selection counts.
- `Figure3_timescale_population_units.csv`: retained MouseV2 timescale units and RF-proxy fields.
- `Figure3_timescale_matched_draw_coverage.csv`: session × probe coverage across the ten matched-flash draws.
- `Figure3_robust_identity_comparison.png` and `Figure3_robust_identity_comparison.pdf`: rendered figure.
- `Figure3_robust_identity_panel_A.png` and `Figure3_robust_identity_panel_A.pdf`: standalone identity-effect panel.
- `Figure3_single_panel_identity_variation.png` and `Figure3_single_panel_identity_variation.pdf`: single-panel visual summary without printed statistics.
- `Figure3_session_level_variation_comparison.png` and `Figure3_session_level_variation_comparison.pdf`: matched session-level probe/area comparison in the earlier split-panel style.
- `Figure3_session_level_variation_with_Visual_Coding_VISp.png` and `Figure3_session_level_variation_with_Visual_Coding_VISp.pdf`: the same comparison with Visual Coding VISp added as a V1 reference.
- `Figure3_hierarchy_fits_with_session_histograms.png` and `Figure3_hierarchy_fits_with_session_histograms.pdf`: hierarchy-score fits with session-level marginal histograms.
