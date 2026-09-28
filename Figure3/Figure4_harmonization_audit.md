# Figure 4 harmonization audit

## Finding

The newer figures do not uniformly preserve the established cross-dataset harmonization. The four MouseV2 locations and HVA response inputs retain the accepted aligned metrics. The single-cell E control also uses aligned Allen V1 inputs. Adding Central reintroduced native Allen metrics and broader QC, and the RF-adjusted version imposed a legacy Pilot-QC fit-coverage restriction on MouseV2. Timescale trial-count matching remains, but the single-cell aggregation and uncertainty differ from the earlier validated draw-aware analysis.

No figures, source data, filters, or scientific conclusions were changed by this audit. Exact input-level checks and QC counts are saved alongside it.

## What was preserved

All current non-Central response values match their accepted source tables to numerical precision: response-selected preferred-polarity TTFS; harmonized conventional F1/F0; valid Allen timescales; and means of valid MouseV2 matched-150-flash timescale fits. All non-Central plotted populations pass common waveform QC. The RF-qualified subset preserves those incoming raw values exactly before regression.

The F1/F0 contract specifies a common 1-s window, 15 trials per orientation×TF, orientations 0/45/90/135°, TFs 1/2/4/8/15 Hz, SF=.04 cycles/degree, contrast=.8, and the 28 complete Allen Brain Observatory sessions. The current HVA and non-Central control values still use that table. MouseV2 timescale input fits still use 75 bright + 75 dark flashes, matching Allen's trial count.

## Regression 1: Central uses native, not harmonized, Allen data

The Central builder reads `data/unit_table.csv` without common QC, transforms native F1/F0 and released TTFS, and applies the old metric-validity/minimum-cell rules. Its values are confirmed to match those native columns. This is not the harmonized Allen V1 reference already available for comparison with MouseV2. The original session-figure control also read the legacy `Figure3_Allen_V1_session_means.csv`; the later non-Central single-cell control corrected that source, but the Central branch reintroduced it.

| Central metric | Cells | Common QC pass | Common QC fail | Sessions |
|---|---:|---:|---:|---:|
| Response timescale (ms) | 1612 | 880 | 732 | 53 |
| TTFS (ms) | 3874 | 2099 | 1775 | 55 |
| log10 F1/F0 | 8546 | 4111 | 4435 | 56 |

For 1,948 Central F1/F0 neurons shared with the accepted harmonized Allen V1 table, the native mean is -0.200413 log10 versus -0.022007 harmonized, a paired mean change of +0.178406. These are the same cells, so that change is directly attributable to metric processing. This comparison is cell-weighted and is not the historical equal-session bridge mean. Central includes 56 sessions, while the harmonized F1/F0 reference is restricted to 28 complete Brain Observatory sessions. Its apparent difference from MouseV2 must not be read as within-V1 spatial variation.

## Regression 2: RF coverage/selection is not aligned

The MouseV2 RF builder fits only the earlier Pilot-QC subset and then demands supported Gaussian fits (including reliability, fit quality, on-screen centers, and interior widths). Allen uses released RF centers with published-like RF significance/area/SNR/rate gates. These are different estimator/selection contracts, not a validated matched RF population. The separate `Figure4_RF_TTFS_probe_audit.md` establishes that all 1,090 original MouseV2 TTFS cells have raw RF records but only 515 have entries in the parametric-fit table, exactly matching legacy Pilot-QC eligibility. 189 pass support and 158 survive the five-cell floor. Missing parametric fits must not be labeled unreliable RFs.

The new adjustment also differs from the earlier validated Allen-only RF workflow: it uses absolute display coordinates (+50/+10 for MouseV2), a common quadratic surface per comparison, no common-support restriction, and in-sample partial omega-squared. The earlier Allen workflow uses session-V1-relative RF positions, strict overlap support, session-centered outcomes, and held-out-session prediction. The new approach is explicitly exploratory; its stars cannot be treated as reproducing the earlier analysis. The coordinate conversion is not gaze correction.

## Regression 3: timescale draw handling and weighting changed

Current single-cell C/D/E uses one mean of available valid fits per MouseV2 neuron, with equal neuron weights and session-only resampling. Of 773 current MouseV2 timescale neurons, 422 have all ten valid fits, 351 have fewer, and 82 have only one. Allen contributes one estimate per neuron. The trial-matched response sources are preserved, but equal weighting of one-valid-draw and ten-valid-draw neurons and omission of trial-draw uncertainty are changes to the measurement/uncertainty comparison.

The earlier full-cell extension fit each matched draw separately and sampled one draw per resampled MouseV2 session. Its sensitivity work explicitly found that complete-draw inclusion and fitted-error selection changed the timescale contrast. Those checks were not carried forward into the simplified new C/D/E analysis. Anatomical V1 filtering also changes the population relative to that older checkpoint, so its numerical results should not be copied directly.

## Pre-existing limits, not new regressions

Matching a preferred-response TTFS estimator does not establish an independent physical light-onset calibration between datasets. The prior roadmap explicitly excludes interpreting uncalibrated absolute cross-dataset latency shifts as biology. Adding Allen V1 as a fifth MouseV2 location reintroduces sensitivity to dataset-wide offsets. Anatomical V1 restriction, common waveform QC, and metric harmonization also do not match cell type, layer, behavior, or sampling density.

The earlier timescale interim conclusion was that a systematic HVA hierarchy was not robust across fit-population sensitivities. New categorical-effect stars estimate a different quantity and do not overturn that result. The V1 shuffle in Central-inclusive plots holds Central fixed; it is not an omnibus cross-dataset group test.

## Concrete repair sequence

1. Use the harmonized Allen V1 unit sources for Central, with the same waveform QC and metric processing as the HVA/control branches. Use every eligible session for that reference, not only the subset retained for E's paired control; retain metric-specific session eligibility.
2. Fit and validate RFs for the current MouseV2 analysis population missing from the legacy fit table. Review comparable RF estimator/support criteria and coordinate alignment before interpreting RF-adjusted cross-dataset effects; do not substitute raw peak centers or relax thresholds silently.
3. Restore trial-draw-aware timescale estimation/uncertainty and its complete-draw/fit-error sensitivity checks on the current anatomical population. Keep any mean-per-neuron display explicitly separate from inferential weighting.
4. Regenerate unadjusted and RF-qualified/adjusted figures from a shared provenance-checked loader, keeping same-cell controls and the original exploratory versions.

## Inspected evidence

- `artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0/FIGURE3_HARMONIZED_F1_F0_CHECKPOINT.md`
- `data/imports/mousev2_timescale_trial_bridge_v1/README.md`
- `artifacts/figure3/07_big_picture_concrete_first/timescale_population_sensitivity/TIMESCALE_INTERIM_CONCLUSION.md`
- `artifacts/figure3/07_big_picture_concrete_first/retinotopy_adjusted_hva_models/VALIDATION_REPORT.md`
- `scripts/figure3_full_cell_multilevel_extension.py` and current Figure 4 builders
- `ANALYSIS_ROADMAP.md` and the source tables recorded in `_source_checks.csv`

Reproduce with `MPLCONFIGDIR=/tmp/mpl-figure4-cells python scripts/audit_figure4_harmonization.py`.
