# RF-adjusted single-cell C/D/E with Central

This is a separate exploratory candidate. The original and Central-inclusive figures are preserved. Central remains in the middle of C and enters C's V1 effect and E's difference.

## RF selection and coordinates

MouseV2 RFs use supported parametric center estimates (`rf_model_supported`). Allen RFs use the existing published-like RF support table (finite centers, p<.01, RF area<2500 deg², SNR>1, drifting-grating firing rate>0.1 Hz, and an available V1 session anchor). These RF-quality rules differ between sources. RF columns are joined by source, session, and neuron with many-to-one validation. The full eligibility table records each exclusion; the coverage table separates missing RFs from subsequent session/group support exclusions.

The existing fixed display-coordinate conversion maps MouseV2 x/y to Allen azimuth/elevation using +50/+10 degrees. Native Allen coordinates are used, not V1-centered relative coordinates. These are screen coordinates, not verified gaze-corrected retinal positions. No shared-support restriction or RF-location matching is imposed; the model can extrapolate where group coverage differs. The RF coverage figure shows that coverage explicitly.

After RF selection, V1 groups (including Central) require at least 5 cells per session. HVA and control groups require at least 10 TTFS cells or 5 other-metric cells. Control HVA cells must belong to retained individual HVA groups, and only sessions with both control populations remain. The same RF-qualified cells are used in the adjusted and unadjusted companion.

## Adjustment and effect definition

For each metric and comparison separately, fit y = group intercepts + a common quadratic RF surface (azimuth, elevation, azimuth², azimuth×elevation, elevation²). Display axes are centered at (50°,10°) and scaled by 25° for numerical stability only. No RF-by-group interactions are fit. Group terms are included when estimating RF coefficients, so an RF-only fit does not absorb all group-associated variation indiscriminately. This remains a model-dependent descriptive adjustment, not a causal separation of area and retinotopy.

C/D show each neuron's raw outcome minus its fitted RF contribution relative to the pooled RF-feature mean for that comparison. Thus response units and each comparison's grand mean are preserved. Adjusted cell values can fall outside the raw selection range. C and D use independently estimated RF surfaces and reference feature distributions; their absolute adjusted means should not be used to infer a V1–HVA shift. The matched E control models that contrast directly.

Effects are partial omega-squared for adding group labels to the RF-only model: (SSE_RF − SSE_full − q×MSE_full)/(SSE_RF + MSE_full), with q the difference in model ranks and MSE_full using all fitted degrees of freedom. This is the bias-corrected fraction of residual variation after RF adjustment associated with group identity. It is not the original omega-squared denominator and not simply omega-squared evaluated on the adjusted dots. Negative estimates are retained. Unadjusted-subset and adjusted values therefore differ in both conditioning and denominator.

Both models are refitted in each of 5,000 bootstrap replicates (seed 20260918). Whole sessions are resampled within MouseV2 and Allen separately, with the same Allen multiplicities used for Central, HVAs, and the control. The bootstrap difference is computed within replicate. The procedure propagates RF coefficient uncertainty conditional on measured RF centers; uncertainty in the RF-center fits is not modeled. No out-of-sample predictive claim is made.

Location-block shuffles retain each cell's outcome/RF pair, reassign available location labels within source/session, and refit the full model. Central remains fixed, so V1 labels carry ‡: they test the four MouseV2 labels conditional on Central, not an omnibus five-group or Central-versus-periphery null. Conditional exchangeability of those label blocks remains an assumption; RF adjustment does not guarantee it. p-values are one-sided, uncorrected. Difference labels † use whether the 95% session-bootstrap interval includes zero, not a shuffle p-value or equivalence test.

Central still uses legacy response metrics and comes from a different cohort. RF regression does not remove that dataset/processing confounding. The control retains HVA-matched response processing and is restricted to RF-qualified matched sessions.

## Direct comparison

| Metric | Version | V1 effect (%) | HVA effect (%) | HVA − V1 (pp), 95% interval |
|---|---|---:|---:|---:|
| TTFS (ms) | All original cells | 0.02 | 0.83 | +0.81 [-0.25, +2.95] |
| TTFS (ms) | RF subset, unadjusted | 0.16 | 1.61 | +1.45 [-0.20, +5.44] |
| TTFS (ms) | RF adjusted | 0.29 | 3.12 | +2.82 [+0.32, +8.08] |
| log10 F1/F0 | All original cells | 2.59 | 0.21 | -2.39 [-4.24, -0.73] |
| log10 F1/F0 | RF subset, unadjusted | 5.63 | 0.31 | -5.32 [-8.24, -2.36] |
| log10 F1/F0 | RF adjusted | 6.51 | 0.23 | -6.27 [-9.13, -3.14] |
| Response timescale (ms) | All original cells | 0.87 | 2.29 | +1.42 [-0.03, +3.07] |
| Response timescale (ms) | RF subset, unadjusted | 0.47 | 2.17 | +1.70 [+0.40, +3.41] |
| Response timescale (ms) | RF adjusted | 0.25 | 2.03 | +1.78 [+0.60, +3.18] |

Reproduce with `OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_rf_adjusted_candidate.py`. Outputs include adjusted and same-cell unadjusted PDFs, PNGs and SVGs, RF-coverage PDF/PNG, inputs, eligibility/coverage audits, model coefficients/diagnostics, effects, and all resamples.
