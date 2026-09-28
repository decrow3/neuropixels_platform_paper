# Single-cell C/D/E with older Allen V1 added as Central

Central is included in C's five-group V1 effect and E's HVA-minus-V1 difference. Original figures are preserved.

## Central source and meaning

Central uses the original legacy Allen V1 cells from `data/unit_table.csv`, with native metrics and the original metric-specific validity filters and minimum of five cells per session. All original reference sessions are included, rather than restricting Central to E's matched control sessions. Per-session means and counts reconstruct `Figure3_Allen_V1_session_means.csv` exactly. TTFS is converted to ms, F1/F0 is log10-transformed, and timescale remains in ms. No additional common-QC or harmonization filters are applied to Central.

Central is the requested display label, not a coordinate-verified anatomical location. Its dataset and processing pipeline are confounded with its group label. The five-group estimate is consequently a sensitivity analysis of combined V1 populations; differences cannot be attributed specifically to spatial position. This version should not be presented as five matched recording locations.

| Metric | Central cells | Central sessions |
|---|---:|---:|
| Response timescale (ms) | 1,612 | 53 |
| TTFS (ms) | 3,874 | 55 |
| log10 F1/F0 | 8,546 | 56 |

## Estimation and resampling

Every eligible neuron receives equal weight. The original four-location and HVA cells, repeated-fit handling, and E control inputs are retained. Cell-level one-way omega-squared is recomputed after adding Central to V1. It is bias-corrected and can be negative; it is not a multilevel variance-component estimate.

5,000 bootstrap replicates (seed 20260917) resample whole sessions separately within MouseV2 and Allen. Within each metric, the union of eligible Allen sessions is resampled, with identical multiplicities applied to Central, HVAs, and the control. Missing session/population combinations remain missing. This preserves dependence between V1 and HVA effect estimates from shared Allen sessions. E's difference is computed within each joint replicate. The two datasets are not resampled as one exchangeable cohort.

The V1 shuffle holds Central fixed, since its sessions never contain the other four location labels. Its stars/n.s. carry ‡ and test only the assignment of the four MouseV2 labels conditional on the fixed Central distribution; they are not an omnibus test of all five groups or of the Central-versus-peripheral distinction. Gray V1 null estimates may therefore be far from zero. HVA/control shuffles retain their prior interpretation. All shuffle p-values are one-sided and uncorrected. Difference labels † use the 95% bootstrap interval, not a shuffle p-value; intervals including zero do not establish equivalence.

E's V1–pooled-HVA control retains the prior HVA-matched processing and populations. Central's legacy cells are not substituted into that separate control. HVA and control point estimates are unchanged; intervals and shuffle p-values are refreshed under the joint bootstrap and new random stream.

## Effects

| Metric | V1 including Central (%) | Across HVAs (%) | HVA − V1 (pp), 95% interval |
|---|---:|---:|---:|
| TTFS (ms) | 0.02 | 0.83 | +0.81 [-0.25, +2.95] |
| log10 F1/F0 | 2.59 | 0.21 | -2.39 [-4.24, -0.73] |
| Response timescale (ms) | 0.87 | 2.29 | +1.42 [-0.03, +3.07] |

Reproduce with `MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_central_v1_candidate.py`. Inputs, reconstruction audit, effects, resamples, method JSON, and PDF/PNG/SVG exports are saved with the same filename prefix.
