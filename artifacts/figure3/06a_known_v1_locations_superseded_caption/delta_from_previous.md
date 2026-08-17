# 06a_known_v1_locations delta from baseline

**Baseline comparison: EXPECTED CHANGE.**

Scientific change: MouseV2 and Allen units are compared using Allen Welch-spectrum modulation index.
MouseV2 preferred conditions include orientation, temporal frequency, and
spatial frequency. The existing pooled-SF values remain available as
`f1_f0_dg_pooled_sf_legacy` and are not overwritten.

- `Figure3_with_V1sites.png`: CHANGED
- `Figure3_probe_zoom.png`: CHANGED
- `Figure3_split_comparison.png`: CHANGED
- `Figure3_stats.md`: CHANGED

The measured-retinotopy diagnostic was also regenerated with the selected metric.

- `Figure3_rf_position.png`
- `rf_metric_session_probe.csv`
- `rf_metric_correlations.csv`
- `rf_position_report.md`

Population profile: `common_qc` was applied to both datasets
before the metric-specific validity filters.

MouseV2 flash variant: `pooled`; TTFS display: `raw_nwb`.
TTFS is aligned to NWB interval start_time without cross-dataset mean matching.
Response timescale uses AllenSDK bin-center selection (45–285 ms centers).

Within-V1 x positions are categorical display offsets centered on VISp,
not anatomical hierarchy scores; the probe-mean hierarchy fit is removed.
Measured RF azimuth/elevation remains a separate companion view.
