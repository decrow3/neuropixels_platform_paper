# Iteration 4A — harmonized common-QC populations

## Outcome

All Figure 3 entry points and the session-aggregated statistical comparison now
apply one named `common_qc` population profile to both datasets before
metric-specific validity filters. The profile is:

```text
amplitude_cutoff < 0.1
presence_ratio > 0.8
ISI violations ratio < 0.5
```

For MouseV2 this selects 11,242/20,374 units and is verified unit-for-unit to
equal the existing NWB `default_qc` flag. For Allen it selects 43,496/99,180
units across the released table and 21,673/43,861 units in the eight Figure 3
regions. MouseV2 retains 1,176–1,711 units per session, with every session ×
probe group represented.

This fixes the earlier inconsistency in which only the split/statistics path
applied MouseV2 QC. The full hierarchy overlay, probe zoom, split comparison,
measured-RF diagnostic, and statistical companion now declare and use the same
profile.

## Population sensitivity

The table compares the Iteration 3 checkpoints with common QC. Estimates are
session-aggregated omega squared; the final columns are from Iteration 4A.

| Metric | Iteration 3 ω² probes | Iteration 3 ω² areas | Common-QC ω² probes | Common-QC ω² areas | Δω² areas−probes | 95% CI for Δω² | P(Δ≤0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFS | 0.160 | 0.237 | 0.160 | 0.125 | −0.036 | [−0.356, +0.112] | 0.811 |
| log10 modulation index | 0.171 | 0.127 | 0.171 | 0.151 | −0.019 | [−0.334, +0.147] | 0.715 |
| log10 full-condition F1/F0 | −0.080 | 0.005 | −0.080 | 0.006 | +0.086 | [−0.223, +0.135] | 0.407 |
| Response timescale | 0.001 | 0.180 | 0.001 | 0.172 | +0.171 | [−0.222, +0.307] | 0.202 |

The within-V1 estimates are unchanged because Iteration 3 already used the
same MouseV2 default-QC population in the split/statistics path. Allen common
QC has its largest impact on TTFS: the post-V1 point estimate falls from 0.237
to 0.125, reversing the sign of the areas-minus-probes contrast. Modulation
index changes modestly, while F1/F0 and timescale are essentially stable. Every
difference interval crosses zero.

The population sensitivity therefore does not support a claim that post-V1
area labels explain more variation than V1 probe labels. It remains compatible
with the narrower interim statement that within-V1 variation can be comparable
to between-area variation. This is not a final claim update: RF filtering, a
validated hierarchy representation for the known within-V1 locations,
matched-session inference, LP handling, and metric-accommodation work remain
outstanding.

## RF-filter audit and blocked checkpoints

Allen's released table can compute the original `published_like` population:
24,134 units in the target Figure 3 regions, or 13,079 after intersecting with
common QC. MouseV2 cannot yet compute the same mask because no validated
all-session `p_value_rf` or `area_rf` export exists.

The grating-rate criterion is available: `firing_rate_dg` is the preferred
full-condition mean spike count divided by the validated 1.0-second analysis
window. The provisional Pilot RF QC population (4,807 units) is a useful
diagnostic but is not equivalent to the published filter. Consequently:

- `04b_published_like` is blocked rather than run with weakened criteria.
- `04c_rf_area` is blocked until RF area is defined and benchmarked.
- The exact cross-repository work and acceptance condition are frozen in
  [`RF_FILTER_BLOCKER.md`](../../data/imports/RF_FILTER_BLOCKER.md).

## Validation

- Named masks are centralized and fail loudly when required columns are absent.
- MouseV2 common QC is tested for exact equality with `default_qc`.
- Allen common-QC selection is fixed at 43,496 units in a regression test.
- `firing_rate_dg` is tested against preferred-condition mean spikes and
  duration; frozen Iteration 3 imports are read compatibly.
- All 15 focused schema, metric, RF-import, and population tests pass.
- A no-profile baseline rerun remains scientifically equivalent to Iteration 0
  for all three figures and the statistical report.
- Both common-QC split figures and the population-flow figure were visually
  reviewed.

## Review artifacts

- Original released modulation-index metric:
  [`04a_common_qc`](04a_common_qc/Figure3_stats.md)
- Full-condition F1/F0 sensitivity:
  [`04a_common_qc_f1f0_sensitivity`](04a_common_qc_f1f0_sensitivity/Figure3_stats.md)
- Population counts, per-group balance, flow figure, reports, logs, code/input
  hashes, and manifests are stored inside both checkpoints.

## Next executable step

Iteration 5 can proceed in this repository while the RF export is improved in
PilotAnalysis. It should first audit the flash onset provenance and regenerate
a raw TTFS/flash-polarity checkpoint; absolute TTFS must not be calibrated by
forcing MouseV2 to match Allen V1.
