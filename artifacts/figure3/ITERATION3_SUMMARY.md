# Iteration 3 — full-condition drifting-grating metrics

## Outcome

All eight MouseV2 NWBs were reprocessed successfully: 20,374 units before
filtering and 11,242 units after `default_qc`. Preferred grating conditions now
include orientation × temporal frequency × spatial frequency (100 conditions,
15 repeats each). Three quantities remain available side by side:

- `f1_f0_dg`: Allen cycle-fold F1/F0 at the preferred full condition.
- `mod_idx_dg`: Allen Welch-spectrum modulation index at the same condition.
- `f1_f0_dg_pooled_sf_legacy`: the previous value, retained rather than overwritten.

MouseV2's logged presentations are approximately 1.00084 s; the analysis uses
the protocol's nominal 1.0-s duration and exactly 1,000 one-millisecond bins.
Unlike the first implementation, bins contain spike counts rather than binary
spike-presence flags.

## Validation

- Eight raw DANDI:001568 asset paths, sizes, and SHA-256 hashes are recorded.
- Every session has the expected unit-ID range and unit count.
- Every preferred condition contains 15 trials and uses a 1.0-s window.
- Both numerical formulas match the installed AllenSDK 2.16.2 source in direct tests.
- Independent PilotAnalysis preference triplets agree for 95.9–98.2% of shared
  units per session (97.0% mean across sessions).
- 2,002/20,374 units (9.8%) have an exact preferred-condition spike-count tie;
  this falls to 785/11,242 (7.0%) after default QC.
- Excluding tied units leaves the session × probe summaries highly stable:
  Pearson r = 0.996 for F1/F0 and 0.984 for modulation index. Within-V1 ω²
  changes from −0.080 to −0.088 for F1/F0 and from 0.171 to 0.148 for
  modulation index.

The full-condition and pooled-SF F1/F0 values remain strongly related (median
site-level Spearman rho = 0.899), but the correction is material. Among
default-QC units, the pooled-SF median is 1.013 and the corrected median is
0.792. The modulation-index median is 0.775.

## Scientific comparison

The table reports session-aggregated omega-squared estimates from the two
metric-specific statistical companions.

| Grating metric | Within-V1 probes ω² | Post-V1 areas ω² | Δω² areas−probes | 95% CI for Δω² | P(Δ≤0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| log10 full-condition F1/F0 | −0.080 | 0.005 | +0.085 | [−0.241, +0.130] | 0.401 |
| log10 modulation index | 0.171 | 0.127 | −0.044 | [−0.352, +0.117] | 0.790 |

Corrected F1/F0 shows essentially no categorical structure in either dataset.
The published modulation index shows structure in both, but the within-V1 point
estimate is not smaller than the post-V1 estimate and the difference interval
crosses zero. Thus Iteration 3 does not support a claim that post-V1 area labels
explain more variation than V1 probe labels for the grating metric. It is
consistent with the broader working claim that within-V1 variation can be
comparable to between-area variation, but that claim remains provisional until
population matching, RF filtering, matched-session inference, anatomy, and the
other metric accommodations are completed.

No primary grating metric is selected yet. Both views should remain visible:
F1/F0 provides a conventional simple/complex-cell measure, while `mod_idx_dg`
faithfully reproduces the metric used by the released Figure 3 script.

## Review artifacts

- Raw metric import and diagnostics:
  [`mousev2_grating_metrics_v1`](../../data/imports/mousev2_grating_metrics_v1/README.md)
- Corrected F1/F0 checkpoint:
  [`03a_f1f0_full_condition`](03a_f1f0_full_condition/delta_from_previous.md)
- Published modulation-index checkpoint:
  [`03b_original_modulation_index`](03b_original_modulation_index/delta_from_previous.md)

Each checkpoint contains three main figures, the statistical companion, the
measured-retinotopy figure, command logs, code/input provenance, and a run
manifest. The first `03b` rendering with overly dense y-axis ticks was moved to
`03b_original_modulation_index_superseded_tick_bug`; the canonical checkpoint
contains the corrected figure.

## Still outstanding

- A raw original Allen NWB was not present locally, so the planned whole-session
  low-level-versus-published-table gold-standard test has not run. Formula-level
  equivalence with the installed AllenSDK source is complete.
- RF significance/area and common population masks remain Iteration 4 work.
- The current bootstrap still does not preserve the four matched probes within
  each MouseV2 session.
- Display-latency calibration, flash-polarity sensitivity, LP handling, anatomy,
  and the full CCG analysis remain later iterations.
