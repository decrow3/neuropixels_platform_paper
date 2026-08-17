# MouseV2 grating start-phase bridge

The acquisition source advances grating phase as `TF * current_frame / fps`
and does not reset phase at presentation onset. With 60 stimulus frames plus
75 blank frames, the randomized 135-frame sweep stride produces four starting
phases at 1 and 15 Hz, two at 2 Hz, and one at 4 and 8 Hz. The NWB interval
table does not retain phase, so onset-aligned averaging mixes these source-defined
phases unless they are reconstructed from presentation order.

## Primary result

- Equal-session weighted coherence increases from 0.387 to 0.433 after the source-derived phase rotation; the phase-permutation center is 0.346.
- Among the affected 1/2/15-Hz units, coherence increases from 0.373 to 0.438, versus 0.315 under phase permutation.
- All 8/8 sessions increase; the session-level gain ranges from +0.019 to +0.076.
- Coherent F1 increases by +0.044 log10 on average across sessions.
- Representative Allen coherence remains 0.539 in Brain Observatory and 0.516 in Functional Connectivity, leaving residual gaps of -0.106 and -0.083 after source-phase adjustment.

Every MouseV2 session moves in the predicted direction. The adjustment is fixed
by the acquisition code, 60-Hz frame schedule, and chronological presentation id;
it is not estimated from Allen values or optimized against neural responses.

## Temporal-frequency falsification

- 1 Hz: raw 0.402, source-phase adjusted 0.455, permutation 0.311.
- 2 Hz: raw 0.355, source-phase adjusted 0.430, permutation 0.315.
- 4 Hz: raw 0.423, source-phase adjusted 0.423, permutation 0.423.
- 8 Hz: raw 0.428, source-phase adjusted 0.428, permutation 0.428.
- 15 Hz: raw 0.328, source-phase adjusted 0.406, permutation 0.325.

The 4/8-Hz values are unchanged by construction because their 135-frame stride
contains an integer number of cycles. Improvements at the source-predicted varying
phases, especially relative to permuted phase labels, support starting phase as a
real partial cause of trial-average cancellation. At 1 and 15 Hz the source-sign
adjustments are 0.455 and 0.406, whereas the opposite-sign controls are 0.311 and
0.331; at 2 Hz the two signs are mathematically identical for 0/0.5-cycle phases.

## Interpretation boundary

This bridge identifies a material stimulus-definition difference, but it does not
erase the Allen gap and does not authorize a scalar correction to released
`mod_idx_dg`. Source-phase-adjusted carrier coherence is a mechanism diagnostic;
F1/F0 remains the phase-invariant cross-dataset grating measure. The next residual
test should ask whether presentation-level phase shifts are shared across units and
whether they covary with the recorded running and eye-tracking signals.
