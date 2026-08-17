# MouseV2 source-corrected Welch modulation-index bridge

For each trial, only the preferred temporal-frequency DFT component is
removed, rotated by the source-defined starting phase, and added back. All
non-carrier temporal structure—including the onset transient—is unchanged.
The resulting condition-averaged PSTH is evaluated with the unchanged released
Welch modulation-index function. Phase-permuted and opposite-sign rotations are
mechanism controls; 4/8-Hz units are source-predicted negative controls.

## Primary result

- Equal-session log10 modulation index changes from -0.098 to +0.019 after source-phase carrier correction, versus -0.123 for phase permutation and -0.087 for the opposite sign.
- The affected 1/2/15-Hz center changes from -0.115 to +0.048; the phase-stable 4/8-Hz center changes from -0.054 to -0.054.
- All 8/8 sessions increase (exact two-sided sign-test p=0.0078; paired session gains +0.072 to +0.164 log10).
- Among the 8,004 affected 1/2/15-Hz units, 4,739 increase, 3,222 decrease, and 43 are unchanged within 1e-12 log10. Among 3,238 stable-phase 4/8-Hz units, 0 increase, 0 decrease, and 3,238 are unchanged.
- Source correction changes target PSD by +0.136 log10 and the spectrum-wide PSD SD denominator by +0.016 log10 on average.
- Representative single-session common-window Allen centers are +0.088 (BO) and +0.123 (FC), leaving corrected MouseV2 gaps of -0.069 and -0.104 log10.

## Temporal-frequency controls

- 1 Hz: raw -0.041, source corrected +0.105, permutation -0.143, opposite sign -0.146.
- 2 Hz: raw -0.181, source corrected +0.033, permutation -0.130, opposite sign +0.033.
- 4 Hz: raw -0.068, source corrected -0.068, permutation -0.068, opposite sign -0.068.
- 8 Hz: raw -0.035, source corrected -0.035, permutation -0.035, opposite sign -0.035.
- 15 Hz: raw -0.221, source corrected -0.072, permutation -0.198, opposite sign -0.178.

## Interpretation boundary

This is a target-component mechanism diagnostic, not a replacement field for
released `mod_idx_dg`. It preserves the non-carrier PSTH exactly and asks how
the released estimator responds when the acquisition-defined carrier phase is
made comparable across trials. Any residual gap still requires multi-session
Allen replication and homologous RF/layer/rate population matching.
Session-level tests are recorded in `paired_session_tests.csv`; the unit counts
describe heterogeneity and are not treated as independent inferential replicates.
