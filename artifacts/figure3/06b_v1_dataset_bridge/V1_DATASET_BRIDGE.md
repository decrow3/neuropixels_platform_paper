# V1 cross-dataset bridge checkpoint

## Outcome: claim gate closed

The known grating and flash protocol differences have now been matched in raw
diagnostic bridges. Unreset MouseV2 grating start phase explains a material part
of the coherence loss. Carrying the source-phase correction through the unchanged
Welch estimator substantially narrows, but does not close, the representative Allen
gap; the residual cross-probe state does not provide an additional repair.
The absolute Allen V1 point is therefore not a calibrated MouseV2 baseline.
The defensible current
claim is within-dataset only; no offset or mean-matching correction is applied.

## What the current tables show

- Equal-session log10 modulation index is -0.107 in MouseV2, 0.040 in Allen Brain Observatory, and 0.199 in Allen Functional Connectivity.
- MouseV2 minus Allen Brain Observatory is -0.147 (session-bootstrap 95% CI -0.185 to -0.109); MouseV2 minus Allen Functional Connectivity is -0.306.
- For log10 F1/F0, MouseV2 minus Allen Brain Observatory is instead +0.086 (+0.059 to +0.114). The large downward offset is therefore specific to `mod_idx_dg`, not a general loss of grating modulation.
- Valid pooled-flash timescale is 47.53 ms in MouseV2 and 43.88 ms in Allen Brain Observatory, a +3.65-ms session-level difference (+0.49 to +6.88 ms).
- Timescale validity retains 2,375/11,242 MouseV2 common-QC units and 882/4,121 Allen V1 common-QC units.
- The completed all-session MouseV2 raw bridge restricted preference to Allen's SF = 0.04 cycles/degree. It changed the equal-site mean log10 modulation index by only +0.009 (site range -0.019 to +0.036), so varying SF/preference selection does not explain the dataset offset. The corresponding F1/F0 change was +0.061.
- In representative checksum-verified Allen sessions, the common 1-s/15-trial estimator leaves MouseV2 modulation -0.186 log10 below Brain Observatory and -0.220 below Functional Connectivity.
- The corresponding harmonized F1/F0 differences are only -0.032 and -0.035 log10. The residual modulation gap is therefore metric-specific, not explained by the known spectral-window mismatch.
- The phase decomposition locates the discrepancy after trial averaging: weighted phase coherence is 0.387 in MouseV2 versus 0.539 in representative Allen BO and 0.516 in representative Allen FC.
- MouseV2 mean single-trial F1 amplitude is not smaller (+0.645 log10 versus +0.499 and +0.614), but it loses -0.494 log10 during coherent averaging versus -0.321 and -0.345 in Allen. This supports phase/latency variability as the proximate cause of the Welch gap.
- The frozen acquisition source identifies one cause: MouseV2 grating phase advances from the absolute block frame and is not reset at onset. Reconstructing the 135-frame schedule raises equal-session coherence from 0.387 to 0.433; the affected 1/2/15-Hz units rise from 0.373 to 0.438, with all eight sessions moving in the predicted direction.
- This source-defined adjustment is partial: residual coherence gaps are -0.106 versus representative Allen BO and -0.083 versus representative Allen FC. The predicted phase-stable 4/8-Hz controls are unchanged.
- Source-corrected residual phase has weak matched-trial structure across separate probes: equal-session alignment is 0.173 versus 0.141 after independent within-condition trial shuffling (aggregate p=0.0010); 8/8 sessions individually exceed the one-sided 0.05 threshold.
- That signal does not repair the gap. Correcting each unit only with the other probes changes coherence from 0.433 to 0.429, versus 0.397 for shuffled correspondence. A simple probe-global residual timing shift is therefore unlikely to be the main remaining cause.
- After orientation/TF stratification, a 50% valid-eye-coverage requirement, and linear/quadratic block-time control, residual population phase covaries descriptively with running (0.067 versus 0.039, aggregate p=0.002) and pupil x/y (0.093/0.092 versus 0.039/0.039, p=0.002/0.002); pupil area is not supported (p=0.092). These associations do not establish a behavioral cause of the dataset offset.
- Passing the source-defined carrier correction through the unchanged released Welch estimator raises the MouseV2 equal-session log10 modulation index from -0.098 to +0.019. All 8/8 sessions move upward (exact two-sided sign-test p=0.0078), versus -0.123 under phase permutation and -0.087 for the opposite-sign rotation.
- The effect is protocol-specific: affected 1/2/15-Hz units move from -0.115 to +0.048, while the predicted 4/8-Hz negative control remains -0.054. Corrected MouseV2 remains -0.069 below representative Allen BO and -0.104 below representative Allen FC, so this is a mechanism diagnostic rather than a replacement released field.
- Matching MouseV2 from 300 to Allen's balanced 150 flashes lowers its timescale center to 45.92 ms, explaining 1.61 ms of the original offset and leaving +2.04 ms versus Brain Observatory.

## Identified non-equivalences

1. Allen drifting-grating PSTHs use a 2-s analysis window; MouseV2 gratings last 1 s.
2. With `nperseg=1024`, Welch uses 1,024-sample segments for Allen but is reduced to 1,000 samples for MouseV2.
3. The released `np.searchsorted` lookup lands exactly on 1/2/4/8/15 Hz for the 1-s Mouse grid but on the next higher Welch bins for Allen (for example, 2 Hz maps to 2.930 Hz rather than the nearest 1.953-Hz bin). Thus identical source code does not define an identical spectral measurement.
4. Allen V1 combines Brain Observatory and Functional Connectivity stimulus sets. The latter is visibly shifted in `mod_idx_dg`, while its F1/F0 center is nearly unchanged.
5. Allen uses fixed grating spatial frequency; MouseV2 varies spatial frequency and selects a preferred orientation x TF x SF condition.
6. The available MouseV2 `firing_rate_dg` is preferred-condition rate, whereas the released Allen field is an overall block firing rate. It cannot yet be used as a matched covariate or cross-dataset filter.
7. Flash polarity and trial count are now matched (75 bright + 75 dark). Trial matching changes both the MouseV2 center and the fraction passing the spike-count/error gate; layer/RF population support remains unmatched.
8. MouseV2 grating phase is advanced as `TF * current_frame / fps` without a presentation-onset reset. The 1-s stimulus + 1.25-s blank schedule therefore mixes starting phases at 1, 2, and 15 Hz, whereas 4 and 8 Hz remain phase stable.
9. `mod_idx_dg` is phase-coherence sensitive because it analyzes the trial-averaged PSTH. Source-derived start phase explains part, but not all, of MouseV2's lower coherence. A target-component source-phase correction materially narrows the Welch-index gap with TF-specific and sign/permutation controls, but does not close it; unmatched population support and other dataset differences remain.

The included Monte Carlo table demonstrates estimator sensitivity only; it is not a biological correction and cannot identify how much of the observed offset is protocol versus population.

## Acceptance analysis before the main result is used

1. Completed: verified raw Allen NWBs reproduce common-QC released grating metrics, and all eight MouseV2 sessions are recomputed at SF = 0.04.
2. Completed as a representative-session diagnostic: first 1 s, 15 trials, shared support, fixed frequency grid, and exact target frequency. Expand across Allen sessions before estimating a population dataset coefficient.
3. Completed: retain released `mod_idx_dg` as a historical sensitivity and harmonized F1/F0 as a co-primary diagnostic; the two metrics lead to different cross-dataset conclusions.
4. Completed: MouseV2 flash polarity/trial support is matched to Allen with repeated trial draws and explicit selection-flow reporting.
5. Completed as a representative-session mechanism diagnostic: decompose per-trial F1 amplitude, coherent amplitude, phase coherence, and target/off-target PSD. Phase inconsistency explains why F1/F0 and Welch modulation lead to different cross-dataset conclusions.
6. Completed: reconstruct MouseV2 start phase directly from the frozen acquisition source and chronological presentation id, with TF-specific and phase-permutation controls. It materially raises carrier coherence but leaves a residual Allen gap.
7. Completed: pass the source-phase carrier correction through the unchanged released Welch estimator, preserving the mean and every non-carrier PSTH component. It raises all eight session centers and materially narrows the representative Allen gap, with TF-specific, opposite-sign, and phase-permutation controls.
8. Completed: test residual presentation-level phase across simultaneously recorded probes and against running and eye state, using leave-one-trial-out phase estimates, other-probe prediction, within-condition permutations, and block-time sensitivity. Shared/behavioral structure exists, but the other-probe correction does not restore coherence.
9. Remaining: expand the raw Allen diagnostic across sessions and match homologous RF/layer/population support. Do not mean-match response metrics.
10. Current decision: the pass criterion still fails for absolute modulation index. Restrict the claim to within-dataset results and treat the Allen V1 point as context unless a multi-session/common-population bridge changes this conclusion.

## Protocol provenance

- MouseV2 protocol snapshot: `config/mousev2_stimulus_manifest.json` (1-s gratings, 15 repeats, five SFs; 300 flashes).
- Released implementation: AllenSDK `brain_observatory/ecephys/stimulus_analysis/drifting_gratings.py` (`trial_duration=2.0`, `nperseg=1024`, and `np.searchsorted`).
- Allen Visual Coding documentation: https://allenswdb.github.io/physiology/ephys/visual-coding/vcnp-stimulus.html
- AllenSDK dataset documentation: https://allensdk.readthedocs.io/en/stable/visual_coding_neuropixels.html

## Outputs

- `Figure_v1_dataset_bridge.png`: session-level diagnostic figure.
- `center_summary.csv`: pooled-unit and equal-session centers.
- `session_metric_summary.csv`: one row per session and metric.
- `dataset_contrasts.csv`: session-bootstrap dataset offsets.
- `welch_frequency_lookup.csv`: duration-dependent spectral-bin audit.
- `welch_protocol_sensitivity.csv`: controlled estimator simulation.
- `timescale_coverage.csv` and `tf_session_summary.csv`: selection and TF diagnostics.
- `data/imports/mousev2_grating_common_support_v1/`: all-session raw MouseV2 SF = 0.04 bridge and diagnostic figure.
- `data/imports/allen_v1_raw_bridge_v2/`: checksum-verified raw Allen reproduction and common-window diagnostic.
- `data/imports/mousev2_timescale_trial_bridge_v1/`: balanced 150-flash trial sensitivity and selection flow.
- `data/imports/v1_grating_phase_bridge_v1/`: single-trial amplitude, phase-coherence, TF-stratified, and adjusted diagnostics.
- `data/imports/mousev2_grating_start_phase_bridge_v1/`: acquisition-source phase reconstruction, permutation controls, and residual-gap diagnostic.
- `data/imports/mousev2_grating_corrected_welch_bridge_v1/`: target-component source-phase correction through the released Welch estimator, with TF/sign/permutation controls.
- `data/imports/mousev2_grating_shared_phase_behavior_v1/`: cross-probe residual-phase, behavior, time-control, and permutation diagnostics.
