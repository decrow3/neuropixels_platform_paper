# V1 grating phase-coherence bridge

All values use the harmonized 1-s, 15-trial, SF = 0.04, contrast = 0.8
support. Allen centers use the representative downloaded session per cohort;
the independent reproduction-control sessions remain visible in the tables and figure.

## Primary decomposition

- Weighted phase coherence: MouseV2 0.387; Allen BO 0.539; Allen FC 0.516.
- Log10 mean single-trial F1 amplitude: MouseV2 +0.645; Allen BO +0.499; Allen FC +0.614.
- Log10 coherent F1 amplitude: MouseV2 +0.151; Allen BO +0.178; Allen FC +0.269.
- Log10 target/off-target PSD: MouseV2 -0.014; Allen BO +0.273; Allen FC +0.293.
- Log10 amplitude lost during trial averaging: MouseV2 -0.494; Allen BO -0.321; Allen FC -0.345.
- At 1 Hz, weighted phase coherence is 0.402 in MouseV2 and 0.560 in representative Allen BO; at 2 Hz it is 0.355, 0.520, and 0.516 for MouseV2, Allen BO, and Allen FC.
- After descriptive adjustment for mean-trial F1 amplitude, preferred rate, and TF, the MouseV2 phase-coherence coefficient remains -0.151.

## Interpretation

MouseV2 does not have weaker single-trial grating modulation: its mean-trial
F1 amplitude is comparable to or higher than the representative Allen sessions.
The difference appears when trials are coherently averaged. MouseV2 loses more
amplitude to phase inconsistency, and its target-frequency peak is less distinct
from off-target power. The same direction at 1 and 2 Hz argues against preferred-TF
composition as the sole explanation.

This strongly supports trial-to-trial phase/latency variability as the proximate
mathematical cause of the low Welch modulation index. It does not distinguish
display-timestamp jitter from neural-state, eye-movement, or genuine response-phase
variability without an independent photodiode timing reference. The adjusted model
is descriptive because raw Allen primary support remains one session per cohort.
