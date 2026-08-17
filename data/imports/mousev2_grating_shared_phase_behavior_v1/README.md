# MouseV2 shared residual-phase and behavior bridge

Starting phase was first removed using the frozen acquisition schedule. For each
unit and trial, intrinsic unit phase was then estimated without that trial. The
primary shared-state test predicts each unit only from units on the other three
simultaneously recorded probes. Trial labels are independently shuffled within
unit and condition for the null.

## Cross-probe phase sharing

- Equal-session source-phase-adjusted coherence is 0.433; removing the
  phase predicted from other probes changes it to 0.429, versus
  0.397 with shuffled trial correspondence.
- Cross-probe residual-phase alignment is +0.173, versus
  +0.141 under the trial-shuffled null
  (equal-session aggregate p = 0.0010).
- 8/8 sessions individually exceed the
  one-sided 0.05 permutation threshold.

A positive matched-trial excess means that residual phase displacement is shared
across physically separate probes; it cannot be produced by a target unit predicting
itself. The phase adjustment remains a mechanism diagnostic, not a replacement for
the released modulation index.

## Behavioral covariance

- absolute running speed: condition-controlled 0.061 versus shuffle 0.040 (p = 0.006); after linear/quadratic block-time control 0.067 versus 0.039 (p = 0.002; 6/8 sessions above null mean).
- log pupil area: condition-controlled 0.049 versus shuffle 0.039 (p = 0.090); after linear/quadratic block-time control 0.048 versus 0.039 (p = 0.092; 6/8 sessions above null mean).
- horizontal pupil position: condition-controlled 0.091 versus shuffle 0.039 (p = 0.002); after linear/quadratic block-time control 0.093 versus 0.039 (p = 0.002; 8/8 sessions above null mean).
- vertical pupil position: condition-controlled 0.096 versus shuffle 0.039 (p = 0.002); after linear/quadratic block-time control 0.092 versus 0.039 (p = 0.002; 7/8 sessions above null mean).
- Grating-block position itself: observed 0.039, shuffle 0.039 (aggregate permutation p = 0.437).

Behavior values are summarized during each 1-s grating and standardized within
orientation × TF condition. Eye summaries require at least 50% valid
samples in the stimulus window. The primary sensitivity removes linear and quadratic
grating-block time trends. Associations use only the angle of the probe-balanced
population phase vector and are calibrated by within-condition behavior shuffles. These tests are
descriptive across eight sessions and should not be interpreted from uncorrected
per-session p-values alone.
