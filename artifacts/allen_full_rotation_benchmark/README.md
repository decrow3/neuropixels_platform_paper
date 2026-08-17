# Full-population rotation benchmark

This checkpoint estimates production runtime using the smallest and largest
published-like V1/HVA populations among the 50 newly downloaded Allen Visual
Coding Neuropixels sessions.

Both sessions use all Gabor presentations, with no gaze calibration or
eye-tracking validity filter. The initial checkpoint gives every published-like
unit axis-aligned point and aperture fits, and every held-out evaluation unit a
freely rotated version using the existing five angle starts. A second checkpoint
tests rotation for every published-like unit.

| Session | Selection role | Fit units | Extraction | Evaluation-unit rotation | All-unit rotation |
|---:|---|---:|---:|---:|---:|
| 744228101 | Low population extreme | 186 | 40.98 s | 4 min 28.44 s | 7 min 29.26 s |
| 786091066 | High population extreme | 621 | 49.81 s | 19 min 9.49 s | 31 min 9.36 s |

Across the 404 rotated held-out evaluation units pooled across these sessions, the
rotated aperture model improves held-out Poisson deviance for 58.7% of units.
The median gain is 0.000233 and the median rotated/axis-aligned area ratio is
0.999997. Rotation therefore adds modest predictive information but does not
materially change population RF area in this checkpoint.

For all-unit rotation, interpolating the observed 2.42--3.01 seconds per unit
over the 20,879-unit 58-session population gives 15.55 CPU-hours of fitting.
Greedy session-level allocation projects 3.95 fit-hours with four workers or
2.64 fit-hours with six workers, before extraction, figures, validation, and
additional optimizer-tail allowance.

The high-population session contained one reproducibly slow aperture fit: unit
951025691 consumed 211.6 seconds and had negative held-out rotation gain. The
median model fit took 1.06 seconds and the 99th percentile took 7.27 seconds.
Production outputs now record optimizer evaluations and elapsed time per model
so such tails can be identified rather than hidden in session averages.
