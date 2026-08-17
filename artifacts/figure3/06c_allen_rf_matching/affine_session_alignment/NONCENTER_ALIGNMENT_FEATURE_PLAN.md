# Non-center feature plan for Allen session registration

## Status

The RF-center-to-area-consensus affine diagnostic is rejected as a coordinate
correction. RF centers are the coordinates being corrected and cannot also be
the registration landmarks.

## Candidate independent fields

An expanded reproducible audit now supersedes the compact screen below; see
`../noncenter_registration_feature_audit/ALLEN_BO11_REGISTRATION_FEATURE_AUDIT.md`.
It adds flash latency, modulation/F1–F0, response variability, selectivity, and
timescale metrics. After correcting eccentricity to Allen's released `(0°, 0°)`
origin, RF area is weak (rho = -0.069; 55% session × area gradient-sign
agreement). Dorsal–ventral CCF position is strongest (rho = -0.172), followed
by cortical depth and probe position. Modulation index and flash latency remain
weak.

Candidate features were audited within session × area after subtracting the
local median. The values below are the largest absolute Spearman association
with RF azimuth, elevation, or eccentricity among 10,919 Brain Observatory 1.1
RF-supported visual-cortical units.

| Candidate | Coverage | Maximum absolute rho |
| --- | ---: | ---: |
| Dorsal–ventral CCF coordinate | 74.2% | 0.180 |
| Cortical depth | 100% | 0.118 |
| Probe vertical position | 100% | 0.118 |
| Probe horizontal position | 100% | 0.098 |
| Anterior–posterior CCF coordinate | 74.2% | 0.083 |
| RF lifetime sparseness | 100% | 0.069 |
| Log RF area | 100% | 0.066 |
| Left–right CCF coordinate | 74.2% | 0.061 |
| RF response time to peak | 100% | 0.055 |
| Log RF width/height | 97.8% | 0.046–0.048 |

These are weak landmarks and should be combined rather than treated as direct
surrogates for RF position.

## Defensible registration test

1. Construct leave-one-session-out, area-specific scalar fields for log RF
   area/width/height, CCF coordinates, cortical depth/probe position, RF
   response time, and RF lifetime sparseness.
2. Estimate one bounded, orientation-preserving transform per held-out session
   by maximizing agreement of those non-center fields, weighted by local unit
   density and feature reliability.
3. Compare identity, translation-only, similarity, and tightly bounded affine
   models. Do not select flexibility using SF or TF.
4. Apply the selected transform unchanged to RF coordinates and evaluate SF and
   TF against leave-one-session-out templates.
5. Treat SF-to-TF and TF-to-SF registration only as explicitly cross-fitted
   sensitivities; never optimize and evaluate the same tuning field.
6. Reject registration if non-center features cannot identify stable transforms
   or if independent SF/TF agreement does not improve consistently.

RF size is derived from the same RF stimulus family but is not an RF-center
landmark. It can inform eccentricity/scale weakly; it is unlikely to identify a
full affine transform alone.
