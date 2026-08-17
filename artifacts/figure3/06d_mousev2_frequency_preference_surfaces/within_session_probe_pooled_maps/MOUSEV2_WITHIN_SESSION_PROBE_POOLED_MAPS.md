# MouseV2 maps pooled across simultaneous probes within session

The 8 complete A/B/C/E sessions are shown separately. Within each
session, units from all four simultaneous probes are pooled before smoothing.
Original display-centered RF coordinates are retained: no session translation or
cross-session alignment is applied. Consequently, each row preserves the screen
and unmeasured eye-position state shared by that session's probes.

RF density uses supported RF units. SF and TF use their independently supported
tuning populations, so the three maps do not contain identical unit sets.

| Session | Mouse | Map | Units | Probes | Supported grid |
| --- | ---: | --- | ---: | ---: | ---: |
| site2 | 816305 | rf_density | 120 | 4 | 66.4% |
| site2 | 816305 | sf | 87 | 4 | 58.4% |
| site2 | 816305 | tf | 47 | 4 | 26.1% |
| site3 | 810531 | rf_density | 211 | 4 | 80.6% |
| site3 | 810531 | sf | 159 | 4 | 73.9% |
| site3 | 810531 | tf | 93 | 4 | 63.4% |
| site4 | 810532 | rf_density | 94 | 4 | 59.5% |
| site4 | 810532 | sf | 67 | 4 | 37.4% |
| site4 | 810532 | tf | 51 | 4 | 24.7% |
| site5 | 813810 | rf_density | 237 | 4 | 87.1% |
| site5 | 813810 | sf | 185 | 4 | 82.7% |
| site5 | 813810 | tf | 114 | 4 | 72.9% |
| site6 | 815152 | rf_density | 107 | 4 | 67.3% |
| site6 | 815152 | sf | 88 | 4 | 59.1% |
| site6 | 815152 | tf | 59 | 4 | 34.0% |
| site7 | 816308 | rf_density | 90 | 4 | 30.2% |
| site7 | 816308 | sf | 66 | 4 | 16.3% |
| site7 | 816308 | tf | 44 | 4 | 12.8% |
| site8 | 817334 | rf_density | 88 | 4 | 53.3% |
| site8 | 817334 | sf | 63 | 4 | 31.5% |
| site8 | 817334 | tf | 40 | 4 | 17.9% |
| site9 | 817335 | rf_density | 163 | 4 | 77.5% |
| site9 | 817335 | sf | 128 | 4 | 64.8% |
| site9 | 817335 | tf | 80 | 4 | 47.5% |

## Interpretation boundary

This is the direct within-session intermediate view. It controls design-wise
for session-level screen geometry and shared recording state, but it does not
measure gaze or prove that the eyes were stationary. Differences between session
maps may reflect gaze, targeting, biological variation, or finite sampling.
