# Allen SF/TF preference surfaces over receptive-field position

## Status: nonlinear preference surfaces implemented

These are surfaces of the released per-unit preferred bins, not full
response-amplitude tuning curves. Only Brain Observatory sessions are used:
Functional Connectivity presented a single 2-Hz drifting-grating condition
and therefore cannot identify temporal-frequency preference.

The primary surface uses a session-balanced Gaussian kernel with a
12° bandwidth. Sensitivities use 8°, 12°, 16°. Grid cells require at least three effective sessions and 20 nearby units.
Preferences are smoothed on a log2 scale so adjacent octave steps are equally spaced.

Tuning-quality filters: lifetime sparseness > 0.1, stimulus firing rate > 0.1 Hz, unique preferred bin required.

## Primary surface coverage

| Preference | Area | Units | Sessions | Supported grid | Surface median | Surface 10–90% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SF | AL | 1,404 | 24 | 99.6% | 0.0476 | 0.0416–0.0558 |
| SF | AM | 979 | 25 | 99.1% | 0.047 | 0.0433–0.0668 |
| SF | HVA pooled | 4,749 | 31 | 100.0% | 0.0463 | 0.0441–0.0489 |
| SF | LM | 818 | 23 | 100.0% | 0.0475 | 0.0408–0.0519 |
| SF | PM | 591 | 19 | 83.9% | 0.052 | 0.046–0.0615 |
| SF | RL | 957 | 27 | 86.4% | 0.0396 | 0.0365–0.0479 |
| SF | V1 | 2,196 | 31 | 94.5% | 0.0598 | 0.0565–0.0636 |
| TF | AL | 1,519 | 24 | 99.9% | 2.81 | 2.4–3.65 |
| TF | AM | 1,126 | 25 | 99.8% | 2.45 | 2.11–3.38 |
| TF | HVA pooled | 5,438 | 31 | 100.0% | 2.63 | 2.41–3.26 |
| TF | LM | 931 | 23 | 100.0% | 2.38 | 2.03–2.98 |
| TF | PM | 720 | 19 | 89.2% | 2.2 | 1.87–2.87 |
| TF | RL | 1,142 | 27 | 94.1% | 3.01 | 2.49–3.92 |
| TF | V1 | 2,596 | 31 | 97.1% | 1.94 | 1.76–2.2 |

## HVA differences from paired-session V1 surfaces

Positive values indicate a higher preferred frequency than V1 at the
same RF coordinate; one log2 unit is one octave.

| Preference | Area | Shared grid | Median difference (octaves) | Spatial 10–90% |
| --- | --- | ---: | ---: | ---: |
| SF | AL | 84.0% | -0.348 | -0.553 to -0.076 |
| SF | AM | 91.2% | -0.364 | -0.492 to +0.065 |
| SF | HVA pooled | 94.5% | -0.377 | -0.440 to -0.306 |
| SF | LM | 83.9% | -0.384 | -0.628 to -0.235 |
| SF | PM | 76.0% | -0.145 | -0.364 to +0.115 |
| SF | RL | 79.8% | -0.616 | -0.736 to -0.321 |
| TF | AL | 87.4% | +0.520 | +0.213 to +0.936 |
| TF | AM | 93.7% | +0.311 | +0.046 to +0.834 |
| TF | HVA pooled | 97.1% | +0.416 | +0.283 to +0.764 |
| TF | LM | 87.7% | +0.371 | +0.079 to +0.696 |
| TF | PM | 80.7% | +0.201 | -0.161 to +0.565 |
| TF | RL | 89.9% | +0.630 | +0.317 to +1.070 |

## Interpretation boundary

The surfaces describe how preferred bins vary over achieved RF position.
They do not measure tuning bandwidth, response strength, or a joint Allen
SF × TF response surface because Allen measured SF with static gratings and
TF with drifting gratings. Each HVA is compared with a V1 surface built
only from the same Allen sessions. `delta_from_v1_log2` is defined only
where both that HVA and its paired-session V1 reference meet the local support rule.

## Outputs

- `frequency_preference_surface_grid.csv`: all bandwidths, support diagnostics, and V1 differences.
- `frequency_preference_surface_summary.csv`: area-level coverage and spatial ranges.
- `frequency_preference_population.csv`: source-unit counts by area and preference.
- `Figure_allen_frequency_preference_surfaces.png`: primary SF and TF surfaces.
- `Figure_allen_frequency_preference_differences.png`: HVA-minus-V1 surfaces on shared support.
- `Figure_allen_pooled_hva_frequency_preference_surfaces.png`: two-panel pooled-HVA SF/TF fits.
- `Figure_allen_v1_frequency_preference_surfaces.png`: matched two-panel Allen V1 SF/TF fits.
- `Figure_allen_pooled_hva_frequency_preference_surfaces_polar.png`: pooled-HVA fits in polar RF coordinates.
- `Figure_allen_v1_frequency_preference_surfaces_polar.png`: Allen V1 fits in polar RF coordinates.
- `Figure_allen_rf_occupancy_polar.png`: exact SF-/TF-population RF occupation for V1 and pooled HVAs.
- `run_manifest.json`: input, code, parameters, and output checksums.
