# Allen SF/TF preference surfaces over receptive-field position

## Status: nonlinear preference surfaces implemented

These are surfaces of the released per-unit preferred bins, not full
response-amplitude tuning curves. Only Brain Observatory sessions are used:
Functional Connectivity presented a single 2-Hz drifting-grating condition
and therefore cannot identify temporal-frequency preference.

The primary surface uses a session-balanced Gaussian kernel with a
12° bandwidth. Sensitivities use 8°, 12°, 16°. Grid cells require at least three effective sessions and 20 nearby units.
Preferences are smoothed on a log2 scale so adjacent octave steps are equally spaced.

## Primary surface coverage

| Preference | Area | Units | Sessions | Supported grid | Surface median | Surface 10–90% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SF | AL | 2,036 | 24 | 100.0% | 0.0505 | 0.0446–0.0565 |
| SF | AM | 1,737 | 25 | 99.9% | 0.0495 | 0.0449–0.0661 |
| SF | HVA pooled | 7,733 | 31 | 100.0% | 0.049 | 0.046–0.0512 |
| SF | LM | 1,211 | 23 | 100.0% | 0.0497 | 0.0472–0.0546 |
| SF | PM | 1,107 | 19 | 95.6% | 0.0537 | 0.0497–0.0615 |
| SF | RL | 1,642 | 27 | 97.1% | 0.0418 | 0.0385–0.0528 |
| SF | V1 | 3,186 | 31 | 98.9% | 0.0599 | 0.0571–0.0649 |
| TF | AL | 2,036 | 24 | 100.0% | 3.12 | 2.69–4 |
| TF | AM | 1,737 | 25 | 99.9% | 2.79 | 2.33–3.5 |
| TF | HVA pooled | 7,733 | 31 | 100.0% | 2.79 | 2.55–3.46 |
| TF | LM | 1,211 | 23 | 100.0% | 2.57 | 2.31–2.95 |
| TF | PM | 1,107 | 19 | 95.6% | 2.26 | 2.05–2.69 |
| TF | RL | 1,642 | 27 | 97.1% | 3.14 | 2.68–3.78 |
| TF | V1 | 3,186 | 31 | 98.9% | 2.02 | 1.86–2.37 |

## HVA differences from paired-session V1 surfaces

Positive values indicate a higher preferred frequency than V1 at the
same RF coordinate; one log2 unit is one octave.

| Preference | Area | Shared grid | Median difference (octaves) | Spatial 10–90% |
| --- | --- | ---: | ---: | ---: |
| SF | AL | 94.8% | -0.200 | -0.474 to +0.037 |
| SF | AM | 97.6% | -0.276 | -0.472 to +0.106 |
| SF | HVA pooled | 98.9% | -0.282 | -0.454 to -0.212 |
| SF | LM | 93.2% | -0.294 | -0.456 to -0.142 |
| SF | PM | 92.6% | -0.132 | -0.259 to +0.065 |
| SF | RL | 95.4% | -0.551 | -0.679 to -0.212 |
| TF | AL | 94.8% | +0.591 | +0.287 to +0.967 |
| TF | AM | 97.6% | +0.409 | +0.069 to +0.812 |
| TF | HVA pooled | 98.9% | +0.442 | +0.237 to +0.766 |
| TF | LM | 93.2% | +0.324 | +0.126 to +0.635 |
| TF | PM | 92.6% | +0.135 | -0.050 to +0.426 |
| TF | RL | 95.4% | +0.625 | +0.265 to +0.956 |

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
- `run_manifest.json`: input, code, parameters, and output checksums.
