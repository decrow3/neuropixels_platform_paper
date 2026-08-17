# MouseV2 multi-probe V1 SF/TF preference surfaces

## Status: supported parametric RF and tuning visualization implemented

Base population: `pilot_qc` with supported parametric RF models.
Eligible units: 1,110 across 8 sessions and four V1 probes.
Fitted RF centers within 5° of the tested field boundary: 21.4%.
Mapped extrapolated preferences: SF 74; TF 117.
Coordinate harmonization: MouseV2 display-centered positions were translated
to Allen-style released axes as `azimuth = x + 50°` and `elevation = y + 10°`.
This changes the coordinate labels and polar geometry, not the fitted Cartesian
relationships; it is not a gaze correction or a claim of eye-centered position.

| Preference | Units | Sessions | Supported grid | Surface median | Surface 10–90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF | 843 | 8 | 94.8% | 0.0814 | 0.0705–0.0872 |
| TF | 528 | 8 | 92.1% | 2.02 | 1.93–2.35 |

## Interpretation boundary

MouseV2 jointly varied SF, TF, and orientation. Each mapped preference now
comes from a joint Poisson log-Gaussian × log-Gaussian × von-Mises fit and
is retained only when its
joint tuning, axis-specific tuning, and split-half surface reliability pass
the dataset-wide FDR support contract. Pilot QC remains a unit-quality gate.
SF/TF peaks up to one octave beyond the sampled range are retained when
their fitted width and off-grid optimum remain identifiable; they are
explicitly flagged in the unit table.
RF centers come from supported trial-level elliptical Gaussian fits.
Gaze correction remains unavailable, so positions are display-centered
rather than eye-centered.

The descriptive MouseV2-minus-Allen offset is substantially larger for SF
(median 1.35x) than TF (1.07x). Plausible contributors include different
stimulus families and estimators (continuous joint fits versus released bins),
different sampled unit/RF populations, and retained identifiable extrapolated
MouseV2 peaks. This observation is noted but not adjudicated here.

## Outputs

- `mousev2_frequency_preference_surface_grid.csv`: pooled and per-probe surface grids.
- `mousev2_frequency_preference_surface_summary.csv`: bandwidth and coverage summary.
- `mousev2_rf_coordinate_mapping.csv`: raw-to-Allen-style grid-coordinate audit.
- `Figure_mousev2_v1_frequency_preference_surfaces.png`: pooled Cartesian fits.
- `Figure_mousev2_v1_frequency_preference_surfaces_polar.png`: pooled polar fits.
- `Figure_mousev2_v1_rf_occupancy_by_probe_polar.png`: pooled and per-probe RF occupation.
- `run_manifest.json`: input, code, parameters, and output checksums.
