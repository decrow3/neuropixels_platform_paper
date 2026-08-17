# MouseV2 simultaneous-probe RF and tuning maps

All 8 retained sessions contain the complete A/B/C/E probe quartet
with at least one independently supported RF, SF preference, and TF preference per probe.
RF density uses all supported RF units; SF and TF maps use their own independently
supported tuning populations. Sessions receive equal prior weight.

| Probe | RF units | SF units | TF units |
| --- | ---: | ---: | ---: |
| A | 274 | 227 | 138 |
| B | 285 | 200 | 127 |
| C | 370 | 283 | 188 |
| E | 181 | 133 | 75 |

## Alignment and interpretation boundary

A session reference is the median of the four probe-specific median RF centers.
Each session is translated to the across-session median reference before smoothing;
this preserves every within-session probe offset and prevents high-yield probes from
dominating the alignment. The largest translation was 18.4°. This is a sensitivity analysis for shared screen/eye-position
translation, not a measured gaze correction; rotations, scale changes, and genuine
session-wide retinotopic differences remain unresolved.

Probe E has sparse TF support in several sessions, so its local support is visibly
smaller and should not be interpreted where the grid is masked.
The probes target substantially different RF regions. Pairwise probe summaries
therefore require at least 50 shared supported grid points; unsupported contrasts
are left blank. Retained contrasts are descriptive surfaces, not unit-independent
inferential tests.

## Cross-dataset note retained for later

The MouseV2–Allen median offset was much larger for SF (1.35×) than TF (1.07×).
Plausible contributors include the different stimulus families and preference
estimators (continuous joint fits versus released discrete bins), different sampled
unit/RF populations, and MouseV2's explicitly retained identifiable extrapolated peaks.
The present simultaneous-probe maps do not adjudicate those explanations.

## Outputs

- `Figure_mousev2_simultaneous_probe_rf_sf_tf_polar.png`: aligned RF/SF/TF probe maps.
- `simultaneous_probe_surface_grid.csv`: plotted grid and support diagnostics.
- `simultaneous_probe_session_alignment.csv`: translation audit.
- `simultaneous_probe_population.csv`: exact session-probe populations.
- `simultaneous_probe_pairwise_surface_summary.csv`: descriptive probe contrasts.
