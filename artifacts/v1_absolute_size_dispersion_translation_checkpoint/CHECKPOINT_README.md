# V1 absolute-size and RF-center-dispersion translation checkpoint

This exploratory checkpoint estimates translation from improved V1 RF fits without
using SF, TF, HVA measurements, within-animal RF-size normalization, or an RF-center
screen-edge exclusion.

## Data and design

- 54 sessions and 5,529 V1 aperture-fit units.
- Absolute `log2(axis_area_deg2)` is retained between animals.
- Local RF-center dispersion uses a 15-degree Gaussian neighborhood and contains
  covariance trace plus two normalized anisotropy components.
- Each target is compared with an equal-session, leave-one-animal-out template.
- Independent target-cell halves use the same external template.
- Translation is searched on a 2-degree grid over +/-30 degrees.
- The main run retains all fitted area values.
- The sensitivity retains all 5,529 RF centers for dispersion but omits the 1,334
  parameter-bound area estimates from the size surface. It does not remove RFs based
  on proximity to the stimulus-grid edge.

## Concrete-case result

The parameter-bound-area sensitivity selected cases algorithmically from combined
split-half reproducibility:

| Role | Session | V1 units | Bound fraction | Combined half difference | Size half difference | Dispersion half difference |
|---|---:|---:|---:|---:|---:|---:|
| Most reproducible non-bound | 742951821 | 81 | 14% | 2.0 deg | 44.0 deg | 0.0 deg |
| Median non-bound | 762602078 | 108 | 47% | 19.0 deg | 38.0 deg | 19.0 deg |
| Failure-prone | 715093703 | 93 | 17% | 84.9 deg | 38.5 deg | 56.0 deg |

In these cases, absolute RF size alone has a broad translation objective and poor
split-half agreement. Local dispersion creates visibly structured objectives and is
responsible for the reproducible solution in the best case, but it remains broad or
unstable in the median and failure cases. These examples do not yet establish
population-level reliability.

## Files

- `Figure_v1_absolute_size_dispersion_translation_cases.png`: all-fit concrete cases.
- `translation_optima_all_sessions.csv`: full and split-half optima for all sessions
  and all three evidence modes.
- `selected_case_audit.csv`: auditable selection roles and criteria.
- `uncensored_size_sensitivity/`: parameter-bound-area sensitivity with the same
  output structure.
- `run_manifest.json`: inputs, hashes, definitions, and parameters.

