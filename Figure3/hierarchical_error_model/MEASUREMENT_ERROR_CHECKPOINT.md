# Hierarchical error model — measurement-error checkpoint

_Generated 2026-08-24 with 1000 neuron-bootstrap repetitions per session × group cell._

## Scope

This checkpoint estimates the observed mean and sampling uncertainty for every retained
session × probe/area cell. It does **not** yet fit hierarchical variance components.

The resampling unit is a neuron within a session × group cell. This captures finite-unit
uncertainty in the retained metric values, but not shared trial noise, spike-sorting
dependence, or uncertainty from rerunning each per-unit timescale fit and validity gate.

## Summary

| Dataset | Metric | Cells | Sessions | Median units | Minimum units | Median bootstrap SE | P90 bootstrap SE | Median bootstrap/analytic SE | Spearman log(n) vs log(SE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Within-V1 | TTFS (ms) | 32 | 8 | 231.0 | 145 | 1.127 | 1.264 | 0.992 | -0.878 |
| Within-V1 | log10 F1/F0 | 32 | 8 | 347.0 | 230 | 0.01238 | 0.01433 | 1.001 | -0.721 |
| Within-V1 | Response timescale (ms) | 32 | 8 | 73.0 | 23 | 3.207 | 5.017 | 1.000 | -0.924 |
| Post-V1 | TTFS (ms) | 260 | 58 | 67.0 | 7 | 1.866 | 2.73 | 0.992 | -0.927 |
| Post-V1 | log10 F1/F0 | 264 | 58 | 119.0 | 7 | 0.02596 | 0.03536 | 0.993 | -0.810 |
| Post-V1 | Response timescale (ms) | 243 | 58 | 24.0 | 5 | 3.847 | 7.164 | 0.974 | -0.813 |

## Claim gate for the hierarchical fit

Proceed only if bootstrap SEs are finite and positive, extreme cells are traceable to
small or intrinsically heterogeneous unit samples, and bootstrap/analytic SE ratios
remain near one for the mean estimand. A large departure would indicate a coding or
transformation problem rather than useful extra uncertainty information.

## Required human checkpoint

Review the selected lowest, typical, and highest-error cells in
`measurement_error_diagnostic_cases.csv`. The next stage will fit a model only after
these measurement-error inputs are accepted.

## Artifacts

- `session_group_measurement_error.csv`: model-ready observed means and known-error inputs.
- `measurement_error_summary.csv`: dataset × metric precision audit.
- `measurement_error_diagnostic_cases.csv`: transparent case selection for review.
- `measurement_error_diagnostics.png` and PDF: precision and sanity-check plots.
