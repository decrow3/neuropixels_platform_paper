# Preserved failed Iteration 1 attempt

This first Iteration 1 run stopped in `eta_squared_comparison.py` after the
three figures completed. The reproducibility refactor had removed `_site_dirs`,
but one reporting-only raw-unit count still referenced it.

The count now comes from the validated central session configuration. No metric,
filter, statistic, or figure calculation changed. The subsequent
`01_reproducible_baseline` run completed and passed equivalence against
Iteration 0.
