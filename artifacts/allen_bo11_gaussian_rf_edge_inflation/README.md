# Allen BO 1.1 Gaussian RF fits inflate near mapping-grid edges

## Answer

In the 31-session, six-area significant-RF cohort, the session-median edge/interior ratio was **11.0×** (bootstrap 95% CI **8.4–12.5×**). Edge means a released threshold-mask center ≤10° from any boundary; interior means >20° from every boundary. The edge median was larger in **31/31 sessions** (two-sided sign-test p = 9.3e-10).

At the unit level, **73.7%** of edge fits versus **7.7%** of interior fits had at least one fitted Gaussian sigma larger than the full 80° sampled span. The corresponding Gaussian-center off-screen rates were **85.1%** and **2.5%**.

## Interpretation

This is a strong boundary-associated numerical failure, not evidence for biological RFs hundreds of degrees wide. Allen fits an unbounded five-parameter Gaussian by least squares, does not constrain its center or sigmas, and ignores the returned fit-success flag when releasing the metrics. `on_screen_rf` checks only whether the fitted Gaussian center lies inside the 9×9 array; it does not test whether the fitted width is supported by the sampled field.

The edge effect is largely carried by fits whose Gaussian center extrapolates off screen. Among fits with `on_screen_rf == True`, the >80° rates were **7.6%** at the edge and **5.4%** in the interior; their unit-level median maximum sigmas were **18.2°** and **27.4°**, respectively.

The result is not dependent on choosing the larger axis alone: the median session-level edge/interior ratio for the geometric mean of |width| and |height| was **4.6×**. All six cortical areas show the same direction in the population summary.

## Definitions and cohort

The source population contains **10,919 units** with p_value_rf ≤ 0.009 from 31 BO 1.1 sessions and areas AL, AM, LM, PM, RL, V1. Four units lacked one or both Gaussian dimensions, leaving **10,915** analyzed fits.

`width_rf` and `height_rf` are Gaussian sigma parameters converted to degrees. Because the Gaussian uses each width only after squaring it, the parameter sign is non-identifiable; this analysis uses absolute magnitudes. The primary scalar is max(|width_rf|, |height_rf|), and “larger than the mapped span” means that scalar exceeds 80°.

The edge distance is computed from released `azimuth_rf`/`elevation_rf`, which come from the thresholded peak-connected component, not from the Gaussian center. That distinction is intentional: it asks whether an RF whose reproducible released location lies near the sampled boundary receives an unstable Gaussian size estimate.

## Caveat

Filtering to `on_screen_rf == True` removes the systematic edge inflation in this cohort, but it is not a general containment criterion: a Gaussian center can be on screen while a large fraction of its fitted profile lies outside the sampled support. For downstream RF size, released thresholded `area_rf` remains the safer Allen metric; the Gaussian dimensions should be treated as censored or refit with explicit bounds and edge-aware uncertainty.
