# Analytic aperture RF comparison: session 746083955

This analysis compares the ordinary point-center Gaussian RF model with a model
that analytically integrates the same Gaussian RF over the known 10-degree-radius
circular stimulus aperture. It does not render the Gabor carrier, rasterize the
aperture, or model physical screen clipping.

All 318 published-like QC units fit successfully. The predeclared held-out
evaluation population contains 160 units (23 V1 and 137 HVA). Confidence
intervals below are unit-bootstrap 95% intervals.

## Main results

- Nominal aperture versus point held-out Poisson-deviance improvement was
  essentially zero: median 0.0000021 (95% CI -0.0000034 to 0.0000116), with
  54.4% of units improving (46.9% to 61.9%). Thus the coarse stimulus grid does
  not distinguish the two response parameterizations by prediction alone.
- The aperture model's median latent half-maximum area was 0.874 times the point
  estimate (log2 ratio -0.194, -0.242 to -0.169).
- By area, the median ratio was 0.699 in V1 (0.506 to 0.791; n=23) and 0.885 in
  HVA (0.858 to 0.893; n=137).
- The nominal aperture fit was censored for 25.6% of evaluation units. In 13.1%
  at least one latent Gaussian sigma reached the predeclared 2-degree lower
  bound (30.4% of V1 and 10.2% of HVA units). These fits mean "unresolved below
  the aperture scale," not a precise 2-degree RF width.
- The median size reduction was similar near the sampled-grid edge (log2 ratio
  -0.177; n=78) and in the interior (-0.224; n=82). This does not support the
  aperture correction being only an edge artifact.
- With the previously selected population gaze transform (x gain 0, y gain
  0.5), the aperture model's median held-out gaze improvement was 0.000240
  (0.000055 to 0.000405); 58.8% improved (51.3% to 66.3%). The paired difference
  between aperture- and point-model gaze benefit was essentially zero, so
  aperture handling did not reveal a stronger gaze-correction effect in this
  session.

## Interpretation

The known aperture can be included cheaply and accurately as an analytic
forward model. It provides a useful sensitivity analysis and a plausible
deconvolved latent Gaussian size, but it is weakly identifiable for RFs near or
below the aperture scale. Report the point and aperture estimates together,
flag all bound-censored fits, and validate recovery on synthetic RFs before
using the aperture estimates as replacement population sizes.

`latent_halfmax_area_deg2` is the area enclosed by the fitted Gaussian's
half-maximum contour. It is not Allen's released connected-component `area_rf`.

