# Population effect of freely rotated Allen RF fits

Both the point-center and analytic circular-aperture models were refit with one
additional ellipse-angle parameter for all 318 published-like QC units in
session 746083955. Fits used five angle starts and the existing trial split.
The predeclared 160-unit evaluation split is primary; the 158 calibration units
were not used for the conclusions below.

All 636 rotated fits completed successfully. The rotated model is nested within
the axis-aligned model: its training Anscombe objective was never worse beyond
1e-8 numerical tolerance.

## Held-out results

- Point model: median deviance gain 0.000247 (unit-bootstrap 95% CI 0.000022 to
  0.000829); 58.8% of units improved (51.3% to 66.3%).
- Aperture model: median gain 0.000215 (-0.000002 to 0.001015); 56.9% improved
  (49.4% to 64.4%).
- Point and aperture unit-level gains strongly agreed (Spearman rho 0.874). The
  paired median aperture-minus-point gain was 0.0000012 (-0.0000022 to
  0.0000072), providing no evidence that either spatial model benefits more
  from rotation.
- HVA point fits showed the clearest evidence: median gain 0.000329 (0.000051
  to 0.001374), with 60.6% improving (52.6% to 68.6%). HVA aperture estimates
  were similarly centered but less certain: median 0.000406 (-0.000011 to
  0.001345).
- V1 was inconclusive (n=23): point median -0.000468 and aperture median
  0.000080, with wide intervals spanning zero.

## Geometry and diagnostics

- Median log2 area change was approximately zero for both models. Rotation
  changes individual orientation and eccentricity but does not systematically
  correct population RF area.
- Among 112 non-bound aperture evaluation fits with axis ratio at least 1.2,
  median absolute tilt was 22.8 degrees and 75% exceeded 10 degrees. Large
  fitted angles were not uniformly associated with held-out improvement.
- Aperture fits remained more frequently resolution-bound: 16.3% reached the
  2-degree sigma floor and 8.8% reached an upper sigma bound. Excluding both
  bounds left 120 units and a median gain of 0.000297 (-0.000070 to 0.001308).

Rotation is therefore useful for a subset of units, particularly some HVA
units, but is not a uniformly superior population model. It should remain an
optional extension selected or weighted by held-out evidence, with angles
reported only for sufficiently elliptical, non-bound fits.

