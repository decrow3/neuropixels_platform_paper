# RF fit improvement: initial concrete-evidence checkpoint

## Question

Can Allen's `width_rf`, `height_rf`, and `area_rf` failure modes be identified on
native BO 1.1 spike-count maps before selecting a replacement estimator?

## Matched cases

Both units come from native session 737581020 and were processed with the AllenSDK
2.2.0 RF code path. Unit 951867908 is the well-conditioned on-screen control; unit
951868026 is the significant boundary case whose released Gaussian dimensions are
approximately 1,400 degrees.

## Immediate result

For the control, adding a nonnegative constant baseline retains a compact RF and
reduces map RMSE from 15.94 to 4.75 spike counts per grid location. Allen estimated
`width_rf = 8.73 deg` and `height_rf = 10.86 deg`; the baseline-aware screen-bounded
fit estimates 9.51 and 7.56 degrees.

For the pathological case, the observed 9 x 9 map is dominated by an approximately
constant background near 40 counts per location. Allen's no-baseline Gaussian moves
its center to `(x=317.95, y=337.69)` pixels and expands its sigmas to `(142.00,
133.25)` pixels so that one remote, nearly flat Gaussian tail mimics that background.
Adding a baseline of 39.45 counts collapses the fitted dimensions to 3.50 and 4.57
degrees with slightly lower in-sample RMSE (11.06 versus 11.35).

This identifies a concrete cause of the extreme released dimensions: the original
model has no constant response term and no bounds on center or sigma.

## Area does not yet have a validated replacement

Allen's threshold mask selects three central pixels for the control (`area_rf = 300
deg2`). For the pathological case, it selects four pixels along the rightmost grid
column (`area_rf = 400 deg2`). That component is different from the small isolated
feature selected by the baseline-aware Gaussian near `(x=2.97, y=6.33)`.

The disagreement is itself diagnostic: the released center/area and Gaussian
dimensions need not describe the same apparent feature. The threshold area is also
discrete in 100 deg2 increments and becomes a censored lower bound when its component
touches a grid edge.

## What remains unsupported

The smaller bounded dimensions are not yet an improved scientific estimate. In the
pathological case, the fitted residual bump may simply be an isolated noise peak. A
hard sigma bound can prevent numerical explosions but cannot establish that an RF is
present or recover RF area outside the sampled screen.

## Smallest useful next step

Use the 45 repeats at each spatial location for held-out validation. Compare:

1. baseline-only;
2. baseline plus screen-bounded Gaussian;
3. baseline plus a boundary-censored Gaussian whose center may lie slightly outside;
4. the original Allen Gaussian.

The candidate should be accepted only when it improves held-out Poisson deviance or
mean-squared prediction relative to baseline-only. Width and height should be
released only for accepted fits. Area should be reported as both observed threshold
area and a separately labeled model-based ellipse area, with an edge-censored flag.
