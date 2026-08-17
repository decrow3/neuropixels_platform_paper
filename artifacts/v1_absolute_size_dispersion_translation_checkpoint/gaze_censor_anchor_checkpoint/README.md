# V1/screen offset: gaze and held-out censoring checkpoint

## Question

Can variables not used by the improved RF-center fit anchor the otherwise unknown
session translation between the animal's retinotopic map and screen coordinates?

## Estimand

For each of 45 sessions, a quadratic CCF-to-RF geometry was learned from all
other animals. The session offset is the robust median of improved V1 RF center
minus the RF progression predicted from fixed CCF locations. Offsets are reported
relative to the median session because the population geometry has no absolute
visual-field intercept.

This offset is highly stable to random cell halves (azimuth Pearson r=0.995,
elevation r=0.992; median two-dimensional half discrepancy 1.22 degrees). This
establishes measurement stability within a session, not that the offset is purely
eye/screen translation rather than animal-specific map geometry.

## Absolute gaze

Absolute filtered spherical gaze was read during the Gabor block for all 40
CCF-usable sessions with eye data. Unlike the earlier gaze-correction pilot, the
session mean was retained.

The expected coordinate-matched relationships were absent:

- gaze x to anatomy-residual azimuth: Pearson r=0.011, p=0.946;
- gaze y to anatomy-residual elevation: Pearson r=0.118, p=0.468.

Separate analyses of the Brain Observatory 1.1 and Functional Connectivity
session types were also null. Leave-one-session-out linear gaze prediction had
negative R-squared for both axes (-0.100 azimuth and -0.085 elevation). The small
MAE differences from a constant (0.17 and 0.48 degrees) therefore do not amount
to useful offset prediction. A nominal cross-axis gaze-x/elevation correlation
(r=0.320, p=0.044) was not present within either session type and was not a
predeclared coordinate-matched relationship.

Allen explicitly cautions that absolute screen-gaze estimates have no accuracy
guarantee because of rig-component degrees of freedom. These data are consistent
with using gaze only for within-session fluctuation, not as an absolute anchor.

## Held-out `on_screen_rf`

The censoring diagnostic used V1 units excluded from the improved-fit population.
An empirical on-screen probability field was learned from other animals and
scored over candidate target-session translations.

In the three concrete cases, the censor-only optimum was broad and reached or
approached the +/-30 degree boundary. Applying the anatomy-derived offset improved
held-out log loss in only one case:

- 760345702: -0.0337 nats/cell (worse);
- 798911424: +0.0099 nats/cell (better, very small);
- 781842082: -0.0114 nats/cell (worse).

Thus `on_screen_rf` supplies weak edge/coverage information but does not identify
a reliable two-dimensional translation. It is also a released RF-map-derived
label, so it should be treated as held-out cells rather than a wholly independent
measurement modality.

## Checkpoint conclusion

The strongest current session signal remains the internally stable V1 RF-to-CCF
offset plus the full V1/HVA topology. Absolute Allen gaze and the released
on-screen label do not independently anchor that offset. The next high-value
test is simultaneous LGd/LP RF centers as a genuinely separate biological
measurement; a second option is to return to raw Gabor response likelihood near
screen boundaries rather than the coarse `on_screen_rf` label.

## Files

- `Figure_gaze_and_censor_anchor_checkpoint.png`: cohort gaze comparisons and
  concrete held-out censoring loss surfaces.
- `all_session_anatomy_offsets.csv`: LOAO offsets and split-half diagnostics.
- `gaze_summary.csv`: absolute Gabor-block gaze summaries.
- `geometry_gaze_comparison.csv`: merged 40-session table.
- `gaze_offset_correlations.csv`: matched- and cross-axis correlations.
- `gaze_offset_loo_prediction.csv`: leave-one-session-out prediction.
- `concrete_case_censor_results.csv`: held-out censoring results.
- `anatomy_offset_split_half_summary.json`: offset repeatability.
- `run_manifest.json`: inputs and interpretation notes.
