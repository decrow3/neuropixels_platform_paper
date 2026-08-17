# LGd population retinotopic-map data sufficiency

## Answer

There is enough LGd data to attempt a **coarse, strongly regularized population
map at approximately 150-250 um spatial scale**. There is not enough independent
sampling for an unconstrained fine-scale three-dimensional map, nor do the
released RF centers yet support precise session offsets.

## Available observations

- 2,582 LGd units across 33 sessions;
- 2,357 have released RF centers and complete 3-D CCF coordinates;
- 1,329 have `p_value_rf < 0.01`;
- 314 are both significant and classified on-screen, across 29 sessions;
- the median session has 8 such clean units.

The 1,329 significant observations at screen boundaries should not simply be
discarded. Their fitted centers are censored, but their raw response maps can
provide inequality or boundary-likelihood information after improved fitting.

## Cross-animal anatomical support

Among the 314 clean units, the median nearest unit from another animal is 96 um.

| CCF radius | At least one other animal | At least three other animals |
|---:|---:|---:|
| 100 um | 53.2% | 3.5% |
| 150 um | 98.4% | 30.9% |
| 200 um | 99.7% | 79.0% |
| 250 um | 100% | 91.4% |

Thus the corpus has good cross-animal overlap around 200-250 um, but not at
100 um. A map claiming substantially finer independent resolution would be
unsupported.

## Identifiability limitation

Individual LGd probe trajectories are nearly parallel. The eigenvalues of the
probe-direction second moment are 0.025, 0.060, and 0.915. Only 7 sessions have
two or more LGd probes.

This is crucial because an unknown translation is fitted per session. Anatomical
variation between sessions cannot by itself identify the retinotopic gradient:
it is confounded with those session translations. Within-probe progression
identifies the dominant direction, while the seven multi-probe sessions provide
most of the independent transverse information.

The pooled within-session CCF design is technically full-rank, with variance
fractions 0.691, 0.210, and 0.099, but the two weaker directions are supported by
far fewer independent trajectories than the raw cell count suggests.

## Existing map recovery

Using released p<0.01, on-screen RF centers and leave-one-animal-out quadratic
CCF geometry:

- 19 of 26 evaluable sessions had positive centered held-out R-squared;
- median held-out R-squared was 0.092;
- several well-sampled sessions reached 0.24-0.55;
- several broad or multi-probe sessions nevertheless failed.

This establishes a real population mapping signal but shows that the present
released-fit model is not reliable enough for per-session offset inference.

## Recommended map attempt

1. Refit all LGd Gabor responses with the improved RF model.
2. Preserve boundary-censored units through a censored or raw-response
   likelihood rather than an on-screen exclusion.
3. Fit a smooth 3-D vector field `CCF -> (azimuth, elevation)` with a separate
   translation intercept per session.
4. Test fixed smoothness scales of roughly 150, 200, 250, and 300 um. Select the
   scale only from held-out within-session RF differences.
5. Give multi-probe sessions explicit diagnostic weight because they identify
   transverse gradients; do not allow their larger cell counts to dominate.
6. Validate on held-out animals, physical probe sections, and held-out probes.
7. Infer thalamic session offsets only after the frozen map passes those tests.

The map's overall visual-field constant remains unidentified without an external
anchor, but its relative geometry can still provide the anatomical correction
needed for the V1/thalamus offset comparison.

## Outputs

- `Figure_LGd_population_sampling_audit.png`: anatomical coverage, RF colors,
  cross-animal support, and held-out map recovery.
- `Figure_LGd_concrete_sessions.png`: strong, typical, and contradictory cases.
- `session_sampling_metrics.csv`: session-level counts, spans, and validation.
- `cross_animal_spatial_support.csv`: support as a function of CCF radius.
- `probe_trajectory_directions.csv`: trajectory orientation audit.
- `concrete_case_selection.csv`: auditable case roles.
- `summary.json`: principal numerical results.
