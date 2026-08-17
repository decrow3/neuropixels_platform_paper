# Extended leave-one-animal-out V1 support control

## Cohort and fixed analysis

The same estimator used in the three-case checkpoint was applied to all 45
sessions with sufficient V1 CCF support. Each animal contributes one session.
The mean CCF-to-RF geometry is trained on six robust physical blocks from the
other 44 animals after removing a translation from every training session.

The quadratic AP/ML map is primary and the affine map is a prespecified
sensitivity model. Each target's held-out block gradient is evaluated against a
constant session-mean prediction before using the map for a sampling correction.

## Population results

- Median held-out gradient R2 versus the session mean: 0.324.
- 40/45 sessions (88.9%) have positive held-out R2.
- Affine and quadratic held-out R2 values agree closely (Spearman rho=0.978).
- Median sampling-only contribution: 2.18% of raw RF covariance trace.
- 42/45 sessions (93.3%) have a median contribution below 10%.

Three sessions exceed 10%:

| Session | Held-out gradient R2 | Median sampling contribution | Maximum consecutive tangential CCF step | Total tangential span |
|---|---:|---:|---:|---:|
| 754829445 | 0.795 | 41.4% | 722 um | 1456 um |
| 754312389 | 0.887 | 22.9% | 133 um | 517 um |
| 789848216 | 0.597 | 12.1% | 94 um | 258 um |

Sampling contribution is moderately associated with maximum consecutive CCF
step (rho=0.49) and total tangential span (rho=0.55).

## Concrete selected cases

Selection roles were defined algorithmically and saved before drill-down:

| Session | Role | Gradient R2 | Sampling contribution |
|---|---|---:|---:|
| 754829445 | Largest sampling correction | 0.795 | 41.4% |
| 754312389 | Strongest held-out gradient | 0.887 | 22.9% |
| 797828357 | Typical held-out gradient | 0.324 | 2.7% |
| 759883607 | Worst held-out gradient | -1.204 | 3.1% |
| 781842082 | Largest affine/quadratic disagreement | 0.716 | 7.0% |

Session 754829445 contains two extreme jumps between consecutive physical blocks
(628 and 722 um in tangential AP/ML CCF). Its RF block medians follow the external
map well, so its large correction is consistent with the intended sampling
confound. Session 754312389 instead shows a smooth, wide tangential trajectory and
a strong held-out RF gradient; continuous anatomical extent can therefore also
inflate a raw covariance neighborhood.

The worst gradient case has a small estimated correction. Its external map should
not be trusted for detailed residual structure, but it also cannot explain away
much of the raw dispersion.

## Interpretation

The independently learned population geometry generalizes across most animals.
For most sessions, mean-map variation caused by anatomical support is much smaller
than raw RF scatter. A small minority have substantial and biologically coherent
support corrections; one also exposes pronounced CCF sampling discontinuities.

This result supports using the conditional covariance estimator, with held-out
gradient validity and correction magnitude retained as session-level diagnostics.
It does not yet establish that the corrected covariance yields reproducible
absolute translations.

## Next checkpoint

Before refitting translations, test the three high-correction sessions and the
five negative-R2 sessions for sensitivity to physical block count, AP/ML surface
projection, and individual RF-fit outliers. Then compare raw versus corrected
covariance translation stability on the remaining validated sessions.

## Outputs

- `Figure_v1_cross_animal_support_population_summary.png`
- `Figure_v1_cross_animal_support_selected_cases.png`
- `all_session_model_audit.csv`
- `selected_followup_cases.csv`
- `all_session_block_predictions.csv.gz`
- `all_session_unit_support_decomposition.csv.gz`
- `population_summary.json`

Reproduce from the repository root with:

```bash
python -m scripts.extend_v1_cross_animal_mean_map_support --overwrite
```
