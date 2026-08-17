# Pooled pairwise anisotropy split-half checkpoint

## Question

Do the two concrete V1 sessions contain enough data to recover a pooled residual-RF
anisotropy axis independently in two cell halves?

For residual RF vectors `e_i = RF_i - predicted_RF_i(CCF)`, the estimator was

`A2 = sum[(delta_az + i delta_el)^2] / sum[delta_az^2 + delta_el^2]`,

where deltas are cell-pair differences. `abs(A2)` is anisotropy magnitude and half
its complex angle is the unoriented anisotropy axis.

Cells were randomly split within each of six physical probe blocks. Splitting was
repeated 1,000 times. Pairs defined the statistic, but cells—not pairs—were the
randomization unit. An orientation null independently rotated each cell residual
while retaining its magnitude.

## Main 0--300 um results

| Session | Full abs(A2) | Full axis | Median half-axis difference | P90 difference | Fraction within 15 deg | Null fraction within 15 deg | Magnitude null p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 760345702 | 0.370 | +24.9 deg | 47.9 deg | 58.8 deg | 0.4% | 15.3% | 0.036 |
| 798911424 | 0.317 | -30.5 deg | 26.6 deg | 56.8 deg | 29.3% | 17.8% | 0.198 |

Neither session passes a convincing split-half reliability criterion.

For `760345702`, the full-data magnitude exceeds the orientation null, but the axis
is less reproducible than chance across complementary halves. This dissociation is
caused by influential residual cells: removing the two largest residuals changes
the pooled axis by 42.0 deg; removing the five largest changes it by 49.6 deg.

For `798911424`, median half-axis agreement is better than the orientation null,
but the p90 remains 56.8 deg and the full magnitude does not exceed the cell-level
orientation null. Removing the five largest residuals changes the axis by 20.2 deg.

## Anatomical-separation sensitivity

The inferred reliability is not stable across CCF pair-distance bands. For
`760345702`, median half-axis differences range from 33.6 to 60.7 deg. For
`798911424`, they range from 26.6 to 36.2 deg. The apparently strongest far-pair
anisotropy is based on only 135 and 252 pairs and is particularly vulnerable to
shared-cell and outlier influence.

## Interpretation

There are enough cells to run the split-half test, and the test is informative:
the conventional length-squared pairwise anisotropy estimator is not reliable
enough to use for registration in these sessions. The thousands of nominal pairs
do not overcome the limited number of independent cells.

This does not yet rule out all pooled directional estimators. The displayed
failure specifically motivates a robust sensitivity test using equal-weight
pair angles or winsorized pair lengths. Such an estimator should only be pursued
if it yields consistent axes across cell halves and CCF-distance bands without
being driven by a few residual cells.

## Outputs

- `Figure_v1_pooled_pairwise_anisotropy_split_half.png`
- `pooled_pairwise_full_data_moments.csv`
- `pooled_pairwise_split_half_estimates.csv.gz`
- `pooled_pairwise_split_half_agreement.csv.gz`
- `pooled_pairwise_split_half_metrics.csv`
- `run_manifest.json`
- Analysis script: `scripts/test_v1_pooled_pairwise_anisotropy_split_half.py`
