# Absolute V1 RF-size corroboration checkpoint

## Question

Does an independently measured V1 property, absolute improved-fit RF size,
corroborate retinal translations estimated from anatomy-corrected local RF-center
scatter?

RF size is never used to estimate the covariance translation.

## Cases fixed before RF-size outcomes

The five sessions with negative held-out CCF-to-RF gradient R2 were selected as
informative failures. Session 754312389, which had the strongest positive gradient
R2 in the 45-session support audit, was fixed as a positive control.

## Nested design

For every target session:

1. The target animal is excluded from every CCF-to-RF geometry fit.
2. Each training animal's conditional scatter is calculated with a geometry fit
   excluding both the target and that training animal.
3. Training-animal translations are estimated from conditional scatter without
   the target animal.
4. An animal-balanced absolute log2 RF-area surface is built from those translated
   training animals.
5. Target cells are divided deterministically into independent halves. Conditional
   scatter from one half estimates a translation; absolute RF sizes from the other
   half evaluate it. The direction is then reversed.
6. A null permutes held-out RF sizes over their fixed RF locations while keeping
   the covariance shift fixed.

All 4,699 improved V1 RF area fits from the 45 CCF-usable sessions are retained in
the primary analysis. No within-animal normalization or RF-center edge exclusion
is applied. A sensitivity analysis excludes parameter-bound area values from both
target and template size calculations.

## Result

RF size does not corroborate conditional-scatter translation in these cases.

| Session | Role | Covariance-size optimum distance | Mean independent-half size gain | Both gains positive | Shuffle p |
|---|---|---:|---:|---|---:|
| 759883607 | Negative gradient R2 | 30.5 deg | -0.0248 | No | 0.87 |
| 773418906 | Negative gradient R2 | 34.4 deg | -0.0012 | No | 0.37 |
| 829720705 | Negative gradient R2 | 29.7 deg | -0.0001 | No | 0.61 |
| 831882777 | Negative gradient R2 | 58.8 deg | -0.0161 | No | 0.50 |
| 840012044 | Negative gradient R2 | 39.4 deg | -0.0005 | No | 0.52 |
| 754312389 | Strongest positive-gradient control | 50.9 deg | -0.0006 | No | 0.63 |

Positive gain means lower held-out absolute-size loss at the independently
estimated covariance shift than at zero shift. No session improves in both split
directions. The size optima are broad and frequently approach the search boundary.

The uncensored-only sensitivity is also mixed and very small: covariance-versus-
zero gains range from -0.0042 to +0.0038. It does not rescue agreement.

## Interpretation

This is a failure of RF size as an independent validator, not evidence that the
conditional-scatter estimand is absent. It reproduces the earlier observation that
absolute V1 RF-size surfaces are too weak and heterogeneous to identify stable
translations. Even the strongest positive CCF-gradient control receives no size
corroboration, so the negative-gradient sessions are not uniquely responsible.

The result argues against using RF size to accept covariance translations. A
translation should not be considered validated merely because its covariance
objective is well structured. Other independent measurements, or a substantially
better-established absolute RF-size field, are still needed.

## Outputs

- `Figure_v1_rf_size_corroboration_cases.png`
- `case_selection.csv`
- `rf_size_corroboration_summary.csv`
- `rf_size_corroboration_landscapes.csv.gz`
- `rf_size_corroboration_nulls.csv.gz`
- `run_manifest.json`

Reproduce from the repository root with:

```bash
python -m scripts.test_v1_rf_size_corroboration --overwrite
```
