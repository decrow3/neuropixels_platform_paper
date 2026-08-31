# TTFS session-variance parameterization checkpoint

_Generated 2026-08-24; 4 chains × 2500 draws after 1500 tuning._

The non-centered fit is the same scientific model as the centered pilot. The second
specification fixes only the V1 session SD to zero while retaining the non-centered HVA fit.

| Specification | V1 identity median | HVA identity median | Median HVA−V1 | 95% interval | P(HVA>V1) |
|---|---:|---:|---:|---:|---:|
| Non-centered session effects | 1.483 | 2.680 | +1.184 | [-2.196, +4.176] | 0.813 |
| V1 session SD fixed to zero | 1.420 | 2.680 | +1.260 | [-2.159, +4.242] | 0.816 |

## Convergence gate

Maximum split-Rhat = 1.0021; minimum ESS = 1635.
Production extension requires split-Rhat < 1.01 for every sampled component and
adequate ESS, plus a stable identity contrast across the two specifications.
