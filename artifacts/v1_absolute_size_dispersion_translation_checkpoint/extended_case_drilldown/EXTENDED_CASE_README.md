# Extended covariance-trace case comparison

Four sessions were selected after fitting covariance-trace translation across all
usable sessions. The original three drill-down sessions were excluded from selection.

## Selection roles

| Session | Role | V1 units | Trace half difference | Full-dispersion half difference |
|---|---|---:|---:|---:|
| 760345702 | CCF-available trace success | 92 | 2.0 deg | 10.0 deg |
| 719161530 | Typical trace case | 83 | 14.4 deg | 37.6 deg |
| 759883607 | Trace rescues full-dispersion instability | 92 | 8.5 deg | 58.5 deg |
| 835479236 | Trace failure/boundary | 74 | 60.0 deg | 10.0 deg |

## Exact-support covariance-trace shuffle

The null keeps every RF center fixed and permutes covariance-trace values across
those locations independently in the full and split target sets.

| Session | Real half difference | Null median | Null as or more reproducible | Null full fits as well as real |
|---|---:|---:|---:|---:|
| 760345702 | 2.0 deg | 4.2 deg | 22% | 0% |
| 719161530 | 14.4 deg | 18.9 deg | 17% | 0% |
| 759883607 | 8.5 deg | 4.0 deg | 87% | 0% |
| 835479236 | 60.0 deg | 35.2 deg | 90% | 26% |

The real covariance-to-location organization improves full-session template fit in
the first three cases. Split stability is less discriminating because the exact RF
support can constrain the optimum even after descriptor shuffling.

## Physical sampling

| Session | RF-space versus CCF-neighbor covariance trace rho | Pairwise CCF versus RF distance rho |
|---|---:|---:|
| 760345702 | -0.30 | 0.07 |
| 719161530 | 0.15 | 0.15 |
| 759883607 | 0.23 | 0.11 |
| 835479236 | 0.49 | 0.42 |

The failure case has the strongest relationship between physical sampling and the
RF dispersion statistic. This supports explicitly separating RF-space covariance
from CCF-neighborhood covariance before using dispersion for registration.

## Interpretation

- Covariance trace is preferable to anisotropy, which often selects boundaries.
- A low split-half difference is not sufficient evidence because RF support can
  stabilize shuffled descriptors.
- Full-session loss relative to an exact-support shuffle and weak CCF dependence are
  useful additional gates.
- The current cases remain exploratory and do not yet establish a production
  translation.

