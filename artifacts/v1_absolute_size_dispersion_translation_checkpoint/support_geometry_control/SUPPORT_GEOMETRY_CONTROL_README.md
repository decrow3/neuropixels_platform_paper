# V1 RF-dispersion support-geometry checkpoint

## Corrected question

The nuisance is not every dispersion component predictable from anatomical
position. Genuine retinotopic dependence of RF scatter should be anatomically
organized. The narrower nuisance is RF covariance introduced when a local RF-space
neighborhood pools cells from separated anatomical positions whose expected mean
RFs differ.

For each session and mean-map bandwidth, this checkpoint:

1. fits leave-one-cell-out smooth azimuth and elevation over probe vertical
   position;
2. retains the exact cells and weights in every 15-degree RF-space neighborhood;
3. calculates covariance of their predicted mean RFs (sampling-only covariance);
4. calculates covariance of their residual RF vectors (conditional scatter).

## Three preregistered case roles

| Session | Role |
|---|---|
| 760345702 | Previous covariance-trace success |
| 719161530 | Previous typical case |
| 835479236 | Previous failure and strongest CCF association |

## Initial evidence

At a 250-um mean-map bandwidth, the median sampling-only covariance is 0.7%,
2.0%, and 2.1% of raw covariance in the success, typical, and failure cases. At
120 um, these values increase to 6.3%, 15.2%, and 21.6%, respectively.

The sampling-only pattern correlates positively with raw covariance in session
760345702 (rho=0.70 at 250 um), negatively in 719161530 (rho=-0.77), and weakly
negatively in 835479236 (rho=-0.24). Thus anatomical support can share spatial
organization with raw covariance without explaining much of its magnitude, and it
does not explain the raw spatial pattern consistently across these cases.

## Current interpretation

The prior along-shank residualization was over-broad and its rejection of the
dispersion anchor is superseded. This narrower diagnostic does not show that
support geometry explains most RF covariance. The failure case is most sensitive
to a finer mean-map scale, which is consistent with some sampling inflation, but
the result depends materially on the unresolved biological smoothness scale.

The conditional residual covariance can exceed raw covariance because the raw,
predicted-mean, and residual covariance matrices contain cross-covariance terms.
It should therefore be treated as a direct conditional-scatter estimate rather
than interpreted as a literal nonnegative raw-minus-nuisance remainder.

## Outputs

- `Figure_v1_support_geometry_control_cases.png`
- `Figure_v1_support_geometry_bandwidth_sensitivity.png`
- `session_bandwidth_audit.csv`
- `unit_neighborhood_support_decomposition.csv.gz`
- `analysis_metadata.json`

This is an exploratory concrete-case checkpoint, not a population result or a
validated registration.
