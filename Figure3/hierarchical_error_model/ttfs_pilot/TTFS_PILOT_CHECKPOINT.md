# Hierarchical identity model — TTFS pilot checkpoint

_Generated 2026-08-24; minimum 5 units; 4 chains × 1500 retained draws after 1000 tuning iterations._

## Model

Observed session × group means are modeled with their neuron-bootstrap SE as known
measurement error. Dataset-specific stable identity, session, and session × identity
scales receive half-normal priors on a common TTFS scale. Gaussian group and session
effects are sampled conditionally; scales use log-slice sampling.

## Posterior summaries

| Dataset | Component | Mean | Median | 95% interval | split-Rhat | ESS |
|---|---|---:|---:|---:|---:|---:|
| Within-V1 | sigma_identity | 1.755 | 1.515 | [0.193, 4.724] | 1.0014 | 1348 |
| Within-V1 | sigma_session | 0.796 | 0.691 | [0.018, 2.293] | 1.0180 | 258 |
| Within-V1 | sigma_interaction | 2.060 | 2.032 | [1.392, 2.930] | 1.0018 | 2823 |
| Post-V1 | sigma_identity | 2.869 | 2.663 | [1.460, 5.403] | 1.0008 | 3242 |
| Post-V1 | sigma_session | 1.887 | 1.886 | [1.212, 2.583] | 1.0012 | 780 |
| Post-V1 | sigma_interaction | 2.995 | 2.988 | [2.597, 3.422] | 1.0000 | 2243 |

## Primary pilot contrast

HVA−V1 stable identity SD: posterior mean +1.113 ms, 95% interval [-2.227, +4.133] ms, P(HVA > V1) = 0.804.

## Claim gate

The identity and interaction scales pass the pilot convergence screen. The Within-V1
session scale does not: its near-zero funnel yields split-Rhat above 1.01 and low ESS.
This pilot is therefore diagnostic, not production inference. Do not extend the current
parameterization to the remaining metrics until that geometry is resolved.

## Next smallest step

Compare a non-centered session parameterization against a simpler no-session-random-effect
sensitivity model for TTFS. Retain the richer model only if convergence improves and the
identity contrast is stable.

## Artifacts

- `posterior_chains.npz`: retained scale chains.
- `posterior_summary.csv`: component summaries and convergence diagnostics.
- `identity_contrast.csv`: direct stable-identity contrast.
- `pilot_diagnostics.png` and PDF: traces and marginal posteriors.
