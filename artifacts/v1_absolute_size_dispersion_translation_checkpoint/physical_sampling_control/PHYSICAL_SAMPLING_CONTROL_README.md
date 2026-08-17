# Physical-sampling control for V1 covariance-trace registration

## Question

Does the RF-space covariance-trace translation survive removal of structure that is
predictable from physical position along the V1 probe?

## Design

- Seven previously inspected sessions were retained without selecting new outcomes.
- Probe vertical position supplies a physical along-shank coordinate for every V1
  unit. Where 3D CCF is available, its first principal component correlates
  0.999-1.000 with probe vertical position.
- Raw covariance trace is decomposed into a leave-one-cell-out, cross-fitted
  along-shank prediction and a residual.
- Six nonoverlapping shank-position blocks are formed per session. Alternating
  blocks define two physically disjoint target halves containing 28-60 cells each.
- Translation is fit separately from raw trace, shank-predicted trace, and residual
  trace against the same leave-one-animal-out templates.
- A shank-preserving null permutes residual trace within physical blocks while
  retaining RF centers, block membership, and the shank-predicted component.

## Physical-half translation differences

| Session | Raw trace | Shank-predicted trace | Residual trace |
|---|---:|---:|---:|
| 742951821 | 0.0 deg | 2.0 deg | 20.0 deg |
| 762602078 | 69.3 deg | 4.0 deg | 10.0 deg |
| 715093703 | 34.5 deg | 62.1 deg | 19.8 deg |
| 760345702 | 2.0 deg | 2.0 deg | 21.6 deg |
| 719161530 | 10.2 deg | 6.0 deg | 30.0 deg |
| 759883607 | 4.5 deg | 6.0 deg | 22.6 deg |
| 835479236 | 80.7 deg | 2.8 deg | 51.6 deg |

The three clearest raw-trace successes (742951821, 760345702, and 759883607)
are also stable in the shank-predicted component and become substantially less
stable after residualization.

## Shank-preserving residual null

| Session | Real residual half difference | Null median | Null as or more stable |
|---|---:|---:|---:|
| 742951821 | 20.0 deg | 16.1 deg | 71% |
| 762602078 | 10.0 deg | 10.2 deg | 48% |
| 715093703 | 19.8 deg | 25.2 deg | 31% |
| 760345702 | 21.6 deg | 22.4 deg | 49% |
| 719161530 | 30.0 deg | 20.5 deg | 74% |
| 759883607 | 22.6 deg | 21.4 deg | 56% |
| 835479236 | 51.6 deg | 24.7 deg | 80% |

The real residual spatial organization fits the full-session residual template
better than all 100 nulls in every session. However, its translation does not show
better physical-half reproducibility than the null in a consistent way.

## Interpretation

The along-shank predictor explains only -0.03 to 0.14 of cross-fitted trace variance,
but its smooth component is sufficient to stabilize several translation optima.
Removing that component eliminates the strongest apparent raw-trace successes.

Therefore the current covariance-trace translation does not pass the physical-
sampling control and should not be used for registration. Residual covariance may
contain genuine spatial organization, but it does not identify a stable translation
across physically independent subsets.

