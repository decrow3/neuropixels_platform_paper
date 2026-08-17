# Full covariance tensor with nested covariance bootstrap

## Question

Can the directional components of anatomy-corrected local RF covariance resolve
the trace-only annulus seen for session `798911424` while preserving the compact
localization of `760345702`?

The covariance tensor was represented without angular wraparound as:

- `log2(trace)`;
- `(Caz,az - Cel,el) / trace`;
- `2 Caz,el / trace`.

The last two values jointly encode normalized ellipse anisotropy and orientation.

## Bootstrap estimand

Each of 100 repeats sampled cells with replacement within six physical probe
blocks. Local covariance was then recomputed for the held-out session and every
session contributing to its population template. This preserves the broad
physical sampling distribution while propagating cell-level uncertainty through
both sides of the covariance match.

The bootstrap remains conditional on the previously fitted nested leave-one-animal-
out CCF-to-RF mean maps. It does not refit those anatomical mean maps.

## Results

| Session | Component | Full optimum (az, el) | Bootstrap median distance | Bootstrap p90 distance | Boundary fraction |
|---|---|---:|---:|---:|---:|
| 760345702 | trace | (+10, 0) deg | 5.8 deg | 14.2 deg | 0% |
| 760345702 | anisotropy | (-14, +30) deg | 28.0 deg | 70.0 deg | 90% |
| 760345702 | full tensor | (+4, -4) deg | 12.7 deg | 22.1 deg | 1% |
| 798911424 | trace | (-28, -12) deg | 41.2 deg | 64.6 deg | 48% |
| 798911424 | anisotropy | (-10, +30) deg | 8.0 deg | 20.0 deg | 100% |
| 798911424 | full tensor | (-8, +30) deg | 8.0 deg | 60.0 deg | 89% |

The apparently smaller median displacement for anisotropy in `798911424` is not
successful localization: every anisotropy-only optimum lies on the search boundary.
It reflects selection along the upper boundary rather than a closed interior basin.

Full covariance therefore does **not** resolve the trace annulus. For
`798911424`, the combined tensor remains boundary-dominated and can move about
60 deg across bootstrap repeats. For `760345702`, trace remains the most stable
descriptor; adding anisotropy increases p90 uncertainty from 14.2 to 22.1 deg and
moves the full-data optimum.

The nested result also qualifies the earlier conditional bootstrap. Once local
covariance and the population template are re-estimated, trace uncertainty for
`760345702` rises from roughly 4.5 deg to 14.2 deg. The compact trace basin is real
relative to the annular case, but its practical uncertainty is larger than the
fixed-descriptor analysis suggested.

## Integrity checks

- The diagonal tensor components reconstruct residual trace with maximum numerical
  error 0 in the audited population.
- Maximum normalized anisotropy was 0.911; no value exceeded the positive-
  semidefinite bound of 1.
- The analysis uses a 4-deg population-surface grid and a 2-deg translation grid.
  This accounts for the 2--4 deg shift relative to the earlier fine-grid trace
  optima but cannot explain 20--70 deg instability.

## Interpretation and next checkpoint

The current evidence supports treating covariance trace as an exploratory
localization cue with session-specific identifiability. Raw anisotropy should not
be added to the registration objective yet.

The smallest useful next test is to measure the repeatability of the two
anisotropy components themselves across independent cell/physical splits in these
same sessions. If they are intrinsically unreliable, the tensor should be reduced
toward an isotropic covariance using reliability-calibrated shrinkage. If they are
repeatable but disagree across animals, the failure lies in the population tensor
template rather than within-session estimation.

## Outputs

- `Figure_v1_covariance_tensor_nested_bootstrap.png`
- `tensor_nested_bootstrap_metrics.csv`
- `tensor_full_data_optima.csv`
- `tensor_nested_bootstrap_optima.csv.gz`
- `tensor_full_data_landscapes.csv.gz`
- `run_manifest.json`
- Analysis script: `scripts/test_v1_covariance_tensor_nested_bootstrap.py`
