# Selected-case V1 dispersion drill-down

## What the current dispersion means

The registration statistic is computed in observed RF space before translation.
For every cell, neighboring RF centers from the same session receive 15-degree
Gaussian weights. Their covariance supplies:

- log2 covariance trace;
- normalized azimuth-versus-elevation anisotropy;
- normalized cross-covariance.

CCF coordinates do not enter this calculation. A candidate translation changes
where these fixed descriptors are sampled against the leave-one-animal-out
template; it does not change the descriptors themselves.

Physical sampling can nevertheless influence the RF point cloud indirectly. The
post hoc CCF diagnostic therefore asks whether physical cortical proximity predicts
RF-center proximity or the local RF-space covariance statistic.

## Selected-case findings

- The leave-one-out absolute-size templates retain 0.51-0.76 log2 units of range
  across the corrected cells. Template washout is therefore not sufficient to
  explain the poor RF-size split-half registration.
- Covariance trace is more credible than anisotropy. In the best case its two half
  translations differ by 6 degrees and remain away from the translation bounds.
  Anisotropy repeatedly selects boundary solutions.
- Exact-support shuffles keep every RF center and the descriptor distribution but
  permute descriptors across locations. Real full-session dispersion loss is lower
  than all 100 shuffles in each selected case.
- Split-half reproducibility is more nuanced: 8% of shuffles are as reproducible as
  the 0-degree best-case split difference, none are as reproducible as the 19-degree
  median-case difference, and 33% are as reproducible as the 56-degree failure-case
  difference.
- Unit-level CCF coordinates are unavailable for the best case. In the other two
  cases, RF-space covariance trace correlates about 0.30-0.32 with a 250-um
  CCF-neighborhood RF covariance trace. Pairwise physical-versus-RF distance
  correlations are only 0.05 and 0.13.
- AP, ML, and DV associations cannot be separated within these single-probe V1
  samples because the CCF coordinates are nearly collinear along the probe shank.

The evidence supports retaining covariance trace as an exploratory registration
feature. It does not yet establish a physical-sampling-independent translation.

## Outputs

- `Figure_selected_v1_dispersion_drilldown.png`
- `component_translation_optima.csv`
- `support_matched_descriptor_shuffle.csv`
- `absolute_size_template_gradient_summary.csv`
- `ccf_sampling_diagnostics.csv`
- `sampled_pairwise_ccf_rf_distances.csv`
- `run_manifest.json`

