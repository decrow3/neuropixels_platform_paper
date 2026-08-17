# Iteration 6A — known V1 locations without pseudo-hierarchy scores

## Outcome

The anatomical interpretation is corrected: the recordings are known to be in
V1 and their within-V1 locations are known experimentally. The missing value is
not V1 identity, but a numerical anatomical hierarchy score for those
locations.

The primary comparison already treats probe A/B/C/E as categorical groups, so
it does not require such a score. Iteration 6A removes the misleading visual
implication that the probes occupy known positions along the published
inter-area hierarchy axis.

## Figure change

In the reviewed `display_only` mode:

- MouseV2 session/probe markers use small symmetric offsets centered on the
  published VISp point solely to prevent overplotting;
- every relevant figure and caption labels those offsets as non-metric;
- the former line fitted through B/C/A/E pseudo-hierarchy positions is removed;
- Allen areas retain their published anatomical hierarchy coordinates and
  regression;
- measured RF azimuth/elevation remains a distinct two-dimensional companion
  view rather than being relabeled as anatomy or hierarchy.

The categorical split comparison and omega-squared analysis are unchanged.
The historical geometry remains available as `legacy_pseudo_hierarchy` only so
the frozen baseline can still be reproduced.

## Metadata audit

The recordings' known V1 localization is not represented by registered
coordinates in the eight NWBs currently used here. Every file has electrode
`location == "unknown"`, no CCF coordinate fields, and only probe-relative
spike localization (`estimated_x/y/z`). The exact anatomical localization
source should be versioned if a physical-coordinate panel is wanted, but that
is not required for the categorical variance comparison.

See [`WITHIN_V1_LOCATION_AUDIT.md`](../../data/imports/WITHIN_V1_LOCATION_AUDIT.md)
for the distinction among anatomical location, RF position, cortical depth,
display offsets, and hierarchy score, plus the optional import contract.

## Statistical result

Because this iteration changes representation rather than grouping, data, or
inference, the pooled-flash effect sizes are identical to Iteration 5:

| Metric | ω² probes | ω² post-V1 areas | Δω² areas−probes | 95% CI for Δω² |
| --- | ---: | ---: | ---: | ---: |
| TTFS | 0.160 | 0.125 | −0.036 | [−0.356, +0.112] |
| Modulation index | 0.171 | 0.151 | −0.019 | [−0.334, +0.147] |
| Timescale | 0.033 | 0.172 | +0.140 | [−0.230, +0.304] |

No interval excludes zero. The claim remains: within-V1 variation can be
comparable to between-area variation, with no resolved evidence that post-V1
area identity explains more. This iteration makes the display fairer but does
not strengthen the statistical evidence.

## Fairness assessment after this correction

For the core three-metric categorical comparison, lack of probe hierarchy
scores is no longer a fairness concern because no score is assigned or used.
The main remaining analysis issues are:

1. inference should preserve the four matched probes within each MouseV2
   session and the correlated areas within Allen sessions;
2. LP should be separated from the cortical post-V1 comparison;
3. the final RF-quality population remains blocked on a validated all-session
   RF significance/area export;
4. absolute cross-dataset TTFS still lacks physical-onset calibration, although
   that does not affect the within-MouseV2 categorical variance comparison.

## Review artifact and validation

- Checkpoint: [`06a_known_v1_locations`](06a_known_v1_locations/Figure3_stats.md)
- Location-aware probe view:
  [`Figure3_probe_zoom.png`](06a_known_v1_locations/Figure3_probe_zoom.png)
- Measured-retinotopy companion:
  [`Figure3_rf_position.png`](06a_known_v1_locations/Figure3_rf_position.png)

All checkpoint commands passed. The new display-position helper is tested for
symmetry around VISp and rejection of undeclared modes. The legacy/default run
remains pixel- and report-equivalent to the frozen baseline.

## Next executable step

Proceed to Iteration 7 locally: replace independent within-group resampling
with a session-blocked or hierarchical analysis that preserves matched probes
and correlated area measurements. Run cortical areas without LP as the primary
sensitivity, with LP-inclusive results reported separately.
