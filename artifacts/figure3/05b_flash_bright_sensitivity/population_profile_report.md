# Figure 3 population profiles

Selected checkpoint profile: `common_qc`.

## Common QC

Rule applied to both datasets: `amplitude_cutoff < 0.1; presence_ratio > 0.8; ISI violations ratio < 0.5`.

- Allen: 43,496/99,180 across the full unit table; 21,673 in the eight Figure 3 regions.
- MouseV2: 11,242/20,374 units.
- MouseV2 per-site range: 1,176–1,711 units.
- MouseV2 per-probe pooled range: 2,611–3,146 units.

MouseV2 `common_qc` is verified to equal the NWB `default_qc` flag exactly.

## RF-filter status

`published_like` is computable for Allen but unavailable for MouseV2.
The provisional RF peak import has no significance p-value or area and uses
`per_unit_unsmoothed_argmax_spike_counts` with gaze correction `none`.
`firing_rate_dg` is available as preferred-condition mean spikes divided by
the validated 1.0-s analysis duration.
Pilot RF diagnostic QC retains 4,807 units, but it must not
be described as the published RF filter.

RF import schema version: 1.
