# Within-V1 location audit

## Established experimental fact

The MouseV2 recordings are known to be in V1, and the sampled part of V1 is
known from the experimental localization. V1 identity is therefore not an
outstanding anatomical validation problem for the present comparison.

The missing quantity is different: the four within-V1 recording locations do
not have numerical anatomical hierarchy scores comparable to the published
scores assigned to LGN, V1, LM, RL, LP, AL, PM, and AM.

## Coordinate types must remain distinct

| Quantity | Available? | Meaning | Valid use |
| --- | --- | --- | --- |
| Probe identity (A/B/C/E) | Yes, all units and sessions | Known recording location/category within V1 | Primary categorical within-V1 grouping |
| V1 anatomical subregion | Present per electrode in the refreshed NWBs | Physical part of V1 sampled | Methods and anatomical-location figures |
| Cortical depth/layer | Present in processed paper tables | Position along the cortical depth axis | Layer/depth sensitivity, not a surface hierarchy score |
| RF azimuth/elevation | Versioned provisionally for 32 session × probe groups | Position in visual space | Two-dimensional retinotopic companion analysis |
| NWB `estimated_x/y/z` | Present | Spike-waveform center-of-mass coordinates relative to the probe | Unit localization on the probe only |
| CCF/surface coordinates | Present per electrode as `x`, `y`, and `z` in the refreshed NWBs | Anatomical coordinate in registered CCF space | Anatomical position and probe-track analyses |
| Within-V1 hierarchy score | Not available | Hypothetical scalar extension of the inter-area hierarchy | Must not be inferred from plotting order or the response metrics under test |

## NWB metadata audit

The original local DANDI:001568 draft snapshot was inspected in August 2026. In
every session:

- `/general/extracellular_ephys/electrodes/location` contains `unknown`;
- the units table has no anterior–posterior, medial–lateral, dorsal–ventral, or
  CCF coordinate columns;
- `estimated_x`, `estimated_y`, and `estimated_z` are present but are
  spike-localization/probe-relative values, not registered anatomical
  coordinates;
- probe labels A, B, C, and E are present and map units completely.

The Dandiset draft was subsequently revised. The current local snapshot assembled
2026-09-25 is stored at
`/media/huklaban5/Data/MouseV2/dandi_refresh_20260925/001568`; it combines the
corrected subject 815152 asset with the seven unchanged assets from the 2026-09-02
snapshot. The earlier frozen snapshot remains at `/media/huklaban5/Data/MouseV2/001568`.

All eight refreshed NWBs were inspected. In every session:

- the electrodes table has `x`, `y`, and `z` columns;
- all 1,920 rows per session (15,360 total) have finite values for all three
  coordinates;
- `location` has atlas structure labels with no remaining `unknown` values;
- 3,913 electrode rows across the cohort are labeled in VISp layers;
- unit counts, `/units/id`, `/units/electrodes`, and
  `/units/electrodes_index` exactly match the earlier snapshot.

The coordinate update therefore supplies the missing machine-readable anatomy
without changing unit identity or the unit-to-electrode mapping. For subject
815152, the 2026-09-25 correction changes the CCF/atlas association among probes
B, C, and E while retaining the same unit IDs and extremum-channel assignments.

## Paper-facing decision

The primary variance comparison treats A/B/C/E as four categorical locations
within known V1. It does not require a hierarchy score for those locations.

For plots that also show the published inter-area hierarchy:

- MouseV2 values use small, symmetric horizontal offsets centered on VISp only
  to avoid overplotting;
- those offsets are labeled non-metric and are never used in inference;
- no regression is fitted through the four probe positions;
- measured RF azimuth/elevation remains a separate two-dimensional view.

The historical geometry is retained only as `legacy_pseudo_hierarchy` for
regression reproduction.

## Anatomical-coordinate use

New anatomical analyses should use the refreshed NWBs and preserve the DANDI
asset revision, coordinate convention, and atlas-label provenance. A compact
paper-facing export should include at least:

```text
subject_id
probe
v1_subregion
anterior_posterior_coordinate
medial_lateral_coordinate
coordinate_space
registration_or_targeting_method
source_record
uncertainty_or_resolution
```

The CCF coordinates should not be converted into a hierarchy score unless an
independent anatomical or connectivity model supplies and validates that
mapping.
