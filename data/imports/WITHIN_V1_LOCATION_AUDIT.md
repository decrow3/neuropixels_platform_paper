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
| V1 anatomical subregion | Known experimentally; exact source record not yet versioned here | Physical part of V1 sampled | Methods description and, once imported, anatomical-location figure |
| Cortical depth/layer | Present in processed paper tables | Position along the cortical depth axis | Layer/depth sensitivity, not a surface hierarchy score |
| RF azimuth/elevation | Versioned provisionally for 32 session × probe groups | Position in visual space | Two-dimensional retinotopic companion analysis |
| NWB `estimated_x/y/z` | Present | Spike-waveform center-of-mass coordinates relative to the probe | Unit localization on the probe only |
| CCF/surface coordinates | Not encoded in the eight NWBs inspected | Anatomical coordinate in a registered brain/surface space | Requires a separate versioned localization export |
| Within-V1 hierarchy score | Not available | Hypothetical scalar extension of the inter-area hierarchy | Must not be inferred from plotting order or the response metrics under test |

## NWB metadata audit

All eight DANDI:001568 NWBs were inspected. In every session:

- `/general/extracellular_ephys/electrodes/location` contains `unknown`;
- the units table has no anterior–posterior, medial–lateral, dorsal–ventral, or
  CCF coordinate columns;
- `estimated_x`, `estimated_y`, and `estimated_z` are present but are
  spike-localization/probe-relative values, not registered anatomical
  coordinates;
- probe labels A, B, C, and E are present and map units completely.

Thus the known anatomical localization exists outside the machine-readable NWB
fields currently consumed by this repository. That absence does not invalidate
the categorical probe comparison, but it prevents this repository from making
an exact anatomical-coordinate panel without an explicit source export.

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

## Optional anatomical-coordinate import

If an exact anatomical surface view is desired, the existing localization
record should be exported with at least:

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

This repository should validate and snapshot that table. It should not convert
those coordinates into a hierarchy score unless an independent anatomical or
connectivity model supplies and validates that mapping.
