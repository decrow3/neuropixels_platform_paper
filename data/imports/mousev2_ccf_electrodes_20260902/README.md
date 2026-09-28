# MouseV2 CCF-electrode snapshot (2026-09-02)

This record identifies the revised protected `DANDI:001568/draft` snapshot used
for MouseV2 anatomy. The eight NWBs are stored outside the repository at the
path recorded in `snapshot_manifest.json`; the previous frozen snapshot was
retained rather than overwritten.

All 15,360 electrode rows have finite `x`, `y`, and `z`, and every `location`
entry contains an atlas label rather than `unknown`. The refreshed files retain
the previous unit counts, unit IDs, electrode-region vectors, and their ragged
indices exactly. The DANDI client checksum-validated every downloaded asset.

The manifest records DANDI asset IDs and SHA-256 values so this mutable draft
revision remains identifiable if the Dandiset changes again. Coordinate-axis
names and units should be taken from the SmartSPIM/CCF pipeline provenance
before assigning `x`, `y`, and `z` to named anatomical axes in a paper figure.
