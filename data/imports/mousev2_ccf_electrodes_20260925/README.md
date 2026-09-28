# MouseV2 CCF-electrode snapshot (2026-09-25)

This provenance record describes the DANDI:001568 draft snapshot used to rebuild
the MouseV2 anatomical panels after the corrected subject 815152 ephys NWB was
uploaded.

The local root is
`/media/huklaban5/Data/MouseV2/dandi_refresh_20260925/001568`. Subject 815152 is
the newly downloaded DANDI asset; the seven unchanged subject directories are
symbolic links to the verified 2026-09-02 snapshot. This avoids duplicating about
66 GB while presenting the figure scripts with one complete eight-session root.

Validation recorded in `snapshot_manifest.json` confirms:

- 8 ephys NWBs and 15,360 electrode rows;
- finite CCF `x`, `y`, and `z` on every electrode row;
- no `unknown` atlas locations;
- 3,913 VISp-labelled electrode rows;
- 20,374 unchanged unit IDs, with the unit-to-extremum-channel mapping preserved;
- a new subject 815152 SHA-256 digest and corrected probe-to-track association.

The correction changes subject 815152's anatomical geometry and VISp gate, so
the CCF tracks, surface projection, Figure 4 anatomical panel, location-control
statistics, and dependent legends were regenerated from this root.
