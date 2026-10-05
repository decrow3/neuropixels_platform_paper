# Neuropixels Platform Paper — Unified-V2 / OpenScope Figure 5 extension

This repository is a fork of the Allen Institute's figure code for
Siegle et al. (2021), *Survey of spiking in the mouse visual system reveals
functional hierarchy*. The original figure scripts (`Figure1/`–`Figure4/`,
`ExtDataFigure*/`) are kept as archival reproduction code.

## Purpose of this branch

This branch (`codex/minimal-figure5`) contains the minimal, reproducible build
of **Figure 5** for the unified-V2 manuscript. Figure 5 asks whether
hierarchy-associated response properties differ more **among higher visual
areas (HVAs)** than **among locations within V1**:

- time to first spike (TTFS),
- phase modulation (log10 F1/F0), and
- response timescale.

Everything needed to rebuild and check the figure is in `Figure5/`:

| Command | What it does |
| --- | --- |
| `python Figure5/build.py` | Renders the figure from frozen, hash-locked tables and rejects any numerical change. |
| `python Figure5/prepare.py check` / `regenerate` / `verify` | Rebuilds those tables from hash-locked metric sources. |

See [Figure5/README.md](Figure5/README.md) for the figure contract. Other
analyses in this repository (Figure 3 overlays, cross-dataset bridge
diagnostics, RF and frequency-preference surfaces) are exploratory. They are
not inputs to Figure 5 unless they are listed in
[Figure5/dependencies.json](Figure5/dependencies.json).

## Datasets and the role of "MouseV2"

| Name in code | Name in manuscript | Source | Role in Figure 5 |
| --- | --- | --- | --- |
| **MouseV2** | OpenScope electrophysiology dataset | Allen Institute OpenScope project (DANDI 001568); 8 mice, each recorded simultaneously with four Neuropixels probes (A, B, C, E) at separated V1 locations | **Within-V1 benchmark.** Probes are grouped as anterior (A), lateral (E), posterior (C) and medial (B) V1 (panels B, D, E, F). |
| **Allen** (`data/unit_table.csv`) | Allen Brain Observatory Visual Coding–Neuropixels | Siegle et al. (2021) | **Across-HVA comparison** for LM, RL, AL, PM and AM (panels A, C, E, F). Allen V1 from the same sessions is the **positive control** for the V1-vs-HVA mean difference (panel E, left). |

"MouseV2" is the internal project name and appears throughout the code, file
paths and tables. It refers only to the dataset. It is **not** a claim that
these recordings come from area V2: every MouseV2 unit used in Figure 5 lies in
V1 (VISp). The unified-V2 hypothesis concerns how the Allen HVAs are
interpreted; the MouseV2 recordings supply the within-area comparison for
judging it.

Which MouseV2 units count as V1, and where each probe enters the cortex, are
frozen snapshots (`Figure5/sources/`,
`artifacts/figure3/06s_mousev2_ccf_surface_projection/`), because the
underlying DANDI draft changed after the analysis.

# Neuropixels Platform Paper

<img src="icon.png" width="400">

This repository contains code for generating the figures in the pre-print describing the scientific findings from the Allen Institute's [Neuropixels data collection platform](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels).

Most of the figures can be generated from publicly available NWB files and unit tables, available via the AllenSDK. See [the AllenSDK documentation](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) for an overview of the dataset and example notebooks for accessing it.

Each figure in the pre-print has a folder with associated code. Data required for figure generation that's not available in the October 2019 release (e.g., from the change detection experiments) is included in the `data` directory.

This repository is still a work in progress; we will continue to clean up the code and add documentation throughout the review process.


## Level of Support

We are not currently supporting this code, but simply releasing it to the community AS IS.  We are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.


## Terms of Use

See [Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-use/)


© 2019 Allen Institute for Brain Science