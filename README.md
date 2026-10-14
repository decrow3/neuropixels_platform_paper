# Neuropixels Platform Paper — Unified-V2 / OpenScope Figure 5 extension

This repository is a fork of the Allen Institute's figure code for
Siegle et al. (2021), *Survey of spiking in the mouse visual system reveals
functional hierarchy*. The original figure scripts (`Figure1/`–`Figure4/`,
`ExtDataFigure*/`) are kept as archival reproduction code.

## Purpose of this branch

This branch (`mousev2-figure5`) contains the minimal, reproducible build
of **Figure 5** for the unified-V2 manuscript. Figure 5 asks whether
hierarchy-associated response properties differ more **among higher visual
areas (HVAs)** than **among locations within V1**:

- time to first spike (TTFS),
- phase modulation (log10 F1/F0), and
- response timescale.

The figure code, frozen analysis tables, and current exports live in
`MouseV2-Figure5/`, following the original one-folder-per-figure layout.
All runtime code, compact metric inputs, anatomical snapshots, and atlas/connectivity
references are included in that folder.

| Command | What it does |
| --- | --- |
| `python MouseV2-Figure5/Figure5.py` | Renders the figure from frozen, hash-locked tables and rejects any numerical change. |
| `python MouseV2-Figure5/reproduce.py` | Rebuilds anatomy, the registry, statistical results, and the figure from bundled sources. |
| `python MouseV2-Figure5/extract.py run --help` | Shows how to recompute response metrics from the historical NWB recordings. |

View the current [Figure 5](MouseV2-Figure5/Figure5.png) or download the
[PDF](MouseV2-Figure5/Figure5.pdf). See
[MouseV2-Figure5/README.md](MouseV2-Figure5/README.md) for the build instructions
and historical NWB-source requirements, and
[MouseV2-Figure5/dependencies.json](MouseV2-Figure5/dependencies.json) for the
production dependencies. Exploratory analyses are retained on the research
branch; they are not part of this figure build.

## Datasets and the role of "MouseV2"

| Name in code | Name in manuscript | Source | Role in Figure 5 |
| --- | --- | --- | --- |
| **MouseV2** | OpenScope electrophysiology dataset | Allen Institute OpenScope project (DANDI 001568); 8 mice, each recorded simultaneously with four Neuropixels probes (A, B, C, E) at separated V1 locations | **Within-V1 benchmark.** Probes are grouped as anterior (A), lateral (E), posterior (C) and medial (B) V1 (panels B, D, E, F). |
| **Allen** (compact source: `MouseV2-Figure5/sources/metrics/allen_units.csv.gz`) | Allen Brain Observatory Visual Coding–Neuropixels | Siegle et al. (2021) | **Across-HVA comparison** for LM, RL, AL, PM and AM (panels A, C, E, F). Allen V1 from the same sessions is the **positive control** for the V1-vs-HVA mean difference (panel E, left). |

"MouseV2" is the internal project name and appears throughout the code, file
paths and tables. It refers only to the dataset. It is **not** a claim that
these recordings come from area V2: every MouseV2 unit used in Figure 5 lies in
V1 (VISp). The unified-V2 hypothesis concerns how the Allen HVAs are
interpreted; the MouseV2 recordings supply the within-area comparison for
judging it.

Which MouseV2 units count as V1, and where each probe enters the cortex, are
derived from the historical scalar snapshots in `MouseV2-Figure5/sources/anatomy/`,
because the underlying DANDI draft changed after the analysis. The derivation
code and both anatomical snapshot dates are documented in
[MouseV2-Figure5/METHODS.md](MouseV2-Figure5/METHODS.md).

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