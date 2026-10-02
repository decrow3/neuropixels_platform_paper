# Neuropixels Platform Paper

<img src="icon.png" width="400">

This repository contains code for generating the figures in the pre-print describing the scientific findings from the Allen Institute's [Neuropixels data collection platform](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels).

Most of the figures can be generated from publicly available NWB files and unit tables, available via the AllenSDK. See [the AllenSDK documentation](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) for an overview of the dataset and example notebooks for accessing it.

Each figure in the pre-print has a folder with associated code. Data required for figure generation that's not available in the October 2019 release (e.g., from the change detection experiments) is included in the `data` directory.

This repository is still a work in progress; we will continue to clean up the code and add documentation throughout the review process.

## New V1 benchmark and Figure 5

The `Figure5` directory is a minimal extension for comparing functional
variation across higher visual areas with variation across spatially separated
recording locations within V1. The locked publication figure, its exact input
tables, and a dependency manifest are included.

Rebuild and validate the canonical figure from the repository root with:

```bash
python Figure5/build.py
```

See [`Figure5/README.md`](Figure5/README.md) for the figure contract and
[`Figure5/dependencies.json`](Figure5/dependencies.json) for the complete list
of added production dependencies.


## Level of Support

We are not currently supporting this code, but simply releasing it to the community AS IS.  We are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.


## Terms of Use

See [Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-use/)


© 2019 Allen Institute for Brain Science
