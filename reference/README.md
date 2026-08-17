# Exploratory PyNWB Analysis

This directory contains an interactive reference analysis for inspecting
MouseV2 NWB files. It is separate from the batch-processing and Figure 3
workflow documented in the [repository README](../README.md).

## What it does

`interactive_analysis.py` is organized as VS Code/Jupyter-style `# %%` cells.
It includes utilities for:

- discovering and opening NWB files with PyNWB;
- inspecting units and stimulus interval tables;
- estimating and plotting receptive fields;
- calculating orientation selectivity and grating tuning;
- fitting spatial- and temporal-frequency tuning curves; and
- exporting per-unit exploratory summaries and figures.

Outputs are written under `results/` relative to the directory from which the
script is run.

## Requirements

Use a modern Python environment with NumPy, pandas, SciPy, Matplotlib, and
PyNWB. PyNWB and HDMF versions must support the schema used by the input NWB
file.

## Usage

For interactive exploration, open `interactive_analysis.py` in an editor that
supports Python cells and run the cells in order.

The script also accepts an NWB file or directory from the command line:

```bash
python reference/interactive_analysis.py --nwb /path/to/session.nwb
```

Use `--show` to display plots interactively. Without it, Matplotlib uses a
non-interactive backend and saves or closes figures as directed by the cells.

## Assumptions and caveats

- The units table must contain ragged `spike_times` arrays.
- Receptive-field interval tables are expected to contain x/y stimulus
  positions; the default names are `x_position` and `y_position`.
- Grating tables must contain orientation and may contain temporal and spatial
  frequency.
- Stimulus table names and columns vary among NWB exports. Adjust the lookup
  helpers when working with a different schema.
- The source currently contains a developer-local default data directory.
  Passing `--nwb` overrides it and is the recommended approach.
- This analysis is exploratory and does not generate the paper-compatible
  TTFS, F1/F0, or response-timescale tables used by the current Figure 3
  scripts. Use `generate_retinotopic_csvs.py` for those outputs.

The project rationale and references live in
[problemstatement.md](../problemstatement.md) rather than being duplicated
here.
