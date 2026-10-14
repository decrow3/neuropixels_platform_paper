# MouseV2 Figure 5

Figure 5 v17 (corrected tests) is the canonical main-text figure. Build it from
the repository root with:

```bash
python -m pip install -r MouseV2-Figure5/requirements.txt
python MouseV2-Figure5/Figure5.py
```

The locked build was verified with Python 3.12.4 and the package versions in
`requirements.txt`. The Allen repository's original `environment.yml` remains
unchanged for reproducing the published figures.

## Files

As in the original `Figure3/` and `Figure4/` directories, the figure script and
current exports live together. The `MouseV2-Figure5/` name distinguishes the
manuscript extension from the original Allen figures.

| File or directory | Purpose |
| --- | --- |
| `Figure5.py` | Build and validate the current figure. |
| `Figure5.png`, `Figure5.pdf`, `Figure5.svg` | Current figure exports. |
| `Figure5_legend.md`, `Figure5_*.csv`, `Figure5_*.csv.gz` | Caption and numerical output tables. |
| `render.py`, `panels.py`, `schematics.py`, `figure_data.py` | Figure-specific rendering helpers. |
| `data/` | Frozen analysis inputs used by the renderer. |
| `sources/` | Frozen anatomical metadata, cortical artwork, and caption source. |
| `prepare.py`, `source_manifest.json` | Regeneration pipeline and source checksums. |
| `dependencies.json`, `requirements.txt` | Production dependencies and pinned environment. |

The atlas and connectivity references remain under `data/reference/` at the
repository root. Running `Figure5.py` from either the repository root or this
folder produces the same outputs. The build does not overwrite this README.

## Regenerating the data inputs

The figure build intentionally consumes frozen analysis tables. Their compact
preparation pipeline is separate:

```bash
python MouseV2-Figure5/prepare.py check
python MouseV2-Figure5/prepare.py download-public
python MouseV2-Figure5/prepare.py regenerate
python MouseV2-Figure5/prepare.py verify
```

`check` verifies every upstream file against `source_manifest.json` before any
analysis runs. `download-public` obtains only missing Allen CCF and Harris
reference files. `regenerate` writes a parallel tree under
`MouseV2-Figure5/regenerated/`; it never overwrites the canonical inputs.
With 5,000 replicates it compares every regenerated table to the locked result,
allowing only floating-point roundoff attributable to the numerical backend.

The metric-level MouseV2 and Allen tables are the external data-release
boundary. They are not silently substituted or downloaded from mutable draft
locations: the exact file sizes and SHA-256 hashes are declared in
`source_manifest.json`. The anatomical entry-point table is retained as a
frozen source because the underlying MouseV2 DANDI draft changed after the
canonical analysis. For the same reason, `MouseV2-Figure5/sources/` preserves the
value-free unit IDs and recording assignments admitted by the original CCF
preflight. Response measurements are rebuilt from the separately hashed metric
tables; the frozen design snapshot contains no response values.

To verify the downstream statistics independently when the original external
source bundle is unavailable, start from the committed locked registry:

```bash
python MouseV2-Figure5/prepare.py regenerate --derived-only
```

The canonical exports and their supporting tables live in
`MouseV2-Figure5/`. The entry point performs a dependency
preflight and rejects changes to the locked numerical tables and bootstrap
content. The
`Figure5_layout_manifest.json` is marked `canonical: true` after a successful build.

## What is locked

- Panels A–F and the v17 layout.
- Bonferroni adjustment across the six panel-F component shuffle tests.
- The direct inter-HVA minus intra-V1 omega-squared contrasts.
- The exact numerical estimates and panel-F test table.
- The repository-local handmade cortical vector source.

## Dependency boundary

`dependencies.json` records the production code and data closure. Historical
Figure 5 variants, methods-companion reports, previews, caches, and tests are
not canonical inputs. The accepted visual changes from the historical v1–v17
development sequence are consolidated in `render.py`; those versioned
development modules are no longer part of the repository.
