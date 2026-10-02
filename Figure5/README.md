# Canonical Figure 5

Figure 5 v17 (corrected tests) is the canonical main-text figure. Build it from
the repository root with:

```bash
python -m pip install -r Figure5/requirements.txt
python Figure5/build.py
```

The locked build was verified with Python 3.12.4 and the package versions in
`requirements.txt`. The Allen repository's original `environment.yml` remains
unchanged for reproducing the published figures.

## Regenerating the data inputs

The figure build intentionally consumes frozen analysis tables. Their compact
preparation pipeline is separate:

```bash
python Figure5/prepare.py check
python Figure5/prepare.py download-public
python Figure5/prepare.py regenerate
python Figure5/prepare.py verify
```

`check` verifies every upstream file against `source_manifest.json` before any
analysis runs. `download-public` obtains only missing Allen CCF and Harris
reference files. `regenerate` writes a parallel tree under
`artifacts/figure5_regenerated/`; it never overwrites the canonical inputs.
With 5,000 replicates it compares every regenerated table to the locked result,
allowing only floating-point roundoff attributable to the numerical backend.

The metric-level MouseV2 and Allen tables are the external data-release
boundary. They are not silently substituted or downloaded from mutable draft
locations: the exact file sizes and SHA-256 hashes are declared in
`source_manifest.json`. The anatomical entry-point table is retained as a
frozen source because the underlying MouseV2 DANDI draft changed after the
canonical analysis. For the same reason, `Figure5/sources/` preserves the
value-free unit IDs and recording assignments admitted by the original CCF
preflight. Response measurements are rebuilt from the separately hashed metric
tables; the frozen design snapshot contains no response values.

To verify the downstream statistics independently when the original external
source bundle is unavailable, start from the committed locked registry:

```bash
python Figure5/prepare.py regenerate --derived-only
```

The canonical exports and their supporting tables live in
`artifacts/figure5/v17_corrected_tests/`. The entry point performs a dependency
preflight and rejects changes to the locked numerical tables and bootstrap
content. The
layout manifest is marked `canonical: true` after a successful build.

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
development sequence are consolidated in `Figure5/render.py`; those versioned
development modules are no longer part of the repository.
