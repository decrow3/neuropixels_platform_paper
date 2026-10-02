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
