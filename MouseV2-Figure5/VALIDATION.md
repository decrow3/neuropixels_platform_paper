# Reproduction validation

Validated on 2026-10-14 before publishing this consolidation.

## Full statistical reconstruction

`python MouseV2-Figure5/reproduce.py --output-dir <empty-directory>` completed
with all 5,000 bootstrap/shuffle replicates and the original random seeds.
Both anatomical products matched their frozen references. All seven analysis
tables matched: display counts byte-for-byte, and the other six numerically
at `rtol=1e-10`, `atol=1e-12`. The figure rendered from those regenerated tables
and passed its numerical checks, including all 27 display counts.

The rendering environment used Python 3.12 and the versions in
`requirements.txt`. The quick build also passed with only this figure folder
copied into an otherwise empty directory. With the original font selection,
its PNG was pixel-identical to the committed figure. Different installed
Liberation Sans versions can change text rasterization.

## Raw recording checks

The extraction environment used Python 3.10 and the versions in
`requirements-extraction.txt`. All eight MouseV2 sessions and all 58 Allen
sessions were processed in separate runs, then combined and passed through
the complete-extraction verification gate:

| Extracted table | Verified rows |
| --- | ---: |
| MouseV2 response-selected TTFS | 1,413 |
| MouseV2 common-support F1/F0 | 20,374 |
| MouseV2 matched-flash timescale draws | 112,420 |
| Allen TTFS | 20,340 |
| Allen harmonized F1/F0 | 9,342 |

All matched the bundled metric sources at `rtol=1e-7`, `atol=1e-8`, including
population membership and validity flags. Rebuilding the registry from these
extracted metrics and anatomical metadata reproduced the 35,409-row canonical
registry at `rtol=1e-10`, `atol=1e-12`. These raw runs checked NWB file sizes;
they did not rehash every full NWB. Use `--hash-raw` for that additional check.

Published Allen timescales remain source values. An optional raw audit of
session 798911424 covered 547 units: spike counts matched, while nonlinear
fit estimates were not uniformly identical (median absolute tau difference
approximately 2.0e-7 ms; maximum approximately 844 ms). This audit does not
replace the published fits. Historical MouseV2 draft availability remains
an external prerequisite for repeating raw extraction; see `METHODS.md`.

## Regression checks

All ten focused tests passed in the extraction environment. Nine passed in
the rendering environment, with the h5py-only NWB test skipped as documented.
These cover source integrity, anatomical reconstruction, registry membership,
original random-stream preservation, TTFS edge cases, single-trial F1/F0,
balanced flash sampling, NWB unit-ID mapping, and rejection of incomplete
reproduction outputs.

The original Allen figure code, helpers, environment specification, and
original README text were verified unchanged against `upstream/master`.
