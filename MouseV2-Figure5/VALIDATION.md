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
replace the published fits. Public MouseV2 assets are now supported for repeating raw extraction; see
the public-access validation below and `METHODS.md`.

## Regression checks

All ten focused tests passed in the extraction environment. Nine passed in
the rendering environment, with the h5py-only NWB test skipped as documented.
These cover source integrity, anatomical reconstruction, registry membership,
original random-stream preservation, TTFS edge cases, single-trial F1/F0,
balanced flash sampling, NWB unit-ID mapping, and rejection of incomplete
reproduction outputs.

The original Allen figure code, helpers, environment specification, and
original README text were verified unchanged against `upstream/master`.

## Anonymous public MouseV2 access

On 2026-10-14, unauthenticated DANDI API requests reported Dandiset 001568 as
`OPEN`, with eight assets and no numbered published release. Anonymous HTTP
range requests successfully read the HDF5 signature from all eight asset
URLs. No API key, authorization header, account, or DANDI client was used.
The public asset IDs, sizes, SHA-256 hashes, and download URLs are now pinned
in `raw_sources.json`.

All eight public SHA-256 values match previously recorded anatomical-snapshot
hashes (September 25 for site6; September 2 for the other sites). Local copies
of these snapshots were compared against the original response-metric NWBs:
all scalar unit fields consumed by extraction, unit IDs, sorted spike times,
and both consumed stimulus tables were exactly equal for every session.
Full local-file SHA-256 was additionally rechecked for sites2 and3 during this
update; the other six relied on the recorded snapshot provenance and file
sizes for local identification. No 75-GB download or repeat statistical
bootstrap was needed for this input-equivalence check.

The updated suite passes all 12 tests in the extraction environment (11 plus
one documented h5py skip in the rendering environment). New tests cover
anonymous downloads, checksum validation, reuse of existing downloads,
acceptance of public and original extraction inputs, rejection of corruption,
and cleanup of partial downloads. Public downloads always verify the full
SHA-256 before accepting a file.
