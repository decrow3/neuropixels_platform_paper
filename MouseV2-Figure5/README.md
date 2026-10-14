# MouseV2 Figure 5

This folder contains the code and compact source tables needed to reproduce
Figure 5 of the unified-V2 manuscript. The original Allen figure folders are
unchanged. No exploratory scripts, private filesystem paths, or files from
another branch are needed for the default reproduction.

## First-time setup

Use Python 3.12 in a separate environment from the original Allen environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r MouseV2-Figure5/requirements.txt
```

The renderer uses the system **Liberation Sans** font (Liberation Fonts 2 for
the committed exports). Install it before rendering, for example with
`sudo apt install fonts-liberation2` on Debian/Ubuntu. Font versions can alter
text rasterization; numerical checks are independent of fonts.

Run these commands from the repository root. Scripts also work from inside
this folder with the folder prefix omitted.

## Reproduce the figure and results

For a quick build from the bundled, locked analysis tables:

```bash
python MouseV2-Figure5/Figure5.py
```

The current `Figure5.png`, `Figure5.pdf`, and `Figure5.svg` live beside
`Figure5.py`, following the original repository's figure-folder convention.
The build checks every numerical output against the canonical reference.

To reconstruct anatomical assignments, the eligible-unit registry, all
statistical results, and a new figure from the bundled metric-level sources:

```bash
python MouseV2-Figure5/reproduce.py
```

This runs 5,000 bootstrap/shuffle replicates with the original seeds. It is
substantially slower than the quick build. Progress is printed by analysis
stage. Results go to the ignored `work/reproduction/` directory:

- `anatomy/`: regenerated unit locations and probe entry points;
- `MouseV2-Figure5/data/`: regenerated registry and statistical tables;
- `figure/`: the figure rendered **from those regenerated tables**;
- `reproduction.json`: commands, runtime, and verification status.

The reference inputs and current exports are preserved. A failed comparison
stops the run. Once the statistical tables are complete, resume verification
and rendering with `python MouseV2-Figure5/reproduce.py --resume`. Resume
checks append to `resume_events.jsonl`, preserving the original
`reproduction.json`. If the original run stopped before writing that report,
resuming does not reconstruct its missing provenance.

## Recompute metrics from NWB recordings

Raw extraction is optional for the bundled-source reproduction above.
[MouseV2 DANDI 001568](https://dandiarchive.org/dandiset/001568/draft) is open:
**no DANDI API key, account, or additional permission is needed**. The eight
public NWBs total about 75 GB. The downloader uses the specific public asset
IDs and SHA-256 checksums in `raw_sources.json`, rather than whichever files
happen to be in a future draft. There was no numbered published version when
access was checked on October 14, 2026.

Create a separate **Python 3.10** environment and install
`requirements-extraction.txt`. This preserves the numerical-library versions
used for extraction. Download the public recordings anonymously, then extract:

```bash
python MouseV2-Figure5/extract.py download-mouse \
  --mouse-root /path/to/001568
python MouseV2-Figure5/extract.py download-allen \
  --allen-root /path/to/allen_sessions
python MouseV2-Figure5/extract.py run \
  --mouse-root /path/to/001568 \
  --allen-root /path/to/allen_sessions \
  --hash-raw
```

The downloader always verifies SHA-256 before accepting a file. Existing
files are checked and reused. `--sessions site2` supports a smaller download.
The original response-metric NWBs are also accepted for existing local caches.
The public files have updated metadata; the figure's anatomical assignments
continue to come from the bundled snapshots described in `METHODS.md`.

MouseV2 paths are `sub-*/...nwb`; Allen paths are
`session_<id>/session_<id>.nwb`, relative to the roots supplied above. An
optional `--path-map paths.json` maps site names (`site2`, etc.) or Allen
session IDs to existing NWBs. There are no hardcoded local roots.

Extraction recomputes preferred-polarity response-selected TTFS, harmonized
F1/F0, and the ten matched-flash MouseV2 timescale draws. It verifies the
resulting tables against the bundled source values, including unit membership.
`--hash-raw` verifies the matching public or original MouseV2 checksums and
records computed Allen checksums; without it, only file sizes are checked. The extraction manifest
records that distinction. `--cohorts`, `--sessions`, and `--metrics` support
smaller audits, which cannot substitute for a complete extraction.

After a complete extraction, return to the rendering environment and run:

```bash
python MouseV2-Figure5/reproduce.py \
  --metrics-dir MouseV2-Figure5/work/metrics \
  --output-dir MouseV2-Figure5/work/from-nwb
```

The original Allen timescale estimates and unit metadata are published-source
inputs, retained in a compact column subset of the original unit table. An
optional raw audit is available; it writes comparisons without replacing the
published fits:

```bash
python MouseV2-Figure5/extract.py audit-allen-timescales \
  --cohorts allen --sessions 798911424 \
  --allen-root /path/to/allen_sessions \
  --output-dir MouseV2-Figure5/work/allen-timescale-audit
```

See [METHODS.md](METHODS.md) for the estimands, source boundaries, anatomical
snapshot dates, and validation tolerances. See [VALIDATION.md](VALIDATION.md)
for the completed full reconstruction and raw-recording checks.

## Files

| File or directory | Purpose |
| --- | --- |
| `Figure5.py`, `render.py`, `panels.py`, `schematics.py`, `figure_data.py` | Current figure and rendering helpers. |
| `reproduce.py`, `prepare.py` | Complete bundled-source reproduction and statistical inference. |
| `extract.py`, `nwb_io.py`, `flash_metrics.py`, `gratings.py` | Raw NWB extraction and metric definitions. |
| `anatomy.py` | Unit-to-electrode assignments and cortical entry-point derivation. |
| `sources/metrics/` | Compact metric inputs and published Allen metadata/timescales. |
| `sources/anatomy/` | Historical scalar electrode tables and unit links. |
| `sources/reference/` | Allen CCF atlas and Harris connectivity/hierarchy references. |
| `sources/` | Derived anatomical design, cortical artwork, and caption source. |
| `data/` | Locked registry and statistical results for the quick build. |
| `Figure5_*.csv`, `Figure5_*.csv.gz`, `Figure5_legend.md` | Numerical figure outputs and caption. |
| `source_manifest.json`, `raw_sources.json`, `method_provenance.json` | Source checksums, raw recording inventory, and calculation provenance. |
| `dependencies.json`, `requirements*.txt`, `tests/` | Dependency closure, environments, and regression checks. |

To check source integrity or run the focused tests:

```bash
python MouseV2-Figure5/prepare.py check
python -m unittest discover -s MouseV2-Figure5/tests -v
```

The small synthetic NWB-reader test additionally requires h5py; run it in the
extraction environment with:

```bash
python -m unittest discover -s MouseV2-Figure5/tests -k nwb -v
```

The full 5,000-replicate reproduction is the integration check, rather than part
of the short test suite.
