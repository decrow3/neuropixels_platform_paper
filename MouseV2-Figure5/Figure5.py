#!/usr/bin/env python3
"""Build and validate the canonical Figure 5."""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "MouseV2-Figure5"
INPUT_ROOT = ROOT
OUTPUT = CANONICAL
STEM = OUTPUT / "Figure5"

# These are the repository inputs opened by the renderer. Generated files in
# OUTPUT are intentionally omitted: they are checked separately after export.
REQUIRED_INPUTS = [
    "MouseV2-Figure5/sources/mousev2_ccf_surface_entry_points.csv",
    "MouseV2-Figure5/data/response_registry_floor5.csv.gz",
    "MouseV2-Figure5/data/bootstrap_effects.csv.gz",
    "MouseV2-Figure5/data/draw_specific_point_fits.csv",
    "MouseV2-Figure5/data/primary_effects.csv",
    "MouseV2-Figure5/data/matched_control.csv",
    "MouseV2-Figure5/data/display_counts.csv",
    "MouseV2-Figure5/sources/Figure5_caption.md",
    "MouseV2-Figure5/data/estimates_and_tests.csv",
    "MouseV2-Figure5/sources/Original_handmade_cortical_vectors.svg",
    "MouseV2-Figure5/sources/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt",
    "MouseV2-Figure5/sources/reference/allen_ccf_2017_surface/top.nrrd",
    "MouseV2-Figure5/sources/reference/allen_harris_2019_connectivity/harris_visualmedial_scores.csv",
    "MouseV2-Figure5/sources/reference/allen_harris_2019_connectivity/normalized_connection_density_ipsi_ctx.csv",
]

LOCKED_TABLES = {
    "Figure5_anatomical_connections.csv": "e705cc11da3185079bc35707c45dbc851b2033b74b1b77a25d2fd6a94f979204",
    "Figure5_display.csv": "cbe6f1c9a285b717f6bd083ec4f9a888cc952e863649ddb8fd26ae392f0b5757",
    "Figure5_estimates.csv": "319116a60cea3ecbddc8704caa00d3492c9ceb54aed4b55bc728632ff32e61b3",
    "Figure5_mean_spreads.csv": "53fd7ee701d3eacc9e22ae00408cf73935f15a0a7a1bd6326e0e8561ae756b5c",
    "Figure5_probe_positions.csv": "364725b7b60346c479a5f58ef66ce53a3d5e386f319a84bf1a6d1a3e2bccc902",
    "Figure5_F_estimates_and_tests.csv": "886050e765cb86cda1d734542460bb764c4d8cf4396ec5d565ced735d5333e9d",
}

LOCKED_GZIP_CONTENT = {
    "Figure5_spread_bootstrap.csv.gz": "7e87aa192d55522ba9ab6150211d779d2fcc52a041d49682638ef9186c737922",
}

EXPECTED_OUTPUTS = [
    STEM.with_suffix(".pdf"),
    STEM.with_suffix(".png"),
    STEM.with_suffix(".svg"),
    STEM.with_name(STEM.name + "_legend.md"),
    STEM.with_name(STEM.name + "_layout_manifest.json"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_path(relative):
    if relative.startswith("MouseV2-Figure5/data/"):
        return INPUT_ROOT / relative
    return ROOT / relative


def preflight() -> None:
    missing = [path for item in REQUIRED_INPUTS if not (path := input_path(item)).is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing canonical Figure 5 inputs:\n{formatted}")


def equivalent_table(name):
    # Regeneration may differ by floating-point roundoff, never by population,
    # table shape, labels, or meaningful numerical values.
    import pandas as pd
    reference = CANONICAL / name
    if name in LOCKED_TABLES:
        digest = sha256(reference)
        expected = LOCKED_TABLES[name]
    else:
        with gzip.open(reference, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        expected = LOCKED_GZIP_CONTENT[name]
    if digest != expected:
        raise RuntimeError(f'Canonical reference changed: {name}')
    pd.testing.assert_frame_equal(pd.read_csv(OUTPUT / name), pd.read_csv(CANONICAL / name),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)


def validate() -> None:
    missing = [path for path in EXPECTED_OUTPUTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Renderer did not create: {missing}")
    for name, expected in LOCKED_TABLES.items():
        actual = sha256(OUTPUT / name)
        if actual != expected:
            if INPUT_ROOT == ROOT:
                raise RuntimeError(f"Canonical numerical output changed: {name}\nexpected {expected}\nactual   {actual}")
            equivalent_table(name)
    for name, expected in LOCKED_GZIP_CONTENT.items():
        with gzip.open(OUTPUT / name, "rb") as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != expected:
            if INPUT_ROOT == ROOT:
                raise RuntimeError(f"Canonical bootstrap output changed: {name}\nexpected {expected}\nactual   {actual}")
            equivalent_table(name)

    manifest_path = STEM.with_name(STEM.name + "_layout_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        canonical=True,
        canonical_entrypoint="MouseV2-Figure5/Figure5.py",
        canonical_version="v17-corrected-tests",
        input_root=os.path.relpath(INPUT_ROOT, ROOT),
        validation="exact hashes" if INPUT_ROOT == ROOT else "numeric equivalence: rtol=1e-10, atol=1e-12",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    global INPUT_ROOT, OUTPUT, STEM, EXPECTED_OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT,
                        help='Root of a verified prepare.py regenerated tree')
    parser.add_argument('--output-dir', type=Path, default=CANONICAL)
    args = parser.parse_args()
    INPUT_ROOT = args.input_root.resolve()
    OUTPUT = args.output_dir.resolve()
    if INPUT_ROOT != ROOT:
        if OUTPUT == CANONICAL:
            parser.error('Use --output-dir to preserve the canonical reference exports')
        from prepare import verify_outputs
        if not verify_outputs(INPUT_ROOT, full=True):
            raise ValueError('Regenerated inputs failed verification')
    STEM = OUTPUT / 'Figure5'
    EXPECTED_OUTPUTS = [STEM.with_suffix('.' + suffix) for suffix in ('pdf', 'png', 'svg')]
    EXPECTED_OUTPUTS += [OUTPUT / ('Figure5_' + name) for name in ('legend.md', 'layout_manifest.json')]
    preflight()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1790989200")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuropixels-platform-paper-matplotlib")

    import matplotlib

    matplotlib.rcParams["svg.hashsalt"] = "figure5-canonical-v17"
    import render as renderer
    import panels
    import figure_data
    renderer.OUTPUT = OUTPUT
    renderer.STEM = STEM
    renderer.VARIANCE_STATS = input_path('MouseV2-Figure5/data/estimates_and_tests.csv')
    panels.REGISTRY = input_path('MouseV2-Figure5/data/response_registry_floor5.csv.gz')
    panels.DISPLAY_COUNTS = input_path('MouseV2-Figure5/data/display_counts.csv')
    figure_data.PRIMARY = figure_data.CONTROL = INPUT_ROOT / 'MouseV2-Figure5/data'

    renderer.main()
    validate()
    print(f"Canonical Figure 5: {STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
