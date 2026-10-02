#!/usr/bin/env python3
"""Build and validate the canonical Figure 5."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "artifacts/figure5/v17_corrected_tests"
STEM = OUTPUT / "Figure5_CDE_rerun_with_schematic"

# These are the repository inputs opened by the renderer. Generated files in
# OUTPUT are intentionally omitted: they are checked separately after export.
REQUIRED_INPUTS = [
    "artifacts/figure3/06s_mousev2_ccf_surface_projection/mousev2_ccf_surface_entry_points.csv",
    "artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz",
    "artifacts/figure4_rerun/v4_primary_dispersion/bootstrap_effects.csv.gz",
    "artifacts/figure4_rerun/v4_primary_dispersion/draw_specific_point_fits.csv",
    "artifacts/figure4_rerun/v4_primary_dispersion/primary_effects.csv",
    "artifacts/figure4_rerun/v6_matched_control/matched_control.csv",
    "artifacts/figure4_rerun/v8_final/figure4_display_counts.csv",
    "artifacts/figure5/captions/Figure5_caption.md",
    "artifacts/figure5/supplement_location_variance_v1/estimates_and_tests.csv",
    "artifacts/figure5/v17_corrected_tests/Original_handmade_cortical_vectors.svg",
    "data/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt",
    "data/reference/allen_ccf_2017_surface/top.nrrd",
    "data/reference/allen_harris_2019_connectivity/harris_visualmedial_scores.csv",
    "data/reference/allen_harris_2019_connectivity/normalized_connection_density_ipsi_ctx.csv",
]

LOCKED_TABLES = {
    "Figure5_CDE_rerun_with_schematic_anatomical_connections.csv": "e705cc11da3185079bc35707c45dbc851b2033b74b1b77a25d2fd6a94f979204",
    "Figure5_CDE_rerun_with_schematic_display.csv": "cbe6f1c9a285b717f6bd083ec4f9a888cc952e863649ddb8fd26ae392f0b5757",
    "Figure5_CDE_rerun_with_schematic_estimates.csv": "319116a60cea3ecbddc8704caa00d3492c9ceb54aed4b55bc728632ff32e61b3",
    "Figure5_CDE_rerun_with_schematic_mean_spreads.csv": "53fd7ee701d3eacc9e22ae00408cf73935f15a0a7a1bd6326e0e8561ae756b5c",
    "Figure5_CDE_rerun_with_schematic_probe_positions.csv": "364725b7b60346c479a5f58ef66ce53a3d5e386f319a84bf1a6d1a3e2bccc902",
    "Figure5_F_estimates_and_tests.csv": "886050e765cb86cda1d734542460bb764c4d8cf4396ec5d565ced735d5333e9d",
}

LOCKED_GZIP_CONTENT = {
    "Figure5_CDE_rerun_with_schematic_spread_bootstrap.csv.gz": "7e87aa192d55522ba9ab6150211d779d2fcc52a041d49682638ef9186c737922",
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


def preflight() -> None:
    missing = [path for item in REQUIRED_INPUTS if not (path := ROOT / item).is_file()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Missing canonical Figure 5 inputs:\n{formatted}")


def validate() -> None:
    missing = [path for path in EXPECTED_OUTPUTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Renderer did not create: {missing}")
    for name, expected in LOCKED_TABLES.items():
        actual = sha256(OUTPUT / name)
        if actual != expected:
            raise RuntimeError(f"Canonical numerical output changed: {name}\nexpected {expected}\nactual   {actual}")
    for name, expected in LOCKED_GZIP_CONTENT.items():
        with gzip.open(OUTPUT / name, "rb") as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Canonical bootstrap output changed: {name}\nexpected {expected}\nactual   {actual}")

    manifest_path = STEM.with_name(STEM.name + "_layout_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        canonical=True,
        canonical_entrypoint="Figure5/build.py",
        canonical_version="v17-corrected-tests",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Canonical Figure 5\n\n"
        "This directory contains the locked v17 corrected-tests exports and "
        "supporting tables. Rebuild and validate from the repository root with "
        "`python Figure5/build.py`.\n"
    )


def main() -> None:
    preflight()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1790989200")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuropixels-platform-paper-matplotlib")

    import matplotlib

    matplotlib.rcParams["svg.hashsalt"] = "figure5-canonical-v17"
    from Figure5 import render as renderer

    renderer.main()
    validate()
    print(f"Canonical Figure 5: {STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
