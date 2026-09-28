from pathlib import Path
import sys

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from project_mousev2_tracks_to_ccf_surface import (  # noqa: E402
    derive_entry_points,
    read_labels,
    read_nrrd,
)


ROOT = Path(__file__).resolve().parents[1]


def test_allen_top_surface_assets_have_expected_geometry_and_visual_areas():
    atlas = read_nrrd(ROOT / "data/reference/allen_ccf_2017_surface/top.nrrd")
    labels = read_labels(
        ROOT / "data/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt"
    )
    assert atlas.shape == (1140, 1320)  # ML x AP, 10-um CCF voxels
    present = {labels[index] for index in set(atlas.flat) if index in labels}
    assert {"VISp", "VISal", "VISam", "VISl", "VISpm", "VISrl"} <= present


def test_mousev2_entry_projection_inventory_and_independent_visp_agreement():
    atlas = read_nrrd(ROOT / "data/reference/allen_ccf_2017_surface/top.nrrd")
    labels = read_labels(
        ROOT / "data/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt"
    )
    points = pd.read_csv(
        ROOT / "artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/"
        "mousev2_probe_track_points.csv"
    )
    entries = derive_entry_points(points, atlas, labels)
    assert len(entries) == 32
    assert entries.subject_id.nunique() == 8
    assert entries.groupby("subject_id").probe.nunique().eq(4).all()
    assert int(entries.surface_agrees_visp.sum()) == 30
    assert set(entries.loc[~entries.surface_agrees_visp, "surface_atlas_acronym"]) == {
        "VISa", "VISpm"
    }
