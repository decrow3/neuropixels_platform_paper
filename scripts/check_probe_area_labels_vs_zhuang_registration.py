#!/usr/bin/env python3
"""Check whether each probe's own recorded Allen area label lands in the matching Zhuang
compartment, under the default registration (true px/mm scale + V1-anchor translation +
left-right mirror, no rotation/shear/optimizer -- `render_naive_map_over_zhuang_rough_bbox.py`).

For each (session, probe, map_area) the probe's median CCF position is transformed into
Zhuang pixel space and compared against the named-area compartments built the same way
`register_allen_session_to_zhuang.py::build_template()` does (seeded connected components of
the domain, `AREA_SEEDS_XY`). A probe "agrees" if its own map_area is also the nearest Zhuang
compartment at its registered position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import (  # noqa: E402
    AREA_LABELS, AREA_SEEDS_XY, build_template,
)

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"

REFLECT_ML = True
V1_SEED_XY_PX = (200, 240)
ZHUANG_FIG3_SCALE_BAR_PX = 62.0
ZHUANG_FIG3_SCALE_BAR_MM = 0.5
FIG3_TO_FIG9_SIMILARITY_SCALE = 0.8432313316638625

AREA_MARKERS = {"VISp": "o", "VISl": "s", "VISal": "^", "VISrl": "D", "VISam": "P", "VISpm": "X"}
AREA_COLORS = {"VISp": "#2864a8", "VISl": "#d78318", "VISal": "#b33f62", "VISrl": "#5f8f3e", "VISam": "#7356a8", "VISpm": "#999999"}


def main() -> None:
    px_per_mm = ZHUANG_FIG3_SCALE_BAR_PX / ZHUANG_FIG3_SCALE_BAR_MM * FIG3_TO_FIG9_SIMILARITY_SCALE
    template = build_template(ZHUANG_TEMPLATE)
    height, width = template["domain"].shape

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    v1_cells = cells.loc[cells.map_area.eq("VISp")]
    v1_ccf_ap = v1_cells.ccf_ap_mm.median()
    v1_ccf_ml = v1_cells.ccf_ml_mm.median()
    v1_seed_col, v1_seed_row = V1_SEED_XY_PX

    def ccf_to_pixel(ap: np.ndarray, ml: np.ndarray) -> np.ndarray:
        row = v1_seed_row + (ap - v1_ccf_ap) * px_per_mm
        ml_sign = -1.0 if REFLECT_ML else 1.0
        col = v1_seed_col + (ml - v1_ccf_ml) * px_per_mm / ml_sign
        return np.column_stack([row, col])

    probes = (
        cells.groupby(["ecephys_session_id", "ecephys_probe_id", "map_area"], as_index=False)
        .agg(ccf_ap_mm=("ccf_ap_mm", "median"), ccf_ml_mm=("ccf_ml_mm", "median"), cells=("ccf_ap_mm", "size"))
    )
    row_col = ccf_to_pixel(probes.ccf_ap_mm.to_numpy(float), probes.ccf_ml_mm.to_numpy(float))
    probes["pixel_row"] = row_col[:, 0]
    probes["pixel_col"] = row_col[:, 1]
    valid_px = (probes.pixel_row >= 0) & (probes.pixel_row < height) & (probes.pixel_col >= 0) & (probes.pixel_col < width)

    known_areas = list(AREA_SEEDS_XY)
    for area in known_areas:
        clipped = np.clip(row_col, [0, 0], [height - 1, width - 1])
        probes[f"dist_{area}_px"] = template["area_distance"][area](clipped)
    dist_cols = [f"dist_{a}_px" for a in known_areas]
    probes["nearest_zhuang_area"] = np.array(known_areas)[probes[dist_cols].to_numpy().argmin(axis=1)]
    probes["own_area_distance_px"] = probes.apply(
        lambda r: r[f"dist_{r.map_area}_px"] if r.map_area in known_areas else np.nan, axis=1
    )
    probes["agrees"] = probes.own_area_distance_px <= 1.5
    probes["has_zhuang_reference"] = probes.map_area.isin(known_areas)

    checked = probes.loc[probes.has_zhuang_reference & valid_px]
    print(f"probes checked: {len(checked)} (excluded {len(probes) - len(checked)}: no Zhuang reference area or out of pixel bounds)")
    print(f"agreement (own area distance <= 1.5px): {checked.agrees.mean():.1%}")
    print(checked.groupby("map_area").agrees.agg(["size", "mean"]))
    mismatches = checked.loc[~checked.agrees, ["ecephys_session_id", "ecephys_probe_id", "map_area", "nearest_zhuang_area", "own_area_distance_px", "cells"]]
    print("\nmismatches (own area's compartment is >1.5px away):")
    print(mismatches.sort_values("own_area_distance_px", ascending=False).to_string(index=False))

    probes.to_csv(OUTPUT / "probe_area_label_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 8.5))
    boundary = template["boundary"].astype(float)
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    for acronym, (x, y) in AREA_SEEDS_XY.items():
        ax.text(x, y, AREA_LABELS[acronym], ha="center", va="center", fontsize=9, color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
    for area, group in probes.groupby("map_area"):
        ax.scatter(group.pixel_col, group.pixel_row, marker=AREA_MARKERS.get(area, "x"),
                   color=AREA_COLORS.get(area, "black"), s=60, edgecolors="white", linewidths=0.6,
                   label=f"{area} (n={len(group)})")
    ax.set(title="Probe positions (own map_area label) over Zhuang, default registration",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_probe_area_label_check.png", dpi=180)
    plt.close(fig)
    print(OUTPUT / "Figure_probe_area_label_check.png")


if __name__ == "__main__":
    main()
