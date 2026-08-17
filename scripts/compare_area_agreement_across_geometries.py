#!/usr/bin/env python3
"""What happens to LM (VISl) / AL (VISal) compartment agreement under the RF-vector-error-
optimal geometry found by `verify_naive_vs_global_restricted_affine_cv.py` (net rotation ~+0.45
deg, i.e. almost exactly Zhuang's native orientation -- it nearly cancels the historically-
locked -8.1 deg), compared against (a) no rotation at all and (b) the currently locked -8.1 deg
default that was originally introduced BECAUSE it fixed a LM/AL confusion problem
(`check_probe_area_labels_vs_zhuang_registration.py`: no-rotation gave LM 21.4%, AL 20.0%;
-8.1 deg improved these to LM 61.5%, AL 39.3%, per `fit_translation_rotation_naive_to_zhuang.py`).

Same probe-level area-membership methodology throughout: each probe's median CCF position is
transformed into Zhuang pixel space under a candidate geometry and checked against the named-
area compartments from `register_allen_session_to_zhuang.py::build_template()`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import AREA_LABELS, AREA_SEEDS_XY, build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
CV_OPTIMAL = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset/global_restricted_affine_full_fit.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"

V1_SEED_XY_PX = (200, 240)
AREA_MARKERS = {"VISp": "o", "VISl": "s", "VISal": "^", "VISrl": "D", "VISam": "P", "VISpm": "X"}
AREA_COLORS = {"VISp": "#2864a8", "VISl": "#d78318", "VISal": "#b33f62", "VISrl": "#5f8f3e", "VISam": "#7356a8", "VISpm": "#999999"}


def make_transform(rotation_deg: float, scale_px_per_mm: float, tx: float, ty: float, v1_anchor_ap: float, v1_anchor_ml: float):
    theta = np.radians(rotation_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale_reflect = np.diag([-1.0 * scale_px_per_mm, scale_px_per_mm])  # ml_sign=-1 (mirror), locked throughout
    matrix = rotation @ scale_reflect
    v1_seed_col, v1_seed_row = V1_SEED_XY_PX
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
    v1_anchor = np.array([v1_anchor_ap, v1_anchor_ml])

    def ccf_to_pixel(ap: np.ndarray, ml: np.ndarray) -> np.ndarray:
        ccf = np.column_stack([ap, ml])
        delta = ccf - v1_anchor
        delta_ml_ap = delta[:, [1, 0]]
        xy = delta_ml_ap @ matrix.T + pixel_center
        return xy[:, ::-1]  # row, col

    return ccf_to_pixel


def evaluate(name: str, ccf_to_pixel, probes: pd.DataFrame, template: dict, height: int, width: int) -> pd.DataFrame:
    row_col = ccf_to_pixel(probes.ccf_ap_mm.to_numpy(float), probes.ccf_ml_mm.to_numpy(float))
    result = probes.copy()
    result["pixel_row"] = row_col[:, 0]
    result["pixel_col"] = row_col[:, 1]
    valid_px = (result.pixel_row >= 0) & (result.pixel_row < height) & (result.pixel_col >= 0) & (result.pixel_col < width)
    known_areas = list(AREA_SEEDS_XY)
    clipped = np.clip(row_col, [0, 0], [height - 1, width - 1])
    for area in known_areas:
        result[f"dist_{area}_px"] = template["area_distance"][area](clipped)
    dist_cols = [f"dist_{a}_px" for a in known_areas]
    result["nearest_zhuang_area"] = np.array(known_areas)[result[dist_cols].to_numpy().argmin(axis=1)]
    result["own_area_distance_px"] = result.apply(
        lambda r: r[f"dist_{r.map_area}_px"] if r.map_area in known_areas else np.nan, axis=1
    )
    result["agrees"] = result.own_area_distance_px <= 1.5
    result["has_zhuang_reference"] = result.map_area.isin(known_areas)
    result["valid_px"] = valid_px
    result["geometry"] = name
    return result


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    cv_optimal = json.loads(CV_OPTIMAL.read_text())

    template = build_template(ZHUANG_TEMPLATE)
    height, width = template["domain"].shape

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    v1_cells = cells.loc[cells.map_area.eq("VISp")]
    v1_ccf_ap = v1_cells.ccf_ap_mm.median()
    v1_ccf_ml = v1_cells.ccf_ml_mm.median()

    probes = (
        cells.groupby(["ecephys_session_id", "ecephys_probe_id", "map_area"], as_index=False)
        .agg(ccf_ap_mm=("ccf_ap_mm", "median"), ccf_ml_mm=("ccf_ml_mm", "median"), cells=("ccf_ap_mm", "size"))
    )

    base_scale = geometry["fixed_scale_px_per_mm"]
    tx, ty = geometry["fitted_translation_px"]
    geometries = {
        "no_rotation": make_transform(0.0, base_scale, 0.0, 0.0, v1_ccf_ap, v1_ccf_ml),
        "current_locked_-8.1deg": make_transform(geometry["fitted_rotation_deg"], base_scale, tx, ty, v1_ccf_ap, v1_ccf_ml),
        "cv_optimal_+0.45deg": make_transform(cv_optimal["total_rotation_deg"], cv_optimal["total_scale_px_per_mm"], tx, ty, v1_ccf_ap, v1_ccf_ml),
    }

    tables = []
    print(f"{'geometry':26s} {'overall':>8s} {'VISp':>8s} {'VISl(LM)':>9s} {'VISal(AL)':>10s} {'VISrl':>8s} {'VISam':>8s}")
    for name, transform in geometries.items():
        result = evaluate(name, transform, probes, template, height, width)
        checked = result.loc[result.has_zhuang_reference & result.valid_px]
        by_area = checked.groupby("map_area").agrees.mean()
        overall = checked.agrees.mean()
        print(f"{name:26s} {overall:8.1%} " + " ".join(f"{by_area.get(a, float('nan')):8.1%}" for a in ("VISp", "VISl", "VISal", "VISrl", "VISam")))
        tables.append(result)

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUTPUT / "area_agreement_across_geometries.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    boundary = template["boundary"].astype(float)
    for ax, (name, transform) in zip(axes, geometries.items()):
        ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
        for acronym, (x, y) in AREA_SEEDS_XY.items():
            ax.text(x, y, AREA_LABELS[acronym], ha="center", va="center", fontsize=9, color="#555555",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
        result = combined.loc[combined.geometry == name]
        for area, group in result.groupby("map_area"):
            ax.scatter(group.pixel_col, group.pixel_row, marker=AREA_MARKERS.get(area, "x"),
                       color=AREA_COLORS.get(area, "black"), s=55, edgecolors="white", linewidths=0.6,
                       label=f"{area} (n={len(group)})")
        checked = result.loc[result.has_zhuang_reference & result.valid_px]
        ax.set(title=f"{name}\noverall agreement={checked.agrees.mean():.1%}",
               xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Probe area-compartment agreement across candidate geometries", fontsize=13)
    figure_path = OUTPUT / "Figure_area_agreement_across_geometries.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
