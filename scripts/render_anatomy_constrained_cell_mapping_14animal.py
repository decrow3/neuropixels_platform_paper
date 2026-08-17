#!/usr/bin/env python3
"""Anatomy-constrained cell mapping for a 14-animal-cohort session, smoothed Zhuang background.

Follows the established plotting spec from `render_anatomy_constrained_cell_mapping.py` /
`session_798911424/anatomy_constrained_cell_mapping_manifest.json`: anatomy is fixed (cells
plotted at measured CCF), the atlas is inverse-warped into that session's CCF frame by the
exact inverse of the fitted CCF->template affine, both displayed axes (ML horizontal, AP
vertical, mm) run high-to-low, equal aspect, tight bounds padded 0.12 mm with the medial edge
cropped at ML=6.75 mm, cells and atlas share one color scale per panel (viridis 0-90 for
azimuth, zero-centered coolwarm -25/+40 for elevation), marker shape = Allen area, open
circles + probe-suffix labels mark penetration medians.

The one deliberate change from that script: the atlas background here is the Gaussian-
smoothed, support-masked continuous field from
`zhuang2017_figure9/interpolation_field_sign_qa/interpolated_fields_and_field_sign.npz`,
not the raw sparse contour points -- this session's "use the Zhuang map smoothed" decision.
Cells and affine come from the 14-animal cohort (`build_14animal_retinotopy_registration.py`),
not the older single-session pilot's saved CSV/manifest format, since this session
(781842082) was only fit in that cohort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_14animal_retinotopy_registration import production_support  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = 781842082
ZHUANG_SMOOTH = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign.npz"
)
LANDMARKS_14 = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1/registered_probe_landmarks.csv"
OUTPUT_ROOT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1/anatomy_constrained_cell_mapping"

AREA_MARKERS = {"VISp": "o", "VISl": "s", "VISal": "^", "VISrl": "D", "VISam": "P"}
AREA_LABELS = {"VISp": "V1", "VISl": "LM", "VISal": "AL", "VISrl": "RL", "VISam": "AM"}
AREA_SEEDS_XY = {
    "VISp": (200, 240), "VISl": (100, 260), "VISal": (75, 190),
    "VISrl": (180, 80), "VISam": (240, 80),
}
MEDIAL_DISPLAY_CUTOFF_ML_MM = 6.75


def recover_ccf_to_pixel_affine(session_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(linear, intercept): pixel = ccf_mm @ linear.T + intercept, exact via OLS (not refit)."""
    landmarks = pd.read_csv(LANDMARKS_14)
    session = landmarks.loc[landmarks.session_id.eq(session_id)]
    ccf = session[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    pixel = session[["template_x_px", "template_y_px"]].to_numpy(float)
    design = np.column_stack([ccf, np.ones(len(ccf))])
    coefficient, *_ = np.linalg.lstsq(design, pixel, rcond=None)
    residual = float(np.abs(design @ coefficient - pixel).max())
    return coefficient[:2].T, coefficient[2], residual


def background_layers(axis, field: np.ndarray, boundary: np.ndarray, cmap: str, norm, template_to_ccf) -> None:
    rows, cols = np.nonzero(np.isfinite(field))
    values_ccf = template_to_ccf(np.column_stack([cols, rows]))
    axis.scatter(
        values_ccf[:, 0], values_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
        marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True,
    )
    height, width = boundary.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_ccf = template_to_ccf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    grid_ml = grid_ccf[:, 0].reshape(boundary.shape)
    grid_ap = grid_ccf[:, 1].reshape(boundary.shape)
    axis.contour(grid_ml, grid_ap, boundary.astype(float), levels=[0.5], colors="#343434", linewidths=0.55, zorder=2)
    for acronym, (x, y) in AREA_SEEDS_XY.items():
        label_ml, label_ap = template_to_ccf(np.array([[x, y]], dtype=float))[0]
        axis.text(label_ml, label_ap, AREA_LABELS[acronym], ha="center", va="center", fontsize=8,
                  color="#555555", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 1.0}, zorder=2)


def render(cells: pd.DataFrame, smoothed: dict, linear: np.ndarray, intercept: np.ndarray, output: Path, session_id: int) -> None:
    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("visual_azimuth_deg", "azimuth_smoothed_for_gradient_deg", "RF azimuth and smoothed Zhuang azimuth", "viridis", azimuth_norm),
        ("visual_elevation_deg", "elevation_smoothed_for_gradient_deg", "RF elevation and smoothed Zhuang elevation", "coolwarm", elevation_norm),
    )
    inverse_linear = np.linalg.inv(linear)

    def template_to_ccf(points_xy: np.ndarray) -> np.ndarray:
        # pixel = ccf @ linear.T + intercept  =>  ccf = (pixel - intercept) @ inverse_linear.T
        ccf = (np.asarray(points_xy, dtype=float) - intercept) @ inverse_linear.T
        return ccf[:, [1, 0]]  # ML horizontal, AP vertical

    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_y, boundary_x = np.nonzero(boundary)
    warped_boundary = template_to_ccf(np.column_stack([boundary_x, boundary_y]))
    all_ml = np.r_[warped_boundary[:, 0], cells.ccf_ml_mm.to_numpy()]
    all_ap = np.r_[warped_boundary[:, 1], cells.ccf_ap_mm.to_numpy()]
    padding_mm = 0.12
    x_limits = (max(all_ml.min() - padding_mm, MEDIAL_DISPLAY_CUTOFF_ML_MM), all_ml.max() + padding_mm)
    y_limits = (all_ap.min() - padding_mm, all_ap.max() + padding_mm)

    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), constrained_layout=True)
    for axis, (field_col, template_key, title, cmap, norm) in zip(axes, panels):
        background_layers(axis, smoothed[template_key], boundary, cmap, norm, template_to_ccf)
        for area, group in cells.groupby("ecephys_structure_acronym", sort=True):
            axis.scatter(
                group.ccf_ml_mm, group.ccf_ap_mm, c=group[field_col], cmap=cmap, norm=norm,
                marker=AREA_MARKERS[area], s=28, alpha=0.84, linewidths=0.3,
                edgecolors="#202020", zorder=4, rasterized=True,
            )
        probe_centers = cells.groupby("ecephys_probe_id", as_index=False).agg(
            ccf_ap_mm=("ccf_ap_mm", "median"), ccf_ml_mm=("ccf_ml_mm", "median"),
        )
        axis.scatter(probe_centers.ccf_ml_mm, probe_centers.ccf_ap_mm, marker="o", s=115,
                     facecolors="none", edgecolors="#111111", linewidths=1.1, zorder=5)
        for row in probe_centers.itertuples():
            axis.text(row.ccf_ml_mm + 0.025, row.ccf_ap_mm + 0.025, str(int(row.ecephys_probe_id))[-3:],
                      fontsize=7, color="#111111", zorder=6)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees; shared by cells and atlas")
        axis.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)",
                 xlim=x_limits[::-1], ylim=y_limits[::-1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        axis.set_axisbelow(True)

    area_handles = [
        Line2D([], [], marker=AREA_MARKERS[area], markerfacecolor="#aaaaaa", markeredgecolor="#222222",
               markeredgewidth=0.4, linestyle="", markersize=7, label=f"{area}->{AREA_LABELS[area]}")
        for area in sorted(AREA_MARKERS) if area in set(cells.ecephys_structure_acronym)
    ]
    axes[1].legend(handles=area_handles, title="Marker = Allen CCF area", loc="upper right",
                   fontsize=7, title_fontsize=8, frameon=True)
    figure.suptitle(
        f"Session {session_id}: cells fixed in CCF anatomy; smoothed Zhuang map inverse-warped into CCF\n"
        f"n={len(cells)} cells on {cells.ecephys_probe_id.nunique()} penetrations · shared color scale per panel · native azimuth convention",
        fontsize=12.5,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=DEFAULT_SESSION)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / f"session_{args.session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    linear, intercept, residual = recover_ccf_to_pixel_affine(args.session_id)
    print(f"affine recovery max residual: {residual:.2e} px")

    cells, audit = production_support()
    session_cells = cells.loc[cells.session_id.eq(args.session_id)].copy()

    smoothed_npz = np.load(ZHUANG_SMOOTH)
    smoothed = {name: smoothed_npz[name] for name in smoothed_npz.files}

    figure_path = output_dir / "Figure_CCF_cells_with_inverse_warped_smoothed_Zhuang_map.png"
    render(session_cells, smoothed, linear, intercept, figure_path, args.session_id)
    print(figure_path)


if __name__ == "__main__":
    main()
