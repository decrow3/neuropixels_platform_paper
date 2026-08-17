#!/usr/bin/env python3
"""Full color-overlay comparison (not just probe area-membership dots): all-session pooled
cells over the span-matched Zhuang background, under the CURRENT LOCKED geometry (-8.1 deg,
scale 104.6 px/mm) vs. the RF-vector-error CV-OPTIMAL geometry (net +0.45 deg, scale 109.0
px/mm) found by `verify_naive_vs_global_restricted_affine_cv.py`. Both use a single pooled
(naive-style) offset fit under their own geometry, same color scale, for a fair side-by-side --
same structure as `render_naive_vs_per_session_translation_comparison.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
CV_OPTIMAL = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset/global_restricted_affine_full_fit.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"


def build_pixel_to_ccf(rotation_deg, scale_px_per_mm, tx, ty, v1_anchor_ap, v1_anchor_ml):
    theta = np.radians(rotation_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale_reflect = np.diag([-1.0 * scale_px_per_mm, scale_px_per_mm])
    matrix = rotation @ scale_reflect
    inverse_matrix = np.linalg.inv(matrix)
    v1_seed_col, v1_seed_row = 200, 240
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])

    def pixel_to_ccf(row_col: np.ndarray) -> np.ndarray:
        row, col = row_col[:, 0], row_col[:, 1]
        xy = np.column_stack([col, row])
        delta_ml_ap = (xy - pixel_center) @ inverse_matrix.T
        ml = v1_anchor_ml + delta_ml_ap[:, 0]
        ap = v1_anchor_ap + delta_ml_ap[:, 1]
        return np.column_stack([ml, ap])

    def ccf_to_pixel(ccf: np.ndarray) -> np.ndarray:
        delta = ccf - np.array([v1_anchor_ap, v1_anchor_ml])
        delta_ml_ap = delta[:, [1, 0]]
        return delta_ml_ap @ matrix.T + pixel_center

    return pixel_to_ccf, ccf_to_pixel


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    cv_optimal = json.loads(CV_OPTIMAL.read_text())
    tx, ty = geometry["fitted_translation_px"]
    v1_anchor_ap, v1_anchor_ml = geometry["v1_anchor_ccf_ap_ml_mm"]

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)

    row_defs = (
        ("current_locked", geometry["fitted_rotation_deg"], geometry["fixed_scale_px_per_mm"],
         39.53167239753153, -5.12557337808509,
         f"CURRENT LOCKED: rotation={geometry['fitted_rotation_deg']:+.1f} deg, scale={geometry['fixed_scale_px_per_mm']:.1f} px/mm"),
        ("cv_optimal", cv_optimal["total_rotation_deg"], cv_optimal["total_scale_px_per_mm"],
         cv_optimal["offset_az"], cv_optimal["offset_el"],
         f"CV-OPTIMAL: rotation={cv_optimal['total_rotation_deg']:+.1f} deg, scale={cv_optimal['total_scale_px_per_mm']:.1f} px/mm"),
    )

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    col_defs = (
        ("azimuth_span_matched_deg", "viridis", azimuth_norm, "Azimuth"),
        ("elevation_span_matched_deg", "coolwarm", elevation_norm, "Elevation"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.6), constrained_layout=True)
    for row_idx, (row_key, rotation_deg, scale_px_per_mm, offset_az, offset_el, row_title) in enumerate(row_defs):
        pixel_to_ccf, ccf_to_pixel = build_pixel_to_ccf(rotation_deg, scale_px_per_mm, tx, ty, v1_anchor_ap, v1_anchor_ml)
        boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))
        cells["azimuth_corrected"] = cells.normalized_rf_x + offset_az
        cells["elevation_corrected"] = cells.normalized_rf_y + offset_el

        for col_idx, (field_key, cmap, norm, panel_title) in enumerate(col_defs):
            ax = axes[row_idx, col_idx]
            field = smoothed[field_key]
            rows, cols = np.nonzero(np.isfinite(field))
            bg_ccf = pixel_to_ccf(np.column_stack([rows, cols]))
            ax.scatter(bg_ccf[:, 0], bg_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                       marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
            ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.6, color="#343434", zorder=2, rasterized=True)
            cell_col = "azimuth_corrected" if col_idx == 0 else "elevation_corrected"
            ax.scatter(cells.ccf_ml_mm, cells.ccf_ap_mm, c=cells[cell_col], cmap=cmap, norm=norm,
                       s=5, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
            colorbar.set_label("degrees")
            ax.set(title=f"{panel_title}\n{row_title}" if col_idx == 0 else panel_title,
                   xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
            ax.invert_xaxis(); ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
            ax.set_axisbelow(True)

    fig.suptitle(
        "Current locked geometry vs. RF-vector-error CV-optimal geometry -- same span-matched field, "
        "same color scale, single pooled offset each",
        fontsize=13,
    )
    figure_path = OUTPUT / "Figure_current_vs_cv_optimal_geometry_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
