#!/usr/bin/env python3
"""Naive V1-centered pooled cell scatter over the population-registered, smoothed Zhuang map.

Same anatomy-constrained convention as `render_anatomy_constrained_cell_mapping_14animal.py`
(anatomy fixed, atlas inverse-warped into CCF, high-to-low mm axes, equal aspect, shared color
scale), but using ALL 45 sessions' pooled naive cells at once and the population-level
CCF<->Zhuang affine fit in `register_naive_map_to_atlases.py` (173 grid-cell landmarks across
the whole cohort), not a single session's 5-6 penetrations. Naive cell colors are the fitted
global offset added back (`normalized_rf + offset`) so they sit on the same absolute-degree
scale as Zhuang's predictions.
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
ZHUANG_SMOOTH = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign.npz"
)
REGISTRATION_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/run_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"
MEDIAL_DISPLAY_CUTOFF_ML_MM = 6.75


def background_layers(axis, field: np.ndarray, boundary: np.ndarray, cmap: str, norm, template_to_ccf) -> None:
    rows, cols = np.nonzero(np.isfinite(field))
    values_ccf = template_to_ccf(np.column_stack([cols, rows]))
    axis.scatter(values_ccf[:, 0], values_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                 marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
    height, width = boundary.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_ccf = template_to_ccf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    grid_ml = grid_ccf[:, 0].reshape(boundary.shape)
    grid_ap = grid_ccf[:, 1].reshape(boundary.shape)
    axis.contour(grid_ml, grid_ap, boundary.astype(float), levels=[0.5], colors="#343434", linewidths=0.55, zorder=2)


def main() -> None:
    manifest = json.loads(REGISTRATION_MANIFEST.read_text())["zhuang"]
    ccf_center = np.asarray(manifest["ccf_center_ap_ml_mm"], dtype=float)
    template_center = np.asarray(manifest["template_center_xy"], dtype=float)
    matrix = np.asarray(manifest["matrix_px_per_mm"], dtype=float)
    offset = np.asarray(manifest["fitted_offset_deg"], dtype=float)
    inverse_matrix = np.linalg.inv(matrix)

    def template_to_ccf(points_xy: np.ndarray) -> np.ndarray:
        ap_ml = ccf_center + (np.asarray(points_xy, dtype=float) - template_center) @ inverse_matrix.T
        return ap_ml[:, [1, 0]]  # ML horizontal, AP vertical

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    cells["corrected_azimuth"] = cells.normalized_rf_x + offset[0]
    cells["corrected_elevation"] = cells.normalized_rf_y + offset[1]

    smoothed = {k: v for k, v in np.load(ZHUANG_SMOOTH).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("corrected_azimuth", "azimuth_smoothed_for_gradient_deg", "Naive pooled azimuth (offset-corrected) vs. registered Zhuang", "viridis", azimuth_norm),
        ("corrected_elevation", "elevation_smoothed_for_gradient_deg", "Naive pooled elevation (offset-corrected) vs. registered Zhuang", "coolwarm", elevation_norm),
    )

    boundary_y, boundary_x = np.nonzero(boundary)
    warped_boundary = template_to_ccf(np.column_stack([boundary_x, boundary_y]))
    all_ml = np.r_[warped_boundary[:, 0], cells.ccf_ml_mm.to_numpy()]
    all_ap = np.r_[warped_boundary[:, 1], cells.ccf_ap_mm.to_numpy()]
    padding_mm = 0.12
    x_limits = (max(all_ml.min() - padding_mm, MEDIAL_DISPLAY_CUTOFF_ML_MM), all_ml.max() + padding_mm)
    y_limits = (all_ap.min() - padding_mm, all_ap.max() + padding_mm)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (field_col, template_key, title, cmap, norm) in zip(axes, panels):
        background_layers(ax, smoothed[template_key], boundary, cmap, norm, template_to_ccf)
        ax.scatter(cells.ccf_ml_mm, cells.ccf_ap_mm, c=cells[field_col], cmap=cmap, norm=norm,
                   s=5, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees; shared by cells and atlas")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)",
               xlim=x_limits[::-1], ylim=y_limits[::-1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Naive V1-centered pooled map (n={len(cells)} cells, {cells.ecephys_session_id.nunique()} sessions)\n"
        f"over Zhuang, registered at the population level (173 landmarks, fitted offset "
        f"az={offset[0]:+.1f} deg, el={offset[1]:+.1f} deg)",
        fontsize=12.5,
    )
    figure_path = OUTPUT / "Figure_naive_pooled_cells_over_registered_zhuang.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
