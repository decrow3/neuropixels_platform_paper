#!/usr/bin/env python3
"""Naive V1-centered pooled scatter over Zhuang, using the fitted translation+rotation
transform (`fit_translation_rotation_naive_to_zhuang.py`), same spatial-overlay style as
`render_naive_map_over_zhuang_rough_bbox.py`'s rough (translation-only) reference.
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
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
FIT_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"

REFLECT_ML = True
V1_SEED_XY_PX = (200, 240)


def main() -> None:
    fit = json.loads(FIT_MANIFEST.read_text())
    px_per_mm = fit["fixed_scale_px_per_mm"]
    theta = np.radians(fit["fitted_rotation_deg"])
    tx, ty = fit["fitted_translation_px"]
    offset_az, offset_el = fit["fitted_rf_offset_deg"]
    v1_anchor_ap, v1_anchor_ml = fit["v1_anchor_ccf_ap_ml_mm"]
    v1_seed_col, v1_seed_row = V1_SEED_XY_PX
    ml_sign = -1.0 if REFLECT_ML else 1.0

    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale_reflect = np.diag([ml_sign * px_per_mm, px_per_mm])
    matrix = rotation @ scale_reflect
    inverse_matrix = np.linalg.inv(matrix)
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])

    print(f"using fitted rotation={fit['fitted_rotation_deg']:+.1f} deg, "
          f"translation=({tx:+.1f},{ty:+.1f}) px, RF offset=({offset_az:+.1f},{offset_el:+.1f}) deg")

    def pixel_to_ccf(row_col: np.ndarray) -> np.ndarray:
        row, col = row_col[:, 0], row_col[:, 1]
        xy = np.column_stack([col, row])
        delta_ml_ap = (xy - pixel_center) @ inverse_matrix.T
        ml = v1_anchor_ml + delta_ml_ap[:, 0]
        ap = v1_anchor_ap + delta_ml_ap[:, 1]
        return np.column_stack([ml, ap])

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    smoothed = {k: v for k, v in np.load(ZHUANG_SMOOTH).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    panels = (
        ("normalized_rf_x", "azimuth_span_matched_deg", offset_az, "Naive azimuth (fitted offset) vs. Zhuang azimuth (span-matched)", "viridis", azimuth_norm),
        ("normalized_rf_y", "elevation_span_matched_deg", offset_el, "Naive elevation (fitted offset) vs. Zhuang elevation (span-matched)", "coolwarm", elevation_norm),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (naive_col, template_key, anchor_offset, title, cmap, norm) in zip(axes, panels):
        field = smoothed[template_key]
        rows, cols = np.nonzero(np.isfinite(field))
        ccf = pixel_to_ccf(np.column_stack([rows, cols]))
        ax.scatter(ccf[:, 0], ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                   marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
        boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))
        ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.6, color="#343434", zorder=2, rasterized=True)

        corrected = cells[naive_col] + anchor_offset
        ax.scatter(cells.ccf_ml_mm, cells.ccf_ap_mm, c=corrected, cmap=cmap, norm=norm,
                   s=5, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees; shared scale via fitted offset")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Naive V1-centered pooled map (n={len(cells)} cells) over Zhuang,\n"
        f"fitted translation + rotation ({fit['fitted_rotation_deg']:+.1f} deg), scale/mirror fixed, "
        f"area agreement {fit['area_agreement_overall']:.0%}",
        fontsize=12.5,
    )
    figure_path = OUTPUT / "Figure_naive_pooled_cells_over_zhuang_rotation_fit_span_matched.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
