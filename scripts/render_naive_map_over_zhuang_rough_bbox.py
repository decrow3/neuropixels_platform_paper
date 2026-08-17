#!/usr/bin/env python3
"""Naive V1-centered pooled scatter over Zhuang, placed by real scale + V1 anchor (no fit).

Still no fitted rotation/shear/optimizer -- but two things are now grounded in independent,
non-data-fit facts instead of an arbitrary bounding-box stretch:

1. Spatial scale: Zhuang's Figure 9 template is rendered at its own true physical scale
   (`template_px_per_mm`, from the Figure 3 scale bar calibration already established in
   `render_anatomy_constrained_cell_mapping.py`), not stretched to fill whatever bounding box
   the naive data happens to span. Only a translation (and the one open left-right reflection
   flag) is applied on top of that fixed, correct scale.
2. Color scale: the naive map's values are V1-median-relative by construction (zero at V1).
   Zhuang's own predicted azimuth/elevation AT its V1 seed pixel (`AREA_SEEDS_XY["VISp"]`) is
   used as the additive offset, so naive cells and the Zhuang background end up on the same
   absolute-degree color scale via one direct anchor point, not a fitted intercept.
"""

from __future__ import annotations

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
    / "interpolated_fields_and_field_sign_domain_patched.npz"
)  # domain-patched (V-notch waist filled, confirmed correct); original unpatched version stays
   # on disk as the locked naive reference at Figure_naive_pooled_cells_over_zhuang_true_scale*.png
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"

REFLECT_ML = True  # the one open degree of freedom; still not resolved by this rough method.
V1_SEED_XY_PX = (200, 240)  # AREA_SEEDS_XY["VISp"] in register_allen_session_to_zhuang.py
ZHUANG_FIG3_SCALE_BAR_PX = 62.0
ZHUANG_FIG3_SCALE_BAR_MM = 0.5
FIG3_TO_FIG9_SIMILARITY_SCALE = 0.8432313316638625


def main() -> None:
    px_per_mm = ZHUANG_FIG3_SCALE_BAR_PX / ZHUANG_FIG3_SCALE_BAR_MM * FIG3_TO_FIG9_SIMILARITY_SCALE
    print(f"Zhuang template scale: {px_per_mm:.1f} px/mm ({1000.0 / px_per_mm:.1f} um/px)")

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    v1_cells = cells.loc[cells.map_area.eq("VISp")]
    v1_ccf_ap = v1_cells.ccf_ap_mm.median()
    v1_ccf_ml = v1_cells.ccf_ml_mm.median()
    print(f"naive V1 anchor: AP={v1_ccf_ap:.2f} mm, ML={v1_ccf_ml:.2f} mm, n={len(v1_cells)} cells")

    smoothed = {k: v for k, v in np.load(ZHUANG_SMOOTH).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)

    v1_seed_col, v1_seed_row = V1_SEED_XY_PX
    zhuang_azimuth_at_v1 = float(smoothed["azimuth_smoothed_for_gradient_deg"][v1_seed_row, v1_seed_col])
    zhuang_elevation_at_v1 = float(smoothed["elevation_smoothed_for_gradient_deg"][v1_seed_row, v1_seed_col])
    print(f"Zhuang value at V1 seed pixel: azimuth={zhuang_azimuth_at_v1:.1f}, elevation={zhuang_elevation_at_v1:.1f}")

    def pixel_to_ccf(row_col: np.ndarray) -> np.ndarray:
        # row (pixel y, down+) -> AP, same direction (both increase downward in the display).
        # col (pixel x, right+) -> ML, sign set by the one open reflection flag. True isotropic
        # px/mm scale throughout -- no per-axis stretch to fit a bounding box.
        row, col = row_col[:, 0], row_col[:, 1]
        ap = v1_ccf_ap + (row - v1_seed_row) / px_per_mm
        ml_sign = -1.0 if REFLECT_ML else 1.0
        ml = v1_ccf_ml + ml_sign * (col - v1_seed_col) / px_per_mm
        return np.column_stack([ml, ap])

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("normalized_rf_x", "azimuth_smoothed_for_gradient_deg", zhuang_azimuth_at_v1, "Naive azimuth (V1-anchor offset) vs. Zhuang azimuth", "viridis", azimuth_norm),
        ("normalized_rf_y", "elevation_smoothed_for_gradient_deg", zhuang_elevation_at_v1, "Naive elevation (V1-anchor offset) vs. Zhuang elevation", "coolwarm", elevation_norm),
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
        colorbar.set_label("degrees; shared scale via V1-anchor offset")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    reflect_label = "left-right mirrored" if REFLECT_ML else "unreflected"
    fig.suptitle(
        f"Naive V1-centered pooled map (n={len(cells)} cells) over Zhuang,\n"
        f"true scale ({px_per_mm:.0f} px/mm) + V1-anchor placement, {reflect_label} -- no rotation/shear/optimizer fit",
        fontsize=12.5,
    )
    suffix = "_mirrored" if REFLECT_ML else ""
    figure_path = OUTPUT / f"Figure_naive_pooled_cells_over_zhuang_true_scale{suffix}.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
