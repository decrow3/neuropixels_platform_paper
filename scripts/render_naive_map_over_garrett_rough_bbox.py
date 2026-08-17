#!/usr/bin/env python3
"""Naive V1-centered pooled scatter over Garrett 2014, same rough treatment as Zhuang.

Garrett has no independent scale-bar calibration like Zhuang's Figure 3 (its own README:
"absolute millimetre scale remain[s] to be established"), but it does have one usable anchor
built in: Garrett's own coordinate origin IS the V1 centroid (v1_mask center of mass), by
construction of the source extraction. So:

1. Anchor: (x=0, y=0) in Garrett's panel-unit frame = naive data's own V1 median CCF position.
   No seed-pixel lookup needed, unlike Zhuang.
2. Scale: no scale bar exists. First attempt matched V1 size against the naive data's own
   scattered cells directly (0.49 panel-units/mm) and came out roughly 2x too small -- a
   filled 2D mask isn't a fair size comparison against sparse recorded points. Cross-
   calibrating against Zhuang's own V1 compartment instead (which does have a true scale bar)
   gives 0.259 panel-units/mm, implying a ~3.85mm domain width, consistent with Zhuang's own
   ~4.4mm calibrated width. Still a real measurement, just anchored through a second atlas
   instead of noisy ephys sampling.
3. Color scale: Garrett's own value AT (0,0) (the V1 centroid) is the additive offset for the
   naive map's V1-relative values, mirroring the Zhuang V1-seed-pixel offset exactly.
4. Orientation: same open reflection question as Zhuang, plus an open AP/y sign convention
   (Garrett's y axis is "positive up" in the source figure, not tied to any AP direction) --
   both are rough initial guesses here, meant to be corrected by eye same as Zhuang's mirror was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_garrett2014_smoothed_field_and_ccf_affine import build_fields  # noqa: E402
from register_allen_session_to_zhuang import build_template as build_zhuang_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
GARRETT_V1_MASK = ROOT / "artifacts/retinotopy_template/garrett2014_figure5/field_sign_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"

REFLECT_ML = True  # rough initial guess, matching Zhuang's locked choice; not yet verified for Garrett
AP_SIGN = -1.0  # rough initial guess: row increases as panel-y decreases, matching Zhuang's row~AP convention


def main() -> None:
    v1_grid = np.load(GARRETT_V1_MASK)
    v1_mask = v1_grid["v1_mask"]
    gx, gy = v1_grid["x_panel_width"], v1_grid["y_panel_width"]
    garrett_v1_std = np.hypot(gx[v1_mask].std(), gy[v1_mask].std())

    zhuang_template = build_zhuang_template(ZHUANG_TEMPLATE)
    zhuang_v1_mask = zhuang_template["area_masks"]["VISp"]
    zhuang_rows, zhuang_cols = np.nonzero(zhuang_v1_mask)
    zhuang_px_per_mm = 62.0 / 0.5 * 0.8432313316638625
    zhuang_v1_std_mm = np.hypot(zhuang_rows.std(), zhuang_cols.std()) / zhuang_px_per_mm

    panel_units_per_mm = garrett_v1_std / zhuang_v1_std_mm
    print(f"Garrett scale, cross-calibrated via Zhuang V1 compartment: {panel_units_per_mm:.4f} panel-units/mm "
          f"({1.0/panel_units_per_mm:.2f} mm/panel-unit)")

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    v1_cells = cells.loc[cells.map_area.eq("VISp")]
    v1_ccf_ap = v1_cells.ccf_ap_mm.median()
    v1_ccf_ml = v1_cells.ccf_ml_mm.median()
    print(f"naive V1 anchor: AP={v1_ccf_ap:.2f} mm, ML={v1_ccf_ml:.2f} mm, n={len(v1_cells)} cells")

    fields = build_fields()
    axis0, axis1 = fields["x_axis"], fields["y_axis"]  # x_axis increasing, y_axis decreasing (row order)
    boundary = fields["boundary"]
    boundary_rows, boundary_cols = np.nonzero(boundary)

    def panel_to_ccf(row_col: np.ndarray) -> np.ndarray:
        row, col = row_col[:, 0], row_col[:, 1]
        panel_y = axis1[np.clip(row.astype(int), 0, len(axis1) - 1)]
        panel_x = axis0[np.clip(col.astype(int), 0, len(axis0) - 1)]
        ap = v1_ccf_ap + AP_SIGN * panel_y / panel_units_per_mm
        ml_sign = -1.0 if REFLECT_ML else 1.0
        ml = v1_ccf_ml + ml_sign * panel_x / panel_units_per_mm
        return np.column_stack([ml, ap])

    garrett_azimuth_at_v1 = float(fields["azimuth_deg_smoothed_for_gradient"][
        np.argmin(np.abs(axis1)), np.argmin(np.abs(axis0))])
    garrett_elevation_at_v1 = float(fields["elevation_deg_smoothed_for_gradient"][
        np.argmin(np.abs(axis1)), np.argmin(np.abs(axis0))])
    print(f"Garrett value at V1 centroid: azimuth={garrett_azimuth_at_v1:.1f}, elevation={garrett_elevation_at_v1:.1f}")

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("normalized_rf_x", "azimuth_deg_smoothed_for_gradient", garrett_azimuth_at_v1, "Naive azimuth (V1-anchor offset) vs. Garrett azimuth", "viridis", azimuth_norm),
        ("normalized_rf_y", "elevation_deg_smoothed_for_gradient", garrett_elevation_at_v1, "Naive elevation (V1-anchor offset) vs. Garrett elevation", "coolwarm", elevation_norm),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (naive_col, field_key, anchor_offset, title, cmap, norm) in zip(axes, panels):
        field = fields[field_key]
        rows, cols = np.nonzero(np.isfinite(field))
        ccf = panel_to_ccf(np.column_stack([rows, cols]))
        ax.scatter(ccf[:, 0], ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                   marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
        boundary_ccf = panel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))
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
        f"Naive V1-centered pooled map (n={len(cells)} cells) over Garrett 2014,\n"
        f"V1-size-matched scale ({panel_units_per_mm:.2f} panel-units/mm) + V1-anchor placement, "
        f"{reflect_label} -- no rotation/shear/optimizer fit",
        fontsize=12,
    )
    suffix = "_mirrored" if REFLECT_ML else ""
    figure_path = OUTPUT / f"Figure_naive_pooled_cells_over_garrett_true_scale{suffix}.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
