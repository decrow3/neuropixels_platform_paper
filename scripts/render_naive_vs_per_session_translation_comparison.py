#!/usr/bin/env python3
"""Side-by-side comparison: NAIVE (one single pooled RF offset for every session -- no per-
session variation at all) vs. the locked default (each session's own capped RF offset), both
plotted over the SAME canonical span-matched Zhuang background with the SAME color scale, so
the two are directly, fairly comparable. Same anatomical geometry (rotation/translation/scale/
reflection) and same atlas field underlie both -- the ONLY difference is whether the RF offset
is one constant for the whole cohort or fit per session.
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
PER_SESSION_DIR = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"
OUTPUT = PER_SESSION_DIR


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    offset_manifest = json.loads((PER_SESSION_DIR / "per_session_offset_manifest.json").read_text())
    pooled_offset_az = offset_manifest["pooled_offset_az_deg"]
    pooled_offset_el = offset_manifest["pooled_offset_el_deg"]

    theta = np.radians(geometry["fitted_rotation_deg"])
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor_ap, v1_anchor_ml = geometry["v1_anchor_ccf_ap_ml_mm"]
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale_reflect = np.diag([ml_sign * px_per_mm, px_per_mm])
    matrix = rotation @ scale_reflect
    inverse_matrix = np.linalg.inv(matrix)
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])

    def pixel_to_ccf(row_col: np.ndarray) -> np.ndarray:
        row, col = row_col[:, 0], row_col[:, 1]
        xy = np.column_stack([col, row])
        delta_ml_ap = (xy - pixel_center) @ inverse_matrix.T
        ml = v1_anchor_ml + delta_ml_ap[:, 0]
        ap = v1_anchor_ap + delta_ml_ap[:, 1]
        return np.column_stack([ml, ap])

    offsets = pd.read_csv(PER_SESSION_DIR / "per_session_rf_offset.csv")
    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    cells = cells.merge(offsets[["ecephys_session_id", "final_offset_az", "final_offset_el"]],
                         on="ecephys_session_id", how="left", validate="many_to_one")
    if cells.final_offset_az.isna().any():
        raise RuntimeError("cells missing a matching session offset")

    cells["naive_azimuth_deg"] = cells.normalized_rf_x + pooled_offset_az
    cells["naive_elevation_deg"] = cells.normalized_rf_y + pooled_offset_el
    cells["per_session_azimuth_deg"] = cells.normalized_rf_x + cells.final_offset_az
    cells["per_session_elevation_deg"] = cells.normalized_rf_y + cells.final_offset_el

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)
    boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    row_defs = (
        ("naive", "naive_azimuth_deg", "naive_elevation_deg",
         f"NAIVE: one pooled offset for all sessions (az={pooled_offset_az:+.1f}, el={pooled_offset_el:+.1f} deg)"),
        ("per_session", "per_session_azimuth_deg", "per_session_elevation_deg",
         f"PER-SESSION: each session's own capped offset ({int(offsets.capped.sum())}/{len(offsets)} capped)"),
    )
    col_defs = (
        ("azimuth_smoothed_for_gradient_deg" if "azimuth_span_matched_deg" not in smoothed else "azimuth_span_matched_deg",
         "viridis", azimuth_norm, "Azimuth"),
        ("elevation_smoothed_for_gradient_deg" if "elevation_span_matched_deg" not in smoothed else "elevation_span_matched_deg",
         "coolwarm", elevation_norm, "Elevation"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.6), constrained_layout=True)
    for row_idx, (row_key, az_col, el_col, row_title) in enumerate(row_defs):
        for col_idx, (cell_col, (field_key, cmap, norm, panel_title)) in enumerate(
            zip((az_col, el_col), col_defs)
        ):
            ax = axes[row_idx, col_idx]
            field = smoothed[field_key]
            rows, cols = np.nonzero(np.isfinite(field))
            bg_ccf = pixel_to_ccf(np.column_stack([rows, cols]))
            ax.scatter(bg_ccf[:, 0], bg_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                       marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
            ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.6, color="#343434", zorder=2, rasterized=True)
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
        f"Naive (single pooled offset) vs. per-session RF offset -- same geometry, same span-matched field, "
        f"same color scale (n={len(cells)} cells)",
        fontsize=13,
    )
    figure_path = OUTPUT / "Figure_naive_vs_per_session_translation_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
