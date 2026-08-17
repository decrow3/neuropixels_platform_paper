#!/usr/bin/env python3
"""All-session pooled scatter over the canonical span-matched Zhuang map, using each session's
own fitted RF offset (`fit_per_session_rf_offset_to_zhuang_span_matched.py`'s capped, final
offset) rather than one shared pooled offset. Same fixed anatomical registration (rotation,
translation, scale, reflection) and same overlay style as
`render_naive_map_over_zhuang_rotation_fit.py`; the only change is that each cell is corrected
by its own session's offset before plotting, so this is the map to inspect for whether
per-session gaze correction actually tightens agreement with the atlas versus the single
pooled-offset version.
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
PER_SESSION_OFFSETS = (
    ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"
    / "per_session_rf_offset.csv"
)
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
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

    offsets = pd.read_csv(PER_SESSION_OFFSETS)
    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    cells = cells.merge(
        offsets[["ecephys_session_id", "final_offset_az", "final_offset_el", "sufficient_support", "capped"]],
        on="ecephys_session_id", how="left", validate="many_to_one",
    )
    missing_offset = cells.final_offset_az.isna()
    if missing_offset.any():
        raise RuntimeError(f"{missing_offset.sum()} cells have no matching session offset")
    cells["registered_azimuth_deg"] = cells.normalized_rf_x + cells.final_offset_az
    cells["registered_elevation_deg"] = cells.normalized_rf_y + cells.final_offset_el

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)
    boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    panels = (
        ("registered_azimuth_deg", "azimuth_span_matched_deg", "Azimuth: per-session-registered cells vs. Zhuang (span-matched)", "viridis", azimuth_norm),
        ("registered_elevation_deg", "elevation_span_matched_deg", "Elevation: per-session-registered cells vs. Zhuang (span-matched)", "coolwarm", elevation_norm),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (cell_col, field_key, title, cmap, norm) in zip(axes, panels):
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
        colorbar.set_label("degrees; shared scale, per-session offset applied")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    n_capped = int(offsets.capped.fillna(False).sum())
    fig.suptitle(
        f"All-session pooled cells (n={len(cells)}) over Zhuang, span-matched field,\n"
        f"per-session RF offset applied ({n_capped}/{len(offsets)} sessions capped), "
        f"anatomical registration fixed (rotation {geometry['fitted_rotation_deg']:+.1f} deg, scale/mirror locked)",
        fontsize=12,
    )
    figure_path = OUTPUT / "Figure_all_cells_over_zhuang_per_session_registered.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
