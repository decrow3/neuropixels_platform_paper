#!/usr/bin/env python3
"""Side-by-side comparison: Allen single units vs. MouseV2 single units, both plotted at their
own inferred position over the SAME span-matched Zhuang background field, same color scale.

IMPORTANT ASYMMETRY (read before interpreting): Allen's cell positions come from independent
histology (CCF registration) -- entirely independent of the RF value used to color them, exactly
like `render_naive_vs_per_session_translation_comparison.py`. MouseV2 has no such independent
per-unit position; units are placed along a per-probe line fit from those same units' own RF
values, so this panel is partially circular by construction (regardless of which fit is used, the
units would land where they'd have to be for their RF value to roughly track the map).

MouseV2 units come from `direction_search_unit_positions.csv`
(render_mousev2_direction_search_depth_spread.py), NOT `unit_positions_along_shank.csv`
(register_mousev2_units_along_probe_shank.py) -- checked directly (2026-08-18) that the latter's
free 2-endpoint fit does NOT anchor its shallow/entry endpoint to the true anatomical penetration
point at all (only a soft DIRECTIONAL nudge, no positional constraint): median distance from that
fit's own shallow endpoint to the real anatomical entry point was 95px, about half of V1's own
diameter, so plotted units visibly floated away from their own probe's X marker. The
direction-search script fixes the entry point exactly (only angle is fit), so its units are
guaranteed to connect to their own penetration point by construction.
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
ALLEN_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ALLEN_GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
ALLEN_PER_SESSION_OFFSETS = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset/per_session_rf_offset.csv"
MOUSEV2_UNITS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/direction_search_unit_positions.csv"
MOUSEV2_PROBE_POSITIONS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/probe_anatomical_position.csv"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
OUTPUT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang"


def allen_ccf_to_zhuang_px(ml: np.ndarray, ap: np.ndarray, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(geometry["fitted_rotation_deg"])
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor_ap, v1_anchor_ml = geometry["v1_anchor_ccf_ap_ml_mm"]
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = rotation @ np.diag([ml_sign * px_per_mm, px_per_mm])
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
    delta_ml_ap = np.column_stack([ml - v1_anchor_ml, ap - v1_anchor_ap])
    xy = delta_ml_ap @ matrix.T + pixel_center
    return xy[:, 0], xy[:, 1]  # col, row


def main() -> None:
    geometry = json.loads(ALLEN_GEOMETRY_MANIFEST.read_text())
    allen_offsets = pd.read_csv(ALLEN_PER_SESSION_OFFSETS)
    allen = pd.read_csv(ALLEN_CELLS)
    allen = allen.merge(allen_offsets[["ecephys_session_id", "final_offset_az", "final_offset_el"]],
                         on="ecephys_session_id", how="left", validate="many_to_one")
    allen = allen.dropna(subset=["final_offset_az", "final_offset_el"])
    allen["azimuth_deg"] = allen.normalized_rf_x + allen.final_offset_az
    allen["elevation_deg"] = allen.normalized_rf_y + allen.final_offset_el
    allen["zhuang_col"], allen["zhuang_row"] = allen_ccf_to_zhuang_px(
        allen.left_right_ccf_coordinate.to_numpy(float) / 1000.0,
        allen.anterior_posterior_ccf_coordinate.to_numpy(float) / 1000.0,
        geometry,
    )
    print(f"Allen cells: {len(allen)}")

    # Allen has no separate per-probe "penetration point" file -- approximated as each probe's
    # own median CCF position (a straight Neuropixels track's ml/ap barely varies with depth, so
    # the median across that probe's own recorded cells is a reasonable single-point stand-in).
    allen_penetrations = allen.groupby("ecephys_probe_id")[["left_right_ccf_coordinate", "anterior_posterior_ccf_coordinate"]].median().reset_index()
    allen_penetrations["zhuang_col"], allen_penetrations["zhuang_row"] = allen_ccf_to_zhuang_px(
        allen_penetrations.left_right_ccf_coordinate.to_numpy(float) / 1000.0,
        allen_penetrations.anterior_posterior_ccf_coordinate.to_numpy(float) / 1000.0,
        geometry,
    )
    print(f"Allen penetrations (approx, per-probe median position): {len(allen_penetrations)}")

    mousev2 = pd.read_csv(MOUSEV2_UNITS)
    mousev2 = mousev2.rename(columns={"inferred_row": "zhuang_row", "inferred_col": "zhuang_col",
                                       "observed_azimuth_deg": "azimuth_deg", "observed_elevation_deg": "elevation_deg"})
    print(f"MouseV2 units: {len(mousev2)}")

    mousev2_penetrations = pd.read_csv(MOUSEV2_PROBE_POSITIONS)
    print(f"MouseV2 penetrations (anatomical entry points): {len(mousev2_penetrations)}")

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    row_defs = (
        ("Allen", allen, allen_penetrations, f"Allen: {len(allen)} single units (independent CCF anatomical position)"),
        ("MouseV2", mousev2, mousev2_penetrations, f"MouseV2: {len(mousev2)} single units (RF-fit position -- see module docstring)"),
    )
    col_defs = (
        (az_field, "viridis", azimuth_norm, "Azimuth", "azimuth_deg"),
        (el_field, "coolwarm", elevation_norm, "Elevation", "elevation_deg"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.6), constrained_layout=True)
    for row_idx, (row_key, data, penetrations, row_title) in enumerate(row_defs):
        for col_idx, (field, cmap, norm, panel_title, value_col) in enumerate(col_defs):
            ax = axes[row_idx, col_idx]
            rows, cols = np.nonzero(np.isfinite(field))
            ax.scatter(cols, rows, c=field[rows, cols], cmap=cmap, norm=norm,
                       marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
            ax.scatter(boundary_cols, boundary_rows, s=0.6, color="#343434", zorder=2, rasterized=True)
            dot_size = 4 if row_key == "Allen" else 10
            dot_alpha = 0.5 if row_key == "Allen" else 0.75
            ax.scatter(data.zhuang_col, data.zhuang_row, c=data[value_col], cmap=cmap, norm=norm,
                       s=dot_size, alpha=dot_alpha, linewidths=0, zorder=3, rasterized=True)
            ax.scatter(penetrations.zhuang_col, penetrations.zhuang_row, marker="x", s=70, color="black",
                       linewidths=1.6, zorder=4, label=f"penetration (n={len(penetrations)})")
            ax.legend(fontsize=7, loc="upper right")
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
            colorbar.set_label("degrees")
            ax.set(title=f"{panel_title}\n{row_title}" if col_idx == 0 else panel_title,
                   xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
            ax.set_axisbelow(True)

    fig.suptitle(
        "Allen vs. MouseV2 single units over the same span-matched Zhuang field, same color scale\n"
        "(Allen position is independent anatomy; MouseV2 position is RF-fit -- NOT a symmetric comparison, see module docstring)",
        fontsize=12,
    )
    figure_path = OUTPUT / "Figure_allen_vs_mousev2_units_on_map.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
