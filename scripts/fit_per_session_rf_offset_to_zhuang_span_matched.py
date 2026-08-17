#!/usr/bin/env python3
"""Per-session RF offset (gaze/eye-position translation) against the canonical, span-matched
Zhuang atlas -- `rescale_zhuang_field_to_naive_span.py`'s span-matched field is now the map we
register against, per the user's "these are the maps we should try to register against."

Design: the anatomical registration (rotation, pixel-space translation, scale, reflection --
i.e. the CCF-to-atlas-pixel coordinate frame) is a one-time calibration common to all sessions
(fixed at `fit_translation_rotation_naive_to_zhuang.py`'s fitted values; CCF is already a shared
anatomical frame across animals, so there is no reason this part should vary per session). What
genuinely varies per session/animal is the RF offset itself -- absolute gaze/eye-position -- so
ONLY that is fit per session, exactly the original project goal (a per-session translation
delta_s that should agree across V1/HVA/LGd) but anchored to an external atlas instead of an
internal EM fixed point.

Because the fitted rotation/translation were optimized against the raw (unsmoothed, un-span-
matched) Zhuang interpolation, and the offset we want is under the NEW canonical span-matched
field, the population offset is recomputed here (pooled Huber location across all sessions
under the fixed geometry + span-matched field) rather than reused from that manifest.

Guard against "overly dramatic deviations per session" (explicit user instruction): each
session's raw per-session offset is a robust (Huber) location estimate, which already resists
within-session outlier cells, but a session whose own probes happen to sample an atypical
patch of the map (or land near a Zhuang domain edge) can still produce a large apparent
deviation from the population offset that is more likely a registration/geometry artifact than
real between-animal gaze variance. So each session's deviation from the pooled offset is
additionally capped in magnitude at a data-driven quantile of the observed deviation
distribution (direction preserved, magnitude clipped) -- a soft Huber-style cap, not a silent
discard.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402

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
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"

MIN_VALID_CELLS = 20
DEVIATION_CAP_QUANTILE = 0.85  # cap per-session deviation magnitude at this quantile of the observed spread


def build_geometry(geometry: dict):
    theta = np.radians(geometry["fitted_rotation_deg"])
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]

    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale_reflect = np.diag([ml_sign * px_per_mm, px_per_mm])
    matrix = rotation @ scale_reflect
    pixel_center = np.array([v1_seed_col, v1_seed_row]) + np.array([tx, ty])

    def ccf_to_pixel(ccf: np.ndarray) -> np.ndarray:
        delta = ccf - v1_anchor  # columns [ap, ml]
        delta_ml_ap = delta[:, [1, 0]]
        return delta_ml_ap @ matrix.T + pixel_center

    return ccf_to_pixel


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    ccf_to_pixel = build_geometry(geometry)

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    azimuth_field = smoothed["azimuth_span_matched_deg"]
    elevation_field = smoothed["elevation_span_matched_deg"]
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    height, width = azimuth_field.shape
    row_axis = np.arange(height)
    col_axis = np.arange(width)
    azimuth_interp = RegularGridInterpolator((row_axis, col_axis), azimuth_field, bounds_error=False, fill_value=np.nan)
    elevation_interp = RegularGridInterpolator((row_axis, col_axis), elevation_field, bounds_error=False, fill_value=np.nan)

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    ccf = cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    xy = ccf_to_pixel(ccf)
    row_col = xy[:, ::-1]
    predicted_az = azimuth_interp(row_col)
    predicted_el = elevation_interp(row_col)
    predicted = np.column_stack([predicted_az, predicted_el])
    naive_rf = cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    residual = predicted - naive_rf
    valid = np.isfinite(residual).all(axis=1)
    cells["valid"] = valid
    cells["pixel_row"] = row_col[:, 0]
    cells["pixel_col"] = row_col[:, 1]
    cells["predicted_azimuth_deg"] = predicted_az
    cells["predicted_elevation_deg"] = predicted_el
    print(f"cells valid inside span-matched domain: {valid.sum()}/{len(cells)} ({valid.mean():.1%})")

    global_offset = huber_location(residual[valid])
    print(f"pooled (population) offset under canonical map: az={global_offset[0]:+.1f}, el={global_offset[1]:+.1f} deg")

    rows = []
    raw_deviations = []
    for sid, group in cells.groupby("ecephys_session_id"):
        idx = group.index.to_numpy()
        keep = valid[idx]
        n_valid = int(keep.sum())
        if n_valid < MIN_VALID_CELLS:
            rows.append({"ecephys_session_id": int(sid), "n_cells": len(group), "n_valid": n_valid,
                         "raw_offset_az": np.nan, "raw_offset_el": np.nan, "sufficient_support": False})
            continue
        session_residual = residual[idx][keep]
        raw_offset = huber_location(session_residual)
        deviation = raw_offset - global_offset
        raw_deviations.append(deviation)
        rows.append({
            "ecephys_session_id": int(sid), "n_cells": len(group), "n_valid": n_valid,
            "raw_offset_az": raw_offset[0], "raw_offset_el": raw_offset[1],
            "deviation_az": deviation[0], "deviation_el": deviation[1],
            "deviation_magnitude_deg": float(np.linalg.norm(deviation)),
            "sufficient_support": True,
        })

    table = pd.DataFrame(rows)
    raw_deviations = np.array(raw_deviations)
    cap_radius = float(np.quantile(table.loc[table.sufficient_support, "deviation_magnitude_deg"], DEVIATION_CAP_QUANTILE))
    print(f"deviation magnitude distribution (deg): "
          f"median={table.deviation_magnitude_deg.median():.1f}, "
          f"p75={table.deviation_magnitude_deg.quantile(.75):.1f}, "
          f"p{int(DEVIATION_CAP_QUANTILE*100)}={cap_radius:.1f}, "
          f"max={table.deviation_magnitude_deg.max():.1f}")

    def cap_and_apply(row):
        if not row.sufficient_support:
            return pd.Series({"final_offset_az": global_offset[0], "final_offset_el": global_offset[1],
                               "shrink_factor": 0.0, "capped": False})
        magnitude = row.deviation_magnitude_deg
        shrink_factor = min(1.0, cap_radius / magnitude) if magnitude > 0 else 1.0
        final = global_offset + shrink_factor * np.array([row.deviation_az, row.deviation_el])
        return pd.Series({"final_offset_az": final[0], "final_offset_el": final[1],
                           "shrink_factor": shrink_factor, "capped": shrink_factor < 1.0})

    table = pd.concat([table, table.apply(cap_and_apply, axis=1)], axis=1)
    n_capped = int(table.capped.sum())
    print(f"sessions capped (deviation magnitude > p{int(DEVIATION_CAP_QUANTILE*100)}={cap_radius:.1f} deg): {n_capped}/{table.sufficient_support.sum()}")
    print(f"sessions with insufficient support (< {MIN_VALID_CELLS} valid cells), forced to pooled offset: "
          f"{int((~table.sufficient_support).sum())}")

    manifest = {
        "geometry_source": str(GEOMETRY_MANIFEST),
        "field_source": str(ZHUANG_SPAN_MATCHED),
        "min_valid_cells": MIN_VALID_CELLS,
        "deviation_cap_quantile": DEVIATION_CAP_QUANTILE,
        "deviation_cap_radius_deg": cap_radius,
        "pooled_offset_az_deg": float(global_offset[0]),
        "pooled_offset_el_deg": float(global_offset[1]),
        "n_sessions": int(len(table)),
        "n_sessions_capped": n_capped,
        "n_sessions_insufficient_support": int((~table.sufficient_support).sum()),
    }
    (OUTPUT / "per_session_offset_manifest.json").write_text(json.dumps(manifest, indent=2))
    table.to_csv(OUTPUT / "per_session_rf_offset.csv", index=False)
    print(f"\nwrote {OUTPUT / 'per_session_rf_offset.csv'}")

    # Summary figure: raw vs. capped per-session offsets around the pooled value.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    supported = table.loc[table.sufficient_support]
    for _, row in supported.iterrows():
        ax.plot([row.raw_offset_az, row.final_offset_az], [row.raw_offset_el, row.final_offset_el],
                color="#b33f62" if row.capped else "#999999", linewidth=0.8, alpha=0.6, zorder=1)
    ax.scatter(supported.raw_offset_az, supported.raw_offset_el, s=22, facecolors="none",
               edgecolors="#b33f62", linewidths=1.0, label="raw per-session (Huber)", zorder=2)
    ax.scatter(supported.final_offset_az, supported.final_offset_el, s=26, color="#2864a8",
               label="final (capped)", zorder=3)
    ax.scatter([global_offset[0]], [global_offset[1]], s=140, marker="*", color="black",
               label="pooled offset", zorder=4)
    circle = plt.Circle(global_offset, cap_radius, fill=False, linestyle="--", color="#555555", linewidth=1.0)
    ax.add_patch(circle)
    ax.set(title=f"Per-session RF offset vs. pooled offset\n"
                 f"cap radius = p{int(DEVIATION_CAP_QUANTILE*100)} of deviation magnitude = {cap_radius:.1f} deg "
                 f"({n_capped}/{len(supported)} sessions capped)",
           xlabel="azimuth offset (deg)", ylabel="elevation offset (deg)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_per_session_offset_summary.png", dpi=170)
    plt.close(fig)
    print(OUTPUT / "Figure_per_session_offset_summary.png")

    # Multi-page PDF: one page per session, cells over the shared span-matched background,
    # annotated with raw vs. final (capped) offset -- same overlay style used throughout.
    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    rows_bg, cols_bg = np.nonzero(np.isfinite(azimuth_field))
    boundary_rows, boundary_cols = np.nonzero(boundary)

    def pixel_to_ccf(row_col_pts: np.ndarray) -> np.ndarray:
        v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
        v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
        theta = np.radians(geometry["fitted_rotation_deg"])
        tx, ty = geometry["fitted_translation_px"]
        px_per_mm = geometry["fixed_scale_px_per_mm"]
        ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale_reflect = np.diag([ml_sign * px_per_mm, px_per_mm])
        matrix = rotation @ scale_reflect
        inverse_matrix = np.linalg.inv(matrix)
        pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
        row, col = row_col_pts[:, 0], row_col_pts[:, 1]
        xy_pts = np.column_stack([col, row])
        delta_ml_ap = (xy_pts - pixel_center) @ inverse_matrix.T
        ml = v1_anchor[1] + delta_ml_ap[:, 0]
        ap = v1_anchor[0] + delta_ml_ap[:, 1]
        return np.column_stack([ml, ap])

    bg_ccf = pixel_to_ccf(np.column_stack([rows_bg, cols_bg]))
    boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))

    with PdfPages(OUTPUT / "Figure_per_session_registration_pages.pdf") as pdf:
        for sid, group in cells.groupby("ecephys_session_id"):
            info = table.loc[table.ecephys_session_id == sid].iloc[0]
            fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
            for ax, field, naive_col, anchor_offset, title, cmap, norm in (
                (axes[0], azimuth_field, "normalized_rf_x", info.final_offset_az, "Azimuth", "viridis", azimuth_norm),
                (axes[1], elevation_field, "normalized_rf_y", info.final_offset_el, "Elevation", "coolwarm", elevation_norm),
            ):
                ax.scatter(bg_ccf[:, 0], bg_ccf[:, 1], c=field[rows_bg, cols_bg], cmap=cmap, norm=norm,
                           marker="s", s=1.2, alpha=0.4, linewidths=0, zorder=1, rasterized=True)
                ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.5, color="#343434", zorder=2, rasterized=True)
                corrected = group[naive_col] + anchor_offset
                ax.scatter(group.ccf_ml_mm, group.ccf_ap_mm, c=corrected, cmap=cmap, norm=norm,
                           s=16, edgecolors="black", linewidths=0.3, alpha=0.9, zorder=3)
                ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="box")
            support_text = "sufficient" if info.sufficient_support else "insufficient (forced to pooled)"
            cap_text = "capped" if info.get("capped", False) else "uncapped"
            fig.suptitle(
                f"Session {int(sid)} (n={info.n_cells} cells, {info.n_valid} valid, {support_text})\n"
                f"raw offset=({info.get('raw_offset_az', np.nan):+.1f},{info.get('raw_offset_el', np.nan):+.1f}) deg, "
                f"final offset=({info.final_offset_az:+.1f},{info.final_offset_el:+.1f}) deg [{cap_text}], "
                f"shrink={info.shrink_factor:.2f}",
                fontsize=11.5,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(OUTPUT / "Figure_per_session_registration_pages.pdf")


if __name__ == "__main__":
    main()
