#!/usr/bin/env python3
"""Jointly resolve two interdependent calibrations before locking a default registration:

1. The Zhuang field's value-span gain (elevation/azimuth), which should match the pooled span
   of the RF-offset-corrected cells.
2. The per-session RF offset fit, which samples the (gain-corrected) Zhuang field as its
   target.

These are circular: (1) is computed FROM the per-session-registered cells, but (2) is fit
AGAINST the field that (1) produces. The initial one-shot span-match
(`rescale_zhuang_field_to_naive_span.py`) was calibrated against a single GLOBAL offset, not
per-session offsets; once per-session heterogeneity is allowed
(`fit_per_session_rf_offset_to_zhuang_span_matched.py`), the pooled spread widens (between-
session offset variance is now real signal, not collapsed away), so the gain that was "matched"
under the single-offset regime no longer is (azimuth 0.99->1.13, elevation 1.24->1.39 measured
after one round of per-session fitting). This script alternates the two steps to convergence:
recompute gain from the current per-session-registered cells -> rebuild the span-matched field
-> refit per-session offsets against it -> recompute gain -> ... until the gain stops moving.

Gain is always computed as the IQR ratio against the RAW (unsmoothed-gain, i.e. before any
span-matching) Zhuang field, not the previous iteration's rescaled field, to avoid compounding
drift from repeatedly stretching an already-stretched field.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
DOMAIN_PATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched.npz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
SPAN_MATCH_OUTPUT = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
PER_SESSION_OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"
FINAL_OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"

V1_SEED_ROW_COL = (240, 200)
MIN_VALID_CELLS = 20
DEVIATION_CAP_QUANTILE = 0.85
MAX_ITER = 8
GAIN_TOL = 0.02


def robust_span(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75])
    p5, p95 = np.percentile(values, [5, 95])
    return {"n": int(values.size), "min": float(values.min()), "max": float(values.max()),
            "std": float(values.std()), "iqr": float(q3 - q1), "p5_95_span": float(p95 - p5),
            "median": float(np.median(values))}


def build_ccf_to_pixel(geometry: dict):
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
        delta = ccf - v1_anchor
        delta_ml_ap = delta[:, [1, 0]]
        return delta_ml_ap @ matrix.T + pixel_center

    return ccf_to_pixel


def build_span_matched_field(zhuang_az_raw, zhuang_el_raw, gain_az, gain_el):
    row, col = V1_SEED_ROW_COL
    anchor_az = float(zhuang_az_raw[row, col])
    anchor_el = float(zhuang_el_raw[row, col])
    rescaled_az = np.where(np.isfinite(zhuang_az_raw), anchor_az + gain_az * (zhuang_az_raw - anchor_az), np.nan)
    rescaled_el = np.where(np.isfinite(zhuang_el_raw), anchor_el + gain_el * (zhuang_el_raw - anchor_el), np.nan)
    return rescaled_az.astype(np.float32), rescaled_el.astype(np.float32), anchor_az, anchor_el


def fit_per_session_offsets(cells, azimuth_field, elevation_field, ccf_to_pixel):
    height, width = azimuth_field.shape
    row_axis, col_axis = np.arange(height), np.arange(width)
    azimuth_interp = RegularGridInterpolator((row_axis, col_axis), azimuth_field, bounds_error=False, fill_value=np.nan)
    elevation_interp = RegularGridInterpolator((row_axis, col_axis), elevation_field, bounds_error=False, fill_value=np.nan)

    ccf = cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    xy = ccf_to_pixel(ccf)
    row_col = xy[:, ::-1]
    predicted = np.column_stack([azimuth_interp(row_col), elevation_interp(row_col)])
    naive_rf = cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    residual = predicted - naive_rf
    valid = np.isfinite(residual).all(axis=1)

    global_offset = huber_location(residual[valid])

    rows = []
    for sid, group in cells.groupby("ecephys_session_id"):
        idx = group.index.to_numpy()
        pos = cells.index.get_indexer(idx)
        keep = valid[pos]
        n_valid = int(keep.sum())
        if n_valid < MIN_VALID_CELLS:
            rows.append({"ecephys_session_id": int(sid), "n_cells": len(group), "n_valid": n_valid,
                         "final_offset_az": global_offset[0], "final_offset_el": global_offset[1],
                         "sufficient_support": False, "capped": False, "shrink_factor": 0.0})
            continue
        raw_offset = huber_location(residual[pos][keep])
        deviation = raw_offset - global_offset
        rows.append({"ecephys_session_id": int(sid), "n_cells": len(group), "n_valid": n_valid,
                     "raw_offset_az": raw_offset[0], "raw_offset_el": raw_offset[1],
                     "deviation_az": deviation[0], "deviation_el": deviation[1],
                     "deviation_magnitude_deg": float(np.linalg.norm(deviation)),
                     "sufficient_support": True})
    table = pd.DataFrame(rows)
    supported = table.sufficient_support
    cap_radius = float(np.quantile(table.loc[supported, "deviation_magnitude_deg"], DEVIATION_CAP_QUANTILE))

    def cap_row(row):
        if not row.sufficient_support:
            return pd.Series({"final_offset_az": global_offset[0], "final_offset_el": global_offset[1],
                               "shrink_factor": 0.0, "capped": False})
        magnitude = row.deviation_magnitude_deg
        shrink = min(1.0, cap_radius / magnitude) if magnitude > 0 else 1.0
        final = global_offset + shrink * np.array([row.deviation_az, row.deviation_el])
        return pd.Series({"final_offset_az": final[0], "final_offset_el": final[1],
                           "shrink_factor": shrink, "capped": shrink < 1.0})

    table = pd.concat([table, table.apply(cap_row, axis=1)], axis=1)
    return table, global_offset, cap_radius, valid


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    ccf_to_pixel = build_ccf_to_pixel(geometry)

    domain_patched = {k: v for k, v in np.load(DOMAIN_PATCHED).items()}
    zhuang_az_raw = domain_patched["azimuth_smoothed_for_gradient_deg"]
    zhuang_el_raw = domain_patched["elevation_smoothed_for_gradient_deg"]
    zhuang_az_stats = robust_span(zhuang_az_raw)
    zhuang_el_stats = robust_span(zhuang_el_raw)

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    gain_az, gain_el = 1.0, 1.0  # start from the raw (unscaled) field
    history = []
    for iteration in range(1, MAX_ITER + 1):
        az_field, el_field, anchor_az, anchor_el = build_span_matched_field(zhuang_az_raw, zhuang_el_raw, gain_az, gain_el)
        table, global_offset, cap_radius, valid = fit_per_session_offsets(cells, az_field, el_field, ccf_to_pixel)

        merged = cells.merge(table[["ecephys_session_id", "final_offset_az", "final_offset_el"]],
                              on="ecephys_session_id", how="left")
        registered_az = merged.normalized_rf_x + merged.final_offset_az
        registered_el = merged.normalized_rf_y + merged.final_offset_el
        reg_az_stats = robust_span(registered_az)
        reg_el_stats = robust_span(registered_el)

        new_gain_az = reg_az_stats["iqr"] / zhuang_az_stats["iqr"]
        new_gain_el = reg_el_stats["iqr"] / zhuang_el_stats["iqr"]

        delta_az = abs(new_gain_az - gain_az)
        delta_el = abs(new_gain_el - gain_el)
        print(f"iter {iteration}: gain_az {gain_az:.3f}->{new_gain_az:.3f} (d={delta_az:.3f}), "
              f"gain_el {gain_el:.3f}->{new_gain_el:.3f} (d={delta_el:.3f}), "
              f"pooled_offset=({global_offset[0]:+.1f},{global_offset[1]:+.1f}), "
              f"n_capped={int(table.capped.sum())}/{table.sufficient_support.sum()}")
        history.append({"iteration": iteration, "gain_az": gain_az, "gain_el": gain_el,
                         "new_gain_az": new_gain_az, "new_gain_el": new_gain_el,
                         "pooled_offset_az": float(global_offset[0]), "pooled_offset_el": float(global_offset[1]),
                         "registered_az_iqr": reg_az_stats["iqr"], "registered_el_iqr": reg_el_stats["iqr"]})

        converged = delta_az < GAIN_TOL and delta_el < GAIN_TOL
        gain_az, gain_el = new_gain_az, new_gain_el
        if converged:
            print(f"converged after {iteration} iterations")
            break
    else:
        print(f"WARNING: did not converge within {MAX_ITER} iterations; using last values")

    # Final pass at the converged gains (fields/table above already reflect gain_az/gain_el
    # used to produce them in the last loop body -- rebuild once more explicitly for clarity).
    az_field, el_field, anchor_az, anchor_el = build_span_matched_field(zhuang_az_raw, zhuang_el_raw, gain_az, gain_el)
    table, global_offset, cap_radius, valid = fit_per_session_offsets(cells, az_field, el_field, ccf_to_pixel)
    merged = cells.merge(table[["ecephys_session_id", "final_offset_az", "final_offset_el", "sufficient_support", "capped"]],
                          on="ecephys_session_id", how="left")
    merged["registered_azimuth_deg"] = merged.normalized_rf_x + merged.final_offset_az
    merged["registered_elevation_deg"] = merged.normalized_rf_y + merged.final_offset_el
    final_az_stats = robust_span(merged.registered_azimuth_deg)
    final_el_stats = robust_span(merged.registered_elevation_deg)

    print("\nfinal span check:")
    print(f"  azimuth:   registered IQR={final_az_stats['iqr']:.1f}  zhuang IQR={zhuang_az_stats['iqr']:.1f}  "
          f"ratio={final_az_stats['iqr']/zhuang_az_stats['iqr']:.3f}  gain applied={gain_az:.3f}")
    print(f"  elevation: registered IQR={final_el_stats['iqr']:.1f}  zhuang IQR={zhuang_el_stats['iqr']:.1f}  "
          f"ratio={final_el_stats['iqr']/zhuang_el_stats['iqr']:.3f}  gain applied={gain_el:.3f}")

    # -- write span-matched field --
    domain_patched["azimuth_span_matched_deg"] = az_field
    domain_patched["elevation_span_matched_deg"] = el_field
    span_matched_path = SPAN_MATCH_OUTPUT / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
    np.savez_compressed(span_matched_path, **domain_patched)
    print(f"\nwrote {span_matched_path}")

    span_manifest = {
        "method": "iterative: gain recomputed from per-session-registered cells, per-session offsets refit against the "
                  "resulting field, repeated to convergence",
        "iterations": history,
        "v1_seed_row_col": list(V1_SEED_ROW_COL),
        "final_gain_az": gain_az, "final_gain_el": gain_el,
        "anchor_az_deg": anchor_az, "anchor_el_deg": anchor_el,
        "final_registered_azimuth_stats": final_az_stats,
        "final_registered_elevation_stats": final_el_stats,
        "zhuang_raw_azimuth_stats": zhuang_az_stats,
        "zhuang_raw_elevation_stats": zhuang_el_stats,
    }
    (SPAN_MATCH_OUTPUT / "domain_patched_span_match_manifest.json").write_text(json.dumps(span_manifest, indent=2))

    # -- write per-session offset table --
    PER_SESSION_OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(PER_SESSION_OUTPUT / "per_session_rf_offset.csv", index=False)
    per_session_manifest = {
        "geometry_source": str(GEOMETRY_MANIFEST),
        "field_source": str(span_matched_path),
        "min_valid_cells": MIN_VALID_CELLS,
        "deviation_cap_quantile": DEVIATION_CAP_QUANTILE,
        "deviation_cap_radius_deg": cap_radius,
        "pooled_offset_az_deg": float(global_offset[0]),
        "pooled_offset_el_deg": float(global_offset[1]),
        "n_sessions": int(len(table)),
        "n_sessions_capped": int(table.capped.sum()),
        "n_sessions_insufficient_support": int((~table.sufficient_support).sum()),
        "converged_after_iterations": len(history),
    }
    (PER_SESSION_OUTPUT / "per_session_offset_manifest.json").write_text(json.dumps(per_session_manifest, indent=2))
    print(f"wrote {PER_SESSION_OUTPUT / 'per_session_rf_offset.csv'}")

    # -- convergence trace figure --
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(hist_df.iteration, hist_df.new_gain_az, "o-", label="azimuth gain", color="#2864a8")
    ax.plot(hist_df.iteration, hist_df.new_gain_el, "o-", label="elevation gain", color="#b33f62")
    ax.set(title="Span-match gain convergence across iterations", xlabel="iteration", ylabel="IQR gain")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(SPAN_MATCH_OUTPUT / "Figure_span_match_gain_convergence.png", dpi=160)
    plt.close(fig)

    # -- span comparison histogram (final) --
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, reg_vals, raw_vals, sm_vals, title in (
        (axes[0], merged.registered_azimuth_deg, zhuang_az_raw, az_field, "Azimuth (deg)"),
        (axes[1], merged.registered_elevation_deg, zhuang_el_raw, el_field, "Elevation (deg)"),
    ):
        reg_vals = np.asarray(reg_vals, dtype=np.float64); reg_vals = reg_vals[np.isfinite(reg_vals)]
        raw_vals = np.asarray(raw_vals, dtype=np.float64); raw_vals = raw_vals[np.isfinite(raw_vals)]
        sm_vals = np.asarray(sm_vals, dtype=np.float64); sm_vals = sm_vals[np.isfinite(sm_vals)]
        bins = np.linspace(min(reg_vals.min(), raw_vals.min()), max(reg_vals.max(), raw_vals.max()), 60)
        ax.hist(reg_vals, bins=bins, density=True, histtype="step", linewidth=1.8, color="#222222", label="per-session-registered cells")
        ax.hist(raw_vals, bins=bins, density=True, histtype="step", linewidth=1.4, color="#b33f62", label="Zhuang (raw)")
        ax.hist(sm_vals, bins=bins, density=True, histtype="step", linewidth=1.4, linestyle="--", color="#2864a8", label="Zhuang (span-matched, converged)")
        ax.set(title=title, xlabel="deg", ylabel="density")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Per-session-registered cells vs. Zhuang value-span (converged)", fontsize=12)
    fig.savefig(SPAN_MATCH_OUTPUT / "Figure_naive_vs_zhuang_span_comparison.png", dpi=170)
    plt.close(fig)
    print(SPAN_MATCH_OUTPUT / "Figure_naive_vs_zhuang_span_comparison.png")

    # -- default registration overlay figure --
    boundary = domain_patched["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)
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

    boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))
    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    panels = (
        ("registered_azimuth_deg", az_field, "Azimuth: per-session-registered cells vs. Zhuang (span-matched, converged)", "viridis", azimuth_norm),
        ("registered_elevation_deg", el_field, "Elevation: per-session-registered cells vs. Zhuang (span-matched, converged)", "coolwarm", elevation_norm),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (cell_col, field, title, cmap, norm) in zip(axes, panels):
        rows, cols = np.nonzero(np.isfinite(field))
        bg_ccf = pixel_to_ccf(np.column_stack([rows, cols]))
        ax.scatter(bg_ccf[:, 0], bg_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                   marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
        ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.6, color="#343434", zorder=2, rasterized=True)
        ax.scatter(merged.ccf_ml_mm, merged.ccf_ap_mm, c=merged[cell_col], cmap=cmap, norm=norm,
                   s=5, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees; shared scale, per-session offset applied")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"DEFAULT REGISTRATION: all-session pooled cells (n={len(merged)}) over Zhuang, span-matched (converged gains "
        f"az={gain_az:.2f}, el={gain_el:.2f}),\nper-session RF offset applied ({int(table.capped.sum())}/{len(table)} "
        f"sessions capped), anatomical registration fixed (rotation {geometry['fitted_rotation_deg']:+.1f} deg)",
        fontsize=12,
    )
    figure_path = FINAL_OUTPUT / "Figure_default_registration_all_cells_over_zhuang.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
