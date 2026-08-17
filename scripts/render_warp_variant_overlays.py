#!/usr/bin/env python3
"""Overlay figures for the three warp variants tested in
`fit_per_session_incremental_warp_cv.py` / `fit_per_session_anisotropic_scale_cv.py`:
rotation+offset, scale+offset (anisotropic AP/ML), and rotation+scale+offset -- rendered the
same way as the locked default (`Figure_default_registration_all_cells_over_zhuang.png`).

IMPORTANT: none of these were adopted. The cross-validated tests showed rotation gives a
marginal, borderline held-out improvement (+0.89 deg, below the practical bar) and anisotropic
scale is weaker still (+0.45 deg, not significant), both with signs of overfitting (a negative-
outlier tail for rotation; grid-edge-pinned parameters for scale). The fits here are each
session's own BEST IN-SAMPLE fit (all of that session's cells, no held-out probe) -- i.e. an
upper bound on how good these variants could ever look, not a cross-validated result. They
exist purely to show, visually, what that optimistic upper bound looks like next to the
honestly-validated default -- expect them to look "better" by eye; that is exactly the
overfitting risk the CV work was built to catch, not evidence the warp is real.
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
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_warp_cv"

DTHETA_GRID_DEG = np.linspace(-8, 8, 17)
DSCALE_AXIS_GRID = np.linspace(0.85, 1.15, 9)
MIN_CELLS = 20


def build_interpolators():
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az = smoothed["azimuth_span_matched_deg"]
    el = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(az.shape[0]), np.arange(az.shape[1])
    az_interp = RegularGridInterpolator((row_axis, col_axis), az, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el, bounds_error=False, fill_value=np.nan)
    return smoothed, az_interp, el_interp


def sample(ccf, geometry, dtheta, dscale_ap, dscale_ml, az_interp, el_interp):
    """predicted[k, n, 2] for k joint (dtheta, dscale_ap, dscale_ml) candidates."""
    theta = np.radians(geometry["fitted_rotation_deg"]) + np.atleast_1d(dtheta)
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
    dscale_ap = np.broadcast_to(np.atleast_1d(dscale_ap), theta.shape)
    dscale_ml = np.broadcast_to(np.atleast_1d(dscale_ml), theta.shape)

    delta = ccf - v1_anchor
    delta_ml_ap = delta[:, [1, 0]]
    n_grid = theta.shape[0]
    n_cells = ccf.shape[0]
    predicted = np.full((n_grid, n_cells, 2), np.nan)
    for k in range(n_grid):
        rotation = np.array([[np.cos(theta[k]), -np.sin(theta[k])], [np.sin(theta[k]), np.cos(theta[k])]])
        scale_reflect = np.diag([ml_sign * px_per_mm * dscale_ml[k], px_per_mm * dscale_ap[k]])
        matrix = rotation @ scale_reflect
        xy = delta_ml_ap @ matrix.T + pixel_center
        row_col = xy[:, ::-1]
        predicted[k] = np.column_stack([az_interp(row_col), el_interp(row_col)])
    return predicted


def best_in_sample_fit(ccf, naive_rf, geometry, az_interp, el_interp, dtheta_grid, dscale_ap_grid, dscale_ml_grid):
    theta_mesh, ap_mesh, ml_mesh = np.meshgrid(dtheta_grid, dscale_ap_grid, dscale_ml_grid, indexing="ij")
    dtheta = theta_mesh.ravel()
    dscale_ap = ap_mesh.ravel()
    dscale_ml = ml_mesh.ravel()
    predicted = sample(ccf, geometry, np.radians(dtheta), dscale_ap, dscale_ml, az_interp, el_interp)

    losses = np.full(len(dtheta), np.inf)
    offsets = np.full((len(dtheta), 2), np.nan)
    for k in range(len(dtheta)):
        valid = np.isfinite(predicted[k]).all(axis=1)
        if valid.sum() < MIN_CELLS:
            continue
        offset = huber_location(predicted[k][valid] - naive_rf[valid])
        residual = predicted[k][valid] - offset - naive_rf[valid]
        losses[k] = float(np.median(np.linalg.norm(residual, axis=1)))
        offsets[k] = offset
    best = int(np.argmin(losses))
    return {
        "dtheta_deg": float(dtheta[best]), "dscale_ap": float(dscale_ap[best]), "dscale_ml": float(dscale_ml[best]),
        "offset_az": float(offsets[best, 0]), "offset_el": float(offsets[best, 1]),
        "in_sample_median_error_deg": float(losses[best]),
    }


def fit_variant(cells, geometry, az_interp, el_interp, allow_rotation, allow_scale):
    dtheta_grid = DTHETA_GRID_DEG if allow_rotation else np.array([0.0])
    dscale_grid = DSCALE_AXIS_GRID if allow_scale else np.array([1.0])
    rows = []
    for sid, session_cells in cells.groupby("ecephys_session_id"):
        ccf = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        naive_rf = session_cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
        fit = best_in_sample_fit(ccf, naive_rf, geometry, az_interp, el_interp, dtheta_grid, dscale_grid, dscale_grid)
        fit["ecephys_session_id"] = int(sid)
        fit["n_cells"] = len(session_cells)
        rows.append(fit)
    return pd.DataFrame(rows)


def render_overlay(cells, fits, geometry, smoothed, title, subtitle, output_path):
    merged = cells.merge(fits, on="ecephys_session_id", how="left")

    def ccf_to_pixel_per_row(group):
        theta = np.radians(geometry["fitted_rotation_deg"]) + np.radians(group.dtheta_deg.iloc[0])
        tx, ty = geometry["fitted_translation_px"]
        px_per_mm = geometry["fixed_scale_px_per_mm"]
        ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
        v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
        v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
        pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale_reflect = np.diag([ml_sign * px_per_mm * group.dscale_ml.iloc[0], px_per_mm * group.dscale_ap.iloc[0]])
        matrix = rotation @ scale_reflect
        ccf = group[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        delta = ccf - v1_anchor
        delta_ml_ap = delta[:, [1, 0]]
        return delta_ml_ap @ matrix.T + pixel_center

    az_interp_field = smoothed["azimuth_span_matched_deg"]
    el_interp_field = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(az_interp_field.shape[0]), np.arange(az_interp_field.shape[1])
    az_interp = RegularGridInterpolator((row_axis, col_axis), az_interp_field, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el_interp_field, bounds_error=False, fill_value=np.nan)

    registered_az = np.full(len(merged), np.nan)
    registered_el = np.full(len(merged), np.nan)
    for sid, idx in merged.groupby("ecephys_session_id").groups.items():
        group = merged.loc[idx]
        xy = ccf_to_pixel_per_row(group)
        row_col = xy[:, ::-1]
        pred_az = az_interp(row_col)
        pred_el = el_interp(row_col)
        registered_az[merged.index.get_indexer(idx)] = group.normalized_rf_x + group.offset_az.iloc[0]
        registered_el[merged.index.get_indexer(idx)] = group.normalized_rf_y + group.offset_el.iloc[0]
    merged["registered_azimuth_deg"] = registered_az
    merged["registered_elevation_deg"] = registered_el

    # For the background, use the DEFAULT (population) geometry -- these per-session dtheta/
    # dscale deviations are small and session-specific; the shared background stays fixed so all
    # variants remain visually comparable to the locked default figure.
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

    def pixel_to_ccf(row_col):
        row, col = row_col[:, 0], row_col[:, 1]
        xy = np.column_stack([col, row])
        delta_ml_ap = (xy - pixel_center) @ inverse_matrix.T
        ml = v1_anchor_ml + delta_ml_ap[:, 0]
        ap = v1_anchor_ap + delta_ml_ap[:, 1]
        return np.column_stack([ml, ap])

    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)
    boundary_ccf = pixel_to_ccf(np.column_stack([boundary_rows, boundary_cols]))

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    panels = (
        ("registered_azimuth_deg", az_interp_field, "Azimuth", "viridis", azimuth_norm),
        ("registered_elevation_deg", el_interp_field, "Elevation", "coolwarm", elevation_norm),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), constrained_layout=True)
    for ax, (cell_col, field, panel_title, cmap, norm) in zip(axes, panels):
        rows, cols = np.nonzero(np.isfinite(field))
        bg_ccf = pixel_to_ccf(np.column_stack([rows, cols]))
        ax.scatter(bg_ccf[:, 0], bg_ccf[:, 1], c=field[rows, cols], cmap=cmap, norm=norm,
                   marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
        ax.scatter(boundary_ccf[:, 0], boundary_ccf[:, 1], s=0.6, color="#343434", zorder=2, rasterized=True)
        ax.scatter(merged.ccf_ml_mm, merged.ccf_ap_mm, c=merged[cell_col], cmap=cmap, norm=norm,
                   s=5, alpha=0.55, linewidths=0, zorder=3, rasterized=True)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees")
        ax.set(title=panel_title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)
    import textwrap
    wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=140))
    fig.suptitle(f"{title}\n{wrapped_subtitle}", fontsize=10.5)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    print(output_path)


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    smoothed, az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    variants = (
        ("rotation_offset", True, False, "Rotation + offset (in-sample best fit per session, NOT cross-validated)"),
        ("scale_offset", False, True, "Anisotropic AP/ML scale + offset (in-sample best fit per session, NOT cross-validated)"),
        ("rotation_scale_offset", True, True, "Rotation + anisotropic scale + offset (in-sample best fit per session, NOT cross-validated)"),
    )

    caveat = ("CAVEAT: fit to each session's OWN cells with no held-out probe -- an optimistic upper bound, "
              "not the cross-validated result. CV showed rotation +0.89 deg (below practical bar) and anisotropic "
              "scale +0.45 deg (not significant), both showing overfitting signatures. Default registration "
              "(offset-only) remains the locked/adopted model.")

    summary = {}
    for key, allow_rotation, allow_scale, title in variants:
        print(f"\n=== fitting {key} ===")
        fits = fit_variant(cells, geometry, az_interp, el_interp, allow_rotation, allow_scale)
        fits.to_csv(OUTPUT / f"in_sample_fit_{key}.csv", index=False)
        print(f"median in-sample error: {fits.in_sample_median_error_deg.median():.2f} deg "
              f"(default offset-only in-sample error for comparison printed below)")
        render_overlay(cells, fits, geometry, smoothed, title, caveat,
                        OUTPUT / f"Figure_variant_{key}_over_zhuang.png")
        summary[key] = {"median_in_sample_error_deg": float(fits.in_sample_median_error_deg.median())}

    # baseline (offset only) in-sample error, for reference in the printed comparison
    baseline_fits = fit_variant(cells, geometry, az_interp, el_interp, allow_rotation=False, allow_scale=False)
    print(f"\nbaseline (offset-only) median in-sample error: {baseline_fits.in_sample_median_error_deg.median():.2f} deg")
    summary["offset_only_baseline"] = {"median_in_sample_error_deg": float(baseline_fits.in_sample_median_error_deg.median())}
    (OUTPUT / "in_sample_variant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
