#!/usr/bin/env python3
"""Does naive (translation-only, theta=0) lose to a HEAVILY RESTRICTED GLOBAL affine (one small
shared rotation + one small shared isotropic scale, fit ONCE across the whole cohort -- not per
session) under honest leave-one-SESSION-out cross-validation?

Context: per-session offset just failed this same style of held-out test
(`verify_naive_vs_per_session_offset_cv.py`) -- fit from only 2-5 probes per fold, its variance
outweighed any real per-animal signal. A GLOBAL affine is a completely different bias-variance
regime: leaving out one whole session still leaves ~44 sessions (~14,500+ cells) to constrain
just 2 extra parameters (rotation, isotropic scale), vastly more data per free parameter than
any per-session fit had. This also finally cross-validates, on the SAME RF-vector-error metric
used throughout, the -8.1 deg population rotation that has been baked into the locked geometry
since `fit_translation_rotation_naive_to_zhuang.py` -- which was only ever validated via area-
compartment agreement, never via held-out RF error.

"Heavily restricted": theta in +/-15 deg (comfortably contains the historical -8.1 deg optimum,
found originally with much looser +/-30 deg bounds), isotropic scale in [0.85, 1.15]. Offset is
refit fresh (closed-form Huber location) for every candidate, exactly as throughout. Training
loss is the pooled median residual norm over all training cells (no per-session weighting needed
here -- the largest session is ~4% of pooled training cells, session count imbalance is mild
enough not to need the per-unit aggregation the per-probe tests required).
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
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wilcoxon

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

DTHETA_GRID_DEG = np.linspace(-15, 15, 15)
DSCALE_GRID = np.linspace(0.85, 1.15, 15)


def build_interpolators():
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az = smoothed["azimuth_span_matched_deg"]
    el = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(az.shape[0]), np.arange(az.shape[1])
    az_interp = RegularGridInterpolator((row_axis, col_axis), az, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el, bounds_error=False, fill_value=np.nan)
    return az_interp, el_interp


def sample_grid(ccf, geometry, dtheta_deg, dscale, az_interp, el_interp):
    """predicted[k, n, 2] using the GLOBAL (population) geometry's own placement, with dtheta
    (deg, added to the fixed rotation) and dscale (multiplicative on the fixed scale) applied --
    NOT anchored per-session, one shared transform for all cells at each grid point."""
    theta = np.radians(geometry["fitted_rotation_deg"]) + np.radians(dtheta_deg)
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
    dscale = np.broadcast_to(np.atleast_1d(dscale), theta.shape)

    delta = ccf - v1_anchor
    delta_ml_ap = delta[:, [1, 0]]
    n_grid = theta.shape[0]
    n_cells = ccf.shape[0]
    predicted = np.full((n_grid, n_cells, 2), np.nan)
    for k in range(n_grid):
        rotation = np.array([[np.cos(theta[k]), -np.sin(theta[k])], [np.sin(theta[k]), np.cos(theta[k])]])
        scale_reflect = np.diag([ml_sign * px_per_mm * dscale[k], px_per_mm * dscale[k]])
        matrix = rotation @ scale_reflect
        xy = delta_ml_ap @ matrix.T + pixel_center
        row_col = xy[:, ::-1]
        predicted[k] = np.column_stack([az_interp(row_col), el_interp(row_col)])
    return predicted


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    ccf = cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    naive_rf = cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    session_ids = cells.ecephys_session_id.to_numpy()

    theta_mesh, scale_mesh = np.meshgrid(DTHETA_GRID_DEG, DSCALE_GRID, indexing="ij")
    dtheta_deg = theta_mesh.ravel()
    dscale = scale_mesh.ravel()
    baseline_idx = int(np.argmin(np.abs(dtheta_deg) + np.abs(dscale - 1.0)))
    print(f"precomputing predicted values for {len(dtheta_deg)} grid points x {len(cells)} cells...")
    predicted = sample_grid(ccf, geometry, dtheta_deg, dscale, az_interp, el_interp)
    valid = np.isfinite(predicted).all(axis=2)  # (n_grid, n_cells)
    print("done")

    rows = []
    for held_session in cells.ecephys_session_id.unique():
        held_mask = session_ids == held_session
        train_mask = ~held_mask
        if held_mask.sum() < 5:
            continue

        losses = np.full(len(dtheta_deg), np.inf)
        offsets = np.full((len(dtheta_deg), 2), np.nan)
        for k in range(len(dtheta_deg)):
            keep_train = valid[k] & train_mask
            if keep_train.sum() < 100:
                continue
            offset = huber_location(predicted[k][keep_train] - naive_rf[keep_train])
            residual = predicted[k][keep_train] - offset - naive_rf[keep_train]
            losses[k] = float(np.median(np.linalg.norm(residual, axis=1)))
            offsets[k] = offset
        best_idx = int(np.argmin(losses))

        def held_score(idx):
            keep_held = valid[idx] & held_mask
            if keep_held.sum() == 0:
                return np.nan
            residual = predicted[idx][keep_held] - offsets[idx] - naive_rf[keep_held]
            return float(np.median(np.linalg.norm(residual, axis=1)))

        naive_error = held_score(baseline_idx)
        affine_error = held_score(best_idx)
        if not (np.isfinite(naive_error) and np.isfinite(affine_error)):
            continue
        rows.append({
            "ecephys_session_id": int(held_session), "n_held_cells": int(held_mask.sum()),
            "naive_held_error_deg": naive_error, "affine_held_error_deg": affine_error,
            "chosen_dtheta_deg": float(dtheta_deg[best_idx]), "chosen_dscale": float(dscale[best_idx]),
            "improvement_deg": naive_error - affine_error,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "cv_naive_vs_global_restricted_affine.csv", index=False)

    delta = table.improvement_deg.to_numpy()
    median_improvement = float(np.median(delta))
    frac_improved = float((delta > 0).mean())
    stat, p_value = wilcoxon(delta)
    print(f"folds={len(table)}  median_improvement={median_improvement:+.2f} deg  "
          f"frac_folds_improved={frac_improved:.1%}  wilcoxon_p={p_value:.4g}")
    print(f"naive median held-out error:  {table.naive_held_error_deg.median():.2f} deg")
    print(f"affine median held-out error: {table.affine_held_error_deg.median():.2f} deg")
    print(f"chosen dtheta: median={table.chosen_dtheta_deg.median():+.1f} deg, "
          f"IQR=[{table.chosen_dtheta_deg.quantile(.25):+.1f}, {table.chosen_dtheta_deg.quantile(.75):+.1f}]")
    print(f"chosen dscale: median={table.chosen_dscale.median():.3f}x, "
          f"IQR=[{table.chosen_dscale.quantile(.25):.3f}, {table.chosen_dscale.quantile(.75):.3f}]")

    summary = {
        "n_folds": len(table), "median_improvement_deg": median_improvement,
        "fraction_folds_improved": frac_improved, "wilcoxon_p": float(p_value),
        "naive_median_held_error_deg": float(table.naive_held_error_deg.median()),
        "affine_median_held_error_deg": float(table.affine_held_error_deg.median()),
        "dtheta_grid_deg": [float(DTHETA_GRID_DEG.min()), float(DTHETA_GRID_DEG.max())],
        "dscale_grid": [float(DSCALE_GRID.min()), float(DSCALE_GRID.max())],
        "chosen_dtheta_median_deg": float(table.chosen_dtheta_deg.median()),
        "chosen_dscale_median": float(table.chosen_dscale.median()),
    }
    (OUTPUT / "naive_vs_global_restricted_affine_cv_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    ax = axes[0, 0]
    ax.hist(delta, bins=20, color="#2864a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(median_improvement, color="#b33f62", linewidth=1.5, linestyle="--", label=f"median={median_improvement:+.2f} deg")
    ax.set(title=f"Held-out improvement per session (n={len(table)})\nwilcoxon p={p_value:.3g}, {frac_improved:.0%} sessions improved",
           xlabel="naive_error - affine_error (deg)", ylabel="sessions")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.scatter(table.naive_held_error_deg, table.affine_held_error_deg, s=22, alpha=0.6, color="#2864a8")
    lo = 0
    hi = max(table.naive_held_error_deg.max(), table.affine_held_error_deg.max())
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1, linestyle="--", label="y=x (no change)")
    ax.set(title="Held-out error: naive vs. restricted affine\n(points below the line = affine better)",
           xlabel="naive held-out error (deg)", ylabel="affine held-out error (deg)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    ax.hist(table.chosen_dtheta_deg, bins=DTHETA_GRID_DEG, color="#5f8f3e", alpha=0.85)
    ax.axvline(geometry["fitted_rotation_deg"] * 0, color="grey", linewidth=1, linestyle=":")
    ax.set(title="Chosen global rotation per session left out\n(historical population fit was -8.1 deg from theta=0)",
           xlabel="chosen dtheta (deg)", ylabel="sessions")

    ax = axes[1, 1]
    ax.hist(table.chosen_dscale, bins=DSCALE_GRID, color="#d78318", alpha=0.85)
    ax.set(title="Chosen global isotropic scale per session left out",
           xlabel="chosen dscale (x)", ylabel="sessions")

    fig.suptitle(
        f"Naive (translation-only) vs. heavily restricted GLOBAL affine "
        f"(theta +/-15deg, scale 0.85-1.15x) -- leave-one-session-out CV",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figure_path = OUTPUT / "Figure_naive_vs_global_restricted_affine_cv.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
