#!/usr/bin/env python3
"""Test whether per-session/animal anatomical warping beyond a rigid RF offset is actually
supported by the data, before allowing it.

User's explicit design constraint: warping should be regularized/minimal, not an unconstrained
per-session fit -- and given how few independent landmarks each session has (3-6 probes), it
may not be identifiable at all. So this does NOT fit warp parameters by in-sample goodness of
fit (which would happily overfit 1-3 extra parameters to 3-6 points). Instead it adds ONE
degree of freedom at a time -- first a small per-session rotation deviation, then (only if that
helps) a small per-session uniform scale deviation -- each searched over a small, "smooth" grid
around zero, selected by in-sample training loss WITHIN a leave-one-probe-out cross-validation
fold, and adoption of each new DOF is decided by whether it improves HELD-OUT probe error across
the whole cohort (paired Wilcoxon signed-rank test + a minimum practical-effect-size gate, not
p-value alone -- with ~227 total folds even a trivial effect can be "significant"). The
procedure stops at the first DOF that fails to clear both bars, on the expectation (stated by
the user) that there may not be enough data to support any warping beyond the rigid per-session
offset already locked in `DEFAULT_REGISTRATION_README.md`.

Fixed throughout: anatomical translation (tx,ty), reflection -- these are the shared coordinate-
frame calibration, not something that should vary per animal. RF offset is refit per fold from
training probes only (closed-form Huber location), exactly as in the locked default.
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
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_warp_cv"

MIN_TRAIN_CELLS = 15
DTHETA_GRID_DEG = np.linspace(-8, 8, 17)
DSCALE_GRID = np.linspace(0.85, 1.15, 13)
MIN_PRACTICAL_IMPROVEMENT_DEG = 1.0
ALPHA = 0.05


def build_interpolators():
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az = smoothed["azimuth_span_matched_deg"]
    el = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(az.shape[0]), np.arange(az.shape[1])
    az_interp = RegularGridInterpolator((row_axis, col_axis), az, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el, bounds_error=False, fill_value=np.nan)
    return az_interp, el_interp


def sample_predicted(ccf, geometry, dtheta, dscale, az_interp, el_interp):
    """predicted[k, n, 2] for k grid values (dtheta or dscale, whichever varies), n cells."""
    theta = np.radians(geometry["fitted_rotation_deg"]) + dtheta  # broadcastable array
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"] * dscale
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])

    delta = ccf - v1_anchor
    delta_ml_ap = delta[:, [1, 0]]  # (n, 2) -> [ml, ap]

    theta = np.atleast_1d(theta)
    px_per_mm = np.atleast_1d(np.broadcast_to(px_per_mm, theta.shape))
    n_grid = theta.shape[0]
    n_cells = ccf.shape[0]
    predicted = np.full((n_grid, n_cells, 2), np.nan)
    for k in range(n_grid):
        rotation = np.array([[np.cos(theta[k]), -np.sin(theta[k])], [np.sin(theta[k]), np.cos(theta[k])]])
        scale_reflect = np.diag([ml_sign * px_per_mm[k], px_per_mm[k]])
        matrix = rotation @ scale_reflect
        xy = delta_ml_ap @ matrix.T + pixel_center
        row_col = xy[:, ::-1]
        predicted[k] = np.column_stack([az_interp(row_col), el_interp(row_col)])
    return predicted


def evaluate_dof(cells, geometry, az_interp, el_interp, grid, vary, base_dtheta=0.0, base_dscale=1.0):
    """Leave-one-probe-out CV for one incremental degree of freedom ('rotation' or 'scale').
    Returns a DataFrame with one row per (session, held-out probe) fold: baseline (grid value =
    0 deviation) held-out error vs. best-candidate held-out error, plus the chosen deviation.
    """
    rows = []
    for sid, session_cells in cells.groupby("ecephys_session_id"):
        ccf = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        naive_rf = session_cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
        probes = session_cells["ecephys_probe_id"].to_numpy()
        unique_probes = np.unique(probes)
        if len(unique_probes) < 3:
            continue

        if vary == "rotation":
            predicted = sample_predicted(ccf, geometry, base_dtheta + np.radians(grid), base_dscale, az_interp, el_interp)
        else:
            predicted = sample_predicted(ccf, geometry, base_dtheta, base_dscale * grid, az_interp, el_interp)
        baseline_idx = int(np.argmin(np.abs(grid)))  # grid value closest to 0 = no added deviation

        for held_probe in unique_probes:
            train_mask = probes != held_probe
            held_mask = probes == held_probe
            if train_mask.sum() < MIN_TRAIN_CELLS or held_mask.sum() < 5:
                continue

            def fold_score(grid_idx):
                offset = huber_location(predicted[grid_idx][train_mask] - naive_rf[train_mask])
                held_residual = predicted[grid_idx][held_mask] - offset - naive_rf[held_mask]
                valid = np.isfinite(held_residual).all(axis=1)
                if valid.sum() == 0:
                    return np.nan, offset
                return float(np.median(np.linalg.norm(held_residual[valid], axis=1))), offset

            def train_loss(grid_idx):
                offset = huber_location(predicted[grid_idx][train_mask] - naive_rf[train_mask])
                train_residual = predicted[grid_idx][train_mask] - offset - naive_rf[train_mask]
                per_probe = []
                for probe in unique_probes:
                    if probe == held_probe:
                        continue
                    pm = probes[train_mask] == probe
                    sub = train_residual[pm]
                    valid = np.isfinite(sub).all(axis=1)
                    if valid.sum() > 0:
                        per_probe.append(np.median(np.linalg.norm(sub[valid], axis=1)))
                return float(np.median(per_probe)) if per_probe else np.inf

            losses = np.array([train_loss(k) for k in range(len(grid))])
            best_idx = int(np.argmin(losses))
            baseline_score, _ = fold_score(baseline_idx)
            candidate_score, _ = fold_score(best_idx)
            if not (np.isfinite(baseline_score) and np.isfinite(candidate_score)):
                continue
            rows.append({
                "ecephys_session_id": int(sid), "held_probe": int(held_probe),
                "n_train_cells": int(train_mask.sum()), "n_held_cells": int(held_mask.sum()),
                "baseline_held_error_deg": baseline_score, "candidate_held_error_deg": candidate_score,
                "chosen_deviation": float(grid[best_idx]), "improvement_deg": baseline_score - candidate_score,
            })
    return pd.DataFrame(rows)


def decide(name, table):
    delta = table.improvement_deg.to_numpy()
    median_improvement = float(np.median(delta))
    frac_improved = float((delta > 0).mean())
    try:
        stat, p_value = wilcoxon(delta)
    except ValueError:
        p_value = 1.0
    adopt = median_improvement > MIN_PRACTICAL_IMPROVEMENT_DEG and p_value < ALPHA
    print(f"\n[{name}] folds={len(table)}  median_improvement={median_improvement:+.2f} deg  "
          f"frac_folds_improved={frac_improved:.1%}  wilcoxon_p={p_value:.4g}  "
          f"ADOPT={adopt} (needs median>{MIN_PRACTICAL_IMPROVEMENT_DEG} deg AND p<{ALPHA})")
    return adopt, median_improvement, frac_improved, float(p_value)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    summary = {}

    print("=== Step 1: per-session rotation deviation (grid +/-8 deg) ===")
    rotation_table = evaluate_dof(cells, geometry, az_interp, el_interp, DTHETA_GRID_DEG, vary="rotation")
    rotation_table.to_csv(OUTPUT / "cv_rotation_step.csv", index=False)
    adopt_rotation, med_imp_r, frac_r, p_r = decide("rotation", rotation_table)
    summary["rotation"] = {"adopted": adopt_rotation, "median_improvement_deg": med_imp_r,
                            "fraction_folds_improved": frac_r, "wilcoxon_p": p_r, "n_folds": len(rotation_table)}

    if adopt_rotation:
        print("\nrotation adopted -- proceeding to test scale on top of it")
        print("=== Step 2: per-session scale deviation (grid 0.85-1.15x), on top of rotation ===")
        # use each fold's own chosen rotation as the new base (approx: use per-session median chosen rotation)
        session_dtheta = rotation_table.groupby("ecephys_session_id").chosen_deviation.median()
        rows = []
        for sid, base_dtheta in session_dtheta.items():
            session_cells = cells.loc[cells.ecephys_session_id == sid]
            sub_table = evaluate_dof(session_cells, geometry, az_interp, el_interp, DSCALE_GRID, vary="scale",
                                      base_dtheta=np.radians(base_dtheta))
            rows.append(sub_table)
        scale_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        scale_table.to_csv(OUTPUT / "cv_scale_step.csv", index=False)
        if len(scale_table):
            adopt_scale, med_imp_s, frac_s, p_s = decide("scale | rotation", scale_table)
            summary["scale_given_rotation"] = {"adopted": adopt_scale, "median_improvement_deg": med_imp_s,
                                                "fraction_folds_improved": frac_s, "wilcoxon_p": p_s, "n_folds": len(scale_table)}
        else:
            print("no valid folds for scale step")
            summary["scale_given_rotation"] = {"adopted": False, "note": "no valid folds"}
    else:
        print("\nrotation NOT adopted -- stopping here. Final model stays the locked rigid per-session offset "
              "(no per-session anatomical warping); not enough independent landmarks per session to support it.")

    (OUTPUT / "incremental_warp_decision.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT / 'incremental_warp_decision.json'}")

    # figure: distribution of per-fold improvement for the rotation step (the one everyone gets)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(rotation_table.improvement_deg, bins=30, color="#2864a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(rotation_table.improvement_deg.median(), color="#b33f62", linewidth=1.5, linestyle="--",
               label=f"median={rotation_table.improvement_deg.median():+.2f} deg")
    ax.set(title="Held-out improvement from allowing per-session rotation\n(positive = rotation helps)",
           xlabel="baseline_error - candidate_error (deg)", ylabel="folds (session x held-out probe)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_rotation_cv_improvement.png", dpi=160)
    plt.close(fig)
    print(OUTPUT / "Figure_rotation_cv_improvement.png")


if __name__ == "__main__":
    main()
