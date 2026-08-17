#!/usr/bin/env python3
"""Test whether a per-session/animal ANISOTROPIC scale deviation (separate AP vs. ML gain,
e.g. from real skull/cortex-size anisotropy between animals) is supported by held-out data --
same leave-one-probe-out discipline as `fit_per_session_incremental_warp_cv.py`'s rotation
test, which was NOT adopted (median held-out improvement +0.89 deg, below the 1.0 deg practical
bar, with an overfitting-flavored negative-outlier tail). Isotropic scale was never reached
because the stepwise procedure stops at the first failed increment; anisotropic scale is tested
here as an independent candidate directly on top of the rigid default (rotation deviation fixed
at 0), not nested under the rejected rotation step.

Leave-one-probe-out (not a half-of-every-probe split) is used deliberately: cells within a
probe are at nearly the same CCF position and highly spatially correlated, so a within-probe
split would let a candidate "pass" by matching each probe's own local structure rather than by
generalizing to an unseen anatomical location -- inflating apparent held-out performance for
exactly the overfitting failure mode this whole procedure exists to catch.
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
DSCALE_AXIS_GRID = np.linspace(0.85, 1.15, 9)  # per-axis candidates; joint grid is the outer product
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


def sample_predicted_grid(ccf, geometry, dscale_ap, dscale_ml, az_interp, el_interp):
    """predicted[k, n, 2] for k joint (dscale_ap[k], dscale_ml[k]) candidates, n cells.
    Anisotropic scale is applied in CCF-aligned (ML, AP) space, before the fixed population
    rotation -- i.e. it models animal-to-animal AP/ML size differences, independent of the
    shared anatomy-to-atlas-pixel orientation.
    """
    theta = np.radians(geometry["fitted_rotation_deg"])
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    delta = ccf - v1_anchor
    delta_ml_ap = delta[:, [1, 0]]

    n_grid = len(dscale_ap)
    n_cells = ccf.shape[0]
    predicted = np.full((n_grid, n_cells, 2), np.nan)
    for k in range(n_grid):
        scale_reflect = np.diag([ml_sign * px_per_mm * dscale_ml[k], px_per_mm * dscale_ap[k]])
        matrix = rotation @ scale_reflect
        xy = delta_ml_ap @ matrix.T + pixel_center
        row_col = xy[:, ::-1]
        predicted[k] = np.column_stack([az_interp(row_col), el_interp(row_col)])
    return predicted


def evaluate_anisotropic_scale(cells, geometry, az_interp, el_interp, dscale_ap_grid, dscale_ml_grid):
    ap_mesh, ml_mesh = np.meshgrid(dscale_ap_grid, dscale_ml_grid, indexing="ij")
    dscale_ap = ap_mesh.ravel()
    dscale_ml = ml_mesh.ravel()
    baseline_idx = int(np.argmin(np.abs(dscale_ap - 1.0) + np.abs(dscale_ml - 1.0)))

    rows = []
    for sid, session_cells in cells.groupby("ecephys_session_id"):
        ccf = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        naive_rf = session_cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
        probes = session_cells["ecephys_probe_id"].to_numpy()
        unique_probes = np.unique(probes)
        if len(unique_probes) < 3:
            continue

        predicted = sample_predicted_grid(ccf, geometry, dscale_ap, dscale_ml, az_interp, el_interp)

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
                    return np.nan
                return float(np.median(np.linalg.norm(held_residual[valid], axis=1)))

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

            losses = np.array([train_loss(k) for k in range(len(dscale_ap))])
            best_idx = int(np.argmin(losses))
            baseline_score = fold_score(baseline_idx)
            candidate_score = fold_score(best_idx)
            if not (np.isfinite(baseline_score) and np.isfinite(candidate_score)):
                continue
            rows.append({
                "ecephys_session_id": int(sid), "held_probe": int(held_probe),
                "n_train_cells": int(train_mask.sum()), "n_held_cells": int(held_mask.sum()),
                "baseline_held_error_deg": baseline_score, "candidate_held_error_deg": candidate_score,
                "chosen_dscale_ap": float(dscale_ap[best_idx]), "chosen_dscale_ml": float(dscale_ml[best_idx]),
                "improvement_deg": baseline_score - candidate_score,
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

    print("=== Anisotropic AP/ML scale deviation (9x9 joint grid, 0.85-1.15x per axis) ===")
    table = evaluate_anisotropic_scale(cells, geometry, az_interp, el_interp, DSCALE_AXIS_GRID, DSCALE_AXIS_GRID)
    table.to_csv(OUTPUT / "cv_anisotropic_scale_step.csv", index=False)
    adopt, median_improvement, frac_improved, p_value = decide("anisotropic_scale", table)

    summary = {"adopted": adopt, "median_improvement_deg": median_improvement,
               "fraction_folds_improved": frac_improved, "wilcoxon_p": p_value, "n_folds": len(table)}
    (OUTPUT / "anisotropic_scale_decision.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT / 'anisotropic_scale_decision.json'}")

    if len(table):
        print(f"\nchosen dscale_ap distribution: median={table.chosen_dscale_ap.median():.3f}, "
              f"IQR=[{table.chosen_dscale_ap.quantile(.25):.3f}, {table.chosen_dscale_ap.quantile(.75):.3f}]")
        print(f"chosen dscale_ml distribution: median={table.chosen_dscale_ml.median():.3f}, "
              f"IQR=[{table.chosen_dscale_ml.quantile(.25):.3f}, {table.chosen_dscale_ml.quantile(.75):.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(table.improvement_deg, bins=30, color="#2864a8", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(table.improvement_deg.median(), color="#b33f62", linewidth=1.5, linestyle="--",
                     label=f"median={table.improvement_deg.median():+.2f} deg")
    axes[0].set(title="Held-out improvement: anisotropic AP/ML scale\n(positive = helps)",
                xlabel="baseline_error - candidate_error (deg)", ylabel="folds")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].scatter(table.chosen_dscale_ap, table.chosen_dscale_ml, s=14, alpha=0.4, color="#2864a8")
    axes[1].axhline(1.0, color="grey", linewidth=0.8)
    axes[1].axvline(1.0, color="grey", linewidth=0.8)
    axes[1].set(title="Chosen (dscale_ap, dscale_ml) per fold", xlabel="dscale_ap", ylabel="dscale_ml")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_anisotropic_scale_cv.png", dpi=160)
    plt.close(fig)
    print(OUTPUT / "Figure_anisotropic_scale_cv.png")


if __name__ == "__main__":
    main()
