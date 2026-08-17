#!/usr/bin/env python3
"""Held-out check of the claim "per-session offset significantly improves over naive": every
rotation/scale CV script so far used a per-session-refit offset as its BASELINE without ever
validating that baseline against the even-simpler single-pooled-offset (naive) alternative. The
visual/in-sample comparisons (Figure_naive_vs_per_session_translation_comparison.png,
Figure_naive_vs_per_session_agreement.png) are suggestive but not held-out evidence: a per-
session offset fits its own session's cells better almost mechanically, since it's a free
parameter per session vs. one shared constant. This runs the same leave-one-probe-out
discipline used throughout to test it properly.

For each (session, held-out probe) fold:
  naive offset      = huber_location(residual) over ALL OTHER cells in the ENTIRE cohort
                       (every other session's cells too, held-out probe excluded)
  per_session offset = huber_location(residual) over this session's OTHER probes only
                       (same session, held-out probe excluded)
Both are evaluated on the same held-out probe's cells; anatomical geometry (rotation/
translation/scale/reflection) is identical for both, fixed throughout -- the only thing that
differs is how much data each offset estimate pools across.
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

MIN_TRAIN_CELLS = 15
CAP_RADIUS_DEG = 21.15629391755514  # from the deployed default's own manifest (85th pct of deviation)


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


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    ccf_to_pixel = build_ccf_to_pixel(geometry)

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(az_field.shape[0]), np.arange(az_field.shape[1])
    az_interp = RegularGridInterpolator((row_axis, col_axis), az_field, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el_field, bounds_error=False, fill_value=np.nan)

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    ccf_all = cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    xy_all = ccf_to_pixel(ccf_all)
    row_col_all = xy_all[:, ::-1]
    predicted_all = np.column_stack([az_interp(row_col_all), el_interp(row_col_all)])
    naive_rf_all = cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    residual_all = predicted_all - naive_rf_all
    valid_all = np.isfinite(residual_all).all(axis=1)

    session_ids = cells.ecephys_session_id.to_numpy()
    probe_ids = cells.ecephys_probe_id.to_numpy()

    rows = []
    for sid in cells.ecephys_session_id.unique():
        session_mask = session_ids == sid
        session_probes = np.unique(probe_ids[session_mask])
        if len(session_probes) < 3:
            continue
        for held_probe in session_probes:
            held_mask = session_mask & (probe_ids == held_probe)
            if held_mask.sum() < 5:
                continue
            train_session_mask = session_mask & ~held_mask
            if train_session_mask.sum() < MIN_TRAIN_CELLS:
                continue

            # naive: pooled over the WHOLE cohort, excluding only the held-out probe
            train_global_mask = valid_all & ~held_mask
            naive_offset = huber_location(residual_all[train_global_mask])

            # per-session: pooled over this session's OTHER probes only
            train_session_valid = valid_all & train_session_mask
            per_session_offset = huber_location(residual_all[train_session_valid])

            # deployed-style shrinkage: cap the per-session offset's deviation from naive at the
            # SAME fixed radius the locked default uses, direction preserved
            deviation = per_session_offset - naive_offset
            deviation_magnitude = float(np.linalg.norm(deviation))
            shrink_factor = min(1.0, CAP_RADIUS_DEG / deviation_magnitude) if deviation_magnitude > 0 else 1.0
            capped_offset = naive_offset + shrink_factor * deviation

            held_valid = valid_all & held_mask
            if held_valid.sum() == 0:
                continue
            held_residual = residual_all[held_valid]
            naive_error = float(np.median(np.linalg.norm(held_residual - naive_offset, axis=1)))
            per_session_error = float(np.median(np.linalg.norm(held_residual - per_session_offset, axis=1)))
            capped_error = float(np.median(np.linalg.norm(held_residual - capped_offset, axis=1)))

            rows.append({
                "ecephys_session_id": int(sid), "held_probe": int(held_probe),
                "n_train_cells_session": int(train_session_valid.sum()), "n_held_cells": int(held_valid.sum()),
                "naive_held_error_deg": naive_error, "per_session_held_error_deg": per_session_error,
                "capped_held_error_deg": capped_error, "shrink_factor": shrink_factor,
                "improvement_deg": naive_error - per_session_error,
                "capped_improvement_deg": naive_error - capped_error,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "cv_naive_vs_per_session_offset.csv", index=False)

    def stats_for(col):
        delta = table[col].to_numpy()
        median_improvement = float(np.median(delta))
        frac_improved = float((delta > 0).mean())
        stat, p_value = wilcoxon(delta)
        return delta, median_improvement, frac_improved, float(p_value)

    delta_raw, med_raw, frac_raw, p_raw = stats_for("improvement_deg")
    delta_capped, med_capped, frac_capped, p_capped = stats_for("capped_improvement_deg")
    print(f"RAW per-session:    folds={len(table)}  median_improvement={med_raw:+.2f} deg  "
          f"frac_folds_improved={frac_raw:.1%}  wilcoxon_p={p_raw:.4g}")
    print(f"CAPPED per-session: folds={len(table)}  median_improvement={med_capped:+.2f} deg  "
          f"frac_folds_improved={frac_capped:.1%}  wilcoxon_p={p_capped:.4g}  "
          f"(n_capped_this_run={int((table.shrink_factor < 1.0).sum())})")
    print(f"naive median held-out error:       {table.naive_held_error_deg.median():.2f} deg")
    print(f"raw per-session median held-out error:    {table.per_session_held_error_deg.median():.2f} deg")
    print(f"capped per-session median held-out error: {table.capped_held_error_deg.median():.2f} deg")

    summary = {
        "n_folds": len(table),
        "raw": {"median_improvement_deg": med_raw, "fraction_folds_improved": frac_raw, "wilcoxon_p": p_raw},
        "capped": {"median_improvement_deg": med_capped, "fraction_folds_improved": frac_capped, "wilcoxon_p": p_capped,
                   "cap_radius_deg": CAP_RADIUS_DEG},
        "naive_median_held_error_deg": float(table.naive_held_error_deg.median()),
        "per_session_median_held_error_deg": float(table.per_session_held_error_deg.median()),
        "capped_median_held_error_deg": float(table.capped_held_error_deg.median()),
    }
    (OUTPUT / "naive_vs_per_session_offset_cv_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for row, (delta, median_improvement, frac_improved, p_value, label, err_col) in enumerate((
        (delta_raw, med_raw, frac_raw, p_raw, "RAW per-session", "per_session_held_error_deg"),
        (delta_capped, med_capped, frac_capped, p_capped, "CAPPED per-session (deployed-style)", "capped_held_error_deg"),
    )):
        ax = axes[row, 0]
        ax.hist(delta, bins=30, color="#2864a8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(median_improvement, color="#b33f62", linewidth=1.5, linestyle="--", label=f"median={median_improvement:+.2f} deg")
        ax.set(title=f"{label}: held-out improvement per fold (n={len(table)})\nwilcoxon p={p_value:.3g}, {frac_improved:.0%} folds improved",
               xlabel="naive_error - candidate_error (deg)", ylabel="folds")
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 1]
        ax.scatter(table.naive_held_error_deg, table[err_col], s=14, alpha=0.4, color="#2864a8")
        lo = 0
        hi = max(table.naive_held_error_deg.max(), table[err_col].max())
        ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1, linestyle="--", label="y=x (no change)")
        ax.set(title=f"{label}: held-out error, naive vs. candidate\n(points below the line = candidate better)",
               xlabel="naive held-out error (deg)", ylabel="candidate held-out error (deg)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Naive (single pooled offset) vs. per-session offset -- leave-one-probe-out CV", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figure_path = OUTPUT / "Figure_naive_vs_per_session_offset_cv.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
