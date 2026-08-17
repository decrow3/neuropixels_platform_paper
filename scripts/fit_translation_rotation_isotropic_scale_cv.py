#!/usr/bin/env python3
"""Does translation + rotation + ISOTROPIC scale (jointly) beat the naive (translation-only)
map, under honest leave-one-probe-out cross-validation?

Rotation alone (+0.89 deg median held-out improvement, below the 1.0 deg practical bar) and
anisotropic AP/ML scale alone (+0.45 deg, not significant) each failed independently
(`fit_per_session_incremental_warp_cv.py`, `fit_per_session_anisotropic_scale_cv.py`). This
tests the JOINT model -- one isotropic scale factor (not separate AP/ML gains) plus rotation,
fit together per fold -- against the same leave-one-probe-out discipline, since a combination
could in principle help even where each piece alone did not (or could just compound the same
overfitting risk on already-scarce per-session landmarks).

Same guardrails as before: offset is refit fresh for every candidate (dtheta, dscale) pair
(closed-form Huber location on training probes only); the best candidate is chosen by IN-SAMPLE
training loss (median of per-probe median residual norm, so no training probe dominates by cell
count); held-out score is computed on the left-out probe with that chosen candidate; decision
requires median held-out improvement > 1.0 deg AND paired Wilcoxon p < 0.05 across all folds.
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from render_warp_variant_overlays import build_interpolators, sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_warp_cv"

MIN_TRAIN_CELLS = 15
DTHETA_GRID_DEG = np.linspace(-60, 60, 25)
DSCALE_GRID = np.linspace(0.3, 3.0, 28)
MIN_PRACTICAL_IMPROVEMENT_DEG = 1.0
ALPHA = 0.05


def evaluate(cells, geometry, az_interp, el_interp):
    theta_mesh, scale_mesh = np.meshgrid(DTHETA_GRID_DEG, DSCALE_GRID, indexing="ij")
    dtheta_deg = theta_mesh.ravel()
    dscale = scale_mesh.ravel()
    baseline_idx = int(np.argmin(np.abs(dtheta_deg) + np.abs(dscale - 1.0)))

    rows = []
    for sid, session_cells in cells.groupby("ecephys_session_id"):
        ccf = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        naive_rf = session_cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
        probes = session_cells["ecephys_probe_id"].to_numpy()
        unique_probes = np.unique(probes)
        if len(unique_probes) < 3:
            continue

        predicted = sample(ccf, geometry, np.radians(dtheta_deg), dscale, dscale, az_interp, el_interp)

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

            losses = np.array([train_loss(k) for k in range(len(dtheta_deg))])
            best_idx = int(np.argmin(losses))
            baseline_score = fold_score(baseline_idx)
            candidate_score = fold_score(best_idx)
            if not (np.isfinite(baseline_score) and np.isfinite(candidate_score)):
                continue
            rows.append({
                "ecephys_session_id": int(sid), "held_probe": int(held_probe),
                "n_train_cells": int(train_mask.sum()), "n_held_cells": int(held_mask.sum()),
                "baseline_held_error_deg": baseline_score, "candidate_held_error_deg": candidate_score,
                "chosen_dtheta_deg": float(dtheta_deg[best_idx]), "chosen_dscale": float(dscale[best_idx]),
                "improvement_deg": baseline_score - candidate_score,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    smoothed, az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    print(f"=== translation + rotation + isotropic scale, jointly "
          f"({len(DTHETA_GRID_DEG)}x{len(DSCALE_GRID)} grid, "
          f"theta in [{DTHETA_GRID_DEG.min():.0f},{DTHETA_GRID_DEG.max():.0f}] deg, "
          f"scale in [{DSCALE_GRID.min():.2f},{DSCALE_GRID.max():.2f}]), leave-one-probe-out CV ===")
    table = evaluate(cells, geometry, az_interp, el_interp)
    table.to_csv(OUTPUT / "cv_rotation_isotropic_scale_joint_wider.csv", index=False)

    delta = table.improvement_deg.to_numpy()
    median_improvement = float(np.median(delta))
    frac_improved = float((delta > 0).mean())
    stat, p_value = wilcoxon(delta)
    adopt = bool(median_improvement > MIN_PRACTICAL_IMPROVEMENT_DEG and p_value < ALPHA)
    print(f"folds={len(table)}  median_improvement={median_improvement:+.2f} deg  "
          f"frac_folds_improved={frac_improved:.1%}  wilcoxon_p={p_value:.4g}  ADOPT={adopt}")

    summary = {"median_improvement_deg": median_improvement, "fraction_folds_improved": frac_improved,
               "wilcoxon_p": float(p_value), "n_folds": len(table), "adopted": adopt,
               "practical_bar_deg": MIN_PRACTICAL_IMPROVEMENT_DEG, "alpha": ALPHA}
    (OUTPUT / "rotation_isotropic_scale_joint_wider_decision.json").write_text(json.dumps(summary, indent=2))

    # ---- "why / why not" figure: 4 panels ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))

    ax = axes[0, 0]
    ax.hist(delta, bins=30, color="#2864a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(median_improvement, color="#b33f62", linewidth=1.5, linestyle="--", label=f"median={median_improvement:+.2f} deg")
    ax.axvline(MIN_PRACTICAL_IMPROVEMENT_DEG, color="grey", linewidth=1.2, linestyle=":", label=f"practical bar={MIN_PRACTICAL_IMPROVEMENT_DEG} deg")
    ax.set(title=f"Held-out improvement per fold (n={len(table)})\nwilcoxon p={p_value:.3g}, {frac_improved:.0%} folds improved",
           xlabel="baseline_error - candidate_error (deg)", ylabel="folds")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.scatter(table.baseline_held_error_deg, table.candidate_held_error_deg, s=14, alpha=0.4, color="#2864a8")
    lo = 0
    hi = max(table.baseline_held_error_deg.max(), table.candidate_held_error_deg.max())
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1, linestyle="--", label="y=x (no change)")
    ax.set(title="Held-out error: baseline vs. candidate\n(points below the line = candidate better)",
           xlabel="baseline (offset-only) held-out error (deg)", ylabel="candidate (rot+scale+offset) held-out error (deg)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    ax.hist(table.chosen_dtheta_deg, bins=DTHETA_GRID_DEG, color="#5f8f3e", alpha=0.85)
    ax.set(title="Chosen rotation deviation per fold\n(pileup at grid edges = overfitting signature)",
           xlabel="chosen dtheta (deg)", ylabel="folds")

    ax = axes[1, 1]
    ax.hist(table.chosen_dscale, bins=DSCALE_GRID, color="#d78318", alpha=0.85)
    ax.set(title="Chosen isotropic scale per fold\n(pileup at grid edges = overfitting signature)",
           xlabel="chosen dscale (x)", ylabel="folds")

    fig.suptitle(
        f"Translation + rotation + isotropic scale vs. translation-only (naive map): "
        f"{'ADOPTED' if adopt else 'NOT adopted'}\n"
        f"(needs median improvement > {MIN_PRACTICAL_IMPROVEMENT_DEG} deg AND p < {ALPHA}; "
        f"got {median_improvement:+.2f} deg, p={p_value:.3g})",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figure_path = OUTPUT / "Figure_rotation_isotropic_scale_vs_naive_cv_wider.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
