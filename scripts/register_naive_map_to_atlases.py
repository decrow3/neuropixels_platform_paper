#!/usr/bin/env python3
"""Register the pooled naive (V1-centered) cross-session ephys map to Zhuang and to Garrett.

Unlike the single-session fits (5-6 probe/area landmarks, noisy), this uses the naive map's
full spatial coverage: every CCF grid cell with enough pooled cells becomes one landmark, so
the fit draws on dozens of session-pooled, already-averaged points instead of one session's
sparse penetrations. Because the naive map's RF values are V1-median-relative (not absolute),
the fit adds a free global RF offset (2 params) alongside the usual 6-parameter CCF<->atlas
affine, rather than assuming the naive map's zero point already matches the atlas's.

No area-membership penalty is used here (unlike the single-session `joint_anatomy_rf` model):
this is deliberately an "RF-only" style fit, since the question is whether the pooled empirical
SHAPE matches each atlas's shape, not whether landmarks land in the nominally-correct compartment.
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
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import make_grid  # noqa: E402
from register_allen_session_to_zhuang import (  # noqa: E402
    AREA_SEEDS_XY, build_template as build_zhuang_template, sample_template as sample_zhuang,
)
from build_garrett2014_smoothed_field_and_ccf_affine import build_fields as build_garrett_fields, sample_template as sample_garrett  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"
CCF2 = ["anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate"]
GRID_STEP_UM = 150.0
GRID_MARGIN_UM = 200.0
MIN_CELLS_PER_LANDMARK = 8


def pseudo_huber(values: np.ndarray) -> np.ndarray:
    return 2.0 * (np.sqrt(1.0 + np.square(values)) - 1.0)


def build_landmarks() -> pd.DataFrame:
    cells = pd.read_csv(NAIVE_CELLS)
    points = cells[CCF2].to_numpy(float)
    axis0, axis1, grid = make_grid(points, GRID_STEP_UM, GRID_MARGIN_UM)
    n0, n1 = len(axis0), len(axis1)
    col = np.clip(np.searchsorted(axis1, cells[CCF2[1]].to_numpy(float)) - 1, 0, n1 - 1)
    row = np.clip(np.searchsorted(axis0, cells[CCF2[0]].to_numpy(float)) - 1, 0, n0 - 1)
    cells = cells.assign(_row=row, _col=col)
    rows = []
    for (r, c), group in cells.groupby(["_row", "_col"]):
        if len(group) < MIN_CELLS_PER_LANDMARK:
            continue
        rows.append({
            "ccf_ap_mm": axis0[r] / 1000.0, "ccf_ml_mm": axis1[c] / 1000.0,
            "normalized_rf_x": group.normalized_rf_x.median(), "normalized_rf_y": group.normalized_rf_y.median(),
            "cells": len(group), "sessions": group.ecephys_session_id.nunique(),
            "dominant_area": group.map_area.mode().iloc[0],
        })
    return pd.DataFrame(rows)


def build_zhuang_area_penalty(template: dict, landmarks: pd.DataFrame):
    """Mirrors fit_candidate's area-membership penalty (register_allen_session_to_zhuang.py),
    the piece missing from the first version of this fit that let it settle on a mirrored,
    anatomically-wrong reflection: RF-value agreement alone barely distinguishes true
    handedness from its mirror image, but area identity does.
    """
    areas = landmarks["dominant_area"].tolist()
    height, width = template["domain"].shape
    known = np.array([area in AREA_SEEDS_XY for area in areas])
    print(f"area penalty: {known.sum()}/{len(areas)} landmarks have a known Zhuang-compartment area")

    def area_penalty_fn(xy: np.ndarray) -> np.ndarray:
        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        penalty = np.zeros(len(xy))
        for i, area in enumerate(areas):
            if not known[i]:
                continue
            distance = template["area_distance"][area](row_col[i:i + 1])[0]
            penalty[i] = (distance / 12.0) ** 2
        return penalty

    return area_penalty_fn


def fit_registration(landmarks: pd.DataFrame, sample_fn, bounds: list[tuple[float, float]], reflection_choices, seed: int,
                      area_penalty_fn=None, area_weight: float = 2.0) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    naive_rf = landmarks[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    ccf_center = ccf.mean(axis=0)

    def transform(parameters: np.ndarray, reflection: int) -> np.ndarray:
        center_x, center_y, theta, scale_x, scale_y, shear, offset_x, offset_y = parameters
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matrix = rotation @ np.array([[scale_x, shear * scale_y], [0.0, reflection * scale_y]])
        xy = (ccf - ccf_center) @ matrix.T + np.array([center_x, center_y])
        return xy, matrix, np.array([offset_x, offset_y])

    def make_objective(reflection: int):
        def objective(parameters: np.ndarray) -> float:
            xy, _, offset = transform(parameters, reflection)
            predicted = sample_fn(xy)
            target = naive_rf + offset
            valid = np.isfinite(predicted).all(axis=1)
            if valid.sum() < 5:
                return 50.0
            retinal = float(np.mean(pseudo_huber((predicted[valid] - target[valid]) / 10.0)))
            domain_penalty = float(3.0 * (1 - valid.mean()) ** 2)
            total = retinal + domain_penalty
            if area_penalty_fn is not None:
                total += area_weight * float(np.mean(area_penalty_fn(xy)))
            return total
        return objective

    candidates = []
    for i, reflection in enumerate(reflection_choices):
        result = differential_evolution(
            make_objective(reflection), bounds, seed=seed + i, maxiter=400, popsize=18,
            tol=1e-8, polish=True, workers=1, updating="immediate",
        )
        xy, matrix, offset = transform(result.x, reflection)
        predicted = sample_fn(xy)
        target = naive_rf + offset
        valid = np.isfinite(predicted).all(axis=1)
        candidates.append({
            "reflection": reflection, "objective": float(result.fun), "parameters": result.x,
            "ccf_center": ccf_center, "matrix": matrix, "offset": offset, "xy": xy,
            "predicted": predicted, "target": target, "valid_fraction": float(valid.mean()),
            "median_vector_error_deg": float(np.median(np.linalg.norm((predicted - target)[valid], axis=1))) if valid.sum() else float("nan"),
        })
    return min(candidates, key=lambda item: item["objective"]), candidates


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    landmarks = build_landmarks()
    landmarks.to_csv(OUTPUT / "naive_map_landmarks.csv", index=False)
    print(f"landmarks: {len(landmarks)} grid cells, {landmarks.cells.sum()} cells, "
          f"{landmarks.sessions.sum()} session-cell-incidences pooled")

    zhuang_template = build_zhuang_template(ZHUANG_TEMPLATE)

    def zhuang_sample(xy):
        predicted, _, _ = sample_zhuang(zhuang_template, xy)
        return predicted

    z_height, z_width = zhuang_template["domain"].shape
    zhuang_bounds = [
        (40.0, z_width - 30.0), (20.0, z_height - 20.0), (-np.pi, np.pi), (70.0, 320.0), (70.0, 320.0), (-0.8, 0.8),
        (-60.0, 60.0), (-40.0, 40.0),
    ]
    # v1: RF-only objective picked reflection=-1 by a narrow, spurious margin (1.038 vs
    # 1.111) -- a mirrored, anatomically-wrong map (confirmed against the properly
    # anatomy-constrained 14-animal fit: reflection=+1 objective 1.08 vs -1's 9.68, a ~9x
    # gap). v2: forcing reflection=+1 alone was NOT enough -- without any anatomical
    # constraint the optimizer found a different, wildly over-elongated degenerate affine at
    # a similar RF-only objective. Fixed properly here by adding the area-membership penalty
    # (build_zhuang_area_penalty), the actual missing ingredient, and letting reflection be
    # freely chosen again now that the objective can tell true handedness from its mirror.
    zhuang_area_penalty = build_zhuang_area_penalty(zhuang_template, landmarks)
    zhuang_best, zhuang_candidates = fit_registration(
        landmarks, zhuang_sample, zhuang_bounds, (-1, 1), 20260819, area_penalty_fn=zhuang_area_penalty,
    )
    print(f"Zhuang: reflection={zhuang_best['reflection']}, objective={zhuang_best['objective']:.3f}, "
          f"valid_fraction={zhuang_best['valid_fraction']:.2f}, median_error={zhuang_best['median_vector_error_deg']:.1f} deg")

    garrett_fields = build_garrett_fields()

    def garrett_sample(xy):
        return sample_garrett(garrett_fields, xy)

    garrett_bounds = [
        (-0.3, 0.3), (-0.3, 0.3), (-np.pi, np.pi), (0.02, 2.0), (0.02, 2.0), (-0.8, 0.8),
        (-60.0, 60.0), (-40.0, 40.0),
    ]
    garrett_best, garrett_candidates = fit_registration(landmarks, garrett_sample, garrett_bounds, (-1, 1), 20260820)
    print(f"Garrett: reflection={garrett_best['reflection']}, objective={garrett_best['objective']:.3f}, "
          f"valid_fraction={garrett_best['valid_fraction']:.2f}, median_error={garrett_best['median_vector_error_deg']:.1f} deg")

    manifest = {
        "n_landmarks": len(landmarks),
        "min_cells_per_landmark": MIN_CELLS_PER_LANDMARK,
        "grid_step_um": GRID_STEP_UM,
        "zhuang": {
            "candidates": [{"reflection": c["reflection"], "objective": c["objective"],
                             "valid_fraction": c["valid_fraction"], "median_vector_error_deg": c["median_vector_error_deg"]}
                            for c in zhuang_candidates],
            "selected_reflection": zhuang_best["reflection"],
            "fitted_offset_deg": zhuang_best["offset"].tolist(),
            "ccf_center_ap_ml_mm": zhuang_best["ccf_center"].tolist(),
            "template_center_xy": zhuang_best["parameters"][:2].tolist(),
            "matrix_px_per_mm": zhuang_best["matrix"].tolist(),
        },
        "garrett": {
            "candidates": [{"reflection": c["reflection"], "objective": c["objective"],
                             "valid_fraction": c["valid_fraction"], "median_vector_error_deg": c["median_vector_error_deg"]}
                            for c in garrett_candidates],
            "selected_reflection": garrett_best["reflection"],
            "fitted_offset_deg": garrett_best["offset"].tolist(),
            "ccf_center_ap_ml_mm": garrett_best["ccf_center"].tolist(),
            "template_center_xy": garrett_best["parameters"][:2].tolist(),
            "matrix_panel_units_per_mm": garrett_best["matrix"].tolist(),
        },
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, best, title in ((axes[0], zhuang_best, "Zhuang"), (axes[1], garrett_best, "Garrett")):
        valid = np.isfinite(best["predicted"]).all(axis=1)
        for kind, idx, color in (("azimuth", 0, "#4c78a8"), ("elevation", 1, "#d95f5f")):
            ax.scatter(best["target"][valid, idx], best["predicted"][valid, idx], s=10, alpha=.5, color=color, label=kind)
        lo = min(best["target"][valid].min(), best["predicted"][valid].min())
        hi = max(best["target"][valid].max(), best["predicted"][valid].max())
        ax.plot([lo, hi], [lo, hi], color=".5", lw=1)
        ax.set(xlabel="naive map (offset-corrected, deg)", ylabel=f"{title} predicted (deg)",
               title=f"{title}: valid={best['valid_fraction']:.0%}, median err={best['median_vector_error_deg']:.1f} deg")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Population-level registration of naive pooled map ({len(landmarks)} landmarks) to each atlas")
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_naive_map_registration_agreement.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
