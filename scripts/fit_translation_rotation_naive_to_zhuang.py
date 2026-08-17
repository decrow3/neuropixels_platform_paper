#!/usr/bin/env python3
"""Fit ONLY translation + rotation between the naive pooled map and Zhuang -- scale fixed at
Zhuang's true px/mm (104.6, from the Figure 3 scale bar), reflection fixed at the already-
established mirror. This is the constrained version of the earlier 8-parameter full-affine
fit (register_naive_map_to_atlases.py), which was underconstrained and produced a degenerate,
over-elongated shape. With only 5 free parameters (rotation angle, a small translation on top
of the V1-anchor starting point, and the 2 RF-offset parameters), this should be far better
behaved -- and directly tests whether adding rotation resolves the LM/AL compartment
confusion found in check_probe_area_labels_vs_zhuang_registration.py's translation-only check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import (  # noqa: E402
    AREA_LABELS, AREA_SEEDS_XY, build_template, sample_template,
)
from register_naive_map_to_atlases import build_landmarks, pseudo_huber  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"
V1_SEED_XY_PX = (200, 240)
ZHUANG_PX_PER_MM = 62.0 / 0.5 * 0.8432313316638625
REFLECT_ML = True  # locked, matching the default rough registration
AREA_WEIGHT = 2.0


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    landmarks = build_landmarks()
    print(f"landmarks: {len(landmarks)} grid cells, {landmarks.cells.sum()} cells")

    template = build_template(ZHUANG_TEMPLATE)
    height, width = template["domain"].shape
    ml_sign = -1.0 if REFLECT_ML else 1.0
    v1_seed_col, v1_seed_row = V1_SEED_XY_PX

    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    naive_rf = landmarks[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
    v1_anchor = ccf[landmarks.dominant_area.eq("VISp").to_numpy()].mean(axis=0) if landmarks.dominant_area.eq("VISp").any() else ccf.mean(axis=0)
    areas = landmarks["dominant_area"].tolist()
    known = np.array([a in AREA_SEEDS_XY for a in areas])
    # Precompute per-area landmark indices ONCE, so the objective can do one vectorized
    # area_distance() call per area (~5 calls) instead of one per landmark (~173 calls) --
    # the per-landmark Python loop was exactly what made register_naive_map_to_atlases.py's
    # area-penalty fit time out earlier.
    area_indices = {area: np.array([i for i, a in enumerate(areas) if a == area])
                     for area in set(a for a in areas if a in AREA_SEEDS_XY)}

    def transform(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta, tx, ty, offset_az, offset_el = parameters
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # Acts on [delta_ml, delta_ap] (x-ish, y-ish order), matching pixel_center=[col,row].
        scale_reflect = np.diag([ml_sign * ZHUANG_PX_PER_MM, ZHUANG_PX_PER_MM])
        matrix = rotation @ scale_reflect
        pixel_center = np.array([v1_seed_col, v1_seed_row]) + np.array([tx, ty])
        delta = ccf - v1_anchor  # columns are [ap, ml]
        delta_ml_ap = delta[:, [1, 0]]  # reorder to [ml, ap] = [x-ish, y-ish]
        xy = delta_ml_ap @ matrix.T + pixel_center
        return xy, np.array([offset_az, offset_el])

    def objective(parameters: np.ndarray) -> float:
        xy, offset = transform(parameters)
        predicted, outside, bounds = sample_template(template, xy)
        target = naive_rf + offset
        valid = np.isfinite(predicted).all(axis=1)
        if valid.sum() < 10:
            return 50.0
        retinal = float(np.mean(pseudo_huber((predicted[valid] - target[valid]) / 10.0)))
        domain_penalty = float(3.0 * (1 - valid.mean()) ** 2)
        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        area_penalty = 0.0
        if area_indices:
            squared = []
            for area, idx in area_indices.items():
                distances = template["area_distance"][area](row_col[idx])
                squared.append((distances / 12.0) ** 2)
            area_penalty = float(np.mean(np.concatenate(squared)))
        return retinal + domain_penalty + AREA_WEIGHT * area_penalty

    bounds = [(-np.pi / 6, np.pi / 6), (-80.0, 80.0), (-80.0, 80.0), (-60.0, 60.0), (-40.0, 40.0)]
    result = differential_evolution(
        objective, bounds, seed=20260821, maxiter=500, popsize=20, tol=1e-9, polish=True, workers=1, updating="immediate",
    )
    xy, offset = transform(result.x)
    predicted, _, _ = sample_template(template, xy)
    target = naive_rf + offset
    valid = np.isfinite(predicted).all(axis=1)
    theta, tx, ty = result.x[0], result.x[1], result.x[2]
    print(f"fitted rotation: {np.degrees(theta):+.1f} deg, translation: ({tx:+.1f}, {ty:+.1f}) px, "
          f"RF offset: az={offset[0]:+.1f}, el={offset[1]:+.1f} deg")
    print(f"objective={result.fun:.3f}, valid_fraction={valid.mean():.2%}, "
          f"median_vector_error={np.median(np.linalg.norm((predicted - target)[valid], axis=1)):.1f} deg")

    # per-area agreement check, same style as check_probe_area_labels_vs_zhuang_registration.py
    row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
    agree_rows = []
    for i, area in enumerate(areas):
        if not known[i]:
            continue
        distances = {a: float(template["area_distance"][a](row_col[i:i + 1])[0]) for a in AREA_SEEDS_XY}
        nearest = min(distances, key=distances.get)
        agree_rows.append({"area": area, "own_distance_px": distances[area], "nearest": nearest, "agrees": distances[area] <= 1.5})
    import pandas as pd
    agree = pd.DataFrame(agree_rows)
    print("\nlandmark-level area agreement (rotation+translation fit):")
    print(agree.groupby("area").agrees.agg(["size", "mean"]))
    print(f"overall: {agree.agrees.mean():.1%}")

    manifest = {
        "fixed_scale_px_per_mm": ZHUANG_PX_PER_MM,
        "fixed_reflection_ml": REFLECT_ML,
        "fitted_rotation_deg": float(np.degrees(theta)),
        "fitted_translation_px": [float(tx), float(ty)],
        "fitted_rf_offset_deg": offset.tolist(),
        "v1_anchor_ccf_ap_ml_mm": v1_anchor.tolist(),
        "v1_seed_xy_px": V1_SEED_XY_PX,
        "objective": float(result.fun),
        "valid_fraction": float(valid.mean()),
        "median_vector_error_deg": float(np.median(np.linalg.norm((predicted - target)[valid], axis=1))),
        "area_agreement_overall": float(agree.agrees.mean()),
        "area_agreement_by_area": agree.groupby("area").agrees.mean().to_dict(),
    }
    (OUTPUT / "translation_rotation_fit_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for kind, idx, color in (("azimuth", 0, "#4c78a8"), ("elevation", 1, "#d95f5f")):
        ax.scatter(target[valid, idx], predicted[valid, idx], s=10, alpha=.5, color=color, label=kind)
    lo = min(target[valid].min(), predicted[valid].min())
    hi = max(target[valid].max(), predicted[valid].max())
    ax.plot([lo, hi], [lo, hi], color=".5", lw=1)
    ax.set(xlabel="naive map (offset-corrected, deg)", ylabel="Zhuang predicted (deg)",
           title=f"translation+rotation fit: {np.degrees(theta):+.1f} deg rotation")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_translation_rotation_fit_agreement.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
