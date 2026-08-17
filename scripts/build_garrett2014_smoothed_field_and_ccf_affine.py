#!/usr/bin/env python3
"""Build a smoothed continuous Garrett et al. (2014) retinotopy field and fit its
CCF<->panel affine for one session, mirroring the Zhuang pipeline.

Garrett's template (`artifacts/retinotopy_template/garrett2014_figure5/`) has no continuous
raster and no CCF registration yet (its own README: "figure coordinate frame, not yet AP/ML
CCF space"). This script:
1. Rasterizes the vector contour line segments (`retinotopy_contours.csv.gz`) and area
   boundary segments (`area_boundaries.csv.gz`) onto a regular grid in the same
   image-row/column convention as the Zhuang pipeline (row increases as y decreases, so the
   same `da_dy = -da_drow` chain rule applies downstream).
2. Reuses `reconstruct()`/`normalized_smooth()` from `render_zhuang_interpolated_field_sign_qa.py`
   as-is (those functions are generic, not Zhuang-specific) to get a linear-interpolated,
   nearest-filled surface and a support-masked Gaussian-smoothed version.
3. Fits a CCF(mm)<->Garrett-panel affine for one session via the same probe/area-median
   `differential_evolution` approach as `register_allen_session_to_zhuang.py::fit_candidate`,
   with bounds rescaled for Garrett's much smaller "panel width = 1 unit" coordinate system.
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
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_14animal_retinotopy_registration import make_landmarks, production_support  # noqa: E402
from render_zhuang_interpolated_field_sign_qa import normalized_smooth, reconstruct  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
GARRETT_ROOT = ROOT / "artifacts/retinotopy_template/garrett2014_figure5"
OUTPUT = GARRETT_ROOT / "smoothed_field_and_ccf_affine"
SESSION_ID = 781842082
GRID_N = 320
MARGIN = 0.2
GRADIENT_SIGMA_PX = 2.0


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    contours = pd.read_csv(GARRETT_ROOT / "retinotopy_contours.csv.gz")
    boundaries = pd.read_csv(GARRETT_ROOT / "area_boundaries.csv.gz")
    all_x = np.concatenate([contours.x0, contours.x1, boundaries.x0, boundaries.x1])
    all_y = np.concatenate([contours.y0, contours.y1, boundaries.y0, boundaries.y1])
    x_axis = np.linspace(all_x.min() - MARGIN, all_x.max() + MARGIN, GRID_N)
    y_axis = np.linspace(all_y.max() + MARGIN, all_y.min() - MARGIN, GRID_N)  # decreasing: row up => y up
    return x_axis, y_axis


def to_row_col(x: np.ndarray, y: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col = np.clip(np.round((x - x_axis[0]) / (x_axis[-1] - x_axis[0]) * (len(x_axis) - 1)), 0, len(x_axis) - 1).astype(int)
    row = np.clip(np.round((y - y_axis[0]) / (y_axis[-1] - y_axis[0]) * (len(y_axis) - 1)), 0, len(y_axis) - 1).astype(int)
    return row, col


def rasterize_sparse_values(frame: pd.DataFrame, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    sparse = np.full((len(y_axis), len(x_axis)), np.nan)
    for x, y, value in ((frame.x0, frame.y0, frame.value_deg), (frame.x1, frame.y1, frame.value_deg)):
        row, col = to_row_col(x.to_numpy(float), y.to_numpy(float), x_axis, y_axis)
        sparse[row, col] = value.to_numpy(float)
    return sparse


def rasterize_boundary(boundaries: pd.DataFrame, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    raster = np.zeros((len(y_axis), len(x_axis)), dtype=bool)
    for _, seg in boundaries.iterrows():
        t = np.linspace(0, 1, 6)
        x = seg.x0 + t * (seg.x1 - seg.x0)
        y = seg.y0 + t * (seg.y1 - seg.y0)
        row, col = to_row_col(x, y, x_axis, y_axis)
        raster[row, col] = True
    return raster


def build_fields() -> dict:
    contours = pd.read_csv(GARRETT_ROOT / "retinotopy_contours.csv.gz")
    boundaries = pd.read_csv(GARRETT_ROOT / "area_boundaries.csv.gz")
    x_axis, y_axis = build_grid()

    boundary_raster = rasterize_boundary(boundaries, x_axis, y_axis)
    boundary_thick = ndimage.binary_dilation(boundary_raster, iterations=1)
    domain = ndimage.binary_fill_holes(boundary_thick)

    fields = {}
    for map_name, key in (("azimuth", "azimuth_deg"), ("altitude", "elevation_deg")):
        sparse = rasterize_sparse_values(contours.loc[contours["map"].eq(map_name)], x_axis, y_axis)
        surface, support = reconstruct(sparse)
        smoothed = normalized_smooth(surface, support & domain, GRADIENT_SIGMA_PX)
        fields[key] = surface
        fields[f"{key}_support"] = support & domain
        fields[f"{key}_smoothed_for_gradient"] = smoothed
    fields["domain"] = domain
    fields["boundary"] = boundary_thick
    fields["x_axis"] = x_axis
    fields["y_axis"] = y_axis
    return fields


def sample_template(fields: dict, xy: np.ndarray) -> np.ndarray:
    x_axis, y_axis = fields["x_axis"], fields["y_axis"]
    # RegularGridInterpolator needs increasing axes; y_axis is decreasing, so flip both axis and array.
    y_increasing = y_axis[::-1]
    row_col = np.column_stack([xy[:, 1], xy[:, 0]])
    predicted = np.column_stack([
        RegularGridInterpolator((y_increasing, x_axis), fields["azimuth_deg"][::-1, :],
                                 bounds_error=False, fill_value=np.nan)(row_col),
        RegularGridInterpolator((y_increasing, x_axis), fields["elevation_deg"][::-1, :],
                                 bounds_error=False, fill_value=np.nan)(row_col),
    ])
    return predicted


def pseudo_huber(values: np.ndarray) -> np.ndarray:
    return 2.0 * (np.sqrt(1.0 + np.square(values)) - 1.0)


def fit_candidate_garrett(fields: dict, landmarks: pd.DataFrame, reflection: int, seed: int) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    target = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    ccf_center = ccf.mean(axis=0)

    def transform(parameters: np.ndarray) -> np.ndarray:
        center_x, center_y, theta, scale_x, scale_y, shear = parameters
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matrix = rotation @ np.array([[scale_x, shear * scale_y], [0.0, reflection * scale_y]])
        return (ccf - ccf_center) @ matrix.T + np.array([center_x, center_y]), matrix

    def objective(parameters: np.ndarray) -> float:
        xy, _ = transform(parameters)
        predicted = sample_template(fields, xy)
        valid = np.isfinite(predicted).all(axis=1)
        if valid.sum() < 3:
            return 50.0
        retinal = float(np.mean(pseudo_huber((predicted[valid] - target[valid]) / 10.0)))
        domain_penalty = float(2.0 * (1 - valid.mean()) ** 2)
        return retinal + domain_penalty

    # Panel-width units are ~O(1) across roughly the cortical extent (a few mm), so scale is
    # ~O(0.1-1) panel-units/mm; bounds are left wide since there is no strong prior here.
    bounds = [(-0.3, 0.3), (-0.3, 0.3), (-np.pi, np.pi), (0.02, 2.0), (0.02, 2.0), (-0.8, 0.8)]
    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=300, popsize=15, tol=1e-8, polish=True, workers=1, updating="immediate",
    )
    xy, matrix = transform(result.x)
    predicted = sample_template(fields, xy)
    return {
        "objective": float(result.fun), "parameters": result.x, "reflection": reflection,
        "ccf_center": ccf_center, "template_center": np.array([result.x[0], result.x[1]]),
        "matrix_px_per_mm": matrix, "xy": xy, "predicted": predicted, "target": target,
        "retinal_median_vector_error_deg": float(np.median(np.linalg.norm(predicted - target, axis=1))),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = build_fields()
    np.savez_compressed(
        OUTPUT / "garrett_smoothed_fields.npz",
        azimuth_deg=fields["azimuth_deg"], elevation_deg=fields["elevation_deg"],
        azimuth_smoothed_for_gradient_deg=fields["azimuth_deg_smoothed_for_gradient"],
        elevation_smoothed_for_gradient_deg=fields["elevation_deg_smoothed_for_gradient"],
        domain=fields["domain"], boundary=fields["boundary"],
        x_axis=fields["x_axis"], y_axis=fields["y_axis"],
    )
    print("azimuth support fraction of domain:",
          float((fields["azimuth_deg_support"] & fields["domain"]).sum() / max(fields["domain"].sum(), 1)))
    print("elevation support fraction of domain:",
          float((fields["elevation_deg_support"] & fields["domain"]).sum() / max(fields["domain"].sum(), 1)))

    cells, audit = production_support()
    session_cells = cells.loc[cells.session_id.eq(SESSION_ID)].copy()
    landmarks = make_landmarks(session_cells)
    print(f"session {SESSION_ID}: {len(landmarks)} probe/area landmarks")

    candidates = [fit_candidate_garrett(fields, landmarks, reflection, 20260818 + i) for i, reflection in enumerate((-1, 1))]
    best = min(candidates, key=lambda item: item["objective"])
    print(f"selected reflection={best['reflection']}, objective={best['objective']:.3f}, "
          f"median vector error={best['retinal_median_vector_error_deg']:.1f} deg")

    manifest = {
        "session_id": SESSION_ID,
        "grid_n": GRID_N, "margin": MARGIN, "gradient_sigma_px": GRADIENT_SIGMA_PX,
        "candidates": [{"reflection": c["reflection"], "objective": c["objective"],
                         "median_vector_error_deg": c["retinal_median_vector_error_deg"]} for c in candidates],
        "selected_reflection": best["reflection"],
        "selected_parameters": best["parameters"].tolist(),
        "ccf_center_ap_ml_mm": best["ccf_center"].tolist(),
        "template_center_xy": best["template_center"].tolist(),
        "matrix_panel_units_per_mm": best["matrix_px_per_mm"].tolist(),
    }
    (OUTPUT / f"session_{SESSION_ID}_fit_manifest.json").write_text(json.dumps(manifest, indent=2))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(fields["azimuth_deg_smoothed_for_gradient"], origin="upper",
              extent=[fields["x_axis"][0], fields["x_axis"][-1], fields["y_axis"][-1], fields["y_axis"][0]],
              cmap="viridis")
    ax.contour(fields["domain"].astype(float), levels=[.5], colors="k", linewidths=.6,
               extent=[fields["x_axis"][0], fields["x_axis"][-1], fields["y_axis"][-1], fields["y_axis"][0]])
    ax.scatter(best["xy"][:, 0], best["xy"][:, 1], c="red", s=40, edgecolor="white")
    ax.set(title=f"Garrett smoothed azimuth + session {SESSION_ID} landmarks", xlabel="panel x", ylabel="panel y")
    fig.savefig(OUTPUT / "Figure_garrett_smoothed_azimuth_with_landmarks.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
