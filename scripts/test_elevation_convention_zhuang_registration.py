#!/usr/bin/env python3
"""Test an independent elevation-sign convention against the frozen 14-animal Zhuang fit.

The frozen registration (`build_14animal_retinotopy_registration.py`) searches azimuth
convention ("native" vs "100 - azimuth") and cortical reflection (+-1), but never searches
an elevation sign convention -- `target_rf()` only ever flips column 0. Offline analysis of
the saved `probe_gradient_comparison.csv` found the observed-vs-Zhuang-predicted elevation
gradient is weakly NEGATIVELY correlated (Pearson r=-0.16, Spearman r=-0.17 across 68
probes), and most cleanly so in V1 alone (r=-0.39, n=14, the best-powered area) -- consistent
with a real but fixable sign convention mismatch rather than pure noise. This script adds the
missing elevation-flip candidate axis and re-selects the shared frame to test that directly.
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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import (  # noqa: E402
    AREA_LABELS, affine, build_template, sample_template, transform_ccf,
)
from build_14animal_retinotopy_registration import (  # noqa: E402
    SESSIONS, make_landmarks, production_support,
)
from audit_session_cell_gradient_registration_evidence import oriented_probe_coordinate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1" / "elevation_convention_test"
SEED = 20260817


def target_rf_v2(observed: np.ndarray, azimuth_convention: str, elevation_convention: str) -> np.ndarray:
    result = observed.copy()
    if azimuth_convention == "100_minus_azimuth":
        result[:, 0] = 100.0 - result[:, 0]
    if elevation_convention == "flipped":
        result[:, 1] = -result[:, 1]
    return result


def pseudo_huber(values: np.ndarray) -> np.ndarray:
    return 2.0 * (np.sqrt(1.0 + np.square(values)) - 1.0)


def fit_candidate_v2(
    template: dict,
    landmarks: pd.DataFrame,
    area_weight: float,
    azimuth_convention: str,
    elevation_convention: str,
    reflection: int,
    seed: int,
) -> dict:
    """Copy of register_allen_session_to_zhuang.fit_candidate, extended with an
    independent elevation-sign convention axis (target_rf_v2 instead of target_rf)."""
    from scipy.optimize import differential_evolution

    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    target = target_rf_v2(observed, azimuth_convention, elevation_convention)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = ccf.mean(axis=0)
    height, width = template["domain"].shape

    def objective(parameters: np.ndarray) -> float:
        xy = transform_ccf(ccf, ccf_center, parameters, reflection)
        predicted, outside, bounds = sample_template(template, xy)
        retinal = float(np.mean(pseudo_huber((predicted - target) / 10.0)))
        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        area_distances = np.asarray(
            [template["area_distance"][area](row_col[i:i + 1])[0] for i, area in enumerate(areas)]
        )
        area_penalty = float(np.mean(np.square(area_distances / 12.0)))
        domain_penalty = float(np.mean(np.square(outside / 15.0) + np.square(bounds)))
        scale_x, scale_y, shear = parameters[3], parameters[4], parameters[5]
        geometry_penalty = float(
            0.02 * (np.log(scale_x / 180.0) ** 2 + np.log(scale_y / 180.0) ** 2) + 0.02 * shear**2
        )
        return retinal + area_weight * area_penalty + domain_penalty + geometry_penalty

    bounds = [
        (40.0, width - 30.0), (20.0, height - 20.0), (-np.pi, np.pi),
        (70.0, 320.0), (70.0, 320.0), (-0.8, 0.8),
    ]
    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=250, popsize=12, tol=1e-7,
        polish=True, workers=1, updating="immediate",
    )
    xy = transform_ccf(ccf, ccf_center, result.x, reflection)
    predicted, outside, bounds_distance = sample_template(template, xy)
    row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
    area_distances = np.asarray(
        [template["area_distance"][area](row_col[i:i + 1])[0] for i, area in enumerate(areas)]
    )
    residuals = predicted - target
    center, matrix = affine(result.x, reflection)
    return {
        "objective": float(result.fun), "parameters": result.x, "reflection": reflection,
        "azimuth_convention": azimuth_convention, "elevation_convention": elevation_convention,
        "ccf_center": ccf_center, "template_center": center, "matrix_px_per_mm": matrix,
        "xy": xy, "target": target, "predicted": predicted, "residuals": residuals,
        "area_distances": area_distances,
        "retinal_rmse_deg": float(np.sqrt(np.mean(np.square(residuals)))),
        "retinal_median_vector_error_deg": float(np.median(np.linalg.norm(residuals, axis=1))),
        "mean_area_distance_px": float(area_distances.mean()),
        "landmarks_in_named_area": int(np.sum(area_distances <= 1.5)),
    }


def gradient_table_v2(
    session_id: int, cells: pd.DataFrame, landmarks: pd.DataFrame, fit: dict, template: dict
) -> pd.DataFrame:
    """Copy of build_cross_animal_retinotopy_registration.gradient_table, extended to also
    sign-correct the observed elevation gradient when elevation_convention == 'flipped'."""
    rows = []
    for probe_id, frame in cells.groupby("ecephys_probe_id", sort=True):
        t, direction = oriented_probe_coordinate(frame)
        span = float(np.ptp(t))
        rf_native = frame[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
        coefficient = np.linalg.lstsq(np.column_stack([np.ones(len(t)), t]), rf_native, rcond=None)[0][1]
        observed_delta = coefficient * span
        if fit["azimuth_convention"] == "100_minus_azimuth":
            observed_delta[0] *= -1.0
        if fit["elevation_convention"] == "flipped":
            observed_delta[1] *= -1.0
        landmark = landmarks.loc[landmarks.ecephys_probe_id.eq(probe_id)].iloc[0]
        center = np.array([landmark.ccf_ap_mm, landmark.ccf_ml_mm], dtype=float)
        endpoints = np.vstack([center - direction * span / 2.0, center + direction * span / 2.0])
        endpoint_xy = transform_ccf(endpoints, fit["ccf_center"], fit["parameters"], fit["reflection"])
        predicted_endpoints, _, _ = sample_template(template, endpoint_xy)
        predicted_delta = predicted_endpoints[1] - predicted_endpoints[0]
        rows.append({
            "session_id": session_id, "ecephys_probe_id": int(probe_id),
            "area": landmark.ecephys_structure_acronym, "cells": len(frame), "ccf_span_mm": span,
            "observed_delta_azimuth_deg": observed_delta[0], "observed_delta_elevation_deg": observed_delta[1],
            "predicted_delta_azimuth_deg": predicted_delta[0], "predicted_delta_elevation_deg": predicted_delta[1],
            "azimuth_sign_match": bool(np.sign(predicted_delta[0]) == np.sign(observed_delta[0])),
            "elevation_sign_match": bool(np.sign(predicted_delta[1]) == np.sign(observed_delta[1])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(TEMPLATE_PATH)
    cells, audit = production_support()

    session_data: dict[int, dict] = {}
    candidate_rows = []
    for position, session_id in enumerate(SESSIONS):
        local_cells = cells.loc[cells.session_id.eq(session_id)].copy()
        landmarks = make_landmarks(local_cells)
        fits = {}
        candidate_number = 0
        for azimuth_convention in ("native", "100_minus_azimuth"):
            for elevation_convention in ("native", "flipped"):
                for reflection in (-1, 1):
                    fit = fit_candidate_v2(
                        template, landmarks, 2.0, azimuth_convention, elevation_convention,
                        reflection, SEED + position * 10 + candidate_number,
                    )
                    candidate_number += 1
                    fits[(azimuth_convention, elevation_convention, reflection)] = fit
                    candidate_rows.append({
                        "session_id": session_id,
                        "azimuth_convention": azimuth_convention,
                        "elevation_convention": elevation_convention,
                        "cortical_reflection": reflection,
                        "objective": fit["objective"],
                        "penetration_median_vector_error_deg": fit["retinal_median_vector_error_deg"],
                        "penetrations_in_named_area": fit["landmarks_in_named_area"],
                        "penetrations": len(landmarks),
                    })
        session_data[session_id] = {"cells": local_cells, "landmarks": landmarks, "fits": fits}
        print(f"candidate fits complete: {position + 1}/{len(SESSIONS)}", flush=True)

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUTPUT / "shared_frame_candidate_fits_with_elevation_axis.csv", index=False)
    frame_summary = candidates.groupby(
        ["azimuth_convention", "elevation_convention", "cortical_reflection"]
    ).agg(median_objective=("objective", "median"), mean_objective=("objective", "mean")).reset_index()
    frame_summary = frame_summary.sort_values(["median_objective", "mean_objective"])
    frame_summary.to_csv(OUTPUT / "shared_frame_selection_with_elevation_axis.csv", index=False)
    best = frame_summary.iloc[0]
    selected = (str(best.azimuth_convention), str(best.elevation_convention), int(best.cortical_reflection))
    print("selected frame:", selected)

    gradient_tables = []
    for session_id, item in session_data.items():
        fit = item["fits"][selected]
        gradient_tables.append(gradient_table_v2(session_id, item["cells"], item["landmarks"], fit, template))
    gradients = pd.concat(gradient_tables, ignore_index=True)
    gradients.to_csv(OUTPUT / "probe_gradient_comparison_with_elevation_axis.csv", index=False)

    old_gradients = pd.read_csv(
        ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1" / "probe_gradient_comparison.csv"
    )

    def summarize(frame: pd.DataFrame, label: str) -> dict:
        o = frame.observed_delta_elevation_deg.to_numpy(float)
        p = frame.predicted_delta_elevation_deg.to_numpy(float)
        pr, ppv = pearsonr(o, p)
        sr, spv = spearmanr(o, p)
        return {
            "frame": label,
            "elevation_sign_agreement": float(frame.elevation_sign_match.mean()),
            "azimuth_sign_agreement": float(frame.azimuth_sign_match.mean()),
            "elevation_pearson_r": float(pr), "elevation_pearson_p": float(ppv),
            "elevation_spearman_r": float(sr), "elevation_spearman_p": float(spv),
        }

    comparison = pd.DataFrame([
        summarize(old_gradients, "frozen_14animal_no_elevation_axis"),
        summarize(gradients, "with_elevation_axis"),
    ])
    comparison.to_csv(OUTPUT / "elevation_axis_before_after_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, frame, title in (
        (axes[0], old_gradients, "frozen (no elevation axis)"),
        (axes[1], gradients, "with elevation-flip axis"),
    ):
        ax.axhline(0, color=".85", lw=.6)
        ax.axvline(0, color=".85", lw=.6)
        ax.scatter(frame.observed_delta_elevation_deg, frame.predicted_delta_elevation_deg, s=28, alpha=.75)
        ax.set(xlabel="observed elevation gradient (deg)", ylabel="Zhuang-predicted elevation gradient (deg)",
               title=title)
    fig.suptitle("Per-probe elevation gradient: observed vs. Zhuang-predicted, before/after adding the flip axis")
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_elevation_axis_before_after.png", dpi=180)
    plt.close(fig)

    manifest = {
        "status": "diagnostic: does an independent elevation-sign convention fix the frozen 14-animal elevation-gradient disagreement",
        "sessions": list(SESSIONS),
        "candidate_axes": {
            "azimuth_convention": ["native", "100_minus_azimuth"],
            "elevation_convention": ["native", "flipped"],
            "cortical_reflection": [-1, 1],
        },
        "selected_frame": {
            "azimuth_convention": selected[0], "elevation_convention": selected[1], "cortical_reflection": selected[2],
        },
        "before_after": comparison.to_dict(orient="records"),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
