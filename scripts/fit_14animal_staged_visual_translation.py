#!/usr/bin/env python3
"""Fit cortical geometry first, then estimate one visual translation per animal.

The geometry objective contains only within-animal RF differences, so a common
visual-field translation cannot be absorbed by the anatomy-to-template warp.
After freezing that geometry, a robust observed-minus-template shift is
estimated from the training penetrations.  Leave-one-probe-out evaluation keeps
the held probe out of both stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from build_14animal_retinotopy_registration import (
    OUTPUT as ORIGINAL_OUTPUT,
    SEED,
    SESSIONS,
    TEMPLATE_PATH,
    make_landmarks,
)
from fit_14animal_registration_with_visual_translation import evaluate_rf_size
from register_allen_session_to_zhuang import (
    AREA_COLORS,
    affine,
    build_template,
    fit_candidate,
    sample_template,
    target_rf,
    transform_ccf,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_CELLS = ORIGINAL_OUTPUT / "production_registration_cell_support.csv.gz"
JOINT_OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_translation_v1"
OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_staged_translation_v1"
OFFSET_SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
REFLECTION = 1
CONVENTION = "native"
AREA_WEIGHT = 2.0
OFFSET_BOUND_DEG = 40.0


def robust_loss(residual: np.ndarray, scale: float = 10.0) -> float:
    return float(np.mean(2.0 * (np.sqrt(1.0 + (residual / scale) ** 2) - 1.0)))


def pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(count, k=1)


def raw_profiled_offset(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.clip(np.nanmedian(target - predicted, axis=0), -OFFSET_BOUND_DEG, OFFSET_BOUND_DEG)


def fit_translation_invariant_geometry(
    landmarks: pd.DataFrame,
    template: dict,
    start_fit: dict,
) -> dict:
    """Fit anatomy warp using RF gradients, never absolute RF position."""
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    target = target_rf(observed, CONVENTION)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = ccf.mean(axis=0)
    height, width = template["domain"].shape
    first, second = pair_indices(len(landmarks))

    start = start_fit["parameters"].copy()
    start[:2] = start_fit["template_center"] + (
        (ccf_center - start_fit["ccf_center"]) @ start_fit["matrix_px_per_mm"].T
    )

    def components(parameters: np.ndarray):
        xy = transform_ccf(ccf, ccf_center, parameters, REFLECTION)
        predicted, outside, bounds_distance = sample_template(template, xy)
        predicted_difference = predicted[first] - predicted[second]
        target_difference = target[first] - target[second]
        pair_residual = predicted_difference - target_difference
        gradient_loss = robust_loss(pair_residual)

        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        area_distances = np.asarray([
            template["area_distance"][area](row_col[i:i + 1])[0]
            for i, area in enumerate(areas)
        ])
        area_penalty = float(np.mean(np.square(area_distances / 12.0)))
        domain_penalty = float(np.mean(np.square(outside / 15.0) + np.square(bounds_distance)))
        scale_x, scale_y, shear = parameters[3], parameters[4], parameters[5]
        geometry_penalty = float(
            .02 * (np.log(scale_x / 180.0) ** 2 + np.log(scale_y / 180.0) ** 2)
            + .02 * shear**2
        )
        objective = gradient_loss + AREA_WEIGHT * area_penalty + domain_penalty + geometry_penalty
        return objective, xy, predicted, pair_residual, area_distances

    bounds = [
        (40.0, width - 30.0), (20.0, height - 20.0), (-np.pi, np.pi),
        (70.0, 320.0), (70.0, 320.0), (-.8, .8),
    ]
    result = minimize(
        lambda parameters: components(parameters)[0],
        start,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 1000, "xtol": 1e-6, "ftol": 1e-6},
    )
    objective, xy, predicted, pair_residual, area_distances = components(result.x)
    center, matrix = affine(result.x, REFLECTION)
    offset = raw_profiled_offset(target, predicted)
    absolute_residual = predicted + offset - target
    return {
        "objective": objective,
        "parameters": result.x,
        "ccf_center": ccf_center,
        "template_center": center,
        "matrix_px_per_mm": matrix,
        "xy": xy,
        "target": target,
        "predicted_template": predicted,
        "raw_visual_offset": offset,
        "pair_residuals": pair_residual,
        "area_distances": area_distances,
        "pair_rmse_deg": float(np.sqrt(np.mean(pair_residual**2))),
        "median_centered_vector_error_deg": float(np.median(np.linalg.norm(absolute_residual, axis=1))),
        "mean_area_distance_px": float(area_distances.mean()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def independent_start(landmarks: pd.DataFrame, template: dict, seed: int) -> dict:
    """Generate a training-only numerical start; it is not part of the staged objective."""
    return fit_candidate(template, landmarks, AREA_WEIGHT, CONVENTION, REFLECTION, seed)


def fit_all_folds(cells: pd.DataFrame, template: dict) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    full_fits: dict[int, dict] = {}
    for session_position, session_id in enumerate(SESSIONS):
        local = cells.loc[cells.session_id.eq(session_id)].copy()
        landmarks = make_landmarks(local)
        full_start = independent_start(landmarks, template, SEED + 3000 + session_position)
        full_fits[session_id] = fit_translation_invariant_geometry(landmarks, template, full_start)

        for held_index in range(len(landmarks)):
            held = landmarks.iloc[held_index]
            train = landmarks.drop(index=held_index).reset_index(drop=True)
            # Both initialization and optimization see training penetrations only.
            fold_start = independent_start(
                train, template, SEED + 10000 + session_position * 100 + held_index
            )
            fold = fit_translation_invariant_geometry(train, template, fold_start)
            held_ccf = held[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)[None, :]
            xy = transform_ccf(held_ccf, fold["ccf_center"], fold["parameters"], REFLECTION)
            template_prediction, _, _ = sample_template(template, xy)
            observed = target_rf(
                held[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)[None, :],
                CONVENTION,
            )[0]
            for offset_scale in OFFSET_SCALES:
                offset = offset_scale * fold["raw_visual_offset"]
                prediction = template_prediction[0] + offset
                rows.append({
                    "session_id": session_id,
                    "held_out_probe_id": int(held.ecephys_probe_id),
                    "held_out_area": held.ecephys_structure_acronym,
                    "offset_scale": offset_scale,
                    "raw_visual_offset_azimuth_deg": fold["raw_visual_offset"][0],
                    "raw_visual_offset_elevation_deg": fold["raw_visual_offset"][1],
                    "visual_offset_azimuth_deg": offset[0],
                    "visual_offset_elevation_deg": offset[1],
                    "observed_azimuth_deg": observed[0],
                    "observed_elevation_deg": observed[1],
                    "predicted_azimuth_deg": prediction[0],
                    "predicted_elevation_deg": prediction[1],
                    "vector_error_deg": float(np.linalg.norm(prediction - observed)),
                    "training_pair_rmse_deg": fold["pair_rmse_deg"],
                    "mean_training_area_distance_px": fold["mean_area_distance_px"],
                    "optimizer_success": fold["optimizer_success"],
                })
        print(f"staged geometry folds complete: {session_position + 1}/{len(SESSIONS)}", flush=True)
    return pd.DataFrame(rows), full_fits


def nested_select(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    selection_rows = []
    for outer_session in SESSIONS:
        training = folds.loc[~folds.session_id.eq(outer_session)]
        summary = training.groupby("offset_scale", observed=True).agg(
            median_error_deg=("vector_error_deg", "median"),
            mean_error_deg=("vector_error_deg", "mean"),
        ).reset_index().sort_values(["median_error_deg", "mean_error_deg", "offset_scale"])
        selected_scale = float(summary.iloc[0].offset_scale)
        summary["outer_session_id"] = outer_session
        summary["selected"] = summary.offset_scale.eq(selected_scale)
        selection_rows.append(summary)
        selected_rows.append(
            folds.loc[
                folds.session_id.eq(outer_session) & folds.offset_scale.eq(selected_scale)
            ].assign(selected_offset_scale=selected_scale)
        )
    return pd.concat(selected_rows, ignore_index=True), pd.concat(selection_rows, ignore_index=True)


def production_scale(folds: pd.DataFrame) -> float:
    summary = folds.groupby("offset_scale", observed=True).agg(
        median_error=("vector_error_deg", "median"),
        mean_error=("vector_error_deg", "mean"),
    ).reset_index().sort_values(["median_error", "mean_error", "offset_scale"])
    return float(summary.iloc[0].offset_scale)


def corrected_cells_and_transforms(
    cells: pd.DataFrame, template: dict, full_fits: dict, scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    transform_rows = []
    for session_id in SESSIONS:
        local = cells.loc[cells.session_id.eq(session_id)].copy()
        fit = full_fits[session_id]
        ccf = local[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        xy = transform_ccf(ccf, fit["ccf_center"], fit["parameters"], REFLECTION)
        template_rf, _, _ = sample_template(template, xy)
        observed = target_rf(
            local[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float), CONVENTION
        )
        offset = scale * fit["raw_visual_offset"]
        local["raw_visual_offset_azimuth_deg"] = fit["raw_visual_offset"][0]
        local["raw_visual_offset_elevation_deg"] = fit["raw_visual_offset"][1]
        local["latent_visual_offset_azimuth_deg"] = offset[0]
        local["latent_visual_offset_elevation_deg"] = offset[1]
        local["translation_corrected_azimuth_deg"] = observed[:, 0] - offset[0]
        local["translation_corrected_elevation_deg"] = observed[:, 1] - offset[1]
        local["staged_template_x_px"] = xy[:, 0]
        local["staged_template_y_px"] = xy[:, 1]
        local["staged_template_azimuth_deg"] = template_rf[:, 0]
        local["staged_template_elevation_deg"] = template_rf[:, 1]
        cell_rows.append(local)
        transform_rows.append({
            "session_id": session_id,
            "offset_scale": scale,
            "raw_visual_offset_azimuth_deg": fit["raw_visual_offset"][0],
            "raw_visual_offset_elevation_deg": fit["raw_visual_offset"][1],
            "visual_offset_azimuth_deg": offset[0],
            "visual_offset_elevation_deg": offset[1],
            "visual_offset_magnitude_deg": float(np.linalg.norm(offset)),
            "pair_rmse_deg": fit["pair_rmse_deg"],
            "median_centered_vector_error_deg": fit["median_centered_vector_error_deg"],
            "mean_area_distance_px": fit["mean_area_distance_px"],
            "optimizer_success": fit["optimizer_success"],
            **{f"anatomy_parameter_{i}": value for i, value in enumerate(fit["parameters"])},
        })
    return pd.concat(cell_rows, ignore_index=True), pd.DataFrame(transform_rows)


def offset_stability(folds: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    raw = folds.loc[folds.offset_scale.eq(1)].copy()
    rows = []
    for session_id, local in raw.groupby("session_id", observed=True):
        az = local.raw_visual_offset_azimuth_deg
        el = local.raw_visual_offset_elevation_deg
        full = production.loc[production.session_id.eq(session_id)].iloc[0]
        rows.append({
            "session_id": session_id,
            "folds": len(local),
            "raw_offset_azimuth_full_deg": full.raw_visual_offset_azimuth_deg,
            "raw_offset_elevation_full_deg": full.raw_visual_offset_elevation_deg,
            "raw_offset_azimuth_fold_sd_deg": az.std(ddof=1),
            "raw_offset_elevation_fold_sd_deg": el.std(ddof=1),
            "raw_offset_azimuth_fold_iqr_deg": az.quantile(.75) - az.quantile(.25),
            "raw_offset_elevation_fold_iqr_deg": el.quantile(.75) - el.quantile(.25),
            "raw_offset_vector_fold_rms_deviation_deg": float(np.sqrt(np.mean(
                (az - az.mean()) ** 2 + (el - el.mean()) ** 2
            ))),
        })
    return pd.DataFrame(rows)


def comparison_table(
    folds: pd.DataFrame,
    staged: pd.DataFrame,
    joint: pd.DataFrame,
    conventional_no_offset: pd.DataFrame,
) -> pd.DataFrame:
    baseline = folds.loc[folds.offset_scale.eq(0), [
        "session_id", "held_out_probe_id", "held_out_area", "vector_error_deg"
    ]].rename(columns={"vector_error_deg": "staged_no_offset_error_deg"})
    result = baseline.merge(
        staged[["session_id", "held_out_probe_id", "vector_error_deg", "selected_offset_scale"]]
        .rename(columns={"vector_error_deg": "staged_error_deg"}),
        on=["session_id", "held_out_probe_id"], validate="one_to_one",
    )
    result = result.merge(
        joint[["session_id", "held_out_probe_id", "vector_error_deg", "selected_offset_scale"]]
        .rename(columns={
            "vector_error_deg": "joint_error_deg",
            "selected_offset_scale": "joint_selected_offset_scale",
        }),
        on=["session_id", "held_out_probe_id"], how="left", validate="one_to_one",
    )
    result = result.merge(
        conventional_no_offset[["session_id", "held_out_probe_id", "vector_error_deg"]]
        .rename(columns={"vector_error_deg": "conventional_no_translation_error_deg"}),
        on=["session_id", "held_out_probe_id"], how="left", validate="one_to_one",
    )
    result["staged_gain_vs_no_offset_deg"] = (
        result.staged_no_offset_error_deg - result.staged_error_deg
    )
    result["staged_gain_vs_joint_deg"] = result.joint_error_deg - result.staged_error_deg
    result["staged_gain_vs_conventional_no_translation_deg"] = (
        result.conventional_no_translation_error_deg - result.staged_error_deg
    )
    return result


def select_cases(comparison: pd.DataFrame) -> pd.DataFrame:
    median = comparison.staged_gain_vs_no_offset_deg.median()
    roles = [
        ("largest staged gain", comparison.staged_gain_vs_no_offset_deg.idxmax()),
        ("typical staged effect", (comparison.staged_gain_vs_no_offset_deg - median).abs().idxmin()),
        ("worst staged failure", comparison.staged_gain_vs_no_offset_deg.idxmin()),
    ]
    known_joint_failure = comparison.loc[
        comparison.session_id.eq(757216464) & comparison.held_out_area.eq("VISp")
    ]
    if not known_joint_failure.empty:
        roles.append(("prior joint VISp failure", known_joint_failure.index[0]))
    rows = []
    for role, index in roles:
        row = comparison.loc[index].to_dict()
        row["selection_role"] = role
        row["provenance"] = "algorithmic audit selection; prior failure was pre-specified from joint model"
        rows.append(row)
    return pd.DataFrame(rows)


def render(
    folds: pd.DataFrame,
    staged: pd.DataFrame,
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    transforms: pd.DataFrame,
    stability: pd.DataFrame,
    rf_size: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)

    axis = axes[0, 0]
    curve = folds.groupby("offset_scale", observed=True).vector_error_deg.agg(["median", "mean"]).reset_index()
    axis.plot(curve.offset_scale, curve["median"], "o-", color="#2f6b9a", label="Median")
    axis.plot(curve.offset_scale, curve["mean"], "s-", color="#d18b2c", label="Mean")
    axis.set(xlabel="Retained fraction of post-geometry offset", ylabel="Held-out RF-vector error (deg)", title="Staged shrinkage diagnostic")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.scatter(
        comparison.staged_no_offset_error_deg, comparison.staged_error_deg,
        s=42, c=[AREA_COLORS[a] for a in comparison.held_out_area], alpha=.78,
    )
    lo = min(comparison.staged_no_offset_error_deg.min(), comparison.staged_error_deg.min())
    hi = max(comparison.staged_no_offset_error_deg.max(), comparison.staged_error_deg.max())
    axis.plot([lo, hi], [lo, hi], "--", color="#555555", lw=.9)
    axis.set(xlabel="Staged geometry, no offset (deg)", ylabel="Staged geometry + offset (deg)", title="Animal-held-out scale selection", aspect="equal")

    axis = axes[0, 2]
    ordered = comparison.sort_values("staged_gain_vs_joint_deg")
    colors = [AREA_COLORS[a] for a in ordered.held_out_area]
    axis.scatter(np.arange(len(ordered)), ordered.staged_gain_vs_joint_deg, c=colors, s=32, alpha=.8, label="Prior joint − staged")
    axis.scatter(
        np.arange(len(ordered)), ordered.staged_gain_vs_conventional_no_translation_deg,
        color="#555555", marker="x", s=24, alpha=.55, label="No translation − staged",
    )
    axis.axhline(0, color="#555555", ls="--", lw=.9)
    axis.set(xlabel="Held-out probes, sorted by joint comparison", ylabel="Reference error − staged error (deg)", title="Staged versus prior models")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.quiver(
        np.zeros(len(transforms)), np.zeros(len(transforms)),
        transforms.visual_offset_azimuth_deg, transforms.visual_offset_elevation_deg,
        angles="xy", scale_units="xy", scale=1, color="#2f6b9a", alpha=.8,
    )
    for row in transforms.itertuples():
        axis.text(row.visual_offset_azimuth_deg + .25, row.visual_offset_elevation_deg + .25, str(row.session_id)[-3:], fontsize=7)
    extent = max(10.0, float(np.abs(transforms[["visual_offset_azimuth_deg", "visual_offset_elevation_deg"]].to_numpy()).max()) + 2)
    axis.axhline(0, color="#777777", lw=.8); axis.axvline(0, color="#777777", lw=.8)
    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xlabel="Azimuth offset (deg)", ylabel="Elevation offset (deg)", title="Production offsets after frozen geometry")

    axis = axes[1, 1]
    axis.scatter(stability.raw_offset_vector_fold_rms_deviation_deg, transforms.visual_offset_magnitude_deg, color="#7356a8", s=46)
    for row in stability.itertuples():
        magnitude = transforms.loc[transforms.session_id.eq(row.session_id), "visual_offset_magnitude_deg"].iloc[0]
        axis.text(row.raw_offset_vector_fold_rms_deviation_deg + .1, magnitude, str(row.session_id)[-3:], fontsize=7)
    axis.set(xlabel="LOPO raw-offset RMS instability (deg)", ylabel="Retained full-data offset magnitude (deg)", title="Can individual probes move the offset?")

    axis = axes[1, 2]
    size_wide = rf_size.pivot(index=["cortical_group", "session_id"], columns="system", values="mae_log2").reset_index()
    for group, marker, color in (("V1", "o", "#2864a8"), ("HVA", "s", "#d78318")):
        local = size_wide.loc[size_wide.cortical_group.eq(group)]
        axis.scatter(local.raw_rf_location, local.translation_corrected_rf_location, marker=marker, color=color, s=44, label=group)
    lo = min(size_wide.raw_rf_location.min(), size_wide.translation_corrected_rf_location.min())
    hi = max(size_wide.raw_rf_location.max(), size_wide.translation_corrected_rf_location.max())
    axis.plot([lo, hi], [lo, hi], "--", color="#555555", lw=.9)
    axis.set(xlabel="Raw RF-location size MAE (log2)", ylabel="Staged-corrected size MAE (log2)", title="Independent RF-size validation", aspect="equal")
    axis.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=.14)
    selected_scales = selection.loc[selection.selected, "offset_scale"]
    fig.suptitle(
        "Geometry first, animal visual translation second\n"
        f"68 held-out penetrations; outer-animal selected scales: {dict(selected_scales.value_counts().sort_index())}",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def area_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison.groupby("held_out_area", observed=True).agg(
        folds=("staged_error_deg", "size"),
        no_offset_median_error_deg=("staged_no_offset_error_deg", "median"),
        staged_median_error_deg=("staged_error_deg", "median"),
        joint_median_error_deg=("joint_error_deg", "median"),
        conventional_no_translation_median_error_deg=("conventional_no_translation_error_deg", "median"),
        staged_median_gain_vs_no_offset_deg=("staged_gain_vs_no_offset_deg", "median"),
        staged_mean_gain_vs_no_offset_deg=("staged_gain_vs_no_offset_deg", "mean"),
        staged_median_gain_vs_joint_deg=("staged_gain_vs_joint_deg", "median"),
        staged_median_gain_vs_conventional_no_translation_deg=("staged_gain_vs_conventional_no_translation_deg", "median"),
    ).reset_index()


def refresh_saved_report() -> None:
    """Rebuild comparison tables/figure without repeating expensive geometry fits."""
    folds = pd.read_csv(OUTPUT / "all_offset_scale_leave_one_probe_out.csv")
    staged = pd.read_csv(OUTPUT / "nested_selected_leave_one_probe_out.csv")
    selection = pd.read_csv(OUTPUT / "nested_offset_scale_selection.csv")
    transforms = pd.read_csv(OUTPUT / "selected_animal_visual_translations.csv")
    stability = pd.read_csv(OUTPUT / "leave_one_probe_offset_stability.csv")
    rf_size = pd.read_csv(OUTPUT / "rf_size_prediction_before_after_staged_translation.csv")
    joint = pd.read_csv(JOINT_OUTPUT / "nested_selected_leave_one_probe_out.csv")
    joint_candidates = pd.read_csv(JOINT_OUTPUT / "all_offset_scale_leave_one_probe_out.csv")
    conventional_no_offset = joint_candidates.loc[joint_candidates.offset_scale.eq(0)].copy()
    comparison = comparison_table(folds, staged, joint, conventional_no_offset)
    area_summary(comparison).to_csv(OUTPUT / "held_out_probe_area_summary.csv", index=False)
    select_cases(comparison).to_csv(OUTPUT / "selected_staged_case_audit.csv", index=False)
    comparison.to_csv(OUTPUT / "staged_no_offset_joint_fold_comparison.csv", index=False)
    render(
        folds, staged, comparison, selection, transforms, stability, rf_size,
        OUTPUT / "Figure_14animal_staged_visual_translation_summary.png",
    )
    manifest_path = OUTPUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    held = manifest["held_out_probe"]
    held.update({
        "conventional_no_translation_median_error_deg": float(comparison.conventional_no_translation_error_deg.median()),
        "staged_median_gain_vs_conventional_no_translation_deg": float(comparison.staged_gain_vs_conventional_no_translation_deg.median()),
        "staged_folds_improved_vs_conventional_no_translation": int((comparison.staged_gain_vs_conventional_no_translation_deg > 0).sum()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(INPUT_CELLS, low_memory=False)
    template = build_template(TEMPLATE_PATH)
    folds, full_fits = fit_all_folds(cells, template)
    # Persist the expensive geometry fits before any downstream reporting.
    folds.to_csv(OUTPUT / "all_offset_scale_leave_one_probe_out.csv", index=False)
    staged, selection = nested_select(folds)
    scale = production_scale(folds)
    corrected_cells, transforms = corrected_cells_and_transforms(cells, template, full_fits, scale)
    stability = offset_stability(folds, transforms)
    rf_size = evaluate_rf_size(corrected_cells)
    joint = pd.read_csv(JOINT_OUTPUT / "nested_selected_leave_one_probe_out.csv")
    joint_candidates = pd.read_csv(JOINT_OUTPUT / "all_offset_scale_leave_one_probe_out.csv")
    conventional_no_offset = joint_candidates.loc[joint_candidates.offset_scale.eq(0)].copy()
    comparison = comparison_table(folds, staged, joint, conventional_no_offset)
    areas = area_summary(comparison)
    cases = select_cases(comparison)

    staged.to_csv(OUTPUT / "nested_selected_leave_one_probe_out.csv", index=False)
    selection.to_csv(OUTPUT / "nested_offset_scale_selection.csv", index=False)
    transforms.to_csv(OUTPUT / "selected_animal_visual_translations.csv", index=False)
    stability.to_csv(OUTPUT / "leave_one_probe_offset_stability.csv", index=False)
    comparison.to_csv(OUTPUT / "staged_no_offset_joint_fold_comparison.csv", index=False)
    areas.to_csv(OUTPUT / "held_out_probe_area_summary.csv", index=False)
    cases.to_csv(OUTPUT / "selected_staged_case_audit.csv", index=False)
    corrected_cells.to_csv(OUTPUT / "cells_staged_translation_corrected.csv.gz", index=False, compression="gzip")
    rf_size.to_csv(OUTPUT / "rf_size_prediction_before_after_staged_translation.csv", index=False)
    render(
        folds, staged, comparison, selection, transforms, stability, rf_size,
        OUTPUT / "Figure_14animal_staged_visual_translation_summary.png",
    )

    size_wide = rf_size.pivot(index=["cortical_group", "session_id"], columns="system", values="mae_log2")
    manifest = {
        "status": "exploratory staged geometry-first visual-translation checkpoint",
        "model": {
            "geometry_stage": "translation-invariant pairwise RF-vector differences plus named-area, domain, and affine geometry constraints",
            "translation_stage": "robust median observed-minus-frozen-template RF vector",
            "held_out_rule": "held penetration excluded from both geometry and translation estimation",
            "candidate_offset_scales": list(OFFSET_SCALES),
            "production_offset_scale": scale,
        },
        "held_out_probe": {
            "folds": len(comparison),
            "staged_no_offset_median_error_deg": float(comparison.staged_no_offset_error_deg.median()),
            "staged_translation_median_error_deg": float(comparison.staged_error_deg.median()),
            "joint_translation_median_error_deg": float(comparison.joint_error_deg.median()),
            "conventional_no_translation_median_error_deg": float(comparison.conventional_no_translation_error_deg.median()),
            "staged_median_gain_vs_no_offset_deg": float(comparison.staged_gain_vs_no_offset_deg.median()),
            "staged_mean_gain_vs_no_offset_deg": float(comparison.staged_gain_vs_no_offset_deg.mean()),
            "staged_folds_improved_vs_no_offset": int((comparison.staged_gain_vs_no_offset_deg > 0).sum()),
            "staged_median_gain_vs_joint_deg": float(comparison.staged_gain_vs_joint_deg.median()),
            "staged_folds_improved_vs_joint": int((comparison.staged_gain_vs_joint_deg > 0).sum()),
            "staged_median_gain_vs_conventional_no_translation_deg": float(comparison.staged_gain_vs_conventional_no_translation_deg.median()),
            "staged_folds_improved_vs_conventional_no_translation": int((comparison.staged_gain_vs_conventional_no_translation_deg > 0).sum()),
        },
        "offset_stability": {
            "median_raw_LOPO_rms_deviation_deg": float(stability.raw_offset_vector_fold_rms_deviation_deg.median()),
            "maximum_raw_LOPO_rms_deviation_deg": float(stability.raw_offset_vector_fold_rms_deviation_deg.max()),
            "median_retained_offset_magnitude_deg": float(transforms.visual_offset_magnitude_deg.median()),
        },
        "rf_size_validation": {
            group: {
                "median_raw_mae_log2": float(size_wide.loc[group].raw_rf_location.median()),
                "median_corrected_mae_log2": float(size_wide.loc[group].translation_corrected_rf_location.median()),
                "median_paired_gain_log2": float((size_wide.loc[group].raw_rf_location - size_wide.loc[group].translation_corrected_rf_location).median()),
                "animals_improved": int((size_wide.loc[group].raw_rf_location > size_wide.loc[group].translation_corrected_rf_location).sum()),
            }
            for group in ("V1", "HVA")
        },
        "limitations": [
            "The numerical start uses absolute RF position but is training-only; the converged staged objective contains no absolute RF term.",
            "With only three to six penetrations per animal, area constraints still materially identify the cortical warp.",
            "Offset-scale choices are exploratory on these 14 animals; unseen animals remain the confirmatory test.",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-report", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_report:
        refresh_saved_report()
    else:
        main()
