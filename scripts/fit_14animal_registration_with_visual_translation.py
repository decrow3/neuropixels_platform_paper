#!/usr/bin/env python3
"""Refit 14-animal anatomy registration with a latent visual-field translation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from build_14animal_retinotopy_registration import (
    OUTPUT as ORIGINAL_OUTPUT,
    SEED,
    SESSIONS,
    TEMPLATE_PATH,
    make_landmarks,
)
from register_allen_session_to_zhuang import (
    AREA_COLORS,
    AREA_LABELS,
    affine,
    build_template,
    fit_candidate,
    sample_template,
    target_rf,
    transform_ccf,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_CELLS = ORIGINAL_OUTPUT / "production_registration_cell_support.csv.gz"
RF_SIZE_CELLS = (
    ORIGINAL_OUTPUT / "rf_size_map_alignment/primary_uncensored_interior_rf_size_cells.csv.gz"
)
OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_translation_v1"
OFFSET_SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
REFLECTION = 1
CONVENTION = "native"
AREA_WEIGHT = 2.0
OFFSET_BOUND_DEG = 40.0
OFFSET_PENALTY_WEIGHT = 0.01
HVA_BANDWIDTHS = (0.20, 0.35, 0.55, 0.80, 1.15)


def profiled_offset(target: np.ndarray, predicted: np.ndarray, scale: float) -> np.ndarray:
    raw = np.nanmedian(target - predicted, axis=0)
    return np.clip(scale * raw, -OFFSET_BOUND_DEG, OFFSET_BOUND_DEG)


def fit_profiled_translation(
    landmarks: pd.DataFrame,
    template: dict,
    start_fit: dict,
    offset_scale: float,
) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    target = target_rf(observed, CONVENTION)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = ccf.mean(axis=0)
    height, width = template["domain"].shape
    old_center = start_fit["ccf_center"]
    old_template_center = start_fit["template_center"]
    old_matrix = start_fit["matrix_px_per_mm"]
    start = start_fit["parameters"].copy()
    start[:2] = old_template_center + (ccf_center - old_center) @ old_matrix.T

    def components(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xy = transform_ccf(ccf, ccf_center, parameters, REFLECTION)
        predicted, outside, bounds_distance = sample_template(template, xy)
        offset = profiled_offset(target, predicted, offset_scale)
        residual = predicted + offset - target
        retinal = float(np.mean(2.0 * (np.sqrt(1.0 + (residual / 10.0) ** 2) - 1.0)))
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
        offset_penalty = OFFSET_PENALTY_WEIGHT * float(np.mean(np.square(offset / 20.0)))
        objective = retinal + AREA_WEIGHT * area_penalty + domain_penalty + geometry_penalty + offset_penalty
        return objective, xy, predicted, offset, area_distances

    bounds = [
        (40.0, width - 30.0), (20.0, height - 20.0), (-np.pi, np.pi),
        (70.0, 320.0), (70.0, 320.0), (-.8, .8),
    ]
    result = minimize(
        lambda parameters: components(parameters)[0], start,
        method="Powell", bounds=bounds,
        options={"maxiter": 1000, "xtol": 1e-6, "ftol": 1e-6},
    )
    objective, xy, predicted, offset, area_distances = components(result.x)
    center, matrix = affine(result.x, REFLECTION)
    residual = predicted + offset - target
    return {
        "objective": objective, "parameters": result.x, "reflection": REFLECTION,
        "convention": CONVENTION, "ccf_center": ccf_center,
        "template_center": center, "matrix_px_per_mm": matrix,
        "xy": xy, "target": target, "predicted_template": predicted,
        "predicted_observed": predicted + offset, "visual_offset": offset,
        "residuals": residual, "area_distances": area_distances,
        "median_vector_error_deg": float(np.median(np.linalg.norm(residual, axis=1))),
        "mean_area_distance_px": float(area_distances.mean()),
    }


def fit_all_candidates(cells: pd.DataFrame, template: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    full_fits = {}
    for session_position, session_id in enumerate(SESSIONS):
        local = cells.loc[cells.session_id.eq(session_id)].copy()
        landmarks = make_landmarks(local)
        old_fit = fit_candidate(
            template, landmarks, AREA_WEIGHT, CONVENTION, REFLECTION,
            SEED + 1000 + session_position,
        )
        for offset_scale in OFFSET_SCALES:
            full = fit_profiled_translation(landmarks, template, old_fit, offset_scale)
            full_fits[(session_id, offset_scale)] = full
            for held_index in range(len(landmarks)):
                held = landmarks.iloc[held_index]
                train = landmarks.drop(index=held_index).reset_index(drop=True)
                fold = fit_profiled_translation(train, template, full, offset_scale)
                held_ccf = held[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)[None, :]
                xy = transform_ccf(held_ccf, fold["ccf_center"], fold["parameters"], REFLECTION)
                template_prediction, _, _ = sample_template(template, xy)
                observed = target_rf(
                    held[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)[None, :],
                    CONVENTION,
                )[0]
                prediction = template_prediction[0] + fold["visual_offset"]
                rows.append({
                    "session_id": session_id, "held_out_probe_id": int(held.ecephys_probe_id),
                    "held_out_area": held.ecephys_structure_acronym,
                    "offset_scale": offset_scale,
                    "visual_offset_azimuth_deg": fold["visual_offset"][0],
                    "visual_offset_elevation_deg": fold["visual_offset"][1],
                    "observed_azimuth_deg": observed[0], "observed_elevation_deg": observed[1],
                    "predicted_azimuth_deg": prediction[0], "predicted_elevation_deg": prediction[1],
                    "vector_error_deg": float(np.linalg.norm(prediction - observed)),
                    "mean_training_area_distance_px": fold["mean_area_distance_px"],
                })
        print(f"latent visual translations complete: {session_position + 1}/{len(SESSIONS)}", flush=True)
    return pd.DataFrame(rows), full_fits


def nested_select(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    selected_rows = []
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
        selected_rows.append(folds.loc[
            folds.session_id.eq(outer_session) & folds.offset_scale.eq(selected_scale)
        ].assign(selected_offset_scale=selected_scale))
    return pd.concat(selected_rows, ignore_index=True), pd.concat(selection_rows, ignore_index=True)


def production_scale(folds: pd.DataFrame) -> float:
    summary = folds.groupby("offset_scale", observed=True).agg(
        median_error=("vector_error_deg", "median"), mean_error=("vector_error_deg", "mean")
    ).reset_index().sort_values(["median_error", "mean_error", "offset_scale"])
    return float(summary.iloc[0].offset_scale)


def corrected_cell_table(
    cells: pd.DataFrame, template: dict, full_fits: dict, scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    transform_rows = []
    for session_id in SESSIONS:
        local = cells.loc[cells.session_id.eq(session_id)].copy()
        fit = full_fits[(session_id, scale)]
        ccf = local[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        xy = transform_ccf(ccf, fit["ccf_center"], fit["parameters"], REFLECTION)
        template_rf, _, _ = sample_template(template, xy)
        observed = target_rf(
            local[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float), CONVENTION
        )
        local["latent_visual_offset_azimuth_deg"] = fit["visual_offset"][0]
        local["latent_visual_offset_elevation_deg"] = fit["visual_offset"][1]
        local["translation_corrected_azimuth_deg"] = observed[:, 0] - fit["visual_offset"][0]
        local["translation_corrected_elevation_deg"] = observed[:, 1] - fit["visual_offset"][1]
        local["latent_template_x_px"] = xy[:, 0]
        local["latent_template_y_px"] = xy[:, 1]
        local["latent_template_azimuth_deg"] = template_rf[:, 0]
        local["latent_template_elevation_deg"] = template_rf[:, 1]
        rows.append(local)
        transform_rows.append({
            "session_id": session_id, "offset_scale": scale,
            "visual_offset_azimuth_deg": fit["visual_offset"][0],
            "visual_offset_elevation_deg": fit["visual_offset"][1],
            "visual_offset_magnitude_deg": float(np.linalg.norm(fit["visual_offset"])),
            "median_landmark_vector_error_deg": fit["median_vector_error_deg"],
            "mean_area_distance_px": fit["mean_area_distance_px"],
            **{f"anatomy_parameter_{i}": value for i, value in enumerate(fit["parameters"])},
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(transform_rows)


def balanced_predict(train: pd.DataFrame, test: pd.DataFrame, columns: tuple[str, str], bandwidth: float) -> np.ndarray:
    x_train = train[list(columns)].to_numpy(float)
    x_test = test[list(columns)].to_numpy(float)
    center = np.nanmedian(x_train, axis=0)
    scale = np.nanquantile(x_train, .75, axis=0) - np.nanquantile(x_train, .25, axis=0)
    scale = np.where(scale > 1e-9, scale, np.nanstd(x_train, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    x_test = (x_test - center) / scale
    predictions = []
    for _, animal in train.groupby("session_id", observed=True):
        x_animal = (animal[list(columns)].to_numpy(float) - center) / scale
        distance2 = cdist(x_test, x_animal, metric="sqeuclidean")
        weights = np.exp(-.5 * (distance2 - distance2.min(axis=1, keepdims=True)) / bandwidth**2)
        predictions.append((weights @ animal.group_centered_log2_area.to_numpy(float)) / weights.sum(axis=1))
    return np.mean(np.vstack(predictions), axis=0)


def score_size(test: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    observed = test.group_centered_log2_area.to_numpy(float)
    return {
        "rho": float(spearmanr(observed, prediction).statistic),
        "mae": float(np.median(np.abs(observed - prediction))),
        "constant_mae": float(np.median(np.abs(observed))),
    }


def select_bandwidth(train: pd.DataFrame, columns: tuple[str, str]) -> float:
    candidates = []
    for bandwidth in HVA_BANDWIDTHS:
        rhos = []
        for session_id in sorted(train.session_id.unique()):
            test = train.loc[train.session_id.eq(session_id)]
            inner = train.loc[~train.session_id.eq(session_id)]
            rhos.append(score_size(test, balanced_predict(inner, test, columns, bandwidth))["rho"])
        candidates.append((float(np.nanmedian(rhos)), bandwidth))
    return max(candidates)[1]


def evaluate_rf_size(corrected_cells: pd.DataFrame) -> pd.DataFrame:
    size = pd.read_csv(RF_SIZE_CELLS, low_memory=False)
    corrected_columns = corrected_cells[[
        "ecephys_unit_id", "translation_corrected_azimuth_deg", "translation_corrected_elevation_deg"
    ]]
    size = size.merge(corrected_columns, on="ecephys_unit_id", how="inner", validate="one_to_one")
    size["cortical_group"] = np.where(size.ecephys_structure_acronym.eq("VISp"), "V1", "HVA")
    size["group_centered_log2_area"] = size.log2_rf_area_deg2 - size.groupby(
        ["session_id", "cortical_group"], observed=True
    ).log2_rf_area_deg2.transform("median")
    systems = {
        "raw_rf_location": ("common_azimuth_deg", "common_elevation_deg"),
        "translation_corrected_rf_location": (
            "translation_corrected_azimuth_deg", "translation_corrected_elevation_deg"
        ),
    }
    rows = []
    for cortical_group in ("V1", "HVA"):
        group = size.loc[size.cortical_group.eq(cortical_group)]
        for outer_session in sorted(group.session_id.unique()):
            train = group.loc[~group.session_id.eq(outer_session)]
            test = group.loc[group.session_id.eq(outer_session)]
            for system, columns in systems.items():
                bandwidth = select_bandwidth(train, columns)
                metrics = score_size(test, balanced_predict(train, test, columns, bandwidth))
                rows.append({
                    "session_id": outer_session, "cortical_group": cortical_group,
                    "system": system, "cells": len(test), "bandwidth_iqr_units": bandwidth,
                    "spearman_rho": metrics["rho"], "mae_log2": metrics["mae"],
                    "constant_mae_log2": metrics["constant_mae"],
                    "mae_gain_vs_constant_log2": metrics["constant_mae"] - metrics["mae"],
                })
    return pd.DataFrame(rows)


def select_cases(nested: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = nested.merge(
        baseline[["session_id", "held_out_probe_id", "vector_error_deg"]].rename(columns={"vector_error_deg": "baseline_error"}),
        on=["session_id", "held_out_probe_id"], validate="one_to_one",
    )
    merged["improvement_deg"] = merged.baseline_error - merged.vector_error_deg
    median = merged.improvement_deg.median()
    roles = [
        ("largest held-out gain", merged.improvement_deg.idxmax(), "maximum RF-vector error reduction"),
        ("typical effect", (merged.improvement_deg - median).abs().idxmin(), "closest to median RF-vector error reduction"),
        ("translation failure", merged.improvement_deg.idxmin(), "minimum RF-vector error reduction"),
    ]
    rows = []
    for role, index, criterion in roles:
        row = merged.loc[index]
        rows.append({
            "session_id": int(row.session_id), "held_out_probe_id": int(row.held_out_probe_id),
            "area": row.held_out_area, "selection_role": role, "criterion": criterion,
            "criterion_value": row.improvement_deg, "baseline_error_deg": row.baseline_error,
            "translation_error_deg": row.vector_error_deg,
            "provenance": "algorithmic selection from nested held-out-probe comparison",
        })
    return pd.DataFrame(rows)


def render(
    folds: pd.DataFrame, nested: pd.DataFrame, selection: pd.DataFrame,
    transforms: pd.DataFrame, rf_size: pd.DataFrame, template: dict, output: Path,
) -> None:
    baseline = folds.loc[folds.offset_scale.eq(0)].copy()
    paired = nested.merge(
        baseline[["session_id", "held_out_probe_id", "vector_error_deg"]].rename(columns={"vector_error_deg": "baseline_error"}),
        on=["session_id", "held_out_probe_id"], validate="one_to_one",
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10), constrained_layout=True)
    axis = axes[0, 0]
    summary = folds.groupby("offset_scale", observed=True).vector_error_deg.agg(["median", "mean"]).reset_index()
    axis.plot(summary.offset_scale, summary["median"], marker="o", color="#2f6b9a", label="Median")
    axis.plot(summary.offset_scale, summary["mean"], marker="s", color="#d18b2c", label="Mean")
    axis.set(xlabel="Fraction of profiled visual offset retained", ylabel="Held-out probe RF-vector error (deg)", title="Translation shrinkage diagnostic")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.scatter(paired.baseline_error, paired.vector_error_deg, s=40, c=[AREA_COLORS[a] for a in paired.held_out_area], alpha=.75)
    lo = min(paired.baseline_error.min(), paired.vector_error_deg.min()); hi = max(paired.baseline_error.max(), paired.vector_error_deg.max())
    axis.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
    axis.set(xlabel="No-offset held-out error (deg)", ylabel="Latent-offset held-out error (deg)", title="Nested animal-held-out model selection", aspect="equal")

    axis = axes[0, 2]
    axis.quiver(np.zeros(len(transforms)), np.zeros(len(transforms)),
                transforms.visual_offset_azimuth_deg, transforms.visual_offset_elevation_deg,
                angles="xy", scale_units="xy", scale=1, color="#2f6b9a", alpha=.8)
    for row in transforms.itertuples(): axis.text(row.visual_offset_azimuth_deg+.4, row.visual_offset_elevation_deg+.4, str(row.session_id)[-3:], fontsize=7)
    extent = max(10, float(np.abs(transforms[["visual_offset_azimuth_deg", "visual_offset_elevation_deg"]]).to_numpy().max()) + 3)
    axis.axhline(0, color="#777777", lw=.8); axis.axvline(0, color="#777777", lw=.8)
    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xlabel="Observed − atlas azimuth (deg)", ylabel="Observed − atlas elevation (deg)", title="Production animal visual offsets")

    axis = axes[1, 0]
    for outer, local in selection.loc[selection.selected].groupby("outer_session_id"):
        axis.scatter(str(outer)[-3:], local.offset_scale.iloc[0], color="#2f6b9a", s=40)
    axis.set(xlabel="Outer held-out animal suffix", ylabel="Selected offset scale", title="Does translation selection generalize across animals?", ylim=(-.05, 1.05))
    axis.tick_params(axis="x", rotation=45)

    for column, cortical_group in enumerate(("V1", "HVA"), start=1):
        axis = axes[1, column]
        local = rf_size.loc[rf_size.cortical_group.eq(cortical_group)]
        wide = local.pivot(index="session_id", columns="system", values="mae_log2")
        axis.scatter(wide.raw_rf_location, wide.translation_corrected_rf_location, color="#d18b2c", s=45)
        lo = min(wide.min()); hi = max(wide.max()); axis.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
        for sid, row in wide.iterrows(): axis.text(row.raw_rf_location+.003, row.translation_corrected_rf_location, str(sid)[-3:], fontsize=7)
        axis.set(xlabel="Raw RF-location MAE (log₂)", ylabel="Translation-corrected MAE (log₂)", title=f"{cortical_group} RF-size transfer", aspect="equal")
    for axis in axes.flat: axis.grid(alpha=.14)
    fig.suptitle(
        "Anatomy-constrained registration with a latent animal visual-field translation\n"
        "RF centers fit the offset; RF size is an independent downstream validation target",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(INPUT_CELLS, low_memory=False)
    template = build_template(TEMPLATE_PATH)
    folds, full_fits = fit_all_candidates(cells, template)
    nested, selection = nested_select(folds)
    scale = production_scale(folds)
    corrected_cells, transforms = corrected_cell_table(cells, template, full_fits, scale)
    rf_size = evaluate_rf_size(corrected_cells)
    baseline = folds.loc[folds.offset_scale.eq(0)].copy()
    cases = select_cases(nested, baseline)

    folds.to_csv(OUTPUT / "all_offset_scale_leave_one_probe_out.csv", index=False)
    nested.to_csv(OUTPUT / "nested_selected_leave_one_probe_out.csv", index=False)
    selection.to_csv(OUTPUT / "nested_offset_scale_selection.csv", index=False)
    transforms.to_csv(OUTPUT / "selected_animal_visual_translations.csv", index=False)
    corrected_cells.to_csv(OUTPUT / "cells_translation_corrected_common_coordinates.csv.gz", index=False, compression="gzip")
    rf_size.to_csv(OUTPUT / "rf_size_prediction_before_after_translation.csv", index=False)
    cases.to_csv(OUTPUT / "selected_translation_case_audit.csv", index=False)
    render(
        folds, nested, selection, transforms, rf_size, template,
        OUTPUT / "Figure_14animal_latent_visual_translation_summary.png",
    )

    paired = nested.merge(
        baseline[["session_id", "held_out_probe_id", "vector_error_deg"]].rename(columns={"vector_error_deg": "baseline_error"}),
        on=["session_id", "held_out_probe_id"], validate="one_to_one",
    )
    paired["improvement"] = paired.baseline_error - paired.vector_error_deg
    size_wide = rf_size.pivot(index=["cortical_group", "session_id"], columns="system", values="mae_log2")
    manifest = {
        "status": "exploratory latent visual-translation registration",
        "shared_frame": {"azimuth_convention": CONVENTION, "reflection": REFLECTION},
        "model": {
            "visual_offset": "profiled robust median observed-minus-template RF vector, shrunk by globally/nested selected scale",
            "anatomical_anchor": "Zhuang named-area distance plus domain and affine geometry penalties",
            "candidate_offset_scales": list(OFFSET_SCALES), "production_offset_scale": scale,
        },
        "held_out_probe": {
            "folds": len(paired), "baseline_median_error_deg": float(paired.baseline_error.median()),
            "translation_median_error_deg": float(paired.vector_error_deg.median()),
            "median_improvement_deg": float(paired.improvement.median()),
            "folds_improved": int((paired.improvement > 0).sum()),
        },
        "visual_offsets": {
            "median_magnitude_deg": float(transforms.visual_offset_magnitude_deg.median()),
            "maximum_magnitude_deg": float(transforms.visual_offset_magnitude_deg.max()),
            "azimuth_range_deg": [float(transforms.visual_offset_azimuth_deg.min()), float(transforms.visual_offset_azimuth_deg.max())],
            "elevation_range_deg": [float(transforms.visual_offset_elevation_deg.min()), float(transforms.visual_offset_elevation_deg.max())],
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
            "Offset-scale and production choices are exploratory on these 14 animals; new animals are confirmatory.",
            "Area constraints identify the cortical-versus-visual translation decomposition; without them the two translations are not separately identifiable.",
            "The offset is assumed constant across all probes and visual areas within a session.",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
