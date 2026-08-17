#!/usr/bin/env python3
"""Test whether within-probe RF gradients improve one-session registration.

The independent validation grain is a whole penetration.  Each observed
gradient is represented as the RF change expected from one end of that probe's
sampled AP/ML span to the other, avoiding unstable deg/mm leverage for very
short tracks.  This is an exploratory six-fold comparison, not a population
estimate.
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

from audit_session_cell_gradient_registration_evidence import oriented_probe_coordinate
from fit_deformable_retinotopy_checkpoint import (
    calibrated_values,
    evaluate_base,
    fit_base_model,
    jacobian_ratios,
    map_to_template,
)
from register_allen_session_to_zhuang import build_template, sample_template


ROOT = Path(__file__).resolve().parents[1]
SESSION = 798911424
PILOT = ROOT / f"artifacts/retinotopy_registration_pilot/session_{SESSION}"
DEFAULT_CELLS = PILOT / "cell_scatter_CCF_RF_support.csv"
DEFAULT_LANDMARKS = PILOT / "penetration_landmarks.csv"
DEFAULT_AFFINE = PILOT / "run_manifest.json"
DEFAULT_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
DEFAULT_OUTPUT = PILOT / "cell_gradient_registration_checkpoint2"
AREA_COLORS = {
    "VISp": "#2864a8", "VISl": "#d78318", "VISal": "#b33f62",
    "VISrl": "#5f8f3e", "VISam": "#7356a8",
}


def gradient_evidence(cells: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe_id, frame in cells.groupby("ecephys_probe_id", sort=True):
        t, direction = oriented_probe_coordinate(frame)
        span = float(np.ptp(t))
        rf = frame[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
        design = np.column_stack([np.ones(len(t)), t])
        coefficients = np.linalg.lstsq(design, rf, rcond=None)[0]
        # Bootstrap endpoint-change uncertainty. A fixed 3-degree floor keeps
        # cell resampling precision from standing in for systematic RF error.
        rng = np.random.default_rng(int(probe_id) % (2**32 - 1))
        changes = np.empty((1000, 2))
        for draw in range(len(changes)):
            take = rng.integers(0, len(t), size=len(t))
            fit = np.linalg.lstsq(
                np.column_stack([np.ones(len(t)), t[take]]), rf[take], rcond=None
            )[0]
            changes[draw] = fit[1] * span
        change_se = np.nanstd(changes, axis=0, ddof=1)
        landmark = landmarks.loc[landmarks.ecephys_probe_id.eq(probe_id)].iloc[0]
        rows.append({
            "ecephys_probe_id": int(probe_id),
            "area": landmark.ecephys_structure_acronym,
            "cells": len(frame),
            "ccf_ap_mm": landmark.ccf_ap_mm,
            "ccf_ml_mm": landmark.ccf_ml_mm,
            "direction_ap": direction[0],
            "direction_ml": direction[1],
            "span_mm": span,
            "observed_delta_azimuth_deg": coefficients[1, 0] * span,
            "observed_delta_elevation_deg": coefficients[1, 1] * span,
            "delta_azimuth_scale_deg": max(3.0, float(change_se[0])),
            "delta_elevation_scale_deg": max(3.0, float(change_se[1])),
        })
    return pd.DataFrame(rows)


def predict_gradient_changes(
    parameters: np.ndarray,
    evidence: pd.DataFrame,
    template: dict,
    ccf_center: np.ndarray,
    template_center: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    centers = evidence[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    directions = evidence[["direction_ap", "direction_ml"]].to_numpy(float)
    half_spans = evidence.span_mm.to_numpy(float)[:, None] / 2.0
    minus = centers - directions * half_spans
    plus = centers + directions * half_spans

    def values(ccf: np.ndarray) -> np.ndarray:
        xy = map_to_template(
            ccf, parameters[:6], ccf_center, template_center, affine_matrix
        )
        raw, _, _ = sample_template(template, xy)
        return calibrated_values(raw, parameters, True)

    return values(plus) - values(minus)


def fit_with_gradient_weight(
    train_indices: np.ndarray,
    weight: float,
    start: np.ndarray,
    ccf: np.ndarray,
    observed: np.ndarray,
    areas: list[str],
    evidence: pd.DataFrame,
    template: dict,
    ccf_center: np.ndarray,
    template_center: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    grid_ap, grid_ml = np.meshgrid(
        np.linspace(ccf[:, 0].min() - 0.15, ccf[:, 0].max() + 0.15, 8),
        np.linspace(ccf[:, 1].min() - 0.15, ccf[:, 1].max() + 0.15, 8),
    )
    geometry_grid = np.column_stack([grid_ap.ravel(), grid_ml.ravel()])
    train_evidence = evidence.iloc[train_indices]
    observed_changes = train_evidence[
        ["observed_delta_azimuth_deg", "observed_delta_elevation_deg"]
    ].to_numpy(float)
    change_scales = train_evidence[
        ["delta_azimuth_scale_deg", "delta_elevation_scale_deg"]
    ].to_numpy(float)

    def objective(parameters: np.ndarray) -> float:
        result = evaluate_base(
            parameters, train_indices, ccf, observed, areas, template, ccf_center,
            template_center, affine_matrix, True,
        )
        scaled_rf = (result["predicted"] - result["observed"]) / 10.0
        rf_loss = np.mean(2.0 * (np.sqrt(1.0 + scaled_rf**2) - 1.0))
        area_loss = 2.0 * np.mean((result["area_distance"] / 10.0) ** 2)
        outside_loss = np.mean((result["outside"] / 10.0) ** 2 + result["bounds"] ** 2)
        geometry_loss = 0.15 * np.mean((parameters[:6] / 25.0) ** 2)
        ratios = jacobian_ratios(geometry_grid, parameters[:6], affine_matrix, ccf_center)
        folding_loss = 20.0 * np.mean(np.maximum(0.20 - ratios, 0.0) ** 2)
        calibration_loss = (
            0.08 * np.mean((parameters[6:8] / 20.0) ** 2)
            + 0.08 * np.mean((parameters[8:10] / 0.5) ** 2)
        )
        predicted_changes = predict_gradient_changes(
            parameters, train_evidence, template, ccf_center, template_center, affine_matrix
        )
        scaled_change = (predicted_changes - observed_changes) / change_scales
        gradient_loss = np.mean(2.0 * (np.sqrt(1.0 + scaled_change**2) - 1.0))
        return float(
            rf_loss + area_loss + outside_loss + geometry_loss + folding_loss
            + calibration_loss + weight * gradient_loss
        )

    fit = minimize(
        objective, start, method="Powell",
        bounds=[(-80.0, 80.0)] * 6 + [(-40.0, 40.0), (-40.0, 40.0), (-0.7, 1.1), (-0.7, 1.1)],
        options={"maxiter": 2500, "xtol": 2e-6, "ftol": 2e-6},
    )
    return fit.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--affine-manifest", type=Path, default=DEFAULT_AFFINE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.cells.resolve())
    landmarks_all = pd.read_csv(args.landmarks.resolve())
    landmarks = landmarks_all.loc[landmarks_all.model.eq("joint_anatomy_rf")].copy()
    landmarks = landmarks.sort_values("ecephys_probe_id").reset_index(drop=True)
    evidence = gradient_evidence(cells, landmarks).sort_values("ecephys_probe_id").reset_index(drop=True)
    if not np.array_equal(landmarks.ecephys_probe_id, evidence.ecephys_probe_id):
        raise RuntimeError("Probe order differs between landmarks and gradient evidence")
    template = build_template(args.template.resolve())
    affine_manifest = json.loads(args.affine_manifest.resolve().read_text(encoding="utf-8"))
    affine = affine_manifest["selected_models"]["joint_anatomy_rf"]
    ccf_center = np.asarray(affine["ccf_center_ap_ml_mm"], float)
    template_center = np.asarray(affine["template_center_xy_px"], float)
    affine_matrix = np.asarray(affine["affine_matrix_xy_px_per_ap_ml_mm"], float)
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    areas = landmarks.ecephys_structure_acronym.tolist()
    all_indices = np.arange(len(landmarks))

    full_baseline, _, _ = fit_base_model(
        all_indices, True, ccf, observed, areas, template, ccf_center, template_center, affine_matrix
    )
    full_parameters = {0.0: full_baseline}
    start = full_baseline
    for weight in args.weights:
        start = fit_with_gradient_weight(
            all_indices, weight, start, ccf, observed, areas, evidence, template,
            ccf_center, template_center, affine_matrix,
        )
        full_parameters[weight] = start

    fold_rows = []
    for held_out in all_indices:
        train = all_indices[all_indices != held_out]
        baseline, _, _ = fit_base_model(
            train, True, ccf, observed, areas, template, ccf_center, template_center, affine_matrix
        )
        parameter_by_weight = {0.0: baseline}
        start = baseline
        for weight in args.weights:
            start = fit_with_gradient_weight(
                train, weight, start, ccf, observed, areas, evidence, template,
                ccf_center, template_center, affine_matrix,
            )
            parameter_by_weight[weight] = start
        for weight, parameters in parameter_by_weight.items():
            center_result = evaluate_base(
                parameters, np.array([held_out]), ccf, observed, areas, template,
                ccf_center, template_center, affine_matrix, True,
            )
            center_prediction = center_result["predicted"][0]
            gradient_prediction = predict_gradient_changes(
                parameters, evidence.iloc[[held_out]], template, ccf_center,
                template_center, affine_matrix,
            )[0]
            observed_change = evidence.iloc[held_out][
                ["observed_delta_azimuth_deg", "observed_delta_elevation_deg"]
            ].to_numpy(float)
            fold_rows.append({
                "held_out_probe_id": int(landmarks.iloc[held_out].ecephys_probe_id),
                "held_out_area": areas[held_out],
                "gradient_weight": weight,
                "observed_azimuth_deg": observed[held_out, 0],
                "observed_elevation_deg": observed[held_out, 1],
                "predicted_azimuth_deg": center_prediction[0],
                "predicted_elevation_deg": center_prediction[1],
                "held_out_center_vector_error_deg": float(np.linalg.norm(center_prediction - observed[held_out])),
                "observed_delta_azimuth_deg": observed_change[0],
                "observed_delta_elevation_deg": observed_change[1],
                "predicted_delta_azimuth_deg": gradient_prediction[0],
                "predicted_delta_elevation_deg": gradient_prediction[1],
                "held_out_gradient_vector_error_deg": float(np.linalg.norm(gradient_prediction - observed_change)),
            })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "leave_one_probe_out_gradient_weight_scan.csv", index=False)
    summary = (
        folds.groupby("gradient_weight", as_index=False)
        .agg(
            median_center_error_deg=("held_out_center_vector_error_deg", "median"),
            mean_center_error_deg=("held_out_center_vector_error_deg", "mean"),
            median_gradient_error_deg=("held_out_gradient_vector_error_deg", "median"),
            mean_gradient_error_deg=("held_out_gradient_vector_error_deg", "mean"),
        )
        .sort_values("gradient_weight")
    )
    summary.to_csv(output / "gradient_weight_summary.csv", index=False)
    positive = summary.loc[summary.gradient_weight.gt(0)].sort_values(
        ["median_center_error_deg", "mean_center_error_deg", "gradient_weight"]
    )
    selected_weight = float(positive.iloc[0].gradient_weight)

    full_rows = []
    for weight in (0.0, selected_weight):
        predictions = predict_gradient_changes(
            full_parameters[weight], evidence, template, ccf_center, template_center, affine_matrix
        )
        for index, row in evidence.iterrows():
            full_rows.append({
                **row.to_dict(),
                "gradient_weight": weight,
                "predicted_delta_azimuth_deg": predictions[index, 0],
                "predicted_delta_elevation_deg": predictions[index, 1],
                "gradient_vector_error_deg": float(np.linalg.norm(
                    predictions[index] - row[["observed_delta_azimuth_deg", "observed_delta_elevation_deg"]].to_numpy(float)
                )),
            })
    full = pd.DataFrame(full_rows)
    full.to_csv(output / "full_fit_probe_gradient_comparison.csv", index=False)

    chosen_folds = folds.loc[folds.gradient_weight.isin([0.0, selected_weight])].copy()
    probes = evidence.ecephys_probe_id.tolist()
    labels = [str(x)[-3:] for x in probes]
    x = np.arange(len(probes))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for component, axis, label in (
        ("azimuth", axes[0, 0], "end-to-end azimuth change (°)"),
        ("elevation", axes[0, 1], "end-to-end elevation change (°)"),
    ):
        observed_component = evidence[f"observed_delta_{component}_deg"].to_numpy(float)
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.scatter(x, observed_component, marker="o", s=65, color="#111111", label="observed cells", zorder=4)
        for weight, marker, color, name in (
            (0.0, "x", "#777777", "centroids only"),
            (selected_weight, "D", "#7b3294", f"+ gradients (w={selected_weight:g})"),
        ):
            local = full.loc[full.gradient_weight.eq(weight)].set_index("ecephys_probe_id").loc[probes]
            axis.scatter(x, local[f"predicted_delta_{component}_deg"], marker=marker, s=58,
                         color=color, label=name, zorder=3)
        axis.set_xticks(x, labels)
        axis.set_xlabel("probe ID suffix")
        axis.set_ylabel(label)
        axis.grid(color="#dddddd", linewidth=0.45)
    axes[0, 0].legend(fontsize=8)

    axis = axes[0, 2]
    for weight, marker, color, name in (
        (0.0, "o", "#888888", "centroids only"),
        (selected_weight, "D", "#7b3294", f"+ gradients (w={selected_weight:g})"),
    ):
        local = full.loc[full.gradient_weight.eq(weight)].set_index("ecephys_probe_id").loc[probes]
        axis.scatter(x, local.gradient_vector_error_deg, marker=marker, s=55, color=color, label=name)
    axis.set_xticks(x, labels)
    axis.set(xlabel="probe ID suffix", ylabel="full-fit gradient-vector error (°)", title="Does the fitted map reproduce cell gradients?")
    axis.grid(color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    for metric, axis, title in (
        ("held_out_center_vector_error_deg", axes[1, 0], "Held-out probe RF center"),
        ("held_out_gradient_vector_error_deg", axes[1, 1], "Held-out probe gradient"),
    ):
        for weight, offset, color, name in (
            (0.0, -0.12, "#888888", "centroids only"),
            (selected_weight, 0.12, "#7b3294", "+ gradients"),
        ):
            local = chosen_folds.loc[chosen_folds.gradient_weight.eq(weight)].set_index("held_out_probe_id").loc[probes]
            axis.scatter(x + offset, local[metric], s=55, color=color, label=name)
        axis.set_xticks(x, labels)
        axis.set(xlabel="held-out probe ID suffix", ylabel="vector error (°)", title=title)
        axis.grid(color="#dddddd", linewidth=0.45)
    axes[1, 0].legend(fontsize=8)

    axis = axes[1, 2]
    axis.plot(summary.gradient_weight, summary.median_center_error_deg, "o-", label="median center error")
    axis.plot(summary.gradient_weight, summary.median_gradient_error_deg, "s-", label="median gradient error")
    axis.axvline(selected_weight, color="#7b3294", linestyle="--", linewidth=1)
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set(xlabel="gradient constraint weight", ylabel="six-fold median error (°)", title="Exploratory constraint-weight sensitivity")
    axis.grid(color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    figure.suptitle(
        "Session 798911424: do cell-derived directional gradients improve Zhuang registration?\n"
        "Whole-probe leave-one-out validation; gradients encoded as end-to-end RF change across each sampled AP–ML span",
        fontsize=14,
    )
    figure.savefig(output / "Figure_gradient_constrained_registration_test.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "checkpoint": "multi-case gradient-constrained registration comparison",
        "status": "exploratory; gradient weight selected on the same six LOPO folds shown",
        "session_id": SESSION,
        "independent_validation_unit": "probe penetration",
        "models": {
            "centroids_only": "six equally weighted penetration RF medians",
            "gradient_constrained": "same model plus cell-derived directional end-to-end RF changes",
        },
        "gradient_weights_tested": [0.0] + args.weights,
        "selected_positive_weight": selected_weight,
        "gradient_uncertainty": "cell bootstrap endpoint-change SE with 3-degree systematic floor per RF axis",
        "limitations": [
            "Only six penetrations are available.",
            "Constraint weight selection and evaluation reuse the same six exploratory folds.",
            "Each probe supplies one directional derivative, not a complete 2D gradient.",
            "The model tested is the calibrated quadratic geometry model, without the post-hoc smooth residual field.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_weight": selected_weight, "summary": summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
