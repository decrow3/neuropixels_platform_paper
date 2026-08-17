#!/usr/bin/env python3
"""Exploratory fixed-anatomy, deformable-retinotopy checkpoint for one session."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from register_allen_session_to_zhuang import build_template, load_session, sample_template


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 798911424
SUPPORT = (
    ROOT
    / "artifacts/allen_multisession_rf_validation_v1/07_registration_readiness"
    / "rf_size_visual_anatomy_unit_support.csv"
)
UNITS = ROOT / "data/unit_table.csv"
TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
PILOT = ROOT / f"artifacts/retinotopy_registration_pilot/session_{SESSION_ID}"
OUTPUT = PILOT / "deformable_map_checkpoint1"

AREA_MARKERS = {"VISp": "o", "VISl": "s", "VISal": "^", "VISrl": "D", "VISam": "P"}
AREA_LABELS = {"VISp": "V1", "VISl": "LM", "VISal": "AL", "VISrl": "RL", "VISam": "AM"}
MODEL_LABELS = {
    "geometry_only": "Geometry-only warp",
    "range_expansion": "Warp + global RF calibration",
    "smooth_residual": "Warp + calibration + smooth RF residual",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", type=int, default=SESSION_ID)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--units", type=Path, default=UNITS)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def quadratic_basis(ccf_ap_ml: np.ndarray, center: np.ndarray) -> np.ndarray:
    scaled = (np.asarray(ccf_ap_ml, dtype=float) - center) / 0.8
    ap, ml = scaled[:, 0], scaled[:, 1]
    return np.column_stack([ap**2, ap * ml, ml**2])


def map_to_template(
    ccf_ap_ml: np.ndarray,
    geometry_parameters: np.ndarray,
    ccf_center: np.ndarray,
    template_center: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    baseline = template_center + (np.asarray(ccf_ap_ml) - ccf_center) @ affine_matrix.T
    correction = quadratic_basis(ccf_ap_ml, ccf_center) @ geometry_parameters.reshape(3, 2)
    return baseline + correction


def jacobian_ratios(
    ccf_ap_ml: np.ndarray, geometry_parameters: np.ndarray, affine_matrix: np.ndarray, ccf_center: np.ndarray
) -> np.ndarray:
    scaled = (np.asarray(ccf_ap_ml, dtype=float) - ccf_center) / 0.8
    ap, ml = scaled[:, 0], scaled[:, 1]
    coefficients = geometry_parameters.reshape(3, 2)
    derivative_ap = (2 * ap[:, None] * coefficients[0] + ml[:, None] * coefficients[1]) / 0.8
    derivative_ml = (ap[:, None] * coefficients[1] + 2 * ml[:, None] * coefficients[2]) / 0.8
    jacobians = np.repeat(affine_matrix[None, :, :], len(ccf_ap_ml), axis=0)
    jacobians[:, :, 0] += derivative_ap
    jacobians[:, :, 1] += derivative_ml
    return np.linalg.det(jacobians) / np.linalg.det(affine_matrix)


def calibrated_values(raw: np.ndarray, parameters: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return raw
    offsets = parameters[6:8]
    scales = np.exp(parameters[8:10])
    return np.column_stack(
        [50.0 + scales[0] * (raw[:, 0] - 50.0) + offsets[0], scales[1] * raw[:, 1] + offsets[1]]
    )


def evaluate_base(
    parameters: np.ndarray,
    indices: np.ndarray,
    ccf: np.ndarray,
    observed: np.ndarray,
    areas: list[str],
    template: dict,
    ccf_center: np.ndarray,
    template_center: np.ndarray,
    affine_matrix: np.ndarray,
    calibration: bool,
) -> dict:
    xy = map_to_template(ccf[indices], parameters[:6], ccf_center, template_center, affine_matrix)
    raw, outside, bounds = sample_template(template, xy)
    predicted = calibrated_values(raw, parameters, calibration)
    height, width = template["domain"].shape
    row_column = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
    area_distance = np.asarray(
        [template["area_distance"][areas[index]](row_column[position : position + 1])[0]
         for position, index in enumerate(indices)]
    )
    return {
        "xy": xy,
        "raw": raw,
        "predicted": predicted,
        "observed": observed[indices],
        "area_distance": area_distance,
        "outside": outside,
        "bounds": bounds,
    }


def fit_base_model(
    indices: np.ndarray,
    calibration: bool,
    ccf: np.ndarray,
    observed: np.ndarray,
    areas: list[str],
    template: dict,
    ccf_center: np.ndarray,
    template_center: np.ndarray,
    affine_matrix: np.ndarray,
) -> tuple[np.ndarray, dict, float]:
    parameter_count = 10 if calibration else 6
    start = np.zeros(parameter_count)
    grid_ap, grid_ml = np.meshgrid(
        np.linspace(ccf[:, 0].min() - 0.15, ccf[:, 0].max() + 0.15, 8),
        np.linspace(ccf[:, 1].min() - 0.15, ccf[:, 1].max() + 0.15, 8),
    )
    geometry_grid = np.column_stack([grid_ap.ravel(), grid_ml.ravel()])

    def objective(parameters: np.ndarray) -> float:
        result = evaluate_base(
            parameters, indices, ccf, observed, areas, template, ccf_center, template_center,
            affine_matrix, calibration,
        )
        scaled_rf = (result["predicted"] - result["observed"]) / 10.0
        rf_loss = np.mean(2.0 * (np.sqrt(1.0 + scaled_rf**2) - 1.0))
        area_loss = 2.0 * np.mean((result["area_distance"] / 10.0) ** 2)
        outside_loss = np.mean((result["outside"] / 10.0) ** 2 + result["bounds"] ** 2)
        geometry_loss = 0.15 * np.mean((parameters[:6] / 25.0) ** 2)
        ratios = jacobian_ratios(geometry_grid, parameters[:6], affine_matrix, ccf_center)
        folding_loss = 20.0 * np.mean(np.maximum(0.20 - ratios, 0.0) ** 2)
        calibration_loss = 0.0
        if calibration:
            calibration_loss = (
                0.08 * np.mean((parameters[6:8] / 20.0) ** 2)
                + 0.08 * np.mean((parameters[8:10] / 0.5) ** 2)
            )
        return float(rf_loss + area_loss + outside_loss + geometry_loss + folding_loss + calibration_loss)

    bounds = [(-80.0, 80.0)] * 6
    if calibration:
        bounds += [(-40.0, 40.0), (-40.0, 40.0), (-0.7, 1.1), (-0.7, 1.1)]
    fit = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 3000, "xtol": 1e-7, "ftol": 1e-7},
    )
    result = evaluate_base(
        fit.x, indices, ccf, observed, areas, template, ccf_center, template_center,
        affine_matrix, calibration,
    )
    return fit.x, result, float(fit.fun)


def kernel(x: np.ndarray, centers: np.ndarray, length_mm: float) -> np.ndarray:
    return np.exp(-0.5 * cdist(np.asarray(x, float), np.asarray(centers, float)) ** 2 / length_mm**2)


def fit_residual_field(
    train_ccf: np.ndarray, residual: np.ndarray, length_mm: float, ridge: float
) -> np.ndarray:
    matrix = kernel(train_ccf, train_ccf, length_mm)
    return np.linalg.solve(matrix + ridge * np.eye(len(train_ccf)), residual)


def predict_residual_field(
    locations: np.ndarray,
    train_ccf: np.ndarray,
    coefficients: np.ndarray,
    length_mm: float,
) -> np.ndarray:
    return kernel(locations, train_ccf, length_mm) @ coefficients


def fit_models(landmarks: pd.DataFrame, template: dict, affine_fit: dict) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = np.asarray(affine_fit["ccf_center_ap_ml_mm"], float)
    template_center = np.asarray(affine_fit["template_center_xy_px"], float)
    affine_matrix = np.asarray(affine_fit["affine_matrix_xy_px_per_ap_ml_mm"], float)
    all_indices = np.arange(len(landmarks))

    geometry_parameters, geometry_result, geometry_objective = fit_base_model(
        all_indices, False, ccf, observed, areas, template, ccf_center, template_center, affine_matrix
    )
    range_parameters, range_result, range_objective = fit_base_model(
        all_indices, True, ccf, observed, areas, template, ccf_center, template_center, affine_matrix
    )

    fold_cache = []
    fold_rows = []
    for held_out in all_indices:
        train = all_indices[all_indices != held_out]
        fold = {"held_out": int(held_out), "train": train}
        for name, calibration in (("geometry_only", False), ("range_expansion", True)):
            parameters, train_result, objective = fit_base_model(
                train, calibration, ccf, observed, areas, template, ccf_center, template_center,
                affine_matrix,
            )
            test_result = evaluate_base(
                parameters, np.array([held_out]), ccf, observed, areas, template, ccf_center,
                template_center, affine_matrix, calibration,
            )
            error = float(np.linalg.norm(test_result["predicted"][0] - observed[held_out]))
            fold[name] = {
                "parameters": parameters,
                "train_result": train_result,
                "test_result": test_result,
                "objective": objective,
            }
            fold_rows.append(
                {
                    "model": name,
                    "held_out_probe_id": int(landmarks.iloc[held_out].ecephys_probe_id),
                    "held_out_area": areas[held_out],
                    "observed_azimuth_deg": observed[held_out, 0],
                    "observed_elevation_deg": observed[held_out, 1],
                    "predicted_azimuth_deg": test_result["predicted"][0, 0],
                    "predicted_elevation_deg": test_result["predicted"][0, 1],
                    "rf_vector_error_deg": error,
                    "length_mm": np.nan,
                    "ridge": np.nan,
                }
            )
        fold_cache.append(fold)

    hyper_rows = []
    for length_mm in (0.35, 0.6, 0.9, 1.2, 1.8):
        for ridge in (0.1, 0.3, 1.0, 3.0, 10.0):
            errors = []
            predictions = []
            for fold in fold_cache:
                held_out, train = fold["held_out"], fold["train"]
                base = fold["range_expansion"]
                residual = observed[train] - base["train_result"]["predicted"]
                coefficients = fit_residual_field(ccf[train], residual, length_mm, ridge)
                adjustment = predict_residual_field(
                    ccf[[held_out]], ccf[train], coefficients, length_mm
                )[0]
                prediction = base["test_result"]["predicted"][0] + adjustment
                predictions.append(prediction)
                errors.append(float(np.linalg.norm(prediction - observed[held_out])))
            hyper_rows.append(
                {
                    "length_mm": length_mm,
                    "ridge": ridge,
                    "median_lopo_error_deg": float(np.median(errors)),
                    "mean_lopo_error_deg": float(np.mean(errors)),
                    "errors": errors,
                    "predictions": predictions,
                }
            )
    selected_hyper = min(
        hyper_rows, key=lambda row: (row["median_lopo_error_deg"], row["mean_lopo_error_deg"])
    )
    for held_out, prediction, error in zip(
        all_indices, selected_hyper["predictions"], selected_hyper["errors"]
    ):
        fold_rows.append(
            {
                "model": "smooth_residual",
                "held_out_probe_id": int(landmarks.iloc[held_out].ecephys_probe_id),
                "held_out_area": areas[held_out],
                "observed_azimuth_deg": observed[held_out, 0],
                "observed_elevation_deg": observed[held_out, 1],
                "predicted_azimuth_deg": prediction[0],
                "predicted_elevation_deg": prediction[1],
                "rf_vector_error_deg": error,
                "length_mm": selected_hyper["length_mm"],
                "ridge": selected_hyper["ridge"],
            }
        )

    full_residual = observed - range_result["predicted"]
    residual_coefficients = fit_residual_field(
        ccf, full_residual, selected_hyper["length_mm"], selected_hyper["ridge"]
    )
    smooth_prediction = range_result["predicted"] + predict_residual_field(
        ccf, ccf, residual_coefficients, selected_hyper["length_mm"]
    )
    full_models = {
        "geometry_only": {
            "parameters": geometry_parameters,
            "result": geometry_result,
            "prediction": geometry_result["predicted"],
            "objective": geometry_objective,
        },
        "range_expansion": {
            "parameters": range_parameters,
            "result": range_result,
            "prediction": range_result["predicted"],
            "objective": range_objective,
        },
        "smooth_residual": {
            "parameters": range_parameters,
            "result": range_result,
            "prediction": smooth_prediction,
            "objective": range_objective,
            "residual_coefficients": residual_coefficients,
            "length_mm": selected_hyper["length_mm"],
            "ridge": selected_hyper["ridge"],
        },
    }
    hyper_table = pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"errors", "predictions"}}
         for row in hyper_rows]
    )
    return {
        "ccf": ccf,
        "observed": observed,
        "areas": areas,
        "ccf_center": ccf_center,
        "template_center": template_center,
        "affine_matrix": affine_matrix,
        "full_models": full_models,
        "lopo": pd.DataFrame(fold_rows),
        "hyperparameters": hyper_table,
        "selected_hyper": {
            "length_mm": selected_hyper["length_mm"],
            "ridge": selected_hyper["ridge"],
            "median_lopo_error_deg": selected_hyper["median_lopo_error_deg"],
            "mean_lopo_error_deg": selected_hyper["mean_lopo_error_deg"],
        },
    }


def anatomy_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ml = np.linspace(6.75, 10.55, 330)
    ap = np.linspace(7.0, 10.05, 270)
    grid_ml, grid_ap = np.meshgrid(ml, ap)
    ccf = np.column_stack([grid_ap.ravel(), grid_ml.ravel()])
    return grid_ml, grid_ap, ccf


def model_grid(model_name: str, fit: dict, template: dict, model_bundle: dict) -> dict:
    grid_ml, grid_ap, grid_ccf = anatomy_grid()
    parameters = fit["parameters"]
    xy = map_to_template(
        grid_ccf,
        parameters[:6],
        model_bundle["ccf_center"],
        model_bundle["template_center"],
        model_bundle["affine_matrix"],
    )
    height, width = template["domain"].shape
    clipped = xy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    row_column = clipped[:, ::-1]
    raw = np.column_stack(
        [template["fields"]["azimuth_deg"](row_column), template["fields"]["altitude_deg"](row_column)]
    )
    values = calibrated_values(raw, parameters, model_name != "geometry_only")
    if model_name == "smooth_residual":
        values += predict_residual_field(
            grid_ccf,
            model_bundle["ccf"],
            fit["residual_coefficients"],
            fit["length_mm"],
        )
    domain_interpolator = RegularGridInterpolator(
        (np.arange(height), np.arange(width)), template["domain"].astype(float),
        method="nearest", bounds_error=False, fill_value=0.0,
    )
    domain = domain_interpolator(row_column).reshape(grid_ml.shape) > 0.5
    boundary_distance = ndimage.distance_transform_edt(~template["boundary"]).astype(float)
    boundary_interpolator = RegularGridInterpolator(
        (np.arange(height), np.arange(width)), boundary_distance,
        bounds_error=False, fill_value=20.0,
    )
    borders = boundary_interpolator(row_column).reshape(grid_ml.shape)
    azimuth = values[:, 0].reshape(grid_ml.shape)
    elevation = values[:, 1].reshape(grid_ml.shape)
    azimuth[~domain] = np.nan
    elevation[~domain] = np.nan
    borders[~domain] = np.nan
    ratios = jacobian_ratios(
        grid_ccf, parameters[:6], model_bundle["affine_matrix"], model_bundle["ccf_center"]
    ).reshape(grid_ml.shape)
    return {
        "ml": grid_ml,
        "ap": grid_ap,
        "azimuth": azimuth,
        "elevation": elevation,
        "borders": borders,
        "jacobian_ratio": ratios,
    }


def add_cells(axis, landmarks: pd.DataFrame, field: str, cmap: str, norm) -> None:
    for area, local in landmarks.groupby("ecephys_structure_acronym", sort=True):
        axis.scatter(
            local.ccf_ml_mm,
            local.ccf_ap_mm,
            c=local[field], cmap=cmap, norm=norm, marker=AREA_MARKERS[area],
            s=72, edgecolors="#171717", linewidths=0.7, zorder=6,
        )
    for row in landmarks.itertuples():
        axis.text(
            row.ccf_ml_mm + 0.035,
            row.ccf_ap_mm + 0.035,
            str(int(row.ecephys_probe_id))[-3:],
            fontsize=6.5,
            color="#111111",
            zorder=7,
        )


def render_map_comparison(
    landmarks: pd.DataFrame, template: dict, bundle: dict, output: Path
) -> dict:
    azimuth_norm = Normalize(0, 90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("azimuth", "rf_azimuth_deg", "viridis", azimuth_norm, np.arange(0, 91, 10)),
        ("elevation", "rf_elevation_deg", "coolwarm", elevation_norm, np.arange(-20, 41, 10)),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13.8, 15.0), constrained_layout=True)
    grid_audit = {}
    for row_index, model_name in enumerate(MODEL_LABELS):
        fit = bundle["full_models"][model_name]
        grid = model_grid(model_name, fit, template, bundle)
        grid_audit[model_name] = {
            "minimum_jacobian_ratio": float(np.nanmin(grid["jacobian_ratio"])),
            "median_jacobian_ratio": float(np.nanmedian(grid["jacobian_ratio"])),
            "maximum_jacobian_ratio": float(np.nanmax(grid["jacobian_ratio"])),
        }
        for column_index, (metric, field, cmap, norm, levels) in enumerate(panels):
            axis = axes[row_index, column_index]
            values = grid[metric]
            colors = plt.get_cmap(cmap)(norm(levels))
            axis.contour(
                grid["ml"], grid["ap"], values, levels=levels, colors=colors,
                linewidths=1.15, zorder=2,
            )
            axis.contour(
                grid["ml"], grid["ap"], grid["borders"], levels=[1.0],
                colors="#555555", linewidths=0.65, zorder=3,
            )
            add_cells(axis, landmarks, field, cmap, norm)
            if row_index == 0:
                axis.set_title(
                    "Azimuth contours and penetration medians" if metric == "azimuth"
                    else "Elevation contours and penetration medians",
                    fontsize=11,
                )
            axis.text(
                0.015, 0.02, MODEL_LABELS[model_name], transform=axis.transAxes,
                ha="left", va="bottom", fontsize=9, weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
                zorder=8,
            )
            axis.set(
                xlim=(10.55, 6.75), ylim=(10.05, 7.0),
                xlabel="Medial–lateral CCF (mm)", ylabel="Anterior–posterior CCF (mm)",
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
            axis.set_axisbelow(True)
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = figure.colorbar(scalar, ax=axis, fraction=0.035, pad=0.015)
            colorbar.set_label("degrees; shared by contours and penetrations")
    figure.suptitle(
        f"Session {int(landmarks.session_id.iloc[0])}: fixed CCF anatomy, nested deformable-map models\n"
        "Six penetration medians are the spatial evidence · Zhuang is a regularized population prior",
        fontsize=14,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return grid_audit


def render_selected_cell_map(
    cells: pd.DataFrame,
    template: dict,
    bundle: dict,
    output: Path,
) -> None:
    model_name = "smooth_residual"
    fit = bundle["full_models"][model_name]
    grid = model_grid(model_name, fit, template, bundle)
    azimuth_norm = Normalize(0, 90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    panels = (
        ("azimuth", "visual_azimuth_deg", "viridis", azimuth_norm, np.arange(0, 91, 10), "RF azimuth"),
        ("elevation", "visual_elevation_deg", "coolwarm", elevation_norm, np.arange(-20, 41, 10), "RF elevation"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), constrained_layout=True)
    for axis, (metric, field, cmap, norm, levels, title) in zip(axes, panels):
        colors = plt.get_cmap(cmap)(norm(levels))
        axis.contour(
            grid["ml"], grid["ap"], grid[metric], levels=levels, colors=colors,
            linewidths=1.15, zorder=2,
        )
        axis.contour(
            grid["ml"], grid["ap"], grid["borders"], levels=[1.0],
            colors="#555555", linewidths=0.65, zorder=3,
        )
        for area, local in cells.groupby("ecephys_structure_acronym", sort=True):
            axis.scatter(
                local.ccf_ml_mm,
                local.ccf_ap_mm,
                c=local[field], cmap=cmap, norm=norm, marker=AREA_MARKERS[area],
                s=28, alpha=0.84, edgecolors="#202020", linewidths=0.3,
                zorder=5, rasterized=True,
            )
        centers = (
            cells.groupby("ecephys_probe_id", as_index=False)
            .agg(ccf_ml_mm=("ccf_ml_mm", "median"), ccf_ap_mm=("ccf_ap_mm", "median"))
        )
        axis.scatter(
            centers.ccf_ml_mm, centers.ccf_ap_mm, marker="o", s=115,
            facecolors="none", edgecolors="#111111", linewidths=1.1, zorder=6,
        )
        for row in centers.itertuples():
            axis.text(
                row.ccf_ml_mm + 0.025, row.ccf_ap_mm + 0.025,
                str(int(row.ecephys_probe_id))[-3:], fontsize=7, color="#111111", zorder=7,
            )
        axis.set(
            title=f"{title} cells and selected-map contours",
            xlim=(10.55, 6.75), ylim=(10.05, 7.0),
            xlabel="Medial–lateral CCF (mm)", ylabel="Anterior–posterior CCF (mm)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        axis.set_axisbelow(True)
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, fraction=0.04, pad=0.02
        )
        colorbar.set_label("degrees; shared by contours and cells")
    selected = bundle["selected_hyper"]
    figure.suptitle(
        f"Session {int(cells.session_id.iloc[0])}: selected exploratory deformable map over fixed CCF cells\n"
        f"n={len(cells)} cells · six penetration landmarks · residual length {selected['length_mm']:.2f} mm "
        f"· exploratory LOPO median error {selected['median_lopo_error_deg']:.1f}°",
        fontsize=13,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_tables(landmarks: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    observed = bundle["observed"]
    for model_name, fit in bundle["full_models"].items():
        prediction = fit["prediction"]
        for index, landmark in landmarks.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "ecephys_probe_id": int(landmark.ecephys_probe_id),
                    "ecephys_structure_acronym": landmark.ecephys_structure_acronym,
                    "units": int(landmark.units),
                    "ccf_ap_mm": landmark.ccf_ap_mm,
                    "ccf_ml_mm": landmark.ccf_ml_mm,
                    "observed_azimuth_deg": observed[index, 0],
                    "observed_elevation_deg": observed[index, 1],
                    "predicted_azimuth_deg": prediction[index, 0],
                    "predicted_elevation_deg": prediction[index, 1],
                    "rf_vector_error_deg": float(np.linalg.norm(prediction[index] - observed[index])),
                    "named_area_distance_px": fit["result"]["area_distance"][index],
                }
            )
    details = pd.DataFrame(rows)
    summaries = []
    for model_name, local in details.groupby("model", sort=False):
        lopo = bundle["lopo"].loc[bundle["lopo"].model.eq(model_name)]
        fit = bundle["full_models"][model_name]
        parameters = fit["parameters"]
        summaries.append(
            {
                "model": model_name,
                "in_sample_median_rf_error_deg": local.rf_vector_error_deg.median(),
                "in_sample_mean_rf_error_deg": local.rf_vector_error_deg.mean(),
                "lopo_median_rf_error_deg": lopo.rf_vector_error_deg.median(),
                "lopo_mean_rf_error_deg": lopo.rf_vector_error_deg.mean(),
                "mean_named_area_distance_px": local.named_area_distance_px.mean(),
                "azimuth_offset_deg": parameters[6] if len(parameters) > 6 else 0.0,
                "elevation_offset_deg": parameters[7] if len(parameters) > 7 else 0.0,
                "azimuth_scale": np.exp(parameters[8]) if len(parameters) > 8 else 1.0,
                "elevation_scale": np.exp(parameters[9]) if len(parameters) > 9 else 1.0,
                "residual_length_mm": fit.get("length_mm", np.nan),
                "residual_ridge": fit.get("ridge", np.nan),
            }
        )
    return details, pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template_path = args.template.resolve()
    support_path = args.support.resolve()
    units_path = args.units.resolve()
    pilot_dir = args.pilot_dir.resolve()
    template = build_template(template_path)
    cells, landmarks = load_session(support_path, units_path, args.session_id)
    if "session_id" not in cells:
        cells.insert(0, "session_id", args.session_id)
    landmarks.insert(0, "session_id", args.session_id)
    affine_manifest_path = pilot_dir / "run_manifest.json"
    affine_manifest = json.loads(affine_manifest_path.read_text(encoding="utf-8"))
    affine_fit = affine_manifest["selected_models"]["joint_anatomy_rf"]

    bundle = fit_models(landmarks, template, affine_fit)
    details, summary = build_tables(landmarks, bundle)
    details.to_csv(output / "penetration_model_comparison.csv", index=False, float_format="%.8g")
    bundle["lopo"].to_csv(
        output / "leave_one_penetration_out.csv", index=False, float_format="%.8g"
    )
    bundle["hyperparameters"].to_csv(
        output / "residual_hyperparameter_scan.csv", index=False, float_format="%.8g"
    )
    summary.to_csv(output / "model_summary.csv", index=False, float_format="%.8g")
    figure_path = output / "Figure_deformable_map_model_comparison.png"
    grid_audit = render_map_comparison(landmarks, template, bundle, figure_path)
    selected_figure_path = output / "Figure_selected_deformable_map_cells.png"
    render_selected_cell_map(cells, template, bundle, selected_figure_path)

    manifest = {
        "checkpoint": "initial nested deformable-map comparison",
        "status": "exploratory; six penetrations; residual hyperparameters selected on the same six LOPO folds",
        "session_id": args.session_id,
        "sources": {
            "rf_support": {"path": str(support_path), "sha256": sha256(support_path)},
            "unit_table": {"path": str(units_path), "sha256": sha256(units_path)},
            "zhuang_template": {"path": str(template_path), "sha256": sha256(template_path)},
            "affine_manifest": {
                "path": str(affine_manifest_path), "sha256": sha256(affine_manifest_path)
            },
            "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "models": {
            "geometry_only": "fixed CCF anatomy; prior affine plus six regularized quadratic geometry coefficients",
            "range_expansion": "geometry-only model plus per-axis global offset and positive scale",
            "smooth_residual": "range-expansion model plus two Gaussian-kernel RF residual fields",
        },
        "independent_spatial_grain": "one median landmark per probe penetration",
        "penetrations": len(landmarks),
        "selected_residual_hyperparameters": bundle["selected_hyper"],
        "grid_deformation_audit": grid_audit,
        "regularization": {
            "quadratic_geometry_scale_px": 25.0,
            "global_offset_scale_deg": 20.0,
            "log_global_scale_sd": 0.5,
            "minimum_soft_jacobian_ratio": 0.20,
            "area_distance_scale_px": 10.0,
            "rf_residual_scale_deg": 10.0,
        },
        "chart_contract": {
            "question": "How do increasingly flexible deformable Zhuang priors place RF contours around fixed CCF penetration anatomy?",
            "takeaway": "Separate geometric deformation, global range calibration, and locally smooth RF deviation; compare their case-level and held-out behavior.",
            "family": "paired spatial contour/scatter small multiples",
            "grain": "six penetration medians",
            "renderer": "static Matplotlib",
            "coordinate_semantics": "ML horizontal and AP vertical, both displayed high-to-low; penetration anatomy never moves",
            "output": figure_path.name,
            "selected_all_cell_output": selected_figure_path.name,
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summary": summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
