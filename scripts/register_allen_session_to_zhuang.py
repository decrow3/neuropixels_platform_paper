#!/usr/bin/env python3
"""Pilot CCF/RF registration of one Allen session to Zhuang Figure 9.

The independent spatial units are probe penetrations, not neurons.  The fit is
therefore estimated from penetration-level medians and evaluated both at the
penetration and unit levels.  Two affine models are retained deliberately:
one balances RF agreement with the published area compartments, and the other
uses RF agreement alone to expose anatomy/retinotopy conflicts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT = (
    ROOT
    / "artifacts/allen_multisession_rf_validation_v1/07_registration_readiness"
    / "rf_size_visual_anatomy_unit_support.csv"
)
DEFAULT_UNITS = ROOT / "data/unit_table.csv"
DEFAULT_TEMPLATE = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/retinotopy_registration_pilot"
DEFAULT_SESSION = 798911424

ARTICLE_URL = "https://elifesciences.org/articles/18372"
FIGURE3_URL = (
    "https://iiif.elifesciences.org/lax/18372%2F"
    "elife-18372-fig3-v2.tif/full/full/0/default.tif"
)

# Seeds were visually verified against labeled Zhuang Figure 3C.  They select
# connected compartments from the repeated Figure 9 mean field-sign borders.
AREA_SEEDS_XY = {
    "VISp": (200, 240),  # V1
    "VISl": (100, 260),  # LM
    "VISal": (75, 190),  # AL
    "VISrl": (180, 80),  # RL
    "VISam": (240, 80),  # AM
}
AREA_LABELS = {"VISp": "V1", "VISl": "LM", "VISal": "AL", "VISrl": "RL", "VISam": "AM"}
AREA_COLORS = {
    "VISp": "#2864a8",
    "VISl": "#d78318",
    "VISal": "#b33f62",
    "VISrl": "#5f8f3e",
    "VISam": "#7356a8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pseudo_huber(values: np.ndarray) -> np.ndarray:
    return 2.0 * (np.sqrt(1.0 + np.square(values)) - 1.0)


def build_template(template_path: Path) -> dict:
    source = np.load(template_path)
    boundary = source["mean_field_sign_boundary"].astype(bool)
    domain = ndimage.binary_fill_holes(boundary)
    rows, columns = np.indices(boundary.shape)

    fields = {}
    field_arrays = {}
    for name in ("altitude_deg", "azimuth_deg"):
        sparse = source[name]
        observed = np.isfinite(sparse)
        surface = griddata(
            np.column_stack([columns[observed], rows[observed]]),
            sparse[observed],
            (columns, rows),
            method="linear",
        )
        missing = ~np.isfinite(surface)
        nearest = ndimage.distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        surface[missing] = surface[tuple(nearest[:, missing])]
        field_arrays[name] = surface.astype(np.float32)
        fields[name] = RegularGridInterpolator(
            (np.arange(surface.shape[0]), np.arange(surface.shape[1])),
            surface,
            bounds_error=False,
            fill_value=None,
        )

    walls = ndimage.binary_dilation(boundary, iterations=1)
    components, _ = ndimage.label(domain & ~walls)
    area_masks = {}
    area_distance_arrays = {}
    area_distance = {}
    for acronym, (x, y) in AREA_SEEDS_XY.items():
        component = int(components[y, x])
        if component == 0:
            raise RuntimeError(f"Area seed for {acronym} lies on a border")
        mask = components == component
        distance = ndimage.distance_transform_edt(~mask)
        area_masks[acronym] = mask
        area_distance_arrays[acronym] = distance.astype(np.float32)
        area_distance[acronym] = RegularGridInterpolator(
            (np.arange(mask.shape[0]), np.arange(mask.shape[1])),
            distance,
            bounds_error=False,
            fill_value=100.0,
        )

    outside_distance_array = ndimage.distance_transform_edt(~domain).astype(np.float32)
    outside_distance = RegularGridInterpolator(
        (np.arange(domain.shape[0]), np.arange(domain.shape[1])),
        outside_distance_array,
        bounds_error=False,
        fill_value=100.0,
    )
    return {
        "boundary": boundary,
        "domain": domain,
        "fields": fields,
        "field_arrays": field_arrays,
        "area_masks": area_masks,
        "area_distance_arrays": area_distance_arrays,
        "area_distance": area_distance,
        "outside_distance_array": outside_distance_array,
        "outside_distance": outside_distance,
        "sparse_altitude": source["altitude_deg"],
        "sparse_azimuth": source["azimuth_deg"],
    }


def load_session(support_path: Path, units_path: Path, session_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = pd.read_csv(support_path, low_memory=False)
    units = pd.read_csv(
        units_path,
        usecols=["ecephys_unit_id", "ecephys_probe_id"],
        low_memory=False,
    )
    data = support.loc[support.session_id.eq(session_id) & support.ccf_available].merge(
        units, on="ecephys_unit_id", how="left", validate="one_to_one"
    )
    data = data.loc[data.ecephys_structure_acronym.isin(AREA_SEEDS_XY)].copy()
    if data.empty:
        raise RuntimeError(f"Session {session_id} has no usable CCF/RF observations")
    if data.ecephys_probe_id.isna().any():
        raise RuntimeError("Some selected observations lack probe identifiers")
    landmarks = (
        data.groupby(["ecephys_probe_id", "ecephys_structure_acronym"], as_index=False)
        .agg(
            units=("ecephys_unit_id", "size"),
            ccf_ap_mm=("ccf_ap_mm", "median"),
            ccf_ml_mm=("ccf_ml_mm", "median"),
            rf_azimuth_deg=("visual_azimuth_deg", "median"),
            rf_azimuth_iqr_deg=("visual_azimuth_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            rf_elevation_deg=("visual_elevation_deg", "median"),
            rf_elevation_iqr_deg=("visual_elevation_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .sort_values("ecephys_probe_id")
        .reset_index(drop=True)
    )
    landmarks["zhuang_area"] = landmarks.ecephys_structure_acronym.map(AREA_LABELS)
    return data, landmarks


def affine(parameters: np.ndarray, reflection: int) -> tuple[np.ndarray, np.ndarray]:
    center_x, center_y, theta, scale_x, scale_y, shear = parameters
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix = rotation @ np.array(
        [[scale_x, shear * scale_y], [0.0, reflection * scale_y]]
    )
    return np.array([center_x, center_y]), matrix


def transform_ccf(
    ccf: np.ndarray, ccf_center: np.ndarray, parameters: np.ndarray, reflection: int
) -> np.ndarray:
    center, matrix = affine(parameters, reflection)
    return (ccf - ccf_center) @ matrix.T + center


def target_rf(observed: np.ndarray, convention: str) -> np.ndarray:
    result = observed.copy()
    if convention == "100_minus_azimuth":
        result[:, 0] = 100.0 - result[:, 0]
    return result


def sample_template(template: dict, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = template["domain"].shape
    clipped = xy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    row_col = clipped[:, ::-1]
    predicted = np.column_stack(
        [
            template["fields"]["azimuth_deg"](row_col),
            template["fields"]["altitude_deg"](row_col),
        ]
    )
    outside = template["outside_distance"](row_col)
    bounds = np.linalg.norm((xy - clipped) / 15.0, axis=1)
    return predicted, outside, bounds


def fit_candidate(
    template: dict,
    landmarks: pd.DataFrame,
    area_weight: float,
    convention: str,
    reflection: int,
    seed: int,
) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    target = target_rf(observed, convention)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = ccf.mean(axis=0)
    height, width = template["domain"].shape

    def objective(parameters: np.ndarray) -> float:
        xy = transform_ccf(ccf, ccf_center, parameters, reflection)
        predicted, outside, bounds = sample_template(template, xy)
        retinal = float(np.mean(pseudo_huber((predicted - target) / 10.0)))
        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        area_distances = np.asarray(
            [template["area_distance"][area](row_col[i : i + 1])[0] for i, area in enumerate(areas)]
        )
        area_penalty = float(np.mean(np.square(area_distances / 12.0)))
        domain_penalty = float(np.mean(np.square(outside / 15.0) + np.square(bounds)))
        scale_x, scale_y, shear = parameters[3], parameters[4], parameters[5]
        geometry_penalty = float(
            0.02 * (np.log(scale_x / 180.0) ** 2 + np.log(scale_y / 180.0) ** 2)
            + 0.02 * shear**2
        )
        return retinal + area_weight * area_penalty + domain_penalty + geometry_penalty

    bounds = [
        (40.0, width - 30.0),
        (20.0, height - 20.0),
        (-np.pi, np.pi),
        (70.0, 320.0),
        (70.0, 320.0),
        (-0.8, 0.8),
    ]
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=250,
        popsize=12,
        tol=1e-7,
        polish=True,
        workers=1,
        updating="immediate",
    )
    xy = transform_ccf(ccf, ccf_center, result.x, reflection)
    predicted, outside, bounds_distance = sample_template(template, xy)
    row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
    area_distances = np.asarray(
        [template["area_distance"][area](row_col[i : i + 1])[0] for i, area in enumerate(areas)]
    )
    residuals = predicted - target
    center, matrix = affine(result.x, reflection)
    return {
        "objective": float(result.fun),
        "parameters": result.x,
        "reflection": reflection,
        "convention": convention,
        "ccf_center": ccf_center,
        "template_center": center,
        "matrix_px_per_mm": matrix,
        "xy": xy,
        "target": target,
        "predicted": predicted,
        "residuals": residuals,
        "area_distances": area_distances,
        "outside_distances": outside,
        "bounds_distances": bounds_distance,
        "retinal_rmse_deg": float(np.sqrt(np.mean(np.square(residuals)))),
        "retinal_median_vector_error_deg": float(np.median(np.linalg.norm(residuals, axis=1))),
        "mean_area_distance_px": float(area_distances.mean()),
        "landmarks_in_named_area": int(np.sum(area_distances <= 1.5)),
    }


def fit_model(template: dict, landmarks: pd.DataFrame, model: str, seed: int) -> tuple[dict, pd.DataFrame]:
    area_weight = 2.0 if model == "joint_anatomy_rf" else 0.0
    candidates = []
    rows = []
    candidate_index = 0
    for convention in ("native", "100_minus_azimuth"):
        for reflection in (-1, 1):
            candidate = fit_candidate(
                template,
                landmarks,
                area_weight,
                convention,
                reflection,
                seed + candidate_index,
            )
            candidate_index += 1
            candidates.append(candidate)
            rows.append(
                {
                    "model": model,
                    "azimuth_convention": convention,
                    "cortical_reflection": reflection,
                    "objective": candidate["objective"],
                    "penetration_rf_rmse_deg": candidate["retinal_rmse_deg"],
                    "penetration_median_vector_error_deg": candidate["retinal_median_vector_error_deg"],
                    "mean_named_area_distance_px": candidate["mean_area_distance_px"],
                    "penetrations_in_named_area": candidate["landmarks_in_named_area"],
                    "penetrations": len(landmarks),
                }
            )
    return min(candidates, key=lambda item: item["objective"]), pd.DataFrame(rows)


def add_fit_columns(frame: pd.DataFrame, fit: dict) -> pd.DataFrame:
    result = frame.copy()
    result["template_x_px"] = fit["xy"][:, 0]
    result["template_y_px"] = fit["xy"][:, 1]
    result["target_azimuth_deg"] = fit["target"][:, 0]
    result["target_elevation_deg"] = fit["target"][:, 1]
    result["predicted_azimuth_deg"] = fit["predicted"][:, 0]
    result["predicted_elevation_deg"] = fit["predicted"][:, 1]
    result["display_observed_azimuth_deg"] = result.rf_azimuth_deg
    result["display_observed_elevation_deg"] = result.rf_elevation_deg
    result["display_predicted_azimuth_deg"] = fit["predicted"][:, 0]
    if fit["convention"] == "100_minus_azimuth":
        result["display_predicted_azimuth_deg"] = 100.0 - result["display_predicted_azimuth_deg"]
    result["display_predicted_elevation_deg"] = fit["predicted"][:, 1]
    result["azimuth_residual_deg"] = fit["residuals"][:, 0]
    result["elevation_residual_deg"] = fit["residuals"][:, 1]
    result["rf_vector_error_deg"] = np.linalg.norm(fit["residuals"], axis=1)
    result["named_area_distance_px"] = fit["area_distances"]
    result["inside_named_area"] = fit["area_distances"] <= 1.5
    return result


def evaluate_units(data: pd.DataFrame, template: dict, fit: dict, model: str) -> pd.DataFrame:
    ccf = data[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    xy = transform_ccf(ccf, fit["ccf_center"], fit["parameters"], fit["reflection"])
    predicted, outside, bounds_distance = sample_template(template, xy)
    observed = data[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
    target = target_rf(observed, fit["convention"])
    residual = predicted - target
    height, width = template["domain"].shape
    row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
    area_distance = np.asarray(
        [
            template["area_distance"][area](row_col[i : i + 1])[0]
            for i, area in enumerate(data.ecephys_structure_acronym)
        ]
    )
    return pd.DataFrame(
        {
            "session_id": data.session_id.astype(int),
            "ecephys_unit_id": data.ecephys_unit_id.astype(int),
            "ecephys_probe_id": data.ecephys_probe_id.astype(int),
            "ecephys_structure_acronym": data.ecephys_structure_acronym,
            "model": model,
            "azimuth_convention": fit["convention"],
            "ccf_ap_mm": data.ccf_ap_mm,
            "ccf_ml_mm": data.ccf_ml_mm,
            "template_x_px": xy[:, 0],
            "template_y_px": xy[:, 1],
            "observed_azimuth_deg": observed[:, 0],
            "observed_elevation_deg": observed[:, 1],
            "target_azimuth_deg": target[:, 0],
            "target_elevation_deg": target[:, 1],
            "predicted_azimuth_deg": predicted[:, 0],
            "predicted_elevation_deg": predicted[:, 1],
            "azimuth_residual_deg": residual[:, 0],
            "elevation_residual_deg": residual[:, 1],
            "rf_vector_error_deg": np.linalg.norm(residual, axis=1),
            "named_area_distance_px": area_distance,
            "inside_named_area": area_distance <= 1.5,
            "outside_template_distance_px": outside,
            "outside_image_distance_scaled": bounds_distance,
        }
    )


def render_fit_map(ax, template: dict, landmarks: pd.DataFrame, fit_frame: pd.DataFrame, title: str) -> None:
    ax.contour(template["boundary"].astype(float), levels=[0.5], colors="#333333", linewidths=0.55)
    altitude = template["sparse_altitude"]
    azimuth = template["sparse_azimuth"]
    altitude_overlay = np.zeros((*altitude.shape, 4), dtype=float)
    altitude_overlay[..., :3] = (0.85, 0.36, 0.06)
    altitude_overlay[..., 3] = np.isfinite(altitude) * 0.42
    azimuth_overlay = np.zeros((*azimuth.shape, 4), dtype=float)
    azimuth_overlay[..., :3] = (0.10, 0.38, 0.68)
    azimuth_overlay[..., 3] = np.isfinite(azimuth) * 0.34
    ax.imshow(altitude_overlay)
    ax.imshow(azimuth_overlay)
    for acronym, group in fit_frame.groupby("ecephys_structure_acronym"):
        color = AREA_COLORS[acronym]
        ax.scatter(
            group.template_x_px,
            group.template_y_px,
            s=70,
            facecolors=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        for row in group.itertuples():
            ax.text(row.template_x_px + 5, row.template_y_px - 5, str(int(row.ecephys_probe_id))[-3:], fontsize=7)
    ax.set(title=title, xlabel="Zhuang Figure 9 x (px)", ylabel="Zhuang Figure 9 y (px; down +)")
    ax.set_aspect("equal")
    ax.set_xlim(0, template["domain"].shape[1])
    ax.set_ylim(template["domain"].shape[0], 0)


def render_qa(
    template: dict,
    landmarks: pd.DataFrame,
    joint: pd.DataFrame,
    rf_only: pd.DataFrame,
    unit_results: pd.DataFrame,
    output: Path,
    session_id: int,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    anatomy = axes[0, 0]
    for acronym, group in landmarks.groupby("ecephys_structure_acronym"):
        anatomy.scatter(
            group.ccf_ap_mm,
            group.ccf_ml_mm,
            s=80,
            color=AREA_COLORS[acronym],
            edgecolors="white",
            linewidths=1.0,
            label=f"{acronym}→{AREA_LABELS[acronym]}",
        )
        for row in group.itertuples():
            anatomy.text(row.ccf_ap_mm + 0.025, row.ccf_ml_mm, str(int(row.ecephys_probe_id))[-3:], fontsize=7)
    anatomy.set(
        title="Six independent penetration landmarks",
        xlabel="CCF anterior–posterior (mm)",
        ylabel="CCF medial–lateral (mm)",
        aspect="equal",
    )
    anatomy.legend(fontsize=7, loc="best")

    render_fit_map(axes[0, 1], template, landmarks, joint, "Joint anatomy + RF affine")
    render_fit_map(axes[0, 2], template, landmarks, rf_only, "RF-driven affine (anatomy unconstrained)")

    rf_axis = axes[1, 0]
    for frame, marker, label in ((joint, "o", "joint"), (rf_only, "x", "RF-only")):
        for row in frame.itertuples():
            color = AREA_COLORS[row.ecephys_structure_acronym]
            rf_axis.plot(
                [row.display_observed_azimuth_deg, row.display_predicted_azimuth_deg],
                [row.display_observed_elevation_deg, row.display_predicted_elevation_deg],
                color=color,
                linewidth=0.8,
                alpha=0.7,
            )
            rf_axis.scatter(
                row.display_predicted_azimuth_deg,
                row.display_predicted_elevation_deg,
                marker=marker,
                color=color,
                s=45,
            )
    rf_axis.scatter(
        joint.display_observed_azimuth_deg,
        joint.display_observed_elevation_deg,
        marker="s",
        facecolors="none",
        edgecolors="#222222",
        s=60,
        label="penetration RF target",
    )
    rf_axis.set(
        title="Penetration RF targets and atlas predictions",
        xlabel="Allen-native azimuth (deg)",
        ylabel="Elevation (deg)",
        xlim=(-5, 95),
        ylim=(-30, 35),
    )
    rf_axis.set_aspect("equal")
    rf_axis.legend(
        handles=[
            Line2D([], [], marker="s", markerfacecolor="none", markeredgecolor="#222222", linestyle="", label="target"),
            Line2D([], [], marker="o", color="#555555", linestyle="", label="joint prediction"),
            Line2D([], [], marker="x", color="#555555", linestyle="", label="RF-only prediction"),
        ],
        fontsize=7,
    )

    comparison = pd.concat(
        [
            joint.assign(model="Joint anatomy + RF"),
            rf_only.assign(model="RF only"),
        ],
        ignore_index=True,
    )
    comparison["probe_label"] = comparison.ecephys_structure_acronym + "\n" + comparison.ecephys_probe_id.astype(str).str[-3:]
    labels = joint.ecephys_structure_acronym + "\n" + joint.ecephys_probe_id.astype(str).str[-3:]
    positions = np.arange(len(labels))
    width = 0.36
    error_axis = axes[1, 1]
    for offset, (model, frame) in zip((-width / 2, width / 2), comparison.groupby("model", sort=False)):
        order = frame.set_index("probe_label").loc[labels]
        error_axis.bar(
            positions + offset,
            order.rf_vector_error_deg,
            width=width,
            label=model,
            color="#2864a8" if model.startswith("Joint") else "#d78318",
            alpha=0.85,
        )
    error_axis.set(
        title="Penetration-level retinal mismatch",
        xlabel="Allen area and probe suffix",
        ylabel="RF vector error (deg)",
        xticks=positions,
        xticklabels=labels,
    )
    error_axis.legend(fontsize=8)

    unit_axis = axes[1, 2]
    for model, group in unit_results.groupby("model", sort=False):
        values = np.sort(group.rf_vector_error_deg.to_numpy())
        unit_axis.plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            linewidth=2,
            label=model.replace("_", " "),
            color="#2864a8" if model == "joint_anatomy_rf" else "#d78318",
        )
    unit_axis.set(
        title="Unit-level residual distribution",
        xlabel="RF vector error (deg)",
        ylabel="Cumulative fraction of units",
        xlim=(0, min(100, unit_results.rf_vector_error_deg.quantile(0.99) + 5)),
        ylim=(0, 1),
    )
    unit_axis.legend(fontsize=8)

    figure.suptitle(
        f"Session {session_id}: first CCF + RF registration to Zhuang Figure 9\n"
        "fit unit = penetration median; neuron-level residuals shown only for evaluation",
        fontsize=15,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fit_to_json(fit: dict) -> dict:
    return {
        "objective": fit["objective"],
        "azimuth_convention": fit["convention"],
        "cortical_reflection": fit["reflection"],
        "ccf_center_ap_ml_mm": fit["ccf_center"].tolist(),
        "template_center_xy_px": fit["template_center"].tolist(),
        "affine_matrix_xy_px_per_ap_ml_mm": fit["matrix_px_per_mm"].tolist(),
        "optimizer_parameters": fit["parameters"].tolist(),
        "penetration_rf_rmse_deg": fit["retinal_rmse_deg"],
        "penetration_median_vector_error_deg": fit["retinal_median_vector_error_deg"],
        "mean_named_area_distance_px": fit["mean_area_distance_px"],
        "penetrations_in_named_area": fit["landmarks_in_named_area"],
    }


def write_readme(output: Path, manifest: dict) -> None:
    joint = manifest["selected_models"]["joint_anatomy_rf"]
    rf_only = manifest["selected_models"]["rf_only"]
    text = f"""# Session {manifest['session_id']} registration pilot — initial fit checkpoint

This pilot tests whether one global affine can register a single Allen session
to the Zhuang et al. (2017) Figure 9 population retinotopy using both CCF
surface coordinates and receptive-field centers.

## Case selection

Session {manifest['session_id']} was selected algorithmically from the four
sessions displayed in the registration-readiness PDF: it has complete CCF
coverage for all {manifest['counts']['selected_units']} trusted units and the
largest usable set of named visual areas. Those units occupy only
{manifest['counts']['penetrations']} independent probe penetrations, so the fit
uses penetration medians rather than treating neurons as independent cortical
landmarks.

## Models

1. **Joint anatomy + RF affine** penalizes leaving the Zhuang compartment
   corresponding to the Allen area acronym.
2. **RF-only affine** omits that compartment penalty. It is a diagnostic for
   whether RF agreement is obtained by placing penetrations in the wrong area.

For each model, both cortical handedness choices and both the native and
`100 - azimuth` retinal conventions were tried. The manifest preserves all
candidate scores.

## Initial result

- Joint model: median penetration RF error
  {joint['penetration_median_vector_error_deg']:.1f}°, with
  {joint['penetrations_in_named_area']}/{manifest['counts']['penetrations']}
  penetration landmarks inside their named Zhuang compartments.
- RF-only model: median penetration RF error
  {rf_only['penetration_median_vector_error_deg']:.1f}°, with
  {rf_only['penetrations_in_named_area']}/{manifest['counts']['penetrations']}
  landmarks inside their named compartments.

This is an exploratory registration, not a validated warp. A large improvement
in the RF-only model accompanied by anatomical violations would reject a
single global affine as the final model. The next model should only add local
deformation after this tradeoff is inspected, because six penetrations cannot
support a high-flexibility warp without strong regularization or widefield
landmarks.

## Evidence and derived layers

- `Figure_registration_pilot_QA.png`: CCF landmarks, both transforms, RF
  target/prediction pairs, and unit-level residual distributions.
- `penetration_landmarks.csv`: the six fit observations and both model outputs.
- `unit_registration_residuals.csv.gz`: neuron-level evaluation only.
- `candidate_model_summary.csv`: all handedness/convention candidates.
- `zhuang_interpolated_fields.npz`: explicit linear interpolation of the
  published 5° contours, named-area masks, and border/domain layers.
- `run_manifest.json`: complete provenance, fit parameters, and chart contract.

## Important limitations

1. The continuous Zhuang fields are derived by linear interpolation between
   published contours and nearest fill only at the small unsupported edge.
2. Zhuang area compartments were identified from labeled Figure 3C; they are
   not Allen CCF boundaries.
3. The source RF centers are trusted aperture fits, but within-penetration RF
   dispersion remains substantial in some HVAs.
4. This fit has no held-out penetration and should not be used for inference.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    support_path = args.support.resolve()
    units_path = args.units.resolve()
    template_path = args.template.resolve()
    output = (args.output_root / f"session_{args.session_id}").resolve()
    output.mkdir(parents=True, exist_ok=True)

    template = build_template(template_path)
    data, landmarks = load_session(support_path, units_path, args.session_id)
    joint_fit, joint_candidates = fit_model(template, landmarks, "joint_anatomy_rf", args.seed)
    rf_fit, rf_candidates = fit_model(template, landmarks, "rf_only", args.seed + 100)
    candidates = pd.concat([joint_candidates, rf_candidates], ignore_index=True)

    joint_landmarks = add_fit_columns(landmarks, joint_fit)
    joint_landmarks["model"] = "joint_anatomy_rf"
    rf_landmarks = add_fit_columns(landmarks, rf_fit)
    rf_landmarks["model"] = "rf_only"
    landmark_output = pd.concat([joint_landmarks, rf_landmarks], ignore_index=True)
    landmark_output.to_csv(output / "penetration_landmarks.csv", index=False, float_format="%.9g")
    candidates.to_csv(output / "candidate_model_summary.csv", index=False, float_format="%.9g")

    unit_results = pd.concat(
        [
            evaluate_units(data, template, joint_fit, "joint_anatomy_rf"),
            evaluate_units(data, template, rf_fit, "rf_only"),
        ],
        ignore_index=True,
    )
    with gzip.open(output / "unit_registration_residuals.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        unit_results.to_csv(stream, index=False, float_format="%.9g")

    np.savez_compressed(
        output / "zhuang_interpolated_fields.npz",
        altitude_deg=template["field_arrays"]["altitude_deg"],
        azimuth_deg=template["field_arrays"]["azimuth_deg"],
        domain=template["domain"],
        mean_field_sign_boundary=template["boundary"],
        outside_domain_distance_px=template["outside_distance_array"],
        **{f"area_mask_{name}": mask for name, mask in template["area_masks"].items()},
        **{
            f"area_distance_px_{name}": distance
            for name, distance in template["area_distance_arrays"].items()
        },
    )
    render_qa(
        template,
        landmarks,
        joint_landmarks,
        rf_landmarks,
        unit_results,
        output / "Figure_registration_pilot_QA.png",
        args.session_id,
    )

    manifest = {
        "checkpoint": "single_session_affine_registration_initial_fit",
        "status": "exploratory; visual judgment required before nonlinear deformation",
        "session_id": args.session_id,
        "selection": {
            "rule": "among the four PDF sessions, complete CCF coverage and maximum usable area/probe coverage",
            "source_pdf": str(
                support_path.parent / "RF_size_visual_and_anatomy_maps_by_session.pdf"
            ),
        },
        "sources": {
            "unit_support": {"path": str(support_path), "sha256": sha256(support_path)},
            "unit_table": {"path": str(units_path), "sha256": sha256(units_path)},
            "zhuang_template": {"path": str(template_path), "sha256": sha256(template_path)},
            "zhuang_article": ARTICLE_URL,
            "zhuang_labeled_figure3": FIGURE3_URL,
        },
        "counts": {
            "selected_units": len(data),
            "penetrations": len(landmarks),
            "areas": int(landmarks.ecephys_structure_acronym.nunique()),
        },
        "area_mapping": AREA_LABELS,
        "interpolation": {
            "method": "linear griddata between published contour pixels; nearest fill for unsupported edge pixels",
            "source_evidence": "Zhuang Figure 9C/D 5-degree contours",
        },
        "selected_models": {
            "joint_anatomy_rf": fit_to_json(joint_fit),
            "rf_only": fit_to_json(rf_fit),
        },
        "candidate_models": candidates.to_dict(orient="records"),
        "unit_evaluation": {
            model: {
                "median_rf_vector_error_deg": float(group.rf_vector_error_deg.median()),
                "p90_rf_vector_error_deg": float(group.rf_vector_error_deg.quantile(0.9)),
                "inside_named_area_fraction": float(group.inside_named_area.mean()),
            }
            for model, group in unit_results.groupby("model")
        },
        "chart_contract": {
            "question": "Can one global affine jointly align session CCF geometry, Allen area identity, and RF centers to Zhuang Figure 9?",
            "takeaway": "Compare RF agreement against named-area violations before allowing a nonlinear warp.",
            "family": "spatial registration QA plus paired error comparison",
            "renderer": "static Matplotlib",
            "palette": "fixed area colors; blue joint model; orange RF-only model; marker shape is redundant model encoding",
            "output": "Figure_registration_pilot_QA.png",
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": manifest["counts"],
                "selected_models": manifest["selected_models"],
                "unit_evaluation": manifest["unit_evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
