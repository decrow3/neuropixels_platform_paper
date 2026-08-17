#!/usr/bin/env python3
"""Compare RF estimators under artificial cropping across selected V1/HVA maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import (
    fit_2d_gaussian,
    threshold_rf,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_MAPS = (
    ROOT / "artifacts" / "allen_rf_artificial_cropping" / "checkpoint2" / "native_maps"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_rf_artificial_cropping" / "checkpoint2"
SESSION_ID = 737581020
DIRECTIONS = ("top", "bottom", "left", "right")
MODEL_ORDER = (
    "Allen no baseline",
    "Baseline screen-bounded",
    "Baseline extended",
    "Baseline + DC ring",
)
MODEL_COLORS = {
    "Allen no baseline": "#777777",
    "Baseline screen-bounded": "#39738c",
    "Baseline extended": "#7a6f9b",
    "Baseline + DC ring": "#d97736",
}
AREA_MAP = {
    "VISp": "V1", "VISl": "HVA", "VISrl": "HVA", "VISal": "HVA",
    "VISpm": "HVA", "VISam": "HVA",
}
RING_RADIUS_PX = {"V1": 4.0, "HVA": 5.0}
RING_TOTAL_WEIGHT = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-removed", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(unit_table: Path) -> pd.DataFrame:
    table = pd.read_csv(unit_table, low_memory=False)
    selected = table.loc[
        table["ecephys_session_id"].eq(SESSION_ID)
        & table["ecephys_structure_acronym"].isin(AREA_MAP)
        & table["p_value_rf"].lt(0.01)
        & table["on_screen_rf"].fillna(False).astype(bool)
        & table[["width_rf", "height_rf", "azimuth_rf", "elevation_rf"]].notna().all(axis=1)
        & table["area_rf"].lt(2500)
        & table["snr"].gt(1)
        & table["firing_rate_dg"].gt(0.1)
    ].copy()
    selected["group"] = selected["ecephys_structure_acronym"].map(AREA_MAP)
    selected["distance_to_nearest_edge_deg"] = np.minimum.reduce(
        [
            selected["azimuth_rf"] - 10.0,
            90.0 - selected["azimuth_rf"],
            selected["elevation_rf"] + 30.0,
            50.0 - selected["elevation_rf"],
        ]
    )
    selected["released_major_sigma_deg"] = selected[["width_rf", "height_rf"]].abs().max(axis=1)
    eligible = selected.loc[
        selected["distance_to_nearest_edge_deg"].ge(20)
        & selected["released_major_sigma_deg"].between(5, 40)
    ].copy()
    rows = []
    for group in ("V1", "HVA"):
        local = eligible.loc[eligible["group"].eq(group)].copy()
        for role, quantile in (("typical", 0.50), ("upper quartile", 0.75)):
            target = local["released_major_sigma_deg"].quantile(quantile)
            row = local.loc[(local["released_major_sigma_deg"] - target).abs().idxmin()].copy()
            row["selection_role"] = f"{group} {role}"
            row["selection_quantile"] = quantile
            row["selection_target_sigma_deg"] = target
            row["eligible_group_units"] = len(local)
            rows.append(row)
    columns = [
        "ecephys_unit_id", "selection_role", "selection_quantile",
        "selection_target_sigma_deg", "eligible_group_units", "group",
        "ecephys_structure_acronym", "azimuth_rf", "elevation_rf", "area_rf",
        "width_rf", "height_rf", "released_major_sigma_deg",
        "distance_to_nearest_edge_deg", "p_value_rf", "snr", "firing_rate_dg",
    ]
    return pd.DataFrame(rows)[columns]


def crop_map(matrix: np.ndarray, direction: str, removed: int):
    rows, columns = matrix.shape
    if direction == "top":
        row_slice, column_slice = slice(removed, rows), slice(0, columns)
    elif direction == "bottom":
        row_slice, column_slice = slice(0, rows - removed), slice(0, columns)
    elif direction == "left":
        row_slice, column_slice = slice(0, rows), slice(removed, columns)
    elif direction == "right":
        row_slice, column_slice = slice(0, rows), slice(0, columns - removed)
    else:
        raise ValueError(direction)
    cropped = matrix[row_slice, column_slice]
    return (
        cropped,
        np.arange(columns, dtype=float)[column_slice],
        np.arange(rows, dtype=float)[row_slice],
    )


def gaussian_prediction(parameters, x, y):
    baseline, amplitude, center_y, center_x, sigma_y, sigma_x = parameters
    return baseline + amplitude * np.exp(
        -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2)
    )


def threshold_metrics(matrix, x_coordinates, y_coordinates):
    mask, center_x_local, center_y_local, area_pixels = threshold_rf(matrix, 1.0)
    center_x = center_x_local + x_coordinates.min()
    center_y = center_y_local + y_coordinates.min()
    touches = bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )
    return mask, center_x, center_y, area_pixels, touches


def fit_baseline_variant(
    matrix,
    x_coordinates,
    y_coordinates,
    *,
    center_extension_px,
    sigma_upper_px,
    ring_center=None,
    ring_radius_px=None,
    ring_total_weight=RING_TOTAL_WEIGHT,
):
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)
    baseline = max(float(np.quantile(matrix, 0.2)), 0.0)
    peak_row, peak_column = np.unravel_index(np.argmax(matrix), matrix.shape)
    initial = np.array(
        [baseline, max(float(matrix.max() - baseline), 1e-3),
         y_coordinates[peak_row], x_coordinates[peak_column], 1.5, 1.5]
    )
    lower = np.array(
        [0.0, 0.0, y_coordinates.min() - center_extension_px,
         x_coordinates.min() - center_extension_px, 0.35, 0.35]
    )
    upper = np.array(
        [np.inf, np.inf, y_coordinates.max() + center_extension_px,
         x_coordinates.max() + center_extension_px, sigma_upper_px, sigma_upper_px]
    )
    ring_x = ring_y = None
    if ring_radius_px is not None:
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring_x = ring_center[0] + ring_radius_px * np.cos(angles)
        ring_y = ring_center[1] + ring_radius_px * np.sin(angles)

    def residual(parameters):
        data_residual = (gaussian_prediction(parameters, x_mesh, y_mesh) - matrix).ravel()
        if ring_x is None:
            return data_residual
        ring_modulation = gaussian_prediction(parameters, ring_x, ring_y) - parameters[0]
        ring_residual = np.sqrt(ring_total_weight / len(ring_x)) * ring_modulation
        return np.concatenate([data_residual, ring_residual])

    fit = least_squares(
        residual,
        np.clip(initial, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        max_nfev=20000,
        method="trf",
    )
    prediction = gaussian_prediction(fit.x, x_mesh, y_mesh)
    at_bound = bool(
        np.any(np.isclose(fit.x, lower, atol=1e-5, rtol=0))
        or np.any(np.isclose(fit.x, upper, atol=1e-5, rtol=0))
    )
    ring_rms = np.nan
    if ring_x is not None:
        modulation = gaussian_prediction(fit.x, ring_x, ring_y) - fit.x[0]
        ring_rms = float(np.sqrt(np.mean(modulation**2)))
    return fit.x, {
        "success": bool(fit.success and np.all(np.isfinite(fit.x))),
        "at_bound": at_bound,
        "data_rmse": float(np.sqrt(np.square(prediction - matrix).mean())),
        "ring_modulation_rms": ring_rms,
    }


def analyze_model(matrix, x_coordinates, y_coordinates, model, group, threshold_center):
    if model == "Allen no baseline":
        parameters, success = fit_2d_gaussian(matrix)
        parameters = np.asarray(parameters, dtype=float)
        # Correct Allen's swapped row/column center labels for rectangular crops.
        return {
            "success": bool(success),
            "at_bound": False,
            "baseline": 0.0,
            "center_x_px": parameters[1] + x_coordinates.min(),
            "center_y_px": parameters[2] + y_coordinates.min(),
            "sigma_x_px": abs(parameters[3]),
            "sigma_y_px": abs(parameters[4]),
            "data_rmse": np.nan,
            "ring_modulation_rms": np.nan,
        }
    if model == "Baseline screen-bounded":
        parameters, audit = fit_baseline_variant(
            matrix, x_coordinates, y_coordinates,
            center_extension_px=0.0, sigma_upper_px=4.0,
        )
    elif model == "Baseline extended":
        parameters, audit = fit_baseline_variant(
            matrix, x_coordinates, y_coordinates,
            center_extension_px=2.0, sigma_upper_px=8.0,
        )
    elif model == "Baseline + DC ring":
        parameters, audit = fit_baseline_variant(
            matrix, x_coordinates, y_coordinates,
            center_extension_px=2.0, sigma_upper_px=8.0,
            ring_center=threshold_center, ring_radius_px=RING_RADIUS_PX[group],
        )
    else:
        raise ValueError(model)
    return {
        **audit,
        "baseline": parameters[0],
        "center_x_px": parameters[3],
        "center_y_px": parameters[2],
        "sigma_x_px": parameters[5],
        "sigma_y_px": parameters[4],
    }


def build_trajectories(cases, maps_dir, maximum_removed):
    rows = []
    case_maps = {}
    for case in cases.itertuples(index=False):
        unit_id = int(case.ecephys_unit_id)
        map_path = maps_dir / f"unit_{unit_id}_observed_map.csv"
        matrix = np.loadtxt(map_path, delimiter=",")
        case_maps[unit_id] = matrix
        full_mask, _, _, _, _ = threshold_metrics(
            matrix, np.arange(matrix.shape[1], dtype=float),
            np.arange(matrix.shape[0], dtype=float),
        )
        full_peak_row, full_peak_column = np.unravel_index(np.argmax(matrix), matrix.shape)
        for direction in DIRECTIONS:
            for removed in range(maximum_removed + 1):
                cropped, x_coordinates, y_coordinates = crop_map(matrix, direction, removed)
                retained_mask = full_mask[np.ix_(
                    y_coordinates.astype(int), x_coordinates.astype(int)
                )]
                original_component_censored = bool(retained_mask.sum() < full_mask.sum())
                original_peak_removed = bool(
                    full_peak_row not in y_coordinates or full_peak_column not in x_coordinates
                )
                if original_peak_removed:
                    crop_stratum = "original peak removed"
                elif original_component_censored:
                    crop_stratum = "component censored; peak retained"
                else:
                    crop_stratum = "component intact"
                _, threshold_x, threshold_y, area_pixels, touches = threshold_metrics(
                    cropped, x_coordinates, y_coordinates
                )
                for model in MODEL_ORDER:
                    fit = analyze_model(
                        cropped, x_coordinates, y_coordinates, model, case.group,
                        (threshold_x, threshold_y),
                    )
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "selection_role": case.selection_role,
                            "group": case.group,
                            "direction": direction,
                            "rows_or_columns_removed": removed,
                            "model": model,
                            "remaining_rows": cropped.shape[0],
                            "remaining_columns": cropped.shape[1],
                            "threshold_center_x_px": threshold_x,
                            "threshold_center_y_px": threshold_y,
                            "threshold_area_deg2": area_pixels * 100.0,
                            "threshold_component_touches_crop_edge": touches,
                            "original_threshold_component_censored": original_component_censored,
                            "original_peak_removed": original_peak_removed,
                            "crop_stratum": crop_stratum,
                            **fit,
                            "major_sigma_deg": max(fit["sigma_x_px"], fit["sigma_y_px"]) * 10.0,
                            "minor_sigma_deg": min(fit["sigma_x_px"], fit["sigma_y_px"]) * 10.0,
                        }
                    )
    result = pd.DataFrame(rows)
    for (unit_id, model, direction), indices in result.groupby(
        ["unit_id", "model", "direction"], observed=True
    ).groups.items():
        local = result.loc[indices]
        reference = local.loc[local["rows_or_columns_removed"].eq(0)].iloc[0]
        result.loc[indices, "major_sigma_ratio"] = (
            local["major_sigma_deg"] / reference["major_sigma_deg"]
        )
        result.loc[indices, "absolute_log2_major_sigma_ratio"] = np.abs(
            np.log2(result.loc[indices, "major_sigma_ratio"])
        )
        result.loc[indices, "center_error_deg"] = 10.0 * np.hypot(
            local["center_x_px"] - reference["center_x_px"],
            local["center_y_px"] - reference["center_y_px"],
        )
        result.loc[indices, "threshold_area_ratio"] = (
            local["threshold_area_deg2"] / reference["threshold_area_deg2"]
        )
    return result, case_maps


def nearest_direction(matrix):
    baseline = max(float(np.quantile(matrix, 0.2)), 0.0)
    peak_row, peak_column = np.unravel_index(np.argmax(matrix - baseline), matrix.shape)
    distances = {
        "top": peak_row,
        "bottom": matrix.shape[0] - 1 - peak_row,
        "left": peak_column,
        "right": matrix.shape[1] - 1 - peak_column,
    }
    return min(distances, key=distances.get)


def render_case_figure(trajectories, cases, case_maps, path):
    fig, axes = plt.subplots(len(cases), 3, figsize=(14.2, 3.25 * len(cases)), squeeze=False)
    for row_index, case in enumerate(cases.itertuples(index=False)):
        unit_id = int(case.ecephys_unit_id)
        direction = nearest_direction(case_maps[unit_id])
        local = trajectories.loc[
            trajectories["unit_id"].eq(unit_id)
            & trajectories["direction"].eq(direction)
        ]
        for model in MODEL_ORDER:
            selected = local.loc[local["model"].eq(model)].sort_values("rows_or_columns_removed")
            axes[row_index, 0].plot(
                selected["rows_or_columns_removed"], selected["major_sigma_ratio"],
                marker="o", color=MODEL_COLORS[model], label=model,
            )
            axes[row_index, 1].plot(
                selected["rows_or_columns_removed"], selected["center_error_deg"],
                marker="o", color=MODEL_COLORS[model], label=model,
            )
        area = local.loc[local["model"].eq("Baseline screen-bounded")].sort_values(
            "rows_or_columns_removed"
        )
        axes[row_index, 2].plot(
            area["rows_or_columns_removed"], area["threshold_area_ratio"],
            marker="o", color="#b23a48",
        )
        touching = area.loc[area["threshold_component_touches_crop_edge"]]
        axes[row_index, 2].scatter(
            touching["rows_or_columns_removed"], touching["threshold_area_ratio"],
            marker="x", s=70, color="#222222", label="component touches edge",
        )
        axes[row_index, 0].axhline(1, color="#333333", linestyle="--", linewidth=1)
        axes[row_index, 2].axhline(1, color="#333333", linestyle="--", linewidth=1)
        axes[row_index, 0].set_ylabel(f"{case.selection_role}\nunit {unit_id}\nmajor σ ratio")
        axes[row_index, 1].set_ylabel("Center error (deg)")
        axes[row_index, 2].set_ylabel("Threshold-area ratio")
        for column in range(3):
            axes[row_index, column].set_xlabel(f"{direction} rows/columns removed")
            axes[row_index, column].set_xticks(range(5))
            axes[row_index, column].grid(alpha=0.18)
    axes[0, 0].set_title("Gaussian scale versus full map")
    axes[0, 1].set_title("Gaussian center versus full map")
    axes[0, 2].set_title("Allen threshold area versus full map")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Selected V1/HVA RFs under nearest-edge artificial cropping")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_summary(trajectories):
    cropped = trajectories.loc[trajectories["rows_or_columns_removed"].gt(0)].copy()
    rows = []
    for model, selected in cropped.groupby("model", observed=True):
        rows.append(
            {
                "model": model,
                "crop_fits": len(selected),
                "median_absolute_log2_sigma_error": selected["absolute_log2_major_sigma_ratio"].median(),
                "q90_absolute_log2_sigma_error": selected["absolute_log2_major_sigma_ratio"].quantile(0.9),
                "median_center_error_deg": selected["center_error_deg"].median(),
                "q90_center_error_deg": selected["center_error_deg"].quantile(0.9),
                "failure_or_bound_fraction": (~selected["success"].astype(bool) | selected["at_bound"].astype(bool)).mean(),
                "median_data_rmse": selected["data_rmse"].median(),
            }
        )
    return pd.DataFrame(rows)


def stratified_summary(trajectories):
    cropped = trajectories.loc[trajectories["rows_or_columns_removed"].gt(0)].copy()
    rows = []
    strata = (
        "component intact",
        "component censored; peak retained",
        "original peak removed",
    )
    for stratum in strata:
        local = cropped.loc[cropped["crop_stratum"].eq(stratum)]
        for model in MODEL_ORDER:
            selected = local.loc[local["model"].eq(model)]
            if selected.empty:
                continue
            rows.append(
                {
                    "crop_stratum": stratum,
                    "model": model,
                    "crop_fits": len(selected),
                    "median_absolute_log2_sigma_error": selected["absolute_log2_major_sigma_ratio"].median(),
                    "q90_absolute_log2_sigma_error": selected["absolute_log2_major_sigma_ratio"].quantile(0.9),
                    "median_center_error_deg": selected["center_error_deg"].median(),
                    "q90_center_error_deg": selected["center_error_deg"].quantile(0.9),
                    "failure_or_bound_fraction": (
                        ~selected["success"].astype(bool) | selected["at_bound"].astype(bool)
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def render_aggregate_figure(trajectories, summary, path):
    cropped = trajectories.loc[trajectories["rows_or_columns_removed"].gt(0)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for model in MODEL_ORDER:
        selected = cropped.loc[cropped["model"].eq(model)]
        depth = selected.groupby("rows_or_columns_removed", observed=True)
        x = sorted(selected["rows_or_columns_removed"].unique())
        sigma_median = depth["absolute_log2_major_sigma_ratio"].median().reindex(x)
        center_median = depth["center_error_deg"].median().reindex(x)
        axes[0, 0].plot(x, sigma_median, marker="o", color=MODEL_COLORS[model], label=model)
        axes[0, 1].plot(x, center_median, marker="o", color=MODEL_COLORS[model], label=model)
    order = list(MODEL_ORDER)
    lookup = summary.set_index("model")
    axes[1, 0].bar(
        range(len(order)), lookup.loc[order, "q90_center_error_deg"],
        color=[MODEL_COLORS[m] for m in order], alpha=0.76,
    )
    axes[1, 1].bar(
        range(len(order)), lookup.loc[order, "failure_or_bound_fraction"],
        color=[MODEL_COLORS[m] for m in order], alpha=0.76,
    )
    axes[0, 0].set(title="Median absolute Gaussian-scale error", ylabel="|log₂(crop/full major σ)|")
    axes[0, 1].set(title="Median Gaussian-center error", ylabel="Center error (deg)")
    axes[1, 0].set(title="90th-percentile center error", ylabel="Center error (deg)")
    axes[1, 1].set(title="Optimizer failure or parameter-bound rate", ylabel="Fraction of crop fits")
    for ax in axes[0]:
        ax.set_xlabel("Rows or columns removed")
        ax.set_xticks(range(1, 5))
        ax.grid(alpha=0.18)
    for ax in axes[1]:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["Allen", "baseline\nscreen", "baseline\nextended", "DC ring"], rotation=0)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Artificial-crop stability across four selected native RF maps and four directions")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_existing = output_dir / "native_maps"
    existing_files = [path for path in output_dir.iterdir() if path != expected_existing]
    if existing_files and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint outputs in {output_dir}")
    cases = select_cases(args.unit_table.resolve())
    cases.to_csv(output_dir / "selected_cases.csv", index=False, float_format="%.8g")
    trajectories, case_maps = build_trajectories(
        cases, args.maps_dir.resolve(), args.maximum_removed
    )
    trajectories.to_csv(output_dir / "multicase_crop_trajectories.csv", index=False, float_format="%.8g")
    summary = aggregate_summary(trajectories)
    summary.to_csv(output_dir / "model_crop_stability_summary.csv", index=False, float_format="%.8g")
    strata = stratified_summary(trajectories)
    strata.to_csv(output_dir / "model_crop_stability_by_censoring.csv", index=False, float_format="%.8g")
    case_figure = output_dir / "Figure_selected_case_crop_trajectories.png"
    aggregate_figure = output_dir / "Figure_multicase_crop_stability.png"
    render_case_figure(trajectories, cases, case_maps, case_figure)
    render_aggregate_figure(trajectories, summary, aggregate_figure)

    lookup = summary.set_index("model")
    ring = lookup.loc["Baseline + DC ring"]
    extended = lookup.loc["Baseline extended"]
    screen = lookup.loc["Baseline screen-bounded"]
    allen = lookup.loc["Allen no baseline"]
    strata_lookup = strata.set_index(["crop_stratum", "model"])
    intact_allen = strata_lookup.loc[("component intact", "Allen no baseline")]
    intact_ring = strata_lookup.loc[("component intact", "Baseline + DC ring")]
    censored_allen = strata_lookup.loc[("component censored; peak retained", "Allen no baseline")]
    censored_extended = strata_lookup.loc[("component censored; peak retained", "Baseline extended")]
    censored_ring = strata_lookup.loc[("component censored; peak retained", "Baseline + DC ring")]
    report = [
        "# Multi-case artificial RF cropping with a DC-return ring",
        "",
        "## Design",
        "",
        "Four units were selected algorithmically before inspecting their raw maps: the units nearest the median and 75th percentile released major sigma among eligible interior V1 and HVA fits in native session 737581020. Eligibility required published-like QC, p_value_rf < .01, an on-screen released Gaussian center, a threshold center at least 20° from every grid edge, and released major sigma between 5° and 40°.",
        "",
        "Each native 9×9 map was cropped by one to four rows or columns from all four directions. Every estimator is compared with its own full-map estimate. The DC-ring model fits a nonnegative baseline plus Gaussian, allows the center two pixels beyond the observed crop, and adds 24 soft pseudo-points whose target is the fitted DC baseline. The exploratory return-to-DC radii are 40° for V1 and 50° for HVA; the total ring weight equals four observed grid positions.",
        "",
        "## Result",
        "",
        f"Across {int(ring.crop_fits)} cropped-map fits per model, the median absolute log2 major-sigma errors were **{allen.median_absolute_log2_sigma_error:.3f}** for Allen, **{screen.median_absolute_log2_sigma_error:.3f}** for the screen-bounded baseline model, **{extended.median_absolute_log2_sigma_error:.3f}** for the extended baseline model, and **{ring.median_absolute_log2_sigma_error:.3f}** for the DC ring.",
        "",
        f"The corresponding 90th-percentile center errors were **{allen.q90_center_error_deg:.1f}°**, **{screen.q90_center_error_deg:.1f}°**, **{extended.q90_center_error_deg:.1f}°**, and **{ring.q90_center_error_deg:.1f}°**. Failure-or-bound rates were **{allen.failure_or_bound_fraction:.1%}**, **{screen.failure_or_bound_fraction:.1%}**, **{extended.failure_or_bound_fraction:.1%}**, and **{ring.failure_or_bound_fraction:.1%}**.",
        "",
        f"When the original threshold component remained intact, Allen and the ring were both stable (median absolute log2 sigma error **{intact_allen.median_absolute_log2_sigma_error:.3f}** and **{intact_ring.median_absolute_log2_sigma_error:.3f}**). Once the original component was censored but its peak remained, Allen's median error rose to **{censored_allen.median_absolute_log2_sigma_error:.3f}**; the extended-baseline and ring errors were **{censored_extended.median_absolute_log2_sigma_error:.3f}** and **{censored_ring.median_absolute_log2_sigma_error:.3f}**.",
        "",
        "## Interpretation",
        "",
        "This is an exploratory four-case stress test, not a population validation. It separates the benefit of adding a DC baseline from the incremental effect of the enclosing DC ring. The ring radius and weight remain hyperparameters and should be swept or cross-fit before production use.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json" and path.name != "native_maps"
    }
    manifest = {
        "checkpoint": "multi-case artificial cropping and exploratory DC ring",
        "session_id": SESSION_ID,
        "inputs": {
            "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
            "maps_dir": str(args.maps_dir.resolve()),
        },
        "parameters": {
            "directions": DIRECTIONS,
            "maximum_removed": args.maximum_removed,
            "ring_radius_px": RING_RADIUS_PX,
            "ring_total_weight_in_observed_pixel_equivalents": RING_TOTAL_WEIGHT,
            "ring_points": 24,
            "extended_center_allowance_px": 2.0,
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote multi-case crop validation to {output_dir}")


if __name__ == "__main__":
    main()
