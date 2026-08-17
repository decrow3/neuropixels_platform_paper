#!/usr/bin/env python3
"""Test whether V1 covariance-trace registration is explained by probe sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

from scripts.checkpoint_v1_absolute_size_dispersion_translation import (
    assign_descriptors,
    robust_scale,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_ORIGINAL = CHECKPOINT / "uncensored_size_sensitivity" / "selected_case_audit.csv"
DEFAULT_EXTENDED = CHECKPOINT / "extended_case_selection" / "extended_case_selection.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "physical_sampling_control"

COMPONENTS = (
    "raw_covariance_trace",
    "shank_predicted_trace",
    "shank_residual_trace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--original-selection", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--extended-selection", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def physical_blocks(values: pd.Series, count: int = 6) -> np.ndarray:
    unique = np.sort(values.dropna().unique())
    lookup = {
        value: min(int(index * count / max(len(unique), 1)), count - 1)
        for index, value in enumerate(unique)
    }
    return values.map(lookup).fillna(-1).to_numpy(int)


def kernel_prediction(
    position: np.ndarray, values: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    distance = position[:, None] - position[None, :]
    weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
    np.fill_diagonal(weights, 0.0)
    finite = np.isfinite(values)
    weights[:, ~finite] = 0.0
    denominator = weights.sum(axis=1)
    prediction = weights @ np.nan_to_num(values, nan=0.0) / np.maximum(denominator, 1e-12)
    effective = denominator**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
    prediction[(denominator <= 0) | (effective < 3)] = np.nan
    return prediction, effective


def add_shank_decomposition(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    table = table.copy()
    position = table["probe_vertical_position"].to_numpy(float)
    raw = table["dispersion_log2_trace"].to_numpy(float)
    bandwidths = np.array([40.0, 60.0, 80.0, 120.0, 180.0, 250.0, 400.0])
    rows = []
    predictions = {}
    for bandwidth in bandwidths:
        prediction, effective = kernel_prediction(position, raw, bandwidth)
        valid = np.isfinite(raw) & np.isfinite(prediction)
        mse = float(np.mean((raw[valid] - prediction[valid]) ** 2)) if valid.sum() >= 10 else np.inf
        predictions[bandwidth] = (prediction, effective)
        rows.append((bandwidth, mse, int(valid.sum())))
    selected_bandwidth, selected_mse, selected_n = min(rows, key=lambda item: item[1])
    predicted, effective = predictions[selected_bandwidth]
    residual = raw - predicted
    table["raw_covariance_trace"] = raw
    table["shank_predicted_trace"] = predicted
    table["shank_residual_trace"] = residual
    table["shank_prediction_effective_neighbors"] = effective
    table["physical_block"] = physical_blocks(table["probe_vertical_position"])
    valid = np.isfinite(raw) & np.isfinite(predicted)
    variance_explained = (
        1.0 - np.var(residual[valid]) / np.var(raw[valid])
        if valid.sum() >= 10 and np.var(raw[valid]) > 0
        else np.nan
    )
    rho = spearmanr(position[valid], raw[valid]).statistic if valid.sum() >= 5 else np.nan
    return table, {
        "selected_shank_bandwidth_um": float(selected_bandwidth),
        "cross_fitted_mse": float(selected_mse),
        "cross_fitted_units": int(selected_n),
        "shank_prediction_variance_explained": float(variance_explained),
        "raw_trace_vs_shank_spearman": float(rho),
    }


def smooth_surface(
    table: pd.DataFrame, grid_points: np.ndarray, bandwidth: float = 12.0
) -> np.ndarray:
    points = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    delta = grid_points[:, None, :] - points[None, :, :]
    spatial_weights = np.exp(-0.5 * np.sum(delta**2, axis=2) / bandwidth**2)
    output = np.full((len(grid_points), len(COMPONENTS)), np.nan)
    for index, component in enumerate(COMPONENTS):
        values = table[component].to_numpy(float)
        finite = np.isfinite(values)
        weights = spatial_weights[:, finite]
        denominator = weights.sum(axis=1)
        effective = denominator**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
        supported = effective >= 3
        output[supported, index] = (
            weights[supported] @ values[finite] / denominator[supported]
        )
    return output


def leave_one_out_template(surfaces: dict[int, np.ndarray], held_session: int) -> np.ndarray:
    stack = np.stack([surface for session_id, surface in surfaces.items() if session_id != held_session])
    support = np.sum(np.isfinite(stack), axis=0)
    template = np.divide(
        np.nansum(stack, axis=0), support,
        out=np.full(support.shape, np.nan, dtype=float), where=support > 0,
    )
    template[support < 8] = np.nan
    return template


def make_interpolators(template: np.ndarray, axis: np.ndarray) -> list[RegularGridInterpolator]:
    shaped = template.reshape(len(axis), len(axis), len(COMPONENTS))
    return [
        RegularGridInterpolator(
            (axis, axis), shaped[:, :, index], bounds_error=False, fill_value=np.nan
        )
        for index in range(len(COMPONENTS))
    ]


def fit_component(
    table: pd.DataFrame,
    interpolator: RegularGridInterpolator,
    component: str,
    scale: float,
    shifts: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    points = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    observed = table[component].to_numpy(float)
    losses = np.full(len(shifts), np.nan)
    for index, shift in enumerate(shifts):
        query = (points + shift)[:, [1, 0]]
        predicted = interpolator(query)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        if valid.sum() < 10:
            continue
        residual = (observed[valid] - predicted[valid]) / scale
        absolute = np.abs(residual)
        huber = np.where(absolute <= 1, 0.5 * residual**2, absolute - 0.5)
        losses[index] = float(np.mean(huber)) + 0.75 * (1 - valid.mean())
    if not np.isfinite(losses).any():
        return {
            "shift_azimuth_deg": np.nan,
            "shift_elevation_deg": np.nan,
            "minimum_loss": np.nan,
            "at_bound": True,
        }, losses
    best = int(np.nanargmin(losses))
    result = {
        "shift_azimuth_deg": float(shifts[best, 0]),
        "shift_elevation_deg": float(shifts[best, 1]),
        "minimum_loss": float(losses[best]),
        "at_bound": bool(np.any(np.abs(shifts[best]) >= 30)),
        "basin_grid_points_delta_005": int(np.sum(losses <= losses[best] + 0.05)),
    }
    return result, losses


def prepare_physical_half(table: pd.DataFrame, parity: int) -> tuple[pd.DataFrame, dict[str, float]]:
    selected = table.loc[(table["physical_block"] >= 0) & (table["physical_block"] % 2 == parity)].copy()
    selected = selected.drop(
        columns=[
            "dispersion_log2_trace", "dispersion_anisotropy_x", "dispersion_anisotropy_xy",
            "dispersion_effective_neighbors", *COMPONENTS,
            "shank_prediction_effective_neighbors",
        ],
        errors="ignore",
    )
    selected = assign_descriptors(selected, bandwidth=15.0)
    return add_shank_decomposition(selected)


def within_block_permutation(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = values.copy()
    for block in np.unique(blocks[blocks >= 0]):
        indices = np.flatnonzero(blocks == block)
        output[indices] = values[rng.permutation(indices)]
    return output


def ccf_pc_diagnostic(table: pd.DataFrame) -> dict[str, float]:
    columns = [
        "anterior_posterior_ccf_coordinate",
        "left_right_ccf_coordinate",
        "dorsal_ventral_ccf_coordinate",
    ]
    valid = table.dropna(subset=columns + ["raw_covariance_trace"])
    if len(valid) < 10:
        return {"ccf_units": len(valid), "ccf_pc_vs_probe_vertical_rho": np.nan, "ccf_pc_vs_raw_trace_rho": np.nan}
    coordinates = valid[columns].to_numpy(float)
    coordinates -= coordinates.mean(axis=0)
    _, _, vectors = np.linalg.svd(coordinates, full_matrices=False)
    score = coordinates @ vectors[0]
    vertical = valid["probe_vertical_position"].to_numpy(float)
    if spearmanr(score, vertical).statistic < 0:
        score *= -1
    return {
        "ccf_units": len(valid),
        "ccf_pc_vs_probe_vertical_rho": float(spearmanr(score, vertical).statistic),
        "ccf_pc_vs_raw_trace_rho": float(spearmanr(score, valid["raw_covariance_trace"]).statistic),
    }


def plot_landscape(
    axis_object: plt.Axes,
    losses: np.ndarray,
    shifts: np.ndarray,
    title: str,
    results: dict[str, dict[str, float]],
) -> None:
    shift_axis = np.unique(shifts[:, 0])
    relative = losses.reshape(len(shift_axis), len(shift_axis)) - np.nanmin(losses)
    upper = max(float(np.nanquantile(relative[np.isfinite(relative)], 0.8)), 0.03)
    artist = axis_object.imshow(
        relative, origin="lower",
        extent=[shift_axis.min(), shift_axis.max(), shift_axis.min(), shift_axis.max()],
        cmap="magma_r", norm=Normalize(0, upper), aspect="equal",
    )
    styles = {
        "full": ("*", "white", 85),
        "physical half 0": ("o", "#27c2ff", 40),
        "physical half 1": ("s", "#55e06f", 40),
    }
    for label, result in results.items():
        marker, color, size = styles[label]
        axis_object.scatter(
            result["shift_azimuth_deg"], result["shift_elevation_deg"],
            marker=marker, facecolor=color, edgecolor="black", linewidth=0.7,
            s=size, zorder=5, label=label,
        )
    axis_object.axhline(0, color="#999999", linewidth=0.5)
    axis_object.axvline(0, color="#999999", linewidth=0.5)
    axis_object.set(xlabel="Azimuth correction (deg)", ylabel="Elevation correction (deg)", title=title)
    plt.colorbar(artist, ax=axis_object, fraction=0.046, pad=0.03, label="loss above optimum")


def render(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    results: pd.DataFrame,
    landscapes: dict[tuple[int, str], np.ndarray],
    shifts: np.ndarray,
    nulls: pd.DataFrame,
    audit: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(len(selected), 6, figsize=(25, 4.5 * len(selected)), squeeze=False)
    roles = selected.set_index("ecephys_session_id")["selection_role"].to_dict()
    for row_index, session_id in enumerate(selected["ecephys_session_id"].astype(int)):
        local = population.loc[population["ecephys_session_id"].eq(session_id)]
        local_results = results.loc[results["ecephys_session_id"].eq(session_id)]
        local_audit = audit.loc[audit["ecephys_session_id"].eq(session_id)].iloc[0]
        scatter = axes[row_index, 0].scatter(
            local["rf_azimuth_deg"], local["rf_elevation_deg"],
            c=local["probe_vertical_position"], cmap="viridis", s=23, alpha=0.85,
        )
        axes[row_index, 0].set(
            xlabel="Observed RF azimuth (deg)", ylabel="Observed RF elevation (deg)",
            title=f"{roles[session_id]}\nsession {session_id}: physical sampling", aspect="equal",
        )
        plt.colorbar(scatter, ax=axes[row_index, 0], fraction=0.046, pad=0.03, label="probe vertical position (µm)")

        ordered = local.sort_values("probe_vertical_position")
        axes[row_index, 1].scatter(
            local["probe_vertical_position"], local["raw_covariance_trace"],
            s=15, alpha=0.45, color="#3274a1", label="raw trace",
        )
        axes[row_index, 1].plot(
            ordered["probe_vertical_position"], ordered["shank_predicted_trace"],
            color="#d1495b", linewidth=1.5, label="cross-fitted shank prediction",
        )
        axes[row_index, 1].set(
            xlabel="Probe vertical position (µm)", ylabel="log₂ covariance trace",
            title=(
                f"Physical nuisance fit\nCV R²={local_audit.shank_prediction_variance_explained:.2f}; "
                f"CCF-PC↔shank rho={local_audit.ccf_pc_vs_probe_vertical_rho:.2f}"
            ),
        )
        axes[row_index, 1].legend(fontsize=7)

        for column, component in enumerate(COMPONENTS, start=2):
            component_rows = local_results.loc[local_results["component"].eq(component)]
            labels = {}
            for subset, label in (
                ("full", "full"),
                ("physical_half_0", "physical half 0"),
                ("physical_half_1", "physical half 1"),
            ):
                labels[label] = component_rows.loc[component_rows["target_subset"].eq(subset)].iloc[0].to_dict()
            difference = component_rows["physical_half_vector_difference_deg"].dropna().iloc[0]
            plot_landscape(
                axes[row_index, column], landscapes[(session_id, component)], shifts,
                f"{component.replace('_', ' ')}\nphysical-half difference={difference:.1f}°", labels,
            )

        local_null = nulls.loc[nulls["ecephys_session_id"].eq(session_id)]
        real_difference = local_null["real_residual_physical_half_difference_deg"].iloc[0]
        axes[row_index, 5].hist(
            local_null["null_residual_physical_half_difference_deg"],
            bins=np.arange(0, 90, 5), color="#9b8ac4", alpha=0.85,
        )
        axes[row_index, 5].axvline(real_difference, color="#d1495b", linewidth=2)
        fraction = np.mean(
            local_null["null_residual_physical_half_difference_deg"] <= real_difference
        )
        loss_fraction = np.mean(
            local_null["null_residual_full_minimum_loss"]
            <= local_null["real_residual_full_minimum_loss"]
        )
        axes[row_index, 5].set(
            xlabel="Residual-trace physical-half difference (deg)", ylabel="Shank-preserving nulls",
            title=(
                f"Shank-preserving residual null\n{100*fraction:.0f}% as stable; "
                f"{100*loss_fraction:.0f}% fit as well"
            ),
        )
        for axis_object in axes[row_index]:
            axis_object.grid(alpha=0.12)
    handles, labels = axes[0, 4].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles[:3], labels[:3], loc="upper center", bbox_to_anchor=(0.5, 0.965),
            ncol=3, frameon=False,
        )
    figure.suptitle(
        "Does physical probe sampling explain V1 covariance-trace registration?\n"
        "Raw trace versus cross-fitted along-shank prediction and residual; physically disjoint target blocks",
        fontsize=16, y=0.998,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.935])
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(args.input.resolve(), low_memory=False)
    units = pd.read_csv(
        args.unit_table.resolve(),
        usecols=[
            "ecephys_unit_id", "probe_vertical_position",
            "anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
        ],
        low_memory=False,
    )
    population = population.merge(units, on="ecephys_unit_id", how="left")
    if population["probe_vertical_position"].isna().any():
        raise ValueError("Probe vertical position is required for every V1 unit")

    decomposed_frames = []
    audit_rows = []
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        decomposed, summary = add_shank_decomposition(local)
        ccf_summary = ccf_pc_diagnostic(decomposed)
        decomposed_frames.append(decomposed)
        audit_rows.append({"ecephys_session_id": int(session_id), "v1_units": len(local), **summary, **ccf_summary})
    population = pd.concat(decomposed_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)

    original = pd.read_csv(args.original_selection.resolve())[["ecephys_session_id", "selection_role"]]
    extended = pd.read_csv(args.extended_selection.resolve())[["ecephys_session_id", "selection_role"]]
    selected = pd.concat([original, extended], ignore_index=True).drop_duplicates("ecephys_session_id")
    selected_ids = selected["ecephys_session_id"].astype(int).tolist()

    axis = np.arange(-90.0, 92.0, 2.0)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    surfaces = {
        int(session_id): smooth_surface(local, grid_points)
        for session_id, local in population.groupby("ecephys_session_id", observed=True)
    }
    scales = {
        component: robust_scale(population[component].to_numpy(float), 0.05)
        for component in COMPONENTS
    }
    shift_axis = np.arange(-30.0, 32.0, 2.0)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    result_rows = []
    landscapes = {}
    null_rows = []
    rng = np.random.default_rng(20260816)

    for session_id in selected_ids:
        local = population.loc[population["ecephys_session_id"].eq(session_id)].copy()
        template = leave_one_out_template(surfaces, session_id)
        interpolators = make_interpolators(template, axis)
        half_zero, half_zero_summary = prepare_physical_half(local, 0)
        half_one, half_one_summary = prepare_physical_half(local, 1)
        subsets = {"full": local, "physical_half_0": half_zero, "physical_half_1": half_one}
        local_results = {}
        for component_index, component in enumerate(COMPONENTS):
            for subset, table in subsets.items():
                result, losses = fit_component(
                    table, interpolators[component_index], component, scales[component], shifts
                )
                local_results[(component, subset)] = result
                result_rows.append(
                    {
                        "ecephys_session_id": session_id,
                        "component": component,
                        "target_subset": subset,
                        **result,
                    }
                )
                if subset == "full":
                    landscapes[(session_id, component)] = losses
            first = local_results[(component, "physical_half_0")]
            second = local_results[(component, "physical_half_1")]
            difference = float(
                np.hypot(
                    first["shift_azimuth_deg"] - second["shift_azimuth_deg"],
                    first["shift_elevation_deg"] - second["shift_elevation_deg"],
                )
            )
            for row in result_rows[-3:]:
                row["physical_half_vector_difference_deg"] = difference

        real_full = local_results[("shank_residual_trace", "full")]
        real_first = local_results[("shank_residual_trace", "physical_half_0")]
        real_second = local_results[("shank_residual_trace", "physical_half_1")]
        real_difference = float(
            np.hypot(
                real_first["shift_azimuth_deg"] - real_second["shift_azimuth_deg"],
                real_first["shift_elevation_deg"] - real_second["shift_elevation_deg"],
            )
        )
        for repeat in range(args.null_repeats):
            null_results = {}
            for subset, table in subsets.items():
                null_table = table.copy()
                null_table["shank_residual_trace"] = within_block_permutation(
                    table["shank_residual_trace"].to_numpy(float),
                    table["physical_block"].to_numpy(int), rng,
                )
                result, _ = fit_component(
                    null_table, interpolators[2], "shank_residual_trace",
                    scales["shank_residual_trace"], shifts,
                )
                null_results[subset] = result
            null_difference = float(
                np.hypot(
                    null_results["physical_half_0"]["shift_azimuth_deg"]
                    - null_results["physical_half_1"]["shift_azimuth_deg"],
                    null_results["physical_half_0"]["shift_elevation_deg"]
                    - null_results["physical_half_1"]["shift_elevation_deg"],
                )
            )
            null_rows.append(
                {
                    "ecephys_session_id": session_id,
                    "null_repeat": repeat,
                    "real_residual_full_minimum_loss": real_full["minimum_loss"],
                    "null_residual_full_minimum_loss": null_results["full"]["minimum_loss"],
                    "real_residual_physical_half_difference_deg": real_difference,
                    "null_residual_physical_half_difference_deg": null_difference,
                }
            )

    results = pd.DataFrame(result_rows)
    nulls = pd.DataFrame(null_rows)
    results.to_csv(output / "raw_predicted_residual_translation_optima.csv", index=False)
    nulls.to_csv(output / "shank_preserving_residual_null.csv", index=False)
    audit.to_csv(output / "session_shank_decomposition_audit.csv", index=False)
    population.loc[population["ecephys_session_id"].isin(selected_ids)].to_csv(
        output / "selected_unit_shank_decomposition.csv.gz", index=False, compression="gzip"
    )
    figure_path = output / "Figure_v1_dispersion_physical_sampling_control.png"
    render(population, selected, results, landscapes, shifts, nulls, audit, figure_path)

    manifest = {
        "status": "selected-session physical-sampling control checkpoint",
        "selected_sessions": selected_ids,
        "physical_coordinate": "probe_vertical_position in microns; available for every V1 unit",
        "ccf_sensitivity": "first 3D CCF principal component where available",
        "decomposition": (
            "leave-one-cell-out Gaussian smoothing of covariance trace over probe vertical position; "
            "bandwidth selected per session from 40-400 um by cross-fitted MSE"
        ),
        "physical_split": "six nonoverlapping shank-position blocks; alternating blocks assigned to target halves",
        "null": (
            "residual trace permuted within shank blocks while RF centers, block membership, "
            "and shank-predicted component remain fixed"
        ),
        "components": list(COMPONENTS),
        "null_repeats": args.null_repeats,
        "outputs": [
            "Figure_v1_dispersion_physical_sampling_control.png",
            "raw_predicted_residual_translation_optima.csv",
            "shank_preserving_residual_null.csv",
            "session_shank_decomposition_audit.csv",
            "selected_unit_shank_decomposition.csv.gz",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
