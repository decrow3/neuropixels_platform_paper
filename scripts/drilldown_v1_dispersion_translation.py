#!/usr/bin/env python3
"""Drill into the selected V1 RF-dispersion translation cases.

The analysis separates covariance trace from anisotropy, measures the retained
absolute-size gradient, tests exact-support descriptor shuffles, and audits the
association with physical CCF sampling where unit-level CCF coordinates exist.
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
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

from scripts.checkpoint_v1_absolute_size_dispersion_translation import (
    FEATURES,
    assign_descriptors,
    build_full_session_surfaces,
    deterministic_split,
    leave_one_out_template,
    robust_scale,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_SELECTION = CHECKPOINT / "uncensored_size_sensitivity" / "selected_case_audit.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "selected_case_drilldown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument(
        "--shuffle-component",
        choices=("covariance trace", "anisotropy", "all dispersion"),
        default="all dispersion",
        help="Component whose split reliability is tested by descriptor-location shuffles.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_interpolators(template: np.ndarray, axis: np.ndarray) -> list[RegularGridInterpolator]:
    shaped = template.reshape(len(axis), len(axis), len(FEATURES))
    return [
        RegularGridInterpolator(
            (axis, axis), shaped[:, :, index], bounds_error=False, fill_value=np.nan
        )
        for index in range(len(FEATURES))
    ]


def candidate_predictions(
    table: pd.DataFrame,
    interpolators: list[RegularGridInterpolator],
    shifts: np.ndarray,
) -> np.ndarray:
    points = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    queries = np.concatenate(
        [(points + shift)[:, [1, 0]] for shift in shifts], axis=0
    )
    predicted = np.column_stack([interpolator(queries) for interpolator in interpolators])
    return predicted.reshape(len(shifts), len(points), len(FEATURES))


def component_losses(
    observed: np.ndarray,
    predicted: np.ndarray,
    scales: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    losses = np.full(len(predicted), np.nan)
    for candidate_index in range(len(predicted)):
        local_prediction = predicted[candidate_index][:, list(indices)]
        local_observed = observed[:, list(indices)]
        if local_observed.ndim == 1:
            local_observed = local_observed[:, None]
            local_prediction = local_prediction[:, None]
        valid = np.isfinite(local_observed).all(axis=1) & np.isfinite(local_prediction).all(axis=1)
        if valid.sum() < 10:
            continue
        residual = (local_observed[valid] - local_prediction[valid]) / scales[list(indices)]
        absolute = np.abs(residual)
        huber = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
        losses[candidate_index] = float(np.mean(huber)) + 0.75 * (1.0 - valid.mean())
    return losses


def optimum_row(losses: np.ndarray, shifts: np.ndarray) -> dict[str, float]:
    if not np.isfinite(losses).any():
        return {
            "shift_azimuth_deg": np.nan,
            "shift_elevation_deg": np.nan,
            "minimum_loss": np.nan,
            "basin_grid_points_delta_005": np.nan,
        }
    index = int(np.nanargmin(losses))
    minimum = float(losses[index])
    return {
        "shift_azimuth_deg": float(shifts[index, 0]),
        "shift_elevation_deg": float(shifts[index, 1]),
        "minimum_loss": minimum,
        "basin_grid_points_delta_005": int(np.sum(losses <= minimum + 0.05)),
    }


def physical_neighbor_rf_covariance(
    rf_points: np.ndarray, ccf_points: np.ndarray, bandwidth_um: float = 250.0
) -> tuple[np.ndarray, np.ndarray]:
    delta = ccf_points[None, :, :] - ccf_points[:, None, :]
    weights = np.exp(-0.5 * np.sum(delta**2, axis=2) / bandwidth_um**2)
    np.fill_diagonal(weights, 0.0)
    weight_sum = weights.sum(axis=1)
    effective = weight_sum**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
    mean = weights @ rf_points / np.maximum(weight_sum[:, None], 1e-12)
    centered = rf_points[None, :, :] - mean[:, None, :]
    cxx = np.sum(weights * centered[:, :, 0] ** 2, axis=1) / np.maximum(weight_sum, 1e-12)
    cyy = np.sum(weights * centered[:, :, 1] ** 2, axis=1) / np.maximum(weight_sum, 1e-12)
    trace = np.log2(np.maximum(cxx + cyy, 1e-6))
    trace[effective < 3] = np.nan
    return trace, effective


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan, np.nan, int(valid.sum())
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def ccf_diagnostics(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    pair_tables = {}
    coordinate_columns = [
        "anterior_posterior_ccf_coordinate",
        "left_right_ccf_coordinate",
        "dorsal_ventral_ccf_coordinate",
    ]
    for session_id, local in table.groupby("ecephys_session_id", observed=True):
        session_id = int(session_id)
        valid = local.dropna(subset=coordinate_columns).copy()
        base = {
            "ecephys_session_id": session_id,
            "v1_units": len(local),
            "ccf_units": len(valid),
            "ccf_available": bool(len(valid) >= 10),
        }
        if len(valid) < 10:
            rows.append(base)
            pair_tables[session_id] = pd.DataFrame()
            continue
        rf = valid[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        ccf = valid[coordinate_columns].to_numpy(float)
        physical_trace, effective = physical_neighbor_rf_covariance(rf, ccf)
        valid["physical_neighbor_rf_log2_trace"] = physical_trace
        valid["physical_neighbor_effective"] = effective
        trace = valid["dispersion_log2_trace"].to_numpy(float)
        rho_physical, p_physical, n_physical = safe_spearman(trace, physical_trace)
        base.update(
            {
                "rf_space_vs_ccf_neighbor_trace_rho": rho_physical,
                "rf_space_vs_ccf_neighbor_trace_p": p_physical,
                "rf_space_vs_ccf_neighbor_trace_n": n_physical,
            }
        )
        for label, column in zip(("ap", "ml", "dv"), coordinate_columns):
            rho, pvalue, count = safe_spearman(trace, valid[column].to_numpy(float))
            base[f"rf_space_trace_vs_{label}_rho"] = rho
            base[f"rf_space_trace_vs_{label}_p"] = pvalue
            base[f"rf_space_trace_vs_{label}_n"] = count
        delta_rf = rf[:, None, :] - rf[None, :, :]
        delta_ccf = ccf[:, None, :] - ccf[None, :, :]
        rf_distance = np.sqrt(np.sum(delta_rf**2, axis=2))
        ccf_distance = np.sqrt(np.sum(delta_ccf**2, axis=2))
        upper = np.triu_indices(len(valid), 1)
        rho_pair, p_pair, n_pair = safe_spearman(ccf_distance[upper], rf_distance[upper])
        base.update(
            {
                "pairwise_ccf_vs_rf_distance_rho": rho_pair,
                "pairwise_ccf_vs_rf_distance_p": p_pair,
                "pairwise_ccf_vs_rf_distance_n": n_pair,
            }
        )
        rows.append(base)
        pairs = pd.DataFrame(
            {
                "ccf_distance_um": ccf_distance[upper],
                "rf_distance_deg": rf_distance[upper],
            }
        )
        if len(pairs) > 2500:
            pairs = pairs.sample(2500, random_state=20260816 + session_id)
        pair_tables[session_id] = pairs
    return pd.DataFrame(rows), pair_tables


def size_gradient_summary(
    session_id: int,
    table: pd.DataFrame,
    template: np.ndarray,
    axis: np.ndarray,
    size_shift: np.ndarray,
) -> dict[str, float]:
    surface = template[:, 0].reshape(len(axis), len(axis))
    grad_el, grad_az = np.gradient(surface, axis, axis)
    magnitude = np.sqrt(grad_az**2 + grad_el**2)
    value_interp = RegularGridInterpolator(
        (axis, axis), surface, bounds_error=False, fill_value=np.nan
    )
    gradient_interp = RegularGridInterpolator(
        (axis, axis), magnitude, bounds_error=False, fill_value=np.nan
    )
    corrected = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float) + size_shift
    query = corrected[:, [1, 0]]
    values = value_interp(query)
    gradients = gradient_interp(query)
    return {
        "ecephys_session_id": session_id,
        "template_log2_area_range_at_corrected_cells": float(np.nanmax(values) - np.nanmin(values)),
        "template_median_gradient_log2_per_deg": float(np.nanmedian(gradients)),
        "template_p90_gradient_log2_per_deg": float(np.nanquantile(gradients, 0.9)),
        "corrected_cells_with_template_support": int(np.isfinite(values).sum()),
    }


def plot_landscape(
    axis_object: plt.Axes,
    losses: np.ndarray,
    shifts: np.ndarray,
    title: str,
    optima: list[tuple[str, dict[str, float]]],
) -> None:
    values = np.unique(shifts[:, 0])
    relative = losses.reshape(len(values), len(values)) - np.nanmin(losses)
    upper = max(float(np.nanquantile(relative[np.isfinite(relative)], 0.8)), 0.03)
    artist = axis_object.imshow(
        relative,
        origin="lower",
        extent=[values.min(), values.max(), values.min(), values.max()],
        cmap="magma_r",
        norm=Normalize(0, upper),
        aspect="equal",
    )
    styles = {
        "full": ("*", "white", 85),
        "half 0": ("o", "#27c2ff", 40),
        "half 1": ("s", "#55e06f", 40),
    }
    for label, result in optima:
        marker, color, size = styles[label]
        axis_object.scatter(
            result["shift_azimuth_deg"], result["shift_elevation_deg"],
            marker=marker, facecolor=color, edgecolor="black", linewidth=0.7,
            s=size, zorder=5, label=label,
        )
    axis_object.axhline(0, color="#999999", linewidth=0.5)
    axis_object.axvline(0, color="#999999", linewidth=0.5)
    axis_object.set(
        xlabel="Azimuth correction (deg)", ylabel="Elevation correction (deg)", title=title
    )
    plt.colorbar(artist, ax=axis_object, fraction=0.046, pad=0.03, label="loss above optimum")


def render(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    templates: dict[int, np.ndarray],
    axis: np.ndarray,
    shifts: np.ndarray,
    component_results: pd.DataFrame,
    component_landscapes: dict[tuple[int, str], np.ndarray],
    shuffle: pd.DataFrame,
    gradients: pd.DataFrame,
    ccf: pd.DataFrame,
    pairs: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(len(selected), 6, figsize=(25, 4.8 * len(selected)), squeeze=False)
    roles = selected.set_index("ecephys_session_id")["selection_role"].to_dict()
    for row_index, session_id in enumerate(selected["ecephys_session_id"].astype(int)):
        local = population.loc[population["ecephys_session_id"].eq(session_id)]
        template = templates[session_id]
        size = template[:, 0].reshape(len(axis), len(axis))
        local_results = component_results.loc[component_results["ecephys_session_id"].eq(session_id)]
        size_full = local_results.loc[
            local_results["component"].eq("absolute size")
            & local_results["target_subset"].eq("full")
        ].iloc[0]
        corrected = local[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float) + np.array(
            [size_full.shift_azimuth_deg, size_full.shift_elevation_deg]
        )
        finite = size[np.isfinite(size)]
        artist = axes[row_index, 0].imshow(
            size,
            origin="lower",
            extent=[axis.min(), axis.max(), axis.min(), axis.max()],
            cmap="viridis",
            norm=Normalize(*np.nanquantile(finite, [0.02, 0.98])),
            aspect="equal",
        )
        axes[row_index, 0].scatter(corrected[:, 0], corrected[:, 1], s=9, c="white", alpha=0.5)
        gradient = gradients.loc[gradients["ecephys_session_id"].eq(session_id)].iloc[0]
        axes[row_index, 0].set(
            xlim=(-60, 60), ylim=(-60, 60),
            xlabel="Corrected azimuth (deg)", ylabel="Corrected elevation (deg)",
            title=(
                f"{roles[session_id]}\nsession {session_id}: LOO absolute-size template\n"
                f"range={gradient.template_log2_area_range_at_corrected_cells:.2f} log2; "
                f"median |grad|={gradient.template_median_gradient_log2_per_deg:.3f}/deg"
            ),
        )
        plt.colorbar(artist, ax=axes[row_index, 0], fraction=0.046, pad=0.03, label="template log₂ area")

        for column, component in enumerate(("covariance trace", "anisotropy", "all dispersion"), start=1):
            result_rows = local_results.loc[local_results["component"].eq(component)]
            optima = []
            for subset, label in (("full", "full"), ("half_0", "half 0"), ("half_1", "half 1")):
                row = result_rows.loc[result_rows["target_subset"].eq(subset)].iloc[0]
                optima.append((label, row.to_dict()))
            plot_landscape(
                axes[row_index, column],
                component_landscapes[(session_id, component)],
                shifts,
                component,
                optima,
            )

        local_shuffle = shuffle.loc[shuffle["ecephys_session_id"].eq(session_id)]
        real = local_shuffle["real_split_difference_deg"].iloc[0]
        axes[row_index, 4].hist(
            local_shuffle["null_split_difference_deg"], bins=np.arange(0, 90, 5),
            color="#9b8ac4", alpha=0.85,
        )
        axes[row_index, 4].axvline(real, color="#d1495b", linewidth=2, label=f"real = {real:.1f}°")
        percentile = 100 * np.mean(local_shuffle["null_split_difference_deg"] <= real)
        shuffle_component = local_shuffle["shuffle_component"].iloc[0]
        axes[row_index, 4].set(
            xlabel="Half-to-half optimum difference (deg)", ylabel="Support-matched shuffles",
            title=(
                f"{shuffle_component} location-shuffle control\n"
                f"{percentile:.1f}% of null as or more reproducible"
            ),
        )
        axes[row_index, 4].legend(fontsize=8)

        local_ccf = ccf.loc[ccf["ecephys_session_id"].eq(session_id)].iloc[0]
        local_pairs = pairs[session_id]
        if len(local_pairs):
            axes[row_index, 5].scatter(
                local_pairs["ccf_distance_um"], local_pairs["rf_distance_deg"],
                s=8, alpha=0.18, color="#2878a8",
            )
            axes[row_index, 5].set(
                xlabel="Pairwise CCF distance (µm)", ylabel="Pairwise RF-center distance (deg)",
                title=(
                    "Physical sampling diagnostic\n"
                    f"pairwise rho={local_ccf.pairwise_ccf_vs_rf_distance_rho:.2f}; "
                    f"RF-vs-CCF local trace rho={local_ccf.rf_space_vs_ccf_neighbor_trace_rho:.2f}"
                ),
            )
        else:
            axes[row_index, 5].text(
                0.5, 0.5, "Unit-level CCF coordinates\nnot available for this session",
                ha="center", va="center", transform=axes[row_index, 5].transAxes, fontsize=12,
            )
            axes[row_index, 5].set(title="Physical sampling diagnostic", xticks=[], yticks=[])
        for axis_object in axes[row_index]:
            axis_object.grid(alpha=0.12)
    handles, labels = axes[0, 3].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles[:3], labels[:3], loc="upper center", bbox_to_anchor=(0.5, 0.958),
            ncol=3, frameon=False,
        )
    figure.suptitle(
        "Selected-case drill-down: what drives V1 RF-center dispersion registration?\n"
        "RF-space covariance before translation; no SF/TF and no RF-center edge exclusion",
        fontsize=16, y=0.997,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.925])
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(args.input.resolve(), low_memory=False)
    selected = pd.read_csv(args.selection.resolve())
    selected_ids = selected["ecephys_session_id"].astype(int).tolist()
    axis = np.arange(-90.0, 92.0, 2.0)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    surfaces, _ = build_full_session_surfaces(population, grid_points, bandwidth=12.0)
    templates = {session_id: leave_one_out_template(surfaces, session_id)[0] for session_id in selected_ids}
    scales = np.array(
        [
            robust_scale(population[feature].to_numpy(float), 0.10 if index == 0 else 0.05)
            for index, feature in enumerate(FEATURES)
        ]
    )
    shift_axis = np.arange(-30.0, 32.0, 2.0)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    components = {
        "absolute size": (0,),
        "covariance trace": (1,),
        "anisotropy": (2, 3),
        "all dispersion": (1, 2, 3),
    }
    result_rows = []
    landscapes = {}
    shuffle_rows = []
    gradient_rows = []
    rng = np.random.default_rng(20260816)

    for session_id in selected_ids:
        local = population.loc[population["ecephys_session_id"].eq(session_id)].copy()
        template = templates[session_id]
        interpolators = make_interpolators(template, axis)
        half_zero, half_one = deterministic_split(local, session_id)
        subsets = {"full": local, "half_0": half_zero, "half_1": half_one}
        cached = {}
        for subset, table in subsets.items():
            observed = table[list(FEATURES)].to_numpy(float)
            predicted = candidate_predictions(table, interpolators, shifts)
            cached[subset] = (observed, predicted)
            for component, indices in components.items():
                losses = component_losses(observed, predicted, scales, indices)
                result_rows.append(
                    {
                        "ecephys_session_id": session_id,
                        "target_subset": subset,
                        "component": component,
                        **optimum_row(losses, shifts),
                    }
                )
                if subset == "full":
                    landscapes[(session_id, component)] = losses
        result_frame = pd.DataFrame(result_rows)
        size_result = result_frame.loc[
            result_frame["ecephys_session_id"].eq(session_id)
            & result_frame["target_subset"].eq("full")
            & result_frame["component"].eq("absolute size")
        ].iloc[0]
        gradient_rows.append(
            size_gradient_summary(
                session_id, local, template, axis,
                np.array([size_result.shift_azimuth_deg, size_result.shift_elevation_deg]),
            )
        )

        shuffle_indices = components[args.shuffle_component]
        real_results = result_frame.loc[
            result_frame["ecephys_session_id"].eq(session_id)
            & result_frame["component"].eq(args.shuffle_component)
        ].set_index("target_subset")
        real_split = float(
            np.hypot(
                real_results.loc["half_0", "shift_azimuth_deg"]
                - real_results.loc["half_1", "shift_azimuth_deg"],
                real_results.loc["half_0", "shift_elevation_deg"]
                - real_results.loc["half_1", "shift_elevation_deg"],
            )
        )
        real_full_loss = float(real_results.loc["full", "minimum_loss"])
        for repeat in range(args.shuffle_repeats):
            null_optima = {}
            for subset in ("full", "half_0", "half_1"):
                observed, predicted = cached[subset]
                permuted = observed.copy()
                order = rng.permutation(len(permuted))
                permuted[:, list(shuffle_indices)] = permuted[order][:, list(shuffle_indices)]
                losses = component_losses(permuted, predicted, scales, shuffle_indices)
                null_optima[subset] = optimum_row(losses, shifts)
            null_split = float(
                np.hypot(
                    null_optima["half_0"]["shift_azimuth_deg"]
                    - null_optima["half_1"]["shift_azimuth_deg"],
                    null_optima["half_0"]["shift_elevation_deg"]
                    - null_optima["half_1"]["shift_elevation_deg"],
                )
            )
            shuffle_rows.append(
                {
                    "ecephys_session_id": session_id,
                    "shuffle_repeat": repeat,
                    "shuffle_component": args.shuffle_component,
                    "real_full_minimum_loss": real_full_loss,
                    "null_full_minimum_loss": null_optima["full"]["minimum_loss"],
                    "real_split_difference_deg": real_split,
                    "null_split_difference_deg": null_split,
                }
            )

    component_results = pd.DataFrame(result_rows)
    shuffle = pd.DataFrame(shuffle_rows)
    gradients = pd.DataFrame(gradient_rows)

    units = pd.read_csv(
        args.unit_table.resolve(),
        usecols=[
            "ecephys_unit_id",
            "anterior_posterior_ccf_coordinate",
            "left_right_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
        ],
        low_memory=False,
    )
    selected_population = population.loc[
        population["ecephys_session_id"].isin(selected_ids)
    ].merge(units, on="ecephys_unit_id", how="left")
    ccf, pair_tables = ccf_diagnostics(selected_population)

    component_results.to_csv(output / "component_translation_optima.csv", index=False)
    shuffle.to_csv(output / "support_matched_descriptor_shuffle.csv", index=False)
    gradients.to_csv(output / "absolute_size_template_gradient_summary.csv", index=False)
    ccf.to_csv(output / "ccf_sampling_diagnostics.csv", index=False)
    pd.concat(
        [table.assign(ecephys_session_id=session_id) for session_id, table in pair_tables.items() if len(table)],
        ignore_index=True,
    ).to_csv(output / "sampled_pairwise_ccf_rf_distances.csv", index=False)
    figure_path = output / "Figure_selected_v1_dispersion_drilldown.png"
    render(
        selected_population, selected, templates, axis, shifts, component_results,
        landscapes, shuffle, gradients, ccf, pair_tables, figure_path,
    )
    manifest = {
        "status": "selected-case drill-down checkpoint",
        "selected_sessions": selected_ids,
        "dispersion_definition": (
            "per-cell local covariance of same-session RF centers using 15-degree RF-space "
            "Gaussian weights, computed before translation; no CCF coordinates enter the registration"
        ),
        "controls": {
            "components": list(components),
            "support_matched_shuffle": (
                "exact RF centers retained; covariance descriptors permuted across those centers "
                "independently in full and split target sets"
            ),
            "shuffle_component": args.shuffle_component,
            "ccf": "post hoc association only; 250-um 3D CCF-neighborhood RF covariance",
            "shuffle_repeats": args.shuffle_repeats,
        },
        "outputs": [
            "Figure_selected_v1_dispersion_drilldown.png",
            "component_translation_optima.csv",
            "support_matched_descriptor_shuffle.csv",
            "absolute_size_template_gradient_summary.csv",
            "ccf_sampling_diagnostics.csv",
            "sampled_pairwise_ccf_rf_distances.csv",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
