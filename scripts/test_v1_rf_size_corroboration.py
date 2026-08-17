#!/usr/bin/env python3
"""Test whether held-out absolute V1 RF size corroborates covariance translation."""

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

from scripts.check_v1_cross_animal_mean_map_support import (
    CCF_COLUMNS,
    DEFAULT_INPUT,
    DEFAULT_UNITS,
    RF_COLUMNS,
    fit_fixed_effect_geometry,
    load_population,
    make_block_table,
    predict,
)
from scripts.check_v1_dispersion_support_geometry import support_decomposition


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
SUPPORT_AUDIT = CHECKPOINT / "cross_animal_mean_map_support_extended" / "all_session_model_audit.csv"
DEFAULT_OUTPUT = CHECKPOINT / "rf_size_corroboration_cases"
POSITIVE_CONTROL = 754312389


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--support-audit", type=Path, default=SUPPORT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-blocks", type=int, default=6)
    parser.add_argument("--rf-neighborhood-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--surface-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--translation-step-deg", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def huber_mean(values: np.ndarray) -> float:
    absolute = np.abs(values)
    return float(np.mean(np.where(absolute <= 1, 0.5 * values**2, absolute - 0.5)))


def robust_scale(values: np.ndarray, floor: float = 0.1) -> float:
    finite = values[np.isfinite(values)]
    return max(float(np.nanquantile(finite, .75) - np.nanquantile(finite, .25)), floor)


def smooth_values(
    points: np.ndarray,
    values: np.ndarray,
    grid_points: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(points).all(axis=1)
    points = points[finite]
    values = values[finite]
    delta = grid_points[:, None, :] - points[None, :, :]
    weights = np.exp(-0.5 * np.sum(delta**2, axis=2) / bandwidth**2)
    total = weights.sum(axis=1)
    effective = total**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
    surface = weights @ values / np.maximum(total, 1e-12)
    surface[effective < 3] = np.nan
    return surface


def mean_template(surfaces: dict[int, np.ndarray], excluded: set[int], minimum: int = 8) -> np.ndarray:
    stack = np.stack([surface for session, surface in surfaces.items() if session not in excluded])
    support = np.sum(np.isfinite(stack), axis=0)
    template = np.divide(
        np.nansum(stack, axis=0),
        support,
        out=np.full(support.shape, np.nan, dtype=float),
        where=support > 0,
    )
    template[support < minimum] = np.nan
    return template


def interpolator(surface: np.ndarray, axis: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (axis, axis), surface.reshape(len(axis), len(axis)), bounds_error=False, fill_value=np.nan
    )


def loss_grid(
    points: np.ndarray,
    observed: np.ndarray,
    template_interpolator: RegularGridInterpolator,
    shifts: np.ndarray,
    scale: float,
) -> np.ndarray:
    losses = np.full(len(shifts), np.nan)
    finite_observed = np.isfinite(observed)
    for index, shift in enumerate(shifts):
        predicted = template_interpolator((points + shift)[:, [1, 0]])
        valid = finite_observed & np.isfinite(predicted)
        if valid.sum() < 10:
            continue
        losses[index] = huber_mean((observed[valid] - predicted[valid]) / scale) + .75 * (1 - valid.mean())
    return losses


def best_shift(losses: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.isfinite(losses).any():
        return np.array([np.nan, np.nan]), np.nan
    index = int(np.nanargmin(losses))
    return shifts[index].copy(), float(losses[index])


def loss_at_shift(
    points: np.ndarray,
    observed: np.ndarray,
    template_interpolator: RegularGridInterpolator,
    shift: np.ndarray,
    scale: float,
) -> float:
    predicted = template_interpolator((points + shift)[:, [1, 0]])
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 10:
        return np.nan
    return huber_mean((observed[valid] - predicted[valid]) / scale) + .75 * (1 - valid.mean())


def deterministic_halves(table: pd.DataFrame, session_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260816 + int(session_id))
    order = rng.permutation(len(table))
    return order[0::2], order[1::2]


def corrected_feature(
    local: pd.DataFrame,
    fit: dict,
    rf_bandwidth: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    predicted = predict(local[list(CCF_COLUMNS)].to_numpy(float), fit, "quadratic")
    decomposition = support_decomposition(
        local["probe_vertical_position"].to_numpy(float),
        local[list(RF_COLUMNS)].to_numpy(float),
        predicted,
        rf_bandwidth,
    )
    result = local.reset_index(drop=True).copy()
    result = pd.concat([result, decomposition], axis=1)
    result["log2_conditional_residual_trace"] = np.log2(
        np.maximum(result["residual_trace_deg2"], 1e-6)
    )
    return result, predicted


def nested_session_features(
    target_id: int,
    population: pd.DataFrame,
    blocks: pd.DataFrame,
    ridge: float,
    rf_bandwidth: float,
) -> tuple[dict[int, pd.DataFrame], dict[int, np.ndarray]]:
    target_specimen = int(
        population.loc[population["ecephys_session_id"].eq(target_id), "specimen_id"].iloc[0]
    )
    features: dict[int, pd.DataFrame] = {}
    predictions: dict[int, np.ndarray] = {}
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        session_id = int(session_id)
        session_specimen = int(local["specimen_id"].iloc[0])
        excluded = {target_specimen, session_specimen}
        if session_id == target_id:
            excluded = {target_specimen}
        training = blocks.loc[~blocks["specimen_id"].isin(excluded)].copy()
        fit = fit_fixed_effect_geometry(training, "quadratic", ridge)
        corrected, predicted = corrected_feature(local, fit, rf_bandwidth)
        features[session_id] = corrected
        predictions[session_id] = predicted
    return features, predictions


def build_scatter_surfaces(
    features: dict[int, pd.DataFrame], grid_points: np.ndarray, bandwidth: float
) -> dict[int, np.ndarray]:
    return {
        session_id: smooth_values(
            local[list(RF_COLUMNS)].to_numpy(float),
            local["log2_conditional_residual_trace"].to_numpy(float),
            grid_points,
            bandwidth,
        )
        for session_id, local in features.items()
    }


def training_shifts(
    target_id: int,
    features: dict[int, pd.DataFrame],
    surfaces: dict[int, np.ndarray],
    axis: np.ndarray,
    shifts: np.ndarray,
    scatter_scale: float,
) -> dict[int, np.ndarray]:
    found = {}
    for session_id, local in features.items():
        if session_id == target_id:
            continue
        template = mean_template(surfaces, {target_id, session_id})
        losses = loss_grid(
            local[list(RF_COLUMNS)].to_numpy(float),
            local["log2_conditional_residual_trace"].to_numpy(float),
            interpolator(template, axis),
            shifts,
            scatter_scale,
        )
        found[session_id] = best_shift(losses, shifts)[0]
    return found


def build_size_template(
    target_id: int,
    features: dict[int, pd.DataFrame],
    training_translation: dict[int, np.ndarray],
    grid_points: np.ndarray,
    bandwidth: float,
    uncensored_only: bool,
) -> np.ndarray:
    surfaces = {}
    for session_id, local in features.items():
        if session_id == target_id:
            continue
        values = local["log2_rf_area_all_fits"].to_numpy(float).copy()
        if uncensored_only:
            values[local["axis_censored"].to_numpy(bool)] = np.nan
        shifted = local[list(RF_COLUMNS)].to_numpy(float) + training_translation[session_id]
        surfaces[session_id] = smooth_values(shifted, values, grid_points, bandwidth)
    return mean_template(surfaces, set())


def split_scatter_landscape(
    local: pd.DataFrame,
    predicted: np.ndarray,
    indices: np.ndarray,
    template_interp: RegularGridInterpolator,
    shifts: np.ndarray,
    scatter_scale: float,
    rf_bandwidth: float,
) -> np.ndarray:
    selected = local.iloc[indices].reset_index(drop=True)
    decomposition = support_decomposition(
        selected["probe_vertical_position"].to_numpy(float),
        selected[list(RF_COLUMNS)].to_numpy(float),
        predicted[indices],
        rf_bandwidth,
    )
    observed = np.log2(np.maximum(decomposition["residual_trace_deg2"].to_numpy(float), 1e-6))
    return loss_grid(
        selected[list(RF_COLUMNS)].to_numpy(float), observed, template_interp, shifts, scatter_scale
    )


def plot_landscape(
    ax: plt.Axes,
    losses: np.ndarray,
    shift_axis: np.ndarray,
    title: str,
    markers: list[tuple[np.ndarray, str, str]],
) -> None:
    relative = losses.reshape(len(shift_axis), len(shift_axis)) - np.nanmin(losses)
    upper = max(float(np.nanquantile(relative[np.isfinite(relative)], .8)), .03)
    artist = ax.imshow(
        relative,
        origin="lower",
        extent=[shift_axis.min(), shift_axis.max(), shift_axis.min(), shift_axis.max()],
        cmap="magma_r",
        norm=Normalize(0, upper),
        aspect="equal",
    )
    for shift, label, color in markers:
        ax.scatter(shift[0], shift[1], marker="*" if label == "covariance" else "o", s=80, color=color, edgecolor="black", linewidth=.6, label=label)
    ax.axhline(0, color="0.6", lw=.6)
    ax.axvline(0, color="0.6", lw=.6)
    ax.set(xlabel="azimuth shift (deg)", ylabel="elevation shift (deg)", title=title)
    plt.colorbar(artist, ax=ax, fraction=.046, pad=.03, label="loss above optimum")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    blocks = make_block_table(population, args.physical_blocks)
    usable = blocks.groupby("ecephys_session_id").size()
    usable = usable.index[usable >= 4]
    population = population.loc[population["ecephys_session_id"].isin(usable)].copy()
    blocks = blocks.loc[blocks["ecephys_session_id"].isin(usable)].copy()
    audit = pd.read_csv(args.support_audit.resolve())
    primary_audit = audit.loc[audit["model"].eq("quadratic")]
    negative = primary_audit.loc[
        primary_audit["heldout_gradient_r2_vs_constant"] < 0,
        "ecephys_session_id",
    ].astype(int).sort_values().tolist()
    selected_ids = negative + [POSITIVE_CONTROL]
    roles = {session_id: "negative held-out CCF→RF gradient" for session_id in negative}
    roles[POSITIVE_CONTROL] = "strongest positive-gradient control"
    selection = pd.DataFrame(
        {
            "ecephys_session_id": selected_ids,
            "selection_role": [roles[value] for value in selected_ids],
            "criterion": ["quadratic held-out gradient R2 < 0" for _ in negative]
            + ["maximum quadratic held-out gradient R2 in the 45-session audit"],
            "selection_stage": "fixed before RF-size outcomes",
        }
    ).merge(
        primary_audit[["ecephys_session_id", "heldout_gradient_r2_vs_constant", "median_sampling_fraction"]],
        on="ecephys_session_id",
        how="left",
    )
    selection.to_csv(output / "case_selection.csv", index=False)

    axis = np.arange(-90.0, 90.0 + args.translation_step_deg, args.translation_step_deg)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    shift_axis = np.arange(-args.translation_bound_deg, args.translation_bound_deg + args.translation_step_deg, args.translation_step_deg)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    zero_index = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
    size_scale = robust_scale(population["log2_rf_area_all_fits"].to_numpy(float), .1)
    rng = np.random.default_rng(20260816)

    summary_rows = []
    landscape_rows = []
    null_rows = []
    case_payload = {}
    for target_index, target_id in enumerate(selected_ids, start=1):
        print(f"target {target_index}/{len(selected_ids)}: {target_id}")
        features, predictions = nested_session_features(
            target_id, population, blocks, args.ridge, args.rf_neighborhood_bandwidth_deg
        )
        scatter_scale = robust_scale(
            np.concatenate([local["log2_conditional_residual_trace"].to_numpy(float) for local in features.values()]),
            .05,
        )
        scatter_surfaces = build_scatter_surfaces(features, grid_points, args.surface_bandwidth_deg)
        scatter_template = mean_template(scatter_surfaces, {target_id})
        scatter_interp = interpolator(scatter_template, axis)
        target = features[target_id].reset_index(drop=True)
        full_scatter_losses = loss_grid(
            target[list(RF_COLUMNS)].to_numpy(float),
            target["log2_conditional_residual_trace"].to_numpy(float),
            scatter_interp,
            shifts,
            scatter_scale,
        )
        covariance_shift, covariance_loss = best_shift(full_scatter_losses, shifts)
        train_shifts = training_shifts(
            target_id, features, scatter_surfaces, axis, shifts, scatter_scale
        )
        size_template_all = build_size_template(
            target_id, features, train_shifts, grid_points, args.surface_bandwidth_deg, False
        )
        size_template_uncensored = build_size_template(
            target_id, features, train_shifts, grid_points, args.surface_bandwidth_deg, True
        )
        size_interp_all = interpolator(size_template_all, axis)
        size_interp_uncensored = interpolator(size_template_uncensored, axis)

        full_size_losses = loss_grid(
            target[list(RF_COLUMNS)].to_numpy(float),
            target["log2_rf_area_all_fits"].to_numpy(float),
            size_interp_all,
            shifts,
            size_scale,
        )
        size_shift, size_minimum = best_shift(full_size_losses, shifts)
        optimum_distance = float(np.linalg.norm(covariance_shift - size_shift))
        first, second = deterministic_halves(target, target_id)
        split_rows = []
        real_gains = []
        null_gains = []
        for estimation_label, estimation_indices, validation_indices in (
            ("half_0_to_half_1", first, second),
            ("half_1_to_half_0", second, first),
        ):
            split_scatter_losses = split_scatter_landscape(
                target,
                predictions[target_id],
                estimation_indices,
                scatter_interp,
                shifts,
                scatter_scale,
                args.rf_neighborhood_bandwidth_deg,
            )
            split_shift, _ = best_shift(split_scatter_losses, shifts)
            validation = target.iloc[validation_indices]
            validation_points = validation[list(RF_COLUMNS)].to_numpy(float)
            validation_sizes = validation["log2_rf_area_all_fits"].to_numpy(float)
            zero_loss = loss_at_shift(validation_points, validation_sizes, size_interp_all, np.zeros(2), size_scale)
            shifted_loss = loss_at_shift(validation_points, validation_sizes, size_interp_all, split_shift, size_scale)
            gain = zero_loss - shifted_loss
            real_gains.append(gain)
            for repeat in range(args.null_repeats):
                shuffled = rng.permutation(validation_sizes)
                null_zero = loss_at_shift(validation_points, shuffled, size_interp_all, np.zeros(2), size_scale)
                null_shift = loss_at_shift(validation_points, shuffled, size_interp_all, split_shift, size_scale)
                null_gain = null_zero - null_shift
                null_gains.append(null_gain)
                null_rows.append(
                    {
                        "ecephys_session_id": target_id,
                        "split": estimation_label,
                        "repeat": repeat,
                        "null_size_gain": null_gain,
                        "real_size_gain": gain,
                    }
                )
            split_rows.append(
                {
                    "split": estimation_label,
                    "scatter_shift_azimuth_deg": split_shift[0],
                    "scatter_shift_elevation_deg": split_shift[1],
                    "heldout_size_loss_zero": zero_loss,
                    "heldout_size_loss_scatter_shift": shifted_loss,
                    "heldout_size_gain": gain,
                }
            )

        size_loss_zero = float(full_size_losses[zero_index])
        size_loss_covariance = loss_at_shift(
            target[list(RF_COLUMNS)].to_numpy(float),
            target["log2_rf_area_all_fits"].to_numpy(float),
            size_interp_all,
            covariance_shift,
            size_scale,
        )
        uncensored_values = target["log2_rf_area_all_fits"].to_numpy(float).copy()
        uncensored_values[target["axis_censored"].to_numpy(bool)] = np.nan
        uncensored_zero = loss_at_shift(
            target[list(RF_COLUMNS)].to_numpy(float), uncensored_values, size_interp_uncensored, np.zeros(2), size_scale
        )
        uncensored_covariance = loss_at_shift(
            target[list(RF_COLUMNS)].to_numpy(float), uncensored_values, size_interp_uncensored, covariance_shift, size_scale
        )
        null_probability = float(np.mean(np.asarray(null_gains) >= np.nanmean(real_gains)))
        summary_rows.append(
            {
                "ecephys_session_id": target_id,
                "selection_role": roles[target_id],
                "v1_units": len(target),
                "censored_fraction": float(target["axis_censored"].mean()),
                "covariance_shift_azimuth_deg": covariance_shift[0],
                "covariance_shift_elevation_deg": covariance_shift[1],
                "size_optimum_azimuth_deg": size_shift[0],
                "size_optimum_elevation_deg": size_shift[1],
                "covariance_size_optimum_distance_deg": optimum_distance,
                "full_size_gain_covariance_vs_zero": size_loss_zero - size_loss_covariance,
                "mean_cross_half_size_gain": float(np.nanmean(real_gains)),
                "cross_half_size_gains_both_positive": bool(np.all(np.asarray(real_gains) > 0)),
                "size_shuffle_p_one_sided": null_probability,
                "uncensored_size_gain_covariance_vs_zero": uncensored_zero - uncensored_covariance,
                "scatter_scale": scatter_scale,
                "size_scale": size_scale,
                **{f"{row['split']}_{key}": value for row in split_rows for key, value in row.items() if key != "split"},
            }
        )
        for name, losses in (("conditional_scatter", full_scatter_losses), ("absolute_size", full_size_losses)):
            for shift, loss in zip(shifts, losses):
                landscape_rows.append(
                    {
                        "ecephys_session_id": target_id,
                        "landscape": name,
                        "shift_azimuth_deg": shift[0],
                        "shift_elevation_deg": shift[1],
                        "loss": loss,
                    }
                )
        case_payload[target_id] = {
            "target": target,
            "scatter_losses": full_scatter_losses,
            "size_losses": full_size_losses,
            "covariance_shift": covariance_shift,
            "size_shift": size_shift,
            "real_gains": np.asarray(real_gains),
            "null_gains": np.asarray(null_gains),
        }

    summary = pd.DataFrame(summary_rows)
    landscapes = pd.DataFrame(landscape_rows)
    nulls = pd.DataFrame(null_rows)
    summary.to_csv(output / "rf_size_corroboration_summary.csv", index=False)
    landscapes.to_csv(output / "rf_size_corroboration_landscapes.csv.gz", index=False, compression="gzip")
    nulls.to_csv(output / "rf_size_corroboration_nulls.csv.gz", index=False, compression="gzip")

    fig, axes = plt.subplots(len(selected_ids), 4, figsize=(17, 4.2 * len(selected_ids)))
    for row_index, target_id in enumerate(selected_ids):
        payload = case_payload[target_id]
        local_summary = summary.loc[summary["ecephys_session_id"].eq(target_id)].iloc[0]
        target = payload["target"]
        scatter = axes[row_index, 0].scatter(
            target["rf_azimuth_deg"], target["rf_elevation_deg"],
            c=target["log2_rf_area_all_fits"], cmap="viridis", s=28,
        )
        censored = target.loc[target["axis_censored"]]
        axes[row_index, 0].scatter(censored["rf_azimuth_deg"], censored["rf_elevation_deg"], facecolors="none", edgecolors="#ef476f", s=48, linewidth=.8)
        axes[row_index, 0].set(xlabel="RF azimuth", ylabel="RF elevation", title=f"{target_id}: {roles[target_id]}\nabsolute RF size; bound={local_summary.censored_fraction:.0%}", aspect="equal")
        fig.colorbar(scatter, ax=axes[row_index, 0], label="log₂ RF area")
        plot_landscape(
            axes[row_index, 1], payload["scatter_losses"], shift_axis,
            "Conditional-scatter translation",
            [(payload["covariance_shift"], "covariance", "white")],
        )
        plot_landscape(
            axes[row_index, 2], payload["size_losses"], shift_axis,
            f"Independent size surface\noptimum distance={local_summary.covariance_size_optimum_distance_deg:.1f}°",
            [(payload["covariance_shift"], "covariance", "cyan"), (payload["size_shift"], "size optimum", "lime")],
        )
        axes[row_index, 3].hist(payload["null_gains"], bins=24, color="#9b8ac4", alpha=.85)
        axes[row_index, 3].axvline(np.mean(payload["real_gains"]), color="#d1495b", lw=2)
        axes[row_index, 3].axvline(0, color="0.4", lw=1)
        axes[row_index, 3].set(xlabel="held-out size loss gain: zero − covariance shift", ylabel="shuffled size arrangements", title=f"Independent-cell validation\nmean gain={np.mean(payload['real_gains']):.3f}; p={local_summary.size_shuffle_p_one_sided:.2f}")
    fig.suptitle(
        "Does absolute V1 RF size corroborate conditional-scatter translation?\n"
        "Nested leave-one-animal-out templates; covariance and size evaluated on independent target-cell halves",
        y=.997,
    )
    fig.tight_layout(rect=(0, 0, 1, .985))
    figure_path = output / "Figure_v1_rf_size_corroboration_cases.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "concrete-case exploratory checkpoint",
        "cases": selection.to_dict(orient="records"),
        "rf_size": "absolute log2 improved axis RF area; no animal normalization; all fits primary",
        "censor_sensitivity": "parameter-bound values excluded from both training and target size only in sensitivity",
        "target_independence": "scatter shift estimated on one deterministic cell half and absolute size tested on the other; repeated both directions",
        "nested_exclusion": "target specimen excluded from every CCF geometry, training scatter translation, and RF-size template",
        "null": "held-out target sizes permuted over fixed RF locations; covariance shift held fixed",
        "surface_bandwidth_deg": args.surface_bandwidth_deg,
        "rf_neighborhood_bandwidth_deg": args.rf_neighborhood_bandwidth_deg,
        "translation_grid": {"bound_deg": args.translation_bound_deg, "step_deg": args.translation_step_deg},
        "outputs": [figure_path.name, "case_selection.csv", "rf_size_corroboration_summary.csv", "rf_size_corroboration_landscapes.csv.gz", "rf_size_corroboration_nulls.csv.gz"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(figure_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
