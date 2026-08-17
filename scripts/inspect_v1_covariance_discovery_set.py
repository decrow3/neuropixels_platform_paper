#!/usr/bin/env python3
"""Inspect anatomy-corrected covariance translation in a quality-selected discovery set."""

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
from scripts.check_v1_dispersion_physical_sampling import physical_blocks
from scripts.check_v1_dispersion_support_geometry import support_decomposition
from scripts.test_v1_rf_size_corroboration import (
    best_shift,
    build_scatter_surfaces,
    deterministic_halves,
    interpolator,
    loss_grid,
    mean_template,
    nested_session_features,
    plot_landscape,
    robust_scale,
    smooth_values,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
SUPPORT_AUDIT = CHECKPOINT / "cross_animal_mean_map_support_extended" / "all_session_model_audit.csv"
DEFAULT_OUTPUT = CHECKPOINT / "covariance_discovery_set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--support-audit", type=Path, default=SUPPORT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rf-neighborhood-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--surface-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--translation-step-deg", type=float, default=2.0)
    parser.add_argument("--physical-block-count", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quality_table(population: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_path)
    audit = audit.loc[audit["model"].eq("quadratic")].copy()
    rows = []
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        az_span = float(local["rf_azimuth_deg"].quantile(.9) - local["rf_azimuth_deg"].quantile(.1))
        el_span = float(local["rf_elevation_deg"].quantile(.9) - local["rf_elevation_deg"].quantile(.1))
        rows.append(
            {
                "ecephys_session_id": int(session_id),
                "censored_fraction": float(local["axis_censored"].mean()),
                "median_test_deviance": float(local["axis_test_deviance"].median()),
                "robust_azimuth_span_deg": az_span,
                "robust_elevation_span_deg": el_span,
                "robust_rf_support_deg": float(np.hypot(az_span, el_span)),
            }
        )
    quality = audit.merge(pd.DataFrame(rows), on="ecephys_session_id", how="inner")
    gates = {
        "gate_units": quality["target_units"] >= 80,
        "gate_gradient": quality["heldout_gradient_r2_vs_constant"] >= .5,
        "gate_sampling": quality["median_sampling_fraction"] < .1,
        "gate_ccf_step": quality["maximum_consecutive_tangential_ccf_step_um"] < 150,
        "gate_censoring": quality["censored_fraction"] < .3,
        "gate_rf_support": quality["robust_rf_support_deg"] >= 20,
        "gate_fit_quality": quality["median_test_deviance"] >= .9,
    }
    for name, values in gates.items():
        quality[name] = values
    quality["discovery_selected"] = np.column_stack(list(gates.values())).all(axis=1)
    return quality.sort_values("heldout_gradient_r2_vs_constant", ascending=False)


def subset_feature(
    local: pd.DataFrame,
    predicted: np.ndarray,
    indices: np.ndarray,
    rf_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = local.iloc[indices].reset_index(drop=True)
    decomposition = support_decomposition(
        selected["probe_vertical_position"].to_numpy(float),
        selected[list(RF_COLUMNS)].to_numpy(float),
        predicted[indices],
        rf_bandwidth,
    )
    values = np.log2(np.maximum(decomposition["residual_trace_deg2"].to_numpy(float), 1e-6))
    return selected[list(RF_COLUMNS)].to_numpy(float), values


def optimum_metrics(losses: np.ndarray, shifts: np.ndarray, bound: float) -> dict[str, float]:
    shift, minimum = best_shift(losses, shifts)
    return {
        "shift_azimuth_deg": float(shift[0]),
        "shift_elevation_deg": float(shift[1]),
        "minimum_loss": minimum,
        "basin_grid_points_delta_005": int(np.sum(losses <= minimum + .05)),
        "at_bound": bool(np.any(np.abs(shift) >= bound)),
    }


def average_surfaces(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    support = np.sum(np.isfinite(stack), axis=0)
    return np.divide(
        np.nansum(stack, axis=0),
        support,
        out=np.full(support.shape, np.nan, dtype=float),
        where=support > 0,
    )


def matched_subset_template(
    target_id: int,
    features: dict[int, pd.DataFrame],
    predictions: dict[int, np.ndarray],
    subset_kind: str,
    grid_points: np.ndarray,
    surface_bandwidth: float,
    rf_bandwidth: float,
    block_count: int,
) -> np.ndarray:
    session_surfaces: dict[int, np.ndarray] = {}
    for session_id, local in features.items():
        if session_id == target_id:
            continue
        subset_indices: list[np.ndarray] = []
        if subset_kind == "cell":
            subset_indices.extend(deterministic_halves(local, session_id))
        elif subset_kind == "physical":
            block_label = physical_blocks(local["probe_vertical_position"], block_count)
            for parity in (0, 1):
                subset_indices.append(
                    np.flatnonzero((block_label >= 0) & (block_label % 2 == parity))
                )
        else:
            raise ValueError(subset_kind)
        surfaces = []
        for indices in subset_indices:
            if len(indices) < 10:
                continue
            points, values = subset_feature(local, predictions[session_id], indices, rf_bandwidth)
            surfaces.append(
                smooth_values(points, values, grid_points, surface_bandwidth)
            )
        if surfaces:
            session_surfaces[session_id] = average_surfaces(surfaces)
    return mean_template(session_surfaces, set())


def square_limits(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, float]:
    combined = np.vstack([observed, predicted])
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2
    half = max(float(np.ptp(combined[:, 0])), float(np.ptp(combined[:, 1])), 6.0) / 2 + 1
    return center, half


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    blocks = make_block_table(population, args.physical_block_count)
    usable = blocks.groupby("ecephys_session_id").size()
    usable = usable.index[usable >= 4]
    population = population.loc[population["ecephys_session_id"].isin(usable)].copy()
    blocks = blocks.loc[blocks["ecephys_session_id"].isin(usable)].copy()
    quality = quality_table(population, args.support_audit.resolve())
    selected = quality.loc[quality["discovery_selected"]].copy()
    selected_ids = selected["ecephys_session_id"].astype(int).tolist()
    quality.to_csv(output / "discovery_quality_audit_all_sessions.csv", index=False)
    selected.to_csv(output / "discovery_set_selection.csv", index=False)

    axis = np.arange(-90.0, 90.0 + args.translation_step_deg, args.translation_step_deg)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    shift_axis = np.arange(-args.translation_bound_deg, args.translation_bound_deg + args.translation_step_deg, args.translation_step_deg)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    rng = np.random.default_rng(20260816)

    result_rows = []
    null_rows = []
    landscape_rows = []
    payload = {}
    for target_index, target_id in enumerate(selected_ids, start=1):
        print(f"target {target_index}/{len(selected_ids)}: {target_id}")
        features, predictions = nested_session_features(
            target_id,
            population,
            blocks,
            args.ridge,
            args.rf_neighborhood_bandwidth_deg,
        )
        scatter_scale = robust_scale(
            np.concatenate(
                [local["log2_conditional_residual_trace"].to_numpy(float) for local in features.values()]
            ),
            .05,
        )
        surfaces = build_scatter_surfaces(features, grid_points, args.surface_bandwidth_deg)
        template = mean_template(surfaces, {target_id})
        template_interps = {
            "full": interpolator(template, axis),
            "cell": interpolator(
                matched_subset_template(
                    target_id, features, predictions, "cell", grid_points,
                    args.surface_bandwidth_deg, args.rf_neighborhood_bandwidth_deg,
                    args.physical_block_count,
                ),
                axis,
            ),
            "physical": interpolator(
                matched_subset_template(
                    target_id, features, predictions, "physical", grid_points,
                    args.surface_bandwidth_deg, args.rf_neighborhood_bandwidth_deg,
                    args.physical_block_count,
                ),
                axis,
            ),
        }
        target = features[target_id].reset_index(drop=True)
        predicted = predictions[target_id]

        variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "full": (
                target[list(RF_COLUMNS)].to_numpy(float),
                target["log2_conditional_residual_trace"].to_numpy(float),
            )
        }
        first, second = deterministic_halves(target, target_id)
        variants["cell_half_0"] = subset_feature(target, predicted, first, args.rf_neighborhood_bandwidth_deg)
        variants["cell_half_1"] = subset_feature(target, predicted, second, args.rf_neighborhood_bandwidth_deg)
        block_label = physical_blocks(target["probe_vertical_position"], args.physical_block_count)
        for parity in (0, 1):
            indices = np.flatnonzero((block_label >= 0) & (block_label % 2 == parity))
            variants[f"physical_half_{parity}"] = subset_feature(
                target, predicted, indices, args.rf_neighborhood_bandwidth_deg
            )

        local_results = {}
        local_losses = {}
        for variant, (points, values) in variants.items():
            template_kind = (
                "cell" if variant.startswith("cell_")
                else "physical" if variant.startswith("physical_")
                else "full"
            )
            losses = loss_grid(
                points, values, template_interps[template_kind], shifts, scatter_scale
            )
            metrics = optimum_metrics(losses, shifts, args.translation_bound_deg)
            local_results[variant] = metrics
            local_losses[variant] = losses
            result_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "target_subset": variant,
                    "subset_units": len(points),
                    **metrics,
                }
            )
        cell_difference = float(
            np.hypot(
                local_results["cell_half_0"]["shift_azimuth_deg"] - local_results["cell_half_1"]["shift_azimuth_deg"],
                local_results["cell_half_0"]["shift_elevation_deg"] - local_results["cell_half_1"]["shift_elevation_deg"],
            )
        )
        physical_difference = float(
            np.hypot(
                local_results["physical_half_0"]["shift_azimuth_deg"] - local_results["physical_half_1"]["shift_azimuth_deg"],
                local_results["physical_half_0"]["shift_elevation_deg"] - local_results["physical_half_1"]["shift_elevation_deg"],
            )
        )
        for row in result_rows[-len(variants):]:
            row["cell_half_vector_difference_deg"] = cell_difference
            row["physical_half_vector_difference_deg"] = physical_difference

        full_points, full_values = variants["full"]
        null_minima = []
        null_distances = []
        full_shift = np.array(
            [local_results["full"]["shift_azimuth_deg"], local_results["full"]["shift_elevation_deg"]]
        )
        for repeat in range(args.null_repeats):
            permuted = rng.permutation(full_values)
            losses = loss_grid(
                full_points, permuted, template_interps["full"], shifts, scatter_scale
            )
            null_shift, null_minimum = best_shift(losses, shifts)
            null_minima.append(null_minimum)
            null_distances.append(float(np.linalg.norm(null_shift - full_shift)))
            null_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "repeat": repeat,
                    "null_minimum_loss": null_minimum,
                    "null_shift_azimuth_deg": null_shift[0],
                    "null_shift_elevation_deg": null_shift[1],
                    "null_distance_from_real_shift_deg": null_distances[-1],
                }
            )
        real_minimum = local_results["full"]["minimum_loss"]
        null_fit_p = float(np.mean(np.asarray(null_minima) <= real_minimum))
        for row in result_rows[-len(variants):]:
            row["exact_support_shuffle_fit_p"] = null_fit_p

        target_blocks = blocks.loc[blocks["ecephys_session_id"].eq(target_id)].copy()
        target_specimen = int(target["specimen_id"].iloc[0])
        geometry_fit = fit_fixed_effect_geometry(
            blocks.loc[blocks["specimen_id"].ne(target_specimen)], "quadratic", args.ridge
        )
        block_prediction = predict(target_blocks[list(CCF_COLUMNS)].to_numpy(float), geometry_fit, "quadratic")
        display_translation = target_blocks[list(RF_COLUMNS)].to_numpy(float).mean(axis=0) - block_prediction.mean(axis=0)
        block_prediction_display = block_prediction + display_translation

        for variant, losses in local_losses.items():
            for shift, loss in zip(shifts, losses):
                landscape_rows.append(
                    {
                        "ecephys_session_id": target_id,
                        "target_subset": variant,
                        "shift_azimuth_deg": shift[0],
                        "shift_elevation_deg": shift[1],
                        "loss": loss,
                    }
                )
        payload[target_id] = {
            "target": target,
            "target_blocks": target_blocks,
            "block_prediction": block_prediction_display,
            "results": local_results,
            "losses": local_losses,
            "null_minima": np.asarray(null_minima),
            "real_minimum": real_minimum,
            "null_fit_p": null_fit_p,
            "cell_difference": cell_difference,
            "physical_difference": physical_difference,
        }

    results = pd.DataFrame(result_rows)
    nulls = pd.DataFrame(null_rows)
    landscapes = pd.DataFrame(landscape_rows)
    results.to_csv(output / "discovery_covariance_translation_results.csv", index=False)
    nulls.to_csv(output / "discovery_exact_support_shuffle_null.csv.gz", index=False, compression="gzip")
    landscapes.to_csv(output / "discovery_covariance_landscapes.csv.gz", index=False, compression="gzip")

    fig, axes = plt.subplots(len(selected_ids), 6, figsize=(25, 4.3 * len(selected_ids)))
    trace_limits = np.nanquantile(
        np.concatenate([payload[value]["target"]["log2_conditional_residual_trace"].to_numpy(float) for value in selected_ids]),
        [.02, .98],
    )
    size_limits = np.nanquantile(population["log2_rf_area_all_fits"], [.02, .98])
    marker_styles = {
        "full": ("*", "white", 105),
        "cell_half_0": ("o", "#24c4f0", 45),
        "cell_half_1": ("s", "#56db6c", 45),
        "physical_half_0": ("^", "#ff9f1c", 50),
        "physical_half_1": ("D", "#d36ad3", 42),
    }
    for row_index, target_id in enumerate(selected_ids):
        local = payload[target_id]
        target = local["target"]
        target_blocks = local["target_blocks"]
        observed_blocks = target_blocks[list(RF_COLUMNS)].to_numpy(float)
        predicted_blocks = local["block_prediction"]

        ax = axes[row_index, 0]
        ax.scatter(observed_blocks[:, 0], observed_blocks[:, 1], s=48, label="observed block")
        ax.scatter(predicted_blocks[:, 0], predicted_blocks[:, 1], marker="x", s=55, label="LOAO anatomy prediction")
        for observed, predicted_value in zip(observed_blocks, predicted_blocks):
            ax.plot([observed[0], predicted_value[0]], [observed[1], predicted_value[1]], color="0.65", lw=.8)
        center, half = square_limits(observed_blocks, predicted_blocks)
        quality_row = selected.loc[selected["ecephys_session_id"].eq(target_id)].iloc[0]
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), xlabel="RF azimuth", ylabel="RF elevation", title=f"{target_id}: independent anatomy gate\nR²={quality_row.heldout_gradient_r2_vs_constant:.2f}; correction={100*quality_row.median_sampling_fraction:.1f}%")
        ax.set_aspect("equal", adjustable="box")
        if row_index == 0:
            ax.legend(frameon=False, fontsize=7)

        ax = axes[row_index, 1]
        scatter = ax.scatter(target["rf_azimuth_deg"], target["rf_elevation_deg"], c=target["log2_conditional_residual_trace"], cmap="cividis", norm=Normalize(*trace_limits), s=28)
        ax.set(xlabel="RF azimuth", ylabel="RF elevation", title="Anatomy-corrected local scatter", aspect="equal")
        fig.colorbar(scatter, ax=ax, label="log₂ residual covariance trace")

        ax = axes[row_index, 2]
        scatter = ax.scatter(target["rf_azimuth_deg"], target["rf_elevation_deg"], c=target["log2_rf_area_all_fits"], cmap="viridis", norm=Normalize(*size_limits), s=28)
        censored = target.loc[target["axis_censored"]]
        ax.scatter(censored["rf_azimuth_deg"], censored["rf_elevation_deg"], facecolors="none", edgecolors="#ef476f", s=48, linewidth=.8)
        ax.set(xlabel="RF azimuth", ylabel="RF elevation", title=f"Descriptive absolute RF size\nbound={target.axis_censored.mean():.0%}", aspect="equal")
        fig.colorbar(scatter, ax=ax, label="log₂ RF area")

        markers = []
        for variant, result in local["results"].items():
            _, color, _ = marker_styles[variant]
            markers.append((np.array([result["shift_azimuth_deg"], result["shift_elevation_deg"]]), variant, color))
        plot_landscape(
            axes[row_index, 3],
            local["losses"]["full"],
            shift_axis,
            f"Conditional-scatter objective\nbasin={local['results']['full']['basin_grid_points_delta_005']} grid points",
            markers,
        )

        ax = axes[row_index, 4]
        for variant, result in local["results"].items():
            marker, color, size = marker_styles[variant]
            ax.scatter(result["shift_azimuth_deg"], result["shift_elevation_deg"], marker=marker, color=color, edgecolor="black", linewidth=.6, s=size, label=variant.replace("_", " "))
        ax.axhline(0, color="0.6", lw=.6)
        ax.axvline(0, color="0.6", lw=.6)
        ax.set(xlim=(-32, 32), ylim=(-32, 32), xlabel="azimuth shift", ylabel="elevation shift", title=f"Reproducibility\ncell Δ={local['cell_difference']:.1f}°; physical Δ={local['physical_difference']:.1f}°", aspect="equal")
        if row_index == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")

        ax = axes[row_index, 5]
        ax.hist(local["null_minima"], bins=22, color="#9b8ac4", alpha=.85)
        ax.axvline(local["real_minimum"], color="#d1495b", lw=2)
        ax.set(xlabel="minimum conditional-scatter loss", ylabel="exact-support shuffles", title=f"Descriptor-location null\np={local['null_fit_p']:.2f}")

    fig.suptitle(
        "V1 covariance discovery set selected without covariance-registration outcomes\n"
        "Full field, cell halves, interleaved physical blocks, and exact-support shuffle",
        y=.998,
    )
    fig.tight_layout(rect=(0, 0, 1, .988))
    figure_path = output / "Figure_v1_covariance_discovery_set.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "high-confidence discovery-set checkpoint",
        "selection_is_independent_of_covariance_registration_outcomes": True,
        "selection_gates": {
            "target_units": ">=80",
            "heldout_ccf_rf_gradient_r2": ">=0.5",
            "median_sampling_fraction": "<0.10",
            "maximum_consecutive_tangential_ccf_step_um": "<150",
            "censored_fraction": "<0.30",
            "robust_rf_support_deg": ">=20",
            "median_test_deviance": ">=0.9",
        },
        "selected_sessions": selected_ids,
        "evaluation": {
            "covariance": "log2 local covariance trace of RF residual vectors after independently learned CCF mean map",
            "cell_split": "deterministic independent halves; covariance recomputed within each half",
            "split_templates": "training sessions recomputed at matching cell-half or physical-half density; the two subset surfaces are averaged within training animal before animal-balanced templating",
            "physical_split": "six physical blocks; alternating blocks assigned to halves; covariance recomputed",
            "null": "conditional-scatter descriptor permuted over exact RF locations",
            "rf_size": "descriptive absolute improved-fit log2 area only; not an acceptance criterion",
        },
        "outputs": [figure_path.name, "discovery_set_selection.csv", "discovery_quality_audit_all_sessions.csv", "discovery_covariance_translation_results.csv", "discovery_exact_support_shuffle_null.csv.gz", "discovery_covariance_landscapes.csv.gz"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(figure_path)
    print(selected[["ecephys_session_id", "target_units", "heldout_gradient_r2_vs_constant", "median_sampling_fraction", "maximum_consecutive_tangential_ccf_step_um", "censored_fraction", "robust_rf_support_deg", "median_test_deviance"]].to_string(index=False))
    print(results.loc[results["target_subset"].eq("full")].to_string(index=False))


if __name__ == "__main__":
    main()
