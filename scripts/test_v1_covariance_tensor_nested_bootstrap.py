#!/usr/bin/env python3
"""Test whether the full residual RF covariance tensor resolves trace ambiguity.

This is a two-session, concrete-case diagnostic. Cell resampling is stratified by
physical probe block and local covariance is re-estimated in both the held-out
session and every session contributing to its population template. The fitted
cross-animal CCF-to-RF mean maps are held fixed.
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

from scripts.check_v1_cross_animal_mean_map_support import (
    DEFAULT_INPUT,
    DEFAULT_UNITS,
    RF_COLUMNS,
    load_population,
    make_block_table,
)
from scripts.check_v1_dispersion_physical_sampling import physical_blocks
from scripts.check_v1_dispersion_support_geometry import support_decomposition
from scripts.test_v1_rf_size_corroboration import (
    best_shift,
    huber_mean,
    nested_session_features,
    robust_scale,
    smooth_values,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_OUTPUT = CHECKPOINT / "covariance_tensor_nested_bootstrap"
CASES = {
    760345702: "trace point-localizing candidate",
    798911424: "trace annular-ambiguity candidate",
}
TENSOR_FEATURES = (
    "log2_conditional_residual_trace",
    "conditional_anisotropy_axis",
    "conditional_anisotropy_cross",
)
COMPONENTS = {
    "trace": (0,),
    "anisotropy": (1, 2),
    "full_tensor": (0, 1, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rf-neighborhood-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--surface-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--surface-step-deg", type=float, default=4.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--translation-step-deg", type=float, default=2.0)
    parser.add_argument("--physical-block-count", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def attach_tensor_features(
    table: pd.DataFrame,
    predicted_rf: np.ndarray,
    rf_bandwidth: float,
) -> pd.DataFrame:
    """Recompute anatomy-corrected local covariance and retain all 3 components."""
    result = table.reset_index(drop=True).copy()
    decomposition = support_decomposition(
        result["probe_vertical_position"].to_numpy(float),
        result[list(RF_COLUMNS)].to_numpy(float),
        np.asarray(predicted_rf, float),
        rf_bandwidth,
    )
    for column in decomposition:
        result[column] = decomposition[column].to_numpy()
    trace = np.maximum(result["residual_trace_deg2"].to_numpy(float), 1e-6)
    cxx = result["residual_cov_azimuth_deg2"].to_numpy(float)
    cyy = result["residual_cov_elevation_deg2"].to_numpy(float)
    cxy = result["residual_cov_azimuth_elevation_deg2"].to_numpy(float)
    result["log2_conditional_residual_trace"] = np.log2(trace)
    # Double-angle representation of ellipse anisotropy. Together with trace,
    # these three values reconstruct the symmetric 2x2 covariance matrix.
    result["conditional_anisotropy_axis"] = (cxx - cyy) / trace
    result["conditional_anisotropy_cross"] = 2.0 * cxy / trace
    return result


def block_stratified_bootstrap_indices(
    table: pd.DataFrame,
    block_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = physical_blocks(table["probe_vertical_position"], block_count)
    sampled = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members):
            sampled.append(rng.choice(members, size=len(members), replace=True))
    indices = np.concatenate(sampled)
    return indices[rng.permutation(len(indices))]


def tensor_surface(
    table: pd.DataFrame,
    grid_points: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    points = table[list(RF_COLUMNS)].to_numpy(float)
    return np.column_stack(
        [
            smooth_values(points, table[feature].to_numpy(float), grid_points, bandwidth)
            for feature in TENSOR_FEATURES
        ]
    )


def mean_template(surfaces: list[np.ndarray], minimum_sessions: int = 8) -> np.ndarray:
    stack = np.stack(surfaces)
    support = np.sum(np.isfinite(stack), axis=0)
    template = np.divide(
        np.nansum(stack, axis=0),
        support,
        out=np.full(support.shape, np.nan, dtype=float),
        where=support > 0,
    )
    template[support < minimum_sessions] = np.nan
    return template


def tensor_interpolators(template: np.ndarray, axis: np.ndarray):
    return [
        RegularGridInterpolator(
            (axis, axis), template[:, index].reshape(len(axis), len(axis)),
            bounds_error=False, fill_value=np.nan,
        )
        for index in range(template.shape[1])
    ]


def tensor_loss_grid(
    table: pd.DataFrame,
    template_interps,
    shifts: np.ndarray,
    scales: np.ndarray,
    component_indices: tuple[int, ...],
) -> np.ndarray:
    points = table[list(RF_COLUMNS)].to_numpy(float)
    observed = table[list(TENSOR_FEATURES)].to_numpy(float)[:, component_indices]
    local_scales = scales[list(component_indices)]
    losses = np.full(len(shifts), np.nan)
    finite_observed = np.isfinite(observed).all(axis=1)
    for shift_index, shift in enumerate(shifts):
        query = (points + shift)[:, [1, 0]]
        predicted = np.column_stack(
            [template_interps[index](query) for index in component_indices]
        )
        valid = finite_observed & np.isfinite(predicted).all(axis=1)
        if valid.sum() < 10:
            continue
        standardized = (observed[valid] - predicted[valid]) / local_scales
        losses[shift_index] = huber_mean(standardized.ravel()) + .75 * (1 - valid.mean())
    return losses


def relative_surface(ax, losses: np.ndarray, shift_axis: np.ndarray, title: str):
    image = losses.reshape(len(shift_axis), len(shift_axis))
    image = image - np.nanmin(image)
    upper = max(float(np.nanquantile(image[np.isfinite(image)], .80)), .01)
    artist = ax.imshow(
        image,
        origin="lower",
        extent=[shift_axis.min(), shift_axis.max(), shift_axis.min(), shift_axis.max()],
        cmap="magma_r",
        norm=Normalize(0, upper),
        aspect="equal",
    )
    ax.axhline(0, color="0.6", lw=.5)
    ax.axvline(0, color="0.6", lw=.5)
    ax.set(xlabel="azimuth shift", ylabel="elevation shift", title=title)
    plt.colorbar(artist, ax=ax, fraction=.046, pad=.03, label="loss above optimum")


def distance_from(optima: pd.DataFrame, reference: np.ndarray) -> np.ndarray:
    return np.hypot(
        optima["shift_azimuth_deg"].to_numpy(float) - reference[0],
        optima["shift_elevation_deg"].to_numpy(float) - reference[1],
    )


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

    axis = np.arange(-92.0, 92.0 + args.surface_step_deg, args.surface_step_deg)
    mesh_az, mesh_el = np.meshgrid(axis, axis)
    grid_points = np.column_stack([mesh_az.ravel(), mesh_el.ravel()])
    shift_axis = np.arange(
        -args.translation_bound_deg,
        args.translation_bound_deg + args.translation_step_deg,
        args.translation_step_deg,
    )
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])

    metric_rows = []
    optimum_rows = []
    bootstrap_rows = []
    landscape_rows = []
    payload = {}

    for target_id, role in CASES.items():
        print(f"Preparing target {target_id}: {role}", flush=True)
        base_features, predictions = nested_session_features(
            target_id, population, blocks, args.ridge,
            args.rf_neighborhood_bandwidth_deg,
        )
        features = {
            session_id: attach_tensor_features(
                local, predictions[session_id], args.rf_neighborhood_bandwidth_deg
            )
            for session_id, local in base_features.items()
        }
        scales = np.array(
            [
                robust_scale(
                    np.concatenate(
                        [local[feature].to_numpy(float) for local in features.values()]
                    ),
                    .05 if index == 0 else .10,
                )
                for index, feature in enumerate(TENSOR_FEATURES)
            ]
        )
        full_surfaces = {
            session_id: tensor_surface(local, grid_points, args.surface_bandwidth_deg)
            for session_id, local in features.items()
        }
        template = mean_template(
            [surface for session_id, surface in full_surfaces.items() if session_id != target_id]
        )
        interps = tensor_interpolators(template, axis)
        full_losses = {}
        full_shifts = {}
        target = features[target_id]
        for component, indices in COMPONENTS.items():
            losses = tensor_loss_grid(target, interps, shifts, scales, indices)
            optimum, minimum = best_shift(losses, shifts)
            full_losses[component] = losses
            full_shifts[component] = optimum
            optimum_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "estimate": "full_data",
                    "component": component,
                    "shift_azimuth_deg": optimum[0],
                    "shift_elevation_deg": optimum[1],
                    "minimum_loss": minimum,
                    "at_search_boundary": bool(np.any(np.abs(optimum) >= args.translation_bound_deg)),
                }
            )
            for shift_index, shift in enumerate(shifts):
                landscape_rows.append(
                    {
                        "ecephys_session_id": target_id,
                        "component": component,
                        "shift_azimuth_deg": shift[0],
                        "shift_elevation_deg": shift[1],
                        "loss": losses[shift_index],
                    }
                )

        root_seed = np.random.SeedSequence([20260816, target_id, 3])
        repeat_seeds = root_seed.spawn(args.bootstrap_repeats)
        for repeat, repeat_seed in enumerate(repeat_seeds):
            if repeat % 10 == 0:
                print(f"  nested bootstrap {repeat}/{args.bootstrap_repeats}", flush=True)
            rng = np.random.default_rng(repeat_seed)
            resampled_tables = {}
            resampled_surfaces = []
            for session_id, local in features.items():
                indices = block_stratified_bootstrap_indices(
                    local, args.physical_block_count, rng
                )
                boot = attach_tensor_features(
                    local.iloc[indices].reset_index(drop=True),
                    predictions[session_id][indices],
                    args.rf_neighborhood_bandwidth_deg,
                )
                resampled_tables[session_id] = boot
                if session_id != target_id:
                    resampled_surfaces.append(
                        tensor_surface(boot, grid_points, args.surface_bandwidth_deg)
                    )
            boot_template = mean_template(resampled_surfaces)
            boot_interps = tensor_interpolators(boot_template, axis)
            for component, indices in COMPONENTS.items():
                losses = tensor_loss_grid(
                    resampled_tables[target_id], boot_interps, shifts, scales, indices
                )
                optimum, minimum = best_shift(losses, shifts)
                bootstrap_rows.append(
                    {
                        "ecephys_session_id": target_id,
                        "bootstrap_repeat": repeat,
                        "component": component,
                        "shift_azimuth_deg": optimum[0],
                        "shift_elevation_deg": optimum[1],
                        "minimum_loss": minimum,
                        "at_search_boundary": bool(np.any(np.abs(optimum) >= args.translation_bound_deg)),
                    }
                )

        local_boot = pd.DataFrame(bootstrap_rows)
        local_boot = local_boot.loc[local_boot["ecephys_session_id"].eq(target_id)]
        for component in COMPONENTS:
            component_boot = local_boot.loc[local_boot["component"].eq(component)]
            distance = distance_from(component_boot, full_shifts[component])
            metric_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "role": role,
                    "component": component,
                    "full_shift_azimuth_deg": full_shifts[component][0],
                    "full_shift_elevation_deg": full_shifts[component][1],
                    "bootstrap_median_distance_deg": float(np.median(distance)),
                    "bootstrap_p90_distance_deg": float(np.quantile(distance, .90)),
                    "bootstrap_p95_distance_deg": float(np.quantile(distance, .95)),
                    "bootstrap_boundary_fraction": float(component_boot["at_search_boundary"].mean()),
                    "bootstrap_azimuth_iqr_deg": float(
                        component_boot["shift_azimuth_deg"].quantile(.75)
                        - component_boot["shift_azimuth_deg"].quantile(.25)
                    ),
                    "bootstrap_elevation_iqr_deg": float(
                        component_boot["shift_elevation_deg"].quantile(.75)
                        - component_boot["shift_elevation_deg"].quantile(.25)
                    ),
                }
            )
        payload[target_id] = {
            "target": target,
            "losses": full_losses,
            "shifts": full_shifts,
            "bootstrap": local_boot,
        }

    metrics = pd.DataFrame(metric_rows)
    optima = pd.DataFrame(optimum_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    landscapes = pd.DataFrame(landscape_rows)
    metrics.to_csv(output / "tensor_nested_bootstrap_metrics.csv", index=False)
    optima.to_csv(output / "tensor_full_data_optima.csv", index=False)
    bootstraps.to_csv(output / "tensor_nested_bootstrap_optima.csv.gz", index=False, compression="gzip")
    landscapes.to_csv(output / "tensor_full_data_landscapes.csv.gz", index=False, compression="gzip")

    fig, axes = plt.subplots(2, 6, figsize=(24, 8.3))
    for row, (target_id, role) in enumerate(CASES.items()):
        local = payload[target_id]
        target = local["target"]
        ax = axes[row, 0]
        magnitude = np.hypot(
            target["conditional_anisotropy_axis"],
            target["conditional_anisotropy_cross"],
        )
        angle = .5 * np.arctan2(
            target["conditional_anisotropy_cross"],
            target["conditional_anisotropy_axis"],
        )
        scatter = ax.scatter(
            target["rf_azimuth_deg"], target["rf_elevation_deg"],
            c=magnitude, cmap="viridis", vmin=0, vmax=1, s=24,
        )
        length = 3.5 * magnitude.to_numpy(float)
        ax.quiver(
            target["rf_azimuth_deg"], target["rf_elevation_deg"],
            length * np.cos(angle), length * np.sin(angle),
            angles="xy", scale_units="xy", scale=1, width=.003, color="black", alpha=.65,
        )
        ax.set(
            xlabel="RF azimuth", ylabel="RF elevation", aspect="equal",
            title=f"{target_id}: {role}\nresidual covariance anisotropy",
        )
        plt.colorbar(scatter, ax=ax, fraction=.046, pad=.03, label="normalized anisotropy")

        for column, component in enumerate(("trace", "anisotropy", "full_tensor"), start=1):
            shift = local["shifts"][component]
            relative_surface(
                axes[row, column], local["losses"][component], shift_axis,
                f"{component.replace('_', ' ')} objective\noptimum=({shift[0]:.0f}, {shift[1]:.0f}) deg",
            )
            axes[row, column].scatter(*shift, marker="*", s=100, color="cyan", edgecolor="black")

        for column, component in zip((4, 5), ("trace", "full_tensor")):
            ax = axes[row, column]
            local_boot = local["bootstrap"].loc[local["bootstrap"]["component"].eq(component)]
            ax.scatter(
                local_boot["shift_azimuth_deg"], local_boot["shift_elevation_deg"],
                s=18, alpha=.35, color="#4477aa",
            )
            shift = local["shifts"][component]
            ax.scatter(*shift, marker="*", s=120, color="white", edgecolor="black")
            metric = metrics.loc[
                metrics["ecephys_session_id"].eq(target_id)
                & metrics["component"].eq(component)
            ].iloc[0]
            ax.set(
                xlim=(-32, 32), ylim=(-32, 32), aspect="equal",
                xlabel="azimuth shift", ylabel="elevation shift",
                title=(
                    f"nested {component.replace('_', ' ')} bootstrap\n"
                    f"median={metric.bootstrap_median_distance_deg:.1f} deg; "
                    f"p90={metric.bootstrap_p90_distance_deg:.1f} deg"
                ),
            )

    fig.suptitle(
        "Does full anatomy-corrected RF covariance resolve trace-only localization ambiguity?\n"
        "Physical-block-stratified bootstrap re-estimates target and population-template covariance; CCF mean maps fixed",
        y=.998,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    figure_path = output / "Figure_v1_covariance_tensor_nested_bootstrap.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "two-case full-tensor nested covariance-bootstrap checkpoint",
        "cases": [{"ecephys_session_id": key, "role": value} for key, value in CASES.items()],
        "features": {
            "trace": "log2 residual covariance trace",
            "anisotropy_axis": "(Caz,az-Cel,el)/trace",
            "anisotropy_cross": "2*Caz,el/trace",
        },
        "bootstrap": (
            "cells sampled with replacement within six physical probe blocks; local covariance "
            "and every contributing session surface re-estimated each repeat"
        ),
        "conditional_on": "nested leave-one-animal-out CCF-to-RF mean-map fits and their predictions",
        "bootstrap_repeats": args.bootstrap_repeats,
        "surface_step_deg": args.surface_step_deg,
        "translation_step_deg": args.translation_step_deg,
        "outputs": [
            figure_path.name,
            "tensor_nested_bootstrap_metrics.csv",
            "tensor_full_data_optima.csv",
            "tensor_nested_bootstrap_optima.csv.gz",
            "tensor_full_data_landscapes.csv.gz",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(figure_path)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
