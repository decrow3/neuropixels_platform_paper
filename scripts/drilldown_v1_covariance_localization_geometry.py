#!/usr/bin/env python3
"""Decompose bowl versus annular V1 covariance-localization objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import ndimage
from scipy.stats import spearmanr

from scripts.check_v1_cross_animal_mean_map_support import (
    DEFAULT_INPUT,
    DEFAULT_UNITS,
    RF_COLUMNS,
    load_population,
    make_block_table,
)
from scripts.check_v1_dispersion_physical_sampling import physical_blocks
from scripts.test_v1_rf_size_corroboration import (
    best_shift,
    build_scatter_surfaces,
    interpolator,
    mean_template,
    nested_session_features,
    robust_scale,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_OUTPUT = CHECKPOINT / "covariance_localization_geometry_drilldown"
CASES = {
    760345702: "bowl-like robust candidate",
    798911424: "annular regional-ambiguity case",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rf-neighborhood-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--surface-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--translation-step-deg", type=float, default=2.0)
    parser.add_argument("--physical-block-count", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def huber_mean(values: np.ndarray) -> float:
    absolute = np.abs(values)
    return float(np.mean(np.where(absolute <= 1, .5 * values**2, absolute - .5)))


def loss_components(
    points: np.ndarray,
    observed: np.ndarray,
    template_interp,
    shifts: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mismatch = np.full(len(shifts), np.nan)
    coverage_penalty = np.full(len(shifts), np.nan)
    coverage = np.zeros(len(shifts))
    finite_observed = np.isfinite(observed)
    for index, shift in enumerate(shifts):
        predicted = template_interp((points + shift)[:, [1, 0]])
        valid = finite_observed & np.isfinite(predicted)
        coverage[index] = valid.mean()
        if valid.sum() < 10:
            continue
        mismatch[index] = huber_mean((observed[valid] - predicted[valid]) / scale)
        coverage_penalty[index] = .75 * (1 - coverage[index])
    return mismatch + coverage_penalty, mismatch, coverage_penalty, coverage


def local_hessian(
    losses: np.ndarray, shifts: np.ndarray, optimum: np.ndarray, radius: float = 10.0
) -> tuple[np.ndarray, np.ndarray, int]:
    delta = shifts - optimum
    selected = np.isfinite(losses) & (np.sqrt(np.sum(delta**2, axis=1)) <= radius)
    x, y = delta[selected, 0], delta[selected, 1]
    design = np.column_stack([np.ones(selected.sum()), x, y, .5 * x**2, x * y, .5 * y**2])
    coefficient, *_ = np.linalg.lstsq(design, losses[selected], rcond=None)
    hessian = np.array([[coefficient[3], coefficient[4]], [coefficient[4], coefficient[5]]])
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    return eigenvalues, eigenvectors, int(selected.sum())


def low_loss_topology(
    losses: np.ndarray, grid_size: int, threshold_delta: float = .05
) -> dict[str, float]:
    image = losses.reshape(grid_size, grid_size)
    mask = np.isfinite(image) & (image <= np.nanmin(image) + threshold_delta)
    labels, components = ndimage.label(mask, structure=np.ones((3, 3), int))
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    hole_labels, hole_count = ndimage.label(holes, structure=np.ones((3, 3), int))
    component_sizes = np.bincount(labels.ravel())[1:] if components else np.array([])
    hole_sizes = np.bincount(hole_labels.ravel())[1:] if hole_count else np.array([])
    return {
        "low_loss_components": int(components),
        "largest_low_loss_component_points": int(component_sizes.max()) if len(component_sizes) else 0,
        "enclosed_holes": int(hole_count),
        "largest_hole_points": int(hole_sizes.max()) if len(hole_sizes) else 0,
        "low_loss_points": int(mask.sum()),
    }


def circle_fit(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    target = -(points[:, 0] ** 2 + points[:, 1] ** 2)
    coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = -coefficient[:2] / 2
    radius2 = np.sum(center**2) - coefficient[2]
    radius = float(np.sqrt(max(radius2, 0)))
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return center, radius, rmse


def radial_profile(
    losses: np.ndarray, shifts: np.ndarray, center: np.ndarray, bin_width: float = 3.0
) -> pd.DataFrame:
    radius = np.sqrt(np.sum((shifts - center) ** 2, axis=1))
    finite = np.isfinite(losses)
    bins = np.arange(0, np.nanmax(radius[finite]) + bin_width, bin_width)
    labels = np.digitize(radius, bins) - 1
    rows = []
    for index in range(len(bins) - 1):
        selected = finite & (labels == index)
        if selected.sum() < 3:
            continue
        rows.append(
            {
                "radius_deg": float(np.mean(radius[selected])),
                "mean_relative_loss": float(np.mean(losses[selected] - np.nanmin(losses))),
                "minimum_relative_loss": float(np.min(losses[selected] - np.nanmin(losses))),
                "grid_points": int(selected.sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_optima(
    points: np.ndarray,
    values: np.ndarray,
    blocks: np.ndarray,
    template_interp,
    shifts: np.ndarray,
    scale: float,
    repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    n = len(points)
    unique_blocks = np.unique(blocks[blocks >= 0])
    for repeat in range(repeats):
        indices = rng.integers(0, n, n)
        total, *_ = loss_components(points[indices], values[indices], template_interp, shifts, scale)
        shift, minimum = best_shift(total, shifts)
        rows.append(
            {
                "resample_type": "cell bootstrap",
                "repeat": repeat,
                "shift_azimuth_deg": shift[0],
                "shift_elevation_deg": shift[1],
                "minimum_loss": minimum,
            }
        )
    for omitted in unique_blocks:
        indices = np.flatnonzero(blocks != omitted)
        total, *_ = loss_components(points[indices], values[indices], template_interp, shifts, scale)
        shift, minimum = best_shift(total, shifts)
        rows.append(
            {
                "resample_type": "leave-one-physical-block-out",
                "repeat": int(omitted),
                "shift_azimuth_deg": shift[0],
                "shift_elevation_deg": shift[1],
                "minimum_loss": minimum,
            }
        )
    return pd.DataFrame(rows)


def plot_surface(ax, values, shift_axis, title, cmap="magma_r", diverging=False):
    image = values.reshape(len(shift_axis), len(shift_axis))
    if diverging:
        limit = max(float(np.nanquantile(np.abs(image[np.isfinite(image)]), .98)), 1e-6)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    else:
        relative = image - np.nanmin(image)
        upper = max(float(np.nanquantile(relative[np.isfinite(relative)], .8)), .01)
        image = relative
        norm = Normalize(0, upper)
    artist = ax.imshow(
        image,
        origin="lower",
        extent=[shift_axis.min(), shift_axis.max(), shift_axis.min(), shift_axis.max()],
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.axhline(0, color="0.6", lw=.5)
    ax.axvline(0, color="0.6", lw=.5)
    ax.set(xlabel="azimuth shift", ylabel="elevation shift", title=title)
    plt.colorbar(artist, ax=ax, fraction=.046, pad=.03)
    return artist


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

    axis = np.arange(-90.0, 90.0 + args.translation_step_deg, args.translation_step_deg)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    shift_axis = np.arange(-args.translation_bound_deg, args.translation_bound_deg + args.translation_step_deg, args.translation_step_deg)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    metrics_rows = []
    landscape_rows = []
    radial_rows = []
    bootstrap_frames = []
    payload = {}
    for target_id, role in CASES.items():
        print(f"target {target_id}: {role}")
        # Keep inference streams independent so changing the number of null
        # permutations cannot silently change the bootstrap sample.
        null_rng = np.random.default_rng(np.random.SeedSequence([20260816, target_id, 1]))
        bootstrap_rng = np.random.default_rng(np.random.SeedSequence([20260816, target_id, 2]))
        features, _ = nested_session_features(
            target_id, population, blocks, args.ridge, args.rf_neighborhood_bandwidth_deg
        )
        scale = robust_scale(
            np.concatenate([local["log2_conditional_residual_trace"].to_numpy(float) for local in features.values()]),
            .05,
        )
        surfaces = build_scatter_surfaces(features, grid_points, args.surface_bandwidth_deg)
        template = mean_template(surfaces, {target_id})
        template_interp = interpolator(template, axis)
        target = features[target_id].reset_index(drop=True)
        points = target[list(RF_COLUMNS)].to_numpy(float)
        values = target["log2_conditional_residual_trace"].to_numpy(float)
        total, mismatch, coverage_penalty, coverage = loss_components(
            points, values, template_interp, shifts, scale
        )
        total_shift, total_minimum = best_shift(total, shifts)
        mismatch_shift, mismatch_minimum = best_shift(mismatch, shifts)
        coverage_shift, _ = best_shift(coverage_penalty, shifts)
        eigenvalues, eigenvectors, hessian_points = local_hessian(total, shifts, total_shift)
        topology = low_loss_topology(total, len(shift_axis))

        low_threshold = np.nanquantile(total[np.isfinite(total)], .2)
        low_points = shifts[np.isfinite(total) & (total <= low_threshold)]
        circle_center, circle_radius, circle_rmse = circle_fit(low_points)
        profile = radial_profile(total, shifts, circle_center)
        profile["ecephys_session_id"] = target_id
        radial_rows.append(profile)

        null_mismatch = []
        null_total_minima = []
        for repeat in range(args.null_repeats):
            shuffled = null_rng.permutation(values)
            null_total, null_feature, *_ = loss_components(
                points, shuffled, template_interp, shifts, scale
            )
            null_mismatch.append(null_feature)
            null_total_minima.append(float(np.nanmin(null_total)))
        null_mismatch_mean = np.nanmean(np.stack(null_mismatch), axis=0)
        descriptor_advantage = null_mismatch_mean - mismatch

        block_label = physical_blocks(target["probe_vertical_position"], args.physical_block_count)
        boot = bootstrap_optima(
            points, values, block_label, template_interp, shifts, scale,
            args.bootstrap_repeats, bootstrap_rng,
        )
        boot["ecephys_session_id"] = target_id
        bootstrap_frames.append(boot)
        cell_boot = boot.loc[boot["resample_type"].eq("cell bootstrap")]
        block_jack = boot.loc[boot["resample_type"].eq("leave-one-physical-block-out")]
        cell_distance = np.sqrt(
            (cell_boot["shift_azimuth_deg"] - total_shift[0]) ** 2
            + (cell_boot["shift_elevation_deg"] - total_shift[1]) ** 2
        )
        block_distance = np.sqrt(
            (block_jack["shift_azimuth_deg"] - total_shift[0]) ** 2
            + (block_jack["shift_elevation_deg"] - total_shift[1]) ** 2
        )
        finite_loss = np.isfinite(total) & np.isfinite(coverage_penalty) & np.isfinite(mismatch)
        metrics_rows.append(
            {
                "ecephys_session_id": target_id,
                "role": role,
                "units": len(target),
                "total_shift_azimuth_deg": total_shift[0],
                "total_shift_elevation_deg": total_shift[1],
                "total_minimum_loss": total_minimum,
                "mismatch_only_shift_azimuth_deg": mismatch_shift[0],
                "mismatch_only_shift_elevation_deg": mismatch_shift[1],
                "total_vs_mismatch_optimum_distance_deg": float(np.linalg.norm(total_shift - mismatch_shift)),
                "coverage_only_shift_azimuth_deg": coverage_shift[0],
                "coverage_only_shift_elevation_deg": coverage_shift[1],
                "total_vs_coverage_optimum_distance_deg": float(np.linalg.norm(total_shift - coverage_shift)),
                "coverage_at_total_optimum": float(coverage[int(np.nanargmin(total))]),
                "total_vs_mismatch_surface_rho": float(spearmanr(total[finite_loss], mismatch[finite_loss]).statistic),
                "total_vs_coverage_surface_rho": float(spearmanr(total[finite_loss], coverage_penalty[finite_loss]).statistic),
                "hessian_eigenvalue_small": float(eigenvalues[0]),
                "hessian_eigenvalue_large": float(eigenvalues[1]),
                "hessian_anisotropy_ratio": float(eigenvalues[1] / max(eigenvalues[0], 1e-12)),
                "hessian_fit_grid_points": hessian_points,
                "circle_center_azimuth_deg": circle_center[0],
                "circle_center_elevation_deg": circle_center[1],
                "low_loss_circle_radius_deg": circle_radius,
                "low_loss_circle_rmse_deg": circle_rmse,
                "cell_bootstrap_median_distance_deg": float(np.median(cell_distance)),
                "cell_bootstrap_p90_distance_deg": float(np.quantile(cell_distance, .9)),
                "block_jackknife_max_distance_deg": float(np.max(block_distance)),
                "exact_support_shuffle_fit_p": float(np.mean(np.asarray(null_total_minima) <= total_minimum)),
                **topology,
            }
        )
        for index, shift in enumerate(shifts):
            landscape_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "shift_azimuth_deg": shift[0],
                    "shift_elevation_deg": shift[1],
                    "total_loss": total[index],
                    "feature_mismatch": mismatch[index],
                    "coverage_penalty": coverage_penalty[index],
                    "coverage_fraction": coverage[index],
                    "mean_shuffle_feature_mismatch": null_mismatch_mean[index],
                    "descriptor_advantage_over_shuffle": descriptor_advantage[index],
                }
            )
        payload[target_id] = {
            "target": target,
            "total": total,
            "mismatch": mismatch,
            "coverage_penalty": coverage_penalty,
            "descriptor_advantage": descriptor_advantage,
            "total_shift": total_shift,
            "mismatch_shift": mismatch_shift,
            "coverage_shift": coverage_shift,
            "circle_center": circle_center,
            "circle_radius": circle_radius,
            "profile": profile,
            "bootstrap": boot,
        }

    metrics = pd.DataFrame(metrics_rows)
    landscapes = pd.DataFrame(landscape_rows)
    radial = pd.concat(radial_rows, ignore_index=True)
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True)
    metrics.to_csv(output / "localization_geometry_metrics.csv", index=False)
    landscapes.to_csv(output / "localization_loss_components.csv.gz", index=False, compression="gzip")
    radial.to_csv(output / "localization_radial_profiles.csv", index=False)
    bootstraps.to_csv(output / "localization_conditional_resamples.csv.gz", index=False, compression="gzip")

    fig, axes = plt.subplots(2, 7, figsize=(28, 8.5))
    trace_limits = np.nanquantile(
        np.concatenate([payload[value]["target"]["log2_conditional_residual_trace"].to_numpy(float) for value in CASES]),
        [.02, .98],
    )
    for row_index, (target_id, role) in enumerate(CASES.items()):
        local = payload[target_id]
        metric = metrics.loc[metrics["ecephys_session_id"].eq(target_id)].iloc[0]
        target = local["target"]
        ax = axes[row_index, 0]
        scatter = ax.scatter(
            target["rf_azimuth_deg"], target["rf_elevation_deg"],
            c=target["log2_conditional_residual_trace"], cmap="cividis",
            norm=Normalize(*trace_limits), s=28,
        )
        ax.set(xlabel="RF azimuth", ylabel="RF elevation", title=f"{target_id}: {role}\nconditional scatter field", aspect="equal")
        fig.colorbar(scatter, ax=ax, label="log₂ residual trace")

        plot_surface(axes[row_index, 1], local["total"], shift_axis, "Total objective")
        axes[row_index, 1].scatter(*local["total_shift"], marker="*", s=90, color="cyan", edgecolor="black")
        plot_surface(axes[row_index, 2], local["mismatch"], shift_axis, f"Feature mismatch only\noptimum Δ={metric.total_vs_mismatch_optimum_distance_deg:.1f}°")
        axes[row_index, 2].scatter(*local["mismatch_shift"], marker="*", s=90, color="cyan", edgecolor="black")
        plot_surface(axes[row_index, 3], local["coverage_penalty"], shift_axis, f"Coverage penalty only\nρ(total)={metric.total_vs_coverage_surface_rho:.2f}")
        axes[row_index, 3].scatter(*local["coverage_shift"], marker="*", s=90, color="lime", edgecolor="black")
        plot_surface(axes[row_index, 4], local["descriptor_advantage"], shift_axis, "Real descriptor advantage over shuffle", cmap="coolwarm", diverging=True)

        ax = axes[row_index, 5]
        profile = local["profile"]
        ax.plot(profile["radius_deg"], profile["mean_relative_loss"], marker="o", ms=3, label="mean")
        ax.plot(profile["radius_deg"], profile["minimum_relative_loss"], marker=".", label="minimum")
        ax.axvline(local["circle_radius"], color="#d1495b", ls="--", label="low-loss circle fit")
        ax.set(xlabel="radius from fitted low-loss center (deg)", ylabel="loss above optimum", title=f"Radial geometry\nr={metric.low_loss_circle_radius_deg:.1f}°, circle RMSE={metric.low_loss_circle_rmse_deg:.1f}°")
        ax.legend(frameon=False, fontsize=7)

        ax = axes[row_index, 6]
        boot = local["bootstrap"]
        cell = boot.loc[boot["resample_type"].eq("cell bootstrap")]
        block = boot.loc[boot["resample_type"].eq("leave-one-physical-block-out")]
        ax.scatter(cell["shift_azimuth_deg"], cell["shift_elevation_deg"], s=14, alpha=.28, color="#4477aa", label="cell bootstrap")
        ax.scatter(block["shift_azimuth_deg"], block["shift_elevation_deg"], s=50, marker="D", color="#ee7733", edgecolor="black", linewidth=.5, label="leave-one-block-out")
        ax.scatter(*local["total_shift"], marker="*", s=110, color="white", edgecolor="black", label="full")
        ax.set(xlim=(-32, 32), ylim=(-32, 32), xlabel="azimuth shift", ylabel="elevation shift", title=f"Conditional uncertainty\ncell p90={metric.cell_bootstrap_p90_distance_deg:.1f}°; block max={metric.block_jackknife_max_distance_deg:.1f}°", aspect="equal")
        ax.legend(frameon=False, fontsize=6, loc="upper left")

    fig.suptitle(
        "What makes a covariance objective localizing? Total loss = feature mismatch + coverage penalty\n"
        "Resampling is conditional on the full-session covariance descriptor; it does not re-estimate RF covariance",
        y=.998,
    )
    fig.tight_layout(rect=(0, 0, 1, .975))
    figure_path = output / "Figure_v1_covariance_localization_geometry.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "two-case mechanism drill-down",
        "cases": [{"ecephys_session_id": key, "role": value} for key, value in CASES.items()],
        "objective": "Huber feature mismatch plus 0.75*(1-coverage)",
        "topology_threshold": "minimum total loss + 0.05",
        "curvature": "quadratic fit within 10 degrees of grid optimum",
        "circle_fit": "algebraic circle fit to lowest 20% of finite total-loss grid points",
        "resampling_limit": "cell bootstrap and leave-one-block-out reuse full-session conditional-scatter descriptors; they quantify matching uncertainty, not covariance-estimation uncertainty",
        "randomization": "independent deterministic random streams by session and inference type",
        "null_repeats": args.null_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "outputs": [figure_path.name, "localization_geometry_metrics.csv", "localization_loss_components.csv.gz", "localization_radial_profiles.csv", "localization_conditional_resamples.csv.gz"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(figure_path)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
