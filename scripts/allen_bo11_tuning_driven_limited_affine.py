#!/usr/bin/env python3
"""Exploratory SF/TF-driven limited-affine stacking of Allen BO 1.1 sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from scripts.allen_frequency_preference_surfaces import polar_coordinates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06c_allen_rf_matching"
    / "simultaneous_v1_hva_session_maps"
    / "allen_bo11_simultaneous_v1_hva_surface_grid.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06c_allen_rf_matching"
    / "tuning_driven_limited_affine"
)
GROUPS = ("V1", "HVA pooled")
PREFERENCES = ("sf", "tf")
MAP_KEYS = tuple((group, preference) for group in GROUPS for preference in PREFERENCES)
CENTER_DEG = np.array([50.0, 10.0])
PARAMETER_NAMES = ("translation_azimuth_deg", "translation_elevation_deg", "rotation_deg", "log_scale_azimuth", "log_scale_elevation", "shear")
BOUNDS = ((-15.0, 15.0), (-15.0, 15.0), (-12.0, 12.0), (np.log(0.85), np.log(1.15)), (np.log(0.85), np.log(1.15)), (-0.12, 0.12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-grid", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--optimizer-generations", type=int, default=35)
    parser.add_argument("--optimizer-population", type=int, default=6)
    parser.add_argument("--minimum-shared-grid-points", type=int, default=50)
    parser.add_argument("--regularization-weight", type=float, default=0.025)
    parser.add_argument("--coverage-penalty-weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def affine_matrix(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx, ty, rotation_deg, log_sx, log_sy, shear = np.asarray(parameters, dtype=float)
    theta = np.deg2rad(rotation_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shape = np.array([[np.exp(log_sx), shear], [0.0, np.exp(log_sy)]])
    return rotation @ shape, np.array([tx, ty])


def normalized_parameter_penalty(parameters: np.ndarray) -> float:
    normalized = []
    for value, (lower, upper) in zip(parameters, BOUNDS):
        half_range = 0.5 * (upper - lower)
        normalized.append(value / half_range)
    return float(np.mean(np.square(normalized)))


def load_maps(path: Path) -> tuple[dict[tuple[int, str, str], dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    table = pd.read_csv(path, low_memory=False)
    table = table.loc[table["group"].isin(GROUPS) & table["map"].isin(PREFERENCES)].copy()
    az_grid = np.sort(table["azimuth_deg"].unique())
    el_grid = np.sort(table["elevation_deg"].unique())
    maps = {}
    for (session_id, group, preference), selected in table.groupby(
        ["ecephys_session_id", "group", "map"], observed=True
    ):
        selected = selected.sort_values(["elevation_deg", "azimuth_deg"])
        value = selected["estimate_log2"].to_numpy(float).reshape(len(el_grid), len(az_grid))
        evidence = selected["local_units"].to_numpy(float).reshape(len(el_grid), len(az_grid))
        supported = selected["supported"].to_numpy(bool).reshape(len(el_grid), len(az_grid))
        evidence = np.where(supported & np.isfinite(value), np.sqrt(np.maximum(evidence, 0.0)), 0.0)
        finite_value = np.where(np.isfinite(value), value, 0.0)
        maps[(int(session_id), str(group), str(preference))] = {
            "value": value,
            "evidence": evidence,
            "source_units": int(selected["source_units"].iloc[0]),
            "interpolate_evidence": RegularGridInterpolator(
                (el_grid, az_grid), evidence, bounds_error=False, fill_value=0.0
            ),
            "interpolate_numerator": RegularGridInterpolator(
                (el_grid, az_grid), finite_value * evidence, bounds_error=False, fill_value=0.0
            ),
        }
    return maps, az_grid, el_grid


def warp_map(
    source: dict[str, np.ndarray],
    parameters: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    matrix, translation = affine_matrix(parameters)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    target_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    inverse = np.linalg.inv(matrix)
    source_points = (target_points - CENTER_DEG - translation) @ inverse.T + CENTER_DEG
    query = np.column_stack([source_points[:, 1], source_points[:, 0]])
    warped_evidence = source["interpolate_evidence"](query).reshape(len(el_grid), len(az_grid))
    warped_numerator = source["interpolate_numerator"](query).reshape(len(el_grid), len(az_grid))
    warped_value = np.divide(
        warped_numerator,
        warped_evidence,
        out=np.full_like(warped_numerator, np.nan),
        where=warped_evidence > 1e-6,
    )
    return {"value": warped_value, "evidence": warped_evidence}


def warp_all(
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    parameters: dict[int, np.ndarray],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    return {
        key: warp_map(value, parameters[key[0]], az_grid, el_grid)
        for key, value in source_maps.items()
    }


def template_from_maps(
    warped_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    group: str,
    preference: str,
    *,
    exclude_session: int | None = None,
) -> dict[str, np.ndarray]:
    selected = [
        value
        for (session_id, local_group, local_preference), value in warped_maps.items()
        if local_group == group
        and local_preference == preference
        and session_id != exclude_session
    ]
    normalized_weights = []
    for item in selected:
        evidence = item["evidence"].copy()
        positive = evidence[evidence > 0]
        scale = np.median(positive) if len(positive) else 1.0
        normalized_weights.append(evidence / max(scale, 1e-9))
    denominator = np.sum(normalized_weights, axis=0)
    numerator = np.sum(
        [np.nan_to_num(item["value"]) * weight for item, weight in zip(selected, normalized_weights)],
        axis=0,
    )
    value = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return {"value": value, "evidence": denominator}


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)
    covariance = np.average((x - mean_x) * (y - mean_y), weights=weights)
    variance_x = np.average(np.square(x - mean_x), weights=weights)
    variance_y = np.average(np.square(y - mean_y), weights=weights)
    if variance_x <= 1e-12 or variance_y <= 1e-12:
        return np.nan
    return float(covariance / np.sqrt(variance_x * variance_y))


def map_agreement(
    session_map: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    minimum_points: int,
) -> dict[str, float]:
    valid = (
        np.isfinite(session_map["value"])
        & np.isfinite(template["value"])
        & (session_map["evidence"] > 0)
        & (template["evidence"] > 0)
    )
    count = int(valid.sum())
    source_count = int((session_map["evidence"] > 0).sum())
    if count < minimum_points:
        return {"correlation": np.nan, "rmse": np.nan, "shared_points": count, "coverage": count / max(source_count, 1)}
    weights = np.sqrt(session_map["evidence"][valid] * template["evidence"][valid])
    x = session_map["value"][valid]
    y = template["value"][valid]
    return {
        "correlation": weighted_correlation(x, y, weights),
        "rmse": float(np.sqrt(np.average(np.square(x - y), weights=weights))),
        "shared_points": count,
        "coverage": count / max(source_count, 1),
    }


def optimize_session(
    session_id: int,
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    templates: dict[tuple[str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    minimum_points: int,
    regularization_weight: float,
    coverage_penalty_weight: float,
    generations: int,
    population: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    def objective(parameters: np.ndarray) -> float:
        losses = []
        for group, preference in MAP_KEYS:
            source = source_maps[(session_id, group, preference)]
            warped = warp_map(source, parameters, az_grid, el_grid)
            agreement = map_agreement(warped, templates[(group, preference)], minimum_points)
            if not np.isfinite(agreement["correlation"]):
                losses.append(2.0)
                continue
            losses.append(
                1.0
                - agreement["correlation"]
                + coverage_penalty_weight * (1.0 - agreement["coverage"])
            )
        return float(np.mean(losses) + regularization_weight * normalized_parameter_penalty(parameters))

    result = differential_evolution(
        objective,
        BOUNDS,
        seed=seed,
        maxiter=generations,
        popsize=population,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-4,
    )
    return result.x, float(result.fun)


def fit_transforms(
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    iterations: int,
    minimum_points: int,
    regularization_weight: float,
    coverage_penalty_weight: float,
    generations: int,
    population: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    sessions = sorted({key[0] for key in source_maps})
    parameters = {session_id: np.zeros(6) for session_id in sessions}
    history = []
    for iteration in range(iterations):
        warped = warp_all(source_maps, parameters, az_grid, el_grid)
        updated = {}
        for session_index, session_id in enumerate(sessions):
            templates = {
                key: template_from_maps(warped, *key, exclude_session=session_id)
                for key in MAP_KEYS
            }
            fitted, loss = optimize_session(
                session_id,
                source_maps,
                templates,
                az_grid,
                el_grid,
                minimum_points=minimum_points,
                regularization_weight=regularization_weight,
                coverage_penalty_weight=coverage_penalty_weight,
                generations=generations,
                population=population,
                seed=20260811 + iteration * 100 + session_index,
            )
            updated[session_id] = fitted
            row = {"iteration": iteration + 1, "ecephys_session_id": session_id, "objective": loss}
            row.update(dict(zip(PARAMETER_NAMES, fitted)))
            history.append(row)
        parameters = updated
    return parameters, pd.DataFrame(history)


def evaluate_model(
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    parameters: dict[int, np.ndarray],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    minimum_points: int,
    model: str,
) -> pd.DataFrame:
    warped = warp_all(source_maps, parameters, az_grid, el_grid)
    rows = []
    for session_id in sorted(parameters):
        for group, preference in MAP_KEYS:
            template = template_from_maps(warped, group, preference, exclude_session=session_id)
            agreement = map_agreement(warped[(session_id, group, preference)], template, minimum_points)
            rows.append(
                {
                    "ecephys_session_id": session_id,
                    "group": group,
                    "preference": preference,
                    "model": model,
                    "weighted_correlation": agreement["correlation"],
                    "weighted_rmse_octaves": agreement["rmse"],
                    "shared_grid_points": agreement["shared_points"],
                    "coverage_fraction": agreement["coverage"],
                }
            )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    joined = raw.merge(
        aligned,
        on=["ecephys_session_id", "group", "preference"],
        suffixes=("_raw", "_aligned"),
        validate="one_to_one",
    )
    rows = []
    for (group, preference), selected in joined.groupby(["group", "preference"], observed=True):
        valid = selected.dropna(subset=["weighted_correlation_raw", "weighted_correlation_aligned"])
        rows.append(
            {
                "group": group,
                "preference": preference,
                "sessions": len(valid),
                "median_correlation_raw": valid["weighted_correlation_raw"].median(),
                "median_correlation_aligned": valid["weighted_correlation_aligned"].median(),
                "median_paired_correlation_change": np.median(valid["weighted_correlation_aligned"] - valid["weighted_correlation_raw"]),
                "median_rmse_raw_octaves": valid["weighted_rmse_octaves_raw"].median(),
                "median_rmse_aligned_octaves": valid["weighted_rmse_octaves_aligned"].median(),
                "median_paired_rmse_change_octaves": np.median(valid["weighted_rmse_octaves_aligned"] - valid["weighted_rmse_octaves_raw"]),
                "median_coverage_raw": valid["coverage_fraction_raw"].median(),
                "median_coverage_aligned": valid["coverage_fraction_aligned"].median(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_templates(
    maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    return {key: template_from_maps(maps, *key) for key in MAP_KEYS}


def render_results(
    raw_metrics: pd.DataFrame,
    aligned_metrics: pd.DataFrame,
    raw_templates: dict[tuple[str, str], dict[str, np.ndarray]],
    aligned_templates: dict[tuple[str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
) -> None:
    joined = raw_metrics.merge(
        aligned_metrics,
        on=["ecephys_session_id", "group", "preference"],
        suffixes=("_raw", "_aligned"),
    )
    fig = plt.figure(figsize=(16.0, 11.5))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.33, wspace=0.28)
    for index, (group, preference) in enumerate(MAP_KEYS):
        ax = fig.add_subplot(grid[0, index])
        selected = joined.loc[joined["group"].eq(group) & joined["preference"].eq(preference)].dropna(
            subset=["weighted_correlation_raw", "weighted_correlation_aligned"]
        )
        ax.scatter(selected["weighted_correlation_raw"], selected["weighted_correlation_aligned"], s=26, alpha=0.75)
        ax.plot([-1, 1], [-1, 1], linestyle="--", color="#777777", linewidth=1)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        delta = np.median(selected["weighted_correlation_aligned"] - selected["weighted_correlation_raw"])
        ax.set_title(f"{group} · {preference.upper()}\nmedian paired Δr={delta:+.3f}")
        ax.set_xlabel("raw correlation")
        if index == 0:
            ax.set_ylabel("tuning-fitted affine correlation")
        ax.grid(alpha=0.2)

    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(az_mesh, el_mesh)
    for column, (group, preference) in enumerate(MAP_KEYS):
        ax = fig.add_subplot(grid[1, column], projection="polar")
        raw = raw_templates[(group, preference)]["value"]
        aligned = aligned_templates[(group, preference)]["value"]
        difference = aligned - raw
        finite = difference[np.isfinite(difference)]
        limit = np.quantile(np.abs(finite), 0.98)
        artist = ax.pcolormesh(theta, radius, difference, shading="gouraud", cmap="coolwarm", norm=Normalize(-limit, limit))
        ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_ylim(0, 110); ax.set_rlabel_position(65)
        ax.set_thetamin(-70); ax.set_thetamax(70)
        ax.set_title(f"{group} · {preference.upper()}\naligned minus raw template", fontsize=10)
        fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.08, label="change (octaves)")
    fig.suptitle(
        "Exploratory tuning-driven limited-affine stacking of Allen BO 1.1 sessions\n"
        "same transform jointly fits V1/HVA SF/TF; density/evidence weighted; in-sample pilot",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_stacked_templates(
    raw_templates: dict[tuple[str, str], dict[str, np.ndarray]],
    aligned_templates: dict[tuple[str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
) -> None:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(az_mesh, el_mesh)
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 8.2), subplot_kw={"projection": "polar"})
    for column, key in enumerate(MAP_KEYS):
        group, preference = key
        raw = np.exp2(raw_templates[key]["value"])
        aligned = np.exp2(aligned_templates[key]["value"])
        finite = np.concatenate([raw[np.isfinite(raw)], aligned[np.isfinite(aligned)]])
        limits = np.quantile(finite, [0.02, 0.98])
        norm = Normalize(*limits)
        artist = None
        for row, (label, values) in enumerate((("Raw stack", raw), ("Tuning-fitted stack", aligned))):
            ax = axes[row, column]
            artist = ax.pcolormesh(
                theta,
                radius,
                values,
                shading="gouraud",
                cmap="viridis" if preference == "sf" else "plasma",
                norm=norm,
            )
            ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_ylim(0, 110); ax.set_rlabel_position(65)
            ax.set_thetamin(-70); ax.set_thetamax(70)
            ax.grid(color="#B5B5B5", linewidth=0.65)
            ax.set_title(f"{label}\n{group} · {preference.upper()}", fontsize=10)
        colorbar = fig.colorbar(artist, ax=axes[:, column].tolist(), fraction=0.025, pad=0.055, extend="both")
        colorbar.set_label("cycles/deg" if preference == "sf" else "Hz")
    fig.suptitle(
        "Allen BO 1.1 raw and tuning-fitted session stacks\n"
        "one shared limited affine per session jointly fits V1/HVA SF/TF",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.035, right=0.95, bottom=0.06, top=0.86, wspace=0.31, hspace=0.31)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, transform_table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Allen BO 1.1 tuning-driven limited-affine stacking pilot",
        "",
        "## Status: deliberately exploratory and in-sample",
        "",
        "One orientation-preserving affine transform per session was optimized jointly on",
        "V1 SF, V1 TF, pooled-HVA SF, and pooled-HVA TF maps. The four maps receive equal",
        "objective weight; grid comparisons are weighted by local unit-density evidence.",
        "This directly uses the outcomes whose agreement is reported and is therefore a",
        "visibility/feasibility pilot, not validation or an estimated gaze correction.",
        "",
        "Transform bounds: translation ±15°, rotation ±12°, independent scales 0.85–1.15,",
        "and shear ±0.12. Positive scales prohibit reflection.",
        "",
        "| Group | Preference | Sessions | Median r raw | Median r aligned | Median paired Δr | Median RMSE raw | Median RMSE aligned | Median paired ΔRMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.group} | {row.preference.upper()} | {int(row.sessions)} | "
            f"{row.median_correlation_raw:.3f} | {row.median_correlation_aligned:.3f} | "
            f"{row.median_paired_correlation_change:+.3f} | {row.median_rmse_raw_octaves:.3f} | "
            f"{row.median_rmse_aligned_octaves:.3f} | {row.median_paired_rmse_change_octaves:+.3f} |"
        )
    final = transform_table.loc[transform_table["iteration"].eq(transform_table["iteration"].max())]
    lines.extend(
        [
            "",
            "## Boundary behavior",
            "",
        ]
    )
    for parameter, (lower, upper) in zip(PARAMETER_NAMES, BOUNDS):
        tolerance = max(1e-4, 0.01 * (upper - lower))
        boundary = ((final[parameter] - lower).abs() <= tolerance) | ((final[parameter] - upper).abs() <= tolerance)
        lines.append(f"- `{parameter}`: {int(boundary.sum())}/{len(final)} sessions within 1% of a bound.")
    lines.extend(
        [
            "",
        "Large apparent gains or frequent boundary solutions indicate registration",
            "flexibility rather than biological validation. The aligned templates must not",
            "replace unaligned maps without an independent landmark or held-out replication.",
            "",
            "`Figure_allen_bo11_tuning_driven_stacked_templates.png` shows the raw and",
            "tuning-fitted aggregate maps on identical per-panel color scales.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_maps, az_grid, el_grid = load_maps(args.surface_grid.resolve())
    sessions = sorted({key[0] for key in source_maps})
    identity = {session_id: np.zeros(6) for session_id in sessions}
    fitted, history = fit_transforms(
        source_maps,
        az_grid,
        el_grid,
        iterations=args.iterations,
        minimum_points=args.minimum_shared_grid_points,
        regularization_weight=args.regularization_weight,
        coverage_penalty_weight=args.coverage_penalty_weight,
        generations=args.optimizer_generations,
        population=args.optimizer_population,
    )
    raw_metrics = evaluate_model(source_maps, identity, az_grid, el_grid, args.minimum_shared_grid_points, "raw")
    aligned_metrics = evaluate_model(source_maps, fitted, az_grid, el_grid, args.minimum_shared_grid_points, "tuning_fitted_affine")
    metrics = pd.concat([raw_metrics, aligned_metrics], ignore_index=True)
    summary = summarize(raw_metrics, aligned_metrics)
    raw_warped = warp_all(source_maps, identity, az_grid, el_grid)
    aligned_warped = warp_all(source_maps, fitted, az_grid, el_grid)
    raw_templates = aggregate_templates(raw_warped)
    aligned_templates = aggregate_templates(aligned_warped)
    transform_rows = []
    final_iteration = history["iteration"].max()
    for session_id, parameters in fitted.items():
        matrix, translation = affine_matrix(parameters)
        singular = np.linalg.svd(matrix, compute_uv=False)
        row = {"ecephys_session_id": session_id, "iteration": final_iteration, "determinant": np.linalg.det(matrix), "minimum_singular_value": singular.min(), "maximum_singular_value": singular.max()}
        row.update(dict(zip(PARAMETER_NAMES, parameters)))
        transform_rows.append(row)
    final_transforms = pd.DataFrame(transform_rows)
    full_history = history.merge(final_transforms[["ecephys_session_id", "determinant", "minimum_singular_value", "maximum_singular_value"]], on="ecephys_session_id", how="left")
    metrics.to_csv(output_dir / "session_map_agreement.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "agreement_summary.csv", index=False, float_format="%.6g")
    full_history.to_csv(output_dir / "limited_affine_transform_history.csv", index=False, float_format="%.6g")
    figure_path = output_dir / "Figure_allen_bo11_tuning_driven_limited_affine.png"
    render_results(raw_metrics, aligned_metrics, raw_templates, aligned_templates, az_grid, el_grid, figure_path)
    render_stacked_templates(
        raw_templates,
        aligned_templates,
        az_grid,
        el_grid,
        output_dir / "Figure_allen_bo11_tuning_driven_stacked_templates.png",
    )
    write_report(summary, history, output_dir / "ALLEN_BO11_TUNING_DRIVEN_LIMITED_AFFINE.md")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_tuning_driven_limited_affine",
        "status": "exploratory in-sample tuning-driven registration pilot",
        "input": {"path": str(args.surface_grid.resolve()), "sha256": sha256(args.surface_grid.resolve())},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "joint_maps": [f"{group}:{preference}" for group, preference in MAP_KEYS],
            "bounds": {name: list(bounds) for name, bounds in zip(PARAMETER_NAMES, BOUNDS)},
            "iterations": args.iterations,
            "optimizer_generations": args.optimizer_generations,
            "optimizer_population": args.optimizer_population,
            "minimum_shared_grid_points": args.minimum_shared_grid_points,
            "regularization_weight": args.regularization_weight,
            "coverage_penalty_weight": args.coverage_penalty_weight,
            "interpretation": "in-sample feasibility only; SF/TF drive and evaluate alignment",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen tuning-driven limited-affine pilot written to {output_dir}")


if __name__ == "__main__":
    main()
