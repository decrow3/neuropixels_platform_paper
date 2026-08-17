#!/usr/bin/env python3
"""Plot Allen V1 RF size after excluding RF centers near stimulus-grid edges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_INPUT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_OUTPUT = AUDIT / "rf_size_interior_v1"
BO_COHORT = "Brain Observatory 1.1"
GRID_LIMITS = (10.0, 90.0, -30.0, 50.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-exclusion-deg", type=float, default=10.0)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--minimum-effective-sessions", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_population(table: pd.DataFrame) -> pd.DataFrame:
    selected = table.loc[table["cohort"].eq(BO_COHORT) & table["area"].eq("V1")].copy()
    selected = selected.dropna(subset=["azimuth_rf", "elevation_rf", "area_rf"])
    selected = selected.loc[selected["area_rf"].gt(0)].copy()
    az_min, az_max, el_min, el_max = GRID_LIMITS
    selected["distance_to_nearest_grid_edge_deg"] = np.minimum.reduce(
        [
            selected["azimuth_rf"] - az_min,
            az_max - selected["azimuth_rf"],
            selected["elevation_rf"] - el_min,
            el_max - selected["elevation_rf"],
        ]
    )
    selected["log2_rf_area_deg2"] = np.log2(selected["area_rf"])
    selected["rf_eccentricity_deg"] = np.hypot(selected["azimuth_rf"], selected["elevation_rf"])
    standardized = pd.Series(np.nan, index=selected.index, dtype=float)
    for _, indices in selected.groupby("ecephys_session_id", observed=True).groups.items():
        values = selected.loc[indices, "log2_rf_area_deg2"]
        center = values.median()
        scale = values.quantile(0.75) - values.quantile(0.25)
        if np.isfinite(scale) and scale > 1e-9:
            standardized.loc[indices] = (values - center) / scale
    selected["session_standardized_log2_rf_area"] = standardized.clip(-3, 3)
    return selected


def session_balanced_surface(
    population: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    bandwidth_deg: float,
    minimum_effective_sessions: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = population[["azimuth_rf", "elevation_rf"]].to_numpy(float)
    values = population["session_standardized_log2_rf_area"].to_numpy(float)
    sessions = population["ecephys_session_id"].to_numpy()
    unique_sessions, inverse = np.unique(sessions, return_inverse=True)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    estimates = np.full(len(grid), np.nan)
    effective = np.zeros(len(grid))
    for index, target in enumerate(grid):
        kernel = np.exp(-0.5 * np.sum(np.square((points - target) / bandwidth_deg), axis=1))
        session_weight = np.bincount(inverse, weights=kernel, minlength=len(unique_sessions))
        session_numerator = np.bincount(inverse, weights=kernel * values, minlength=len(unique_sessions))
        valid = session_weight > 1e-9
        local_session_values = session_numerator[valid] / session_weight[valid]
        local_session_weights = session_weight[valid]
        if local_session_weights.size:
            effective[index] = local_session_weights.sum() ** 2 / np.square(local_session_weights).sum()
        if effective[index] >= minimum_effective_sessions:
            estimates[index] = np.average(local_session_values, weights=local_session_weights)
    shape = (len(el_grid), len(az_grid))
    return estimates.reshape(shape), effective.reshape(shape)


def radial_summary(population: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0, 111, 10)
    local = population.copy()
    local["eccentricity_bin"] = pd.cut(local["rf_eccentricity_deg"], bins, include_lowest=True)
    session_bins = (
        local.groupby(["ecephys_session_id", "eccentricity_bin"], observed=True)
        ["session_standardized_log2_rf_area"].median().reset_index()
    )
    summary = (
        session_bins.groupby("eccentricity_bin", observed=True)["session_standardized_log2_rf_area"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), sessions="size")
        .reset_index()
    )
    summary["eccentricity_deg"] = summary["eccentricity_bin"].map(lambda x: x.mid).astype(float)
    return summary


def cutoff_sensitivity(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff in (0.0, 5.0, 10.0, 15.0, 20.0):
        selected = population.loc[population["distance_to_nearest_grid_edge_deg"].ge(cutoff)]
        rhos = {"eccentricity": [], "azimuth": [], "elevation": []}
        for _, group in selected.groupby("ecephys_session_id", observed=True):
            if len(group) < 10:
                continue
            for label, coordinate in (
                ("eccentricity", "rf_eccentricity_deg"),
                ("azimuth", "azimuth_rf"),
                ("elevation", "elevation_rf"),
            ):
                rho = spearmanr(group[coordinate], group["log2_rf_area_deg2"]).statistic
                if np.isfinite(rho):
                    rhos[label].append(float(rho))
        rows.append(
            {
                "edge_exclusion_deg": cutoff,
                "units": len(selected),
                "sessions": selected["ecephys_session_id"].nunique(),
                **{
                    f"{stat}_session_rho_size_vs_{label}": value
                    for label, values in rhos.items()
                    for stat, value in (
                        ("median", np.median(values)),
                        ("q25", np.quantile(values, .25)),
                        ("q75", np.quantile(values, .75)),
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def coordinate_summary(population: pd.DataFrame, coordinate: str, bins: np.ndarray) -> pd.DataFrame:
    local = population.copy()
    local["coordinate_bin"] = pd.cut(local[coordinate], bins, include_lowest=True)
    session_bins = (
        local.groupby(["ecephys_session_id", "coordinate_bin"], observed=True)
        ["session_standardized_log2_rf_area"].median().reset_index()
    )
    summary = (
        session_bins.groupby("coordinate_bin", observed=True)["session_standardized_log2_rf_area"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), sessions="size")
        .reset_index()
    )
    summary["coordinate_deg"] = summary["coordinate_bin"].map(lambda x: x.mid).astype(float)
    summary["coordinate"] = coordinate.removesuffix("_rf")
    return summary


def render_coordinate_figure(
    coordinate: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cutoff: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    specifications = (
        ("azimuth", "RF azimuth (deg)", "#4c78a8"),
        ("elevation", "RF elevation (deg)", "#e45756"),
    )
    for row, (label, axis_label, color) in enumerate(specifications):
        local = coordinate.loc[coordinate["coordinate"].eq(label) & coordinate["sessions"].ge(8)].sort_values("coordinate_deg")
        axes[row, 0].fill_between(local["coordinate_deg"], local["q25"], local["q75"], color=color, alpha=.2)
        axes[row, 0].plot(local["coordinate_deg"], local["median"], color=color, marker="o", linewidth=2)
        axes[row, 0].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[row, 0].set(xlabel=axis_label, ylabel="Session median standardized log₂ RF area", title=f"Interior V1 RF size versus {label}")
        median = f"median_session_rho_size_vs_{label}"
        q25 = f"q25_session_rho_size_vs_{label}"
        q75 = f"q75_session_rho_size_vs_{label}"
        axes[row, 1].fill_between(sensitivity["edge_exclusion_deg"], sensitivity[q25], sensitivity[q75], color=color, alpha=.2)
        axes[row, 1].plot(sensitivity["edge_exclusion_deg"], sensitivity[median], color=color, marker="o", linewidth=2)
        axes[row, 1].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[row, 1].axvline(cutoff, color="#a13d2d", linewidth=1)
        axes[row, 1].set(xlabel="Excluded distance from nearest edge (deg)", ylabel="Median within-session Spearman ρ", title=f"Size–{label} association across edge cutoffs")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.suptitle("Allen BO 1.1 interior V1 RF size by azimuth and elevation", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_figure(
    population: pd.DataFrame,
    interior: pd.DataFrame,
    surface: np.ndarray,
    effective: np.ndarray,
    radial: pd.DataFrame,
    sensitivity: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    cutoff: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4))
    excluded = population.loc[population["distance_to_nearest_grid_edge_deg"].lt(cutoff)]
    axes[0, 0].scatter(excluded["azimuth_rf"], excluded["elevation_rf"], s=7, color="#bbbbbb", alpha=.22, label="Excluded")
    axes[0, 0].scatter(interior["azimuth_rf"], interior["elevation_rf"], s=7, color="#3d6f8e", alpha=.16, label="Interior")
    rectangle = plt.Rectangle((10 + cutoff, -30 + cutoff), 80 - 2 * cutoff, 80 - 2 * cutoff, fill=False, color="#a13d2d", linewidth=1.5)
    axes[0, 0].add_patch(rectangle)
    axes[0, 0].set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title=f"V1 RF centers retained ≥{cutoff:g}° from every edge")
    axes[0, 0].legend(frameon=False)

    finite = surface[np.isfinite(surface)]
    limit = max(float(np.quantile(np.abs(finite), .98)), .1)
    artist = axes[0, 1].pcolormesh(az_grid, el_grid, surface, shading="gouraud", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[0, 1].contour(az_grid, el_grid, effective, levels=[5, 10, 20], colors="#333333", linewidths=.6, alpha=.55)
    axes[0, 1].set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title="Interior-only, session-balanced RF-size surface")
    cbar = fig.colorbar(artist, ax=axes[0, 1], fraction=.045, pad=.03)
    cbar.set_label("Within-session standardized log₂ RF area (IQR units)")

    supported = radial.loc[radial["sessions"].ge(8)].sort_values("eccentricity_deg")
    axes[1, 0].fill_between(supported["eccentricity_deg"], supported["q25"], supported["q75"], color="#4c78a8", alpha=.2)
    axes[1, 0].plot(supported["eccentricity_deg"], supported["median"], color="#4c78a8", marker="o", linewidth=2)
    axes[1, 0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="RF eccentricity from Allen (0°, 0°) (deg)", ylabel="Session median standardized log₂ RF area", title="Interior V1 size versus eccentricity")

    axes[1, 1].fill_between(sensitivity["edge_exclusion_deg"], sensitivity["q25_session_rho_size_vs_eccentricity"], sensitivity["q75_session_rho_size_vs_eccentricity"], color="#f58518", alpha=.2)
    axes[1, 1].plot(sensitivity["edge_exclusion_deg"], sensitivity["median_session_rho_size_vs_eccentricity"], color="#f58518", marker="o", linewidth=2)
    axes[1, 1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1, 1].axvline(cutoff, color="#a13d2d", linewidth=1)
    axes[1, 1].set(xlabel="Excluded distance from nearest edge (deg)", ylabel="Median within-session Spearman ρ", title="Size–eccentricity result across edge cutoffs")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.suptitle("Allen BO 1.1 V1 RF size away from RF-stimulus boundaries", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    population = prepare_population(pd.read_csv(args.input.resolve(), low_memory=False))
    interior = population.loc[population["distance_to_nearest_grid_edge_deg"].ge(args.edge_exclusion_deg)].copy()
    az_grid = np.linspace(10 + args.edge_exclusion_deg, 90 - args.edge_exclusion_deg, 31)
    el_grid = np.linspace(-30 + args.edge_exclusion_deg, 50 - args.edge_exclusion_deg, 31)
    surface, effective = session_balanced_surface(interior, az_grid, el_grid, args.bandwidth_deg, args.minimum_effective_sessions)
    radial = radial_summary(interior)
    sensitivity = cutoff_sensitivity(population)
    coordinate = pd.concat(
        [
            coordinate_summary(interior, "azimuth_rf", np.arange(30, 71, 5)),
            coordinate_summary(interior, "elevation_rf", np.arange(-10, 31, 5)),
        ],
        ignore_index=True,
    )
    interior.to_csv(output_dir / "allen_bo11_v1_rf_size_interior_units.csv", index=False, float_format="%.6g")
    radial.to_csv(output_dir / "allen_bo11_v1_rf_size_interior_radial.csv", index=False, float_format="%.6g")
    sensitivity.to_csv(output_dir / "allen_bo11_v1_rf_size_edge_cutoff_sensitivity.csv", index=False, float_format="%.6g")
    coordinate.to_csv(output_dir / "allen_bo11_v1_rf_size_interior_coordinates.csv", index=False, float_format="%.6g")
    figure_path = output_dir / "Figure_allen_bo11_v1_rf_size_interior.png"
    render_figure(population, interior, surface, effective, radial, sensitivity, az_grid, el_grid, args.edge_exclusion_deg, figure_path)
    coordinate_figure = output_dir / "Figure_allen_bo11_v1_rf_size_by_azimuth_elevation.png"
    render_coordinate_figure(coordinate, sensitivity, args.edge_exclusion_deg, coordinate_figure)
    selected = sensitivity.loc[sensitivity["edge_exclusion_deg"].eq(args.edge_exclusion_deg)].iloc[0]
    report = [
        "# Allen BO 1.1 interior V1 RF size",
        "",
        f"RF centers within {args.edge_exclusion_deg:g}° of any released RF-grid boundary were excluded.",
        f"The retained population contains **{len(interior)}/{len(population)} units** from **{interior['ecephys_session_id'].nunique()} sessions**.",
        "RF area is log2 transformed and median/IQR standardized within session before spatial aggregation.",
        "The map first estimates each session locally and then combines sessions, preventing unit-rich sessions from dominating.",
        "",
        f"At this cutoff, the median within-session association between RF size and Allen-origin eccentricity is **rho = {selected.median_session_rho_size_vs_eccentricity:+.3f}**.",
        f"Separately, the median associations are **rho = {selected.median_session_rho_size_vs_azimuth:+.3f}** for azimuth and **rho = {selected.median_session_rho_size_vs_elevation:+.3f}** for elevation.",
        "The cutoff-sensitivity panel shows whether that relationship persists as progressively more boundary-adjacent RF centers are removed.",
        "This removes center estimates near the sampled grid boundary; it cannot guarantee that a large RF centered inside the boundary was fully contained by the stimulus support.",
    ]
    (output_dir / "ALLEN_BO11_V1_RF_SIZE_INTERIOR.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output_dir.iterdir() if p.is_file() and p.name != "run_manifest.json"}
    manifest = {
        "checkpoint": "06c_allen_bo11_v1_rf_size_interior",
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input.resolve())},
        "parameters": {"cohort": BO_COHORT, "area": "V1", "grid_limits_deg": GRID_LIMITS, "edge_exclusion_deg": args.edge_exclusion_deg, "bandwidth_deg": args.bandwidth_deg, "minimum_effective_sessions": args.minimum_effective_sessions},
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen interior V1 RF-size diagnostic written to {output_dir}")


if __name__ == "__main__":
    main()
