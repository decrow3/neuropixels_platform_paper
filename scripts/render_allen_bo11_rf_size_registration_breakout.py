#!/usr/bin/env python3
"""Break out Allen interior V1 RF-size registration and its tuning maps."""

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

from scripts.allen_bo11_tuning_driven_limited_affine import (
    MAP_KEYS,
    PARAMETER_NAMES,
    aggregate_templates,
    load_maps,
    polar_coordinates,
    template_from_maps,
    warp_all,
)
from scripts.allen_bo11_tuning_weighted_session_surfaces import weighted_gaussian_surface
from scripts.render_allen_bo11_registration_comparison import (
    DEFAULT_CCF_TRANSFORMS,
    DEFAULT_RF_SIZE_TRANSFORMS,
    DEFAULT_SURFACE_GRID,
    load_ccf_parameters,
    load_rf_size_parameters,
)
from scripts.render_allen_bo11_v1_rf_size_interior import DEFAULT_INPUT, prepare_population


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_OUTPUT = AUDIT / "v1_rf_size_registration_breakout"
EDGE_EXCLUSION_DEG = 20.0
BANDWIDTH_DEG = 8.0
DISPLAY_AZ_LIMITS = (0.0, 100.0)
DISPLAY_EL_LIMITS = (-40.0, 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tuning-grid", type=Path, default=DEFAULT_SURFACE_GRID)
    parser.add_argument("--transforms", type=Path, default=DEFAULT_RF_SIZE_TRANSFORMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-exclusion-deg", type=float, default=EDGE_EXCLUSION_DEG)
    parser.add_argument("--minimum-effective-local-units", type=float, default=5.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_absolute_size_maps(
    population: pd.DataFrame,
    sessions: list[int],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    minimum_effective_local_units: float = 5.0,
) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    targets = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = {}
    for session_id in sessions:
        selected = population.loc[population["ecephys_session_id"].eq(session_id)].dropna(
            subset=["azimuth_rf", "elevation_rf", "log2_rf_area_deg2"]
        )
        surface = weighted_gaussian_surface(
            selected[["azimuth_rf", "elevation_rf"]].to_numpy(float),
            selected["log2_rf_area_deg2"].to_numpy(float),
            np.ones(len(selected)),
            targets,
            bandwidth_deg=BANDWIDTH_DEG,
            minimum_effective_local_units=minimum_effective_local_units,
        )
        value = surface["estimate_log2"].reshape(len(el_grid), len(az_grid))
        effective = surface["effective_local_units"].reshape(len(el_grid), len(az_grid))
        supported = surface["supported"].reshape(len(el_grid), len(az_grid))
        evidence = np.where(supported & np.isfinite(value), np.sqrt(np.maximum(effective, 0)), 0.0)
        finite = np.where(np.isfinite(value), value, 0.0)
        maps[(session_id, "V1", "rf_size_absolute")] = {
            "value": value,
            "evidence": evidence,
            "source_units": len(selected),
            "interpolate_evidence": RegularGridInterpolator(
                (el_grid, az_grid), evidence, bounds_error=False, fill_value=0.0
            ),
            "interpolate_numerator": RegularGridInterpolator(
                (el_grid, az_grid), finite * evidence, bounds_error=False, fill_value=0.0
            ),
        }
    return maps


def render_tiling(
    ax: plt.Axes,
    template: dict[str, np.ndarray],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    title: str,
) -> None:
    values = template["value"]
    evidence = template["evidence"]
    artist = ax.pcolormesh(
        az_grid,
        el_grid,
        np.exp2(values),
        shading="gouraud",
        cmap="YlGnBu",
        norm=Normalize(250, 750),
        alpha=0.72,
    )
    ax.contour(az_grid, el_grid, evidence, levels=3, colors="#333333", linewidths=.55, alpha=.45)
    ax.set(
        xlim=DISPLAY_AZ_LIMITS,
        ylim=DISPLAY_EL_LIMITS,
        aspect="equal",
        xlabel="RF azimuth (deg)",
        ylabel="RF elevation (deg)",
        title=title,
    )
    ax.grid(alpha=.16)
    return artist


def render_session_grid(
    figure: plt.Figure,
    specification,
    maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    sessions: list[int],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    title: str,
    parameters: dict[int, np.ndarray] | None = None,
) -> None:
    columns = 6
    rows = int(np.ceil(len(sessions) / columns))
    axes = specification.subgridspec(rows, columns, wspace=.08, hspace=.28)
    for index, session_id in enumerate(sessions):
        row, column = divmod(index, columns)
        ax = figure.add_subplot(axes[row, column])
        source = maps[(session_id, "V1", "rf_size_absolute")]
        ax.pcolormesh(
            az_grid,
            el_grid,
            np.exp2(source["value"]),
            shading="gouraud",
            cmap="YlGnBu",
            norm=Normalize(250, 750),
        )
        if column == 0:
            ax.set_ylabel("RF elevation (deg)", fontsize=8)
            ax.set_yticks([-40, 10, 60])
        else:
            ax.set_yticks([])
        if row == rows - 1:
            ax.set_xticks([0, 50, 100])
        else:
            ax.set_xticks([])
        ax.tick_params(axis="both", labelsize=7)
        label = str(session_id)[-4:]
        if parameters is not None:
            shift = parameters[session_id]
            label += f" ({shift[0]:+.0f},{shift[1]:+.0f})°"
        ax.set_title(label, fontsize=7.2, pad=2)
        ax.set(xlim=DISPLAY_AZ_LIMITS, ylim=DISPLAY_EL_LIMITS, aspect="equal")
        ax.grid(alpha=.12)
    for index in range(len(sessions), rows * columns):
        ax = figure.add_subplot(axes[index // columns, index % columns])
        ax.axis("off")
    label_axis = figure.add_subplot(specification, frameon=False)
    label_axis.set_xticks([])
    label_axis.set_yticks([])
    label_axis.set_title(title, fontsize=12, pad=10)


def render_tuning_row(
    figure: plt.Figure,
    specification,
    templates: dict[tuple[str, str], dict[str, np.ndarray]],
    raw_templates: dict[tuple[str, str], dict[str, np.ndarray]],
    tuning_az: np.ndarray,
    tuning_el: np.ndarray,
    row_label: str,
) -> None:
    local_grid = specification.subgridspec(1, 4, wspace=.34)
    az_mesh, el_mesh = np.meshgrid(tuning_az, tuning_el)
    theta, radius = polar_coordinates(az_mesh, el_mesh)
    for column, key in enumerate(MAP_KEYS):
        group, preference = key
        ax = figure.add_subplot(local_grid[0, column], projection="polar")
        values = np.exp2(templates[key]["value"])
        reference = np.exp2(raw_templates[key]["value"])
        finite = reference[np.isfinite(reference)]
        limits = np.quantile(finite, [.02, .98])
        artist = ax.pcolormesh(
            theta,
            radius,
            values,
            shading="gouraud",
            cmap="viridis" if preference == "sf" else "plasma",
            norm=Normalize(*limits),
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_ylim(0, 110)
        ax.set_rlabel_position(65)
        ax.set_thetamin(-70)
        ax.set_thetamax(70)
        ax.grid(color="#B5B5B5", linewidth=.65)
        ax.set_title(f"{row_label} · {group}\n{preference.upper()} preference", fontsize=10)
        bar = figure.colorbar(artist, ax=ax, fraction=.026, pad=.07, extend="both")
        bar.set_label("cycles/deg" if preference == "sf" else "Hz", fontsize=9)


def render_figure(
    raw_size_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    aligned_size_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    raw_size: dict[str, np.ndarray],
    aligned_size: dict[str, np.ndarray],
    raw_tuning: dict[tuple[str, str], dict[str, np.ndarray]],
    aligned_tuning: dict[tuple[str, str], dict[str, np.ndarray]],
    sessions: list[int],
    parameters: dict[int, np.ndarray],
    size_az: np.ndarray,
    size_el: np.ndarray,
    tuning_az: np.ndarray,
    tuning_el: np.ndarray,
    edge_exclusion_deg: float,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(22.0, 22.5))
    grid = figure.add_gridspec(4, 1, height_ratios=[2.65, 1.25, 1.15, 1.15], hspace=.33)
    comparison_grid = grid[0, 0].subgridspec(1, 2, wspace=.10)
    render_session_grid(figure, comparison_grid[0, 0], raw_size_maps, sessions, size_az, size_el, "1 · Raw session RF-size maps")
    render_session_grid(figure, comparison_grid[0, 1], aligned_size_maps, sessions, size_az, size_el, "2 · Registered session RF-size maps · titles include (az, el) shift", parameters=parameters)

    aggregate_grid = grid[1, 0].subgridspec(1, 2, wspace=.20)
    top_axes = [figure.add_subplot(aggregate_grid[0, 0]), figure.add_subplot(aggregate_grid[0, 1])]
    artist = render_tiling(top_axes[0], raw_size, size_az, size_el, "3 · Raw interior V1 RF-size tiling")
    render_tiling(top_axes[1], aligned_size, size_az, size_el, "3 · Registered interior V1 RF-size tiling")
    size_colorbar_axis = figure.add_axes([.925, .393, .007, .12])
    colorbar = figure.colorbar(artist, cax=size_colorbar_axis, extend="both")
    colorbar.set_label("Fitted RF area (deg²)")
    render_tuning_row(figure, grid[2, 0], raw_tuning, raw_tuning, tuning_az, tuning_el, "4 · Raw stack")
    render_tuning_row(figure, grid[3, 0], aligned_tuning, raw_tuning, tuning_az, tuning_el, "5 · Shifted stack")
    figure.suptitle(
        f"Allen BO 1.1 V1 RF-size registration · {edge_exclusion_deg:g}° edge exclusion · {len(sessions)} sessions\n"
        "session maps → aggregate RF-size tiling → independent raw and shifted SF/TF stacks",
        fontsize=16,
    )
    figure.subplots_adjust(left=.05, right=.90, bottom=.035, top=.94)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_registered_size_zoom(
    aligned_size: dict[str, np.ndarray],
    size_az: np.ndarray,
    size_el: np.ndarray,
    edge_exclusion_deg: float,
    session_count: int,
    output_path: Path,
) -> None:
    figure, ax = plt.subplots(figsize=(9.2, 7.7))
    area = np.exp2(aligned_size["value"])
    finite = area[np.isfinite(area)]
    limits = np.quantile(finite, [.05, .95])
    artist = ax.pcolormesh(
        size_az, size_el, area, shading="gouraud", cmap="YlGnBu", norm=Normalize(*limits)
    )
    ax.contour(size_az, size_el, aligned_size["evidence"], levels=3,
               colors="#333333", linewidths=.65, alpha=.5)
    ax.set(
        aspect="equal", xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)",
        title=(f"3b · Registered V1 RF-size tiling · focused 5–95% scale\n"
               f"{edge_exclusion_deg:g}° edge exclusion · {session_count} sessions · "
               f"{limits[0]:.0f}–{limits[1]:.0f} deg²"),
    )
    ax.grid(alpha=.16)
    supported = aligned_size["evidence"] > 0
    rows, columns = np.where(supported)
    if len(rows):
        padding = 5.0
        ax.set_xlim(max(DISPLAY_AZ_LIMITS[0], size_az[columns].min() - padding), min(DISPLAY_AZ_LIMITS[1], size_az[columns].max() + padding))
        ax.set_ylim(max(DISPLAY_EL_LIMITS[0], size_el[rows].min() - padding), min(DISPLAY_EL_LIMITS[1], size_el[rows].max() + padding))
    colorbar = figure.colorbar(artist, ax=ax, fraction=.046, pad=.04, extend="both")
    colorbar.set_label("Fitted RF area (deg²)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_maps, tuning_az, tuning_el = load_maps(args.tuning_grid.resolve())
    all_sessions = sorted({key[0] for key in tuning_maps})
    transform_table = pd.read_csv(args.transforms.resolve())
    sessions = sorted(set(all_sessions) & set(transform_table["ecephys_session_id"].astype(int)))
    tuning_maps = {key: value for key, value in tuning_maps.items() if key[0] in sessions}
    parameters = load_rf_size_parameters(args.transforms.resolve(), sessions)

    population = prepare_population(pd.read_csv(args.support.resolve(), low_memory=False))
    population = population.loc[
        population["ecephys_session_id"].isin(sessions)
        & population["distance_to_nearest_grid_edge_deg"].ge(args.edge_exclusion_deg)
    ].copy()
    size_az = np.linspace(*DISPLAY_AZ_LIMITS, 61)
    size_el = np.linspace(*DISPLAY_EL_LIMITS, 61)
    size_maps = build_absolute_size_maps(
        population,
        sessions,
        size_az,
        size_el,
        minimum_effective_local_units=args.minimum_effective_local_units,
    )
    raw_size = template_from_maps(size_maps, "V1", "rf_size_absolute")
    aligned_size_maps = warp_all(size_maps, parameters, size_az, size_el)
    aligned_size = template_from_maps(aligned_size_maps, "V1", "rf_size_absolute")
    raw_tuning = aggregate_templates(tuning_maps)
    aligned_tuning = aggregate_templates(warp_all(tuning_maps, parameters, tuning_az, tuning_el))

    figure_path = output_dir / "Figure_allen_bo11_rf_size_registration_breakout.png"
    render_figure(
        size_maps,
        aligned_size_maps,
        raw_size,
        aligned_size,
        raw_tuning,
        aligned_tuning,
        sessions,
        parameters,
        size_az,
        size_el,
        tuning_az,
        tuning_el,
        args.edge_exclusion_deg,
        figure_path,
    )
    zoom_path = output_dir / "Figure_allen_bo11_registered_rf_size_3b_zoom.png"
    render_registered_size_zoom(
        aligned_size,
        size_az,
        size_el,
        args.edge_exclusion_deg,
        len(sessions),
        zoom_path,
    )
    report = [
        "# Allen BO 1.1 RF-size registration breakout",
        "",
        f"The figure uses {len(sessions)} sessions retained by the supplied transform table.",
        f"RF-size surfaces use V1 RF centers at least {args.edge_exclusion_deg:g}° from every RF-stimulus edge.",
        "The upper paired blocks tile each session's fitted absolute V1 RF-area surface in raw coordinates on the left and registered coordinates on the right; registered titles list (azimuth, elevation) translation in degrees.",
        "Directly below, the paired panels aggregate the raw and registered RF-size tiling without equivalent-circle overlays.",
        "RF-size maps are rendered on an expanded 0–100° azimuth × -40–60° elevation canvas so translated support remains visible.",
        "Rows 4 and 5 show raw and shifted SF/TF stacks on shared per-column color scales. These tuning maps were not used to fit the translations.",
    ]
    (output_dir / "ALLEN_BO11_RF_SIZE_REGISTRATION_BREAKOUT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "checkpoint": "06c_allen_bo11_rf_size_registration_breakout",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "tuning_grid": {"path": str(args.tuning_grid.resolve()), "sha256": sha256(args.tuning_grid.resolve())},
            "transforms": {"path": str(args.transforms.resolve()), "sha256": sha256(args.transforms.resolve())},
        },
        "parameters": {"sessions": sessions, "edge_exclusion_deg": args.edge_exclusion_deg, "bandwidth_deg": BANDWIDTH_DEG, "minimum_effective_local_units": args.minimum_effective_local_units, "circle_overlays": False, "display_azimuth_limits_deg": DISPLAY_AZ_LIMITS, "display_elevation_limits_deg": DISPLAY_EL_LIMITS},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
