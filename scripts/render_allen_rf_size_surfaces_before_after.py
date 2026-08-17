#!/usr/bin/env python3
"""Render matched point-versus-aperture RF-size surfaces for V1 and HVAs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS = (
    ROOT
    / "artifacts"
    / "allen_multisession_rf_validation_v1"
    / "03_geometry"
    / "all_session_unit_geometry_fits.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "allen_multisession_rf_validation_v1"
    / "07_registration_readiness"
)
SESSIONS = (746083955, 755434585, 760693773, 798911424)
GROUPS = ("V1", "HVA")
MODELS = ("point", "aperture")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def matched_interior(fits: pd.DataFrame) -> pd.DataFrame:
    keys = ["session_id", "ecephys_unit_id"]
    usable = fits.loc[
        ~fits.axis_censored.astype(bool) & fits.axis_edge_distance_deg.gt(10)
    ].copy()
    counts = usable.groupby(keys, observed=True).spatial_model.nunique()
    matched = counts.loc[counts.eq(2)].index
    usable = usable.set_index(keys).loc[matched].reset_index()
    usable["azimuth_deg"] = usable.axis_center_x_deg + 50
    usable["elevation_deg"] = usable.axis_center_y_deg + 10
    usable["log2_area_deg2"] = np.log2(usable.axis_area_deg2)
    return usable


def kernel_surface(local, az_grid, el_grid, bandwidth=15, minimum_effective=3):
    points = local[["azimuth_deg", "elevation_deg"]].to_numpy(float)
    values = local.log2_area_deg2.to_numpy(float)
    surface = np.full((len(el_grid), len(az_grid)), np.nan)
    effective = np.zeros_like(surface)
    for row, elevation in enumerate(el_grid):
        for column, azimuth in enumerate(az_grid):
            distance = np.sqrt(np.sum((points - [azimuth, elevation]) ** 2, axis=1))
            weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
            if weights.sum() > 0:
                effective[row, column] = weights.sum() ** 2 / np.square(weights).sum()
            if effective[row, column] >= minimum_effective and np.sum(distance <= 20) >= 3:
                surface[row, column] = np.average(values, weights=weights)
            else:
                effective[row, column] = 0
    return surface, effective


def build_maps(data, az_grid, el_grid):
    maps = {}
    rows = []
    for session_id in SESSIONS:
        for group in GROUPS:
            for model in MODELS:
                local = data.loc[
                    data.session_id.eq(session_id)
                    & data.group.eq(group)
                    & data.spatial_model.eq(model)
                ]
                surface, effective = kernel_surface(local, az_grid, el_grid)
                maps[(session_id, group, model)] = (surface, effective, len(local))
                for r, elevation in enumerate(el_grid):
                    for c, azimuth in enumerate(az_grid):
                        rows.append(
                            {
                                "session_id": session_id,
                                "group": group,
                                "spatial_model": model,
                                "azimuth_deg": azimuth,
                                "elevation_deg": elevation,
                                "log2_area_deg2": surface[r, c],
                                "effective_units": effective[r, c],
                                "source_units": len(local),
                            }
                        )
    return maps, pd.DataFrame(rows)


def shared_limits(maps):
    values = np.concatenate(
        [surface[np.isfinite(surface)] for surface, _, _ in maps.values()]
    )
    return tuple(np.nanquantile(values, [0.02, 0.98]))


def support_contours(axis, az_grid, el_grid, effective):
    maximum = np.nanmax(effective)
    levels = [value for value in (3, 6, 12) if value <= maximum]
    if levels:
        axis.contour(
            az_grid,
            el_grid,
            effective,
            levels=levels,
            colors="#333333",
            linewidths=0.55,
        )


def render_by_session(maps, az_grid, el_grid, path):
    vmin, vmax = shared_limits(maps)
    columns = (("V1", "point"), ("V1", "aperture"), ("HVA", "point"), ("HVA", "aperture"))
    fig, axes = plt.subplots(
        len(SESSIONS), 4, figsize=(15, 15), sharex=True, sharey=True, constrained_layout=True
    )
    for row, session_id in enumerate(SESSIONS):
        for column, (group, model) in enumerate(columns):
            surface, effective, count = maps[(session_id, group, model)]
            axis = axes[row, column]
            image = axis.pcolormesh(
                az_grid,
                el_grid,
                surface,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            support_contours(axis, az_grid, el_grid, effective)
            axis.set_aspect("equal")
            axis.set_title(f"{session_id} · {group} · {model}\nmatched interior n={count}")
            if row == len(SESSIONS) - 1:
                axis.set_xlabel("RF azimuth (deg)")
            if column == 0:
                axis.set_ylabel("RF elevation (deg)")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.62, pad=0.02)
    colorbar.set_label("Smoothed log₂ latent half-max RF area (deg²)")
    fig.suptitle("RF-size surfaces before and after analytic aperture correction", fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pooled_maps(maps):
    pooled = {}
    for group in GROUPS:
        for model in MODELS:
            surfaces = np.stack([maps[(sid, group, model)][0] for sid in SESSIONS])
            support = np.sum(np.isfinite(surfaces), axis=0)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(surfaces, axis=0)
            mean[support < 2] = np.nan
            pooled[(group, model)] = (mean, support)
    return pooled


def render_pooled(maps, az_grid, el_grid, path):
    pooled = pooled_maps(maps)
    absolute = np.concatenate(
        [pooled[(group, model)][0][np.isfinite(pooled[(group, model)][0])] for group in GROUPS for model in MODELS]
    )
    vmin, vmax = np.nanquantile(absolute, [0.02, 0.98])
    differences = {
        group: pooled[(group, "aperture")][0] - pooled[(group, "point")][0]
        for group in GROUPS
    }
    difference_limit = max(
        0.25,
        float(
            np.nanquantile(
                np.abs(np.concatenate([value[np.isfinite(value)] for value in differences.values()])),
                0.98,
            )
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for row, group in enumerate(GROUPS):
        for column, model in enumerate(MODELS):
            surface, support = pooled[(group, model)]
            image = axes[row, column].pcolormesh(
                az_grid, el_grid, surface, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[row, column].contour(
                az_grid, el_grid, support, levels=[1.5, 2.5, 3.5], colors="#333333", linewidths=0.55
            )
            axes[row, column].set_title(f"{group} · {'before: point' if model == 'point' else 'after: aperture'}")
        delta = differences[group]
        delta_image = axes[row, 2].pcolormesh(
            az_grid,
            el_grid,
            delta,
            shading="auto",
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        axes[row, 2].set_title(f"{group} · after − before")
        for column in range(3):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("RF azimuth (deg)")
        axes[row, 0].set_ylabel("RF elevation (deg)")
    absolute_bar = fig.colorbar(image, ax=axes[:, :2], shrink=0.75, pad=0.02)
    absolute_bar.set_label("Session-balanced mean log₂ RF area (deg²)")
    difference_bar = fig.colorbar(delta_image, ax=axes[:, 2], shrink=0.75, pad=0.02)
    difference_bar.set_label("Change in log₂ RF area (aperture − point)")
    fig.suptitle("Session-balanced RF-size surfaces: point model versus aperture correction", fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(args.fits.resolve(), low_memory=False)
    data = matched_interior(fits)
    az_grid = np.linspace(10, 90, 33)
    el_grid = np.linspace(-30, 50, 33)
    maps, grid = build_maps(data, az_grid, el_grid)
    data.to_csv(output / "matched_interior_point_aperture_units.csv", index=False, float_format="%.9g")
    grid.to_csv(output / "point_aperture_rf_size_surface_grid.csv", index=False, float_format="%.9g")
    render_by_session(
        maps,
        az_grid,
        el_grid,
        output / "Figure_rf_size_surfaces_point_vs_aperture_by_session.png",
    )
    render_pooled(
        maps,
        az_grid,
        el_grid,
        output / "Figure_rf_size_surfaces_point_vs_aperture_pooled.png",
    )
    print(
        data.groupby(["session_id", "group", "spatial_model"], observed=True)
        .axis_area_deg2.agg(["count", "median"])
        .to_string()
    )


if __name__ == "__main__":
    main()
