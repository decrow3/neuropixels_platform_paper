#!/usr/bin/env python3
"""Fit animal-balanced RF-size surfaces over observed RF location for V1/HVAs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from analyze_14animal_v1_hva_rf_location_size_relationships import (
    prepare,
    select_bandwidth,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/retinotopy_cross_animal_registration_14_v1/rf_size_map_alignment"
    / "v1_vs_hva_relationships/rf_size_surfaces"
)
COORDINATES = ("common_azimuth_deg", "common_elevation_deg")
AZIMUTH = np.linspace(15, 85, 71)
ELEVATION = np.linspace(-20, 40, 61)
MIN_EFFECTIVE_CELLS_PER_ANIMAL = 3.0
MIN_NEARBY_CELLS_PER_ANIMAL = 3
NEARBY_RADIUS_DEG = 20.0
MIN_SUPPORTING_ANIMALS = 7


def animal_balanced_surface(
    data: pd.DataFrame,
    target: str,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_mesh, elevation_mesh = np.meshgrid(AZIMUTH, ELEVATION)
    grid = np.column_stack([azimuth_mesh.ravel(), elevation_mesh.ravel()])
    coordinates = data[list(COORDINATES)].to_numpy(float)
    center = np.nanmedian(coordinates, axis=0)
    scale = np.nanquantile(coordinates, .75, axis=0) - np.nanquantile(coordinates, .25, axis=0)
    scale = np.where(scale > 1e-9, scale, np.nanstd(coordinates, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    grid_scaled = (grid - center) / scale

    predictions = []
    effective = []
    supported = []
    for _, animal in data.groupby("session_id", observed=True):
        animal_coordinates = animal[list(COORDINATES)].to_numpy(float)
        animal_scaled = (animal_coordinates - center) / scale
        distance2 = np.sum((grid_scaled[:, None, :] - animal_scaled[None, :, :]) ** 2, axis=2)
        weights = np.exp(-.5 * distance2 / bandwidth**2)
        weight_sum = weights.sum(axis=1)
        ess = np.divide(
            weight_sum**2,
            np.square(weights).sum(axis=1),
            out=np.zeros_like(weight_sum),
            where=np.square(weights).sum(axis=1) > 0,
        )
        distance_deg = np.sqrt(np.sum((grid[:, None, :] - animal_coordinates[None, :, :]) ** 2, axis=2))
        near = (distance_deg <= NEARBY_RADIUS_DEG).sum(axis=1)
        valid = (
            (ess >= MIN_EFFECTIVE_CELLS_PER_ANIMAL)
            & (near >= MIN_NEARBY_CELLS_PER_ANIMAL)
            & (weight_sum > 1e-12)
        )
        prediction = np.divide(
            weights @ animal[target].to_numpy(float),
            weight_sum,
            out=np.full(len(grid), np.nan),
            where=weight_sum > 1e-12,
        )
        prediction[~valid] = np.nan
        predictions.append(prediction)
        effective.append(ess)
        supported.append(valid)

    prediction_stack = np.vstack(predictions)
    support_stack = np.vstack(supported)
    support_count = support_stack.sum(axis=0)
    consensus = np.divide(
        np.nansum(prediction_stack, axis=0), support_count,
        out=np.full(prediction_stack.shape[1], np.nan), where=support_count > 0,
    )
    consensus[support_count < MIN_SUPPORTING_ANIMALS] = np.nan
    mean_effective = np.divide(
        np.nansum(np.where(support_stack, np.vstack(effective), np.nan), axis=0), support_count,
        out=np.full(prediction_stack.shape[1], np.nan), where=support_count > 0,
    )
    shape = (len(ELEVATION), len(AZIMUTH))
    return consensus.reshape(shape), support_count.reshape(shape), mean_effective.reshape(shape)


def build_surfaces(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    surfaces = {}
    grid_rows = []
    selection_rows = []
    for cortical_group in ("V1", "HVA"):
        local = data.loc[data.cortical_group.eq(cortical_group)].copy()
        bandwidth, history = select_bandwidth(
            local, COORDINATES, "group_centered_log2_rf_area"
        )
        for row in history:
            selection_rows.append({"cortical_group": cortical_group, **row})
        absolute_log2, support, effective = animal_balanced_surface(
            local, "log2_rf_area_deg2", bandwidth
        )
        relative, support_relative, _ = animal_balanced_surface(
            local, "group_centered_log2_rf_area", bandwidth
        )
        if not np.array_equal(support, support_relative):
            raise AssertionError("Absolute and relative surface support differs")
        surfaces[cortical_group] = {
            "absolute_area_deg2": np.exp2(absolute_log2),
            "relative_log2_octaves": relative,
            "supporting_animals": support,
            "mean_effective_cells": effective,
            "bandwidth_iqr_units": bandwidth,
            "cells": len(local),
            "animals": local.session_id.nunique(),
        }
        for row_index, elevation in enumerate(ELEVATION):
            for column_index, azimuth in enumerate(AZIMUTH):
                grid_rows.append({
                    "cortical_group": cortical_group,
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "animal_balanced_rf_area_deg2": surfaces[cortical_group]["absolute_area_deg2"][row_index, column_index],
                    "animal_balanced_relative_log2_octaves": relative[row_index, column_index],
                    "supporting_animals": int(support[row_index, column_index]),
                    "mean_effective_cells_per_supporting_animal": effective[row_index, column_index],
                    "bandwidth_iqr_units": bandwidth,
                })
    return surfaces, pd.DataFrame(grid_rows), pd.DataFrame(selection_rows)


def support_contours(axis: plt.Axes, support: np.ndarray) -> None:
    levels = [value for value in (7, 10, 13) if np.nanmax(support) >= value]
    if levels:
        contours = axis.contour(
            AZIMUTH, ELEVATION, support, levels=levels,
            colors="#222222", linewidths=.7,
        )
        axis.clabel(contours, inline=True, fontsize=7, fmt=lambda value: f"{int(value)} animals")


def render(data: pd.DataFrame, surfaces: dict, output: Path) -> None:
    absolute_values = np.concatenate([
        value["absolute_area_deg2"][np.isfinite(value["absolute_area_deg2"])]
        for value in surfaces.values()
    ])
    absolute_limits = np.nanquantile(absolute_values, [.02, .98])
    relative_values = np.concatenate([
        value["relative_log2_octaves"][np.isfinite(value["relative_log2_octaves"])]
        for value in surfaces.values()
    ])
    relative_limit = max(.25, float(np.nanquantile(np.abs(relative_values), .98)))

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8), sharex=True, sharey=True, constrained_layout=True)
    absolute_artist = relative_artist = support_artist = None
    for row, cortical_group in enumerate(("V1", "HVA")):
        local = data.loc[data.cortical_group.eq(cortical_group)]
        surface = surfaces[cortical_group]
        absolute_artist = axes[row, 0].pcolormesh(
            AZIMUTH, ELEVATION, surface["absolute_area_deg2"], shading="auto",
            cmap="viridis", norm=LogNorm(vmin=absolute_limits[0], vmax=absolute_limits[1]),
        )
        support_contours(axes[row, 0], surface["supporting_animals"])
        relative_artist = axes[row, 1].pcolormesh(
            AZIMUTH, ELEVATION, surface["relative_log2_octaves"], shading="auto",
            cmap="coolwarm", vmin=-relative_limit, vmax=relative_limit,
        )
        support_contours(axes[row, 1], surface["supporting_animals"])
        support_artist = axes[row, 2].pcolormesh(
            AZIMUTH, ELEVATION, surface["supporting_animals"], shading="auto",
            cmap="Blues", vmin=0, vmax=14,
        )
        axes[row, 2].contour(
            AZIMUTH, ELEVATION, surface["supporting_animals"],
            levels=[MIN_SUPPORTING_ANIMALS-.5], colors="#222222", linewidths=1.0,
        )
        for column in range(3):
            axes[row, column].scatter(
                local.common_azimuth_deg, local.common_elevation_deg,
                s=3, c="#222222", alpha=.07, linewidths=0,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("RF azimuth (deg)")
            axes[row, column].grid(alpha=.10)
        axes[row, 0].set_title(
            f"{cortical_group}: absolute RF-size surface\n"
            f"n={surface['cells']:,} cells; bandwidth={surface['bandwidth_iqr_units']:.2f} IQR"
        )
        axes[row, 1].set_title(
            f"{cortical_group}: animal-normalized surface\n"
            + ("V1 median removed" if cortical_group == "V1" else "HVA median removed; area means retained")
        )
        axes[row, 2].set_title(f"{cortical_group}: cross-animal support")
        axes[row, 0].set_ylabel("RF elevation (deg)")

    fig.colorbar(absolute_artist, ax=axes[:, 0], fraction=.035, pad=.02, label="RF half-max area (deg²; log color scale)")
    fig.colorbar(relative_artist, ax=axes[:, 1], fraction=.035, pad=.02, label="Relative log₂ RF area (octaves)")
    fig.colorbar(support_artist, ax=axes[:, 2], fraction=.035, pad=.02, label="Animals with local support")
    fig.suptitle(
        "Animal-balanced RF-size surfaces over observed visual-field location\n"
        "uncensored aperture fits, RF centers >10° from stimulus edge; surface shown where ≥7/14 animals support it",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = prepare()
    surfaces, grid, selection = build_surfaces(data)
    grid.to_csv(OUTPUT / "animal_balanced_rf_size_surface_grid.csv.gz", index=False, compression="gzip")
    selection.to_csv(OUTPUT / "surface_bandwidth_leave_one_animal_out_selection.csv", index=False)
    figure = OUTPUT / "Figure_V1_HVA_RF_size_surfaces_over_RF_location.png"
    render(data, surfaces, figure)
    chart_contract = {
        "question": "What smooth RF-size surface is supported over observed RF azimuth/elevation, separately in V1 and HVAs?",
        "family": "heatmap with cross-animal support contours",
        "grain": "full-data animal-balanced consensus; one local smooth per animal before averaging",
        "bandwidth_selection": "leave-one-animal-out rank prediction, separately for V1 and HVA",
        "palette": "viridis for absolute positive area; blue/red diverging for signed animal-normalized deviations; blue support",
        "support_rule": "at least 3 effective and 3 nearby cells per animal; display consensus for at least 7 animals",
        "output": str(figure),
    }
    (OUTPUT / "chart_contract.json").write_text(json.dumps(chart_contract, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "exploratory descriptive consensus surfaces with cross-validated bandwidth",
        "groups": {
            group: {
                "cells": value["cells"], "animals": value["animals"],
                "bandwidth_iqr_units": value["bandwidth_iqr_units"],
                "supported_grid_fraction": float(np.mean(value["supporting_animals"] >= MIN_SUPPORTING_ANIMALS)),
            }
            for group, value in surfaces.items()
        },
        "absolute_target": "log2 analytic-aperture half-max area, back-transformed to deg2",
        "relative_target": "log2 area minus animal V1 median or overall animal HVA median; LM/AL/RL/AM means retained",
        "grid": {"azimuth_deg": [float(AZIMUTH.min()), float(AZIMUTH.max()), len(AZIMUTH)],
                 "elevation_deg": [float(ELEVATION.min()), float(ELEVATION.max()), len(ELEVATION)]},
        "minimum_supporting_animals": MIN_SUPPORTING_ANIMALS,
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
