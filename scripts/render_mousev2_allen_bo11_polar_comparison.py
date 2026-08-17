#!/usr/bin/env python3
"""Stack MouseV2 and Allen Brain Observatory 1.1 V1 SF/TF polar maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import polar_coordinates  # noqa: E402


DEFAULT_MOUSE = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06d_mousev2_frequency_preference_surfaces"
    / "mousev2_frequency_preference_surface_grid.csv"
)
DEFAULT_ALLEN = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06c_allen_rf_matching"
    / "frequency_preference_surfaces_tuning_enriched"
    / "frequency_preference_surface_grid.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06d_mousev2_frequency_preference_surfaces"
    / "Figure_mousev2_allen_bo11_v1_frequency_preference_surfaces_polar.png"
)
DEFAULT_DIFFERENCE_OUTPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06d_mousev2_frequency_preference_surfaces"
    / "Figure_mousev2_minus_allen_bo11_v1_frequency_preference_polar.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse-grid", type=Path, default=DEFAULT_MOUSE)
    parser.add_argument("--allen-grid", type=Path, default=DEFAULT_ALLEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--difference-output", type=Path, default=DEFAULT_DIFFERENCE_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_surface(
    table: pd.DataFrame, *, preference: str, area: str, bandwidth_deg: float
) -> pd.DataFrame:
    selected = table.loc[
        table["preference"].eq(preference)
        & table["area"].eq(area)
        & np.isclose(table["bandwidth_deg"], bandwidth_deg)
    ].sort_values(["elevation_deg", "azimuth_deg"])
    if selected.empty:
        raise ValueError(f"No {area} {preference} surface at {bandwidth_deg:g}°")
    if selected.duplicated(["azimuth_deg", "elevation_deg"]).any():
        raise ValueError(f"Duplicate grid coordinates for {area} {preference}")
    return selected


def render_comparison(
    mouse: pd.DataFrame,
    allen: pd.DataFrame,
    output_path: Path,
    *,
    bandwidth_deg: float,
) -> dict[str, object]:
    definitions = (
        ("MouseV2 multi-probe V1", mouse, "Multi-probe V1"),
        ("Allen Brain Observatory 1.1 V1", allen, "V1"),
    )
    preferences = ("sf", "tf")
    labels = {"sf": "preferred SF (cycles/degree)", "tf": "preferred TF (Hz)"}
    color_maps = {"sf": "cividis", "tf": "magma"}
    selected: dict[tuple[int, str], pd.DataFrame] = {}
    for row, (_, table, area) in enumerate(definitions):
        for preference in preferences:
            selected[(row, preference)] = select_surface(
                table,
                preference=preference,
                area=area,
                bandwidth_deg=bandwidth_deg,
            )

    reference = selected[(0, "sf")]
    azimuth = np.sort(reference["azimuth_deg"].unique())
    elevation = np.sort(reference["elevation_deg"].unique())
    expected_coordinates = reference[["azimuth_deg", "elevation_deg"]].to_numpy()
    for surface in selected.values():
        if not np.array_equal(
            surface[["azimuth_deg", "elevation_deg"]].to_numpy(), expected_coordinates
        ):
            raise ValueError("MouseV2 and Allen surface grids do not match")

    azimuth_mesh, elevation_mesh = np.meshgrid(azimuth, elevation)
    theta, eccentricity = polar_coordinates(
        azimuth_mesh.ravel(), elevation_mesh.ravel(),
        center_azimuth_deg=0.0, center_elevation_deg=0.0,
    )
    theta_mesh = theta.reshape(len(elevation), len(azimuth))
    eccentricity_mesh = eccentricity.reshape(len(elevation), len(azimuth))
    theta_degrees = np.rad2deg(theta)
    radial_limit = np.ceil(np.max(eccentricity) / 10.0) * 10.0

    limits = {}
    for preference in preferences:
        combined = np.concatenate(
            [
                selected[(row, preference)]["estimate_log2"].dropna().to_numpy()
                for row in range(2)
            ]
        )
        limits[preference] = np.quantile(combined, [0.02, 0.98])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 10.8),
        subplot_kw={"projection": "polar"},
    )
    artists = {}
    for row, (population, _, _) in enumerate(definitions):
        for column, preference in enumerate(preferences):
            ax = axes[row, column]
            surface = selected[(row, preference)]
            values = surface["estimate_log2"].to_numpy().reshape(
                len(elevation), len(azimuth)
            )
            lower, upper = limits[preference]
            artists[preference] = ax.pcolormesh(
                theta_mesh,
                eccentricity_mesh,
                values,
                shading="gouraud",
                cmap=color_maps[preference],
                norm=Normalize(vmin=lower, vmax=upper),
            )
            finite = values[np.isfinite(values)]
            contour_levels = np.linspace(lower, upper, 6)[1:-1]
            usable = contour_levels[
                (contour_levels > np.min(finite)) & (contour_levels < np.max(finite))
            ]
            if len(usable):
                ax.contour(
                    theta_mesh,
                    eccentricity_mesh,
                    values,
                    levels=usable,
                    colors="white",
                    linewidths=0.65,
                    alpha=0.65,
                )
            ax.set_thetamin(np.floor(theta_degrees.min() / 10.0) * 10.0)
            ax.set_thetamax(np.ceil(theta_degrees.max() / 10.0) * 10.0)
            ax.set_ylim(0, radial_limit)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rlabel_position(65)
            units = int(surface["source_units"].iloc[0])
            sessions = int(surface["source_sessions"].iloc[0])
            ax.set_title(
                f"{population}\n{units:,} units; {sessions} sessions",
                fontsize=11,
                pad=17,
            )
            if row == 1:
                ax.set_xlabel(
                    "polar angle from Allen (0°, 0°)", labelpad=17
                )
            if column == 0:
                ax.text(
                    np.deg2rad(64),
                    radial_limit * 0.58,
                    "eccentricity (deg)",
                    rotation=-26,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    for column, preference in enumerate(preferences):
        lower, upper = limits[preference]
        ticks = np.linspace(lower, upper, 5)
        colorbar = fig.colorbar(
            artists[preference],
            ax=axes[:, column],
            fraction=0.035,
            pad=0.075,
            extend="both",
        )
        colorbar.set_ticks(ticks)
        precision = 3 if preference == "sf" else 2
        colorbar.set_ticklabels([f"{np.exp2(value):.{precision}f}" for value in ticks])
        colorbar.set_label(labels[preference])
        fig.text(
            0.245 if column == 0 else 0.705,
            0.91,
            f"Preferred {preference.upper()}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

    fig.suptitle(
        "MouseV2 and Allen Brain Observatory 1.1 V1 frequency preference",
        fontsize=16,
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        f"shared color scales; polar center = Allen (0°, 0°); bandwidth {bandwidth_deg:g}°",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.015,
        "MouseV2: supported trial-fitted continuous preferences.  "
        "Allen 1.1: tuning-enriched released preferred bins; SF and TF came from different stimulus families.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.89, bottom=0.075, top=0.86, hspace=0.34, wspace=0.19)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "bandwidth_deg": bandwidth_deg,
        "polar_center_deg": [0.0, 20.0],
        "shared_color_limits_log2": {
            preference: list(map(float, limits[preference])) for preference in preferences
        },
        "mouse_method": "supported trial-fitted continuous preferences",
        "allen_method": "tuning-enriched released preferred bins",
    }


def difference_grid(
    mouse: pd.DataFrame, allen: pd.DataFrame, *, bandwidth_deg: float
) -> pd.DataFrame:
    frames = []
    for preference in ("sf", "tf"):
        mouse_surface = select_surface(
            mouse,
            preference=preference,
            area="Multi-probe V1",
            bandwidth_deg=bandwidth_deg,
        )
        allen_surface = select_surface(
            allen,
            preference=preference,
            area="V1",
            bandwidth_deg=bandwidth_deg,
        )
        merged = mouse_surface[
            ["azimuth_deg", "elevation_deg", "estimate_log2", "supported"]
        ].rename(
            columns={
                "estimate_log2": "mousev2_estimate_log2",
                "supported": "mousev2_supported",
            }
        ).merge(
            allen_surface[
                ["azimuth_deg", "elevation_deg", "estimate_log2", "supported"]
            ].rename(
                columns={
                    "estimate_log2": "allen_estimate_log2",
                    "supported": "allen_supported",
                }
            ),
            on=["azimuth_deg", "elevation_deg"],
            validate="one_to_one",
        )
        merged["preference"] = preference
        merged["shared_supported"] = (
            merged["mousev2_supported"]
            & merged["allen_supported"]
            & merged[["mousev2_estimate_log2", "allen_estimate_log2"]].notna().all(axis=1)
        )
        merged["mousev2_minus_allen_log2"] = (
            merged["mousev2_estimate_log2"] - merged["allen_estimate_log2"]
        ).where(merged["shared_supported"])
        merged["mousev2_over_allen_ratio"] = np.exp2(
            merged["mousev2_minus_allen_log2"]
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def render_difference(
    differences: pd.DataFrame,
    output_path: Path,
    *,
    bandwidth_deg: float,
) -> list[dict[str, object]]:
    preferences = ("sf", "tf")
    reference = differences.loc[differences["preference"].eq("sf")].sort_values(
        ["elevation_deg", "azimuth_deg"]
    )
    azimuth = np.sort(reference["azimuth_deg"].unique())
    elevation = np.sort(reference["elevation_deg"].unique())
    azimuth_mesh, elevation_mesh = np.meshgrid(azimuth, elevation)
    theta, eccentricity = polar_coordinates(
        azimuth_mesh.ravel(), elevation_mesh.ravel(),
        center_azimuth_deg=0.0, center_elevation_deg=0.0,
    )
    theta_mesh = theta.reshape(len(elevation), len(azimuth))
    eccentricity_mesh = eccentricity.reshape(len(elevation), len(azimuth))
    theta_degrees = np.rad2deg(theta)
    radial_limit = np.ceil(np.max(eccentricity) / 10.0) * 10.0

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 6.2), subplot_kw={"projection": "polar"}
    )
    summaries = []
    for column, preference in enumerate(preferences):
        selected = differences.loc[differences["preference"].eq(preference)].sort_values(
            ["elevation_deg", "azimuth_deg"]
        )
        values = selected["mousev2_minus_allen_log2"].to_numpy().reshape(
            len(elevation), len(azimuth)
        )
        finite = values[np.isfinite(values)]
        limit = max(float(np.quantile(np.abs(finite), 0.98)), 0.05)
        ax = axes[column]
        artist = ax.pcolormesh(
            theta_mesh,
            eccentricity_mesh,
            values,
            shading="gouraud",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        levels = np.linspace(-limit, limit, 7)
        ax.contour(
            theta_mesh,
            eccentricity_mesh,
            values,
            levels=levels,
            colors="black",
            linewidths=0.55,
            alpha=0.55,
        )
        ax.set_thetamin(np.floor(theta_degrees.min() / 10.0) * 10.0)
        ax.set_thetamax(np.ceil(theta_degrees.max() / 10.0) * 10.0)
        ax.set_ylim(0, radial_limit)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(65)
        median = float(np.median(finite))
        p10, p90 = map(float, np.quantile(finite, [0.1, 0.9]))
        ratio = float(np.exp2(median))
        shared_fraction = float(selected["shared_supported"].mean())
        ax.set_title(
            f"Preferred {preference.upper()}\n"
            f"median {median:+.3f} octaves ({ratio:.2f}×); shared grid {shared_fraction:.1%}",
            pad=18,
            fontsize=11,
        )
        ax.set_xlabel("polar angle from Allen (0°, 0°)", labelpad=18)
        ax.text(
            np.deg2rad(64),
            radial_limit * 0.58,
            "eccentricity (deg)",
            rotation=-26,
            ha="center",
            va="center",
            fontsize=9,
        )
        colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.09, extend="both")
        colorbar.set_label("MouseV2 − Allen V1 preference (octaves)")
        summaries.append(
            {
                "preference": preference,
                "shared_grid_fraction": shared_fraction,
                "median_difference_octaves": median,
                "p10_difference_octaves": p10,
                "p90_difference_octaves": p90,
                "median_mousev2_over_allen_ratio": ratio,
                "surface_correlation": float(
                    np.corrcoef(
                        selected.loc[selected["shared_supported"], "mousev2_estimate_log2"],
                        selected.loc[selected["shared_supported"], "allen_estimate_log2"],
                    )[0, 1]
                ),
            }
        )
    fig.suptitle(
        "MouseV2 minus Allen Brain Observatory 1.1 V1 frequency preference\n"
        f"shared supported RF coordinates; bandwidth {bandwidth_deg:g}°",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "Descriptive contrast: MouseV2 uses trial-fitted continuous preferences; Allen uses tuning-enriched released bins.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.11, top=0.82, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def main() -> None:
    args = parse_args()
    mouse_path = args.mouse_grid.resolve()
    allen_path = args.allen_grid.resolve()
    output_path = args.output.resolve()
    difference_output_path = args.difference_output.resolve()
    mouse = pd.read_csv(mouse_path)
    allen = pd.read_csv(allen_path)
    parameters = render_comparison(
        mouse,
        allen,
        output_path,
        bandwidth_deg=args.bandwidth_deg,
    )
    differences = difference_grid(mouse, allen, bandwidth_deg=args.bandwidth_deg)
    difference_grid_path = difference_output_path.with_suffix(".csv")
    differences.to_csv(difference_grid_path, index=False, float_format="%.7g")
    difference_summary = render_difference(
        differences, difference_output_path, bandwidth_deg=args.bandwidth_deg
    )
    manifest_path = output_path.with_suffix(".json")
    manifest = {
        "status": "MouseV2 over Allen Brain Observatory 1.1 V1 polar comparison",
        "inputs": {
            "mousev2": {"path": str(mouse_path), "sha256": sha256(mouse_path)},
            "allen_bo11_tuning_enriched": {
                "path": str(allen_path),
                "sha256": sha256(allen_path),
            },
        },
        "parameters": parameters,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "difference_summary": difference_summary,
        "outputs": {
            "stacked_comparison": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
            },
            "difference_figure": {
                "path": str(difference_output_path),
                "bytes": difference_output_path.stat().st_size,
                "sha256": sha256(difference_output_path),
            },
            "difference_grid": {
                "path": str(difference_grid_path),
                "bytes": difference_grid_path.stat().st_size,
                "sha256": sha256(difference_grid_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MouseV2/Allen BO 1.1 polar comparison written to {output_path}")


if __name__ == "__main__":
    main()
