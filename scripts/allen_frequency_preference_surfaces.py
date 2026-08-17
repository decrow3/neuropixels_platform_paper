#!/usr/bin/env python3
"""Map Allen SF and TF preferences over achieved RF azimuth/elevation.

The released table contains preferred bins rather than full tuning curves.  This
checkpoint therefore estimates nonlinear *preference* surfaces, using only
Brain Observatory sessions where multiple SF and TF values were presented.
Gaussian smoothing is session-balanced and unsupported grid cells are masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_config  # noqa: E402


DEFAULT_AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_OUTPUT = DEFAULT_AUDIT / "frequency_preference_surfaces"
AREA_ORDER = ("V1", "LM", "RL", "AL", "PM", "AM")
HVA_ORDER = AREA_ORDER[1:]
POOLED_HVA = "HVA pooled"
BO_COHORT = "Brain Observatory 1.1"
PREFERENCES = {
    "sf": {
        "column": "pref_sf_sg",
        "label": "preferred SF (cycles/degree)",
        "values": (0.02, 0.04, 0.08, 0.16, 0.32),
    },
    "tf": {
        "column": "pref_tf_dg",
        "label": "preferred TF (Hz)",
        "values": (1.0, 2.0, 4.0, 8.0, 15.0),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--bandwidths-deg",
        type=float,
        nargs="+",
        default=(8.0, 12.0, 16.0),
        help="Gaussian spatial bandwidth sensitivities.",
    )
    parser.add_argument("--primary-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-effective-sessions", type=float, default=3.0)
    parser.add_argument("--minimum-local-units", type=int, default=20)
    parser.add_argument(
        "--minimum-lifetime-sparseness",
        type=float,
        default=None,
        help="Metric-specific condition-selectivity threshold; omit for the inclusive surface.",
    )
    parser.add_argument(
        "--minimum-stimulus-firing-rate",
        type=float,
        default=None,
        help="Metric-specific static/drifting-grating firing-rate threshold in Hz.",
    )
    parser.add_argument(
        "--require-unique-preference",
        action="store_true",
        help="Exclude units whose released preferred SF or TF has a tied maximum.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bandwidths(bandwidths: list[float], primary: float) -> tuple[float, ...]:
    values = tuple(sorted(set(float(value) for value in bandwidths)))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Every bandwidth must be positive")
    if not any(np.isclose(primary, value) for value in values):
        raise ValueError("Primary bandwidth must be included in --bandwidths-deg")
    return values


def load_preference_units(
    audit_dir: Path,
    config_path: Path | None,
    *,
    minimum_lifetime_sparseness: float | None = None,
    minimum_stimulus_firing_rate: float | None = None,
    require_unique_preference: bool = False,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    support_path = audit_dir / "rf_unit_common_support.csv"
    manifest_path = audit_dir / "run_manifest.json"
    if not support_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing RF-audit contract in {audit_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["parameters"]["population"] != "published_like":
        raise ValueError("Preference surfaces require the published_like RF audit")

    support = pd.read_csv(support_path, low_memory=False)
    support = support.loc[
        support["cohort"].eq(BO_COHORT) & support["area"].isin(AREA_ORDER)
    ].copy()
    config = load_config(config_path)
    unit_path = Path(config["allen_unit_table"])
    if not unit_path.is_absolute():
        unit_path = ROOT / unit_path
    preferences = pd.read_csv(
        unit_path,
        usecols=[
            "ecephys_unit_id",
            "pref_sf_sg",
            "pref_tf_dg",
            "pref_sf_multi_sg",
            "pref_tf_multi_dg",
            "firing_rate_sg",
            "firing_rate_dg",
            "fano_sg",
            "fano_dg",
            "lifetime_sparseness_sg",
            "lifetime_sparseness_dg",
        ],
        low_memory=False,
    )
    if preferences["ecephys_unit_id"].duplicated().any():
        raise ValueError("Released Allen table contains duplicate unit IDs")
    result = support.merge(preferences, on="ecephys_unit_id", how="left", validate="one_to_one")
    for specification in PREFERENCES.values():
        column = specification["column"]
        result[column] = pd.to_numeric(result[column], errors="coerce")
        valid = result[column].isin(specification["values"])
        result.loc[~valid, column] = np.nan
    tuning_fields = {
        "sf": ("lifetime_sparseness_sg", "firing_rate_sg", "pref_sf_multi_sg"),
        "tf": ("lifetime_sparseness_dg", "firing_rate_dg", "pref_tf_multi_dg"),
    }
    for preference, (sparseness, firing_rate, multiple) in tuning_fields.items():
        fano = "fano_sg" if preference == "sf" else "fano_dg"
        for column in (sparseness, firing_rate, multiple, fano):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        eligible = result[PREFERENCES[preference]["column"]].notna()
        if minimum_lifetime_sparseness is not None:
            eligible &= result[sparseness].gt(float(minimum_lifetime_sparseness))
        if minimum_stimulus_firing_rate is not None:
            eligible &= result[firing_rate].gt(float(minimum_stimulus_firing_rate))
        if require_unique_preference:
            eligible &= result[multiple].eq(0)
        result[f"tuning_eligible_{preference}"] = eligible
    return result, unit_path, manifest


def session_balanced_gaussian_surface(
    points: np.ndarray,
    values: np.ndarray,
    sessions: np.ndarray,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> dict[str, np.ndarray]:
    """Estimate a nonlinear surface with equal total prior weight per session."""
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    sessions = np.asarray(sessions)
    grid_points = np.asarray(grid_points, dtype=float)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(values)
    points, values, sessions = points[finite], values[finite], sessions[finite]
    if not len(values):
        empty = np.full(len(grid_points), np.nan)
        return {
            "estimate_log2": empty,
            "effective_sessions": np.zeros(len(grid_points)),
            "local_units": np.zeros(len(grid_points), dtype=int),
            "supported": np.zeros(len(grid_points), dtype=bool),
        }

    unique_sessions, inverse, counts = np.unique(sessions, return_inverse=True, return_counts=True)
    prior = 1.0 / counts[inverse]
    distance_squared = np.sum(
        (grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2
    )
    kernel = np.exp(-0.5 * distance_squared / float(bandwidth_deg) ** 2)
    weights = kernel * prior[None, :]
    denominator = weights.sum(axis=1)
    estimate = np.divide(
        weights @ values,
        denominator,
        out=np.full(len(grid_points), np.nan),
        where=denominator > 0,
    )

    session_weights = np.zeros((len(grid_points), len(unique_sessions)))
    for index in range(len(unique_sessions)):
        session_weights[:, index] = weights[:, inverse == index].sum(axis=1)
    effective_sessions = np.divide(
        session_weights.sum(axis=1) ** 2,
        np.square(session_weights).sum(axis=1),
        out=np.zeros(len(grid_points)),
        where=np.square(session_weights).sum(axis=1) > 0,
    )
    local_units = (distance_squared <= (1.5 * float(bandwidth_deg)) ** 2).sum(axis=1)
    supported = (
        (effective_sessions >= float(minimum_effective_sessions))
        & (local_units >= int(minimum_local_units))
    )
    estimate[~supported] = np.nan
    return {
        "estimate_log2": estimate,
        "effective_sessions": effective_sessions,
        "local_units": local_units.astype(int),
        "supported": supported,
    }


def build_grid(units: pd.DataFrame, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 10:
        raise ValueError("grid size must be at least 10")
    azimuth = units["azimuth_rf"].to_numpy(dtype=float)
    elevation = units["elevation_rf"].to_numpy(dtype=float)
    az_limits = np.quantile(azimuth[np.isfinite(azimuth)], [0.01, 0.99])
    el_limits = np.quantile(elevation[np.isfinite(elevation)], [0.01, 0.99])
    az_grid = np.linspace(*az_limits, grid_size)
    el_grid = np.linspace(*el_limits, grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    return az_grid, el_grid, points


def polar_coordinates(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    *,
    center_azimuth_deg: float = 0.0,
    center_elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert visual-field coordinates to polar angle and eccentricity."""
    delta_azimuth = np.asarray(azimuth_deg, dtype=float) - center_azimuth_deg
    delta_elevation = np.asarray(elevation_deg, dtype=float) - center_elevation_deg
    theta_rad = np.arctan2(delta_elevation, delta_azimuth)
    eccentricity_deg = np.hypot(delta_azimuth, delta_elevation)
    return theta_rad, eccentricity_deg


def rf_occupancy_counts(
    units: pd.DataFrame,
    azimuth_edges_deg: np.ndarray,
    elevation_edges_deg: np.ndarray,
) -> np.ndarray:
    """Count units in elevation × azimuth visual-field bins."""
    counts, _, _ = np.histogram2d(
        pd.to_numeric(units["elevation_rf"], errors="coerce"),
        pd.to_numeric(units["azimuth_rf"], errors="coerce"),
        bins=[elevation_edges_deg, azimuth_edges_deg],
    )
    return counts.astype(int)


def estimate_surfaces(
    units: pd.DataFrame,
    grid_points: np.ndarray,
    bandwidths: tuple[float, ...],
    *,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for preference, specification in PREFERENCES.items():
        column = specification["column"]
        for area in AREA_ORDER + (POOLED_HVA,):
            area_mask = units["area"].isin(HVA_ORDER) if area == POOLED_HVA else units["area"].eq(area)
            group = units.loc[area_mask & units[f"tuning_eligible_{preference}"]].dropna(
                subset=[column, "azimuth_rf", "elevation_rf"]
            )
            points = group[["azimuth_rf", "elevation_rf"]].to_numpy(dtype=float)
            values = np.log2(group[column].to_numpy(dtype=float))
            sessions = group["ecephys_session_id"].to_numpy()
            for bandwidth in bandwidths:
                surface = session_balanced_gaussian_surface(
                    points,
                    values,
                    sessions,
                    grid_points,
                    bandwidth_deg=bandwidth,
                    minimum_effective_sessions=minimum_effective_sessions,
                    minimum_local_units=minimum_local_units,
                )
                if area == "V1":
                    reference_group = group
                    reference_surface = surface
                else:
                    reference_group = units.loc[
                        units["area"].eq("V1")
                        & units["ecephys_session_id"].isin(group["ecephys_session_id"])
                        & units[f"tuning_eligible_{preference}"]
                    ].dropna(subset=[column, "azimuth_rf", "elevation_rf"])
                    reference_surface = session_balanced_gaussian_surface(
                        reference_group[["azimuth_rf", "elevation_rf"]].to_numpy(dtype=float),
                        np.log2(reference_group[column].to_numpy(dtype=float)),
                        reference_group["ecephys_session_id"].to_numpy(),
                        grid_points,
                        bandwidth_deg=bandwidth,
                        minimum_effective_sessions=minimum_effective_sessions,
                        minimum_local_units=minimum_local_units,
                    )
                frame = pd.DataFrame(
                    {
                        "preference": preference,
                        "area": area,
                        "bandwidth_deg": bandwidth,
                        "azimuth_deg": grid_points[:, 0],
                        "elevation_deg": grid_points[:, 1],
                        "estimate_log2": surface["estimate_log2"],
                        "estimate_preference": np.exp2(surface["estimate_log2"]),
                        "effective_sessions": surface["effective_sessions"],
                        "local_units": surface["local_units"],
                        "supported": surface["supported"],
                        "source_units": len(group),
                        "source_sessions": group["ecephys_session_id"].nunique(),
                        "paired_v1_estimate_log2": reference_surface["estimate_log2"],
                        "paired_v1_supported": reference_surface["supported"],
                        "paired_v1_source_units": len(reference_group),
                        "paired_v1_source_sessions": reference_group["ecephys_session_id"].nunique(),
                    }
                )
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize_surfaces(surfaces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in surfaces.groupby(
        ["preference", "area", "bandwidth_deg"], observed=True, sort=True
    ):
        supported = group.loc[group["supported"] & group["estimate_log2"].notna()]
        shared = group.loc[group["shared_v1_support"] & group["delta_from_v1_log2"].notna()]
        rows.append(
            {
                "preference": keys[0],
                "area": keys[1],
                "bandwidth_deg": keys[2],
                "source_units": int(group["source_units"].iloc[0]),
                "source_sessions": int(group["source_sessions"].iloc[0]),
                "supported_grid_fraction": float(group["supported"].mean()),
                "surface_median_preference": float(np.exp2(supported["estimate_log2"].median())) if len(supported) else np.nan,
                "surface_p10_preference": float(np.exp2(supported["estimate_log2"].quantile(0.1))) if len(supported) else np.nan,
                "surface_p90_preference": float(np.exp2(supported["estimate_log2"].quantile(0.9))) if len(supported) else np.nan,
                "shared_v1_grid_fraction": float(group["shared_v1_support"].mean()),
                "median_delta_from_v1_log2": float(shared["delta_from_v1_log2"].median()) if len(shared) else np.nan,
                "p10_delta_from_v1_log2": float(shared["delta_from_v1_log2"].quantile(0.1)) if len(shared) else np.nan,
                "p90_delta_from_v1_log2": float(shared["delta_from_v1_log2"].quantile(0.9)) if len(shared) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_v1_differences(surfaces: pd.DataFrame) -> pd.DataFrame:
    if {"paired_v1_estimate_log2", "paired_v1_supported"}.issubset(surfaces.columns):
        result = surfaces.copy()
        result["v1_estimate_log2"] = result["paired_v1_estimate_log2"]
        result["v1_supported"] = result["paired_v1_supported"]
        result["shared_v1_support"] = result["supported"] & result["v1_supported"].fillna(False)
        result["delta_from_v1_log2"] = (
            result["estimate_log2"] - result["v1_estimate_log2"]
        ).where(result["shared_v1_support"])
        return result
    keys = ["preference", "bandwidth_deg", "azimuth_deg", "elevation_deg"]
    v1 = surfaces.loc[surfaces["area"].eq("V1"), keys + ["estimate_log2", "supported"]].rename(
        columns={"estimate_log2": "v1_estimate_log2", "supported": "v1_supported"}
    )
    result = surfaces.merge(v1, on=keys, how="left", validate="many_to_one")
    result["shared_v1_support"] = result["supported"] & result["v1_supported"].fillna(False)
    result["delta_from_v1_log2"] = (
        result["estimate_log2"] - result["v1_estimate_log2"]
    ).where(result["shared_v1_support"])
    return result


def render_figure(
    surfaces: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    primary_bandwidth: float,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 7.2))
    grid_spec = fig.add_gridspec(2, len(AREA_ORDER) + 1, width_ratios=[1] * len(AREA_ORDER) + [0.055])
    axes = np.empty((2, len(AREA_ORDER)), dtype=object)
    for row_index in range(2):
        for column_index in range(len(AREA_ORDER)):
            axes[row_index, column_index] = fig.add_subplot(
                grid_spec[row_index, column_index],
                sharex=axes[0, 0] if (row_index, column_index) != (0, 0) else None,
                sharey=axes[0, 0] if (row_index, column_index) != (0, 0) else None,
            )
    for row_index, (preference, specification) in enumerate(PREFERENCES.items()):
        log_values = np.log2(np.asarray(specification["values"], dtype=float))
        normalization = Normalize(vmin=log_values.min(), vmax=log_values.max())
        image_artist = None
        for column_index, area in enumerate(AREA_ORDER):
            ax = axes[row_index, column_index]
            selected = surfaces.loc[
                surfaces["preference"].eq(preference)
                & surfaces["area"].eq(area)
                & np.isclose(surfaces["bandwidth_deg"], primary_bandwidth)
            ].sort_values(["elevation_deg", "azimuth_deg"])
            matrix = selected["estimate_log2"].to_numpy().reshape(len(el_grid), len(az_grid))
            image_artist = ax.imshow(
                matrix,
                origin="lower",
                extent=[az_grid.min(), az_grid.max(), el_grid.min(), el_grid.max()],
                aspect="auto",
                cmap="viridis",
                norm=normalization,
                interpolation="bilinear",
            )
            if row_index == 0:
                ax.set_title(area)
            if column_index == 0:
                ax.set_ylabel(f"{preference.upper()}\nRF elevation (deg)")
            if row_index == 1:
                ax.set_xlabel("RF azimuth (deg)")
            ax.spines[["top", "right"]].set_visible(False)
        colorbar_axis = fig.add_subplot(grid_spec[row_index, -1])
        colorbar = fig.colorbar(image_artist, cax=colorbar_axis)
        colorbar.set_ticks(log_values)
        colorbar.set_ticklabels([f"{value:g}" for value in specification["values"]])
        colorbar.set_label(specification["label"])
    fig.suptitle(
        f"Allen nonlinear frequency-preference surfaces (session-balanced, bandwidth {primary_bandwidth:g}°)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.9, wspace=0.12, hspace=0.16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_difference_figure(
    surfaces: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    primary_bandwidth: float,
    output_path: Path,
) -> None:
    hva_order = AREA_ORDER[1:]
    fig = plt.figure(figsize=(15.5, 7.2))
    grid_spec = fig.add_gridspec(2, len(hva_order) + 1, width_ratios=[1] * len(hva_order) + [0.055])
    axes = np.empty((2, len(hva_order)), dtype=object)
    for row_index, preference in enumerate(PREFERENCES):
        selected_preference = surfaces.loc[
            surfaces["preference"].eq(preference)
            & np.isclose(surfaces["bandwidth_deg"], primary_bandwidth)
            & surfaces["area"].isin(hva_order)
        ]
        finite_delta = selected_preference["delta_from_v1_log2"].dropna().abs()
        limit = max(0.25, float(finite_delta.quantile(0.98)))
        normalization = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image_artist = None
        for column_index, area in enumerate(hva_order):
            axes[row_index, column_index] = fig.add_subplot(
                grid_spec[row_index, column_index],
                sharex=axes[0, 0] if (row_index, column_index) != (0, 0) else None,
                sharey=axes[0, 0] if (row_index, column_index) != (0, 0) else None,
            )
            ax = axes[row_index, column_index]
            selected = selected_preference.loc[selected_preference["area"].eq(area)].sort_values(
                ["elevation_deg", "azimuth_deg"]
            )
            matrix = selected["delta_from_v1_log2"].to_numpy().reshape(len(el_grid), len(az_grid))
            image_artist = ax.imshow(
                matrix,
                origin="lower",
                extent=[az_grid.min(), az_grid.max(), el_grid.min(), el_grid.max()],
                aspect="auto",
                cmap="RdBu_r",
                norm=normalization,
                interpolation="bilinear",
            )
            if row_index == 0:
                ax.set_title(area)
            if column_index == 0:
                ax.set_ylabel(f"{preference.upper()}\nRF elevation (deg)")
            if row_index == 1:
                ax.set_xlabel("RF azimuth (deg)")
            ax.spines[["top", "right"]].set_visible(False)
        colorbar_axis = fig.add_subplot(grid_spec[row_index, -1])
        colorbar = fig.colorbar(image_artist, cax=colorbar_axis)
        colorbar.set_label(f"{preference.upper()} preference difference from V1 (octaves)")
    fig.suptitle(
        f"Allen HVA minus V1 nonlinear preference surfaces (shared support, bandwidth {primary_bandwidth:g}°)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.065, right=0.93, bottom=0.09, top=0.9, wspace=0.12, hspace=0.16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_compact_surface_figure(
    surfaces: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    primary_bandwidth: float,
    output_path: Path,
    *,
    area: str,
    title_population: str,
) -> None:
    """Render only the two nonlinear preference fits for one population."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharex=True, sharey=True)
    color_maps = {"sf": "cividis", "tf": "magma"}
    for index, (preference, specification) in enumerate(PREFERENCES.items()):
        ax = axes[index]
        selected = surfaces.loc[
            surfaces["preference"].eq(preference)
            & surfaces["area"].eq(area)
            & np.isclose(surfaces["bandwidth_deg"], primary_bandwidth)
        ].sort_values(["elevation_deg", "azimuth_deg"])
        matrix = selected["estimate_log2"].to_numpy().reshape(len(el_grid), len(az_grid))
        finite_values = matrix[np.isfinite(matrix)]
        color_limits = np.quantile(finite_values, [0.02, 0.98])
        color_ticks = np.linspace(color_limits[0], color_limits[1], 5)
        image_artist = ax.imshow(
            matrix,
            origin="lower",
            extent=[az_grid.min(), az_grid.max(), el_grid.min(), el_grid.max()],
            aspect="auto",
            cmap=color_maps[preference],
            norm=Normalize(vmin=color_limits[0], vmax=color_limits[1]),
            interpolation="bilinear",
        )
        finite = np.isfinite(matrix)
        if finite.any() and np.nanmax(matrix) > np.nanmin(matrix):
            contour_levels = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 6)[1:-1]
            ax.contour(az_grid, el_grid, matrix, levels=contour_levels, colors="white", linewidths=0.65, alpha=0.65)
        ax.set_title(f"Preferred {preference.upper()}")
        ax.set_xlabel("RF azimuth (deg)")
        ax.set_ylabel("RF elevation (deg)")
        ax.spines[["top", "right"]].set_visible(False)
        colorbar = fig.colorbar(
            image_artist, ax=ax, fraction=0.046, pad=0.035, extend="both"
        )
        colorbar.set_ticks(color_ticks)
        if preference == "sf":
            colorbar.set_ticklabels([f"{np.exp2(value):.3f}" for value in color_ticks])
        else:
            colorbar.set_ticklabels([f"{np.exp2(value):.2f}" for value in color_ticks])
        colorbar.set_label(specification["label"])
    fig.suptitle(
        f"{title_population} frequency-preference surfaces (bandwidth {primary_bandwidth:g}°; fitted 2–98% color range)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.13, top=0.84, wspace=0.28)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_polar_surface_figure(
    surfaces: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    primary_bandwidth: float,
    output_path: Path,
    *,
    area: str,
    title_population: str,
    center_azimuth_deg: float = 0.0,
    center_elevation_deg: float = 0.0,
) -> None:
    """Render compact SF/TF fits in polar visual-field coordinates."""
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, eccentricity = polar_coordinates(
        az_mesh.ravel(),
        el_mesh.ravel(),
        center_azimuth_deg=center_azimuth_deg,
        center_elevation_deg=center_elevation_deg,
    )
    theta_limits = np.degrees(theta)
    radial_limit = np.ceil(np.max(eccentricity) / 10) * 10
    color_maps = {"sf": "cividis", "tf": "magma"}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.0),
        subplot_kw={"projection": "polar"},
    )
    for index, (preference, specification) in enumerate(PREFERENCES.items()):
        ax = axes[index]
        selected = surfaces.loc[
            surfaces["preference"].eq(preference)
            & surfaces["area"].eq(area)
            & np.isclose(surfaces["bandwidth_deg"], primary_bandwidth)
        ].sort_values(["elevation_deg", "azimuth_deg"])
        values = selected["estimate_log2"].to_numpy()
        finite_values = values[np.isfinite(values)]
        color_limits = np.quantile(finite_values, [0.02, 0.98])
        color_ticks = np.linspace(color_limits[0], color_limits[1], 5)
        theta_mesh = theta.reshape(len(el_grid), len(az_grid))
        eccentricity_mesh = eccentricity.reshape(len(el_grid), len(az_grid))
        value_mesh = values.reshape(len(el_grid), len(az_grid))
        image_artist = ax.pcolormesh(
            theta_mesh,
            eccentricity_mesh,
            value_mesh,
            shading="gouraud",
            cmap=color_maps[preference],
            norm=Normalize(vmin=color_limits[0], vmax=color_limits[1]),
        )
        contour_levels = np.linspace(finite_values.min(), finite_values.max(), 6)[1:-1]
        ax.contour(
            theta_mesh,
            eccentricity_mesh,
            value_mesh,
            levels=contour_levels,
            colors="white",
            linewidths=0.65,
            alpha=0.65,
        )
        ax.set_thetamin(np.floor(theta_limits.min() / 10) * 10)
        ax.set_thetamax(np.ceil(theta_limits.max() / 10) * 10)
        ax.set_ylim(0, radial_limit)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(65)
        ax.set_title(f"Preferred {preference.upper()}", pad=18)
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
        colorbar = fig.colorbar(
            image_artist, ax=ax, fraction=0.046, pad=0.09, extend="both"
        )
        colorbar.set_ticks(color_ticks)
        if preference == "sf":
            colorbar.set_ticklabels([f"{np.exp2(value):.3f}" for value in color_ticks])
        else:
            colorbar.set_ticklabels([f"{np.exp2(value):.2f}" for value in color_ticks])
        colorbar.set_label(specification["label"])
    fig.suptitle(
        f"{title_population} frequency-preference surfaces in polar RF coordinates\n"
        f"Allen origin = (0° azimuth, 0° elevation); bandwidth {primary_bandwidth:g}°",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.11, top=0.82, wspace=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_polar_occupancy_figure(
    units: pd.DataFrame,
    output_path: Path,
    *,
    center_azimuth_deg: float = 0.0,
    center_elevation_deg: float = 0.0,
) -> None:
    """Show RF-bin occupation for the exact SF- and TF-eligible populations."""
    azimuth_edges = np.arange(5.0, 96.0, 10.0)
    elevation_edges = np.arange(-35.0, 56.0, 10.0)
    az_mesh, el_mesh = np.meshgrid(azimuth_edges, elevation_edges)
    theta_edges, radius_edges = polar_coordinates(
        az_mesh,
        el_mesh,
        center_azimuth_deg=center_azimuth_deg,
        center_elevation_deg=center_elevation_deg,
    )
    theta_limits = np.degrees(theta_edges)
    radial_limit = np.ceil(np.max(radius_edges) / 10) * 10
    population_definitions = (
        ("V1", units["area"].eq("V1")),
        ("Pooled HVAs", units["area"].isin(HVA_ORDER)),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 10.2),
        subplot_kw={"projection": "polar"},
    )
    for row_index, preference in enumerate(PREFERENCES):
        count_matrices = []
        eligible = units[f"tuning_eligible_{preference}"]
        for _, population_mask in population_definitions:
            count_matrices.append(
                rf_occupancy_counts(
                    units.loc[eligible & population_mask],
                    azimuth_edges,
                    elevation_edges,
                )
            )
        maximum_count = max(int(matrix.max()) for matrix in count_matrices)
        image_artist = None
        for column_index, ((population_label, population_mask), counts) in enumerate(
            zip(population_definitions, count_matrices)
        ):
            ax = axes[row_index, column_index]
            masked_counts = np.ma.masked_equal(counts, 0)
            image_artist = ax.pcolormesh(
                theta_edges,
                radius_edges,
                masked_counts,
                shading="flat",
                cmap="inferno",
                norm=LogNorm(vmin=1, vmax=max(1, maximum_count)),
            )
            selected = units.loc[eligible & population_mask]
            ax.set_thetamin(np.floor(theta_limits.min() / 10) * 10)
            ax.set_thetamax(np.ceil(theta_limits.max() / 10) * 10)
            ax.set_ylim(0, radial_limit)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rlabel_position(65)
            ax.grid(color="#B5B5B5", linewidth=0.8)
            ax.set_title(
                f"{population_label}: {preference.upper()} population\n"
                f"{len(selected):,} units, {selected['ecephys_session_id'].nunique()} sessions",
                pad=18,
            )
            if row_index == 1:
                ax.set_xlabel(
                    "polar angle from (0° azimuth, 20° elevation)", labelpad=18
                )
        colorbar = fig.colorbar(
            image_artist,
            ax=axes[row_index, :],
            fraction=0.025,
            pad=0.055,
        )
        colorbar.set_label("units per 10° × 10° RF bin (log scale)")
    fig.suptitle(
        "Allen RF-location occupancy for frequency-preference populations",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        "polar center = Allen (0° azimuth, 0° elevation)",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.07, top=0.84, wspace=0.25, hspace=0.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    units: pd.DataFrame,
    summary: pd.DataFrame,
    bandwidths: tuple[float, ...],
    primary_bandwidth: float,
    minimum_lifetime_sparseness: float | None,
    minimum_stimulus_firing_rate: float | None,
    require_unique_preference: bool,
    output_path: Path,
) -> None:
    primary = summary.loc[np.isclose(summary["bandwidth_deg"], primary_bandwidth)]
    lines = [
        "# Allen SF/TF preference surfaces over receptive-field position",
        "",
        "## Status: nonlinear preference surfaces implemented",
        "",
        "These are surfaces of the released per-unit preferred bins, not full",
        "response-amplitude tuning curves. Only Brain Observatory sessions are used:",
        "Functional Connectivity presented a single 2-Hz drifting-grating condition",
        "and therefore cannot identify temporal-frequency preference.",
        "",
        "The primary surface uses a session-balanced Gaussian kernel with a",
        f"{primary_bandwidth:g}° bandwidth. Sensitivities use "
        + ", ".join(f"{value:g}°" for value in bandwidths)
        + ". Grid cells require at least three effective sessions and 20 nearby units.",
        "Preferences are smoothed on a log2 scale so adjacent octave steps are equally spaced.",
        "",
        "Tuning-quality filters: "
        + (
            f"lifetime sparseness > {minimum_lifetime_sparseness:g}"
            if minimum_lifetime_sparseness is not None
            else "no lifetime-sparseness threshold"
        )
        + ", "
        + (
            f"stimulus firing rate > {minimum_stimulus_firing_rate:g} Hz"
            if minimum_stimulus_firing_rate is not None
            else "no additional stimulus firing-rate threshold"
        )
        + (", unique preferred bin required." if require_unique_preference else ", tied preferred bins retained."),
        "",
        "## Primary surface coverage",
        "",
        "| Preference | Area | Units | Sessions | Supported grid | Surface median | Surface 10–90% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in primary.sort_values(["preference", "area"]).iterrows():
        lines.append(
            f"| {row.preference.upper()} | {row.area} | {int(row.source_units):,} | "
            f"{int(row.source_sessions)} | {row.supported_grid_fraction:.1%} | "
            f"{row.surface_median_preference:.3g} | "
            f"{row.surface_p10_preference:.3g}–{row.surface_p90_preference:.3g} |"
        )
    lines.extend(
        [
            "",
            "## HVA differences from paired-session V1 surfaces",
            "",
            "Positive values indicate a higher preferred frequency than V1 at the",
            "same RF coordinate; one log2 unit is one octave.",
            "",
            "| Preference | Area | Shared grid | Median difference (octaves) | Spatial 10–90% |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in primary.loc[~primary["area"].eq("V1")].sort_values(
        ["preference", "area"]
    ).iterrows():
        lines.append(
            f"| {row.preference.upper()} | {row.area} | {row.shared_v1_grid_fraction:.1%} | "
            f"{row.median_delta_from_v1_log2:+.3f} | "
            f"{row.p10_delta_from_v1_log2:+.3f} to {row.p90_delta_from_v1_log2:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The surfaces describe how preferred bins vary over achieved RF position.",
            "They do not measure tuning bandwidth, response strength, or a joint Allen",
            "SF × TF response surface because Allen measured SF with static gratings and",
            "TF with drifting gratings. Each HVA is compared with a V1 surface built",
            "only from the same Allen sessions. `delta_from_v1_log2` is defined only",
            "where both that HVA and its paired-session V1 reference meet the local support rule.",
            "",
            "## Outputs",
            "",
            "- `frequency_preference_surface_grid.csv`: all bandwidths, support diagnostics, and V1 differences.",
            "- `frequency_preference_surface_summary.csv`: area-level coverage and spatial ranges.",
            "- `frequency_preference_population.csv`: source-unit counts by area and preference.",
            "- `Figure_allen_frequency_preference_surfaces.png`: primary SF and TF surfaces.",
            "- `Figure_allen_frequency_preference_differences.png`: HVA-minus-V1 surfaces on shared support.",
            "- `Figure_allen_pooled_hva_frequency_preference_surfaces.png`: two-panel pooled-HVA SF/TF fits.",
            "- `Figure_allen_v1_frequency_preference_surfaces.png`: matched two-panel Allen V1 SF/TF fits.",
            "- `Figure_allen_pooled_hva_frequency_preference_surfaces_polar.png`: pooled-HVA fits in polar RF coordinates.",
            "- `Figure_allen_v1_frequency_preference_surfaces_polar.png`: Allen V1 fits in polar RF coordinates.",
            "- `Figure_allen_rf_occupancy_polar.png`: exact SF-/TF-population RF occupation for V1 and pooled HVAs.",
            "- `run_manifest.json`: input, code, parameters, and output checksums.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bandwidths = validate_bandwidths(args.bandwidths_deg, args.primary_bandwidth_deg)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    units, unit_path, audit_manifest = load_preference_units(
        args.audit_dir.resolve(),
        args.config,
        minimum_lifetime_sparseness=args.minimum_lifetime_sparseness,
        minimum_stimulus_firing_rate=args.minimum_stimulus_firing_rate,
        require_unique_preference=args.require_unique_preference,
    )
    az_grid, el_grid, grid_points = build_grid(units, args.grid_size)
    surfaces = estimate_surfaces(
        units,
        grid_points,
        bandwidths,
        minimum_effective_sessions=args.minimum_effective_sessions,
        minimum_local_units=args.minimum_local_units,
    )
    surfaces = add_v1_differences(surfaces)
    summary = summarize_surfaces(surfaces)
    population_frames = []
    for preference, specification in PREFERENCES.items():
        eligible = units.loc[units[f"tuning_eligible_{preference}"]].copy()
        frame = (
            eligible.groupby(["area", specification["column"]], observed=True)
            .agg(
                units=("ecephys_unit_id", "size"),
                sessions=("ecephys_session_id", "nunique"),
            )
            .reset_index()
            .rename(columns={specification["column"]: "preference_value"})
        )
        frame.insert(0, "preference", preference)
        population_frames.append(frame)
    population = pd.concat(population_frames, ignore_index=True)

    surfaces.to_csv(output_dir / "frequency_preference_surface_grid.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "frequency_preference_surface_summary.csv", index=False, float_format="%.6g")
    population.to_csv(output_dir / "frequency_preference_population.csv", index=False)
    render_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_frequency_preference_surfaces.png",
    )
    render_difference_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_frequency_preference_differences.png",
    )
    render_compact_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_pooled_hva_frequency_preference_surfaces.png",
        area=POOLED_HVA,
        title_population="Pooled Allen HVA",
    )
    render_compact_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_v1_frequency_preference_surfaces.png",
        area="V1",
        title_population="Allen V1",
    )
    render_polar_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_pooled_hva_frequency_preference_surfaces_polar.png",
        area=POOLED_HVA,
        title_population="Pooled Allen HVA",
    )
    render_polar_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_allen_v1_frequency_preference_surfaces_polar.png",
        area="V1",
        title_population="Allen V1",
    )
    render_polar_occupancy_figure(
        units,
        output_dir / "Figure_allen_rf_occupancy_polar.png",
    )
    write_report(
        units,
        summary,
        bandwidths,
        args.primary_bandwidth_deg,
        args.minimum_lifetime_sparseness,
        args.minimum_stimulus_firing_rate,
        args.require_unique_preference,
        output_dir / "ALLEN_FREQUENCY_PREFERENCE_SURFACES.md",
    )

    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_frequency_preference_surfaces",
        "status": "nonlinear SF/TF preference surfaces implemented",
        "input": {"path": str(unit_path), "sha256": sha256(unit_path)},
        "audit_manifest_sha256": sha256(args.audit_dir.resolve() / "run_manifest.json"),
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "cohort": BO_COHORT,
            "population": audit_manifest["parameters"]["population"],
            "bandwidths_deg": list(bandwidths),
            "primary_bandwidth_deg": args.primary_bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_effective_sessions": args.minimum_effective_sessions,
            "minimum_local_units": args.minimum_local_units,
            "session_balanced": True,
            "preference_scale": "log2",
            "minimum_lifetime_sparseness": args.minimum_lifetime_sparseness,
            "minimum_stimulus_firing_rate_hz": args.minimum_stimulus_firing_rate,
            "require_unique_preference": args.require_unique_preference,
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Allen frequency-preference surfaces written to {output_dir}")


if __name__ == "__main__":
    main()
