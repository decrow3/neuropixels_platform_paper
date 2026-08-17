#!/usr/bin/env python3
"""Plot RF size over RF location for the Allen non-center registration pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from scripts.allen_bo11_noncenter_similarity_alignment import (
    AUDIT,
    BO_COHORT,
    DEFAULT_SUPPORT,
    DEFAULT_UNITS,
    robust_area_standardize,
)
from scripts.allen_bo11_tuning_weighted_session_surfaces import weighted_gaussian_surface
from scripts.render_allen_bo11_simultaneous_v1_hva_session_maps import group_definitions
from scipy.interpolate import RegularGridInterpolator
from scripts.allen_bo11_tuning_driven_limited_affine import (
    CENTER_DEG,
    affine_matrix,
    polar_coordinates,
    template_from_maps,
    warp_all,
)
from scripts.render_allen_bo11_registration_comparison import (
    DEFAULT_NONCENTER_TRANSFORMS,
    load_noncenter_parameters,
)


DEFAULT_OUTPUT = AUDIT / "noncenter_similarity_alignment" / "rf_size_evidence"
GROUPS = ("V1", "HVA pooled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--noncenter-transforms", type=Path, default=DEFAULT_NONCENTER_TRANSFORMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-effective-local-units", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_aligned_coordinates(
    population: pd.DataFrame,
    parameters: dict[int, np.ndarray],
) -> pd.DataFrame:
    result = population.copy()
    aligned = np.full((len(result), 2), np.nan)
    for session_id, indices in result.groupby("ecephys_session_id", observed=True).groups.items():
        matrix, translation = affine_matrix(parameters[int(session_id)])
        points = result.loc[indices, ["azimuth_rf", "elevation_rf"]].to_numpy(float)
        transformed = (points - CENTER_DEG) @ matrix.T + CENTER_DEG + translation
        aligned[result.index.get_indexer(indices)] = transformed
    result["aligned_azimuth_rf"] = aligned[:, 0]
    result["aligned_elevation_rf"] = aligned[:, 1]
    result["raw_eccentricity_deg"] = np.hypot(
        result["azimuth_rf"], result["elevation_rf"]
    )
    result["aligned_eccentricity_deg"] = np.hypot(
        result["aligned_azimuth_rf"], result["aligned_elevation_rf"],
    )
    return result


def session_balanced_radial_summary(
    population: pd.DataFrame,
    group: str,
    coordinate: str,
    bins: np.ndarray,
) -> pd.DataFrame:
    mask = population["area"].eq("V1") if group == "V1" else population["area"].ne("V1")
    selected = population.loc[mask, ["ecephys_session_id", coordinate, "standardized_area_rf"]].dropna()
    selected["bin"] = pd.cut(selected[coordinate], bins=bins, include_lowest=True)
    session_bins = (
        selected.groupby(["ecephys_session_id", "bin"], observed=True)["standardized_area_rf"]
        .median()
        .reset_index()
    )
    summary = (
        session_bins.groupby("bin", observed=True)["standardized_area_rf"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            sessions="size",
        )
        .reset_index()
    )
    summary["eccentricity_deg"] = summary["bin"].map(lambda interval: interval.mid).astype(float)
    summary["registration"] = "aligned" if coordinate.startswith("aligned") else "raw"
    summary["group"] = group
    return summary


def build_rf_area_maps(
    population: pd.DataFrame,
    sessions: list[int],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_local_units: float,
) -> tuple[dict[tuple[int, str, str], dict[str, np.ndarray]], pd.DataFrame]:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = {}
    audit = []
    for session_id in sessions:
        session = population.loc[population["ecephys_session_id"].eq(session_id)]
        for group, mask in group_definitions(session):
            if group not in GROUPS:
                continue
            selected = session.loc[mask].dropna(
                subset=["standardized_area_rf", "azimuth_rf", "elevation_rf"]
            )
            surface = weighted_gaussian_surface(
                selected[["azimuth_rf", "elevation_rf"]].to_numpy(float),
                selected["standardized_area_rf"].to_numpy(float),
                np.ones(len(selected)),
                grid_points,
                bandwidth_deg=bandwidth_deg,
                minimum_effective_local_units=minimum_effective_local_units,
            )
            value = surface["estimate_log2"].reshape(len(el_grid), len(az_grid))
            effective = surface["effective_local_units"].reshape(len(el_grid), len(az_grid))
            supported = surface["supported"].reshape(len(el_grid), len(az_grid))
            evidence = np.where(supported & np.isfinite(value), np.sqrt(np.maximum(effective, 0)), 0.0)
            finite_value = np.where(np.isfinite(value), value, 0.0)
            maps[(session_id, group, "area_rf")] = {
                "value": value,
                "evidence": evidence,
                "source_units": len(selected),
                "interpolate_evidence": RegularGridInterpolator(
                    (el_grid, az_grid), evidence, bounds_error=False, fill_value=0.0
                ),
                "interpolate_numerator": RegularGridInterpolator(
                    (el_grid, az_grid), finite_value * evidence, bounds_error=False, fill_value=0.0
                ),
            }
            audit.append(
                {
                    "ecephys_session_id": session_id,
                    "group": group,
                    "feature": "area_rf",
                    "source_units": len(selected),
                    "supported_grid_fraction": float(supported.mean()),
                }
            )
    return maps, pd.DataFrame(audit)


def render_figure(
    raw_templates: dict[str, dict[str, np.ndarray]],
    aligned_templates: dict[str, dict[str, np.ndarray]],
    radial: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    source_audit: pd.DataFrame,
    selected_model: str,
    output_path: Path,
) -> None:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(
        az_mesh, el_mesh, center_azimuth_deg=0.0, center_elevation_deg=0.0
    )
    all_values = np.concatenate(
        [
            templates[group]["value"][np.isfinite(templates[group]["value"])]
            for templates in (raw_templates, aligned_templates)
            for group in GROUPS
        ]
    )
    limit = float(np.quantile(np.abs(all_values), 0.98))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig = plt.figure(figsize=(17.2, 9.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], hspace=0.42, wspace=0.30)
    artists = []
    for row, group in enumerate(GROUPS):
        for column, (label, templates) in enumerate(
            (("Raw coordinates", raw_templates), (f"Non-center {selected_model}", aligned_templates))
        ):
            ax = fig.add_subplot(grid[row, column], projection="polar")
            artist = ax.pcolormesh(
                theta,
                radius,
                templates[group]["value"],
                shading="gouraud",
                cmap="coolwarm",
                norm=norm,
            )
            artists.append(artist)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_ylim(0, 105)
            ax.set_rlabel_position(65)
            ax.grid(color="#B5B5B5", linewidth=0.65)
            audit = source_audit.loc[
                source_audit["group"].eq(group) & source_audit["feature"].eq("area_rf")
            ]
            ax.set_title(
                f"{group} · {label}\n"
                f"median n/session={audit.source_units.median():.0f}; "
                f"supported grid={audit.supported_grid_fraction.median():.0%}",
                fontsize=10,
            )
        ax = fig.add_subplot(grid[row, 2])
        selected = radial.loc[radial["group"].eq(group)]
        styles = {
            "raw": ("Raw coordinates", "#4C78A8", "o"),
            "aligned": (f"Non-center {selected_model}", "#F58518", "s"),
        }
        for registration, (label, color, marker) in styles.items():
            local = selected.loc[selected["registration"].eq(registration)].sort_values("eccentricity_deg")
            supported = local["sessions"].ge(10)
            local = local.loc[supported]
            ax.fill_between(
                local["eccentricity_deg"], local["q25"], local["q75"], color=color, alpha=0.16
            )
            ax.plot(
                local["eccentricity_deg"], local["median"], color=color, marker=marker,
                linewidth=1.8, markersize=5, label=label,
            )
        ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xlim(0, 105)
        ax.set_xlabel("RF eccentricity from Allen (0°, 0°) (deg)")
        ax.set_ylabel("log₂ RF area, standardized within area (IQR units)")
        ax.set_title(f"{group} · session-balanced radial trend")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    colorbar_axis = fig.add_axes([0.625, 0.28, 0.012, 0.44])
    colorbar = fig.colorbar(artists[0], cax=colorbar_axis, extend="both")
    colorbar.set_label("standardized log₂ RF area (IQR units)")
    fig.suptitle(
        "RF size over RF location: evidence used by the non-center registration\n"
        "maps are session/evidence weighted; bands show across-session IQR of bin medians",
        fontsize=14,
        y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.07, top=0.86)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_table = pd.read_csv(args.noncenter_transforms.resolve())
    sessions = sorted(transform_table["ecephys_session_id"].astype(int).unique())
    parameters, selected_model = load_noncenter_parameters(args.noncenter_transforms.resolve(), sessions)
    support = pd.read_csv(args.support.resolve(), low_memory=False)
    support = support.loc[
        support["cohort"].eq(BO_COHORT) & support["ecephys_session_id"].isin(sessions)
    ].copy()
    if "area_rf" not in support:
        metrics = pd.read_csv(args.unit_table.resolve(), usecols=["ecephys_unit_id", "area_rf"])
        support = support.merge(metrics, on="ecephys_unit_id", how="left", validate="one_to_one")
    support["standardized_area_rf"] = robust_area_standardize(support, "area_rf", "log2")
    aligned_population = add_aligned_coordinates(support, parameters)
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    feature_maps, surface_audit = build_rf_area_maps(
        support.copy(), sessions, az_grid, el_grid,
        bandwidth_deg=args.bandwidth_deg,
        minimum_effective_local_units=args.minimum_effective_local_units,
    )
    area_maps = {key: value for key, value in feature_maps.items() if key[2] == "area_rf"}
    raw_templates = {
        group: template_from_maps(area_maps, group, "area_rf") for group in GROUPS
    }
    aligned_maps = warp_all(area_maps, parameters, az_grid, el_grid)
    aligned_templates = {
        group: template_from_maps(aligned_maps, group, "area_rf") for group in GROUPS
    }
    bins = np.arange(0.0, 111.0, 10.0)
    radial = pd.concat(
        [
            session_balanced_radial_summary(aligned_population, group, coordinate, bins)
            for group in GROUPS
            for coordinate in ("raw_eccentricity_deg", "aligned_eccentricity_deg")
        ],
        ignore_index=True,
    )
    radial.to_csv(output_dir / "rf_size_radial_session_summary.csv", index=False, float_format="%.6g")
    surface_audit.loc[surface_audit["feature"].eq("area_rf")].to_csv(
        output_dir / "rf_size_surface_support.csv", index=False, float_format="%.6g"
    )
    figure_path = output_dir / "Figure_allen_bo11_rf_size_by_rf_location.png"
    render_figure(
        raw_templates, aligned_templates, radial, az_grid, el_grid,
        surface_audit, selected_model, figure_path,
    )
    report = [
        "# RF size evidence for the Allen BO 1.1 non-center registration",
        "",
        "The plotted quantity is log2 released RF area, robustly standardized within each",
        "anatomical area before smoothing. This is exactly the RF-size field used by the",
        f"selected non-center **{selected_model}** model. Raw RF area is not a Gaussian width",
        "estimate; released `width_rf` and `height_rf` were excluded because their scales are",
        "not reliable as a size/shape decomposition.",
        "",
        "The maps compare raw coordinates with the selected transform. The radial panel first",
        "takes the median within session × eccentricity bin, then displays the median and IQR",
        "across sessions; bins with fewer than ten sessions are omitted.",
        "",
        "RF area is weakly associated with Allen-origin eccentricity and is not sufficient to identify",
        "a two-dimensional translation. It was combined with directional CCF/probe fields and",
        "weak latency regularizers in the fourth-row fit.",
        "",
        "The radial relationship is non-monotonic and weak: size varies through intermediate",
        "eccentricities and falls at the mapped periphery. The peripheral decline may partly reflect RFs",
        "being truncated by the finite mapping display, causing released area to be underestimated.",
        "RF area should therefore be treated as a reproducible scalar pattern, not an uncensored",
        "biological size-versus-eccentricity calibration.",
    ]
    (output_dir / "ALLEN_BO11_RF_SIZE_BY_RF_LOCATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_rf_size_evidence",
        "status": "diagnostic for fourth-row non-center registration",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "transforms": {"path": str(args.noncenter_transforms.resolve()), "sha256": sha256(args.noncenter_transforms.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": [int(session_id) for session_id in sessions],
            "selected_model": selected_model,
            "bandwidth_deg": args.bandwidth_deg,
            "minimum_effective_local_units": args.minimum_effective_local_units,
            "rf_area_transform": "log2, median centered and IQR scaled within anatomical area",
            "radial_center_deg": [0.0, 0.0],
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen RF-size evidence written to {output_dir}")


if __name__ == "__main__":
    main()
