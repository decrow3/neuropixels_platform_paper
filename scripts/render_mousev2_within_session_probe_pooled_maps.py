#!/usr/bin/env python3
"""Pool simultaneous MouseV2 probes within session and render RF/SF/TF maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import (  # noqa: E402
    polar_coordinates,
    session_balanced_gaussian_surface,
)
from scripts.mousev2_frequency_preference_surfaces import (  # noqa: E402
    DEFAULT_GRATINGS,
    DEFAULT_RF,
    DEFAULT_TUNING,
    PREFERENCES,
    load_mousev2_units,
)
from scripts.render_mousev2_simultaneous_probe_maps import (  # noqa: E402
    complete_simultaneous_sessions,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06d_mousev2_frequency_preference_surfaces"
    / "within_session_probe_pooled_maps"
)
MAP_ORDER = ("rf_density", "sf", "tf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-table", type=Path, default=DEFAULT_RF)
    parser.add_argument("--grating-dir", type=Path, default=DEFAULT_GRATINGS)
    parser.add_argument("--tuning-support", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-local-units", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within_session_rf_density(
    group: pd.DataFrame,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_local_units: int,
) -> dict[str, np.ndarray]:
    """Gaussian RF density for one session, with all four probes pooled."""
    points = group[["azimuth_rf", "elevation_rf"]].to_numpy(float)
    distance_squared = np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * distance_squared / bandwidth_deg**2)
    density = kernel.mean(axis=1)
    local_units = (distance_squared <= (1.5 * bandwidth_deg) ** 2).sum(axis=1)
    supported = local_units >= minimum_local_units
    density[~supported] = np.nan
    return {"density": density, "local_units": local_units, "supported": supported}


def estimate_within_session_maps(
    units: pd.DataFrame,
    sessions: list[str],
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for site in sessions:
        session = units.loc[units["site"].eq(site)]
        subject_id = session["subject_id"].dropna().iloc[0]
        rf_group = session.loc[session["analysis_eligible"]]
        rf = within_session_rf_density(
            rf_group,
            grid_points,
            bandwidth_deg=bandwidth_deg,
            minimum_local_units=minimum_local_units,
        )
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "subject_id": subject_id,
                    "map": "rf_density",
                    "azimuth_deg": grid_points[:, 0],
                    "elevation_deg": grid_points[:, 1],
                    "estimate": rf["density"],
                    "estimate_log2": np.nan,
                    "local_units": rf["local_units"],
                    "supported": rf["supported"],
                    "source_units": len(rf_group),
                    "source_probes": rf_group["probe"].nunique(),
                }
            )
        )
        for preference, specification in PREFERENCES.items():
            group = session.loc[session[f"tuning_eligible_{preference}"]]
            surface = session_balanced_gaussian_surface(
                group[["azimuth_rf", "elevation_rf"]].to_numpy(float),
                np.log2(group[specification["column"]].to_numpy(float)),
                group["site"].to_numpy(),
                grid_points,
                bandwidth_deg=bandwidth_deg,
                minimum_effective_sessions=1.0,
                minimum_local_units=minimum_local_units,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "subject_id": subject_id,
                        "map": preference,
                        "azimuth_deg": grid_points[:, 0],
                        "elevation_deg": grid_points[:, 1],
                        "estimate": np.exp2(surface["estimate_log2"]),
                        "estimate_log2": surface["estimate_log2"],
                        "local_units": surface["local_units"],
                        "supported": surface["supported"],
                        "source_units": len(group),
                        "source_probes": group["probe"].nunique(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def render_maps(
    maps: pd.DataFrame,
    sessions: list[str],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
    *,
    bandwidth_deg: float,
) -> None:
    if len(sessions) != 8:
        raise ValueError("The compact overview currently expects eight sessions")
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(
        az_mesh, el_mesh, center_azimuth_deg=50.0, center_elevation_deg=10.0
    )
    theta_degrees = np.degrees(theta)
    radial_limit = np.ceil(np.nanmax(radius) / 10) * 10
    labels = {
        "rf_density": "RF density",
        "sf": "Preferred SF",
        "tf": "Preferred TF",
    }
    colorbar_labels = {
        "rf_density": "RF density (within-session a.u.)",
        "sf": "Preferred SF (cycles/deg)",
        "tf": "Preferred TF (Hz)",
    }
    cmaps = {"rf_density": "magma", "sf": "viridis", "tf": "plasma"}
    norms: dict[str, Normalize] = {}
    for map_name in MAP_ORDER:
        finite = maps.loc[maps["map"].eq(map_name), "estimate"].dropna().to_numpy(float)
        limits = np.quantile(finite, [0.02, 0.98])
        norms[map_name] = Normalize(vmin=limits[0], vmax=limits[1])

    fig, axes = plt.subplots(
        4,
        6,
        figsize=(19.5, 13.8),
        subplot_kw={"projection": "polar"},
    )
    artists: dict[str, object] = {}
    for session_index, site in enumerate(sessions):
        row = session_index // 2
        group_start = (session_index % 2) * 3
        for metric_index, map_name in enumerate(MAP_ORDER):
            column = group_start + metric_index
            ax = axes[row, column]
            selected = maps.loc[
                maps["site"].eq(site) & maps["map"].eq(map_name)
            ].sort_values(["elevation_deg", "azimuth_deg"])
            values = selected["estimate"].to_numpy().reshape(len(el_grid), len(az_grid))
            artist = ax.pcolormesh(
                theta,
                radius,
                values,
                shading="gouraud",
                cmap=cmaps[map_name],
                norm=norms[map_name],
            )
            artists[map_name] = artist
            ax.set_thetamin(np.floor(theta_degrees.min() / 10) * 10)
            ax.set_thetamax(np.ceil(theta_degrees.max() / 10) * 10)
            ax.set_ylim(0, radial_limit)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_rlabel_position(65)
            ax.grid(color="#B5B5B5", linewidth=0.6)
            subject = int(selected["subject_id"].iloc[0])
            units = int(selected["source_units"].iloc[0])
            support = selected["supported"].mean()
            ax.set_title(
                f"{site} · mouse {subject}\n{labels[map_name]}",
                fontsize=9,
                pad=10,
            )
            ax.text(
                0.56,
                0.035,
                f"n={units:,}; grid={support:.0%}",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
            )

    # One shared scale for each metric; place them below the figure to avoid
    # squeezing individual polar panels.
    positions = ((0.15, 0.025), (0.425, 0.025), (0.70, 0.025))
    for map_name, (left, bottom) in zip(MAP_ORDER, positions):
        colorbar_axis = fig.add_axes([left, bottom, 0.18, 0.012])
        colorbar = fig.colorbar(artists[map_name], cax=colorbar_axis, orientation="horizontal", extend="both")
        colorbar.set_label(colorbar_labels[map_name], fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "MouseV2 maps pooled across four simultaneous V1 probes within each session\n"
        f"original display coordinates; no cross-session alignment; bandwidth {bandwidth_deg:g}°\n"
        "independent RF/SF/TF populations; polar center = MouseV2 display center (50°, 10°)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.075, top=0.885, wspace=0.25, hspace=0.34)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(maps: pd.DataFrame, sessions: list[str], output_path: Path) -> None:
    summary = (
        maps.groupby(["site", "subject_id", "map"], observed=True)
        .agg(
            source_units=("source_units", "first"),
            source_probes=("source_probes", "first"),
            supported_grid_fraction=("supported", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# MouseV2 maps pooled across simultaneous probes within session",
        "",
        f"The {len(sessions)} complete A/B/C/E sessions are shown separately. Within each",
        "session, units from all four simultaneous probes are pooled before smoothing.",
        "Original display-centered RF coordinates are retained: no session translation or",
        "cross-session alignment is applied. Consequently, each row preserves the screen",
        "and unmeasured eye-position state shared by that session's probes.",
        "",
        "RF density uses supported RF units. SF and TF use their independently supported",
        "tuning populations, so the three maps do not contain identical unit sets.",
        "",
        "| Session | Mouse | Map | Units | Probes | Supported grid |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.site} | {int(row.subject_id)} | {row['map']} | "
            f"{int(row.source_units):,} | {int(row.source_probes)} | "
            f"{row.supported_grid_fraction:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is the direct within-session intermediate view. It controls design-wise",
            "for session-level screen geometry and shared recording state, but it does not",
            "measure gaze or prove that the eyes were stationary. Differences between session",
            "maps may reflect gaze, targeting, biological variation, or finite sampling.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    units, grating_paths = load_mousev2_units(
        args.rf_table.resolve(),
        args.grating_dir.resolve(),
        qc_profile="pilot_qc",
        require_unique_preference=True,
        tuning_support_path=args.tuning_support.resolve(),
    )
    sessions = complete_simultaneous_sessions(units)
    if not sessions:
        raise ValueError("No complete simultaneous A/B/C/E sessions")
    units = units.loc[units["site"].isin(sessions)].copy()
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = estimate_within_session_maps(
        units,
        sessions,
        grid_points,
        bandwidth_deg=args.bandwidth_deg,
        minimum_local_units=args.minimum_local_units,
    )
    figure_path = output_dir / "Figure_mousev2_within_session_probe_pooled_rf_sf_tf_polar.png"
    render_maps(
        maps,
        sessions,
        az_grid,
        el_grid,
        figure_path,
        bandwidth_deg=args.bandwidth_deg,
    )
    maps.to_csv(output_dir / "within_session_probe_pooled_surface_grid.csv", index=False, float_format="%.6g")
    write_report(
        maps,
        sessions,
        output_dir / "MOUSEV2_WITHIN_SESSION_PROBE_POOLED_MAPS.md",
    )
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06d_mousev2_within_session_probe_pooled_maps",
        "status": "complete-quartet probes pooled within original session coordinates",
        "inputs": {
            "rf_table": {"path": str(args.rf_table.resolve()), "sha256": sha256(args.rf_table.resolve())},
            "grating_tables": [
                {"path": str(path), "sha256": sha256(path)} for path in grating_paths
            ],
            "tuning_support": {
                "path": str(args.tuning_support.resolve()),
                "sha256": sha256(args.tuning_support.resolve()),
            },
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "complete_sessions": sessions,
            "probe_pooling": "A/B/C/E pooled within session",
            "bandwidth_deg": args.bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_local_units": args.minimum_local_units,
            "coordinate_alignment": "none; original display coordinates",
            "gaze_correction": "none",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MouseV2 within-session probe-pooled maps written to {output_dir}")


if __name__ == "__main__":
    main()
