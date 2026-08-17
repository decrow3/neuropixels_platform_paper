#!/usr/bin/env python3
"""Render simultaneous Allen BO 1.1 V1-versus-HVA RF/SF/TF maps by session."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import (  # noqa: E402
    DEFAULT_AUDIT,
    HVA_ORDER,
    PREFERENCES,
    load_preference_units,
    polar_coordinates,
    session_balanced_gaussian_surface,
)
from scripts.render_mousev2_within_session_probe_pooled_maps import (  # noqa: E402
    within_session_rf_density,
)


DEFAULT_OUTPUT = DEFAULT_AUDIT / "simultaneous_v1_hva_session_maps"
MAP_ORDER = ("rf_density", "sf", "tf")
PRIMARY_GROUPS = ("V1", "HVA pooled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-local-units", type=int, default=10)
    parser.add_argument("--minimum-lifetime-sparseness", type=float, default=0.1)
    parser.add_argument("--minimum-stimulus-firing-rate", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simultaneous_v1_hva_sessions(units: pd.DataFrame) -> list[int]:
    """Require V1 and at least one HVA for RF, SF, and TF populations."""
    retained = []
    for session_id, session in units.groupby("ecephys_session_id", observed=True):
        valid = True
        for flag in (None, "tuning_eligible_sf", "tuning_eligible_tf"):
            selected = session if flag is None else session.loc[session[flag]]
            valid &= selected["area"].eq("V1").any()
            valid &= selected["area"].isin(HVA_ORDER).any()
        if valid:
            retained.append(int(session_id))
    return sorted(retained)


def group_definitions(session: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    definitions = [
        ("V1", session["area"].eq("V1")),
        ("HVA pooled", session["area"].isin(HVA_ORDER)),
    ]
    definitions.extend(
        (area, session["area"].eq(area))
        for area in HVA_ORDER
        if session["area"].eq(area).any()
    )
    return definitions


def estimate_maps(
    units: pd.DataFrame,
    sessions: list[int],
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for session_id in sessions:
        session = units.loc[units["ecephys_session_id"].eq(session_id)]
        specimen_id = int(session["specimen_id"].iloc[0])
        for group_label, group_mask in group_definitions(session):
            rf_group = session.loc[group_mask]
            rf = within_session_rf_density(
                rf_group,
                grid_points,
                bandwidth_deg=bandwidth_deg,
                minimum_local_units=minimum_local_units,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "ecephys_session_id": session_id,
                        "specimen_id": specimen_id,
                        "group": group_label,
                        "map": "rf_density",
                        "azimuth_deg": grid_points[:, 0],
                        "elevation_deg": grid_points[:, 1],
                        "estimate": rf["density"],
                        "estimate_log2": np.nan,
                        "local_units": rf["local_units"],
                        "supported": rf["supported"],
                        "source_units": len(rf_group),
                        "source_probes": rf_group["ecephys_probe_id"].nunique(),
                        "source_areas": "+".join(sorted(rf_group["area"].unique())),
                    }
                )
            )
            for preference, specification in PREFERENCES.items():
                group = session.loc[group_mask & session[f"tuning_eligible_{preference}"]]
                surface = session_balanced_gaussian_surface(
                    group[["azimuth_rf", "elevation_rf"]].to_numpy(float),
                    np.log2(group[specification["column"]].to_numpy(float)),
                    group["ecephys_session_id"].to_numpy(),
                    grid_points,
                    bandwidth_deg=bandwidth_deg,
                    minimum_effective_sessions=1.0,
                    minimum_local_units=minimum_local_units,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "ecephys_session_id": session_id,
                            "specimen_id": specimen_id,
                            "group": group_label,
                            "map": preference,
                            "azimuth_deg": grid_points[:, 0],
                            "elevation_deg": grid_points[:, 1],
                            "estimate": np.exp2(surface["estimate_log2"]),
                            "estimate_log2": surface["estimate_log2"],
                            "local_units": surface["local_units"],
                            "supported": surface["supported"],
                            "source_units": len(group),
                            "source_probes": group["ecephys_probe_id"].nunique(),
                            "source_areas": "+".join(sorted(group["area"].unique())),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def global_norms(maps: pd.DataFrame) -> dict[str, Normalize]:
    norms = {}
    primary = maps.loc[maps["group"].isin(PRIMARY_GROUPS)]
    for map_name in MAP_ORDER:
        finite = primary.loc[primary["map"].eq(map_name), "estimate"].dropna().to_numpy(float)
        limits = np.quantile(finite, [0.02, 0.98])
        norms[map_name] = Normalize(vmin=limits[0], vmax=limits[1])
    return norms


def render_session_figure(
    maps: pd.DataFrame,
    session_id: int,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    norms: dict[str, Normalize],
    *,
    bandwidth_deg: float,
) -> plt.Figure:
    selected_session = maps.loc[maps["ecephys_session_id"].eq(session_id)]
    specimen_id = int(selected_session["specimen_id"].iloc[0])
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(az_mesh, el_mesh)
    theta_degrees = np.degrees(theta)
    radial_limit = np.ceil(np.nanmax(radius) / 10) * 10
    labels = {"rf_density": "RF density", "sf": "Preferred SF", "tf": "Preferred TF"}
    units_labels = {
        "rf_density": "RF density (within-session a.u.)",
        "sf": "Preferred SF (cycles/deg)",
        "tf": "Preferred TF (Hz)",
    }
    cmaps = {"rf_density": "magma", "sf": "viridis", "tf": "plasma"}
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.8), subplot_kw={"projection": "polar"})
    artists = {}
    for row, group_label in enumerate(PRIMARY_GROUPS):
        for column, map_name in enumerate(MAP_ORDER):
            ax = axes[row, column]
            selected = selected_session.loc[
                selected_session["group"].eq(group_label)
                & selected_session["map"].eq(map_name)
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
            ax.grid(color="#B5B5B5", linewidth=0.65)
            units = int(selected["source_units"].iloc[0])
            probes = int(selected["source_probes"].iloc[0])
            support = selected["supported"].mean()
            areas = str(selected["source_areas"].iloc[0])
            ax.set_title(f"{group_label} · {labels[map_name]}", fontsize=10, pad=12)
            ax.text(
                0.55,
                0.035,
                f"n={units:,}; probes={probes}; grid={support:.0%}",
                transform=ax.transAxes,
                ha="center",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.3},
            )
            if row == 1 and column == 0:
                ax.text(
                    0.5,
                    -0.16,
                    f"areas: {areas}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                )
    for column, map_name in enumerate(MAP_ORDER):
        colorbar = fig.colorbar(
            artists[map_name],
            ax=axes[:, column].tolist(),
            fraction=0.028,
            pad=0.06,
            aspect=25,
            extend="both",
        )
        colorbar.set_label(units_labels[map_name], fontsize=9)
    fig.suptitle(
        f"Allen Brain Observatory 1.1 simultaneous visual-area map\n"
        f"session {session_id} · mouse {specimen_id} · bandwidth {bandwidth_deg:g}°",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.012,
        "All probes are simultaneous within each stimulus block; RF, static-SF, and drifting-TF blocks are sequential.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.08, top=0.84, wspace=0.26, hspace=0.32)
    return fig


def write_report(maps: pd.DataFrame, sessions: list[int], output_path: Path) -> None:
    primary = maps.loc[maps["group"].isin(PRIMARY_GROUPS)]
    summary = (
        primary.groupby(["ecephys_session_id", "specimen_id", "group", "map"], observed=True)
        .agg(
            source_units=("source_units", "first"),
            source_probes=("source_probes", "first"),
            source_areas=("source_areas", "first"),
            supported_grid_fraction=("supported", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Allen Brain Observatory 1.1 simultaneous V1/HVA session atlas",
        "",
        f"The atlas contains {len(sessions)} sessions with V1 and at least one simultaneously",
        "recorded HVA contributing supported RF, SF, and TF populations. Each page pools all",
        "simultaneous HVA probes and compares them with V1 in the same session.",
        "",
        "RF, SF, and TF maps use independent eligible unit populations. RF comes from the",
        "Gabor mapping block, SF from static gratings, and TF from drifting gratings. Probes",
        "recorded simultaneously within each block, but the three blocks occurred sequentially.",
        "No cross-session alignment is applied.",
        "",
        "Area-specific LM/RL/AL/PM/AM surfaces are retained in the grid CSV even though the",
        "primary atlas pools them for more stable within-session support.",
        "",
        "## Outputs",
        "",
        "- `Figure_allen_bo11_simultaneous_v1_hva_session_atlas.pdf`: one page per session.",
        "- `session_figures/`: the same pages as individual PNG files.",
        "- `allen_bo11_simultaneous_v1_hva_surface_grid.csv`: pooled and area-specific grids.",
        "- `allen_bo11_simultaneous_v1_hva_population.csv`: exact populations and support.",
        "",
        "## Interpretation boundary",
        "",
        "This is a within-session V1-versus-HVA view, not an Allen analogue of four V1 probes.",
        "Most Allen sessions contain one V1 probe; the HVA row can combine several probes and",
        "areas. Differences may reflect area, probe targeting, RF coverage, or finite sampling.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary.to_csv(
        output_path.parent / "allen_bo11_simultaneous_v1_hva_population.csv",
        index=False,
        float_format="%.6g",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "session_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    units, unit_path, audit_manifest = load_preference_units(
        args.audit_dir.resolve(),
        args.config,
        minimum_lifetime_sparseness=args.minimum_lifetime_sparseness,
        minimum_stimulus_firing_rate=args.minimum_stimulus_firing_rate,
        require_unique_preference=True,
    )
    sessions = simultaneous_v1_hva_sessions(units)
    if not sessions:
        raise ValueError("No simultaneous V1/HVA sessions with RF, SF, and TF support")
    units = units.loc[units["ecephys_session_id"].isin(sessions)].copy()
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = estimate_maps(
        units,
        sessions,
        grid_points,
        bandwidth_deg=args.bandwidth_deg,
        minimum_local_units=args.minimum_local_units,
    )
    maps.to_csv(
        output_dir / "allen_bo11_simultaneous_v1_hva_surface_grid.csv",
        index=False,
        float_format="%.6g",
    )
    norms = global_norms(maps)
    atlas_path = output_dir / "Figure_allen_bo11_simultaneous_v1_hva_session_atlas.pdf"
    with PdfPages(atlas_path) as pdf:
        for session_id in sessions:
            fig = render_session_figure(
                maps,
                session_id,
                az_grid,
                el_grid,
                norms,
                bandwidth_deg=args.bandwidth_deg,
            )
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(
                figure_dir / f"allen_bo11_session_{session_id}_v1_hva_rf_sf_tf.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)
    write_report(
        maps,
        sessions,
        output_dir / "ALLEN_BO11_SIMULTANEOUS_V1_HVA_SESSION_MAPS.md",
    )
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "checkpoint": "06c_allen_bo11_simultaneous_v1_hva_session_maps",
        "status": "within-session simultaneous V1-versus-pooled-HVA atlas",
        "input": {"path": str(unit_path.resolve()), "sha256": sha256(unit_path.resolve())},
        "audit_manifest_sha256": hashlib.sha256(
            json.dumps(audit_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "primary_groups": list(PRIMARY_GROUPS),
            "area_specific_grids": list(HVA_ORDER),
            "bandwidth_deg": args.bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_local_units": args.minimum_local_units,
            "minimum_lifetime_sparseness": args.minimum_lifetime_sparseness,
            "minimum_stimulus_firing_rate": args.minimum_stimulus_firing_rate,
            "require_unique_preference": True,
            "cross_session_alignment": "none",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Allen simultaneous V1/HVA session atlas written to {output_dir}")


if __name__ == "__main__":
    main()
