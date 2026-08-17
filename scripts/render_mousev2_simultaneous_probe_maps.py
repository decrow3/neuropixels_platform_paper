#!/usr/bin/env python3
"""Render independent RF/SF/TF maps for complete simultaneous MouseV2 probe quartets."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
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
    MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG,
    MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG,
    PREFERENCES,
    PROBE_ORDER,
    load_mousev2_units,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "figure3"
    / "06d_mousev2_frequency_preference_surfaces"
    / "simultaneous_probe_maps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-table", type=Path, default=DEFAULT_RF)
    parser.add_argument("--grating-dir", type=Path, default=DEFAULT_GRATINGS)
    parser.add_argument("--tuning-support", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-effective-sessions", type=float, default=3.0)
    parser.add_argument("--minimum-local-units", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_simultaneous_sessions(units: pd.DataFrame) -> list[str]:
    """Require every probe to contribute RF-, SF-, and TF-supported units."""
    eligible_sessions: set[str] | None = None
    for flag in ("analysis_eligible", "tuning_eligible_sf", "tuning_eligible_tf"):
        counts = (
            units.loc[units[flag]]
            .groupby(["site", "probe"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=PROBE_ORDER, fill_value=0)
        )
        complete = set(counts.index[counts.gt(0).all(axis=1)].astype(str))
        eligible_sessions = complete if eligible_sessions is None else eligible_sessions & complete
    return sorted(eligible_sessions or set())


def align_simultaneous_rf_translation(
    units: pd.DataFrame,
    sessions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove a session-wide RF translation estimated with equal probe weight.

    The median RF center is first computed within each probe, then across the
    four simultaneous probes.  This avoids allowing a high-yield probe to set
    the session reference.  Only a common translation is removed; within-session
    probe geometry is unchanged.
    """
    result = units.loc[units["site"].isin(sessions)].copy()
    selected = result.loc[result["analysis_eligible"]]
    probe_centers = (
        selected.groupby(["site", "probe"], observed=True)[
            ["stimulus_x_deg", "stimulus_y_deg"]
        ]
        .median()
        .reset_index()
    )
    counts = probe_centers.groupby("site", observed=True)["probe"].nunique()
    if not counts.eq(len(PROBE_ORDER)).all():
        raise ValueError("Alignment requires all four probes in every retained session")
    references = probe_centers.groupby("site", observed=True)[
        ["stimulus_x_deg", "stimulus_y_deg"]
    ].median()
    grand_reference = references.median()
    shifts = references - grand_reference
    shifts.columns = ["session_shift_x_deg", "session_shift_y_deg"]
    audit = references.rename(
        columns={
            "stimulus_x_deg": "session_reference_x_deg",
            "stimulus_y_deg": "session_reference_y_deg",
        }
    ).join(shifts)
    audit["aligned_reference_x_deg"] = grand_reference["stimulus_x_deg"]
    audit["aligned_reference_y_deg"] = grand_reference["stimulus_y_deg"]
    audit["simultaneous_probes"] = len(PROBE_ORDER)
    audit = audit.reset_index()
    result = result.merge(
        audit[["site", "session_shift_x_deg", "session_shift_y_deg"]],
        on="site",
        how="left",
        validate="many_to_one",
    )
    result["aligned_stimulus_x_deg"] = (
        result["stimulus_x_deg"] - result["session_shift_x_deg"]
    )
    result["aligned_stimulus_y_deg"] = (
        result["stimulus_y_deg"] - result["session_shift_y_deg"]
    )
    result["azimuth_rf_aligned"] = (
        result["aligned_stimulus_x_deg"] + MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG
    )
    result["elevation_rf_aligned"] = (
        result["aligned_stimulus_y_deg"] + MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG
    )
    return result, audit


def session_balanced_rf_density(
    group: pd.DataFrame,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> dict[str, np.ndarray]:
    points = group[["azimuth_rf_aligned", "elevation_rf_aligned"]].to_numpy(float)
    sessions = group["site"].to_numpy()
    distance_squared = np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * distance_squared / bandwidth_deg**2)
    unique_sessions = np.unique(sessions)
    by_session = np.column_stack(
        [kernel[:, sessions == session].mean(axis=1) for session in unique_sessions]
    )
    density = by_session.mean(axis=1)
    effective_sessions = np.divide(
        by_session.sum(axis=1) ** 2,
        np.square(by_session).sum(axis=1),
        out=np.zeros(len(grid_points)),
        where=np.square(by_session).sum(axis=1) > 0,
    )
    local_units = (distance_squared <= (1.5 * bandwidth_deg) ** 2).sum(axis=1)
    supported = (
        (effective_sessions >= minimum_effective_sessions)
        & (local_units >= minimum_local_units)
    )
    density[~supported] = np.nan
    return {
        "density": density,
        "effective_sessions": effective_sessions,
        "local_units": local_units,
        "supported": supported,
    }


def estimate_maps(
    units: pd.DataFrame,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for probe in PROBE_ORDER:
        rf_group = units.loc[units["analysis_eligible"] & units["probe"].eq(probe)]
        rf = session_balanced_rf_density(
            rf_group,
            grid_points,
            bandwidth_deg=bandwidth_deg,
            minimum_effective_sessions=minimum_effective_sessions,
            minimum_local_units=minimum_local_units,
        )
        frames.append(
            pd.DataFrame(
                {
                    "map": "rf_density",
                    "probe": probe,
                    "azimuth_deg": grid_points[:, 0],
                    "elevation_deg": grid_points[:, 1],
                    "estimate": rf["density"],
                    "estimate_log2": np.nan,
                    "effective_sessions": rf["effective_sessions"],
                    "local_units": rf["local_units"],
                    "supported": rf["supported"],
                    "source_units": len(rf_group),
                    "source_sessions": rf_group["site"].nunique(),
                }
            )
        )
        for preference, specification in PREFERENCES.items():
            group = units.loc[
                units[f"tuning_eligible_{preference}"] & units["probe"].eq(probe)
            ]
            surface = session_balanced_gaussian_surface(
                group[["azimuth_rf_aligned", "elevation_rf_aligned"]].to_numpy(float),
                np.log2(group[specification["column"]].to_numpy(float)),
                group["site"].to_numpy(),
                grid_points,
                bandwidth_deg=bandwidth_deg,
                minimum_effective_sessions=minimum_effective_sessions,
                minimum_local_units=minimum_local_units,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "map": preference,
                        "probe": probe,
                        "azimuth_deg": grid_points[:, 0],
                        "elevation_deg": grid_points[:, 1],
                        "estimate": np.exp2(surface["estimate_log2"]),
                        "estimate_log2": surface["estimate_log2"],
                        "effective_sessions": surface["effective_sessions"],
                        "local_units": surface["local_units"],
                        "supported": surface["supported"],
                        "source_units": len(group),
                        "source_sessions": group["site"].nunique(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def pairwise_surface_summary(maps: pd.DataFrame) -> pd.DataFrame:
    minimum_shared_points = 50
    rows = []
    keys = ["azimuth_deg", "elevation_deg"]
    for preference in ("sf", "tf"):
        selected = maps.loc[maps["map"].eq(preference)]
        for first, second in combinations(PROBE_ORDER, 2):
            left = selected.loc[selected["probe"].eq(first), keys + ["estimate_log2", "supported"]]
            right = selected.loc[selected["probe"].eq(second), keys + ["estimate_log2", "supported"]]
            joined = left.merge(right, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
            shared = joined.loc[
                joined["supported_first"]
                & joined["supported_second"]
                & joined[["estimate_log2_first", "estimate_log2_second"]].notna().all(axis=1)
            ].copy()
            delta = shared["estimate_log2_first"] - shared["estimate_log2_second"]
            comparison_supported = len(shared) >= minimum_shared_points
            rows.append(
                {
                    "preference": preference,
                    "first_probe": first,
                    "second_probe": second,
                    "contrast": f"{first} minus {second}",
                    "shared_grid_points": len(shared),
                    "shared_grid_fraction": len(shared) / len(joined),
                    "comparison_supported": comparison_supported,
                    "median_difference_octaves": delta.median() if comparison_supported else np.nan,
                    "p10_difference_octaves": delta.quantile(0.1) if comparison_supported else np.nan,
                    "p90_difference_octaves": delta.quantile(0.9) if comparison_supported else np.nan,
                    "surface_correlation": shared["estimate_log2_first"].corr(
                        shared["estimate_log2_second"]
                    ) if comparison_supported else np.nan,
                }
            )
    return pd.DataFrame(rows)


def render_maps(
    maps: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
    *,
    bandwidth_deg: float,
    session_count: int,
) -> None:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(
        az_mesh, el_mesh, center_azimuth_deg=50.0, center_elevation_deg=10.0
    )
    theta_degrees = np.degrees(theta)
    radial_limit = np.ceil(np.nanmax(radius) / 10) * 10
    map_order = ("rf_density", "sf", "tf")
    labels = {
        "rf_density": "RF density\n(equal-session a.u.)",
        "sf": "Preferred SF (cycles/deg)",
        "tf": "Preferred TF (Hz)",
    }
    cmaps = {"rf_density": "magma", "sf": "viridis", "tf": "plasma"}
    norms: dict[str, Normalize] = {}
    for map_name in map_order:
        finite = maps.loc[maps["map"].eq(map_name), "estimate"].dropna().to_numpy(float)
        limits = np.quantile(finite, [0.02, 0.98])
        norms[map_name] = Normalize(vmin=limits[0], vmax=limits[1])

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16.8, 12.8),
        subplot_kw={"projection": "polar"},
    )
    artists: dict[str, object] = {}
    for row, map_name in enumerate(map_order):
        for column, probe in enumerate(PROBE_ORDER):
            ax = axes[row, column]
            selected = maps.loc[
                maps["map"].eq(map_name) & maps["probe"].eq(probe)
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
            support = selected["supported"].mean()
            if row == 0:
                ax.set_title(f"Probe {probe}", pad=14, fontsize=12)
            ax.text(
                0.56,
                0.04,
                f"n={units:,}; grid={support:.0%}",
                transform=ax.transAxes,
                fontsize=8,
                ha="center",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
            )
            if column == 0:
                ax.set_ylabel(labels[map_name], labelpad=38, fontsize=10)
    for row, map_name in enumerate(map_order):
        colorbar = fig.colorbar(
            artists[map_name],
            ax=axes[row, :].tolist(),
            fraction=0.018,
            pad=0.035,
            aspect=30,
            extend="both",
        )
        colorbar.set_label(labels[map_name])
    fig.suptitle(
        "MouseV2 independent RF and tuning maps for simultaneous V1 probe quartets\n"
        f"{session_count} complete sessions; equal session weight; common-translation RF alignment; bandwidth {bandwidth_deg:g}°",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.012,
        "Polar center = MouseV2 display center (50° azimuth, 10° elevation on the translated axes). Alignment removes only each session's shared RF translation; it is not gaze correction.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.055, top=0.89, wspace=0.25, hspace=0.28)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    units: pd.DataFrame,
    audit: pd.DataFrame,
    maps: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_path: Path,
) -> None:
    populations = (
        units.groupby("probe", observed=True)[
            ["analysis_eligible", "tuning_eligible_sf", "tuning_eligible_tf"]
        ]
        .sum()
        .reindex(PROBE_ORDER)
    )
    maximum_shift = np.hypot(
        audit["session_shift_x_deg"], audit["session_shift_y_deg"]
    ).max()
    lines = [
        "# MouseV2 simultaneous-probe RF and tuning maps",
        "",
        f"All {audit['site'].nunique()} retained sessions contain the complete A/B/C/E probe quartet",
        "with at least one independently supported RF, SF preference, and TF preference per probe.",
        "RF density uses all supported RF units; SF and TF maps use their own independently",
        "supported tuning populations. Sessions receive equal prior weight.",
        "",
        "| Probe | RF units | SF units | TF units |",
        "| --- | ---: | ---: | ---: |",
    ]
    for probe, row in populations.iterrows():
        lines.append(
            f"| {probe} | {int(row.analysis_eligible):,} | "
            f"{int(row.tuning_eligible_sf):,} | {int(row.tuning_eligible_tf):,} |"
        )
    lines.extend(
        [
            "",
            "## Alignment and interpretation boundary",
            "",
            "A session reference is the median of the four probe-specific median RF centers.",
            "Each session is translated to the across-session median reference before smoothing;",
            "this preserves every within-session probe offset and prevents high-yield probes from",
            "dominating the alignment. The largest translation was "
            f"{maximum_shift:.1f}°. This is a sensitivity analysis for shared screen/eye-position",
            "translation, not a measured gaze correction; rotations, scale changes, and genuine",
            "session-wide retinotopic differences remain unresolved.",
            "",
            "Probe E has sparse TF support in several sessions, so its local support is visibly",
            "smaller and should not be interpreted where the grid is masked.",
            "The probes target substantially different RF regions. Pairwise probe summaries",
            "therefore require at least 50 shared supported grid points; unsupported contrasts",
            "are left blank. Retained contrasts are descriptive surfaces, not unit-independent",
            "inferential tests.",
            "",
            "## Cross-dataset note retained for later",
            "",
            "The MouseV2–Allen median offset was much larger for SF (1.35×) than TF (1.07×).",
            "Plausible contributors include the different stimulus families and preference",
            "estimators (continuous joint fits versus released discrete bins), different sampled",
            "unit/RF populations, and MouseV2's explicitly retained identifiable extrapolated peaks.",
            "The present simultaneous-probe maps do not adjudicate those explanations.",
            "",
            "## Outputs",
            "",
            "- `Figure_mousev2_simultaneous_probe_rf_sf_tf_polar.png`: aligned RF/SF/TF probe maps.",
            "- `simultaneous_probe_surface_grid.csv`: plotted grid and support diagnostics.",
            "- `simultaneous_probe_session_alignment.csv`: translation audit.",
            "- `simultaneous_probe_population.csv`: exact session-probe populations.",
            "- `simultaneous_probe_pairwise_surface_summary.csv`: descriptive probe contrasts.",
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
    units, audit = align_simultaneous_rf_translation(units, sessions)
    population = (
        units.groupby(["site", "probe"], observed=True)[
            ["analysis_eligible", "tuning_eligible_sf", "tuning_eligible_tf"]
        ]
        .sum()
        .reset_index()
    )
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = estimate_maps(
        units,
        grid_points,
        bandwidth_deg=args.bandwidth_deg,
        minimum_effective_sessions=args.minimum_effective_sessions,
        minimum_local_units=args.minimum_local_units,
    )
    pairwise = pairwise_surface_summary(maps)
    figure_path = output_dir / "Figure_mousev2_simultaneous_probe_rf_sf_tf_polar.png"
    render_maps(
        maps,
        az_grid,
        el_grid,
        figure_path,
        bandwidth_deg=args.bandwidth_deg,
        session_count=len(sessions),
    )
    maps.to_csv(output_dir / "simultaneous_probe_surface_grid.csv", index=False, float_format="%.6g")
    audit.to_csv(output_dir / "simultaneous_probe_session_alignment.csv", index=False, float_format="%.6g")
    population.to_csv(output_dir / "simultaneous_probe_population.csv", index=False)
    pairwise.to_csv(
        output_dir / "simultaneous_probe_pairwise_surface_summary.csv",
        index=False,
        float_format="%.6g",
    )
    write_report(
        units,
        audit,
        maps,
        pairwise,
        output_dir / "MOUSEV2_SIMULTANEOUS_PROBE_MAPS.md",
    )
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06d_mousev2_simultaneous_probe_maps",
        "status": "complete-quartet translation-aligned sensitivity maps",
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
            "probe_order": list(PROBE_ORDER),
            "complete_sessions": sessions,
            "bandwidth_deg": args.bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_effective_sessions": args.minimum_effective_sessions,
            "minimum_local_units": args.minimum_local_units,
            "session_weighting": "equal prior weight",
            "alignment": "median across four probe-specific median RF centers; translation only",
            "gaze_correction": "none; alignment is sensitivity analysis",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MouseV2 simultaneous-probe maps written to {output_dir}")


if __name__ == "__main__":
    main()
