#!/usr/bin/env python3
"""Map supported parametric MouseV2 SF/TF preferences over supported RF centers."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import (  # noqa: E402
    polar_coordinates,
    render_compact_surface_figure,
    render_polar_surface_figure,
    rf_occupancy_counts,
    session_balanced_gaussian_surface,
    validate_bandwidths,
)


DEFAULT_RF = ROOT / "data" / "imports" / "mousev2_parametric_rf_v1" / "rf_unit_fits.csv"
DEFAULT_GRATINGS = ROOT / "data" / "imports" / "mousev2_grating_metrics_v1"
DEFAULT_TUNING = (
    ROOT / "data" / "imports" / "mousev2_frequency_tuning_v1" / "frequency_tuning_support.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "figure3" / "06d_mousev2_frequency_preference_surfaces"
POOLED_AREA = "Multi-probe V1"
PROBE_ORDER = ("A", "B", "C", "E")
MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG = 50.0
MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG = 10.0
RAW_RF_LIMITS_DEG = (-40.0, 40.0)
PREFERENCES = {
    "sf": {
        "column": "pref_sf_supported_dg",
        "values": (0.02, 0.04, 0.08, 0.16, 0.32),
        "fit_bounds": (0.01, 0.64),
    },
    "tf": {
        "column": "pref_tf_supported_dg",
        "values": (1.0, 2.0, 4.0, 8.0, 15.0),
        "fit_bounds": (0.5, 30.0),
    },
}


def to_allen_display_coordinates(
    stimulus_x_deg: pd.Series | np.ndarray,
    stimulus_y_deg: pd.Series | np.ndarray,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """Map display-centered MouseV2 positions onto Allen's released RF axes.

    Both experiments used nine positions at 10-degree spacing. MouseV2 saved
    PsychoPy positions relative to display center (-40..40 on each axis), while
    AllenSDK reports the corresponding grid as azimuth 10..90 and elevation
    -30..50. This is a fixed display-coordinate translation, not gaze correction.
    """
    return (
        stimulus_x_deg + MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG,
        stimulus_y_deg + MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-table", type=Path, default=DEFAULT_RF)
    parser.add_argument("--grating-dir", type=Path, default=DEFAULT_GRATINGS)
    parser.add_argument("--tuning-support", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qc-profile", choices=("pilot_qc", "default_qc"), default="pilot_qc")
    parser.add_argument("--allow-preference-ties", action="store_true")
    parser.add_argument("--bandwidths-deg", type=float, nargs="+", default=(8.0, 12.0, 16.0))
    parser.add_argument("--primary-bandwidth-deg", type=float, default=12.0)
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


def load_mousev2_units(
    rf_path: Path,
    grating_dir: Path,
    *,
    qc_profile: str,
    require_unique_preference: bool,
    tuning_support_path: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    rf = pd.read_csv(rf_path, low_memory=False)
    parametric_rf = "rf_model_supported" in rf.columns
    if parametric_rf:
        required_rf = {
            "supported_rf_center_x_deg",
            "supported_rf_center_y_deg",
            "rf_model_supported",
        }
        missing_rf = required_rf.difference(rf.columns)
        if missing_rf:
            raise ValueError(f"{rf_path} lacks columns {sorted(missing_rf)}")
        rf["rf_center_x_deg"] = rf["supported_rf_center_x_deg"]
        rf["rf_center_y_deg"] = rf["supported_rf_center_y_deg"]
    metric_paths = [Path(path) for path in sorted(glob.glob(str(grating_dir / "site*" / "grating_metrics.csv")))]
    if not metric_paths:
        raise FileNotFoundError(f"No per-site grating metrics in {grating_dir}")
    gratings = pd.concat([pd.read_csv(path, low_memory=False) for path in metric_paths], ignore_index=True)
    if rf["unit_id"].duplicated().any() or gratings["unit_id"].duplicated().any():
        raise ValueError("MouseV2 RF or grating table contains duplicate unit IDs")
    units = rf.merge(gratings, on="unit_id", how="inner", validate="one_to_one")
    if len(units) != len(rf) or (not parametric_rf and len(units) != len(gratings)):
        raise ValueError(
            f"Incomplete RF/grating join: RF={len(rf)}, grating={len(gratings)}, joined={len(units)}"
        )
    if tuning_support_path is not None:
        tuning = pd.read_csv(tuning_support_path, low_memory=False)
        required_tuning = {
            "unit_id",
            "sf_preference_supported",
            "tf_preference_supported",
            "pref_sf_supported_dg",
            "pref_tf_supported_dg",
        }
        missing = required_tuning.difference(tuning.columns)
        if missing:
            raise ValueError(f"{tuning_support_path} lacks columns {sorted(missing)}")
        if tuning["unit_id"].duplicated().any():
            raise ValueError("MouseV2 tuning-support table contains duplicate unit IDs")
        # RF/grating inputs own the canonical session metadata for spatial maps.
        tuning = tuning.drop(columns=["site", "subject_id"], errors="ignore")
        units = units.merge(tuning, on="unit_id", how="left", validate="one_to_one")
        if units[["sf_preference_supported", "tf_preference_supported"]].isna().any().any():
            raise ValueError("Tuning-support table does not cover every RF/grating unit")
    else:
        # Compatibility path for frozen provisional inputs and focused tests.
        units["sf_preference_supported"] = True
        units["tf_preference_supported"] = True
        units["pref_sf_supported_dg"] = units["pref_sf_dg"]
        units["pref_tf_supported_dg"] = units["pref_tf_dg"]

    for column in (
        "rf_center_x_deg",
        "rf_center_y_deg",
        "pref_sf_supported_dg",
        "pref_tf_supported_dg",
        "preferred_condition_ties_dg",
    ):
        units[column] = pd.to_numeric(units[column], errors="coerce")
    eligible = units[qc_profile].fillna(False).astype(bool)
    if parametric_rf:
        eligible &= units["rf_model_supported"].fillna(False).astype(bool)
    if require_unique_preference and tuning_support_path is None:
        eligible &= units["preferred_condition_ties_dg"].eq(1)
    eligible &= units[["rf_center_x_deg", "rf_center_y_deg"]].notna().all(axis=1)
    units["analysis_eligible"] = eligible
    units["tuning_eligible_sf"] = (
        eligible
        & units["sf_preference_supported"].fillna(False).astype(bool)
        & units["pref_sf_supported_dg"].between(
            *PREFERENCES["sf"]["fit_bounds"]
        )
    )
    units["tuning_eligible_tf"] = (
        eligible
        & units["tf_preference_supported"].fillna(False).astype(bool)
        & units["pref_tf_supported_dg"].between(
            *PREFERENCES["tf"]["fit_bounds"]
        )
    )
    units["stimulus_x_deg"] = units["rf_center_x_deg"]
    units["stimulus_y_deg"] = units["rf_center_y_deg"]
    units["azimuth_rf"], units["elevation_rf"] = to_allen_display_coordinates(
        units["stimulus_x_deg"], units["stimulus_y_deg"]
    )
    units["ecephys_session_id"] = units["site"]
    units["area"] = POOLED_AREA
    return units, metric_paths


def estimate_surfaces(
    units: pd.DataFrame,
    grid_points: np.ndarray,
    bandwidths: tuple[float, ...],
    *,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    frames = []
    group_definitions = ((POOLED_AREA, units["probe"].isin(PROBE_ORDER)),) + tuple(
        (f"Probe {probe}", units["probe"].eq(probe)) for probe in PROBE_ORDER
    )
    for preference, specification in PREFERENCES.items():
        for group_label, group_mask in group_definitions:
            group = units.loc[group_mask & units[f"tuning_eligible_{preference}"]].copy()
            points = group[["azimuth_rf", "elevation_rf"]].to_numpy(dtype=float)
            values = np.log2(group[specification["column"]].to_numpy(dtype=float))
            sessions = group["ecephys_session_id"].to_numpy()
            for bandwidth in bandwidths:
                result = session_balanced_gaussian_surface(
                    points,
                    values,
                    sessions,
                    grid_points,
                    bandwidth_deg=bandwidth,
                    minimum_effective_sessions=minimum_effective_sessions,
                    minimum_local_units=minimum_local_units,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "preference": preference,
                            "area": group_label,
                            "bandwidth_deg": bandwidth,
                            "azimuth_deg": grid_points[:, 0],
                            "elevation_deg": grid_points[:, 1],
                            "estimate_log2": result["estimate_log2"],
                            "estimate_preference": np.exp2(result["estimate_log2"]),
                            "effective_sessions": result["effective_sessions"],
                            "local_units": result["local_units"],
                            "supported": result["supported"],
                            "source_units": len(group),
                            "source_sessions": group["site"].nunique(),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def summarize_surfaces(surfaces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in surfaces.groupby(["preference", "area", "bandwidth_deg"], observed=True):
        supported = group.loc[group["supported"] & group["estimate_log2"].notna()]
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
            }
        )
    return pd.DataFrame(rows)


def render_occupancy(units: pd.DataFrame, output_path: Path) -> None:
    selected = units.loc[units["analysis_eligible"]].copy()
    azimuth_edges = np.arange(5.0, 96.0, 10.0)
    elevation_edges = np.arange(-35.0, 56.0, 10.0)
    az_mesh, el_mesh = np.meshgrid(azimuth_edges, elevation_edges)
    theta_edges, radius_edges = polar_coordinates(
        az_mesh, el_mesh, center_azimuth_deg=50.0, center_elevation_deg=10.0
    )
    radial_limit = np.ceil(np.max(radius_edges) / 10) * 10
    definitions = ((POOLED_AREA, selected),) + tuple(
        (f"Probe {probe}", selected.loc[selected["probe"].eq(probe)]) for probe in PROBE_ORDER
    )
    matrices = [rf_occupancy_counts(group, azimuth_edges, elevation_edges) for _, group in definitions]
    maximum = max(int(matrix.max()) for matrix in matrices)
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.8), subplot_kw={"projection": "polar"})
    image_artist = None
    for ax, ((label, group), counts) in zip(axes, zip(definitions, matrices)):
        image_artist = ax.pcolormesh(
            theta_edges,
            radius_edges,
            np.ma.masked_equal(counts, 0),
            shading="flat",
            cmap="inferno",
            norm=LogNorm(vmin=1, vmax=max(1, maximum)),
        )
        ax.set_ylim(0, radial_limit)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(65)
        ax.grid(color="#B5B5B5", linewidth=0.7)
        ax.set_title(f"{label}\n{len(group):,} units", pad=15, fontsize=10)
    colorbar_axis = fig.add_axes([0.945, 0.16, 0.012, 0.68])
    colorbar = fig.colorbar(image_artist, cax=colorbar_axis)
    colorbar.set_label("units per 10° × 10° RF bin (log scale)")
    fig.suptitle(
        "MouseV2 multi-probe V1 supported parametric RF-location occupancy\n"
        "Allen-style display axes; polar center = MouseV2 display center (50°, 10°)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.035, right=0.91, bottom=0.08, top=0.8, wspace=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    units: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    qc_profile: str,
    require_unique_preference: bool,
    primary_bandwidth: float,
    output_path: Path,
) -> None:
    selected = units.loc[units["analysis_eligible"]]
    edge = selected["stimulus_x_deg"].abs().ge(35) | selected["stimulus_y_deg"].abs().ge(35)
    extrapolated_sf = int(
        (units["tuning_eligible_sf"] & units.get("sf_preference_extrapolated", False)).sum()
    )
    extrapolated_tf = int(
        (units["tuning_eligible_tf"] & units.get("tf_preference_extrapolated", False)).sum()
    )
    primary = summary.loc[
        np.isclose(summary["bandwidth_deg"], primary_bandwidth)
        & summary["area"].eq(POOLED_AREA)
    ]
    lines = [
        "# MouseV2 multi-probe V1 SF/TF preference surfaces",
        "",
        "## Status: supported parametric RF and tuning visualization implemented",
        "",
        f"Base population: `{qc_profile}` with supported parametric RF models.",
        f"Eligible units: {len(selected):,} across {selected['site'].nunique()} sessions and four V1 probes.",
        f"Fitted RF centers within 5° of the tested field boundary: {edge.mean():.1%}.",
        f"Mapped extrapolated preferences: SF {extrapolated_sf:,}; TF {extrapolated_tf:,}.",
        "Coordinate harmonization: MouseV2 display-centered positions were translated",
        "to Allen-style released axes as `azimuth = x + 50°` and `elevation = y + 10°`.",
        "This changes the coordinate labels and polar geometry, not the fitted Cartesian",
        "relationships; it is not a gaze correction or a claim of eye-centered position.",
        "",
        "| Preference | Units | Sessions | Supported grid | Surface median | Surface 10–90% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in primary.sort_values("preference").iterrows():
        lines.append(
            f"| {row.preference.upper()} | {int(row.source_units):,} | {int(row.source_sessions)} | "
            f"{row.supported_grid_fraction:.1%} | {row.surface_median_preference:.3g} | "
            f"{row.surface_p10_preference:.3g}–{row.surface_p90_preference:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "MouseV2 jointly varied SF, TF, and orientation. Each mapped preference now",
            "comes from a joint Poisson log-Gaussian × log-Gaussian × von-Mises fit and",
            "is retained only when its",
            "joint tuning, axis-specific tuning, and split-half surface reliability pass",
            "the dataset-wide FDR support contract. Pilot QC remains a unit-quality gate.",
            "SF/TF peaks up to one octave beyond the sampled range are retained when",
            "their fitted width and off-grid optimum remain identifiable; they are",
            "explicitly flagged in the unit table.",
            "RF centers come from supported trial-level elliptical Gaussian fits.",
            "Gaze correction remains unavailable, so positions are display-centered",
            "rather than eye-centered.",
            "",
            "## Outputs",
            "",
            "- `mousev2_frequency_preference_surface_grid.csv`: pooled and per-probe surface grids.",
            "- `mousev2_frequency_preference_surface_summary.csv`: bandwidth and coverage summary.",
            "- `mousev2_rf_coordinate_mapping.csv`: raw-to-Allen-style grid-coordinate audit.",
            "- `Figure_mousev2_v1_frequency_preference_surfaces.png`: pooled Cartesian fits.",
            "- `Figure_mousev2_v1_frequency_preference_surfaces_polar.png`: pooled polar fits.",
            "- `Figure_mousev2_v1_rf_occupancy_by_probe_polar.png`: pooled and per-probe RF occupation.",
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
    units, metric_paths = load_mousev2_units(
        args.rf_table.resolve(),
        args.grating_dir.resolve(),
        qc_profile=args.qc_profile,
        require_unique_preference=not args.allow_preference_ties,
        tuning_support_path=args.tuning_support.resolve(),
    )
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    surfaces = estimate_surfaces(
        units,
        grid_points,
        bandwidths,
        minimum_effective_sessions=args.minimum_effective_sessions,
        minimum_local_units=args.minimum_local_units,
    )
    summary = summarize_surfaces(surfaces)
    surfaces.to_csv(output_dir / "mousev2_frequency_preference_surface_grid.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "mousev2_frequency_preference_surface_summary.csv", index=False, float_format="%.6g")
    raw_grid = np.arange(RAW_RF_LIMITS_DEG[0], RAW_RF_LIMITS_DEG[1] + 1.0, 10.0)
    coordinate_mapping = pd.DataFrame(
        {
            "mousev2_stimulus_x_deg": raw_grid,
            "allen_style_azimuth_deg": raw_grid + MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG,
            "mousev2_stimulus_y_deg": raw_grid,
            "allen_style_elevation_deg": raw_grid + MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG,
        }
    )
    coordinate_mapping.to_csv(output_dir / "mousev2_rf_coordinate_mapping.csv", index=False)
    render_compact_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_mousev2_v1_frequency_preference_surfaces.png",
        area=POOLED_AREA,
        title_population="MouseV2 multi-probe V1",
    )
    render_polar_surface_figure(
        surfaces,
        az_grid,
        el_grid,
        args.primary_bandwidth_deg,
        output_dir / "Figure_mousev2_v1_frequency_preference_surfaces_polar.png",
        area=POOLED_AREA,
        title_population="MouseV2 multi-probe V1",
    )
    render_occupancy(
        units,
        output_dir / "Figure_mousev2_v1_rf_occupancy_by_probe_polar.png",
    )
    write_report(
        units,
        summary,
        qc_profile=args.qc_profile,
        require_unique_preference=not args.allow_preference_ties,
        primary_bandwidth=args.primary_bandwidth_deg,
        output_path=output_dir / "MOUSEV2_FREQUENCY_PREFERENCE_SURFACES.md",
    )
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06d_mousev2_frequency_preference_surfaces",
        "status": "trial-supported frequency preference surfaces implemented",
        "inputs": {
            "rf_table": {"path": str(args.rf_table.resolve()), "sha256": sha256(args.rf_table.resolve())},
            "grating_tables": [
                {"path": str(path), "sha256": sha256(path)} for path in metric_paths
            ],
            "tuning_support": {
                "path": str(args.tuning_support.resolve()),
                "sha256": sha256(args.tuning_support.resolve()),
            },
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "qc_profile": args.qc_profile,
            "preference_gate": "trial-level joint/axis tuning plus split-half reliability",
            "bandwidths_deg": list(bandwidths),
            "primary_bandwidth_deg": args.primary_bandwidth_deg,
            "polar_center_deg": [0.0, 20.0],
            "rf_coordinate_system": "Allen-style released display axes",
            "mousev2_stimulus_x_to_azimuth_offset_deg": MOUSEV2_X_TO_ALLEN_AZIMUTH_OFFSET_DEG,
            "mousev2_stimulus_y_to_elevation_offset_deg": MOUSEV2_Y_TO_ALLEN_ELEVATION_OFFSET_DEG,
            "rf_method": "supported trial-level Poisson elliptical Gaussian",
            "gaze_correction": "none",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MouseV2 frequency-preference surfaces written to {output_dir}")


if __name__ == "__main__":
    main()
