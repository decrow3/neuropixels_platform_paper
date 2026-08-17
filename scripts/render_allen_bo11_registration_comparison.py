#!/usr/bin/env python3
"""Compare five Allen session-map registrations, including V1 CCF and RF size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from scripts.allen_bo11_tuning_driven_limited_affine import (
    MAP_KEYS,
    PARAMETER_NAMES,
    aggregate_templates,
    evaluate_model,
    load_maps,
    polar_coordinates,
    summarize,
    warp_all,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_SURFACE_GRID = AUDIT / "tuning_weighted_session_surfaces" / "allen_bo11_tuning_weighted_surface_grid.csv"
DEFAULT_TUNING_TRANSFORMS = AUDIT / "tuning_driven_limited_affine_weighted" / "limited_affine_transform_history.csv"
DEFAULT_RF_UNITS = AUDIT / "rf_unit_common_support.csv"
DEFAULT_CCF_TRANSFORMS = AUDIT / "ccf_retinotopy_alignment" / "selected_ccf_retinotopy_transforms.csv"
DEFAULT_RF_SIZE_TRANSFORMS = AUDIT / "v1_rf_size_translation_fixed_penalty_bound_30" / "selected_v1_rf_size_translations.csv"
DEFAULT_OUTPUT = AUDIT / "tuning_driven_limited_affine_weighted" / "five_registration_comparison"
BO_COHORT = "Brain Observatory 1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-grid", type=Path, default=DEFAULT_SURFACE_GRID)
    parser.add_argument("--tuning-transforms", type=Path, default=DEFAULT_TUNING_TRANSFORMS)
    parser.add_argument("--rf-units", type=Path, default=DEFAULT_RF_UNITS)
    parser.add_argument("--ccf-transforms", type=Path, default=DEFAULT_CCF_TRANSFORMS)
    parser.add_argument("--rf-size-transforms", type=Path, default=DEFAULT_RF_SIZE_TRANSFORMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-shared-grid-points", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v1_rf_center_translation_parameters(
    rf_units: pd.DataFrame,
    sessions: list[int],
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Translate each session's median V1 RF center to the cross-session median."""
    selected = rf_units.loc[
        rf_units["cohort"].eq(BO_COHORT)
        & rf_units["area"].eq("V1")
        & rf_units["ecephys_session_id"].isin(sessions)
    ].copy()
    centers = (
        selected.groupby("ecephys_session_id", observed=True)
        .agg(
            v1_rf_center_azimuth_deg=("azimuth_rf", "median"),
            v1_rf_center_elevation_deg=("elevation_rf", "median"),
            v1_rf_units=("ecephys_unit_id", "size"),
            v1_rf_probes=("ecephys_probe_id", "nunique"),
        )
        .reset_index()
    )
    missing = sorted(set(sessions) - set(centers["ecephys_session_id"].astype(int)))
    if missing:
        raise ValueError(f"Missing V1 RF centers for sessions: {missing}")
    reference_azimuth = float(centers["v1_rf_center_azimuth_deg"].median())
    reference_elevation = float(centers["v1_rf_center_elevation_deg"].median())
    centers["reference_azimuth_deg"] = reference_azimuth
    centers["reference_elevation_deg"] = reference_elevation
    centers["translation_azimuth_deg"] = reference_azimuth - centers["v1_rf_center_azimuth_deg"]
    centers["translation_elevation_deg"] = reference_elevation - centers["v1_rf_center_elevation_deg"]
    parameters = {}
    for row in centers.itertuples(index=False):
        parameters[int(row.ecephys_session_id)] = np.array(
            [row.translation_azimuth_deg, row.translation_elevation_deg, 0.0, 0.0, 0.0, 0.0]
        )
    return parameters, centers


def load_tuning_parameters(path: Path, sessions: list[int]) -> dict[int, np.ndarray]:
    history = pd.read_csv(path)
    final = history.loc[history["iteration"].eq(history["iteration"].max())]
    parameters = {
        int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float)
        for _, row in final.iterrows()
    }
    missing = sorted(set(sessions) - set(parameters))
    if missing:
        raise ValueError(f"Missing tuning-fitted transforms for sessions: {missing}")
    return {session_id: parameters[session_id] for session_id in sessions}


def load_ccf_parameters(path: Path, sessions: list[int]) -> tuple[dict[int, np.ndarray], str, list[int]]:
    table = pd.read_csv(path)
    available = table.loc[table["ccf_available"].eq(True)].copy()
    parameters = {
        int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float)
        for _, row in available.iterrows()
    }
    included = sorted(set(sessions) & set(parameters))
    if not included:
        raise ValueError("No map sessions have V1 CCF-derived transforms")
    models = available["selected_model"].dropna().unique()
    if len(models) != 1:
        raise ValueError("Expected one globally selected CCF retinotopy model")
    return {session_id: parameters[session_id] for session_id in included}, str(models[0]), included


def load_rf_size_parameters(path: Path, sessions: list[int]) -> dict[int, np.ndarray]:
    table = pd.read_csv(path)
    parameters = {
        int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float)
        for _, row in table.iterrows()
    }
    missing = sorted(set(sessions) - set(parameters))
    if missing:
        raise ValueError(f"Missing V1 RF-size transforms for sessions: {missing}")
    return {session_id: parameters[session_id] for session_id in sessions}


def render_registration_templates(
    template_rows: list[tuple[str, dict[tuple[str, str], dict[str, np.ndarray]]]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
) -> None:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    theta, radius = polar_coordinates(az_mesh, el_mesh)
    row_count = len(template_rows)
    fig, axes = plt.subplots(row_count, 4, figsize=(16.4, 3.72 * row_count), subplot_kw={"projection": "polar"})
    for column, key in enumerate(MAP_KEYS):
        group, preference = key
        values_by_row = [np.exp2(templates[key]["value"]) for _, templates in template_rows]
        finite = np.concatenate([values[np.isfinite(values)] for values in values_by_row])
        limits = np.quantile(finite, [0.02, 0.98])
        norm = Normalize(*limits)
        artist = None
        for row, ((label, _), values) in enumerate(zip(template_rows, values_by_row)):
            ax = axes[row, column]
            artist = ax.pcolormesh(
                theta,
                radius,
                values,
                shading="gouraud",
                cmap="viridis" if preference == "sf" else "plasma",
                norm=norm,
            )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_ylim(0, 110)
            ax.set_rlabel_position(65)
            ax.set_thetamin(-70)
            ax.set_thetamax(70)
            ax.grid(color="#B5B5B5", linewidth=0.65)
            ax.set_title(f"{label}\n{group} · {preference.upper()}", fontsize=10)
        colorbar = fig.colorbar(
            artist,
            ax=axes[:, column].tolist(),
            fraction=0.019,
            pad=0.05,
            extend="both",
        )
        colorbar.set_label("cycles/deg" if preference == "sf" else "Hz")
    fig.suptitle(
        "Allen BO 1.1 quality-weighted session stacks under five registrations\n"
        "rows 2, 4, and 5 use V1 RF center, CCF→RF, and interior RF-size landmarks; SF/TF fit only row 3",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.035, right=0.95, bottom=0.045, top=0.89, wspace=0.31, hspace=0.38)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_maps, az_grid, el_grid = load_maps(args.surface_grid.resolve())
    all_sessions = sorted({key[0] for key in source_maps})
    ccf_parameters, ccf_model, sessions = load_ccf_parameters(args.ccf_transforms.resolve(), all_sessions)
    source_maps = {key: value for key, value in source_maps.items() if key[0] in sessions}
    identity = {session_id: np.zeros(6) for session_id in sessions}
    rf_parameters, rf_audit = v1_rf_center_translation_parameters(
        pd.read_csv(args.rf_units.resolve(), low_memory=False), sessions
    )
    tuning_parameters = load_tuning_parameters(args.tuning_transforms.resolve(), sessions)
    rf_size_parameters = load_rf_size_parameters(args.rf_size_transforms.resolve(), sessions)
    registrations = [
        ("Raw stack", "raw", identity),
        ("V1-RF-center stack", "v1_rf_center_translation", rf_parameters),
        ("Tuning-fitted stack", "tuning_fitted_affine", tuning_parameters),
        ("CCF→V1-RF stack", ccf_model, ccf_parameters),
        ("Interior V1 RF-size stack", "interior_v1_rf_size_translation", rf_size_parameters),
    ]
    template_rows = []
    metrics = []
    for label, model, parameters in registrations:
        warped = warp_all(source_maps, parameters, az_grid, el_grid)
        template_rows.append((label, aggregate_templates(warped)))
        metrics.append(
            evaluate_model(
                source_maps,
                parameters,
                az_grid,
                el_grid,
                args.minimum_shared_grid_points,
                model,
            )
        )
    metrics_table = pd.concat(metrics, ignore_index=True)
    raw_metrics = metrics[0]
    summary = pd.concat(
        [summarize(raw_metrics, local).assign(registration=model) for (_, model, _), local in zip(registrations[1:], metrics[1:])],
        ignore_index=True,
    )
    figure_path = output_dir / "Figure_allen_bo11_five_registration_stacked_templates.png"
    render_registration_templates(template_rows, az_grid, el_grid, figure_path)
    rf_audit.to_csv(output_dir / "v1_rf_center_translation_audit.csv", index=False, float_format="%.6g")
    metrics_table.to_csv(output_dir / "five_registration_session_map_agreement.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "five_registration_agreement_summary.csv", index=False, float_format="%.6g")
    lines = [
        "# Allen BO 1.1 five-registration comparison",
        "",
        "The middle row applies one translation per session: the median center of that",
        "session's supported V1 RF units is moved to the cross-session median V1 RF center.",
        "The identical translation is applied to V1 and HVA SF/TF maps. There is no rotation,",
        "scale, shear, or tuning-driven fitting in this middle-row registration.",
        "",
        "This matches the center-registration concept, but is distinct from the rejected",
        "all-area RF-consensus affine diagnostic, which used multiple area centers and allowed",
        "scale, shear, rotation, and reflection.",
        "",
        f"The fourth row uses the **{ccf_model}** model on the {len(sessions)}/{len(all_sessions)} sessions with reconstructed V1 CCF coordinates.",
        "For each session, a robust session-balanced CCF→RF model was learned from V1 units in all other sessions.",
        "The median held-out V1 prediction residual defines one translation shared by V1 and simultaneous HVA maps.",
        "The comparison restricts every row to the same CCF-available sessions. RF size, HVA units, SF, and TF were not used to fit or select row 4.",
        "",
        "The fifth row fits translation from the interior V1 RF-size surface, excluding RF centers within 20° of a stimulus-grid edge.",
        "RF size is log2 transformed and standardized within session, then matched to a leave-one-session-out template.",
        "The ±30° range was selected as the least censoring exploratory bound after cross-half predictive comparison; the per-degree regularization remained fixed.",
        "SF, TF, and HVA data were not used for the fifth-row transform.",
        "",
        "| Registration | Group | Map | Median paired Δr versus raw |",
        "| --- | --- | --- | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.registration} | {row.group} | {row.preference.upper()} | "
            f"{row.median_paired_correlation_change:+.3f} |"
        )
    (output_dir / "ALLEN_BO11_FIVE_REGISTRATION_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_five_registration_comparison",
        "status": "raw, V1-RF-center, tuning-fitted, V1 CCF-to-RF, and interior V1 RF-size registration comparison",
        "inputs": {
            "surface_grid": {"path": str(args.surface_grid.resolve()), "sha256": sha256(args.surface_grid.resolve())},
            "tuning_transforms": {"path": str(args.tuning_transforms.resolve()), "sha256": sha256(args.tuning_transforms.resolve())},
            "rf_units": {"path": str(args.rf_units.resolve()), "sha256": sha256(args.rf_units.resolve())},
            "ccf_transforms": {"path": str(args.ccf_transforms.resolve()), "sha256": sha256(args.ccf_transforms.resolve())},
            "rf_size_transforms": {"path": str(args.rf_size_transforms.resolve()), "sha256": sha256(args.rf_size_transforms.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "excluded_without_v1_ccf": sorted(set(all_sessions) - set(sessions)),
            "v1_center_statistic": "within-session median of supported V1 RF unit centers",
            "v1_center_reference": "cross-session median of session V1 RF centers",
            "middle_row_transform": "translation only; shared by V1 and HVA SF/TF maps",
            "fourth_row_transform": f"{ccf_model}; leave-one-session-out V1 CCF-to-RF residual; tuning held out",
            "fifth_row_transform": "interior V1 RF-size translation; 20-degree edge exclusion; fixed penalty; +/-30-degree bound; tuning held out",
            "minimum_shared_grid_points": args.minimum_shared_grid_points,
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen five-registration comparison written to {output_dir}")


if __name__ == "__main__":
    main()
