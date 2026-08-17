#!/usr/bin/env python3
"""Fit Allen BO 1.1 session transforms from non-center scalar fields only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from scripts.allen_bo11_tuning_driven_limited_affine import (
    CENTER_DEG,
    map_agreement,
    template_from_maps,
    warp_all,
    warp_map,
)
from scripts.allen_bo11_tuning_weighted_session_surfaces import weighted_gaussian_surface
from scripts.render_allen_bo11_simultaneous_v1_hva_session_maps import (
    group_definitions,
    simultaneous_v1_hva_sessions,
)
from scripts.allen_frequency_preference_surfaces import load_preference_units


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_SUPPORT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = AUDIT / "noncenter_similarity_alignment"
BO_COHORT = "Brain Observatory 1.1"
GROUPS = ("V1", "HVA pooled")
FEATURES = {
    "area_rf": {
        "label": "log2 RF area",
        "transform": "log2",
        "weight": 0.27,
    },
    "dorsal_ventral_ccf_coordinate": {
        "label": "dorsal–ventral CCF coordinate",
        "transform": "identity",
        "weight": 1.00,
    },
    "probe_horizontal_position": {
        "label": "probe horizontal position",
        "transform": "identity",
        "weight": 0.68,
    },
    "time_to_peak_rf": {
        "label": "RF response time-to-peak",
        "transform": "identity",
        "weight": 0.34,
    },
    "time_to_first_spike_fl": {
        "label": "flash first-spike latency",
        "transform": "identity",
        "weight": 0.10,
    },
}
FEATURE_KEYS = tuple((group, feature) for group in GROUPS for feature in FEATURES)
TRANSLATION_BOUNDS = ((-15.0, 15.0), (-15.0, 15.0))
SIMILARITY_BOUNDS = (
    (-15.0, 15.0),
    (-15.0, 15.0),
    (-8.0, 8.0),
    (np.log(0.92), np.log(1.08)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-effective-local-units", type=float, default=10.0)
    parser.add_argument("--minimum-shared-grid-points", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--optimizer-generations", type=int, default=12)
    parser.add_argument("--optimizer-population", type=int, default=4)
    parser.add_argument("--regularization-weight", type=float, default=0.04)
    parser.add_argument("--coverage-penalty-weight", type=float, default=0.25)
    parser.add_argument(
        "--similarity-selection-threshold",
        type=float,
        default=0.02,
        help="Minimum median regularized-objective gain required over translation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_parameters(parameters: np.ndarray, model: str) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    if model == "translation":
        return np.array([parameters[0], parameters[1], 0.0, 0.0, 0.0, 0.0])
    if model == "similarity":
        tx, ty, rotation, log_scale = parameters
        return np.array([tx, ty, rotation, log_scale, log_scale, 0.0])
    raise ValueError(f"Unknown model: {model}")


def robust_area_standardize(table: pd.DataFrame, feature: str, transform: str) -> pd.Series:
    values = pd.to_numeric(table[feature], errors="coerce")
    if transform == "log2":
        values = np.log2(values.where(values > 0))
    elif transform != "identity":
        raise ValueError(f"Unknown feature transform: {transform}")
    result = pd.Series(np.nan, index=table.index, dtype=float)
    for _, indices in table.groupby("area", observed=True).groups.items():
        local = values.loc[indices]
        median = local.median()
        scale = local.quantile(0.75) - local.quantile(0.25)
        if not np.isfinite(scale) or scale <= 1e-9:
            continue
        result.loc[indices] = ((local - median) / scale).clip(-4.0, 4.0)
    return result


def build_feature_maps(
    population: pd.DataFrame,
    sessions: list[int],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_local_units: float,
) -> tuple[dict[tuple[int, str, str], dict[str, np.ndarray]], pd.DataFrame]:
    grid_points = np.column_stack(np.meshgrid(az_grid, el_grid)).reshape(-1, 2)
    # np.column_stack(meshgrid) does not preserve the desired Cartesian order.
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = {}
    audit_rows = []
    for feature, specification in FEATURES.items():
        population[f"standardized_{feature}"] = robust_area_standardize(
            population, feature, str(specification["transform"])
        )
    for session_id in sessions:
        session = population.loc[population["ecephys_session_id"].eq(session_id)]
        for group, mask in group_definitions(session):
            if group not in GROUPS:
                continue
            for feature, specification in FEATURES.items():
                value_column = f"standardized_{feature}"
                selected = session.loc[mask].dropna(subset=[value_column, "azimuth_rf", "elevation_rf"])
                surface = weighted_gaussian_surface(
                    selected[["azimuth_rf", "elevation_rf"]].to_numpy(float),
                    selected[value_column].to_numpy(float),
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
                maps[(session_id, group, feature)] = {
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
                audit_rows.append(
                    {
                        "ecephys_session_id": session_id,
                        "group": group,
                        "feature": feature,
                        "feature_weight": specification["weight"],
                        "source_units": len(selected),
                        "supported_grid_fraction": float(supported.mean()),
                    }
                )
    return maps, pd.DataFrame(audit_rows)


def normalized_penalty(parameters: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> float:
    normalized = [value / (0.5 * (upper - lower)) for value, (lower, upper) in zip(parameters, bounds)]
    return float(np.mean(np.square(normalized)))


def feature_templates(
    warped: dict[tuple[int, str, str], dict[str, np.ndarray]],
    session_id: int,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    return {
        key: template_from_maps(warped, *key, exclude_session=session_id)
        for key in FEATURE_KEYS
    }


def session_objective(
    session_id: int,
    compact_parameters: np.ndarray,
    model: str,
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    templates: dict[tuple[str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    minimum_points: int,
    regularization_weight: float,
    coverage_penalty_weight: float,
) -> float:
    packed = pack_parameters(compact_parameters, model)
    losses = []
    weights = []
    for group, feature in FEATURE_KEYS:
        source = source_maps[(session_id, group, feature)]
        warped = warp_map(source, packed, az_grid, el_grid)
        agreement = map_agreement(warped, templates[(group, feature)], minimum_points)
        if not np.isfinite(agreement["correlation"]):
            continue
        losses.append(
            1.0
            - agreement["correlation"]
            + coverage_penalty_weight * (1.0 - agreement["coverage"])
        )
        weights.append(float(FEATURES[feature]["weight"]))
    if not losses:
        return 2.0
    bounds = TRANSLATION_BOUNDS if model == "translation" else SIMILARITY_BOUNDS
    return float(np.average(losses, weights=weights) + regularization_weight * normalized_penalty(compact_parameters, bounds))


def fit_model(
    model: str,
    source_maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    iterations: int,
    minimum_points: int,
    regularization_weight: float,
    coverage_penalty_weight: float,
    generations: int,
    population: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    sessions = sorted({key[0] for key in source_maps})
    parameters = {session_id: np.zeros(6) for session_id in sessions}
    history = []
    bounds = TRANSLATION_BOUNDS if model == "translation" else SIMILARITY_BOUNDS
    for iteration in range(iterations):
        warped = warp_all(source_maps, parameters, az_grid, el_grid)
        updated = {}
        for session_index, session_id in enumerate(sessions):
            templates = feature_templates(warped, session_id)
            result = differential_evolution(
                lambda compact: session_objective(
                    session_id,
                    compact,
                    model,
                    source_maps,
                    templates,
                    az_grid,
                    el_grid,
                    minimum_points=minimum_points,
                    regularization_weight=regularization_weight,
                    coverage_penalty_weight=coverage_penalty_weight,
                ),
                bounds,
                seed=20260812 + iteration * 100 + session_index + (1000 if model == "similarity" else 0),
                maxiter=generations,
                popsize=population,
                polish=True,
                updating="immediate",
                workers=1,
                tol=1e-4,
            )
            packed = pack_parameters(result.x, model)
            updated[session_id] = packed
            history.append(
                {
                    "model": model,
                    "iteration": iteration + 1,
                    "ecephys_session_id": session_id,
                    "regularized_objective": float(result.fun),
                    "translation_azimuth_deg": packed[0],
                    "translation_elevation_deg": packed[1],
                    "rotation_deg": packed[2],
                    "isotropic_log_scale": packed[3],
                    "isotropic_scale": np.exp(packed[3]),
                }
            )
        parameters = updated
    return parameters, pd.DataFrame(history)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_units, _, _ = load_preference_units(args.audit_dir.resolve(), None)
    sessions = simultaneous_v1_hva_sessions(tuning_units)
    support = pd.read_csv(args.support.resolve(), low_memory=False)
    support = support.loc[
        support["cohort"].eq(BO_COHORT) & support["ecephys_session_id"].isin(sessions)
    ].copy()
    metric_columns = ["ecephys_unit_id", *[feature for feature in FEATURES if feature not in support.columns]]
    metrics = pd.read_csv(args.unit_table.resolve(), usecols=metric_columns, low_memory=False)
    population = support.merge(metrics, on="ecephys_unit_id", how="left", validate="one_to_one")
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    feature_maps, surface_audit = build_feature_maps(
        population,
        sessions,
        az_grid,
        el_grid,
        bandwidth_deg=args.bandwidth_deg,
        minimum_effective_local_units=args.minimum_effective_local_units,
    )
    translation_parameters, translation_history = fit_model(
        "translation",
        feature_maps,
        az_grid,
        el_grid,
        iterations=args.iterations,
        minimum_points=args.minimum_shared_grid_points,
        regularization_weight=args.regularization_weight,
        coverage_penalty_weight=args.coverage_penalty_weight,
        generations=args.optimizer_generations,
        population=args.optimizer_population,
    )
    similarity_parameters, similarity_history = fit_model(
        "similarity",
        feature_maps,
        az_grid,
        el_grid,
        iterations=args.iterations,
        minimum_points=args.minimum_shared_grid_points,
        regularization_weight=args.regularization_weight,
        coverage_penalty_weight=args.coverage_penalty_weight,
        generations=args.optimizer_generations,
        population=args.optimizer_population,
    )
    history = pd.concat([translation_history, similarity_history], ignore_index=True)
    final = history.loc[history["iteration"].eq(args.iterations)]
    objective_summary = (
        final.groupby("model", observed=True)["regularized_objective"]
        .agg(["median", "mean", "max"])
        .reset_index()
    )
    medians = objective_summary.set_index("model")["median"]
    gain = float(medians["translation"] - medians["similarity"])
    selected_model = "similarity" if gain >= args.similarity_selection_threshold else "translation"
    selected_parameters = similarity_parameters if selected_model == "similarity" else translation_parameters
    selected_rows = []
    for session_id, packed in selected_parameters.items():
        selected_rows.append(
            {
                "ecephys_session_id": session_id,
                "selected_model": selected_model,
                "translation_azimuth_deg": packed[0],
                "translation_elevation_deg": packed[1],
                "rotation_deg": packed[2],
                "log_scale_azimuth": packed[3],
                "log_scale_elevation": packed[4],
                "shear": packed[5],
                "isotropic_scale": np.exp(packed[3]),
            }
        )
    selected_table = pd.DataFrame(selected_rows)
    surface_audit.to_csv(output_dir / "noncenter_feature_surface_audit.csv", index=False, float_format="%.6g")
    history.to_csv(output_dir / "noncenter_transform_history.csv", index=False, float_format="%.6g")
    objective_summary.to_csv(output_dir / "noncenter_model_objective_summary.csv", index=False, float_format="%.6g")
    selected_table.to_csv(output_dir / "selected_noncenter_transforms.csv", index=False, float_format="%.6g")
    lines = [
        "# Allen BO 1.1 non-center feature registration",
        "",
        "Transforms were fitted without SF, TF, RF-center consensus, or modulation index.",
        "The scalar fields are log2 RF area, dorsal–ventral CCF position, probe-horizontal",
        "position, RF response time-to-peak, and flash first-spike latency. Feature weights",
        "follow the independent spatial-gradient audit; the two latency fields are weak",
        "regularizers.",
        "",
        "Translation-only and tightly bounded similarity models were both fitted. Similarity",
        "is selected only when its median regularized non-center objective improves by at least",
        f"{args.similarity_selection_threshold:.3f}.",
        "",
        f"Selected model: **{selected_model}**. Median objective gain over translation: {gain:+.3f}.",
        "",
        "The selected transform is an exploratory landmark-based registration. SF/TF were not",
        "used for fitting or model selection and can therefore be used as independent outcomes.",
    ]
    (output_dir / "ALLEN_BO11_NONCENTER_SIMILARITY_ALIGNMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_noncenter_similarity_alignment",
        "status": "exploratory non-center-feature registration; tuning held out",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "features": FEATURES,
            "groups": list(GROUPS),
            "translation_bounds": TRANSLATION_BOUNDS,
            "similarity_bounds": [list(bounds) for bounds in SIMILARITY_BOUNDS],
            "bandwidth_deg": args.bandwidth_deg,
            "minimum_effective_local_units": args.minimum_effective_local_units,
            "iterations": args.iterations,
            "optimizer_generations": args.optimizer_generations,
            "optimizer_population": args.optimizer_population,
            "regularization_weight": args.regularization_weight,
            "coverage_penalty_weight": args.coverage_penalty_weight,
            "similarity_selection_threshold": args.similarity_selection_threshold,
            "selected_model": selected_model,
            "tuning_used": False,
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen non-center registration written to {output_dir}; selected {selected_model}")


if __name__ == "__main__":
    main()
