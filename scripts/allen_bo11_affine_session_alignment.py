#!/usr/bin/env python3
"""Test RF-only affine session alignment against independent SF/TF agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import (  # noqa: E402
    DEFAULT_AUDIT,
    HVA_ORDER,
    PREFERENCES,
    load_preference_units,
    session_balanced_gaussian_surface,
)
from scripts.render_allen_bo11_simultaneous_v1_hva_session_maps import (  # noqa: E402
    simultaneous_v1_hva_sessions,
)


DEFAULT_OUTPUT = DEFAULT_AUDIT / "affine_session_alignment"
GROUPS = ("V1", "HVA pooled")
COORDINATE_SCALE_DEG = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-session-units", type=int, default=10)
    parser.add_argument("--minimum-reference-units", type=int, default=20)
    parser.add_argument("--minimum-reference-sessions", type=float, default=3.0)
    parser.add_argument("--minimum-shared-grid-points", type=int, default=50)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def area_rf_centers(units: pd.DataFrame) -> pd.DataFrame:
    centers = (
        units.groupby(["ecephys_session_id", "area"], observed=True)
        .agg(
            center_azimuth_deg=("azimuth_rf", "median"),
            center_elevation_deg=("elevation_rf", "median"),
            rf_units=("ecephys_unit_id", "size"),
            rf_probes=("ecephys_probe_id", "nunique"),
        )
        .reset_index()
    )
    return centers


def leave_one_session_consensus(centers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session_id in sorted(centers["ecephys_session_id"].unique()):
        reference = centers.loc[centers["ecephys_session_id"].ne(session_id)]
        consensus = (
            reference.groupby("area", observed=True)
            .agg(
                target_azimuth_deg=("center_azimuth_deg", "median"),
                target_elevation_deg=("center_elevation_deg", "median"),
                target_sessions=("ecephys_session_id", "nunique"),
            )
            .reset_index()
        )
        observed = centers.loc[centers["ecephys_session_id"].eq(session_id)]
        joined = observed.merge(consensus, on="area", how="inner", validate="one_to_one")
        joined["ecephys_session_id"] = session_id
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def fit_regularized_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit row-vector target = source @ linear + translation near identity."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    normalized = source / COORDINATE_SCALE_DEG
    design = np.column_stack([normalized, np.ones(len(source))])
    delta = (target - source) / COORDINATE_SCALE_DEG
    root_weight = np.sqrt(weights / np.mean(weights))[:, None]
    weighted_design = design * root_weight
    weighted_delta = delta * root_weight
    penalty = np.diag([ridge_lambda, ridge_lambda, ridge_lambda * 0.1])
    theta = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_delta,
    )
    linear = np.eye(2) + theta[:2, :]
    translation = COORDINATE_SCALE_DEG * theta[2, :]
    return linear, translation


def apply_affine(points: np.ndarray, linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ linear + translation


def select_ridge_lambda(
    correspondences: pd.DataFrame,
    candidates: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> pd.DataFrame:
    rows = []
    for ridge_lambda in candidates:
        errors = []
        evidence = []
        for _, session in correspondences.groupby("ecephys_session_id", observed=True):
            if len(session) < 3:
                continue
            for held_out in session.index:
                training = session.drop(index=held_out)
                source = training[["center_azimuth_deg", "center_elevation_deg"]].to_numpy(float)
                target = training[["target_azimuth_deg", "target_elevation_deg"]].to_numpy(float)
                weights = np.sqrt(training["rf_units"].to_numpy(float))
                linear, translation = fit_regularized_affine(
                    source, target, weights, ridge_lambda
                )
                test_source = session.loc[[held_out], ["center_azimuth_deg", "center_elevation_deg"]].to_numpy(float)
                test_target = session.loc[[held_out], ["target_azimuth_deg", "target_elevation_deg"]].to_numpy(float)
                prediction = apply_affine(test_source, linear, translation)
                errors.append(float(np.linalg.norm(prediction - test_target)))
                evidence.append(float(np.sqrt(session.loc[held_out, "rf_units"])))
        errors_array = np.asarray(errors)
        evidence_array = np.asarray(evidence)
        rows.append(
            {
                "ridge_lambda": ridge_lambda,
                "held_out_area_predictions": len(errors_array),
                "weighted_mean_error_deg": np.average(errors_array, weights=evidence_array),
                "weighted_median_error_deg": weighted_quantile(
                    errors_array, evidence_array, 0.5
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["selected"] = result["weighted_mean_error_deg"].eq(
        result["weighted_mean_error_deg"].min()
    )
    return result


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))


def estimate_transforms(
    correspondences: pd.DataFrame,
    ridge_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transform_rows = []
    residual_rows = []
    for session_id, session in correspondences.groupby("ecephys_session_id", observed=True):
        source = session[["center_azimuth_deg", "center_elevation_deg"]].to_numpy(float)
        target = session[["target_azimuth_deg", "target_elevation_deg"]].to_numpy(float)
        weights = np.sqrt(session["rf_units"].to_numpy(float))
        linear, translation = fit_regularized_affine(source, target, weights, ridge_lambda)
        aligned = apply_affine(source, linear, translation)
        before = np.linalg.norm(source - target, axis=1)
        after = np.linalg.norm(aligned - target, axis=1)
        singular_values = np.linalg.svd(linear, compute_uv=False)
        transform_rows.append(
            {
                "ecephys_session_id": int(session_id),
                "shared_areas": len(session),
                "ridge_lambda": ridge_lambda,
                "linear_az_to_az": linear[0, 0],
                "linear_az_to_el": linear[0, 1],
                "linear_el_to_az": linear[1, 0],
                "linear_el_to_el": linear[1, 1],
                "translation_azimuth_deg": translation[0],
                "translation_elevation_deg": translation[1],
                "determinant": np.linalg.det(linear),
                "minimum_singular_value": singular_values.min(),
                "maximum_singular_value": singular_values.max(),
                "weighted_rf_center_error_before_deg": np.average(before, weights=weights),
                "weighted_rf_center_error_after_deg": np.average(after, weights=weights),
            }
        )
        local = session[["area", "rf_units"]].copy()
        local["ecephys_session_id"] = int(session_id)
        local["rf_center_error_before_deg"] = before
        local["rf_center_error_after_deg"] = after
        residual_rows.append(local)
    return pd.DataFrame(transform_rows), pd.concat(residual_rows, ignore_index=True)


def add_aligned_coordinates(units: pd.DataFrame, transforms: pd.DataFrame) -> pd.DataFrame:
    result = units.merge(transforms, on="ecephys_session_id", how="inner", validate="many_to_one")
    source = result[["azimuth_rf", "elevation_rf"]].to_numpy(float)
    aligned = np.empty_like(source)
    for session_id, indices in result.groupby("ecephys_session_id", observed=True).groups.items():
        row = transforms.loc[transforms["ecephys_session_id"].eq(session_id)].iloc[0]
        linear = np.array(
            [
                [row.linear_az_to_az, row.linear_az_to_el],
                [row.linear_el_to_az, row.linear_el_to_el],
            ]
        )
        translation = np.array([row.translation_azimuth_deg, row.translation_elevation_deg])
        aligned[result.index.get_indexer(indices)] = apply_affine(
            result.loc[indices, ["azimuth_rf", "elevation_rf"]].to_numpy(float),
            linear,
            translation,
        )
    result["aligned_azimuth_rf"] = aligned[:, 0]
    result["aligned_elevation_rf"] = aligned[:, 1]
    return result


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)
    covariance = np.average((x - mean_x) * (y - mean_y), weights=weights)
    variance_x = np.average(np.square(x - mean_x), weights=weights)
    variance_y = np.average(np.square(y - mean_y), weights=weights)
    if variance_x <= 0 or variance_y <= 0:
        return np.nan
    return float(covariance / np.sqrt(variance_x * variance_y))


def agreement_metrics(
    units: pd.DataFrame,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_session_units: int,
    minimum_reference_units: int,
    minimum_reference_sessions: float,
    minimum_shared_grid_points: int,
) -> pd.DataFrame:
    rows = []
    coordinate_sets = {
        "raw": ("azimuth_rf", "elevation_rf"),
        "affine": ("aligned_azimuth_rf", "aligned_elevation_rf"),
    }
    for coordinate_model, coordinate_columns in coordinate_sets.items():
        for group_label in GROUPS:
            group_mask = units["area"].eq("V1") if group_label == "V1" else units["area"].isin(HVA_ORDER)
            for preference, specification in PREFERENCES.items():
                population = units.loc[group_mask & units[f"tuning_eligible_{preference}"]].copy()
                values = np.log2(population[specification["column"]].to_numpy(float))
                population["preference_log2"] = values
                for session_id in sorted(population["ecephys_session_id"].unique()):
                    held_out = population.loc[population["ecephys_session_id"].eq(session_id)]
                    reference = population.loc[population["ecephys_session_id"].ne(session_id)]
                    session_surface = session_balanced_gaussian_surface(
                        held_out[list(coordinate_columns)].to_numpy(float),
                        held_out["preference_log2"].to_numpy(float),
                        held_out["ecephys_session_id"].to_numpy(),
                        grid_points,
                        bandwidth_deg=bandwidth_deg,
                        minimum_effective_sessions=1.0,
                        minimum_local_units=minimum_session_units,
                    )
                    reference_surface = session_balanced_gaussian_surface(
                        reference[list(coordinate_columns)].to_numpy(float),
                        reference["preference_log2"].to_numpy(float),
                        reference["ecephys_session_id"].to_numpy(),
                        grid_points,
                        bandwidth_deg=bandwidth_deg,
                        minimum_effective_sessions=minimum_reference_sessions,
                        minimum_local_units=minimum_reference_units,
                    )
                    shared = (
                        session_surface["supported"]
                        & reference_surface["supported"]
                        & np.isfinite(session_surface["estimate_log2"])
                        & np.isfinite(reference_surface["estimate_log2"])
                    )
                    count = int(shared.sum())
                    supported = count >= minimum_shared_grid_points
                    if supported:
                        x = session_surface["estimate_log2"][shared]
                        y = reference_surface["estimate_log2"][shared]
                        weights = np.sqrt(
                            session_surface["local_units"][shared]
                            * reference_surface["local_units"][shared]
                        ) * np.sqrt(reference_surface["effective_sessions"][shared])
                        difference = x - y
                        correlation = weighted_correlation(x, y, weights)
                        rmse = float(np.sqrt(np.average(np.square(difference), weights=weights)))
                        bias = float(np.average(difference, weights=weights))
                    else:
                        correlation = rmse = bias = np.nan
                    rows.append(
                        {
                            "ecephys_session_id": int(session_id),
                            "group": group_label,
                            "preference": preference,
                            "coordinate_model": coordinate_model,
                            "session_units": len(held_out),
                            "reference_units": len(reference),
                            "shared_grid_points": count,
                            "shared_grid_fraction": count / len(grid_points),
                            "comparison_supported": supported,
                            "weighted_surface_correlation": correlation,
                            "weighted_rmse_octaves": rmse,
                            "weighted_bias_octaves": bias,
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def summarize_agreement(metrics: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    wide = metrics.pivot(
        index=["ecephys_session_id", "group", "preference"],
        columns="coordinate_model",
        values=["weighted_surface_correlation", "weighted_rmse_octaves"],
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    rng = np.random.default_rng(20260811)
    rows = []
    for (group, preference), selected in wide.groupby(["group", "preference"], observed=True):
        valid = selected.dropna(
            subset=[
                "weighted_surface_correlation_raw",
                "weighted_surface_correlation_affine",
                "weighted_rmse_octaves_raw",
                "weighted_rmse_octaves_affine",
            ]
        ).copy()
        correlation_delta = (
            valid["weighted_surface_correlation_affine"]
            - valid["weighted_surface_correlation_raw"]
        ).to_numpy(float)
        rmse_delta = (
            valid["weighted_rmse_octaves_affine"]
            - valid["weighted_rmse_octaves_raw"]
        ).to_numpy(float)
        corr_ci = bootstrap_median_ci(correlation_delta, bootstrap_iterations, rng)
        rmse_ci = bootstrap_median_ci(rmse_delta, bootstrap_iterations, rng)
        corr_p = wilcoxon(correlation_delta).pvalue if np.any(correlation_delta) else 1.0
        rmse_p = wilcoxon(rmse_delta).pvalue if np.any(rmse_delta) else 1.0
        rows.append(
            {
                "group": group,
                "preference": preference,
                "paired_sessions": len(valid),
                "median_correlation_raw": valid["weighted_surface_correlation_raw"].median(),
                "median_correlation_affine": valid["weighted_surface_correlation_affine"].median(),
                "median_correlation_change": np.median(correlation_delta),
                "correlation_change_ci_low": corr_ci[0],
                "correlation_change_ci_high": corr_ci[1],
                "correlation_change_wilcoxon_p": corr_p,
                "median_rmse_raw_octaves": valid["weighted_rmse_octaves_raw"].median(),
                "median_rmse_affine_octaves": valid["weighted_rmse_octaves_affine"].median(),
                "median_rmse_change_octaves": np.median(rmse_delta),
                "rmse_change_ci_low": rmse_ci[0],
                "rmse_change_ci_high": rmse_ci[1],
                "rmse_change_wilcoxon_p": rmse_p,
            }
        )
    return pd.DataFrame(rows)


def render_agreement(metrics: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    wide = metrics.pivot(
        index=["ecephys_session_id", "group", "preference"],
        columns="coordinate_model",
        values="weighted_surface_correlation",
    ).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
    for row, group in enumerate(GROUPS):
        for column, preference in enumerate(("sf", "tf")):
            ax = axes[row, column]
            selected = wide.loc[
                wide["group"].eq(group) & wide["preference"].eq(preference)
            ].dropna(subset=["raw", "affine"])
            ax.scatter(selected["raw"], selected["affine"], s=27, alpha=0.72, color="#35618D")
            limits = (-1.0, 1.0)
            ax.plot(limits, limits, color="#777777", linestyle="--", linewidth=1)
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            row_summary = summary.loc[
                summary["group"].eq(group) & summary["preference"].eq(preference)
            ].iloc[0]
            ax.set_title(f"{group} · {preference.upper()} · n={len(selected)} sessions")
            ax.text(
                0.04,
                0.94,
                f"median Δr={row_summary.median_correlation_change:+.3f}\n"
                f"median ΔRMSE={row_summary.median_rmse_change_octaves:+.3f} oct",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            if row == 1:
                ax.set_xlabel("raw leave-one-session-out correlation")
            if column == 0:
                ax.set_ylabel("RF-affine leave-one-session-out correlation")
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Does RF-only affine session alignment improve independent SF/TF agreement?\n"
        "density/evidence-weighted surfaces; diagonal = no change",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    cv: pd.DataFrame,
    transforms: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_lambda = cv.loc[cv["selected"], "ridge_lambda"].iloc[0]
    lines = [
        "# REJECTED correction — Allen BO 1.1 RF-center affine diagnostic",
        "",
        "**Status: invalid as a coordinate correction.** RF centers were used both as the",
        "coordinates to transform and as the area-consensus registration landmarks. Holding",
        "SF/TF out of transform fitting does not remove that circular retinotopic assumption.",
        "This output is retained only as a documented failure mode and must not generate",
        "aligned primary maps.",
        "",
        "One global affine transform per session was estimated only from RF evidence.",
        "Area-specific RF centers were matched to leave-one-session-out area consensus",
        "centers with square-root unit-count weights. Ridge strength was selected by",
        "held-out-area RF prediction, without consulting SF or TF. The selected lambda",
        f"was {selected_lambda:g}.",
        "",
        "The unchanged transform was then applied to every V1/HVA unit. Agreement was",
        "evaluated against leave-one-session-out SF/TF templates with weights based on",
        "local unit density and effective reference-session evidence.",
        "",
        "| Group | Preference | Sessions | Median r raw | Median r affine | Median Δr (95% bootstrap CI) | Median RMSE raw | Median RMSE affine | Median ΔRMSE (octaves) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.group} | {row.preference.upper()} | {int(row.paired_sessions)} | "
            f"{row.median_correlation_raw:.3f} | {row.median_correlation_affine:.3f} | "
            f"{row.median_correlation_change:+.3f} ({row.correlation_change_ci_low:+.3f}, "
            f"{row.correlation_change_ci_high:+.3f}) | {row.median_rmse_raw_octaves:.3f} | "
            f"{row.median_rmse_affine_octaves:.3f} | {row.median_rmse_change_octaves:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "Weighted RMSE decreases after affine alignment for all four group-by-frequency",
            "combinations, but spatial-pattern correlation does not improve consistently:",
            "both V1 correlations decline, pooled-HVA SF is essentially unchanged, and only",
            "pooled-HVA TF shows a positive median correlation change. The RF-selected transforms",
            "also compress the visual field strongly. Therefore the RMSE reduction is not accepted",
            "as evidence that a free global affine transform recovers a common tuning map.",
            "",
            "## Interpretation boundary",
            "",
            "This is a sensitivity analysis, not gaze calibration. A fitted affine transform",
            "can absorb true retinotopic targeting differences as well as screen/eye geometry.",
            "The independent tuning evaluation limits circularity, but area-center consensus",
            "and affine regularization remain modeling assumptions. Improvement should be",
            "accepted only if it is consistent across SF and TF, V1 and HVAs, correlation and",
            "RMSE, without extreme determinants or singular values.",
            "",
            f"Across sessions, affine determinants ranged {transforms.determinant.min():.3f}–"
            f"{transforms.determinant.max():.3f}; singular values ranged "
            f"{transforms.minimum_singular_value.min():.3f}–"
            f"{transforms.maximum_singular_value.max():.3f}. "
            f"{int(transforms.determinant.lt(0).sum())} session(s) reverse orientation.",
            "A physically constrained translation/rotation/scale sensitivity is required before",
            "using an alignment in the primary maps.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    units, unit_path, audit_manifest = load_preference_units(
        args.audit_dir.resolve(),
        args.config,
        minimum_lifetime_sparseness=0.1,
        minimum_stimulus_firing_rate=0.1,
        require_unique_preference=True,
    )
    sessions = simultaneous_v1_hva_sessions(units)
    units = units.loc[units["ecephys_session_id"].isin(sessions)].copy()
    centers = area_rf_centers(units)
    correspondences = leave_one_session_consensus(centers)
    cv = select_ridge_lambda(correspondences)
    selected_lambda = float(cv.loc[cv["selected"], "ridge_lambda"].iloc[0])
    transforms, residuals = estimate_transforms(correspondences, selected_lambda)
    aligned_units = add_aligned_coordinates(units, transforms)
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    metrics = agreement_metrics(
        aligned_units,
        grid_points,
        bandwidth_deg=args.bandwidth_deg,
        minimum_session_units=args.minimum_session_units,
        minimum_reference_units=args.minimum_reference_units,
        minimum_reference_sessions=args.minimum_reference_sessions,
        minimum_shared_grid_points=args.minimum_shared_grid_points,
    )
    summary = summarize_agreement(metrics, args.bootstrap_iterations)
    cv.to_csv(output_dir / "affine_regularization_rf_cross_validation.csv", index=False, float_format="%.6g")
    transforms.to_csv(output_dir / "session_affine_transforms.csv", index=False, float_format="%.6g")
    residuals.to_csv(output_dir / "rf_area_center_residuals.csv", index=False, float_format="%.6g")
    metrics.to_csv(output_dir / "sf_tf_leave_one_session_agreement.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "sf_tf_agreement_summary.csv", index=False, float_format="%.6g")
    figure_path = output_dir / "Figure_allen_bo11_affine_sf_tf_agreement.png"
    render_agreement(metrics, summary, figure_path)
    write_report(
        cv,
        transforms,
        summary,
        output_dir / "ALLEN_BO11_AFFINE_SESSION_ALIGNMENT.md",
    )
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_affine_session_alignment",
        "status": "rejected diagnostic; RF-center landmarks are circular for coordinate correction",
        "input": {"path": str(unit_path.resolve()), "sha256": sha256(unit_path.resolve())},
        "audit_manifest_sha256": hashlib.sha256(
            json.dumps(audit_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "ridge_candidates": cv["ridge_lambda"].tolist(),
            "selected_ridge_lambda": selected_lambda,
            "rf_center_weight": "sqrt unit count",
            "tuning_agreement_weight": "sqrt held-out/reference local units times sqrt effective reference sessions",
            "bandwidth_deg": args.bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_session_units": args.minimum_session_units,
            "minimum_reference_units": args.minimum_reference_units,
            "minimum_reference_sessions": args.minimum_reference_sessions,
            "minimum_shared_grid_points": args.minimum_shared_grid_points,
            "bootstrap_iterations": args.bootstrap_iterations,
            "tuning_leakage": "none; transforms and ridge selected from RF centers only",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Allen RF-affine agreement analysis written to {output_dir}")


if __name__ == "__main__":
    main()
