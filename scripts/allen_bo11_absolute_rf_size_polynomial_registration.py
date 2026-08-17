#!/usr/bin/env python3
"""Register Allen BO 1.1 sessions to a shared absolute RF-size polynomial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from scripts.allen_bo11_tuning_driven_limited_affine import (
    PARAMETER_NAMES,
    evaluate_model,
    load_maps,
    summarize,
)
from scripts.render_allen_bo11_registration_comparison import DEFAULT_SURFACE_GRID
from scripts.render_allen_bo11_v1_rf_size_interior import DEFAULT_INPUT, prepare_population


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_COHORT = AUDIT / "ccf_retinotopy_alignment" / "selected_ccf_retinotopy_transforms.csv"
DEFAULT_OUTPUT = AUDIT / "v1_absolute_rf_size_polynomial_edge35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tuning-grid", type=Path, default=DEFAULT_SURFACE_GRID)
    parser.add_argument("--session-table", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-exclusion-deg", type=float, default=35.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--minimum-units-for-shift", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--polynomial-degree", type=int, choices=(1, 2), help="Force model degree; default selects degree and penalty by CV")
    parser.add_argument("--regularization-weight", type=float, help="Force translation penalty after CV audit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polynomial_features(azimuth: np.ndarray, elevation: np.ndarray, degree: int) -> np.ndarray:
    x = (np.asarray(azimuth) - 50.0) / 10.0
    y = (np.asarray(elevation) - 10.0) / 10.0
    columns = [np.ones_like(x), x, y]
    if degree >= 2:
        columns.extend([x * x, x * y, y * y])
    return np.column_stack(columns)


def fit_model(
    table: pd.DataFrame,
    sessions: list[int],
    *,
    degree: int,
    regularization_weight: float,
    translation_bound_deg: float,
    minimum_units_for_shift: int,
) -> dict[str, object]:
    session_lookup = {session_id: index for index, session_id in enumerate(sessions)}
    session_index = table["ecephys_session_id"].map(session_lookup).to_numpy(int)
    counts = np.bincount(session_index, minlength=len(sessions))
    eligible_indices = np.flatnonzero(counts >= minimum_units_for_shift)
    eligible_lookup = {session_index: index for index, session_index in enumerate(eligible_indices)}
    local_shift_index = np.array([eligible_lookup.get(value, -1) for value in session_index])
    y = table["log2_rf_area_deg2"].to_numpy(float)
    azimuth = table["azimuth_rf"].to_numpy(float)
    elevation = table["elevation_rf"].to_numpy(float)
    weights = 1.0 / np.sqrt(np.maximum(counts[session_index], 1))
    feature_count = 3 if degree == 1 else 6
    initial_features = polynomial_features(azimuth, elevation, degree)
    beta = np.linalg.lstsq(initial_features * weights[:, None], y * weights, rcond=None)[0]
    initial = np.r_[beta, np.zeros(2 * len(eligible_indices))]
    lower = np.r_[np.full(feature_count, -np.inf), np.full(2 * len(eligible_indices), -translation_bound_deg)]
    upper = np.r_[np.full(feature_count, np.inf), np.full(2 * len(eligible_indices), translation_bound_deg)]

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifts_az = np.zeros(len(sessions))
        shifts_el = np.zeros(len(sessions))
        shifts_az[eligible_indices] = parameters[feature_count : feature_count + len(eligible_indices)]
        shifts_el[eligible_indices] = parameters[feature_count + len(eligible_indices) :]
        return parameters[:feature_count], shifts_az, shifts_el

    def residuals(parameters: np.ndarray) -> np.ndarray:
        local_beta, shifts_az, shifts_el = unpack(parameters)
        prediction = polynomial_features(
            azimuth + shifts_az[session_index], elevation + shifts_el[session_index], degree
        ) @ local_beta
        # Equal total squared-error weight per session. The shift penalty fixes
        # the otherwise arbitrary common translation gauge at mean zero.
        data_residual = weights * (y - prediction)
        penalty_scale = np.sqrt(regularization_weight / max(len(eligible_indices), 1))
        shift_penalty = penalty_scale * np.r_[shifts_az[eligible_indices] / 10.0,
                                              shifts_el[eligible_indices] / 10.0]
        return np.r_[data_residual, shift_penalty]

    fit = least_squares(
        residuals, initial, bounds=(lower, upper), max_nfev=4000,
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    beta, shifts_az, shifts_el = unpack(fit.x)
    prediction = polynomial_features(
        azimuth + shifts_az[session_index], elevation + shifts_el[session_index], degree
    ) @ beta
    return {
        "beta": beta,
        "translation_azimuth_deg": shifts_az,
        "translation_elevation_deg": shifts_el,
        "prediction": prediction,
        "residual": y - prediction,
        "counts": counts,
        "eligible": counts >= minimum_units_for_shift,
        "cost": fit.cost,
        "success": fit.success,
    }


def make_folds(table: pd.DataFrame, folds: int) -> np.ndarray:
    labels = np.zeros(len(table), dtype=int)
    for session_id, indices in table.groupby("ecephys_session_id", observed=True).groups.items():
        selected = np.asarray(list(indices))
        rng = np.random.default_rng(20260817 + int(session_id))
        labels[selected] = np.arange(len(selected))[rng.permutation(len(selected))] % folds
    return labels


def cross_validate(
    table: pd.DataFrame,
    sessions: list[int],
    folds: int,
    translation_bound_deg: float,
    minimum_units_for_shift: int,
) -> pd.DataFrame:
    fold_labels = make_folds(table, folds)
    candidates = (0.01, 0.03, 0.08, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    rows = []
    for degree in (1, 2):
        for penalty in candidates:
            residuals = []
            residual_sessions = []
            for fold in range(folds):
                train = table.loc[fold_labels != fold]
                test = table.loc[fold_labels == fold]
                fitted = fit_model(
                    train, sessions, degree=degree, regularization_weight=penalty,
                    translation_bound_deg=translation_bound_deg,
                    minimum_units_for_shift=max(3, minimum_units_for_shift - 1),
                )
                lookup = {session_id: index for index, session_id in enumerate(sessions)}
                indices = test["ecephys_session_id"].map(lookup).to_numpy(int)
                prediction = polynomial_features(
                    test["azimuth_rf"].to_numpy(float) + fitted["translation_azimuth_deg"][indices],
                    test["elevation_rf"].to_numpy(float) + fitted["translation_elevation_deg"][indices],
                    degree,
                ) @ fitted["beta"]
                residuals.extend(test["log2_rf_area_deg2"].to_numpy(float) - prediction)
                residual_sessions.extend(test["ecephys_session_id"].astype(int))
            scored = pd.DataFrame({"residual": residuals, "session": residual_sessions})
            session_rmse = scored.groupby("session")["residual"].apply(lambda value: np.sqrt(np.mean(np.square(value))))
            rows.append({
                "degree": degree, "regularization_weight": penalty,
                "session_balanced_cv_rmse_log2": float(np.sqrt(np.mean(np.square(session_rmse)))),
                "median_session_cv_rmse_log2": float(session_rmse.median()),
            })
    return pd.DataFrame(rows)


def transform_table(
    fitted: dict[str, object], sessions: list[int], penalty: float, degree: int
) -> pd.DataFrame:
    residual = np.asarray(fitted["residual"])
    rows = []
    for index, session_id in enumerate(sessions):
        rows.append({
            "ecephys_session_id": session_id,
            "selected_model": f"absolute_log2_rf_area_degree_{degree}_translation",
            "translation_azimuth_deg": fitted["translation_azimuth_deg"][index],
            "translation_elevation_deg": fitted["translation_elevation_deg"][index],
            "rotation_deg": 0.0,
            "log_scale_azimuth": 0.0,
            "log_scale_elevation": 0.0,
            "shear": 0.0,
            "interior_v1_units": int(fitted["counts"][index]),
            "fit_identifiable": bool(fitted["eligible"][index]),
            "regularization_weight": penalty,
            "translation_at_bound": bool(
                abs(fitted["translation_azimuth_deg"][index]) >= 29.4
                or abs(fitted["translation_elevation_deg"][index]) >= 29.4
            ),
        })
    return pd.DataFrame(rows)


def render_diagnostic(
    population: pd.DataFrame,
    fitted: dict[str, object],
    transforms: pd.DataFrame,
    cv: pd.DataFrame,
    degree: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.8))
    azimuth = np.linspace(15, 85, 141)
    elevation = np.linspace(-25, 45, 141)
    az_mesh, el_mesh = np.meshgrid(azimuth, elevation)
    log_area = polynomial_features(az_mesh.ravel(), el_mesh.ravel(), degree) @ fitted["beta"]
    area = np.exp2(log_area.reshape(az_mesh.shape))
    interior = (az_mesh >= 45) & (az_mesh <= 55) & (el_mesh >= 5) & (el_mesh <= 15)
    display = np.where(interior, area, np.nan)
    limits = np.quantile(area[interior], [.02, .98])
    image = axes[0, 0].pcolormesh(azimuth, elevation, display, shading="gouraud",
                                  cmap="YlGnBu", norm=Normalize(*limits))
    axes[0, 0].quiver(
        population.groupby("ecephys_session_id")["azimuth_rf"].mean(),
        population.groupby("ecephys_session_id")["elevation_rf"].mean(),
        transforms["translation_azimuth_deg"], transforms["translation_elevation_deg"],
        angles="xy", scale_units="xy", scale=1, color="#6a3d9a", alpha=.75,
    )
    model_label = "planar" if degree == 1 else "quadratic"
    axes[0, 0].set(title=f"Shared absolute {model_label} RF-size field\nand inferred session translations")
    figure.colorbar(image, ax=axes[0, 0], label="Predicted RF area (deg²)", extend="both")

    axes[0, 1].scatter(
        population["log2_rf_area_deg2"], fitted["prediction"], s=13, alpha=.3, color="#356a9a"
    )
    limits_scatter = [population["log2_rf_area_deg2"].min(), population["log2_rf_area_deg2"].max()]
    axes[0, 1].plot(limits_scatter, limits_scatter, color="#555555", linewidth=1)
    axes[0, 1].set(xlabel="Observed log₂ RF area", ylabel="Model-predicted log₂ RF area",
                   title="Unit-level fit")

    for degree, color, label in ((1, "#999999", "Plane"), (2, "#4c78a8", "Quadratic")):
        selected = cv.loc[cv.degree.eq(degree)]
        axes[1, 0].plot(selected["regularization_weight"], selected["session_balanced_cv_rmse_log2"],
                        marker="o", color=color, label=label)
    axes[1, 0].set(xscale="log", xlabel="Translation regularization weight",
                   ylabel="Session-balanced held-out RMSE (log₂ area)",
                   title="Within-session-fold cross-validation")
    axes[1, 0].legend()

    axes[1, 1].quiver(
        np.zeros(len(transforms)), np.zeros(len(transforms)),
        transforms["translation_azimuth_deg"], transforms["translation_elevation_deg"],
        angles="xy", scale_units="xy", scale=1, color="#6a3d9a", alpha=.75,
    )
    axes[1, 1].axhline(0, color="#888888", linewidth=.8)
    axes[1, 1].axvline(0, color="#888888", linewidth=.8)
    extent = max(12.0, float(np.max(np.abs(transforms[["translation_azimuth_deg", "translation_elevation_deg"]]))) + 2)
    axes[1, 1].set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
                   xlabel="Azimuth translation (deg)", ylabel="Elevation translation (deg)",
                   title="Absolute-RF-size translations")
    for ax in axes.ravel():
        ax.grid(alpha=.16)
    axes[0, 0].set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", aspect="equal")
    figure.suptitle(f"Allen BO 1.1: absolute RF-size {model_label} registration · 35° exclusion", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.session_table.resolve())
    if "ccf_available" in cohort:
        cohort = cohort.loc[cohort["ccf_available"].fillna(False).astype(bool)]
    cohort_sessions = set(cohort["ecephys_session_id"].astype(int))
    tuning_maps, tuning_az, tuning_el = load_maps(args.tuning_grid.resolve())
    sessions = sorted(cohort_sessions & {key[0] for key in tuning_maps})
    population = prepare_population(pd.read_csv(args.support.resolve(), low_memory=False))
    population = population.loc[
        population["ecephys_session_id"].isin(sessions)
        & population["distance_to_nearest_grid_edge_deg"].ge(args.edge_exclusion_deg)
    ].dropna(subset=["log2_rf_area_deg2"]).copy().reset_index(drop=True)
    sessions = [session_id for session_id in sessions if session_id in set(population["ecephys_session_id"])]
    tuning_maps = {key: value for key, value in tuning_maps.items() if key[0] in sessions}

    cv = cross_validate(
        population, sessions, args.cv_folds, args.translation_bound_deg, args.minimum_units_for_shift
    )
    selection_table = cv if args.polynomial_degree is None else cv.loc[cv.degree.eq(args.polynomial_degree)]
    selected_index = selection_table["session_balanced_cv_rmse_log2"].idxmin()
    selected_degree = int(cv.loc[selected_index, "degree"])
    selected_penalty = (
        float(args.regularization_weight)
        if args.regularization_weight is not None
        else float(cv.loc[selected_index, "regularization_weight"])
    )
    fitted = fit_model(
        population, sessions, degree=selected_degree, regularization_weight=selected_penalty,
        translation_bound_deg=args.translation_bound_deg,
        minimum_units_for_shift=args.minimum_units_for_shift,
    )
    transforms = transform_table(fitted, sessions, selected_penalty, selected_degree)

    parameters = {
        int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float)
        for _, row in transforms.iterrows()
    }
    identity = {session_id: np.zeros(6) for session_id in sessions}
    raw_metrics = evaluate_model(tuning_maps, identity, tuning_az, tuning_el, 50, "raw")
    aligned_metrics = evaluate_model(tuning_maps, parameters, tuning_az, tuning_el, 50,
                                     "absolute_rf_size_polynomial")
    tuning_summary = summarize(raw_metrics, aligned_metrics)

    transforms.to_csv(output_dir / "selected_absolute_rf_size_polynomial_translations.csv",
                      index=False, float_format="%.6g")
    cv.to_csv(output_dir / "absolute_rf_size_polynomial_cross_validation.csv",
              index=False, float_format="%.6g")
    tuning_summary.to_csv(output_dir / "absolute_rf_size_polynomial_tuning_summary.csv",
                          index=False, float_format="%.6g")
    term_names = ["intercept", "azimuth", "elevation"]
    if selected_degree == 2:
        term_names.extend(["azimuth_squared", "azimuth_by_elevation", "elevation_squared"])
    coefficients = pd.DataFrame({
        "term": term_names,
        "coefficient_log2_area": fitted["beta"],
    })
    coefficients.to_csv(output_dir / "absolute_rf_size_polynomial_coefficients.csv",
                        index=False, float_format="%.6g")
    render_diagnostic(
        population, fitted, transforms, cv,
        selected_degree,
        output_dir / "Figure_allen_bo11_absolute_rf_size_polynomial_registration.png",
    )
    manifest = {
        "checkpoint": "06c_allen_bo11_absolute_rf_size_polynomial_registration",
        "status": "exploratory absolute-RF-size low-dimensional registration",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "tuning_grid": {"path": str(args.tuning_grid.resolve()), "sha256": sha256(args.tuning_grid.resolve())},
            "session_table": {"path": str(args.session_table.resolve()), "sha256": sha256(args.session_table.resolve())},
        },
        "parameters": {
            "sessions": sessions, "edge_exclusion_deg": args.edge_exclusion_deg,
            "response": "absolute log2 RF area; no session centering, scaling, or intercept",
            "polynomial_degree": selected_degree, "degree_forced": args.polynomial_degree is not None,
            "selected_regularization_weight": selected_penalty,
            "regularization_forced": args.regularization_weight is not None,
            "translation_bound_deg": args.translation_bound_deg,
            "minimum_units_for_shift": args.minimum_units_for_shift,
            "selection": "minimum session-balanced within-session-fold CV RMSE",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
