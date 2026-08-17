#!/usr/bin/env python3
"""Validate the 35-degree Allen V1 RF-size registration surface."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from scripts.allen_bo11_tuning_driven_limited_affine import template_from_maps, warp_all
from scripts.allen_bo11_v1_rf_size_translation_alignment import build_session_maps
from scripts.render_allen_bo11_registration_comparison import load_rf_size_parameters
from scripts.render_allen_bo11_rf_size_registration_breakout import (
    DISPLAY_AZ_LIMITS,
    DISPLAY_EL_LIMITS,
    build_absolute_size_maps,
)
from scripts.render_allen_bo11_v1_rf_size_interior import DEFAULT_INPUT, prepare_population


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
TRANSFORMS = AUDIT / "v1_rf_size_translation_edge35_bound30" / "selected_v1_rf_size_translations.csv"
OUTPUT = AUDIT / "v1_rf_size_surface_validation_edge35"
EDGE_EXCLUSION_DEG = 35.0


def session_center(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["centered_standardized_log2_rf_area"] = (
        result["session_standardized_log2_rf_area"]
        - result.groupby("ecephys_session_id", observed=True)["session_standardized_log2_rf_area"].transform("mean")
    )
    return result


def coordinates(table: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(table["azimuth_rf"].to_numpy(float) - 50.0) / 5.0,
                            (table["elevation_rf"].to_numpy(float) - 10.0) / 5.0])


def balanced_weights(groups: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(groups), dtype=float)
    for session_id in np.unique(groups):
        selected = groups == session_id
        weights[selected] = 1.0 / selected.sum()
    return weights


def polynomial_model(degree: int) -> object:
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=1.0, fit_intercept=True),
    )


def leave_one_session_out_predictions(table: pd.DataFrame, degree: int) -> np.ndarray:
    x = coordinates(table)
    y = table["centered_standardized_log2_rf_area"].to_numpy(float)
    groups = table["ecephys_session_id"].to_numpy(int)
    predictions = np.full(len(table), np.nan)
    for session_id in np.unique(groups):
        test = groups == session_id
        train = ~test
        model = polynomial_model(degree)
        model.fit(x[train], y[train], ridge__sample_weight=balanced_weights(groups[train]))
        local = model.predict(x[test])
        predictions[test] = local - local.mean()
    return predictions


def cv_summary(table: pd.DataFrame, degrees: tuple[int, ...] = (1, 2, 3)) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["centered_standardized_log2_rf_area"].to_numpy(float)
    sessions = table["ecephys_session_id"].to_numpy(int)
    summary_rows = []
    session_rows = []
    for degree in degrees:
        prediction = leave_one_session_out_predictions(table, degree)
        pooled_r2 = 1.0 - np.sum(np.square(y - prediction)) / np.sum(np.square(y))
        for session_id in np.unique(sessions):
            selected = sessions == session_id
            rho = np.nan
            if selected.sum() >= 3 and np.std(y[selected]) > 0 and np.std(prediction[selected]) > 0:
                rho = pearsonr(y[selected], prediction[selected]).statistic
            session_rows.append({"degree": degree, "ecephys_session_id": session_id,
                                 "units": int(selected.sum()), "pearson_r": rho})
        local = pd.DataFrame(session_rows).loc[lambda x: x.degree.eq(degree)]
        summary_rows.append({
            "degree": degree,
            "terms": int(PolynomialFeatures(degree, include_bias=False).fit(np.zeros((1, 2))).n_output_features_),
            "pooled_cv_r2": pooled_r2,
            "median_session_r": local["pearson_r"].median(),
            "sessions_with_defined_r": int(local["pearson_r"].notna().sum()),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(session_rows)


def permutation_null(table: pd.DataFrame, degree: int = 2, repetitions: int = 300) -> np.ndarray:
    rng = np.random.default_rng(20260812)
    original = table["centered_standardized_log2_rf_area"].to_numpy(float).copy()
    groups = table["ecephys_session_id"].to_numpy(int)
    results = []
    for _ in range(repetitions):
        permuted = original.copy()
        for session_id in np.unique(groups):
            selected = np.flatnonzero(groups == session_id)
            permuted[selected] = permuted[selected][rng.permutation(len(selected))]
        local = table.copy()
        local["centered_standardized_log2_rf_area"] = permuted
        prediction = leave_one_session_out_predictions(local, degree)
        results.append(1.0 - np.sum(np.square(permuted - prediction)) / np.sum(np.square(permuted)))
    return np.asarray(results)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transforms = pd.read_csv(TRANSFORMS)
    sessions = transforms["ecephys_session_id"].astype(int).tolist()
    population = prepare_population(pd.read_csv(DEFAULT_INPUT, low_memory=False))
    population = population.loc[
        population["ecephys_session_id"].isin(sessions)
        & population["distance_to_nearest_grid_edge_deg"].ge(EDGE_EXCLUSION_DEG)
    ].dropna(subset=["session_standardized_log2_rf_area"]).copy()
    population = session_center(population)

    fit_az = np.linspace(45, 55, 31)
    fit_el = np.linspace(5, 15, 31)
    kernel_maps, _ = build_session_maps(
        population, sessions, fit_az, fit_el,
        bandwidth_deg=8.0, minimum_effective_local_units=1.0,
    )
    kernel_target = template_from_maps(kernel_maps, "V1", "rf_size")

    x = coordinates(population)
    y = population["centered_standardized_log2_rf_area"].to_numpy(float)
    groups = population["ecephys_session_id"].to_numpy(int)
    quadratic = polynomial_model(2)
    quadratic.fit(x, y, ridge__sample_weight=balanced_weights(groups))
    az_mesh, el_mesh = np.meshgrid(fit_az, fit_el)
    target_x = np.column_stack([(az_mesh.ravel() - 50) / 5, (el_mesh.ravel() - 10) / 5])
    polynomial_target = quadratic.predict(target_x).reshape(az_mesh.shape)

    display_az = np.linspace(*DISPLAY_AZ_LIMITS, 61)
    display_el = np.linspace(*DISPLAY_EL_LIMITS, 61)
    absolute_maps = build_absolute_size_maps(
        population, sessions, display_az, display_el, minimum_effective_local_units=1.0
    )
    parameters = load_rf_size_parameters(TRANSFORMS, sessions)
    aligned_absolute = template_from_maps(
        warp_all(absolute_maps, parameters, display_az, display_el), "V1", "rf_size_absolute"
    )

    summary, per_session = cv_summary(population)
    null = permutation_null(population)
    quadratic_r2 = float(summary.loc[summary.degree.eq(2), "pooled_cv_r2"].iloc[0])
    permutation_p = float((1 + np.sum(null >= quadratic_r2)) / (1 + len(null)))
    summary["quadratic_permutation_p"] = np.where(summary.degree.eq(2), permutation_p, np.nan)
    summary.to_csv(OUTPUT / "polynomial_cross_validation.csv", index=False, float_format="%.6g")
    per_session.to_csv(OUTPUT / "polynomial_cross_validation_by_session.csv", index=False, float_format="%.6g")
    pd.DataFrame({"quadratic_permuted_pooled_cv_r2": null}).to_csv(
        OUTPUT / "quadratic_permutation_null.csv", index=False, float_format="%.6g"
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.2, 11.6))
    absolute = np.exp2(aligned_absolute["value"])
    finite_absolute = absolute[np.isfinite(absolute)]
    focused_limits = np.quantile(finite_absolute, [.05, .95])
    image = axes[0, 0].pcolormesh(display_az, display_el, absolute, shading="gouraud",
                                  cmap="YlGnBu", norm=Normalize(*focused_limits))
    axes[0, 0].contour(display_az, display_el, aligned_absolute["evidence"], levels=3,
                       colors="#555555", linewidths=.7, alpha=.55)
    axes[0, 0].set(title=f"3b · Registered RF-size surface · focused 5–95% scale\n{focused_limits[0]:.0f}–{focused_limits[1]:.0f} deg²")
    figure.colorbar(image, ax=axes[0, 0], label="Fitted RF area (deg²)", extend="both")

    joint = np.r_[kernel_target["value"][np.isfinite(kernel_target["value"])], polynomial_target.ravel()]
    limit = max(.05, float(np.quantile(np.abs(joint), .98)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = axes[0, 1].pcolormesh(fit_az, fit_el, kernel_target["value"], shading="gouraud",
                                  cmap="coolwarm", norm=norm)
    axes[0, 1].contour(fit_az, fit_el, kernel_target["evidence"], levels=3,
                       colors="#555555", linewidths=.7, alpha=.55)
    axes[0, 1].set(title="Current joint kernel target\n8° bandwidth on a 10° × 10° fitting window")
    figure.colorbar(image, ax=axes[0, 1], label="Within-session standardized log₂ RF area")

    image = axes[1, 0].pcolormesh(fit_az, fit_el, polynomial_target, shading="gouraud",
                                  cmap="coolwarm", norm=norm)
    axes[1, 0].set(title="Regularized quadratic target\n5 spatial terms; sessions equally weighted")
    figure.colorbar(image, ax=axes[1, 0], label="Within-session standardized log₂ RF area")

    axis = axes[1, 1]
    axis.axhline(0, color="#777777", linewidth=.9)
    axis.bar(summary["degree"], summary["pooled_cv_r2"], color="#4c78a8", width=.62)
    null_95 = float(np.quantile(null, .95))
    axis.axhline(null_95, color="#b04a3a", linestyle="--", linewidth=1.2,
                 label=f"Quadratic shuffled 95th percentile ({null_95:+.3f})")
    for row in summary.itertuples(index=False):
        axis.text(row.degree, row.pooled_cv_r2 + (.003 if row.pooled_cv_r2 >= 0 else -.006),
                  f"R² {row.pooled_cv_r2:+.3f}\nmedian r {row.median_session_r:+.2f}",
                  ha="center", va="bottom" if row.pooled_cv_r2 >= 0 else "top", fontsize=9)
    axis.set(xlabel="Polynomial degree", ylabel="Leave-one-session-out pooled R²",
             title=f"Held-out-session prediction\nquadratic permutation p={permutation_p:.3f}",
             xticks=summary["degree"])
    axis.legend(loc="lower right", fontsize=8)

    for ax in axes.ravel():
        if ax is not axes[1, 1]:
            ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", aspect="equal")
        ax.grid(alpha=.16)
    figure.suptitle(
        f"Allen BO 1.1 V1 RF-size surface validation · {EDGE_EXCLUSION_DEG:g}° exclusion · "
        f"{len(sessions)} sessions / {len(population)} units",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(OUTPUT / "Figure_allen_bo11_rf_size_surface_validation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "sessions": len(sessions), "units": len(population), "edge_exclusion_deg": EDGE_EXCLUSION_DEG,
        "focused_rf_area_scale_deg2": focused_limits.tolist(), "quadratic_cv_r2": quadratic_r2,
        "quadratic_permutation_p": permutation_p, "quadratic_null_95_cv_r2": null_95,
    }
    (OUTPUT / "validation_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
