#!/usr/bin/env python3
"""Compare point-center and analytic circular-aperture RF models.

This reuses the cached trial table and spike counts from the population gaze-RF
checkpoint.  It does not render the Gabor carrier or an aperture image.  The
aperture model integrates an axis-aligned Gaussian RF over a 10-degree-radius
disk with fixed Gauss-Legendre quadrature in x and an analytic erf integral in y.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.stats import ncx2


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
DEFAULT_INPUT = ROOT / "artifacts" / "allen_population_gaze_rf" / f"session_{SESSION_ID}"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_aperture_rf_comparison" / f"session_{SESSION_ID}"
APERTURE_RADIUS_DEG = 10.0
QUADRATURE_ORDER = 24
CENTER_EXTENSION_DEG = 20.0
SIGMA_UPPER_DEG = {"V1": 40.0, "HVA": 50.0}
PREDECLARED_CASE = 951888972


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit-units", type=int, default=None,
        help="Deterministic balanced QC subset for a quick checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)


def gaussian_disk_integral(delta_x, delta_y, sigma_x, sigma_y,
                           radius=APERTURE_RADIUS_DEG):
    """Integral of an unnormalized axis-aligned Gaussian over a circular disk."""
    delta_x = np.atleast_1d(np.asarray(delta_x, dtype=float))
    delta_y = np.atleast_1d(np.asarray(delta_y, dtype=float))
    u = (radius * _LEGENDRE_NODES)[:, None]
    weights = (radius * _LEGENDRE_WEIGHTS)[:, None]
    half_height = np.sqrt(np.maximum(radius ** 2 - u ** 2, 0.0))
    root_two = np.sqrt(2.0)
    inner_y = sigma_y * np.sqrt(np.pi / 2.0) * (
        erf((delta_y[None, :] + half_height) / (root_two * sigma_y))
        - erf((delta_y[None, :] - half_height) / (root_two * sigma_y))
    )
    outer_x = np.exp(-0.5 * ((delta_x[None, :] + u) / sigma_x) ** 2)
    return np.sum(weights * outer_x * inner_y, axis=0)


def normalized_aperture_response(delta_x, delta_y, sigma_x, sigma_y):
    overlap = gaussian_disk_integral(delta_x, delta_y, sigma_x, sigma_y)
    peak = gaussian_disk_integral(
        np.array([0.0]), np.array([0.0]), sigma_x, sigma_y
    )[0]
    return overlap / max(float(peak), 1e-12)


def validate_quadrature():
    """Compare against the noncentral-chi-square solution for isotropic RFs."""
    rows = []
    for sigma in (2.0, 5.0, 10.0, 20.0, 40.0):
        for delta_x, delta_y in ((0, 0), (5, 0), (10, 0), (15, 0),
                                 (5, 7), (20, 20)):
            numeric = gaussian_disk_integral(
                np.array([delta_x]), np.array([delta_y]), sigma, sigma
            )[0]
            distance = np.hypot(delta_x, delta_y)
            exact = 2.0 * np.pi * sigma ** 2 * ncx2.cdf(
                (APERTURE_RADIUS_DEG / sigma) ** 2,
                df=2,
                nc=(distance / sigma) ** 2,
            )
            relative_error = abs(numeric - exact) / max(abs(exact), 1e-12)
            rows.append(
                {
                    "sigma_deg": sigma,
                    "delta_x_deg": delta_x,
                    "delta_y_deg": delta_y,
                    "numeric": numeric,
                    "exact": exact,
                    "relative_error": relative_error,
                }
            )
    result = pd.DataFrame(rows)
    if result["relative_error"].max() > 0.003:
        raise AssertionError("Aperture quadrature exceeded the 0.3% validation tolerance")
    return result


def point_spatial(x, y, center_x, center_y, sigma_x, sigma_y):
    return np.exp(
        -0.5 * (((x - center_x) / sigma_x) ** 2
                + ((y - center_y) / sigma_y) ** 2)
    )


def aperture_spatial(x, y, center_x, center_y, sigma_x, sigma_y):
    return normalized_aperture_response(
        x - center_x, y - center_y, sigma_x, sigma_y
    )


def model_prediction(parameters, x, y, orientation_index, spatial_model):
    baseline = parameters[0]
    amplitudes = parameters[1:4]
    center_x, center_y, sigma_x, sigma_y = parameters[4:8]
    if spatial_model == "point":
        spatial = point_spatial(x, y, center_x, center_y, sigma_x, sigma_y)
    elif spatial_model == "aperture":
        spatial = aperture_spatial(x, y, center_x, center_y, sigma_x, sigma_y)
    else:
        raise ValueError(f"Unknown spatial model: {spatial_model}")
    return baseline + amplitudes[orientation_index] * spatial


def initial_parameters(counts, x, y, orientation_index):
    baseline = max(float(np.quantile(counts, 0.20)), 0.0)
    frame = pd.DataFrame(
        {
            "x": np.round(x / 10.0) * 10.0,
            "y": np.round(y / 10.0) * 10.0,
            "count": counts,
        }
    )
    peak_x, peak_y = frame.groupby(["x", "y"], observed=True)["count"].mean().idxmax()
    amplitudes = []
    for orientation in range(3):
        local = counts[orientation_index == orientation]
        amplitudes.append(max(float(np.quantile(local, 0.95) - baseline), 0.1))
    return np.array(
        [baseline] + amplitudes + [float(peak_x), float(peak_y), 15.0, 15.0]
    )


def poisson_deviance(observed, predicted):
    predicted = np.maximum(predicted, 1e-8)
    positive = observed > 0
    terms = predicted - observed
    terms[positive] += observed[positive] * np.log(observed[positive] / predicted[positive])
    return float(2.0 * np.mean(terms))


def fit_unit(counts, x, y, orientation_index, train, test, group,
             spatial_model, start=None):
    sigma_upper = SIGMA_UPPER_DEG[group]
    lower = np.array([0.0, 0.0, 0.0, 0.0, -40.0 - CENTER_EXTENSION_DEG,
                      -40.0 - CENTER_EXTENSION_DEG, 2.0, 2.0])
    upper = np.array([np.inf, np.inf, np.inf, np.inf, 40.0 + CENTER_EXTENSION_DEG,
                      40.0 + CENTER_EXTENSION_DEG, sigma_upper, sigma_upper])
    if start is None:
        start = initial_parameters(
            counts[train], x[train], y[train], orientation_index[train]
        )
    start = np.clip(np.asarray(start, dtype=float), lower + 1e-7, upper - 1e-7)

    def residual(parameters):
        prediction = model_prediction(
            parameters, x[train], y[train], orientation_index[train], spatial_model
        )
        return 2.0 * (
            np.sqrt(np.maximum(prediction, 0) + 3.0 / 8.0)
            - np.sqrt(counts[train] + 3.0 / 8.0)
        )

    result = least_squares(
        residual, start, bounds=(lower, upper), method="trf", max_nfev=1200
    )
    parameters = result.x
    train_prediction = model_prediction(
        parameters, x[train], y[train], orientation_index[train], spatial_model
    )
    test_prediction = model_prediction(
        parameters, x[test], y[test], orientation_index[test], spatial_model
    )
    center_bound = bool(
        np.any(np.isclose(parameters[4:6], lower[4:6], atol=1e-4, rtol=0))
        or np.any(np.isclose(parameters[4:6], upper[4:6], atol=1e-4, rtol=0))
    )
    sigma_upper_bound = bool(
        np.any(np.isclose(parameters[6:8], upper[6:8], atol=1e-4, rtol=0))
    )
    sigma_lower_bound = bool(
        np.any(np.isclose(parameters[6:8], lower[6:8], atol=1e-4, rtol=0))
    )
    return parameters, {
        "success": bool(result.success and np.all(np.isfinite(parameters))),
        "optimizer_status": int(result.status),
        "optimizer_nfev": int(result.nfev),
        "center_bound": center_bound,
        "sigma_lower_bound": sigma_lower_bound,
        "sigma_upper_bound": sigma_upper_bound,
        "censored": bool(center_bound or sigma_lower_bound or sigma_upper_bound),
        "baseline_spikes": float(parameters[0]),
        "amplitude_0_spikes": float(parameters[1]),
        "amplitude_45_spikes": float(parameters[2]),
        "amplitude_90_spikes": float(parameters[3]),
        "mean_amplitude_spikes": float(np.mean(parameters[1:4])),
        "center_x_deg": float(parameters[4]),
        "center_y_deg": float(parameters[5]),
        "sigma_x_deg": float(parameters[6]),
        "sigma_y_deg": float(parameters[7]),
        "latent_halfmax_area_deg2": float(
            2.0 * np.pi * np.log(2.0) * parameters[6] * parameters[7]
        ),
        "train_poisson_deviance": poisson_deviance(counts[train], train_prediction),
        "test_poisson_deviance": poisson_deviance(counts[test], test_prediction),
        "train_anscombe_rmse": float(np.sqrt(np.mean(np.square(residual(parameters))))),
        "test_anscombe_rmse": float(np.sqrt(np.mean(np.square(
            2.0 * (np.sqrt(np.maximum(test_prediction, 0) + 3.0 / 8.0)
                   - np.sqrt(counts[test] + 3.0 / 8.0))
        )))),
    }


def balanced_subset(population, limit):
    eligible = population.loc[population["published_like_qc"]].copy()
    if limit is None or limit >= len(eligible):
        return eligible
    pieces = []
    pools = {}
    for group in ("V1", "HVA"):
        for split in ("calibration", "evaluation"):
            pools[(group, split)] = list(
                eligible.index[
                    eligible["group"].eq(group) & eligible["unit_split"].eq(split)
                ]
            )
    if PREDECLARED_CASE in eligible["ecephys_unit_id"].values:
        index = eligible.index[eligible["ecephys_unit_id"].eq(PREDECLARED_CASE)][0]
        pieces.append(index)
        for pool in pools.values():
            if index in pool:
                pool.remove(index)
    while len(pieces) < limit and any(pools.values()):
        for key in (("V1", "calibration"), ("V1", "evaluation"),
                    ("HVA", "calibration"), ("HVA", "evaluation")):
            if pools[key] and len(pieces) < limit:
                pieces.append(pools[key].pop(0))
    return eligible.loc[pieces].sort_index()


def add_paired_metrics(fits):
    key = ["ecephys_unit_id"]
    point_nominal = fits.loc[
        fits["spatial_model"].eq("point") & fits["gaze_condition"].eq("nominal"),
        key + ["test_poisson_deviance", "latent_halfmax_area_deg2",
               "center_x_deg", "center_y_deg", "sigma_x_deg", "sigma_y_deg"],
    ].rename(
        columns={
            "test_poisson_deviance": "point_nominal_test_deviance",
            "latent_halfmax_area_deg2": "point_nominal_area_deg2",
            "center_x_deg": "point_nominal_center_x_deg",
            "center_y_deg": "point_nominal_center_y_deg",
            "sigma_x_deg": "point_nominal_sigma_x_deg",
            "sigma_y_deg": "point_nominal_sigma_y_deg",
        }
    )
    same_model_nominal = fits.loc[
        fits["gaze_condition"].eq("nominal"),
        key + ["spatial_model", "test_poisson_deviance", "latent_halfmax_area_deg2"],
    ].rename(
        columns={
            "test_poisson_deviance": "same_model_nominal_test_deviance",
            "latent_halfmax_area_deg2": "same_model_nominal_area_deg2",
        }
    )
    result = fits.merge(point_nominal, on=key, how="left")
    result = result.merge(same_model_nominal, on=key + ["spatial_model"], how="left")
    result["deviance_improvement_vs_point_nominal"] = (
        result["point_nominal_test_deviance"] - result["test_poisson_deviance"]
    )
    result["gaze_deviance_improvement_within_model"] = (
        result["same_model_nominal_test_deviance"] - result["test_poisson_deviance"]
    )
    result["log2_latent_area_vs_point_nominal"] = np.log2(
        result["latent_halfmax_area_deg2"] / result["point_nominal_area_deg2"]
    )
    result["log2_sigma_x_vs_point_nominal"] = np.log2(
        result["sigma_x_deg"] / result["point_nominal_sigma_x_deg"]
    )
    result["log2_sigma_y_vs_point_nominal"] = np.log2(
        result["sigma_y_deg"] / result["point_nominal_sigma_y_deg"]
    )
    result["point_nominal_edge_distance_deg"] = np.minimum(
        40.0 - np.abs(result["point_nominal_center_x_deg"]),
        40.0 - np.abs(result["point_nominal_center_y_deg"]),
    )
    result["log2_gaze_area_ratio_within_model"] = np.log2(
        result["latent_halfmax_area_deg2"] / result["same_model_nominal_area_deg2"]
    )
    return result


def summarize(fits):
    rows = []
    augmented = pd.concat([fits, fits.assign(group="all")], ignore_index=True)
    for (model, gaze, split, group), local in augmented.groupby(
        ["spatial_model", "gaze_condition", "unit_split", "group"], observed=True
    ):
        rows.append(
            {
                "spatial_model": model,
                "gaze_condition": gaze,
                "unit_split": split,
                "group": group,
                "units": len(local),
                "median_test_deviance": local["test_poisson_deviance"].median(),
                "median_improvement_vs_point_nominal": local[
                    "deviance_improvement_vs_point_nominal"
                ].median(),
                "fraction_improved_vs_point_nominal": local[
                    "deviance_improvement_vs_point_nominal"
                ].gt(0).mean(),
                "median_gaze_improvement_within_model": local[
                    "gaze_deviance_improvement_within_model"
                ].median(),
                "fraction_gaze_improved_within_model": local[
                    "gaze_deviance_improvement_within_model"
                ].gt(0).mean(),
                "median_log2_latent_area_vs_point_nominal": local[
                    "log2_latent_area_vs_point_nominal"
                ].median(),
                "censored_fraction": local["censored"].mean(),
                "sigma_lower_bound_fraction": local["sigma_lower_bound"].mean(),
            }
        )
    return pd.DataFrame(rows)


def render_figure(fits, path):
    evaluation = fits.loc[fits["unit_split"].eq("evaluation")].copy()
    aperture_nominal = evaluation.loc[
        evaluation["spatial_model"].eq("aperture")
        & evaluation["gaze_condition"].eq("nominal")
    ]
    point_gaze = evaluation.loc[
        evaluation["spatial_model"].eq("point")
        & evaluation["gaze_condition"].eq("gaze_corrected")
    ].set_index("ecephys_unit_id")
    aperture_gaze = evaluation.loc[
        evaluation["spatial_model"].eq("aperture")
        & evaluation["gaze_condition"].eq("gaze_corrected")
    ].set_index("ecephys_unit_id")

    colors = {"V1": "#3366aa", "HVA": "#cc5533"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for group, local in aperture_nominal.groupby("group", observed=True):
        axes[0, 0].scatter(
            local["point_nominal_area_deg2"], local["latent_halfmax_area_deg2"],
            s=18, alpha=0.65, color=colors[group], label=group,
        )
    limits = [8, max(aperture_nominal["point_nominal_area_deg2"].max(),
                     aperture_nominal["latent_halfmax_area_deg2"].max()) * 1.1]
    axes[0, 0].plot(limits, limits, color="0.35", linewidth=1, linestyle="--")
    axes[0, 0].set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                   xlabel="Point-model half-max area (deg²)",
                   ylabel="Aperture-model latent area (deg²)")
    axes[0, 0].legend(frameon=False)

    bins = np.linspace(-3, 1, 33)
    for group, local in aperture_nominal.groupby("group", observed=True):
        axes[0, 1].hist(local["log2_latent_area_vs_point_nominal"], bins=bins,
                        histtype="step", linewidth=2, color=colors[group], label=group)
    axes[0, 1].axvline(0, color="0.35", linewidth=1, linestyle="--")
    axes[0, 1].set(xlabel="log2(aperture latent area / point area)", ylabel="Units")

    common = point_gaze.index.intersection(aperture_gaze.index)
    for group in ("V1", "HVA"):
        ids = [unit_id for unit_id in common if point_gaze.loc[unit_id, "group"] == group]
        axes[1, 0].scatter(
            point_gaze.loc[ids, "gaze_deviance_improvement_within_model"],
            aperture_gaze.loc[ids, "gaze_deviance_improvement_within_model"],
            s=18, alpha=0.65, color=colors[group], label=group,
        )
    gaze_limit = max(
        np.abs(point_gaze["gaze_deviance_improvement_within_model"]).max(),
        np.abs(aperture_gaze["gaze_deviance_improvement_within_model"]).max(),
    )
    axes[1, 0].plot([-gaze_limit, gaze_limit], [-gaze_limit, gaze_limit],
                    color="0.35", linewidth=1, linestyle="--")
    axes[1, 0].axhline(0, color="0.75", linewidth=0.8)
    axes[1, 0].axvline(0, color="0.75", linewidth=0.8)
    axes[1, 0].set(xlabel="Point-model gaze improvement",
                   ylabel="Aperture-model gaze improvement")

    for group, local in aperture_nominal.groupby("group", observed=True):
        axes[1, 1].scatter(
            local["point_nominal_edge_distance_deg"],
            local["log2_latent_area_vs_point_nominal"],
            s=18, alpha=0.65, color=colors[group], label=group,
        )
    axes[1, 1].axhline(0, color="0.35", linewidth=1, linestyle="--")
    axes[1, 1].axvline(10, color="0.55", linewidth=1, linestyle=":")
    axes[1, 1].set(xlabel="Point-fit center distance inside sampled grid (deg)",
                   ylabel="log2(aperture latent area / point area)")
    fig.suptitle(f"Session {SESSION_ID}: analytic aperture versus point RF fits")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    validation = validate_quadrature()
    if args.validate_only:
        print(validation[["relative_error"]].describe().to_string())
        return

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(input_dir / "visual_unit_population.csv", low_memory=False)
    trials = pd.read_csv(input_dir / "gabor_trial_gaze_table.csv", low_memory=False)
    spike_file = np.load(input_dir / "gabor_spike_counts.npz")
    counts = spike_file["counts"]
    if not np.array_equal(
        spike_file["unit_ids"].astype(int), population["ecephys_unit_id"].to_numpy(int)
    ):
        raise ValueError("Cached spike-count unit order does not match population table")

    source_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    chosen_gain_x = float(source_summary["chosen_gain_x"])
    chosen_gain_y = float(source_summary["chosen_gain_y"])
    valid = trials["valid_gaze"].to_numpy(bool)
    train = valid & trials["trial_split"].eq("train").to_numpy(bool)
    test = valid & trials["trial_split"].eq("test").to_numpy(bool)
    orientation = trials["orientation_index"].to_numpy(int)
    nominal_x = trials["x_position"].to_numpy(float)
    nominal_y = trials["y_position"].to_numpy(float)
    coordinates = {
        "nominal": (nominal_x, nominal_y),
        "gaze_corrected": (
            nominal_x - chosen_gain_x * trials["gaze_dx_deg"].to_numpy(float),
            nominal_y - chosen_gain_y * trials["gaze_dy_deg"].to_numpy(float),
        ),
    }

    selected = balanced_subset(population, args.limit_units)
    rows = []
    for progress, unit in enumerate(selected.itertuples(), start=1):
        unit_counts = counts[unit.Index].astype(float)
        for spatial_model in ("point", "aperture"):
            nominal_parameters = None
            for gaze_condition in ("nominal", "gaze_corrected"):
                x, y = coordinates[gaze_condition]
                parameters, metrics = fit_unit(
                    unit_counts, x, y, orientation, train, test, unit.group,
                    spatial_model=spatial_model, start=nominal_parameters,
                )
                if gaze_condition == "nominal":
                    nominal_parameters = parameters.copy()
                rows.append(
                    {
                        "ecephys_unit_id": int(unit.ecephys_unit_id),
                        "group": unit.group,
                        "ecephys_structure_acronym": unit.ecephys_structure_acronym,
                        "unit_split": unit.unit_split,
                        "spatial_model": spatial_model,
                        "gaze_condition": gaze_condition,
                        "gaze_gain_x": 0.0 if gaze_condition == "nominal" else chosen_gain_x,
                        "gaze_gain_y": 0.0 if gaze_condition == "nominal" else chosen_gain_y,
                        **metrics,
                    }
                )
        if progress % 10 == 0 or progress == len(selected):
            print(f"Model comparison: fitted {progress}/{len(selected)} units", flush=True)

    fits = add_paired_metrics(pd.DataFrame(rows))
    comparison_summary = summarize(fits)
    validation.to_csv(output_dir / "quadrature_validation.csv", index=False, float_format="%.9g")
    fits.to_csv(output_dir / "unit_model_comparison.csv", index=False, float_format="%.9g")
    comparison_summary.to_csv(
        output_dir / "model_comparison_summary.csv", index=False, float_format="%.9g"
    )
    render_figure(fits, output_dir / "population_model_comparison.png")
    run_summary = {
        "session_id": SESSION_ID,
        "units": int(len(selected)),
        "limited_checkpoint": args.limit_units is not None,
        "aperture_radius_deg": APERTURE_RADIUS_DEG,
        "quadrature_order": QUADRATURE_ORDER,
        "maximum_quadrature_relative_error": float(validation["relative_error"].max()),
        "gaze_gain_x": chosen_gain_x,
        "gaze_gain_y": chosen_gain_y,
        "carrier_rendered": False,
        "aperture_rasterized": False,
        "screen_clipping_modeled": False,
        "input_dir": str(input_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote point-versus-aperture comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
