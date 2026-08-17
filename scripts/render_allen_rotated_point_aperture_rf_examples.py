#!/usr/bin/env python3
"""Refit the preselected RF examples with freely rotated Gaussian ellipses."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from compare_allen_point_vs_aperture_rf import (
    SIGMA_UPPER_DEG,
    aperture_spatial,
    poisson_deviance,
    point_spatial,
)
from render_allen_point_aperture_rf_examples import parameter_vector


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
GAZE_DIR = ROOT / "artifacts" / "allen_population_gaze_rf" / f"session_{SESSION_ID}"
AXIS_DIR = ROOT / "artifacts" / "allen_aperture_rf_comparison" / f"session_{SESSION_ID}"
EXAMPLE_DIR = ROOT / "artifacts" / "allen_aperture_rf_examples" / f"session_{SESSION_ID}"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_rotated_rf_examples" / f"session_{SESSION_ID}"
ANGLE_STARTS_RAD = np.deg2rad([-89.0, -45.0, 0.0, 45.0, 89.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visualization-html", type=Path, default=None)
    return parser.parse_args()


def wrap_half_turn(theta):
    return (theta + np.pi / 2.0) % np.pi - np.pi / 2.0


def canonicalize(parameters):
    result = np.asarray(parameters, dtype=float).copy()
    if result[6] < result[7]:
        result[6], result[7] = result[7], result[6]
        result[8] += np.pi / 2.0
    result[8] = wrap_half_turn(result[8])
    return result


def rotated_coordinates(x, y, center_x, center_y, theta):
    dx = np.asarray(x, dtype=float) - center_x
    dy = np.asarray(y, dtype=float) - center_y
    cosine, sine = np.cos(theta), np.sin(theta)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def rotated_spatial(x, y, center_x, center_y, sigma_1, sigma_2, theta, model):
    principal_x, principal_y = rotated_coordinates(x, y, center_x, center_y, theta)
    if model == "point":
        return np.exp(-0.5 * ((principal_x / sigma_1) ** 2 + (principal_y / sigma_2) ** 2))
    if model == "aperture":
        return aperture_spatial(
            principal_x, principal_y, 0.0, 0.0, sigma_1, sigma_2
        )
    raise ValueError(model)


def rotated_prediction(parameters, x, y, orientation, model):
    spatial = rotated_spatial(x, y, *parameters[4:9], model=model)
    return parameters[0] + parameters[1:4][orientation] * spatial


def fit_rotated(counts, x, y, orientation, train, test, group, model, axis_start):
    sigma_upper = SIGMA_UPPER_DEG[group]
    lower = np.array([0.0, 0.0, 0.0, 0.0, -60.0, -60.0, 2.0, 2.0, -np.pi])
    upper = np.array([
        np.inf, np.inf, np.inf, np.inf, 60.0, 60.0,
        sigma_upper, sigma_upper, np.pi,
    ])

    def residual(parameters):
        prediction = rotated_prediction(
            parameters, x[train], y[train], orientation[train], model
        )
        return 2.0 * (
            np.sqrt(np.maximum(prediction, 0.0) + 3.0 / 8.0)
            - np.sqrt(counts[train] + 3.0 / 8.0)
        )

    candidates = []
    for angle in ANGLE_STARTS_RAD:
        start = np.r_[axis_start, angle]
        start = np.clip(start, lower + 1e-7, upper - 1e-7)
        result = least_squares(
            residual, start, bounds=(lower, upper), method="trf", max_nfev=1800
        )
        candidates.append((float(np.mean(np.square(result.fun))), result))
    candidates.sort(key=lambda item: item[0])
    best_score, best = candidates[0]
    parameters = canonicalize(best.x)
    train_prediction = rotated_prediction(
        parameters, x[train], y[train], orientation[train], model
    )
    test_prediction = rotated_prediction(
        parameters, x[test], y[test], orientation[test], model
    )
    return parameters, {
        "success": bool(best.success and np.all(np.isfinite(parameters))),
        "optimizer_nfev": int(best.nfev),
        "best_train_anscombe_mse": best_score,
        "second_best_train_anscombe_mse": candidates[1][0],
        "multistart_mse_gap": candidates[1][0] - best_score,
        "train_poisson_deviance": poisson_deviance(counts[train], train_prediction),
        "test_poisson_deviance": poisson_deviance(counts[test], test_prediction),
        "baseline_spikes": parameters[0],
        "mean_amplitude_spikes": parameters[1:4].mean(),
        "center_x_deg": parameters[4],
        "center_y_deg": parameters[5],
        "sigma_major_deg": parameters[6],
        "sigma_minor_deg": parameters[7],
        "major_axis_angle_deg": np.rad2deg(parameters[8]),
        "axis_ratio": parameters[6] / parameters[7],
        "latent_halfmax_area_deg2": 2.0 * np.pi * np.log(2.0) * parameters[6] * parameters[7],
        "sigma_lower_bound": bool(np.isclose(parameters[7], 2.0, atol=1e-4, rtol=0)),
        "sigma_upper_bound": bool(np.isclose(parameters[6], sigma_upper, atol=1e-4, rtol=0)),
    }


def mean_map(parameters, xx, yy, model):
    spatial = rotated_spatial(xx.ravel(), yy.ravel(), *parameters[4:9], model=model)
    spatial = spatial.reshape(xx.shape)
    return parameters[0] + parameters[1:4].mean() * spatial, spatial


def axis_spatial_map(row, xx, yy, model):
    parameters = parameter_vector(row)
    if model == "point" or model == "latent":
        return point_spatial(xx, yy, *parameters[4:8])
    return aperture_spatial(
        xx.ravel(), yy.ravel(), *parameters[4:8]
    ).reshape(xx.shape)


def render(selection, fit_rows, fits, population, counts, trials, figure_path, dark=False):
    if dark:
        plt.style.use("dark_background")
        old_color, new_color = "#f1f5f9", "#22d3ee"
    else:
        plt.style.use("default")
        old_color, new_color = "#334155", "#00bcd4"
    valid_test = trials["valid_gaze"].to_numpy(bool) & trials["trial_split"].eq("test").to_numpy(bool)
    x_trial = trials["x_position"].to_numpy(float)
    y_trial = trials["y_position"].to_numpy(float)
    grid_axis = np.linspace(-40, 40, 81)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    rotated = pd.DataFrame(fit_rows).set_index(["ecephys_unit_id", "spatial_model"])
    fig, axes = plt.subplots(len(selection), 4, figsize=(14, 13), constrained_layout=True)
    titles = [
        "Held-out responses", "Rotated point Gaussian",
        "Rotated aperture response", "Rotated aperture latent Gaussian",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12)

    for row_number, selected in selection.iterrows():
        unit_id = int(selected["ecephys_unit_id"])
        population_index = population.index[population["ecephys_unit_id"].eq(unit_id)][0]
        unit_counts = counts[population_index].astype(float)
        observed = pd.DataFrame({
            "x": x_trial[valid_test], "y": y_trial[valid_test],
            "count": unit_counts[valid_test],
        }).groupby(["y", "x"], observed=True)["count"].mean().unstack("x")
        observed = observed.reindex(index=np.arange(-40, 41, 10), columns=np.arange(-40, 41, 10))

        point_parameters = np.array(rotated.loc[(unit_id, "point"), "parameters"], dtype=float)
        aperture_parameters = np.array(rotated.loc[(unit_id, "aperture"), "parameters"], dtype=float)
        point_map, point_shape = mean_map(point_parameters, xx, yy, "point")
        aperture_map, aperture_shape = mean_map(aperture_parameters, xx, yy, "aperture")
        latent_map, latent_shape = mean_map(aperture_parameters, xx, yy, "point")
        maps = [observed.to_numpy(float), point_map, aperture_map, latent_map]
        vmin = min(float(np.nanmin(array)) for array in maps)
        vmax = max(float(np.nanmax(array)) for array in maps)

        point_axis = fits.loc[
            fits["ecephys_unit_id"].eq(unit_id) & fits["spatial_model"].eq("point")
            & fits["gaze_condition"].eq("nominal")
        ].iloc[0]
        aperture_axis = fits.loc[
            fits["ecephys_unit_id"].eq(unit_id) & fits["spatial_model"].eq("aperture")
            & fits["gaze_condition"].eq("nominal")
        ].iloc[0]
        old_shapes = [None, axis_spatial_map(point_axis, xx, yy, "point"),
                      axis_spatial_map(aperture_axis, xx, yy, "aperture"),
                      axis_spatial_map(aperture_axis, xx, yy, "latent")]
        new_shapes = [None, point_shape, aperture_shape, latent_shape]
        for column, axis_object in enumerate(axes[row_number]):
            image = axis_object.imshow(
                maps[column], origin="lower", extent=(-45, 45, -45, 45),
                interpolation="nearest" if column == 0 else "bilinear",
                cmap="magma", vmin=vmin, vmax=vmax, aspect="equal",
            )
            if column:
                axis_object.contour(
                    xx, yy, old_shapes[column], levels=[0.5], colors=old_color,
                    linewidths=1.2, linestyles="--",
                )
                axis_object.contour(
                    xx, yy, new_shapes[column], levels=[0.5], colors=new_color,
                    linewidths=1.5,
                )
            axis_object.set_xticks([-40, 0, 40])
            axis_object.set_yticks([-40, 0, 40])
            axis_object.set_xlabel("Azimuth (deg)")
            if column == 0:
                axis_object.set_ylabel("Elevation (deg)")
            else:
                axis_object.set_yticklabels([])
            fig.colorbar(image, ax=axis_object, fraction=0.046, pad=0.02, label="spikes / 249 ms")

        point_result = rotated.loc[(unit_id, "point")]
        aperture_result = rotated.loc[(unit_id, "aperture")]
        floor = " · width floor" if aperture_result["sigma_lower_bound"] else ""
        axes[row_number, 0].text(
            -0.34, 0.5,
            f"{selected['selection_role']}\n{unit_id} · {selected['ecephys_structure_acronym']}\n"
            f"point θ {point_result['major_axis_angle_deg']:+.0f}° · Δdev {point_result['rotation_test_gain']:+.4f}\n"
            f"aperture θ {aperture_result['major_axis_angle_deg']:+.0f}° · Δdev {aperture_result['rotation_test_gain']:+.4f}\n"
            f"aperture area {aperture_result['axis_area_deg2']:.0f} → "
            f"{aperture_result['latent_halfmax_area_deg2']:.0f} deg²{floor}",
            transform=axes[row_number, 0].transAxes, ha="right", va="center", fontsize=9,
        )

    fig.suptitle(
        f"Session {SESSION_ID}: RF examples refit with freely rotated ellipses\n"
        f"solid cyan = rotated half-maximum · dashed = prior axis-aligned fit",
        fontsize=14,
    )
    fig.savefig(figure_path, dpi=115, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_visualization(light_path, dark_path, html_path):
    light = base64.b64encode(light_path.read_bytes()).decode("ascii")
    dark = base64.b64encode(dark_path.read_bytes()).decode("ascii")
    fragment = f'''<div id="allen-rotated-rf-examples">
  <h2>Point-center and aperture RF examples with rotation</h2>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{dark}" />
    <img src="data:image/png;base64,{light}" alt="Four preselected Allen RF examples refit with freely rotated point and aperture Gaussian models. Solid cyan contours show rotated fits and dashed contours show the prior axis-aligned fits." />
  </picture>
</div>
<style>
#allen-rotated-rf-examples {{ width: 100%; background: transparent; color: var(--foreground); }}
#allen-rotated-rf-examples img {{ display: block; width: 100%; height: auto; }}
</style>
'''
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(fragment, encoding="utf-8")


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(EXAMPLE_DIR / "example_selection.csv")
    fits = pd.read_csv(AXIS_DIR / "unit_model_comparison.csv")
    trials = pd.read_csv(GAZE_DIR / "gabor_trial_gaze_table.csv")
    population = pd.read_csv(GAZE_DIR / "visual_unit_population.csv", low_memory=False)
    spike_file = np.load(GAZE_DIR / "gabor_spike_counts.npz")
    counts = spike_file["counts"]
    valid = trials["valid_gaze"].to_numpy(bool)
    train = valid & trials["trial_split"].eq("train").to_numpy(bool)
    test = valid & trials["trial_split"].eq("test").to_numpy(bool)
    x = trials["x_position"].to_numpy(float)
    y = trials["y_position"].to_numpy(float)
    orientation = trials["orientation_index"].to_numpy(int)

    fit_rows = []
    for selected in selection.itertuples():
        unit_id = int(selected.ecephys_unit_id)
        population_index = population.index[population["ecephys_unit_id"].eq(unit_id)][0]
        unit_counts = counts[population_index].astype(float)
        for model in ("point", "aperture"):
            axis_row = fits.loc[
                fits["ecephys_unit_id"].eq(unit_id)
                & fits["spatial_model"].eq(model)
                & fits["gaze_condition"].eq("nominal")
            ].iloc[0]
            parameters, metrics = fit_rotated(
                unit_counts, x, y, orientation, train, test, selected.group,
                model, parameter_vector(axis_row),
            )
            fit_rows.append({
                "ecephys_unit_id": unit_id,
                "selection_role": selected.selection_role,
                "group": selected.group,
                "ecephys_structure_acronym": selected.ecephys_structure_acronym,
                "spatial_model": model,
                "parameters": parameters.tolist(),
                "axis_test_poisson_deviance": axis_row["test_poisson_deviance"],
                "rotation_test_gain": axis_row["test_poisson_deviance"] - metrics["test_poisson_deviance"],
                "axis_area_deg2": axis_row["latent_halfmax_area_deg2"],
                "rotation_log2_area_ratio": np.log2(
                    metrics["latent_halfmax_area_deg2"] / axis_row["latent_halfmax_area_deg2"]
                ),
                **metrics,
            })
    table = pd.DataFrame(fit_rows)
    csv_table = table.drop(columns="parameters")
    csv_table.to_csv(output / "rotated_example_fits.csv", index=False, float_format="%.9g")

    light_path = output / "rf_method_examples_rotated.png"
    dark_path = output / "rf_method_examples_rotated_dark.png"
    render(selection, fit_rows, fits, population, counts, trials, light_path, dark=False)
    render(selection, fit_rows, fits, population, counts, trials, dark_path, dark=True)
    if args.visualization_html is not None:
        write_visualization(light_path, dark_path, args.visualization_html.resolve())
    print(csv_table[[
        "ecephys_unit_id", "selection_role", "spatial_model",
        "major_axis_angle_deg", "axis_ratio", "rotation_test_gain",
        "axis_area_deg2", "latent_halfmax_area_deg2", "sigma_lower_bound",
    ]].to_string(index=False))
    print(f"Wrote {light_path}")


if __name__ == "__main__":
    main()
