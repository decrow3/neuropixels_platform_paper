#!/usr/bin/env python3
"""Render auditable examples of point-center and aperture-overlap RF fits."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_allen_point_vs_aperture_rf import (
    PREDECLARED_CASE,
    aperture_spatial,
    point_spatial,
)


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
DEFAULT_GAZE = ROOT / "artifacts" / "allen_population_gaze_rf" / f"session_{SESSION_ID}"
DEFAULT_FITS = (
    ROOT / "artifacts" / "allen_aperture_rf_comparison" / f"session_{SESSION_ID}"
    / "unit_model_comparison.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_aperture_rf_examples" / f"session_{SESSION_ID}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze-dir", type=Path, default=DEFAULT_GAZE)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visualization-html", type=Path, default=None)
    return parser.parse_args()


def select_examples(aperture):
    eligible = aperture.loc[
        aperture["unit_split"].eq("evaluation")
        & aperture["gaze_condition"].eq("nominal")
    ].copy()
    rows = []
    for group in ("V1", "HVA"):
        local = eligible.loc[
            eligible["group"].eq(group)
            & ~eligible["censored"]
            & eligible["point_nominal_edge_distance_deg"].gt(10)
        ].copy()
        median = local["log2_latent_area_vs_point_nominal"].median()
        chosen = local.loc[(local["log2_latent_area_vs_point_nominal"] - median).abs().idxmin()]
        rows.append((
            f"Typical interior {group}", chosen,
            f"closest uncensored interior fit to {group} median area ratio",
        ))

    excluded = {int(row[1]["ecephys_unit_id"]) for row in rows}
    resolved = eligible.loc[
        ~eligible["censored"]
        & eligible["deviance_improvement_vs_point_nominal"].ge(0)
        & ~eligible["ecephys_unit_id"].isin(excluded)
    ]
    chosen = resolved.loc[resolved["log2_latent_area_vs_point_nominal"].idxmin()]
    rows.append((
        "Strongest resolved shrinkage",
        chosen,
        "smallest area ratio among uncensored fits with non-worse held-out deviance",
    ))

    floor = eligible.loc[eligible["ecephys_unit_id"].eq(PREDECLARED_CASE)]
    if floor.empty or not bool(floor.iloc[0]["sigma_lower_bound"]):
        floor = eligible.loc[eligible["sigma_lower_bound"]].sort_values("test_poisson_deviance").head(1)
        criterion = "best held-out deviance among fits reaching the sigma floor"
    else:
        criterion = "predeclared diagnostic case; aperture fit reaches the sigma floor"
    rows.append(("Lower-bound diagnostic", floor.iloc[0], criterion))

    output = []
    for order, (role, row, criterion) in enumerate(rows, start=1):
        output.append({
            "display_order": order,
            "selection_role": role,
            "selection_criterion": criterion,
            "ecephys_unit_id": int(row["ecephys_unit_id"]),
            "group": row["group"],
            "ecephys_structure_acronym": row["ecephys_structure_acronym"],
            "log2_aperture_area_vs_point": row["log2_latent_area_vs_point_nominal"],
            "aperture_to_point_area_ratio": 2 ** row["log2_latent_area_vs_point_nominal"],
            "heldout_deviance_improvement": row["deviance_improvement_vs_point_nominal"],
            "point_center_distance_inside_grid_deg": row["point_nominal_edge_distance_deg"],
            "aperture_sigma_lower_bound": bool(row["sigma_lower_bound"]),
        })
    return pd.DataFrame(output)


def parameter_vector(row):
    return np.array([
        row["baseline_spikes"], row["amplitude_0_spikes"],
        row["amplitude_45_spikes"], row["amplitude_90_spikes"],
        row["center_x_deg"], row["center_y_deg"],
        row["sigma_x_deg"], row["sigma_y_deg"],
    ], dtype=float)


def mean_model_map(parameters, xx, yy, model):
    baseline = parameters[0]
    amplitude = parameters[1:4].mean()
    if model == "point":
        spatial = point_spatial(xx, yy, *parameters[4:8])
    elif model == "aperture":
        spatial = aperture_spatial(xx.ravel(), yy.ravel(), *parameters[4:8]).reshape(xx.shape)
    elif model == "latent":
        spatial = point_spatial(xx, yy, *parameters[4:8])
    else:
        raise ValueError(model)
    return baseline + amplitude * spatial, spatial


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fits = pd.read_csv(args.fits.resolve())
    aperture = fits.loc[fits["spatial_model"].eq("aperture")]
    selection = select_examples(aperture)
    selection.to_csv(output / "example_selection.csv", index=False, float_format="%.9g")

    trials = pd.read_csv(args.gaze_dir.resolve() / "gabor_trial_gaze_table.csv")
    population = pd.read_csv(args.gaze_dir.resolve() / "visual_unit_population.csv", low_memory=False)
    spike_file = np.load(args.gaze_dir.resolve() / "gabor_spike_counts.npz")
    counts = spike_file["counts"]
    valid_test = trials["valid_gaze"].to_numpy(bool) & trials["trial_split"].eq("test").to_numpy(bool)
    x_trial = trials["x_position"].to_numpy(float)
    y_trial = trials["y_position"].to_numpy(float)

    axis = np.linspace(-40, 40, 81)
    xx, yy = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(len(selection), 4, figsize=(14, 13), constrained_layout=True)
    column_titles = [
        "Held-out responses", "Point-center Gaussian", "Aperture-overlap response",
        "Aperture latent Gaussian",
    ]
    for column, title in enumerate(column_titles):
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

        point_row = fits.loc[
            fits["ecephys_unit_id"].eq(unit_id)
            & fits["spatial_model"].eq("point")
            & fits["gaze_condition"].eq("nominal")
        ].iloc[0]
        aperture_row = fits.loc[
            fits["ecephys_unit_id"].eq(unit_id)
            & fits["spatial_model"].eq("aperture")
            & fits["gaze_condition"].eq("nominal")
        ].iloc[0]
        point_parameters = parameter_vector(point_row)
        aperture_parameters = parameter_vector(aperture_row)
        point_map, point_spatial_map = mean_model_map(point_parameters, xx, yy, "point")
        aperture_map, aperture_spatial_map = mean_model_map(aperture_parameters, xx, yy, "aperture")
        latent_map, latent_spatial_map = mean_model_map(aperture_parameters, xx, yy, "latent")
        maps = [observed.to_numpy(float), point_map, aperture_map, latent_map]
        vmin = min(float(np.nanmin(array)) for array in maps)
        vmax = max(float(np.nanmax(array)) for array in maps)

        for column, (axis_object, array) in enumerate(zip(axes[row_number], maps)):
            interpolation = "nearest" if column == 0 else "bilinear"
            image = axis_object.imshow(
                array, origin="lower", extent=(-45, 45, -45, 45),
                interpolation=interpolation, cmap="magma", vmin=vmin, vmax=vmax,
                aspect="equal",
            )
            if column == 1:
                axis_object.contour(xx, yy, point_spatial_map, levels=[0.5], colors="cyan", linewidths=1)
                axis_object.plot(point_parameters[4], point_parameters[5], "+", color="cyan", ms=7)
            elif column == 2:
                axis_object.contour(xx, yy, aperture_spatial_map, levels=[0.5], colors="cyan", linewidths=1)
                axis_object.plot(aperture_parameters[4], aperture_parameters[5], "+", color="cyan", ms=7)
            elif column == 3:
                axis_object.contour(xx, yy, latent_spatial_map, levels=[0.5], colors="cyan", linewidths=1)
                axis_object.plot(aperture_parameters[4], aperture_parameters[5], "+", color="cyan", ms=7)
            axis_object.set_xticks([-40, 0, 40])
            axis_object.set_yticks([-40, 0, 40])
            axis_object.set_xlabel("Azimuth (deg)")
            if column == 0:
                axis_object.set_ylabel("Elevation (deg)")
            else:
                axis_object.set_yticklabels([])
            fig.colorbar(image, ax=axis_object, fraction=0.046, pad=0.02, label="spikes / 249 ms")

        point_area = float(point_row["latent_halfmax_area_deg2"])
        aperture_area = float(aperture_row["latent_halfmax_area_deg2"])
        floor_label = " · width floor" if bool(aperture_row["sigma_lower_bound"]) else ""
        axes[row_number, 0].text(
            -0.34, 0.5,
            f"{selected['selection_role']}\n{unit_id} · {selected['ecephys_structure_acronym']}\n"
            f"area {point_area:.0f} → {aperture_area:.0f} deg²\n"
            f"ratio {aperture_area / point_area:.2f}{floor_label}",
            transform=axes[row_number, 0].transAxes, ha="right", va="center", fontsize=10,
        )

    fig.suptitle(
        f"Session {SESSION_ID}: point-center and analytic aperture RF examples\n"
        "cyan contour = fitted half-maximum; predictions use training-fit parameters",
        fontsize=14,
    )
    figure_path = output / "rf_method_examples.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    if args.visualization_html is not None:
        html_path = args.visualization_html.resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        fragment = f'''<div id="allen-rf-method-examples">
  <h2>Point-center and aperture RF examples</h2>
  <img src="data:image/png;base64,{encoded}" alt="Four selected units from Allen session 746083955. Each row compares held-out response counts, the point-center Gaussian prediction, the analytic aperture-overlap prediction, and the aperture model latent Gaussian. Cyan contours mark fitted half-maximum boundaries." />
</div>
<style>
#allen-rf-method-examples {{ width: 100%; background: transparent; color: var(--foreground); }}
#allen-rf-method-examples img {{ display: block; width: 100%; height: auto; }}
</style>
'''
        html_path.write_text(fragment, encoding="utf-8")
        print(f"Wrote {html_path}")
    print(f"Wrote {figure_path}")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
