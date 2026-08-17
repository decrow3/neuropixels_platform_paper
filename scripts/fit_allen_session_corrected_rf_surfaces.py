#!/usr/bin/env python3
"""Fit one Allen session's visual RF population and compare RF-size surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import (
    ReceptiveFieldMapping,
    fit_2d_gaussian,
    threshold_rf,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NWB = Path(
    "/media/huklaban5/Data/MouseV2/allen_v1_bridge/000021/"
    "sub-718643564/sub-718643564_ses-737581020.nwb"
)
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_session_corrected_rf_surfaces" / "session_737581020"
SESSION_ID = 737581020
AREA_MAP = {
    "VISp": "V1",
    "VISl": "HVA",
    "VISrl": "HVA",
    "VISal": "HVA",
    "VISpm": "HVA",
    "VISam": "HVA",
}
SIGMA_UPPER_PX = {"V1": 4.0, "HVA": 5.0}
CENTER_EXTENSION_PX = 2.0
GRID_AZIMUTH = np.arange(10.0, 91.0, 10.0)
GRID_ELEVATION = np.arange(-30.0, 51.0, 10.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--session-id", type=int, default=SESSION_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--minimum-effective-units", type=float, default=3.0)
    parser.add_argument(
        "--reuse-fits", action="store_true",
        help="Reuse session_rf_fit_population.csv and regenerate summaries/figures.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_no_baseline(shape, parameters):
    amplitude, center_y, center_x, sigma_y, sigma_x = parameters
    y, x = np.indices(shape)
    return amplitude * np.exp(
        -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2)
    )


def gaussian_with_baseline(shape, parameters):
    baseline, amplitude, center_y, center_x, sigma_y, sigma_x = parameters
    return baseline + gaussian_no_baseline(
        shape, np.array([amplitude, center_y, center_x, sigma_y, sigma_x])
    )


def fit_corrected(matrix, group):
    rows, columns = matrix.shape
    baseline = max(float(np.quantile(matrix, 0.20)), 0.0)
    peak_y, peak_x = np.unravel_index(np.argmax(matrix), matrix.shape)
    start = np.array(
        [baseline, max(float(matrix.max() - baseline), 1e-3), peak_y, peak_x, 1.5, 1.5]
    )
    lower = np.array(
        [0.0, 0.0, -CENTER_EXTENSION_PX, -CENTER_EXTENSION_PX, 0.20, 0.20]
    )
    upper = np.array(
        [np.inf, np.inf, rows - 1 + CENTER_EXTENSION_PX,
         columns - 1 + CENTER_EXTENSION_PX, SIGMA_UPPER_PX[group], SIGMA_UPPER_PX[group]]
    )
    result = least_squares(
        lambda p: (gaussian_with_baseline(matrix.shape, p) - matrix).ravel(),
        np.clip(start, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        method="trf",
        max_nfev=20000,
    )
    prediction = gaussian_with_baseline(matrix.shape, result.x)
    at_bound = bool(
        np.any(np.isclose(result.x, lower, atol=1e-5, rtol=0))
        or np.any(np.isclose(result.x, upper, atol=1e-5, rtol=0))
    )
    sigma_upper_bound = bool(
        np.any(np.isclose(result.x[4:6], upper[4:6], atol=1e-5, rtol=0))
    )
    center_bound = bool(
        np.any(np.isclose(result.x[2:4], lower[2:4], atol=1e-5, rtol=0))
        or np.any(np.isclose(result.x[2:4], upper[2:4], atol=1e-5, rtol=0))
    )
    return (
        result.x,
        bool(result.success and np.all(np.isfinite(result.x))),
        at_bound,
        sigma_upper_bound,
        center_bound,
        float(np.sqrt(np.square(prediction - matrix).mean())),
    )


def fit_population(nwb_path, population, session_id=SESSION_ID):
    session = EcephysSession.from_nwb_path(
        nwb_path,
        api_kwargs={
            "amplitude_cutoff_maximum": np.inf,
            "presence_ratio_minimum": -np.inf,
            "isi_violations_maximum": np.inf,
            "filter_by_validity": False,
        },
    )
    unit_ids = population["ecephys_unit_id"].astype(int).tolist()
    analysis = ReceptiveFieldMapping(session, filter=unit_ids, mask_threshold=1.0)
    rows = []
    for index, unit in enumerate(population.itertuples(index=False), start=1):
        unit_id = int(unit.ecephys_unit_id)
        record = {
            "ecephys_unit_id": unit_id,
            "ecephys_session_id": session_id,
            "ecephys_structure_acronym": unit.ecephys_structure_acronym,
            "group": unit.group,
            "p_value_rf": unit.p_value_rf,
            "snr": unit.snr,
            "firing_rate_dg": unit.firing_rate_dg,
            "released_area_rf_deg2": unit.area_rf,
            "released_azimuth_rf_deg": unit.azimuth_rf,
            "released_elevation_rf_deg": unit.elevation_rf,
            "released_width_rf_deg": unit.width_rf,
            "released_height_rf_deg": unit.height_rf,
        }
        try:
            matrix = analysis.get_receptive_field(unit_id).astype(float)
            mask, center_x, center_y, area_pixels = threshold_rf(matrix, 1.0)
            allen_parameters, allen_success = fit_2d_gaussian(matrix)
            allen_parameters = np.asarray(allen_parameters, dtype=float)
            allen_prediction = gaussian_no_baseline(matrix.shape, allen_parameters)
            (
                corrected,
                corrected_success,
                corrected_at_bound,
                corrected_sigma_upper_bound,
                corrected_center_bound,
                corrected_rmse,
            ) = fit_corrected(matrix, unit.group)
            record.update(
                {
                    "map_mean": float(matrix.mean()),
                    "map_max": float(matrix.max()),
                    "threshold_azimuth_deg": float(GRID_AZIMUTH[0] + 10.0 * center_x),
                    "threshold_elevation_deg": float(50.0 - 10.0 * center_y),
                    "threshold_area_deg2": float(area_pixels * 100.0),
                    "threshold_component_touches_edge": bool(
                        mask[0, :].any() or mask[-1, :].any()
                        or mask[:, 0].any() or mask[:, -1].any()
                    ),
                    "allen_success": bool(allen_success and np.all(np.isfinite(allen_parameters))),
                    "allen_center_azimuth_deg": float(GRID_AZIMUTH[0] + 10.0 * allen_parameters[2]),
                    "allen_center_elevation_deg": float(50.0 - 10.0 * allen_parameters[1]),
                    "allen_sigma_x_deg": float(abs(allen_parameters[4]) * 10.0),
                    "allen_sigma_y_deg": float(abs(allen_parameters[3]) * 10.0),
                    "allen_rmse": float(np.sqrt(np.square(allen_prediction - matrix).mean())),
                    "corrected_success": corrected_success,
                    "corrected_at_bound": corrected_at_bound,
                    "corrected_sigma_upper_bound": corrected_sigma_upper_bound,
                    "corrected_center_bound": corrected_center_bound,
                    "corrected_baseline": float(corrected[0]),
                    "corrected_center_azimuth_deg": float(GRID_AZIMUTH[0] + 10.0 * corrected[3]),
                    "corrected_center_elevation_deg": float(50.0 - 10.0 * corrected[2]),
                    "corrected_sigma_x_deg": float(corrected[5] * 10.0),
                    "corrected_sigma_y_deg": float(corrected[4] * 10.0),
                    "corrected_rmse": corrected_rmse,
                    "fit_exception": "",
                }
            )
        except Exception as error:
            record.update(
                {
                    "allen_success": False,
                    "corrected_success": False,
                    "corrected_at_bound": False,
                    "corrected_sigma_upper_bound": False,
                    "corrected_center_bound": False,
                    "fit_exception": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(record)
        if index % 50 == 0 or index == len(population):
            print(f"Fitted {index}/{len(population)} visual units", flush=True)
    result = pd.DataFrame(rows)
    for prefix in ("allen", "corrected"):
        result[f"{prefix}_major_sigma_deg"] = result[
            [f"{prefix}_sigma_x_deg", f"{prefix}_sigma_y_deg"]
        ].max(axis=1)
        result[f"{prefix}_halfmax_area_deg2"] = (
            2.0 * np.pi * np.log(2.0)
            * result[f"{prefix}_sigma_x_deg"] * result[f"{prefix}_sigma_y_deg"]
        )
    result["published_like_qc"] = (
        result["p_value_rf"].lt(0.01)
        & result["released_area_rf_deg2"].lt(2500)
        & result["snr"].gt(1)
        & result["firing_rate_dg"].gt(0.1)
    )
    result["allen_finite"] = result[
        ["allen_sigma_x_deg", "allen_sigma_y_deg",
         "allen_center_azimuth_deg", "allen_center_elevation_deg"]
    ].notna().all(axis=1)
    result["corrected_censored"] = (
        result["corrected_sigma_upper_bound"].astype(bool)
        | result["corrected_center_bound"].astype(bool)
    )
    return result


def kernel_surface(table, value_column, az_grid, el_grid, bandwidth, minimum_effective):
    points = table[["threshold_azimuth_deg", "threshold_elevation_deg"]].to_numpy(float)
    values = np.log2(table[value_column].to_numpy(float))
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    estimates = np.full(az_mesh.size, np.nan)
    effective = np.zeros(az_mesh.size)
    for index, target in enumerate(np.column_stack([az_mesh.ravel(), el_mesh.ravel()])):
        weights = np.exp(-0.5 * np.sum(np.square((points - target) / bandwidth), axis=1))
        if weights.sum() > 0:
            effective[index] = weights.sum() ** 2 / np.square(weights).sum()
        if effective[index] >= minimum_effective:
            estimates[index] = np.average(values, weights=weights)
    shape = az_mesh.shape
    return np.exp2(estimates.reshape(shape)), effective.reshape(shape)


def make_surfaces(fits, bandwidth, minimum_effective):
    primary = fits.loc[
        fits["published_like_qc"]
        & fits["allen_finite"]
        & fits["corrected_success"]
        & ~fits["corrected_censored"]
        & fits[["threshold_azimuth_deg", "threshold_elevation_deg",
               "allen_halfmax_area_deg2", "corrected_halfmax_area_deg2"]].notna().all(axis=1)
        & fits["allen_halfmax_area_deg2"].gt(0)
        & fits["corrected_halfmax_area_deg2"].gt(0)
    ].copy()
    az_grid = np.linspace(10, 90, 65)
    el_grid = np.linspace(-30, 50, 65)
    surfaces = {}
    for group in ("V1", "HVA"):
        local = primary.loc[primary["group"].eq(group)]
        for model in ("allen", "corrected"):
            surfaces[(group, model)] = kernel_surface(
                local, f"{model}_halfmax_area_deg2", az_grid, el_grid,
                bandwidth, minimum_effective,
            )
    return primary, surfaces, az_grid, el_grid


def render_surface_figure(primary, surfaces, az_grid, el_grid, path, bandwidth,
                          minimum_effective, session_id=SESSION_ID):
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 11.2), sharex=True, sharey=True)
    for row, group in enumerate(("V1", "HVA")):
        joint = np.concatenate(
            [surfaces[(group, model)][0][np.isfinite(surfaces[(group, model)][0])]
             for model in ("allen", "corrected")]
        )
        limits = np.quantile(joint, [0.05, 0.95])
        if limits[1] <= limits[0]:
            limits = [joint.min(), joint.max()]
        for column, model in enumerate(("allen", "corrected")):
            surface, effective = surfaces[(group, model)]
            axis = axes[row, column]
            image = axis.pcolormesh(
                az_grid, el_grid, surface, shading="gouraud", cmap="YlGnBu",
                norm=LogNorm(vmin=max(limits[0], 1e-6), vmax=limits[1]),
            )
            local = primary.loc[primary["group"].eq(group)]
            axis.scatter(
                local["threshold_azimuth_deg"], local["threshold_elevation_deg"],
                s=8, facecolors="none", edgecolors="#263238", linewidths=0.45, alpha=0.42,
            )
            if np.nanmax(effective) >= 6:
                levels = [level for level in (3, 6, 12) if level <= np.nanmax(effective)]
                axis.contour(az_grid, el_grid, effective, levels=levels,
                             colors="#444444", linewidths=0.55, alpha=0.45)
            title = "Allen no baseline" if model == "allen" else "Corrected: baseline + bounded Gaussian"
            axis.set_title(f"{group} · {title}\n{len(local)} matched QC units; 5–95% scale {limits[0]:.0f}–{limits[1]:.0f} deg²")
            axis.set(xlabel="Threshold RF azimuth (deg)", ylabel="Threshold RF elevation (deg)",
                     xlim=(10, 90), ylim=(-30, 50), aspect="equal")
            axis.grid(alpha=0.12)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03, extend="both")
            colorbar.set_label("Gaussian half-maximum ellipse area (deg²)")
    fig.suptitle(
        f"Allen session {session_id}: uncorrected versus corrected RF-size surfaces\n"
        f"{bandwidth:g}° Gaussian kernel; matched support; effective local n ≥ {minimum_effective:g}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_diagnostic_figure(fits, primary, path, session_id=SESSION_ID):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    colors = {"V1": "#39738c", "HVA": "#d97736"}
    for group in ("V1", "HVA"):
        local = primary.loc[primary["group"].eq(group)]
        axes[0].scatter(
            local["allen_halfmax_area_deg2"], local["corrected_halfmax_area_deg2"],
            s=17, alpha=0.55, color=colors[group], label=f"{group} (n={len(local)})",
        )
        delta = np.log2(local["corrected_halfmax_area_deg2"] / local["allen_halfmax_area_deg2"])
        axes[1].hist(delta, bins=np.linspace(-6, 2, 33), histtype="step", linewidth=2,
                     color=colors[group], label=f"{group}; median {delta.median():+.2f}")
    joint = np.r_[primary["allen_halfmax_area_deg2"], primary["corrected_halfmax_area_deg2"]]
    lower, upper = np.quantile(joint[np.isfinite(joint) & (joint > 0)], [0.01, 0.99])
    axes[0].plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
                xlabel="Allen half-max area (deg²)", ylabel="Corrected half-max area (deg²)",
                title="Matched unit-level Gaussian area")
    axes[0].legend(frameon=False)
    axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="log₂(corrected / Allen area)", ylabel="Units",
                title="Correction magnitude")
    axes[1].legend(frameon=False)

    audit = []
    for group in ("V1", "HVA"):
        local = fits.loc[fits["group"].eq(group)]
        audit.extend(
            [
                (group, "All visual", len(local)),
                (group, "QC/significant", int(local["published_like_qc"].sum())),
                (group, "Corrected censored", int((local["published_like_qc"] & local["corrected_censored"]).sum())),
                (group, "Primary matched", int((primary["group"].eq(group)).sum())),
            ]
        )
    audit = pd.DataFrame(audit, columns=["group", "stage", "units"])
    stages = ["All visual", "QC/significant", "Corrected censored", "Primary matched"]
    x = np.arange(len(stages))
    for offset, group in ((-0.19, "V1"), (0.19, "HVA")):
        local = audit.loc[audit["group"].eq(group)].set_index("stage").loc[stages]
        axes[2].bar(x + offset, local["units"], width=0.36, color=colors[group], label=group)
        axes[2].set(xticks=x, xticklabels=["All", "QC", "Censored", "Primary"],
                ylabel="Units", title="Fit population and exclusions")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle(f"Allen session {session_id}: RF-fit correction diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT.parent / f"session_{args.session_id}"
    ).resolve()
    if output.exists() and any(output.iterdir()) and not (args.overwrite or args.reuse_fits):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    fit_path = output / "session_rf_fit_population.csv"
    if args.reuse_fits:
        fits = pd.read_csv(fit_path, low_memory=False)
        if "released_elevation_rf_deg" not in fits:
            released = pd.read_csv(
                args.unit_table.resolve(), low_memory=False,
                usecols=["ecephys_unit_id", "azimuth_rf", "elevation_rf"],
            ).rename(columns={
                "azimuth_rf": "released_azimuth_rf_deg",
                "elevation_rf": "released_elevation_rf_deg",
            })
            fits = fits.merge(released, on="ecephys_unit_id", how="left", validate="one_to_one")
        current_error = np.nanmedian(np.abs(
            fits["threshold_elevation_deg"] - fits["released_elevation_rf_deg"]
        ))
        flipped_error = np.nanmedian(np.abs(
            (20.0 - fits["threshold_elevation_deg"]) - fits["released_elevation_rf_deg"]
        ))
        if flipped_error < current_error:
            fits["threshold_elevation_deg"] = 20.0 - fits["threshold_elevation_deg"]
            fits["allen_center_elevation_deg"] = 20.0 - fits["allen_center_elevation_deg"]
            fits["corrected_center_elevation_deg"] = 20.0 - fits["corrected_center_elevation_deg"]
        fits.to_csv(fit_path, index=False, float_format="%.8g")
    else:
        table = pd.read_csv(args.unit_table.resolve(), low_memory=False)
        population = table.loc[
            table["ecephys_session_id"].eq(args.session_id)
            & table["ecephys_structure_acronym"].isin(AREA_MAP)
        ].copy()
        population["group"] = population["ecephys_structure_acronym"].map(AREA_MAP)
        fits = fit_population(args.nwb.resolve(), population, args.session_id)
        fits.to_csv(fit_path, index=False, float_format="%.8g")
    primary, surfaces, az_grid, el_grid = make_surfaces(
        fits, args.bandwidth_deg, args.minimum_effective_units
    )
    primary.to_csv(output / "surface_matched_qc_population.csv", index=False, float_format="%.8g")
    surface_rows = []
    for (group, model), (surface, effective) in surfaces.items():
        for row, elevation in enumerate(el_grid):
            for column, azimuth in enumerate(az_grid):
                surface_rows.append(
                    {"group": group, "model": model, "azimuth_deg": azimuth,
                     "elevation_deg": elevation, "halfmax_area_deg2": surface[row, column],
                     "effective_local_units": effective[row, column]}
                )
    pd.DataFrame(surface_rows).to_csv(
        output / "rf_size_surfaces.csv", index=False, float_format="%.8g"
    )
    surface_figure = output / "Figure_uncorrected_vs_corrected_rf_size_surfaces.png"
    diagnostic_figure = output / "Figure_rf_fit_correction_diagnostics.png"
    render_surface_figure(
        primary, surfaces, az_grid, el_grid, surface_figure,
        args.bandwidth_deg, args.minimum_effective_units, args.session_id,
    )
    render_diagnostic_figure(fits, primary, diagnostic_figure, args.session_id)

    comparison_by_group = {}
    for group, local in primary.groupby("group", observed=True):
        edge_distance = np.minimum.reduce(
            [
                local["threshold_azimuth_deg"] - 10.0,
                90.0 - local["threshold_azimuth_deg"],
                local["threshold_elevation_deg"] + 30.0,
                50.0 - local["threshold_elevation_deg"],
            ]
        )
        group_summary = {"units": len(local)}
        for model in ("allen", "corrected"):
            area = local[f"{model}_halfmax_area_deg2"]
            group_summary[f"{model}_median_halfmax_area_deg2"] = float(area.median())
            group_summary[f"{model}_q95_halfmax_area_deg2"] = float(area.quantile(0.95))
            group_summary[f"{model}_median_rmse"] = float(local[f"{model}_rmse"].median())
            group_summary[f"{model}_spearman_log2_area_vs_edge_distance"] = float(
                spearmanr(edge_distance, np.log2(area)).correlation
            )
            for touching in (False, True):
                selected = local.loc[local["threshold_component_touches_edge"].eq(touching)]
                label = "edge_touching" if touching else "not_edge_touching"
                group_summary[f"{model}_{label}_units"] = len(selected)
                group_summary[f"{model}_{label}_median_halfmax_area_deg2"] = (
                    float(selected[f"{model}_halfmax_area_deg2"].median())
                    if len(selected) else None
                )
        comparison_by_group[group] = group_summary

    summary = {
        "session_id": args.session_id,
        "visual_units_fitted": len(fits),
        "published_like_qc_units": int(fits["published_like_qc"].sum()),
        "allen_success_units": int(fits["allen_success"].sum()),
        "allen_finite_parameter_units": int(fits["allen_finite"].sum()),
        "corrected_success_units": int(fits["corrected_success"].sum()),
        "corrected_censored_qc_units": int(
            (fits["published_like_qc"] & fits["corrected_censored"]).sum()
        ),
        "corrected_sigma_upper_bound_qc_units": int(
            (fits["published_like_qc"] & fits["corrected_sigma_upper_bound"]).sum()
        ),
        "corrected_center_bound_qc_units": int(
            (fits["published_like_qc"] & fits["corrected_center_bound"]).sum()
        ),
        "primary_matched_surface_units": len(primary),
        "primary_by_group": primary["group"].value_counts().to_dict(),
        "median_log2_corrected_over_allen_area_by_group": {
            group: float(np.log2(
                local["corrected_halfmax_area_deg2"] / local["allen_halfmax_area_deg2"]
            ).median())
            for group, local in primary.groupby("group", observed=True)
        },
        "comparison_by_group": comparison_by_group,
        "corrected_model": {
            "baseline": "nonnegative fitted DC level",
            "amplitude": "nonnegative",
            "center_extension_px": CENTER_EXTENSION_PX,
            "sigma_upper_px": SIGMA_UPPER_PX,
            "enclosing_border": False,
        },
        "surface": {
            "coordinates": "direct threshold-map center",
            "metric": "2*pi*ln(2)*sigma_x*sigma_y; half-maximum ellipse area",
            "bandwidth_deg": args.bandwidth_deg,
            "minimum_effective_units": args.minimum_effective_units,
            "support": "identical published-like QC units with finite Allen parameters and successful non-censored corrected fits",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# Allen session {args.session_id}: corrected RF Gaussian surfaces",
        "",
        f"All {len(fits)} units in canonical V1/HVA structures were fitted with Allen's no-baseline Gaussian and a corrected nonnegative-baseline Gaussian. No enclosing border or pseudo-observations were used.",
        "",
        f"The primary surfaces use {len(primary)} matched units that pass published-like RF/QC filters, have finite Allen parameters, and have a successful corrected fit without its center or upper sigma limit being reached. Allen's implementation returns and releases finite parameters even when its least-squares convergence flag is false, so that flag is retained as an audit field rather than used as a selection filter. RF location is the direct threshold-map center for both models, so only the Gaussian size estimate changes.",
        "",
        "The plotted size is Gaussian half-maximum ellipse area, `2*pi*ln(2)*sigma_x*sigma_y`. V1 sigma is bounded at 40 degrees and HVA sigma at 50 degrees; fits reaching an upper sigma or center-extension bound are labeled censored and excluded from both matched surfaces.",
        "",
        f"In V1, median half-maximum area changes from {comparison_by_group['V1']['allen_median_halfmax_area_deg2']:.0f} to {comparison_by_group['V1']['corrected_median_halfmax_area_deg2']:.0f} deg2, and Spearman rho between log2 area and distance from the nearest sampled edge changes from {comparison_by_group['V1']['allen_spearman_log2_area_vs_edge_distance']:+.2f} to {comparison_by_group['V1']['corrected_spearman_log2_area_vs_edge_distance']:+.2f}.",
        f"In pooled HVAs, median half-maximum area changes from {comparison_by_group['HVA']['allen_median_halfmax_area_deg2']:.0f} to {comparison_by_group['HVA']['corrected_median_halfmax_area_deg2']:.0f} deg2. HVA support in this session is strongly boundary concentrated, so its spatial surface should be read within the effective-sample contours.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_path = output / "run_manifest.json"
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir()) if path.is_file() and path != manifest_path
    }
    manifest = {
        "nwb": {"path": str(args.nwb.resolve()), "sha256": sha256(args.nwb.resolve())},
        "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote full-session corrected RF surfaces to {output}")


if __name__ == "__main__":
    main()
