#!/usr/bin/env python3
"""Fit point/aperture and axis-aligned/rotated RF models across cached sessions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_allen_point_vs_aperture_rf import fit_unit, point_spatial, aperture_spatial
from render_allen_rotated_point_aperture_rf_examples import (
    fit_rotated,
    rotated_spatial,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS = (746083955, 755434585, 760693773, 798911424)
DEFAULT_CACHE = ROOT / "artifacts" / "allen_population_gaze_rf"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_multisession_rf_validation_v1" / "03_geometry"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-units", type=int, default=None)
    parser.add_argument(
        "--rotation-limit", type=int, default=80,
        help="Balanced evaluation-neuron subset per session for costly five-start rotation fits.",
    )
    parser.add_argument(
        "--rotation-all-units", action="store_true",
        help="Draw rotation fits from every selected unit instead of evaluation units only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--per-session-only", action="store_true",
        help="Write independent session directories and skip shared aggregate outputs.",
    )
    parser.add_argument(
        "--all-gabor-trials", action="store_true",
        help="Use every Gabor presentation; do not require a valid eye-tracking sample.",
    )
    return parser.parse_args()


def axis_parameter_vector(parameters):
    return np.asarray(parameters, dtype=float)


def select_units(population, limit):
    selected = population.loc[population["published_like_qc"].astype(bool)].copy()
    if limit is None or limit >= len(selected):
        return selected
    pieces = []
    pools = {
        key: list(local.index)
        for key, local in selected.groupby(["group", "unit_split"], observed=True)
    }
    keys = [("V1", "calibration"), ("V1", "evaluation"),
            ("HVA", "calibration"), ("HVA", "evaluation")]
    while len(pieces) < limit and any(pools.values()):
        for key in keys:
            if pools.get(key) and len(pieces) < limit:
                pieces.append(pools[key].pop(0))
    return selected.loc[pieces].sort_index()


def rotation_subset(selected, limit, all_units=False):
    eligible = selected if all_units else selected.loc[selected["unit_split"].eq("evaluation")]
    if limit is None or limit >= len(eligible):
        return set(eligible["ecephys_unit_id"].astype(int))
    ids = []
    pools = {
        group: list(local["ecephys_unit_id"].astype(int))
        for group, local in eligible.groupby("group", observed=True)
    }
    while len(ids) < limit and any(pools.values()):
        for group in ("V1", "HVA"):
            if pools.get(group) and len(ids) < limit:
                ids.append(pools[group].pop(0))
    return set(ids)


def fit_session(session_id, cache_root, output_root, limit, rotation_limit, resume, overwrite,
                all_gabor_trials, rotation_all_units):
    cache = cache_root / f"session_{session_id}"
    output = output_root / f"session_{session_id}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "unit_geometry_fits.csv"
    if result_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"{result_path} exists; use --resume or --overwrite")

    population = pd.read_csv(cache / "visual_unit_population.csv", low_memory=False)
    trials = pd.read_csv(cache / "gabor_trial_gaze_table.csv", low_memory=False)
    spikes = np.load(cache / "gabor_spike_counts.npz")
    counts = spikes["counts"]
    if not np.array_equal(spikes["unit_ids"].astype(int), population["ecephys_unit_id"].to_numpy(int)):
        raise ValueError(f"Session {session_id}: spike/unit order mismatch")
    selected = select_units(population, limit)
    rotation_ids = rotation_subset(selected, rotation_limit, all_units=rotation_all_units)
    valid = (
        np.ones(len(trials), dtype=bool)
        if all_gabor_trials
        else trials["valid_gaze"].to_numpy(bool)
    )
    train = valid & trials["trial_split"].eq("train").to_numpy(bool)
    test = valid & trials["trial_split"].eq("test").to_numpy(bool)
    x = trials["x_position"].to_numpy(float)
    y = trials["y_position"].to_numpy(float)
    orientation = trials["orientation_index"].to_numpy(int)

    existing = pd.DataFrame()
    if resume and result_path.exists():
        existing = pd.read_csv(result_path, low_memory=False)
    rows = existing.to_dict("records")
    done = set(zip(existing.get("ecephys_unit_id", []), existing.get("spatial_model", [])))
    for progress, unit in enumerate(selected.itertuples(), start=1):
        unit_id = int(unit.ecephys_unit_id)
        unit_counts = counts[unit.Index].astype(float)
        for model in ("point", "aperture"):
            if (unit_id, model) in done:
                continue
            model_started = time.perf_counter()
            axis_parameters, axis = fit_unit(
                unit_counts, x, y, orientation, train, test, unit.group,
                spatial_model=model,
            )
            if unit_id in rotation_ids:
                rotated_parameters, rotated = fit_rotated(
                    unit_counts, x, y, orientation, train, test, unit.group,
                    model, axis_parameter_vector(axis_parameters),
                )
            else:
                rotated_parameters = np.array([])
                rotated = {
                    "center_x_deg": np.nan, "center_y_deg": np.nan,
                    "sigma_major_deg": np.nan, "sigma_minor_deg": np.nan,
                    "major_axis_angle_deg": np.nan, "axis_ratio": np.nan,
                    "latent_halfmax_area_deg2": np.nan,
                    "train_poisson_deviance": np.nan, "test_poisson_deviance": np.nan,
                    "sigma_lower_bound": False, "sigma_upper_bound": False,
                    "optimizer_nfev": np.nan,
                }
            edge_distance = min(
                axis["center_x_deg"] + 40.0, 40.0 - axis["center_x_deg"],
                axis["center_y_deg"] + 40.0, 40.0 - axis["center_y_deg"],
            )
            rows.append({
                "session_id": session_id,
                "ecephys_unit_id": unit_id,
                "group": unit.group,
                "ecephys_structure_acronym": unit.ecephys_structure_acronym,
                "unit_split": unit.unit_split,
                "spatial_model": model,
                "released_azimuth_rf_deg": unit.azimuth_rf,
                "released_elevation_rf_deg": unit.elevation_rf,
                "released_area_rf_deg2": unit.area_rf,
                "released_width_rf_deg": unit.width_rf,
                "released_height_rf_deg": unit.height_rf,
                "axis_center_x_deg": axis["center_x_deg"],
                "axis_center_y_deg": axis["center_y_deg"],
                "axis_sigma_x_deg": axis["sigma_x_deg"],
                "axis_sigma_y_deg": axis["sigma_y_deg"],
                "axis_area_deg2": axis["latent_halfmax_area_deg2"],
                "axis_train_deviance": axis["train_poisson_deviance"],
                "axis_test_deviance": axis["test_poisson_deviance"],
                "axis_optimizer_nfev": axis["optimizer_nfev"],
                "axis_censored": axis["censored"],
                "axis_edge_distance_deg": edge_distance,
                "rotation_center_x_deg": rotated["center_x_deg"],
                "rotation_center_y_deg": rotated["center_y_deg"],
                "rotation_sigma_major_deg": rotated["sigma_major_deg"],
                "rotation_sigma_minor_deg": rotated["sigma_minor_deg"],
                "rotation_angle_deg": rotated["major_axis_angle_deg"],
                "rotation_axis_ratio": rotated["axis_ratio"],
                "rotation_area_deg2": rotated["latent_halfmax_area_deg2"],
                "rotation_train_deviance": rotated["train_poisson_deviance"],
                "rotation_test_deviance": rotated["test_poisson_deviance"],
                "rotation_optimizer_nfev": rotated["optimizer_nfev"],
                "rotation_test_gain": (axis["test_poisson_deviance"] - rotated["test_poisson_deviance"]
                                       if unit_id in rotation_ids else np.nan),
                "rotation_train_gain": (axis["train_poisson_deviance"] - rotated["train_poisson_deviance"]
                                        if unit_id in rotation_ids else np.nan),
                "rotation_sigma_lower_bound": rotated["sigma_lower_bound"],
                "rotation_sigma_upper_bound": rotated["sigma_upper_bound"],
                "axis_parameters": json.dumps(axis_parameters.tolist()),
                "rotation_parameters": json.dumps(rotated_parameters.tolist()),
                "model_elapsed_seconds": time.perf_counter() - model_started,
            })
        if progress % 10 == 0 or progress == len(selected):
            pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.9g")
            print(f"Session {session_id}: {progress}/{len(selected)} units", flush=True)
    result = pd.DataFrame(rows).sort_values(["ecephys_unit_id", "spatial_model"])
    result.to_csv(result_path, index=False, float_format="%.9g")
    render_population(result, output / "Figure_geometry_population.png", session_id)
    render_examples(result, population, trials, counts, output / "Figure_geometry_examples.png", session_id)
    return result


def render_population(table, path, session_id):
    evaluation = table.loc[table["unit_split"].eq("evaluation")].copy()
    point = evaluation.loc[evaluation["spatial_model"].eq("point")].set_index("ecephys_unit_id")
    aperture = evaluation.loc[evaluation["spatial_model"].eq("aperture")].set_index("ecephys_unit_id")
    common = point.index.intersection(aperture.index)
    colors = {"V1": "#3366aa", "HVA": "#d97736"}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4), constrained_layout=True)
    for group in ("V1", "HVA"):
        ids = [uid for uid in common if point.loc[uid, "group"] == group]
        axes[0, 0].scatter(point.loc[ids, "axis_area_deg2"], aperture.loc[ids, "axis_area_deg2"],
                           s=18, alpha=.55, color=colors[group], label=f"{group} (n={len(ids)})")
        axes[0, 1].hist(point.loc[ids, "rotation_test_gain"], bins=28, histtype="step",
                        linewidth=2, color=colors[group], label=group)
        axes[1, 0].scatter(aperture.loc[ids, "axis_edge_distance_deg"],
                           np.log2(aperture.loc[ids, "axis_area_deg2"]), s=18, alpha=.5,
                           color=colors[group], label=group)
        axes[1, 1].scatter(point.loc[ids, "rotation_test_gain"],
                           aperture.loc[ids, "rotation_test_gain"], s=18, alpha=.5,
                           color=colors[group], label=group)
    all_area = np.r_[point.loc[common, "axis_area_deg2"], aperture.loc[common, "axis_area_deg2"]]
    lo, hi = np.nanquantile(all_area[all_area > 0], [.01, .99])
    axes[0, 0].plot([lo, hi], [lo, hi], "--", color="#555555", lw=1)
    axes[0, 0].set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
                   xlabel="Point latent half-max area (deg²)",
                   ylabel="Aperture latent half-max area (deg²)", title="Aperture correction")
    axes[0, 1].axvline(0, color="#555555", ls="--", lw=1)
    axes[0, 1].set(xlabel="Point rotation held-out deviance gain", ylabel="Evaluation units",
                   title="Does tilt generalize?")
    axes[1, 0].axvline(0, color="#555555", ls=":", lw=1)
    axes[1, 0].set(xlabel="Distance of fitted center inside sampled grid (deg)",
                   ylabel="log₂ aperture latent area (deg²)", title="Residual edge dependence")
    lim = np.nanquantile(np.abs(np.r_[point.loc[common, "rotation_test_gain"],
                                    aperture.loc[common, "rotation_test_gain"]]), .99)
    lim = max(float(lim), 1e-4)
    axes[1, 1].plot([-lim, lim], [-lim, lim], "--", color="#555555", lw=1)
    axes[1, 1].axhline(0, color="#aaaaaa", lw=.8); axes[1, 1].axvline(0, color="#aaaaaa", lw=.8)
    axes[1, 1].set(xlim=(-lim, lim), ylim=(-lim, lim),
                   xlabel="Point rotation gain", ylabel="Aperture rotation gain",
                   title="Tilt evidence across stimulus models")
    for axis in axes.ravel():
        axis.grid(alpha=.14); axis.legend(frameon=False)
    fig.suptitle(f"Allen session {session_id}: RF geometry on held-out repeats", fontsize=15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def spatial_map(parameters, xx, yy, model, rotated):
    if rotated:
        return rotated_spatial(xx.ravel(), yy.ravel(), *parameters[4:9], model=model).reshape(xx.shape)
    if model == "point":
        return point_spatial(xx, yy, *parameters[4:8])
    return aperture_spatial(xx.ravel(), yy.ravel(), *parameters[4:8]).reshape(xx.shape)


def render_examples(table, population, trials, counts, path, session_id):
    eval_ap = table.loc[(table["unit_split"].eq("evaluation")) &
                        (table["spatial_model"].eq("aperture")) &
                        table["rotation_test_gain"].notna()].copy()
    interior = eval_ap.loc[eval_ap["axis_edge_distance_deg"].gt(15)]
    selections = []
    if len(interior):
        target = interior["axis_area_deg2"].median()
        selections.append((interior.loc[(interior["axis_area_deg2"] - target).abs().idxmin()],
                           "typical interior"))
    edge = eval_ap.sort_values("axis_edge_distance_deg").iloc[0]
    selections.append((edge, "nearest edge"))
    selections.append((eval_ap.loc[eval_ap["rotation_test_gain"].idxmax()], "largest tilt gain"))
    selections.append((eval_ap.loc[eval_ap["rotation_test_gain"].idxmin()], "largest tilt loss"))
    unique = []; seen = set()
    for row, label in selections:
        if int(row.ecephys_unit_id) not in seen:
            unique.append((row, label)); seen.add(int(row.ecephys_unit_id))
    valid_test = trials["valid_gaze"].to_numpy(bool) & trials["trial_split"].eq("test").to_numpy(bool)
    x_trial = trials["x_position"].to_numpy(float); y_trial = trials["y_position"].to_numpy(float)
    grid = np.linspace(-45, 45, 91); xx, yy = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(len(unique), 5, figsize=(16, 3.25 * len(unique)), squeeze=False,
                             constrained_layout=True)
    titles = ["Held-out map", "Point axis", "Point rotated", "Aperture axis", "Aperture rotated"]
    for c, title in enumerate(titles): axes[0, c].set_title(title)
    for r, (selected, label) in enumerate(unique):
        uid = int(selected.ecephys_unit_id)
        pop_index = population.index[population["ecephys_unit_id"].eq(uid)][0]
        observed = pd.DataFrame({"x": x_trial[valid_test], "y": y_trial[valid_test],
                                 "count": counts[pop_index].astype(float)[valid_test]}).groupby(
                                     ["y", "x"], observed=True)["count"].mean().unstack("x")
        observed = observed.reindex(index=np.arange(-40, 41, 10), columns=np.arange(-40, 41, 10))
        axes[r, 0].imshow(observed.to_numpy(float), origin="lower", extent=(-45,45,-45,45),
                          interpolation="nearest", cmap="magma", aspect="equal")
        for c, (model, rotated) in enumerate((("point", False), ("point", True),
                                               ("aperture", False), ("aperture", True)), start=1):
            row = table.loc[(table["ecephys_unit_id"].eq(uid)) &
                            (table["spatial_model"].eq(model))].iloc[0]
            parameters = np.asarray(json.loads(row["rotation_parameters" if rotated else "axis_parameters"]), float)
            shape = spatial_map(parameters, xx, yy, model, rotated)
            axes[r, c].pcolormesh(grid, grid, shape, shading="auto", cmap="magma", vmin=0, vmax=1)
            axes[r, c].contour(xx, yy, shape, levels=[.5], colors="#00e5ff", linewidths=1.25)
            axes[r, c].plot(parameters[4], parameters[5], "+", color="#00e5ff", ms=7)
        axes[r, 0].set_ylabel(f"{label}\nunit {uid} · {selected.group}")
        for axis in axes[r]:
            axis.set(xlim=(-45,45), ylim=(-45,45), xticks=[-40,0,40], yticks=[-40,0,40], aspect="equal")
    fig.suptitle(f"Session {session_id}: auditable RF geometry cases", fontsize=15)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(all_fits):
    evaluation = all_fits.loc[all_fits["unit_split"].eq("evaluation")].copy()
    rows = []
    for keys, local in evaluation.groupby(["session_id", "group", "spatial_model"], observed=True):
        rotated_local = local.loc[local["rotation_test_gain"].notna()]
        rows.append({
            "session_id": keys[0], "group": keys[1], "spatial_model": keys[2],
            "evaluation_units": len(local),
            "rotation_units": int(local["rotation_test_gain"].notna().sum()),
            "median_axis_area_deg2": local["axis_area_deg2"].median(),
            "median_rotation_test_gain": local["rotation_test_gain"].median(),
            "fraction_rotation_gain_positive": rotated_local["rotation_test_gain"].gt(0).mean(),
            "median_rotation_area_ratio": np.nanmedian(local["rotation_area_deg2"] / local["axis_area_deg2"]),
            "axis_censored_fraction": local["axis_censored"].mean(),
            "rotation_bound_fraction": (rotated_local["rotation_sigma_lower_bound"] |
                                        rotated_local["rotation_sigma_upper_bound"]).mean(),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    session_tables = []
    for session_id in args.sessions:
        session_tables.append(fit_session(session_id, args.cache_root.resolve(), output,
                                          args.limit_units, args.rotation_limit,
                                          args.resume, args.overwrite,
                                          args.all_gabor_trials,
                                          args.rotation_all_units))
    if args.per_session_only:
        return
    all_fits = pd.concat(session_tables, ignore_index=True)
    all_fits.to_csv(output / "all_session_unit_geometry_fits.csv", index=False, float_format="%.9g")
    summary = summarize(all_fits)
    summary.to_csv(output / "all_session_geometry_summary.csv", index=False, float_format="%.9g")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
