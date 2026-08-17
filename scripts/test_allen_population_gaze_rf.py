#!/usr/bin/env python3
"""Test population-calibrated gaze correction of Allen Gabor RFs in one session."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
DEFAULT_NWB = Path(
    "/media/huklaban5/Data/MouseV2/allen_v1_bridge/000021/"
    "sub-726170927/sub-726170927_ses-746083955.nwb"
)
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "allen_population_gaze_rf"
AREA_MAP = {
    "VISp": "V1",
    "VISl": "HVA",
    "VISrl": "HVA",
    "VISal": "HVA",
    "VISpm": "HVA",
    "VISam": "HVA",
}
SIGMA_UPPER_DEG = {"V1": 40.0, "HVA": 50.0}
CENTER_EXTENSION_DEG = 20.0
GAINS = (0.0, 0.5, 1.0, 1.5)
RESPONSE_WINDOW_S = 0.249
MIN_GAZE_SAMPLES = 3
RANDOM_SEED = 20260815
PREDECLARED_CASE = 951888972


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--session-id", type=int, default=SESSION_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Write compact unit/trial/spike caches, then stop before fitting.",
    )
    parser.add_argument(
        "--reuse-extracted", action="store_true",
        help="Reuse compact caches in output-dir and skip the NWB load.",
    )
    parser.add_argument(
        "--skip-gaze", action="store_true",
        help="Extract all Gabor trials and spike counts without loading eye tracking.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_session(path):
    return EcephysSession.from_nwb_path(
        path,
        api_kwargs={
            "amplitude_cutoff_maximum": np.inf,
            "presence_ratio_minimum": -np.inf,
            "isi_violations_maximum": np.inf,
            "filter_by_validity": False,
        },
    )


def prepare_population(unit_table, session_id=SESSION_ID):
    table = pd.read_csv(unit_table, low_memory=False)
    result = table.loc[
        table["ecephys_session_id"].eq(session_id)
        & table["ecephys_structure_acronym"].isin(AREA_MAP)
    ].copy()
    result["group"] = result["ecephys_structure_acronym"].map(AREA_MAP)
    result["published_like_qc"] = (
        result["p_value_rf"].lt(0.01)
        & result["area_rf"].lt(2500)
        & result["snr"].gt(1)
        & result["firing_rate_dg"].gt(0.1)
    )
    result = result.sort_values("ecephys_unit_id").reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    result["unit_split"] = "descriptive"
    for group in ("V1", "HVA"):
        indices = result.index[result["published_like_qc"] & result["group"].eq(group)].to_numpy()
        shuffled = rng.permutation(indices)
        midpoint = len(shuffled) // 2
        result.loc[shuffled[:midpoint], "unit_split"] = "calibration"
        result.loc[shuffled[midpoint:], "unit_split"] = "evaluation"
    return result


def presentation_gaze_medians(gabors, gaze):
    time = gaze.index.to_numpy(float)
    x = gaze["filtered_screen_coordinates_spherical_x_deg"].to_numpy(float)
    y = gaze["filtered_screen_coordinates_spherical_y_deg"].to_numpy(float)
    starts = gabors["start_time"].to_numpy(float)
    stops = starts + RESPONSE_WINDOW_S
    median_x = np.full(len(gabors), np.nan)
    median_y = np.full(len(gabors), np.nan)
    samples = np.zeros(len(gabors), dtype=int)
    within_sd = np.full(len(gabors), np.nan)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(time, start, side="left")
        right = np.searchsorted(time, stop, side="right")
        local_x = x[left:right]
        local_y = y[left:right]
        valid = np.isfinite(local_x) & np.isfinite(local_y)
        samples[index] = int(valid.sum())
        if samples[index] >= MIN_GAZE_SAMPLES:
            median_x[index] = float(np.median(local_x[valid]))
            median_y[index] = float(np.median(local_y[valid]))
            within_sd[index] = float(np.sqrt(np.var(local_x[valid]) + np.var(local_y[valid])))
    valid = np.isfinite(median_x) & np.isfinite(median_y)
    center_x = float(np.median(median_x[valid]))
    center_y = float(np.median(median_y[valid]))
    return median_x - center_x, median_y - center_y, samples, within_sd, center_x, center_y


def prepare_trials(session, session_id=SESSION_ID, include_gaze=True):
    presentations = session.stimulus_presentations
    gabors = presentations.loc[presentations["stimulus_name"].eq("gabors")].copy()
    gabors = gabors.sort_values("start_time").reset_index().rename(
        columns={"index": "stimulus_presentation_id"}
    )
    if include_gaze:
        gaze = session.get_screen_gaze_data(include_filtered_data=True)
        if gaze is None:
            raise RuntimeError(f"Session {session_id} has no screen-gaze data")
        dx, dy, gaze_samples, within_sd, gaze_center_x, gaze_center_y = presentation_gaze_medians(
            gabors, gaze
        )
    else:
        dx = np.full(len(gabors), np.nan)
        dy = np.full(len(gabors), np.nan)
        gaze_samples = np.zeros(len(gabors), dtype=int)
        within_sd = np.full(len(gabors), np.nan)
        gaze_center_x = np.nan
        gaze_center_y = np.nan
    gabors["gaze_dx_deg"] = dx
    gabors["gaze_dy_deg"] = dy
    gabors["gaze_samples"] = gaze_samples
    gabors["gaze_within_trial_sd_deg"] = within_sd
    gabors["valid_gaze"] = np.isfinite(dx) & np.isfinite(dy)
    gabors["repeat_index_within_condition"] = gabors.groupby(
        "stimulus_condition_id", observed=True
    ).cumcount()
    gabors["trial_split"] = np.where(
        gabors["repeat_index_within_condition"].mod(3).eq(2), "test", "train"
    )
    orientation_values = sorted(gabors["orientation"].astype(float).unique())
    orientation_map = {value: index for index, value in enumerate(orientation_values)}
    gabors["orientation_index"] = gabors["orientation"].astype(float).map(orientation_map)
    metadata = {
        "gaze_center_spherical_x_deg": gaze_center_x,
        "gaze_center_spherical_y_deg": gaze_center_y,
        "orientation_values_deg": orientation_values,
    }
    return gabors, metadata


def spike_count_matrix(session, population, trials):
    starts = trials["start_time"].to_numpy(float)
    stops = starts + RESPONSE_WINDOW_S
    counts = np.zeros((len(population), len(trials)), dtype=np.int16)
    spike_times = session.spike_times
    for row, unit_id in enumerate(population["ecephys_unit_id"].astype(int)):
        spikes = np.asarray(spike_times.get(unit_id, np.array([])), dtype=float)
        left = np.searchsorted(spikes, starts, side="left")
        right = np.searchsorted(spikes, stops, side="left")
        local = right - left
        counts[row] = np.minimum(local, np.iinfo(np.int16).max).astype(np.int16)
    return counts


def candidate_table(trials):
    rng = np.random.default_rng(RANDOM_SEED + 1)
    # Shuffle only among presentations with usable gaze. Invalid rows remain in
    # place so this control uses exactly the same observations as real gaze.
    permutation = np.arange(len(trials))
    valid_indices = np.flatnonzero(trials["valid_gaze"].to_numpy(dtype=bool))
    permutation[valid_indices] = rng.permutation(valid_indices)
    candidates = []
    for gain_x in GAINS:
        for gain_y in GAINS:
            candidates.append(
                {
                    "candidate": f"gain_x_{gain_x:g}_gain_y_{gain_y:g}",
                    "gain_x": gain_x,
                    "gain_y": gain_y,
                    "control": False,
                    "shuffle": False,
                }
            )
    candidates.extend(
        [
            {"candidate": "sign_reversed", "gain_x": -1.0, "gain_y": -1.0,
             "control": True, "shuffle": False},
            {"candidate": "time_shuffled", "gain_x": 1.0, "gain_y": 1.0,
             "control": True, "shuffle": True},
        ]
    )
    return pd.DataFrame(candidates), permutation


def transformed_coordinates(trials, candidate, permutation):
    dx = trials["gaze_dx_deg"].to_numpy(float)
    dy = trials["gaze_dy_deg"].to_numpy(float)
    if bool(candidate.shuffle):
        dx = dx[permutation]
        dy = dy[permutation]
    x = trials["x_position"].astype(float).to_numpy() - float(candidate.gain_x) * dx
    y = trials["y_position"].astype(float).to_numpy() - float(candidate.gain_y) * dy
    return x, y


def model_prediction(parameters, x, y, orientation_index):
    baseline = parameters[0]
    amplitudes = parameters[1:4]
    center_x, center_y, sigma_x, sigma_y = parameters[4:8]
    spatial = np.exp(
        -0.5 * (((x - center_x) / sigma_x) ** 2 + ((y - center_y) / sigma_y) ** 2)
    )
    return baseline + amplitudes[orientation_index] * spatial


def initial_parameters(counts, x, y, orientation_index):
    baseline = max(float(np.quantile(counts, 0.20)), 0.0)
    rounded_x = np.round(x / 10.0) * 10.0
    rounded_y = np.round(y / 10.0) * 10.0
    frame = pd.DataFrame({"x": rounded_x, "y": rounded_y, "count": counts})
    means = frame.groupby(["x", "y"], observed=True)["count"].mean()
    peak_x, peak_y = means.idxmax()
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


def fit_unit(counts, x, y, orientation_index, train, test, group, start=None):
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
            parameters, x[train], y[train], orientation_index[train]
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
        parameters, x[train], y[train], orientation_index[train]
    )
    test_prediction = model_prediction(
        parameters, x[test], y[test], orientation_index[test]
    )
    center_bound = bool(
        np.any(np.isclose(parameters[4:6], lower[4:6], atol=1e-4, rtol=0))
        or np.any(np.isclose(parameters[4:6], upper[4:6], atol=1e-4, rtol=0))
    )
    sigma_upper_bound = bool(
        np.any(np.isclose(parameters[6:8], upper[6:8], atol=1e-4, rtol=0))
    )
    halfmax_area = float(2.0 * np.pi * np.log(2.0) * parameters[6] * parameters[7])
    mean_amplitude = float(np.mean(parameters[1:4]))
    return parameters, {
        "success": bool(result.success and np.all(np.isfinite(parameters))),
        "optimizer_status": int(result.status),
        "center_bound": center_bound,
        "sigma_upper_bound": sigma_upper_bound,
        "censored": bool(center_bound or sigma_upper_bound),
        "baseline_spikes": float(parameters[0]),
        "mean_amplitude_spikes": mean_amplitude,
        "peak_spikes": float(parameters[0] + np.max(parameters[1:4])),
        "center_x_deg": float(parameters[4]),
        "center_y_deg": float(parameters[5]),
        "sigma_x_deg": float(parameters[6]),
        "sigma_y_deg": float(parameters[7]),
        "major_sigma_deg": float(max(parameters[6], parameters[7])),
        "halfmax_area_deg2": halfmax_area,
        "train_poisson_deviance": poisson_deviance(counts[train], train_prediction),
        "test_poisson_deviance": poisson_deviance(counts[test], test_prediction),
        "train_anscombe_rmse": float(np.sqrt(np.mean(np.square(residual(parameters))))),
        "test_anscombe_rmse": float(np.sqrt(np.mean(np.square(
            2.0 * (np.sqrt(np.maximum(test_prediction, 0) + 3.0 / 8.0)
                   - np.sqrt(counts[test] + 3.0 / 8.0))
        )))),
    }


def run_candidate_sweep(population, trials, counts, candidates, permutation):
    valid = trials["valid_gaze"].to_numpy(bool)
    train = valid & trials["trial_split"].eq("train").to_numpy(bool)
    test = valid & trials["trial_split"].eq("test").to_numpy(bool)
    orientation = trials["orientation_index"].to_numpy(int)
    coordinate_lookup = {
        candidate.candidate: transformed_coordinates(trials, candidate, permutation)
        for candidate in candidates.itertuples(index=False)
    }
    eligible = population.loc[population["published_like_qc"]].copy()
    rows = []
    for progress, unit in enumerate(eligible.itertuples(), start=1):
        unit_counts = counts[unit.Index].astype(float)
        nominal_parameters = None
        for candidate in candidates.itertuples(index=False):
            x, y = coordinate_lookup[candidate.candidate]
            parameters, metrics = fit_unit(
                unit_counts, x, y, orientation, train, test, unit.group,
                start=nominal_parameters,
            )
            if candidate.candidate == "gain_x_0_gain_y_0":
                nominal_parameters = parameters.copy()
            rows.append(
                {
                    "ecephys_unit_id": int(unit.ecephys_unit_id),
                    "group": unit.group,
                    "ecephys_structure_acronym": unit.ecephys_structure_acronym,
                    "unit_split": unit.unit_split,
                    "candidate": candidate.candidate,
                    "gain_x": candidate.gain_x,
                    "gain_y": candidate.gain_y,
                    "control": candidate.control,
                    "shuffle": candidate.shuffle,
                    **metrics,
                }
            )
        if progress % 25 == 0 or progress == len(eligible):
            print(f"Gain sweep: fitted {progress}/{len(eligible)} QC units", flush=True)
    result = pd.DataFrame(rows)
    nominal = result.loc[result["candidate"].eq("gain_x_0_gain_y_0"),
                         ["ecephys_unit_id", "test_poisson_deviance",
                          "test_anscombe_rmse", "halfmax_area_deg2",
                          "mean_amplitude_spikes"]].rename(
        columns={
            "test_poisson_deviance": "nominal_test_poisson_deviance",
            "test_anscombe_rmse": "nominal_test_anscombe_rmse",
            "halfmax_area_deg2": "nominal_halfmax_area_deg2",
            "mean_amplitude_spikes": "nominal_mean_amplitude_spikes",
        }
    )
    result = result.merge(nominal, on="ecephys_unit_id", how="left")
    result["test_deviance_improvement"] = (
        result["nominal_test_poisson_deviance"] - result["test_poisson_deviance"]
    )
    result["test_anscombe_rmse_improvement"] = (
        result["nominal_test_anscombe_rmse"] - result["test_anscombe_rmse"]
    )
    result["log2_area_ratio"] = np.log2(
        result["halfmax_area_deg2"] / result["nominal_halfmax_area_deg2"]
    )
    result["log2_amplitude_ratio"] = np.log2(
        (result["mean_amplitude_spikes"] + 1e-6)
        / (result["nominal_mean_amplitude_spikes"] + 1e-6)
    )
    return result, coordinate_lookup, train, test


def summarize_candidates(results):
    rows = []
    for candidate, local in results.groupby("candidate", observed=True):
        selected = local.loc[local["unit_split"].eq("calibration")]
        rows.append(
            {
                "candidate": candidate,
                "gain_x": local["gain_x"].iloc[0],
                "gain_y": local["gain_y"].iloc[0],
                "control": bool(local["control"].iloc[0]),
                "calibration_units": len(selected),
                "median_test_deviance_improvement": selected["test_deviance_improvement"].median(),
                "fraction_test_deviance_improved": selected["test_deviance_improvement"].gt(0).mean(),
                "median_test_rmse_improvement": selected["test_anscombe_rmse_improvement"].median(),
                "median_log2_area_ratio": selected["log2_area_ratio"].median(),
                "median_log2_amplitude_ratio": selected["log2_amplitude_ratio"].median(),
                "failure_or_censored_fraction": (
                    ~selected["success"].astype(bool) | selected["censored"].astype(bool)
                ).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "median_test_deviance_improvement", ascending=False
    )


def select_best_candidate(summary):
    candidates = summary.loc[~summary["control"]].copy()
    return str(candidates.iloc[0]["candidate"])


def fit_full_population_chosen(population, trials, counts, sweep, coordinate_lookup,
                               train, test, chosen):
    orientation = trials["orientation_index"].to_numpy(int)
    rows = []
    existing = sweep.loc[sweep["candidate"].isin(["gain_x_0_gain_y_0", chosen])].copy()
    existing["fit_scope"] = "gain_sweep_qc"
    rows.extend(existing.to_dict("records"))
    remaining = population.loc[~population["published_like_qc"]]
    full_candidate_names = list(dict.fromkeys(["gain_x_0_gain_y_0", chosen]))
    for progress, unit in enumerate(remaining.itertuples(), start=1):
        unit_counts = counts[unit.Index].astype(float)
        nominal_parameters = None
        for candidate in full_candidate_names:
            x, y = coordinate_lookup[candidate]
            parameters, metrics = fit_unit(
                unit_counts, x, y, orientation, train, test, unit.group,
                start=nominal_parameters,
            )
            if candidate == "gain_x_0_gain_y_0":
                nominal_parameters = parameters.copy()
            rows.append(
                {
                    "ecephys_unit_id": int(unit.ecephys_unit_id),
                    "group": unit.group,
                    "ecephys_structure_acronym": unit.ecephys_structure_acronym,
                    "unit_split": unit.unit_split,
                    "candidate": candidate,
                    "gain_x": np.nan,
                    "gain_y": np.nan,
                    "control": False,
                    "shuffle": False,
                    "fit_scope": "full_population_descriptive",
                    **metrics,
                }
            )
        if progress % 50 == 0 or progress == len(remaining):
            print(f"Full population: fitted {progress}/{len(remaining)} additional units", flush=True)
    return pd.DataFrame(rows)


def evaluation_summary(sweep, chosen):
    selected = sweep.loc[
        sweep["candidate"].eq(chosen) & sweep["unit_split"].eq("evaluation")
    ].copy()
    rows = []
    for group in ("V1", "HVA", "all"):
        local = selected if group == "all" else selected.loc[selected["group"].eq(group)]
        rows.append(
            {
                "group": group,
                "evaluation_units": len(local),
                "median_test_deviance_improvement": local["test_deviance_improvement"].median(),
                "fraction_test_deviance_improved": local["test_deviance_improvement"].gt(0).mean(),
                "median_test_rmse_improvement": local["test_anscombe_rmse_improvement"].median(),
                "median_log2_area_ratio": local["log2_area_ratio"].median(),
                "median_log2_amplitude_ratio": local["log2_amplitude_ratio"].median(),
                "failure_or_censored_fraction": (
                    ~local["success"].astype(bool) | local["censored"].astype(bool)
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def select_cases(sweep, chosen):
    local = sweep.loc[
        sweep["candidate"].eq(chosen) & sweep["unit_split"].eq("evaluation")
    ].copy()
    rows = []
    if PREDECLARED_CASE in sweep["ecephys_unit_id"].values:
        predeclared = sweep.loc[
            sweep["candidate"].eq(chosen)
            & sweep["ecephys_unit_id"].eq(PREDECLARED_CASE)
        ].iloc[0]
        rows.append((predeclared, "predeclared compact V1", "selected before gaze results"))
    largest = local.loc[local["test_deviance_improvement"].idxmax()]
    worst = local.loc[local["test_deviance_improvement"].idxmin()]
    median = local["test_deviance_improvement"].median()
    typical = local.loc[(local["test_deviance_improvement"] - median).abs().idxmin()]
    sharper = local.loc[
        local["test_deviance_improvement"].gt(0)
        & np.isfinite(local["log2_area_ratio"])
    ]
    if len(sharper):
        sharper = sharper.loc[sharper["log2_area_ratio"].idxmin()]
        rows.append((sharper, "largest sharpening with prediction gain",
                     "minimum log2 area ratio among evaluation units with positive test gain"))
    rows.extend(
        [
            (largest, "largest held-out improvement", "maximum test deviance improvement"),
            (typical, "typical held-out change", "closest to median test deviance improvement"),
            (worst, "largest held-out worsening", "minimum test deviance improvement"),
        ]
    )
    output = []
    seen = set()
    for record, role, criterion in rows:
        unit_id = int(record.ecephys_unit_id)
        if unit_id in seen:
            continue
        seen.add(unit_id)
        output.append(
            {
                "ecephys_unit_id": unit_id,
                "selection_role": role,
                "selection_criterion": criterion,
                "group": record.group,
                "ecephys_structure_acronym": record.ecephys_structure_acronym,
                "unit_split": record.unit_split,
                "test_deviance_improvement": record.test_deviance_improvement,
                "log2_area_ratio": record.log2_area_ratio,
                "log2_amplitude_ratio": record.log2_amplitude_ratio,
            }
        )
    return pd.DataFrame(output)


def kernel_map(counts, x, y, grid, bandwidth=4.0):
    x_mesh, y_mesh = np.meshgrid(grid, grid)
    output = np.full(x_mesh.shape, np.nan)
    effective = np.zeros(x_mesh.shape)
    for row in range(len(grid)):
        for column in range(len(grid)):
            weight = np.exp(
                -0.5 * (((x - x_mesh[row, column]) / bandwidth) ** 2
                        + ((y - y_mesh[row, column]) / bandwidth) ** 2)
            )
            if weight.sum() > 0:
                effective[row, column] = weight.sum() ** 2 / np.square(weight).sum()
                if effective[row, column] >= 3:
                    output[row, column] = np.average(counts, weights=weight)
    return output, effective


def render_population_figure(summary, evaluation, sweep, chosen, path, session_id=SESSION_ID):
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    grid = summary.loc[~summary["control"]].pivot(
        index="gain_y", columns="gain_x", values="median_test_deviance_improvement"
    ).sort_index().sort_index(axis=1)
    limit = max(float(np.nanmax(np.abs(grid.to_numpy()))), 1e-6)
    image = axes[0, 0].imshow(
        grid.to_numpy(), origin="lower", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="equal",
    )
    axes[0, 0].set(
        xticks=np.arange(len(grid.columns)), xticklabels=[f"{x:g}" for x in grid.columns],
        yticks=np.arange(len(grid.index)), yticklabels=[f"{y:g}" for y in grid.index],
        xlabel="Horizontal gaze gain", ylabel="Vertical gaze gain",
        title="Calibration-unit held-out improvement",
    )
    chosen_row = summary.loc[summary["candidate"].eq(chosen)].iloc[0]
    chosen_x = list(grid.columns).index(chosen_row.gain_x)
    chosen_y = list(grid.index).index(chosen_row.gain_y)
    axes[0, 0].scatter(chosen_x, chosen_y, marker="*", s=180, color="#222222")
    figure.colorbar(image, ax=axes[0, 0], label="Median test Poisson-deviance improvement")

    selected = sweep.loc[
        sweep["candidate"].eq(chosen) & sweep["unit_split"].eq("evaluation")
    ]
    colors = {"V1": "#39738c", "HVA": "#d97736"}
    for group in ("V1", "HVA"):
        local = selected.loc[selected["group"].eq(group)]
        axes[0, 1].hist(
            local["test_deviance_improvement"], bins=28, histtype="step",
            linewidth=2, color=colors[group], label=f"{group} (n={len(local)})",
        )
        axes[1, 0].scatter(
            local["log2_area_ratio"], local["log2_amplitude_ratio"],
            s=19, alpha=0.55, color=colors[group], label=group,
        )
    axes[0, 1].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set(
        xlabel="Test Poisson-deviance improvement", ylabel="Evaluation units",
        title="Held-out-neuron predictive change",
    )
    axes[0, 1].legend(frameon=False)
    axes[1, 0].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set(
        xlabel="log₂ gaze-corrected / nominal half-max area",
        ylabel="log₂ gaze-corrected / nominal amplitude",
        title="RF sharpness and magnitude on evaluation units",
    )
    axes[1, 0].legend(frameon=False)

    order = ["V1", "HVA", "all"]
    local = evaluation.set_index("group").loc[order]
    axes[1, 1].bar(
        np.arange(len(order)), local["fraction_test_deviance_improved"],
        color=[colors["V1"], colors["HVA"], "#777777"], width=0.65,
    )
    axes[1, 1].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1, 1].set(
        xticks=np.arange(len(order)), xticklabels=order, ylim=(0, 1),
        ylabel="Fraction of evaluation units", title="Units with improved held-out prediction",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    figure.suptitle(
        f"Allen session {session_id}: population-calibrated gaze correction\n"
        f"chosen transform {chosen}; corrected baseline + bounded RF model",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def render_case_figure(cases, population, trials, counts, coordinate_lookup, chosen, sweep, path,
                       session_id=SESSION_ID):
    valid = trials["valid_gaze"].to_numpy(bool)
    grid = np.linspace(-50, 50, 51)
    figure, axes = plt.subplots(len(cases), 3, figsize=(13.2, 3.5 * len(cases)), squeeze=False)
    x_nominal, y_nominal = coordinate_lookup["gain_x_0_gain_y_0"]
    x_corrected, y_corrected = coordinate_lookup[chosen]
    for row, case in enumerate(cases.itertuples(index=False)):
        unit_index = population.index[population["ecephys_unit_id"].eq(case.ecephys_unit_id)][0]
        unit_counts = counts[unit_index].astype(float)[valid]
        nominal_map, nominal_effective = kernel_map(
            unit_counts, x_nominal[valid], y_nominal[valid], grid
        )
        corrected_map, corrected_effective = kernel_map(
            unit_counts, x_corrected[valid], y_corrected[valid], grid
        )
        joint = np.r_[nominal_map[np.isfinite(nominal_map)], corrected_map[np.isfinite(corrected_map)]]
        vmin, vmax = np.quantile(joint, [0.02, 0.98])
        for column, (matrix, title) in enumerate(
            ((nominal_map, "Nominal coordinates"), (corrected_map, "Gaze-corrected coordinates"))
        ):
            image = axes[row, column].imshow(
                matrix, origin="lower", extent=[-50, 50, -50, 50], cmap="viridis",
                vmin=vmin, vmax=vmax, aspect="equal",
            )
            axes[row, column].set_title(title)
            figure.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.03,
                            label="Mean spikes / presentation")
        difference = corrected_map - nominal_map
        finite = difference[np.isfinite(difference)]
        limit = max(float(np.quantile(np.abs(finite), 0.98)), 1e-6)
        image = axes[row, 2].imshow(
            difference, origin="lower", extent=[-50, 50, -50, 50], cmap="coolwarm",
            vmin=-limit, vmax=limit, aspect="equal",
        )
        figure.colorbar(image, ax=axes[row, 2], fraction=0.045, pad=0.03,
                        label="Corrected − nominal spikes")
        axes[row, 2].set_title("Paired map difference")
        axes[row, 0].set_ylabel(
            f"{case.selection_role}\nunit {case.ecephys_unit_id} ({case.group})\n"
            f"test Δdev={case.test_deviance_improvement:+.4f}"
        )
        for axis in axes[row]:
            axis.set(xlabel="Gabor x position (deg)", ylabel=axis.get_ylabel() or "Gabor y position (deg)")
            axis.grid(alpha=0.12)
    figure.suptitle(
        f"Concrete gaze-correction RF maps · session {session_id} · 4° display kernel",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    output = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / f"session_{args.session_id}"
    ).resolve()
    if output.exists() and any(output.iterdir()) and not (args.overwrite or args.reuse_extracted):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse_extracted:
        population = pd.read_csv(output / "visual_unit_population.csv", low_memory=False)
        trials = pd.read_csv(output / "gabor_trial_gaze_table.csv", low_memory=False)
        spike_file = np.load(output / "gabor_spike_counts.npz")
        counts = spike_file["counts"]
        gaze_metadata = json.loads((output / "extraction_summary.json").read_text())["gaze"]
    else:
        population = prepare_population(args.unit_table.resolve(), args.session_id)
        session = load_session(args.nwb.resolve())
        trials, gaze_metadata = prepare_trials(
            session, args.session_id, include_gaze=not args.skip_gaze
        )
        counts = spike_count_matrix(session, population, trials)
        population.to_csv(output / "visual_unit_population.csv", index=False, float_format="%.8g")
        trials.to_csv(output / "gabor_trial_gaze_table.csv", index=False, float_format="%.8g")
        np.savez_compressed(
            output / "gabor_spike_counts.npz",
            unit_ids=population["ecephys_unit_id"].to_numpy(int), counts=counts,
        )
        extraction_summary = {
            "session_id": args.session_id,
            "visual_units": len(population),
            "published_like_qc_units": int(population["published_like_qc"].sum()),
            "gabor_presentations": len(trials),
            "valid_gaze_presentations": int(trials["valid_gaze"].sum()),
            "valid_gaze_fraction": float(trials["valid_gaze"].mean()),
            "gaze": gaze_metadata,
        }
        (output / "extraction_summary.json").write_text(
            json.dumps(extraction_summary, indent=2) + "\n", encoding="utf-8"
        )
        del session
    if args.extract_only:
        print(f"Wrote compact extraction cache to {output}")
        return
    candidates, gaze_permutation = candidate_table(trials)
    sweep, coordinate_lookup, train, test = run_candidate_sweep(
        population, trials, counts, candidates, gaze_permutation
    )
    sweep.to_csv(output / "gaze_gain_unit_fit_sweep.csv", index=False, float_format="%.8g")
    candidate_summary = summarize_candidates(sweep)
    candidate_summary.to_csv(output / "gaze_gain_calibration_summary.csv", index=False, float_format="%.8g")
    chosen = select_best_candidate(candidate_summary)
    evaluation = evaluation_summary(sweep, chosen)
    evaluation.to_csv(output / "heldout_neuron_evaluation.csv", index=False, float_format="%.8g")
    full_population = fit_full_population_chosen(
        population, trials, counts, sweep, coordinate_lookup, train, test, chosen
    )
    full_population.to_csv(
        output / "full_population_nominal_and_chosen_fits.csv", index=False, float_format="%.8g"
    )
    cases = select_cases(sweep, chosen)
    cases.to_csv(output / "selected_concrete_cases.csv", index=False, float_format="%.8g")
    population_figure = output / "Figure_population_gaze_calibration.png"
    case_figure = output / "Figure_concrete_gaze_rf_maps.png"
    render_population_figure(
        candidate_summary, evaluation, sweep, chosen, population_figure, args.session_id
    )
    render_case_figure(
        cases, population, trials, counts, coordinate_lookup, chosen, sweep, case_figure,
        args.session_id,
    )

    chosen_row = candidate_summary.loc[candidate_summary["candidate"].eq(chosen)].iloc[0]
    eval_all = evaluation.loc[evaluation["group"].eq("all")].iloc[0]
    summary = {
        "session_id": args.session_id,
        "visual_units": len(population),
        "published_like_qc_units": int(population["published_like_qc"].sum()),
        "calibration_units": int(population["unit_split"].eq("calibration").sum()),
        "evaluation_units": int(population["unit_split"].eq("evaluation").sum()),
        "gabor_presentations": len(trials),
        "valid_gaze_presentations": int(trials["valid_gaze"].sum()),
        "valid_gaze_fraction": float(trials["valid_gaze"].mean()),
        "training_presentations": int(train.sum()),
        "test_presentations": int(test.sum()),
        "gaze": {
            **gaze_metadata,
            "trial_median_dx_sd_deg": float(trials.loc[trials["valid_gaze"], "gaze_dx_deg"].std()),
            "trial_median_dy_sd_deg": float(trials.loc[trials["valid_gaze"], "gaze_dy_deg"].std()),
            "trace_centered_to_zero": True,
        },
        "chosen_candidate": chosen,
        "chosen_gain_x": float(chosen_row.gain_x),
        "chosen_gain_y": float(chosen_row.gain_y),
        "calibration_median_test_deviance_improvement": float(
            chosen_row.median_test_deviance_improvement
        ),
        "evaluation": {
            key: (
                int(value) if isinstance(value, (np.integer,))
                else float(value) if isinstance(value, (np.floating,))
                else value
            )
            for key, value in eval_all.to_dict().items()
        },
        "rf_model": {
            "response_window_s": RESPONSE_WINDOW_S,
            "baseline": "nonnegative",
            "orientation_amplitudes": 3,
            "center_extension_deg": CENTER_EXTENSION_DEG,
            "sigma_upper_deg": SIGMA_UPPER_DEG,
            "enclosing_border": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        f"# Allen session {args.session_id}: population gaze correction",
        "",
        f"The analysis used {len(population)} canonical visual-area units and {len(trials)} Gabor presentations. The gain sweep was restricted to {int(population['published_like_qc'].sum())} RF/QC units; the chosen transform and nominal control were then fit for the full population.",
        "",
        "The gaze trace is the per-presentation median filtered spherical screen-gaze coordinate, centered to zero over the Gabor block. Corrected stimulus coordinates equal nominal position minus the scaled gaze deviation. No per-neuron gaze transform is fitted.",
        "",
        f"The population-calibration subset selected `{chosen}`. On held-out neurons, the median test Poisson-deviance improvement was {eval_all.median_test_deviance_improvement:+.6f}, and {eval_all.fraction_test_deviance_improved:.1%} of units improved.",
        "",
        "RFs use a nonnegative baseline plus three orientation amplitudes and a shared axis-aligned Gaussian. Centers may extend 20 degrees beyond the sampled grid; sigma is limited to 40 degrees in V1 and 50 degrees in HVAs. Bound-reaching fits are labeled censored. No enclosing border is used.",
        "",
        "This is an exploratory single-session checkpoint. Candidate selection and final evaluation use disjoint neuron sets, and every fit uses separate training and test repeats within each position/orientation condition.",
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
        "parameters": {
            "gain_grid": GAINS,
            "response_window_s": RESPONSE_WINDOW_S,
            "minimum_gaze_samples_per_presentation": MIN_GAZE_SAMPLES,
            "random_seed": RANDOM_SEED,
        },
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote population gaze-correction checkpoint to {output}")


if __name__ == "__main__":
    main()
