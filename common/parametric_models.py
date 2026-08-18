"""Parametric trial-count models for MouseV2 grating tuning and receptive fields."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy.stats import t as student_t


def poisson_deviance_residual(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Signed square-root Poisson deviance residuals."""
    y = np.asarray(observed, dtype=float)
    mu = np.maximum(np.asarray(expected, dtype=float), 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
    return np.sign(y - mu) * np.sqrt(np.maximum(2.0 * term, 0.0))


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    residual = poisson_deviance_residual(observed, expected)
    return float(np.sum(residual**2))


def grating_rate(
    parameters: np.ndarray,
    log2_sf: np.ndarray,
    log2_tf: np.ndarray,
    orientation_rad: np.ndarray,
) -> np.ndarray:
    """Separable log-Gaussian SF/TF and orientation-periodic von Mises model."""
    baseline, amplitude, sf_mu, sf_sigma, tf_mu, tf_sigma, ori_mu, kappa = parameters
    sf_term = np.exp(-0.5 * ((log2_sf - sf_mu) / sf_sigma) ** 2)
    tf_term = np.exp(-0.5 * ((log2_tf - tf_mu) / tf_sigma) ** 2)
    ori_term = np.exp(kappa * (np.cos(2.0 * (orientation_rad - ori_mu)) - 1.0))
    return baseline + amplitude * sf_term * tf_term * ori_term


def elliptical_gaussian_rate(
    parameters: np.ndarray, x_deg: np.ndarray, y_deg: np.ndarray
) -> np.ndarray:
    """Baseline plus a rotated elliptical 2D Gaussian."""
    baseline, amplitude, x0, y0, sigma_major, sigma_minor, theta = parameters
    cosine = np.cos(theta)
    sine = np.sin(theta)
    dx = x_deg - x0
    dy = y_deg - y0
    major = cosine * dx + sine * dy
    minor = -sine * dx + cosine * dy
    return baseline + amplitude * np.exp(
        -0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2)
    )


def _fit_poisson_model(
    observed: np.ndarray,
    exposures: np.ndarray,
    predictor,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_nfev: int,
) -> tuple[np.ndarray, float, bool, int]:
    def residual(parameters: np.ndarray) -> np.ndarray:
        expected = exposures * np.maximum(predictor(parameters), 1e-10)
        return poisson_deviance_residual(observed, expected)

    try:
        fit = least_squares(
            residual,
            x0=np.clip(initial, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            max_nfev=max_nfev,
            method="trf",
        )
        deviance = poisson_deviance(
            observed, exposures * np.maximum(predictor(fit.x), 1e-10)
        )
        return fit.x, deviance, bool(fit.success and np.all(np.isfinite(fit.x))), int(fit.nfev)
    except (ValueError, FloatingPointError, RuntimeError):
        return np.full_like(initial, np.nan), np.nan, False, 0


def fit_grating_cell_model(
    sf_cpd: np.ndarray,
    tf_hz: np.ndarray,
    orientation_deg: np.ndarray,
    spike_totals: np.ndarray,
    trials: np.ndarray,
    *,
    max_nfev: int = 800,
) -> dict[str, float | int | bool]:
    sf = np.asarray(sf_cpd, dtype=float)
    tf = np.asarray(tf_hz, dtype=float)
    ori = np.deg2rad(np.asarray(orientation_deg, dtype=float))
    totals = np.asarray(spike_totals, dtype=float)
    exposures = np.asarray(trials, dtype=float)
    rates = totals / exposures
    log_sf = np.log2(sf)
    log_tf = np.log2(tf)
    best = int(np.argmax(rates))
    low_rate = max(float(np.quantile(rates, 0.1)), 0.0)
    amplitude = max(float(rates[best] - low_rate), 1e-3)
    maximum = max(float(np.max(rates)), 1.0)
    initial = np.array(
        [low_rate, amplitude, log_sf[best], 0.8, log_tf[best], 1.0, ori[best] % np.pi, 1.0]
    )
    # Permit a one-octave extrapolation beyond the sampled SF/TF range. This
    # lets a consistently rising or falling flank locate an off-grid optimum,
    # while finite outer bounds still expose unconstrained fits.
    lower = np.array(
        [0.0, 0.0, np.min(log_sf) - 1.0, 0.15, np.min(log_tf) - 1.0, 0.15, 0.0, 0.0]
    )
    upper = np.array(
        [maximum * 3 + 1, maximum * 6 + 1, np.max(log_sf) + 1.0, 4.0, np.max(log_tf) + 1.0, 5.0, np.pi, 20.0]
    )
    predictor = lambda p: grating_rate(p, log_sf, log_tf, ori)
    parameters, deviance, success, evaluations = _fit_poisson_model(
        totals, exposures, predictor, initial, lower, upper, max_nfev=max_nfev
    )
    null_rate = float(np.sum(totals) / np.sum(exposures))
    null_deviance = poisson_deviance(totals, exposures * null_rate)
    improvement = max(null_deviance - deviance, 0.0) if np.isfinite(deviance) else np.nan
    pseudo_r2 = 1.0 - deviance / null_deviance if null_deviance > 0 and np.isfinite(deviance) else np.nan
    p_value = float(chi2.sf(improvement, 7)) if np.isfinite(improvement) else np.nan
    names = (
        "baseline_spikes",
        "amplitude_spikes",
        "sf_mu_log2",
        "sf_sigma_octaves",
        "tf_mu_log2",
        "tf_sigma_octaves",
        "ori_pref_rad",
        "ori_kappa",
    )
    output = dict(zip(names, map(float, parameters)))
    output.update(
        {
            "parametric_pref_sf_cpd": float(np.exp2(parameters[2])),
            "parametric_pref_tf_hz": float(np.exp2(parameters[4])),
            "parametric_pref_ori_deg": float(np.rad2deg(parameters[6]) % 180.0),
            "parametric_deviance": deviance,
            "parametric_null_deviance": null_deviance,
            "parametric_pseudo_r2": pseudo_r2,
            "parametric_lrt_p": p_value,
            "parametric_fit_success": success,
            "parametric_fit_evaluations": evaluations,
            "parametric_sf_in_tested_range": bool(
                success and np.min(log_sf) <= parameters[2] <= np.max(log_sf)
            ),
            "parametric_tf_in_tested_range": bool(
                success and np.min(log_tf) <= parameters[4] <= np.max(log_tf)
            ),
            "parametric_sf_at_extrapolation_bound": bool(
                not success or parameters[2] <= lower[2] + 0.02 or parameters[2] >= upper[2] - 0.02
            ),
            "parametric_tf_at_extrapolation_bound": bool(
                not success or parameters[4] <= lower[4] + 0.02 or parameters[4] >= upper[4] - 0.02
            ),
        }
    )
    return output


def fit_rf_cell_model(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    spike_totals: np.ndarray,
    trials: np.ndarray,
    *,
    max_nfev: int = 800,
) -> dict[str, float | int | bool]:
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(y_deg, dtype=float)
    totals = np.asarray(spike_totals, dtype=float)
    exposures = np.asarray(trials, dtype=float)
    rates = totals / exposures
    best = int(np.argmax(rates))
    low_rate = max(float(np.quantile(rates, 0.1)), 0.0)
    amplitude = max(float(rates[best] - low_rate), 1e-3)
    maximum = max(float(np.max(rates)), 1.0)
    initial = np.array([low_rate, amplitude, x[best], y[best], 18.0, 12.0, 0.0])
    lower = np.array([0.0, 0.0, np.min(x) - 20.0, np.min(y) - 20.0, 3.0, 3.0, -np.pi / 2])
    upper = np.array([maximum * 3 + 1, maximum * 6 + 1, np.max(x) + 20.0, np.max(y) + 20.0, 80.0, 80.0, np.pi / 2])
    predictor = lambda p: elliptical_gaussian_rate(p, x, y)
    parameters, deviance, success, evaluations = _fit_poisson_model(
        totals, exposures, predictor, initial, lower, upper, max_nfev=max_nfev
    )
    # Canonicalize major/minor axes so angle summaries are comparable.
    if success and parameters[4] < parameters[5]:
        parameters[4], parameters[5] = parameters[5], parameters[4]
        parameters[6] += np.pi / 2
    if success:
        parameters[6] = (parameters[6] + np.pi / 2) % np.pi - np.pi / 2
    null_rate = float(np.sum(totals) / np.sum(exposures))
    null_deviance = poisson_deviance(totals, exposures * null_rate)
    improvement = max(null_deviance - deviance, 0.0) if np.isfinite(deviance) else np.nan
    pseudo_r2 = 1.0 - deviance / null_deviance if null_deviance > 0 and np.isfinite(deviance) else np.nan
    p_value = float(chi2.sf(improvement, 6)) if np.isfinite(improvement) else np.nan
    return {
        "rf_baseline_spikes": float(parameters[0]),
        "rf_amplitude_spikes": float(parameters[1]),
        "rf_center_x_deg": float(parameters[2]),
        "rf_center_y_deg": float(parameters[3]),
        "rf_sigma_major_deg": float(parameters[4]),
        "rf_sigma_minor_deg": float(parameters[5]),
        "rf_theta_deg": float(np.rad2deg(parameters[6])),
        "rf_deviance": deviance,
        "rf_null_deviance": null_deviance,
        "rf_pseudo_r2": pseudo_r2,
        "rf_lrt_p": p_value,
        "rf_fit_success": success,
        "rf_fit_evaluations": evaluations,
        "rf_center_on_screen": bool(
            success
            and np.min(x) <= parameters[2] <= np.max(x)
            and np.min(y) <= parameters[3] <= np.max(y)
        ),
    }


def presentation_counts(
    unit_ids: Iterable[int],
    spikes_by_unit: dict[int, np.ndarray],
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Count spikes for every unit and presentation."""
    ids = [int(unit_id) for unit_id in unit_ids]
    output = np.zeros((len(ids), len(starts)), dtype=np.float32)
    for row, unit_id in enumerate(ids):
        spikes = np.asarray(spikes_by_unit.get(unit_id, []), dtype=float)
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            spikes = np.sort(spikes)
        output[row] = np.searchsorted(spikes, stops, side="left") - np.searchsorted(
            spikes, starts, side="left"
        )
    return output


def aggregate_presentations(
    responses: np.ndarray, codes: np.ndarray, cell_count: int
) -> tuple[np.ndarray, np.ndarray]:
    trials = np.bincount(codes, minlength=cell_count).astype(float)
    totals = np.stack(
        [np.sum(responses[:, codes == code], axis=1) for code in range(cell_count)], axis=1
    )
    return totals, trials


def _split_half_reliability(
    responses: np.ndarray, codes: np.ndarray, split: np.ndarray, cell_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    trial_counts = []
    for half in (0, 1):
        selected = split == half
        totals, trials = aggregate_presentations(
            responses[:, selected], codes[selected], cell_count
        )
        trial_counts.append(trials)
        with np.errstate(divide="ignore", invalid="ignore"):
            means.append(totals / trials[None, :])
    # A stimulus cell with zero presentations in either half divides by
    # zero above; excluding it here keeps that one cell's inf/NaN from
    # silently contaminating every unit's correlation across all cells.
    valid_cells = (trial_counts[0] > 0) & (trial_counts[1] > 0)
    means0 = means[0][:, valid_cells]
    means1 = means[1][:, valid_cells]
    x = means0 - np.mean(means0, axis=1, keepdims=True)
    y = means1 - np.mean(means1, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.sum(x * y, axis=1) / denominator
    correlation = np.clip(correlation, -1.0, 1.0)
    df = int(valid_cells.sum()) - 2
    statistic = np.full(len(correlation), np.nan)
    interior = np.isfinite(correlation) & (np.abs(correlation) < 1)
    statistic[interior] = correlation[interior] * np.sqrt(
        df / (1.0 - correlation[interior] ** 2)
    )
    statistic[correlation == 1] = np.inf
    statistic[correlation == -1] = -np.inf
    p_value = student_t.sf(statistic, df)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = 2.0 * correlation / (1.0 + correlation)
    return correlation, corrected, p_value


def _numeric_design(
    presentations: pd.DataFrame, columns: tuple[str, ...]
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    table = presentations.copy()
    required = {"start_time", "stop_time", *columns}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Presentation table lacks columns {sorted(missing)}")
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    valid = table[list(required)].notna().all(axis=1)
    for column in columns:
        if column in {"spatial_frequency", "temporal_frequency"}:
            valid &= table[column] > 0
    table = table.loc[valid].sort_values("start_time", kind="mergesort").reset_index(drop=True)
    cells = table[list(columns)].drop_duplicates().sort_values(list(columns)).reset_index(drop=True)
    lookup = {tuple(row): index for index, row in cells.iterrows()}
    codes = np.asarray(
        [lookup[tuple(row)] for row in table[list(columns)].itertuples(index=False, name=None)],
        dtype=int,
    )
    return table, codes, cells


def fit_parametric_grating_models(
    unit_ids: Iterable[int],
    spikes_by_unit: dict[int, np.ndarray],
    presentations: pd.DataFrame,
    *,
    max_nfev: int = 800,
) -> pd.DataFrame:
    """Fit the separable SF/TF/orientation model to every unit."""
    ids = np.asarray([int(unit_id) for unit_id in unit_ids], dtype=int)
    table, codes, cells = _numeric_design(
        presentations, ("spatial_frequency", "temporal_frequency", "orientation")
    )
    responses = presentation_counts(
        ids,
        spikes_by_unit,
        table["start_time"].to_numpy(),
        table["stop_time"].to_numpy(),
    )
    totals, trials = aggregate_presentations(responses, codes, len(cells))
    rows = []
    for row, unit_id in enumerate(ids):
        result = fit_grating_cell_model(
            cells["spatial_frequency"].to_numpy(),
            cells["temporal_frequency"].to_numpy(),
            cells["orientation"].to_numpy(),
            totals[row],
            trials,
            max_nfev=max_nfev,
        )
        result["unit_id"] = int(unit_id)
        rows.append(result)
    return pd.DataFrame(rows)


def fit_parametric_rf_models(
    unit_ids: Iterable[int],
    spikes_by_unit: dict[int, np.ndarray],
    presentations: pd.DataFrame,
    *,
    max_nfev: int = 800,
) -> pd.DataFrame:
    """Fit elliptical Gaussian RFs and position-wise split-half reliability."""
    ids = np.asarray([int(unit_id) for unit_id in unit_ids], dtype=int)
    table, codes, cells = _numeric_design(presentations, ("x_position", "y_position"))
    responses = presentation_counts(
        ids,
        spikes_by_unit,
        table["start_time"].to_numpy(),
        table["stop_time"].to_numpy(),
    )
    totals, trials = aggregate_presentations(responses, codes, len(cells))
    split = np.zeros(len(table), dtype=int)
    for cell, indices in table.groupby(["x_position", "y_position"], sort=True).groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        split[ordered] = np.arange(len(ordered)) % 2
    reliability_r, reliability_sb, reliability_p = _split_half_reliability(
        responses, codes, split, len(cells)
    )
    rows = []
    for row, unit_id in enumerate(ids):
        result = fit_rf_cell_model(
            cells["x_position"].to_numpy(),
            cells["y_position"].to_numpy(),
            totals[row],
            trials,
            max_nfev=max_nfev,
        )
        result.update(
            {
                "unit_id": int(unit_id),
                "rf_split_half_r": reliability_r[row],
                "rf_split_half_spearman_brown": reliability_sb[row],
                "rf_reliability_p": reliability_p[row],
                "rf_presentations": len(table),
                "rf_position_cells": len(cells),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)
