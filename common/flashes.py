"""Paper-compatible flash latency and response-timescale metrics.

The released analysis pools all flash presentations. MouseV2 encodes polarity
as contrast, whereas Allen Visual Coding encodes it as color with fixed 0.8
contrast. Both contain equal numbers of bright (+1) and dark (-1) trials, so
this module computes declared polarity sensitivities without changing the
pooled definition.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit


FLASH_VARIANTS = ("pooled", "bright", "dark")
TTFS_START_MS = 30
TTFS_END_MS = 200
TIMESCALE_BIN_EDGES_S = np.arange(0.0, 2.01, 0.01)
TIMESCALE_WINDOW_S = (0.04, 0.29)


def prepare_flash_presentations(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and label a MouseV2 or Allen bright/dark flash table."""
    required = {"start_time", "stop_time"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Flash table is missing columns {missing}")

    result = table.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(required)].isna().any().any():
        raise ValueError("Flash table contains nonnumeric timing or contrast values")
    # Polarity (contrast for MouseV2, color for Allen) is coerced and
    # validated separately below, not here.
    polarity_column = None
    for candidate in ("contrast", "color"):
        if candidate not in result:
            continue
        values = pd.to_numeric(result[candidate], errors="coerce")
        if values.notna().all() and set(values.unique()) == {-1.0, 1.0}:
            result[candidate] = values
            polarity_column = candidate
            break
    if polarity_column is None:
        raise ValueError(
            "Expected flash polarity -1/+1 in contrast (MouseV2) or color (Allen)"
        )
    if not (result["stop_time"] > result["start_time"]).all():
        raise ValueError("Flash stop times must be after start times")
    result["flash_polarity"] = np.where(
        result[polarity_column] > 0, "bright", "dark"
    )
    result["flash_polarity_source"] = polarity_column
    return result.reset_index(drop=True)


def presentation_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return pooled and polarity masks in stable presentation order."""
    prepared = prepare_flash_presentations(table)
    return {
        "pooled": np.ones(len(prepared), dtype=bool),
        "bright": prepared["flash_polarity"].eq("bright").to_numpy(),
        "dark": prepared["flash_polarity"].eq("dark").to_numpy(),
    }


def first_spike_latency_seconds(
    spikes_s: np.ndarray,
    trial_starts_s: np.ndarray,
    *,
    start_ms: int = TTFS_START_MS,
    end_ms: int = TTFS_END_MS,
) -> tuple[float, int]:
    """Median first occupied 1-ms bin, matching the released TTFS method.

    Trials without a spike in ``[start_ms, end_ms)`` are omitted from the
    median. Returned latency is relative to the NWB interval ``start_time``.
    """
    starts = np.asarray(trial_starts_s, dtype=float)
    spikes = np.asarray(spikes_s, dtype=float)
    if starts.size == 0 or spikes.size == 0:
        return np.nan, 0
    if spikes.size > 1 and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)

    window_starts = starts + start_ms / 1000.0
    window_stops = starts + end_ms / 1000.0
    first_indices = np.searchsorted(spikes, window_starts, side="left")
    stop_indices = np.searchsorted(spikes, window_stops, side="left")
    valid = first_indices < stop_indices
    if not np.any(valid):
        return np.nan, 0

    first_spikes = spikes[first_indices[valid]]
    # Match the legacy binary raster: positive relative times are truncated to
    # their occupied millisecond bin before taking the median across trials.
    first_bins_ms = ((first_spikes - starts[valid]) * 1000.0).astype(np.int64)
    return float(np.median(first_bins_ms) / 1000.0), int(valid.sum())


def bin_trial_spike_counts(
    spikes_s: np.ndarray,
    trial_starts_s: np.ndarray,
    bin_edges_s: np.ndarray = TIMESCALE_BIN_EDGES_S,
) -> np.ndarray:
    """Count spikes in trial-aligned bins using left-closed edges."""
    starts = np.asarray(trial_starts_s, dtype=float)
    edges = np.asarray(bin_edges_s, dtype=float)
    if starts.size == 0:
        return np.zeros((0, max(len(edges) - 1, 0)), dtype=float)
    spikes = np.asarray(spikes_s, dtype=float)
    if spikes.size > 1 and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    if spikes.size == 0:
        return np.zeros((len(starts), len(edges) - 1), dtype=float)
    absolute_edges = starts[:, None] + edges[None, :]
    indices = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(indices, axis=1).astype(float, copy=False)


def timescale_bin_mask(
    bin_edges_s: np.ndarray = TIMESCALE_BIN_EDGES_S,
    window_s: tuple[float, float] = TIMESCALE_WINDOW_S,
) -> np.ndarray:
    """Select bins by AllenSDK center coordinates, not their left edges."""
    edges = np.asarray(bin_edges_s, dtype=float)
    centers = edges[:-1] + np.diff(edges) / 2.0
    return (centers >= window_s[0]) & (centers <= window_s[1])


def autocorrelation_curve(spikes_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the released 2-D correlation reduction and time axis."""
    values = np.asarray(spikes_window, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 4:
        raise ValueError("Timescale input needs at least 3 trials and 4 time bins")
    autocorrelation = signal.correlate(values, values, mode="same")
    autocorrelation = np.delete(
        autocorrelation, [autocorrelation.shape[0] // 2], axis=0
    )
    curve = np.mean(autocorrelation, axis=0)
    curve = curve[values.shape[1] // 2 :]
    time_ms = np.linspace(0, values.shape[1] / 2 * 10, len(curve))
    return time_ms, curve


def fit_response_timescale(
    trial_counts: np.ndarray,
    *,
    bin_edges_s: np.ndarray = TIMESCALE_BIN_EDGES_S,
    window_s: tuple[float, float] = TIMESCALE_WINDOW_S,
) -> tuple[float, float, float, bool]:
    """Fit the released bounded exponential to one flash population."""
    mask = timescale_bin_mask(bin_edges_s, window_s)
    selected = np.asarray(trial_counts, dtype=float)[:, mask]
    spike_count = float(np.sum(selected))
    try:
        time_ms, curve = autocorrelation_curve(selected)

        def exponential(t: np.ndarray, amplitude: float, tau: float, offset: float):
            return amplitude * np.exp(-t / tau) + offset

        parameters, covariance = curve_fit(
            exponential,
            time_ms,
            curve,
            p0=(5, 20, 0.1),
            method="trf",
            bounds=([0, 1, -np.inf], [np.inf, 1000, np.inf]),
            maxfev=1_000_000_000,
        )
        errors = np.sqrt(np.diag(covariance))
        return float(parameters[1]), float(errors[1]), spike_count, True
    except Exception:
        return np.nan, np.nan, spike_count, False


def compute_flash_metrics(
    spikes_by_unit: Mapping[int, np.ndarray],
    flash_presentations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pooled, bright, and dark per-unit TTFS and timescale metrics."""
    flashes = prepare_flash_presentations(flash_presentations)
    starts = flashes["start_time"].to_numpy(dtype=float)
    masks = presentation_masks(flashes)
    rows: list[dict[str, object]] = []

    for unit_id, spikes in spikes_by_unit.items():
        spikes = np.asarray(spikes, dtype=float)
        all_counts = bin_trial_spike_counts(spikes, starts)
        row: dict[str, object] = {"unit_id": int(unit_id)}
        for variant in FLASH_VARIANTS:
            selected = masks[variant]
            latency, valid_trials = first_spike_latency_seconds(
                spikes, starts[selected]
            )
            tau, error, spike_count, fit_ok = fit_response_timescale(
                all_counts[selected]
            )
            row[f"time_to_first_spike_{variant}"] = latency
            row[f"ttfs_valid_trials_{variant}"] = valid_trials
            row[f"autocorr_tau_{variant}"] = tau
            row[f"err_ac_{variant}"] = error
            row[f"spike_count_ac_{variant}"] = spike_count
            row[f"timescale_fit_ok_{variant}"] = fit_ok
            row[f"flash_trials_{variant}"] = int(selected.sum())
        rows.append(row)
    return pd.DataFrame(rows)
