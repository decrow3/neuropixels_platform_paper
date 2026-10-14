"""Preferred-polarity TTFS, response selection, and matched-trial timescales."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

TIMESCALE_BIN_EDGES_S = np.arange(0.0, 2.01, 0.01)
TIMESCALE_WINDOW_S = (0.04, 0.29)
WINDOW_EDGES_S = np.arange(4, 30, dtype=float) / 100.0
TRIALS_PER_POLARITY = 75
EDGES_S = np.arange(-0.100, 0.205, 0.005)
CENTERS_MS = (EDGES_S[:-1] + np.diff(EDGES_S) / 2) * 1000
SELECTION_WINDOW = (CENTERS_MS >= 30) & (CENTERS_MS < 180)
BASELINE_WINDOW = (CENTERS_MS >= -100) & (CENTERS_MS < 0)

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

def is_valid_timescale(table: pd.DataFrame) -> pd.Series:
    tau = pd.to_numeric(table["timescale_ms"], errors="coerce")
    error = pd.to_numeric(table["fit_error_ms"], errors="coerce")
    spikes = pd.to_numeric(table["spike_count"], errors="coerce")
    return tau.between(1, 300) & spikes.gt(50) & error.lt(20)


def balanced_subsample_masks(
    flashes: pd.DataFrame,
    *,
    n_subsamples: int,
    seed: int,
) -> list[np.ndarray]:
    bright = np.flatnonzero(flashes["flash_polarity"].eq("bright").to_numpy())
    dark = np.flatnonzero(flashes["flash_polarity"].eq("dark").to_numpy())
    if len(bright) != 150 or len(dark) != 150:
        raise ValueError(
            f"Expected 150 MouseV2 flashes per polarity; got {len(bright)} and {len(dark)}"
        )
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_subsamples):
        selected = np.concatenate(
            [
                rng.choice(bright, TRIALS_PER_POLARITY, replace=False),
                rng.choice(dark, TRIALS_PER_POLARITY, replace=False),
            ]
        )
        mask = np.zeros(len(flashes), dtype=bool)
        mask[selected] = True
        masks.append(mask)
    return masks

def first_spike_ms(spikes: np.ndarray, starts: np.ndarray) -> float:
    """Median first occupied 1-ms bin in [30, 200] ms; empty trials excluded."""
    if len(starts) == 0:
        return np.nan
    lo = np.searchsorted(spikes, starts + 0.030, side="left")
    hi = np.searchsorted(spikes, starts + 0.201, side="left")
    valid = lo < hi
    if not valid.any():
        return np.nan
    # Match a binned implementation rather than returning sub-ms spike times.
    bins = np.floor((spikes[lo[valid]] - starts[valid]) * 1000 + 1e-9)
    return float(np.median(bins))


def preferred_polarity(spikes: np.ndarray, starts: np.ndarray, polarity: np.ndarray) -> float:
    """Historical Flashes preference: largest mean spike count over its 250-ms trial."""
    means = []
    for value in (1.0, -1.0):
        selected = starts[polarity == value]
        left = np.searchsorted(spikes, selected, side="left")
        right = np.searchsorted(spikes, selected + 0.250, side="left")
        means.append(float(np.mean(right - left)))
    return (1.0, -1.0)[int(np.argmax(means))]

def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.minimum(q_ranked, 1.0)
    return q

def counts(spikes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    absolute_edges = starts[:, None] + EDGES_S[None, :]
    indices = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(indices, axis=1).astype(float)


def response_selected_ttfs(spikes_by_id, units, flashes, *, area_adjustment=False):
    """Apply the canonical odd-trial amplitude selection and all-trial latency.

    FDR families are session/polarity for MouseV2 and session/area/polarity for
    Allen. Latency and preferred polarity use all trials, as in the locked
    analysis; only amplitude selection uses alternating trials.
    """
    starts = flashes.start_time.to_numpy(float)
    polarity = np.where(flashes.flash_polarity.eq('bright'), 1.0, -1.0)
    rows = []
    for unit in units.itertuples():
        spikes = spikes_by_id[int(unit.source_unit_id)]
        preferred = preferred_polarity(spikes, starts, polarity)
        for value in (1.0, -1.0):
            indices = np.flatnonzero(polarity == value)[::2]
            rates = counts(spikes, starts[indices]) / np.diff(EDGES_S)
            baseline = rates[:, BASELINE_WINDOW].mean(axis=1)
            response = rates[:, SELECTION_WINDOW].mean(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    pvalue = float(wilcoxon(response, baseline, alternative='greater').pvalue)
                except ValueError:
                    pvalue = 1.0
            rows.append({'unit_id': int(unit.unit_id), 'area_coarse': unit.area_coarse,
                         'polarity': value, 'preferred': preferred,
                         'effect': float(np.mean(response - baseline)), 'pvalue': pvalue,
                         'preferred_0_250_ttfs_ms': first_spike_ms(spikes, starts[polarity == preferred])})
    result = pd.DataFrame(rows)
    result['qvalue'] = np.nan
    families = ['area_coarse', 'polarity'] if area_adjustment else ['polarity']
    for indices in result.groupby(families).groups.values():
        result.loc[indices, 'qvalue'] = bh_adjust(result.loc[indices, 'pvalue'].to_numpy())
    result['selected'] = result.effect.ge(2) & result.qvalue.lt(.01)
    return result.loc[result.polarity.eq(result.preferred)].copy()


def matched_timescales(spikes_by_id, units, flashes, site_number):
    """Ten fixed, balanced 150-flash draws per common-QC MouseV2 unit."""
    masks = balanced_subsample_masks(flashes, n_subsamples=10, seed=20260805 + site_number)
    rows = []
    for unit in units.itertuples():
        counts = bin_trial_spike_counts(spikes_by_id[int(unit.source_unit_id)],
                                       flashes.start_time.to_numpy(float), WINDOW_EDGES_S)
        for draw, mask in enumerate(masks):
            selected = counts[mask]
            spike_count = float(selected.sum())
            tau, error = np.nan, np.nan
            if spike_count > 50:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tau, error, _, _ = fit_response_timescale(selected, bin_edges_s=WINDOW_EDGES_S)
            rows.append({'view': 'mouse_matched_150', 'session_id': site_number,
                         'unit_id': int(unit.unit_id), 'subsample': draw,
                         'timescale_ms': tau, 'fit_error_ms': error, 'spike_count': spike_count})
    result = pd.DataFrame(rows)
    result['valid_timescale'] = is_valid_timescale(result)
    return result[['view', 'session_id', 'unit_id', 'subsample', 'timescale_ms', 'valid_timescale']]


def reproduce_unit(spikes: np.ndarray, starts: np.ndarray) -> dict[str, object]:
    counts = bin_trial_spike_counts(spikes, starts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau, error, spike_count, fit_ok = fit_response_timescale(counts)
    return {
        "reproduced_timescale_ac": tau,
        "reproduced_err_ac": error,
        "reproduced_spike_count_ac": spike_count,
        "reproduced_fit_ok": fit_ok,
    }


def exclude_invalid_spikes(spikes: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Match AllenSDK's removal of spikes falling in declared invalid times."""
    result = np.asarray(spikes, dtype=float)
    if not len(intervals):
        return result
    keep = np.ones(len(result), dtype=bool)
    for start, stop in intervals:
        keep &= ~((result >= start) & (result < stop))
    return result[keep]
