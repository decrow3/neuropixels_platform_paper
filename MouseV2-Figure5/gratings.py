"""Canonical cycle-fold grating metric; extracted from the research implementation."""
from __future__ import annotations
from collections import OrderedDict
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import signal

DIMENSION_ALIASES = OrderedDict(
    [
        ("orientation", ("orientation", "ori", "ORI")),
        (
            "temporal_frequency",
            ("temporal_frequency", "temporal_frequency_hz", "tf", "TF"),
        ),
        (
            "spatial_frequency",
            ("spatial_frequency", "spatial_frequency_cpd", "sf", "SF"),
        ),
        ("contrast", ("contrast",)),
        ("phase", ("phase",)),
    ]
)


OUTPUT_COLUMNS = (
    "unit_id",
    "f1_f0_dg",
    "mod_idx_dg",
    "pref_ori_dg",
    "pref_tf_dg",
    "pref_sf_dg",
    "pref_contrast_dg",
    "pref_phase_dg",
    "preferred_mean_spikes_dg",
    "firing_rate_dg",
    "preferred_condition_ties_dg",
    "preferred_trials_dg",
    "analysis_duration_s_dg",
)


def _empty_metrics(unit_ids: Iterable[int]) -> pd.DataFrame:
    rows = []
    for unit_id in unit_ids:
        row = {column: np.nan for column in OUTPUT_COLUMNS}
        row["unit_id"] = int(unit_id)
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _find_column(table: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    return next((column for column in aliases if column in table.columns), None)


def _bin_trial_spike_counts(
    spikes_s: np.ndarray,
    trial_starts_s: np.ndarray,
    duration_ms: int,
) -> np.ndarray:
    """Count spikes in 1-ms bins, returning trials × time bins.

    This intentionally counts multiple spikes in a bin rather than converting
    the bin to binary; AllenSDK's ``presentationwise_spike_counts`` returns
    counts.
    """
    starts = np.asarray(trial_starts_s, dtype=float)
    if starts.size == 0:
        return np.zeros((0, duration_ms), dtype=np.float32)
    spikes = np.asarray(spikes_s, dtype=float)
    if spikes.size == 0:
        return np.zeros((len(starts), duration_ms), dtype=np.float32)
    if spikes.size > 1 and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    edges = starts[:, None] + np.arange(duration_ms + 1, dtype=float)[None, :] / 1000.0
    indices = np.searchsorted(spikes, edges, side="left")
    return np.diff(indices, axis=1).astype(np.float32, copy=False)


def f1_f0_from_trial_counts(
    trial_counts: np.ndarray,
    temporal_frequency_hz: float,
    trial_duration_s: float,
) -> float:
    """AllenSDK cycle-fold F1/F0 applied to trials × 1-ms spike counts."""
    arr = np.asarray(trial_counts)
    if arr.size == 0:
        return np.nan
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.size)

    cycles_per_trial = int(float(temporal_frequency_hz) * float(trial_duration_s))
    if cycles_per_trial < 1:
        return np.nan
    bins_per_cycle = int(arr.shape[1] / cycles_per_trial)
    if bins_per_cycle < 2:
        return np.nan

    used = cycles_per_trial * bins_per_cycle
    folded = arr[:, :used].reshape(arr.shape[0], cycles_per_trial, bins_per_cycle)
    average_cycle = np.mean(folded, axis=1)
    amplitude = 2.0 * np.abs(np.fft.fft(average_cycle, axis=1)) / bins_per_cycle
    f0 = 0.5 * amplitude[:, 0]
    f1 = amplitude[:, 1]
    selected = f0 > 0.0
    if not np.any(selected):
        return np.nan
    return float(np.nanmean(f1[selected] / f0[selected]))


def welch_modulation_index(
    response_psth: np.ndarray,
    temporal_frequency_hz: float,
    sample_rate_hz: float = 1000.0,
) -> float:
    """AllenSDK ``mod_idx_dg`` on a condition-averaged response PSTH."""
    response = np.asarray(response_psth, dtype=float)
    if response.size == 0:
        return np.nan
    # nperseg=1024 matches AllenSDK's mod_idx_dg default; SciPy silently caps
    # it to len(response) when the PSTH is shorter (~1000 samples for a 1-s
    # grating), so make that explicit rather than relying on the warning.
    nperseg = min(1024, response.size)
    frequencies, psd = signal.welch(response, fs=sample_rate_hz, nperseg=nperseg)
    mean_psd = float(np.mean(psd))
    if mean_psd == 0.0:
        return 0.0
    tf_index = int(np.searchsorted(frequencies, float(temporal_frequency_hz)))
    if not 0 <= tf_index < psd.size:
        return np.nan
    denominator = float(np.sqrt(np.mean(psd**2) - mean_psd**2))
    if denominator == 0.0:
        return np.nan
    return float(abs((psd[tf_index] - mean_psd) / denominator))


def _prepare_conditions(
    stimulus_presentations: pd.DataFrame,
    *,
    min_trials: int,
) -> tuple[list[dict[str, object]], float, int]:
    table = stimulus_presentations.copy()
    if "start_time" not in table.columns:
        raise ValueError("Drifting-grating table has no start_time column")

    resolved: dict[str, str] = {}
    for canonical, aliases in DIMENSION_ALIASES.items():
        column = _find_column(table, aliases)
        if column is not None:
            table[column] = pd.to_numeric(table[column], errors="coerce")
            resolved[canonical] = column

    for required in ("orientation", "temporal_frequency", "spatial_frequency"):
        if required not in resolved:
            raise ValueError(f"Drifting-grating table has no {required} column")

    valid = table["start_time"].notna()
    for canonical in ("orientation", "temporal_frequency", "spatial_frequency"):
        valid &= table[resolved[canonical]].notna()
    valid &= table[resolved["temporal_frequency"]] > 0
    valid &= table[resolved["spatial_frequency"]] > 0
    table = table.loc[valid].copy()
    if table.empty:
        raise ValueError("Drifting-grating table has no valid nonblank presentations")

    if "stop_time" in table.columns:
        observed = pd.to_numeric(table["stop_time"], errors="coerce") - pd.to_numeric(
            table["start_time"], errors="coerce"
        )
        observed_duration = float(np.nanmedian(observed))
    else:
        observed_duration = 1.0
    if not np.isfinite(observed_duration) or observed_duration <= 0:
        raise ValueError("Could not infer a positive grating presentation duration")
    analysis_duration_s = float(round(observed_duration, 2))
    duration_ms = int(round(analysis_duration_s * 1000.0))

    # Required dimensions are always in the key. Optional dimensions enter the
    # key only when the stimulus actually varies them.
    canonical_group_dimensions = ["orientation", "temporal_frequency", "spatial_frequency"]
    for optional in ("contrast", "phase"):
        if optional in resolved and table[resolved[optional]].nunique(dropna=True) > 1:
            canonical_group_dimensions.append(optional)
    group_columns = [resolved[name] for name in canonical_group_dimensions]

    conditions: list[dict[str, object]] = []
    grouped = table.groupby(group_columns, sort=True, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(group) < min_trials:
            continue
        parameters = dict(zip(canonical_group_dimensions, map(float, keys)))
        for optional in ("contrast", "phase"):
            if optional in resolved and optional not in parameters:
                unique = table[resolved[optional]].dropna().unique()
                if len(unique) == 1:
                    parameters[optional] = float(unique[0])
        conditions.append(
            {
                "parameters": parameters,
                "starts": group["start_time"].to_numpy(dtype=float),
            }
        )
    if not conditions:
        raise ValueError(f"No grating condition has at least {min_trials} trials")
    return conditions, analysis_duration_s, duration_ms


def compute_drifting_grating_metrics(
    unit_ids: Iterable[int],
    spikes_by_unit: dict[int, np.ndarray],
    stimulus_presentations: pd.DataFrame,
    *,
    min_trials: int = 5,
) -> pd.DataFrame:
    """Compute full-condition F1/F0 and ``mod_idx_dg`` for every unit.

    Preferred-condition selection uses mean spikes per presentation over the
    same nominal duration used by the two temporal-modulation metrics.  Exact
    ties are resolved by the lexicographically first condition and their count
    is retained for diagnosis.
    """
    unit_ids = [int(unit_id) for unit_id in unit_ids]
    if not unit_ids:
        return _empty_metrics([])
    try:
        conditions, duration_s, duration_ms = _prepare_conditions(
            stimulus_presentations, min_trials=min_trials
        )
    except ValueError:
        raise

    all_starts = np.concatenate([condition["starts"] for condition in conditions])
    condition_codes = np.concatenate(
        [
            np.full(len(condition["starts"]), index, dtype=int)
            for index, condition in enumerate(conditions)
        ]
    )
    condition_trial_counts = np.bincount(
        condition_codes, minlength=len(conditions)
    ).astype(float)

    rows: list[dict[str, float | int]] = []
    for unit_id in unit_ids:
        spikes = np.asarray(
            spikes_by_unit.get(unit_id, np.array([], dtype=float)), dtype=float
        )
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            spikes = np.sort(spikes)
        if spikes.size == 0:
            rows.append(_empty_metrics([unit_id]).iloc[0].to_dict())
            continue

        start_indices = np.searchsorted(spikes, all_starts, side="left")
        stop_indices = np.searchsorted(spikes, all_starts + duration_s, side="left")
        presentation_counts = stop_indices - start_indices
        condition_sums = np.bincount(
            condition_codes, weights=presentation_counts, minlength=len(conditions)
        )
        condition_means = condition_sums / condition_trial_counts
        best_index = int(np.argmax(condition_means))
        best_mean = float(condition_means[best_index])
        tie_count = int(
            np.count_nonzero(
                np.isclose(condition_means, best_mean, rtol=0.0, atol=0.0)
            )
        )
        best = conditions[best_index]
        parameters = best["parameters"]

        counts = _bin_trial_spike_counts(spikes, best["starts"], duration_ms)
        tf = float(parameters["temporal_frequency"])
        f1_f0 = f1_f0_from_trial_counts(counts, tf, duration_s)
        mod_idx = welch_modulation_index(np.mean(counts, axis=0), tf, 1000.0)
        rows.append(
            {
                "unit_id": unit_id,
                "f1_f0_dg": f1_f0,
                "mod_idx_dg": mod_idx,
                "pref_ori_dg": float(parameters["orientation"]),
                "pref_tf_dg": tf,
                "pref_sf_dg": float(parameters["spatial_frequency"]),
                "pref_contrast_dg": float(parameters.get("contrast", np.nan)),
                "pref_phase_dg": float(parameters.get("phase", np.nan)),
                "preferred_mean_spikes_dg": best_mean,
                "firing_rate_dg": best_mean / duration_s,
                "preferred_condition_ties_dg": tie_count,
                "preferred_trials_dg": int(len(best["starts"])),
                "analysis_duration_s_dg": duration_s,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

def prepare_grating_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "orientation",
        "temporal_frequency",
        "spatial_frequency",
        "contrast",
        "start_time",
        "stop_time",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Grating table lacks columns {missing}")
    result = table.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = (
        result[["orientation", "temporal_frequency", "spatial_frequency", "contrast"]]
        .notna()
        .all(axis=1)
        & result["temporal_frequency"].gt(0)
        & result["spatial_frequency"].gt(0)
    )
    return result.loc[valid].copy()


def condition_starts(
    table: pd.DataFrame,
    *,
    common_support: dict[str, object] | None = None,
) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
    selected = prepare_grating_table(table)
    if common_support is not None:
        selected = selected.loc[
            selected["orientation"].isin(common_support["orientation_deg"])
            & selected["temporal_frequency"].isin(
                common_support["temporal_frequency_hz"]
            )
            & np.isclose(
                selected["spatial_frequency"],
                float(common_support["spatial_frequency_cpd"]),
            )
            & np.isclose(selected["contrast"], float(common_support["contrast"]))
        ]
    dimensions = ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]
    # sort=False is essential: AllenSDK's condition IDs preserve first
    # presentation order, and idxmax resolves exact response ties by that order.
    return [
        (tuple(map(float, key)), group["start_time"].to_numpy(dtype=float))
        for key, group in selected.groupby(dimensions, sort=False)
    ]

def trial_f1_f0_components(
    spikes_s: np.ndarray,
    starts_s: np.ndarray,
    temporal_frequency_hz: float,
    *,
    duration_ms: int = 1000,
    trial_duration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trial spike counts and exact cycle-fold F1/F0 ratios.

    The ratio is evaluated directly from occupied 1-ms bins. It is algebraically
    identical to folding the binned trial into cycles and taking the first FFT
    harmonic, while avoiding a dense trials-by-time matrix for every unit.
    """
    starts = np.asarray(starts_s, dtype=float)
    spikes = np.asarray(spikes_s, dtype=float)
    cycles = int(float(temporal_frequency_hz) * float(trial_duration_s))
    bins_per_cycle = int(duration_ms / cycles)
    used_ms = cycles * bins_per_cycle
    counts = np.empty(len(starts), dtype=np.int32)
    ratios = np.full(len(starts), np.nan, dtype=float)
    phase_lookup = np.exp(-2j * np.pi * np.arange(bins_per_cycle) / bins_per_cycle)
    for index, start in enumerate(starts):
        first = np.searchsorted(spikes, start, side="left")
        last = np.searchsorted(spikes, start + duration_ms / 1000.0, side="left")
        counts[index] = last - first
        used_last = np.searchsorted(spikes, start + used_ms / 1000.0, side="left")
        trial_spikes = spikes[first:used_last]
        if len(trial_spikes) == 0:
            continue
        bins = np.floor((trial_spikes - start) * 1000.0 + 1e-7).astype(int)
        bins = bins[(bins >= 0) & (bins < used_ms)]
        if len(bins) == 0:
            continue
        ratios[index] = 2.0 * np.abs(phase_lookup[bins % bins_per_cycle].sum()) / len(bins)
    return counts, ratios


def prepare_unit(
    spikes: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
) -> list[dict[str, object]]:
    prepared = []
    for parameters, starts in conditions:
        counts, ratios = trial_f1_f0_components(spikes, starts, parameters[1])
        prepared.append({"parameters": parameters, "counts": counts, "ratios": ratios})
    return prepared


def summarize_draw(
    prepared: list[dict[str, object]], indices_by_condition: list[np.ndarray]
) -> dict[str, object]:
    means = [
        float(np.mean(condition["counts"][indices]))
        for condition, indices in zip(prepared, indices_by_condition)
    ]
    preferred = int(np.argmax(means))
    condition = prepared[preferred]
    indices = indices_by_condition[preferred]
    ratios = condition["ratios"][indices]
    valid = ratios[np.isfinite(ratios)]
    return {
        "preferred_condition_index": preferred,
        "preferred_orientation_deg": float(condition["parameters"][0]),
        "preferred_tf_hz": float(condition["parameters"][1]),
        "f1_f0": float(valid.mean()) if len(valid) else np.nan,
        "preferred_mean_spikes": float(np.mean(condition["counts"][indices])),
        "selected_trials": int(len(indices)),
        "valid_trials": int(len(valid)),
        "zero_f0_trials": int(len(indices) - len(valid)),
    }
