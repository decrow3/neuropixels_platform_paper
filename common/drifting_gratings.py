"""Low-level drifting-grating metrics for the MouseV2 NWB tables.

The published Allen ``DriftingGratings`` analysis chooses a unit's preferred
stimulus *condition* and computes both F1/F0 and ``mod_idx_dg`` at that
condition.  Allen's drifting-grating stimulus has a fixed spatial frequency;
MouseV2 varies spatial frequency, so its condition key must include SF as well
as orientation and temporal frequency.  Contrast and phase are also included
when they vary.

MouseV2 presentations last approximately 1.00084 seconds because the logged
stop time follows display timing.  The intended protocol duration is 1.0 s.
We therefore round the median observed duration to two decimal places before
constructing 1-ms bins.  This gives exactly 1,000 bins and an integral number
of cycles for every tested temporal frequency (1, 2, 4, 8, and 15 Hz).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats


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


FREQUENCY_SURFACE_COLUMNS = (
    "unit_id",
    "spatial_frequency_cpd",
    "temporal_frequency_hz",
    "mean_spikes",
    "sem_spikes",
    "trials",
    "split_a_mean_spikes",
    "split_b_mean_spikes",
)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving missing values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    ordered = finite[np.argsort(values[finite], kind="mergesort")]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    raw = values[ordered] * len(ordered) / ranks
    raw = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted[ordered] = np.minimum(raw, 1.0)
    return adjusted


def _one_way_f_test(
    responses: np.ndarray,
    group_codes: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized balanced/unbalanced one-way ANOVA across presentation labels."""
    y = np.asarray(responses, dtype=float)
    codes = np.asarray(group_codes, dtype=int)
    if y.ndim != 2 or y.shape[1] != len(codes):
        raise ValueError("responses must be units x presentations")
    counts = np.bincount(codes, minlength=group_count).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every tuning group must contain presentations")
    sums = np.stack(
        [np.sum(y[:, codes == code], axis=1) for code in range(group_count)], axis=1
    )
    means = sums / counts[None, :]
    grand = np.mean(y, axis=1)
    ss_between = np.sum(counts[None, :] * (means - grand[:, None]) ** 2, axis=1)
    fitted = means[:, codes]
    ss_within = np.sum((y - fitted) ** 2, axis=1)
    df_between = group_count - 1
    df_within = y.shape[1] - group_count
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (ss_between / df_between) / (ss_within / df_within)
    statistic[(ss_within == 0) & (ss_between > 0)] = np.inf
    statistic[(ss_within == 0) & (ss_between == 0)] = 0.0
    p_value = stats.f.sf(statistic, df_between, df_within)
    return statistic, p_value


def _rowwise_pearson_test(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Pearson correlation and one-sided p-value for positive reliability."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.sum(x_centered * y_centered, axis=1) / denominator
    correlation = np.clip(correlation, -1.0, 1.0)
    degrees_freedom = x.shape[1] - 2
    t_value = np.full(correlation.shape, np.nan, dtype=float)
    interior = np.isfinite(correlation) & (np.abs(correlation) < 1.0)
    t_value[interior] = correlation[interior] * np.sqrt(
        degrees_freedom / (1.0 - correlation[interior] ** 2)
    )
    t_value[correlation == 1.0] = np.inf
    t_value[correlation == -1.0] = -np.inf
    p_value = stats.t.sf(t_value, degrees_freedom)
    p_value[~np.isfinite(correlation)] = np.nan
    return correlation, p_value


def _frequency_trial_design(
    stimulus_presentations: pd.DataFrame,
    *,
    min_trials_per_sf_tf: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return valid presentations and balanced SF/TF and split-half labels."""
    table = stimulus_presentations.copy()
    resolved = {
        name: _find_column(table, aliases) for name, aliases in DIMENSION_ALIASES.items()
    }
    for required in ("orientation", "temporal_frequency", "spatial_frequency"):
        if resolved[required] is None:
            raise ValueError(f"Drifting-grating table has no {required} column")
    if "start_time" not in table:
        raise ValueError("Drifting-grating table has no start_time column")
    for column in filter(None, resolved.values()):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["start_time"] = pd.to_numeric(table["start_time"], errors="coerce")
    valid = table["start_time"].notna()
    for required in ("orientation", "temporal_frequency", "spatial_frequency"):
        valid &= table[resolved[required]].notna()
    valid &= table[resolved["temporal_frequency"]] > 0
    valid &= table[resolved["spatial_frequency"]] > 0
    table = table.loc[valid].sort_values("start_time", kind="mergesort").reset_index(drop=True)
    if table.empty:
        raise ValueError("Drifting-grating table has no valid nonblank presentations")

    table = table.rename(
        columns={
            resolved["orientation"]: "_orientation",
            resolved["temporal_frequency"]: "_tf",
            resolved["spatial_frequency"]: "_sf",
        }
    )
    sf_values = np.sort(table["_sf"].unique())
    tf_values = np.sort(table["_tf"].unique())
    expected = pd.MultiIndex.from_product([sf_values, tf_values])
    observed_counts = table.groupby(["_sf", "_tf"], observed=True).size().reindex(expected)
    if observed_counts.isna().any() or (observed_counts < min_trials_per_sf_tf).any():
        raise ValueError("SF x TF design is incomplete or lacks minimum trial support")

    sf_lookup = {value: index for index, value in enumerate(sf_values)}
    tf_lookup = {value: index for index, value in enumerate(tf_values)}
    sf_codes = table["_sf"].map(sf_lookup).to_numpy(dtype=int)
    tf_codes = table["_tf"].map(tf_lookup).to_numpy(dtype=int)
    joint_codes = sf_codes * len(tf_values) + tf_codes

    # Alternate repeats within every complete nuisance condition. Flipping the
    # starting half across adjacent nuisance groups balances odd repeat counts
    # within each marginal SF x TF cell.
    nuisance_columns = ["_sf", "_tf", "_orientation"]
    for optional in ("contrast", "phase"):
        column = resolved[optional]
        if column is not None and column in table and table[column].nunique(dropna=True) > 1:
            nuisance_columns.append(column)
    split_codes = np.zeros(len(table), dtype=int)
    for group_number, (_, indices) in enumerate(
        table.groupby(nuisance_columns, sort=True, dropna=False).groups.items()
    ):
        ordered = np.asarray(list(indices), dtype=int)
        split_codes[ordered] = (np.arange(len(ordered)) + group_number) % 2

    if "stop_time" in table:
        durations = pd.to_numeric(table["stop_time"], errors="coerce") - table["start_time"]
        duration_s = float(round(float(np.nanmedian(durations)), 2))
    else:
        duration_s = 1.0
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Could not infer a positive grating presentation duration")
    return table, sf_codes, tf_codes, split_codes, duration_s


def compute_frequency_tuning_surfaces(
    unit_ids: Iterable[int],
    spikes_by_unit: dict[int, np.ndarray],
    stimulus_presentations: pd.DataFrame,
    *,
    min_trials_per_sf_tf: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build marginal trial-level SF x TF surfaces and unadjusted support tests.

    Orientation and any varying nuisance dimensions remain balanced because the
    surface averages every presentation within an SF x TF cell. The returned
    p-values are intentionally unadjusted; dataset-level extraction applies FDR
    after concatenating sessions.
    """
    ids = np.asarray([int(unit_id) for unit_id in unit_ids], dtype=int)
    table, sf_codes, tf_codes, split_codes, duration_s = _frequency_trial_design(
        stimulus_presentations, min_trials_per_sf_tf=min_trials_per_sf_tf
    )
    sf_values = np.sort(table["_sf"].unique())
    tf_values = np.sort(table["_tf"].unique())
    joint_codes = sf_codes * len(tf_values) + tf_codes
    joint_count = len(sf_values) * len(tf_values)
    starts = table["start_time"].to_numpy(dtype=float)

    responses = np.zeros((len(ids), len(starts)), dtype=np.float32)
    for row, unit_id in enumerate(ids):
        spikes = np.asarray(spikes_by_unit.get(int(unit_id), []), dtype=float)
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            spikes = np.sort(spikes)
        responses[row] = np.searchsorted(spikes, starts + duration_s, side="left") - np.searchsorted(
            spikes, starts, side="left"
        )

    cell_counts = np.bincount(joint_codes, minlength=joint_count).astype(float)
    cell_sums = np.stack(
        [np.sum(responses[:, joint_codes == code], axis=1) for code in range(joint_count)],
        axis=1,
    )
    cell_means = cell_sums / cell_counts[None, :]
    cell_ss = np.stack(
        [
            np.sum((responses[:, joint_codes == code] - cell_means[:, code, None]) ** 2, axis=1)
            for code in range(joint_count)
        ],
        axis=1,
    )
    cell_sem = np.sqrt(cell_ss / np.maximum(cell_counts[None, :] - 1.0, 1.0)) / np.sqrt(
        cell_counts[None, :]
    )

    split_means = []
    split_counts = []
    for split in (0, 1):
        counts = np.bincount(joint_codes[split_codes == split], minlength=joint_count).astype(float)
        sums = np.stack(
            [
                np.sum(responses[:, (joint_codes == code) & (split_codes == split)], axis=1)
                for code in range(joint_count)
            ],
            axis=1,
        )
        split_counts.append(counts)
        split_means.append(sums / counts[None, :])

    joint_f, joint_p = _one_way_f_test(responses, joint_codes, joint_count)
    sf_f, sf_p = _one_way_f_test(responses, sf_codes, len(sf_values))
    tf_f, tf_p = _one_way_f_test(responses, tf_codes, len(tf_values))
    reliability_r, reliability_p = _rowwise_pearson_test(split_means[0], split_means[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        reliability_sb = 2.0 * reliability_r / (1.0 + reliability_r)

    best_codes = np.argmax(cell_means, axis=1)
    metrics = pd.DataFrame(
        {
            "unit_id": ids,
            "sf_tf_joint_f": joint_f,
            "sf_tf_joint_p": joint_p,
            "sf_main_f": sf_f,
            "sf_main_p": sf_p,
            "tf_main_f": tf_f,
            "tf_main_p": tf_p,
            "surface_split_half_r": reliability_r,
            "surface_split_half_spearman_brown": reliability_sb,
            "surface_reliability_p": reliability_p,
            "surface_peak_sf_cpd": sf_values[best_codes // len(tf_values)],
            "surface_peak_tf_hz": tf_values[best_codes % len(tf_values)],
            "analysis_duration_s": duration_s,
            "sf_levels": len(sf_values),
            "tf_levels": len(tf_values),
            "presentations": len(starts),
        }
    )

    surface_frames = []
    for code in range(joint_count):
        surface_frames.append(
            pd.DataFrame(
                {
                    "unit_id": ids,
                    "spatial_frequency_cpd": sf_values[code // len(tf_values)],
                    "temporal_frequency_hz": tf_values[code % len(tf_values)],
                    "mean_spikes": cell_means[:, code],
                    "sem_spikes": cell_sem[:, code],
                    "trials": int(cell_counts[code]),
                    "split_a_mean_spikes": split_means[0][:, code],
                    "split_b_mean_spikes": split_means[1][:, code],
                }
            )
        )
    surfaces = pd.concat(surface_frames, ignore_index=True)[list(FREQUENCY_SURFACE_COLUMNS)]
    return metrics, surfaces
