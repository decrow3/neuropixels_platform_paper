#!/usr/bin/env python3
"""Test shared residual grating phase across probes and behavioral covariation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import _bin_trial_spike_counts  # noqa: E402
from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    MOUSE_CONFIG,
    STIMULUS_MANIFEST,
    phase_aware_conditions,
    phase_schedule,
    weighted_phase_coherence,
)


START_PHASE_IMPORT = (
    ROOT / "data" / "imports" / "mousev2_grating_start_phase_bridge_v1"
)
START_PHASE_MANIFEST = START_PHASE_IMPORT / "import_manifest.json"
DEFAULT_OUTPUT = (
    ROOT / "data" / "imports" / "mousev2_grating_shared_phase_behavior_v1"
)
BASE_SEED = 20260806
MIN_VALID_EYE_FRACTION = 0.5
BEHAVIOR_METRICS = {
    "running_abs_median_stim": "absolute running speed",
    "log_pupil_area_median_stim": "log pupil area",
    "pupil_x_median_stim": "horizontal pupil position",
    "pupil_y_median_stim": "vertical pupil position",
}
BLOCK_POSITION_METRIC = "presentation_ordinal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MOUSE_CONFIG)
    parser.add_argument("--stimulus-manifest", type=Path, default=STIMULUS_MANIFEST)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-permutations", type=int, default=1000)
    parser.add_argument("--behavior-permutations", type=int, default=500)
    parser.add_argument("--permutation-seed", type=int, default=BASE_SEED)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def condition_id(parameters: tuple[float, ...]) -> str:
    return f"ori{parameters[0]:g}_tf{parameters[1]:g}"


def trial_coefficients(
    spikes_s: np.ndarray,
    conditions: list[dict[str, object]],
) -> tuple[dict[str, object], np.ndarray]:
    means = []
    for condition in conditions:
        starts = np.asarray(condition["starts"], dtype=float)
        first = np.searchsorted(spikes_s, starts, side="left")
        last = np.searchsorted(spikes_s, starts + 1.0, side="left")
        means.append(float(np.mean(last - first)))
    selected = conditions[int(np.argmax(means))]
    starts = np.asarray(selected["starts"], dtype=float)
    counts = _bin_trial_spike_counts(spikes_s, starts, duration_ms=1000)
    temporal_frequency_hz = float(selected["parameters"][1])
    time_s = np.arange(1000, dtype=float) / 1000.0
    kernel = np.exp(-2j * np.pi * temporal_frequency_hz * time_s)
    coefficients = 2.0 * (counts @ kernel)
    phases = np.asarray(selected["start_phase_cycles"], dtype=float)
    return selected, coefficients * np.exp(-2j * np.pi * phases)


def _normalize_complex(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=complex)
    magnitude = np.abs(values)
    valid = np.isfinite(magnitude) & (magnitude > 0)
    result = np.zeros(values.shape, dtype=complex)
    result[valid] = values[valid] / magnitude[valid]
    return result, valid


def leave_trial_out_residual_phase(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove each unit's intrinsic phase without reusing the target trial."""
    coefficients = np.asarray(coefficients, dtype=complex)
    trial_phase, trial_valid = _normalize_complex(coefficients)
    reference = np.sum(coefficients, axis=1, keepdims=True) - coefficients
    reference_phase, reference_valid = _normalize_complex(reference)
    valid = trial_valid & reference_valid
    residual = np.zeros(coefficients.shape, dtype=complex)
    residual[valid] = trial_phase[valid] * np.conjugate(reference_phase[valid])
    return residual, valid


def _consensus_excluding_targets(
    residual: np.ndarray,
    valid: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all-other-unit and other-probe consensus for every unit/trial."""
    summed = np.sum(residual, axis=0, keepdims=True)
    all_other = summed - residual
    all_other_phase, all_other_valid = _normalize_complex(all_other)

    other_probe = np.empty(residual.shape, dtype=complex)
    other_probe_count = np.empty(residual.shape, dtype=int)
    total_count = np.sum(valid, axis=0, keepdims=True)
    for probe in np.unique(probes):
        selected = probes == probe
        probe_sum = np.sum(residual[selected], axis=0, keepdims=True)
        probe_count = np.sum(valid[selected], axis=0, keepdims=True)
        other_probe[selected] = summed - probe_sum
        other_probe_count[selected] = total_count - probe_count
    other_probe_phase, other_probe_valid = _normalize_complex(other_probe)
    other_probe_valid &= other_probe_count > 0
    other_probe_phase[~other_probe_valid] = 0
    all_other_valid &= (np.sum(valid, axis=0, keepdims=True) - valid) > 0
    all_other_phase[~all_other_valid] = 0
    return all_other_phase, other_probe_phase


def _alignment(
    residual: np.ndarray,
    valid: np.ndarray,
    consensus_phase: np.ndarray,
) -> tuple[np.ndarray, float]:
    consensus_valid = np.abs(consensus_phase) > 0
    selected = valid & consensus_valid
    values = np.full(residual.shape, np.nan, dtype=float)
    values[selected] = np.real(
        residual[selected] * np.conjugate(consensus_phase[selected])
    )
    finite = np.isfinite(values)
    unit_values = np.divide(
        np.nansum(values, axis=1),
        np.sum(finite, axis=1),
        out=np.full(len(values), np.nan),
        where=np.sum(finite, axis=1) > 0,
    )
    return unit_values, float(np.nanmean(values))


def _adjusted_coherence(
    coefficients: np.ndarray,
    consensus_phase: np.ndarray,
) -> np.ndarray:
    rows = []
    for unit_index in range(len(coefficients)):
        valid = np.abs(consensus_phase[unit_index]) > 0
        if np.sum(valid) < 2:
            rows.append(np.nan)
            continue
        adjusted = coefficients[unit_index, valid] * np.conjugate(
            consensus_phase[unit_index, valid]
        )
        rows.append(weighted_phase_coherence(adjusted))
    return np.asarray(rows, dtype=float)


def _probe_balanced_trial_consensus(
    residual: np.ndarray,
    valid: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probe_means = []
    for probe in np.unique(probes):
        selected = probes == probe
        counts = np.sum(valid[selected], axis=0)
        values = np.zeros(residual.shape[1], dtype=complex)
        usable = counts > 0
        values[usable] = np.sum(residual[selected], axis=0)[usable] / counts[usable]
        probe_means.append(values)
    matrix = np.asarray(probe_means)
    consensus = np.mean(matrix, axis=0)
    valid_units = np.sum(valid, axis=0)
    return consensus, valid_units


def shared_phase_condition(
    coefficients: np.ndarray,
    probes: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Measure trial-matched phase shared across units and across probes."""
    coefficients = np.asarray(coefficients, dtype=complex)
    probes = np.asarray(probes, dtype=str)
    if coefficients.ndim != 2 or coefficients.shape[1] < 3:
        raise ValueError("Expected units x trials complex coefficients")
    if len(probes) != len(coefficients) or len(np.unique(probes)) < 2:
        raise ValueError("Shared-phase analysis requires units on at least two probes")
    if permutations < 1:
        raise ValueError("At least one phase permutation is required")

    residual, valid = leave_trial_out_residual_phase(coefficients)
    all_other, other_probe = _consensus_excluding_targets(residual, valid, probes)
    unit_all_alignment, all_alignment = _alignment(residual, valid, all_other)
    unit_cross_alignment, cross_alignment = _alignment(residual, valid, other_probe)
    source_coherence = np.asarray(
        [weighted_phase_coherence(row) for row in coefficients], dtype=float
    )
    all_adjusted = _adjusted_coherence(coefficients, all_other)
    cross_adjusted = _adjusted_coherence(coefficients, other_probe)
    trial_consensus, valid_units = _probe_balanced_trial_consensus(
        residual, valid, probes
    )

    rng = np.random.default_rng(seed)
    permutation_rows = []
    for permutation in range(permutations):
        shuffled = np.empty_like(residual)
        shuffled_valid = np.empty_like(valid)
        for unit_index in range(len(residual)):
            order = rng.permutation(residual.shape[1])
            shuffled[unit_index] = residual[unit_index, order]
            shuffled_valid[unit_index] = valid[unit_index, order]
        null_all, null_cross = _consensus_excluding_targets(
            shuffled, shuffled_valid, probes
        )
        _, null_all_alignment = _alignment(shuffled, shuffled_valid, null_all)
        _, null_cross_alignment = _alignment(shuffled, shuffled_valid, null_cross)
        null_cross_adjusted = _adjusted_coherence(coefficients, null_cross)
        permutation_rows.append(
            {
                "permutation": permutation,
                "all_other_alignment": null_all_alignment,
                "cross_probe_alignment": null_cross_alignment,
                "mean_cross_probe_adjusted_coherence": float(
                    np.nanmean(null_cross_adjusted)
                ),
                "mean_cross_probe_coherence_gain": float(
                    np.nanmean(null_cross_adjusted - source_coherence)
                ),
            }
        )
    permutations_table = pd.DataFrame(permutation_rows)
    return {
        "source_coherence": source_coherence,
        "all_other_adjusted_coherence": all_adjusted,
        "cross_probe_adjusted_coherence": cross_adjusted,
        "unit_all_other_alignment": unit_all_alignment,
        "unit_cross_probe_alignment": unit_cross_alignment,
        "all_other_alignment": all_alignment,
        "cross_probe_alignment": cross_alignment,
        "trial_consensus": trial_consensus,
        "trial_valid_units": valid_units,
        "permutations": permutations_table,
    }


def _read_behavior_signals(nwb_path: Path) -> dict[str, np.ndarray]:
    import h5py

    from generate_retinotopic_csvs import _read_numeric_dset

    paths = {
        "running": "/processing/running/running_speed/data",
        "running_time": "/processing/running/running_speed/timestamps",
        "pupil_area": "/processing/eye_tracking/pupil/area",
        "pupil_x": "/processing/eye_tracking/pupil/data_x",
        "pupil_y": "/processing/eye_tracking/pupil/data_y",
        "eye_time": "/processing/eye_tracking/pupil/timestamps",
        "blink": "/processing/eye_tracking/likely_blink_times/data",
    }
    fid = h5py.h5f.open(str(nwb_path).encode(), flags=h5py.h5f.ACC_RDONLY)
    try:
        result = {name: _read_numeric_dset(fid, path) for name, path in paths.items()}
    finally:
        fid.close()
    if len(result["running"]) != len(result["running_time"]):
        raise ValueError(f"{nwb_path}: running data/timestamp length mismatch")
    eye_length = len(result["eye_time"])
    if any(len(result[name]) != eye_length for name in ("pupil_area", "pupil_x", "pupil_y", "blink")):
        raise ValueError(f"{nwb_path}: eye-tracking data/timestamp length mismatch")
    return result


def _window_values(
    values: np.ndarray,
    timestamps: np.ndarray,
    start: float,
    stop: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    first = int(np.searchsorted(timestamps, start, side="left"))
    last = int(np.searchsorted(timestamps, stop, side="left"))
    selected = np.asarray(values[first:last], dtype=float)
    keep = np.isfinite(selected)
    if valid is not None:
        keep &= valid[first:last]
    return selected[keep]


def trial_behavior_table(
    table: pd.DataFrame,
    signals: dict[str, np.ndarray],
) -> pd.DataFrame:
    ordered = table.sort_values("start_time").reset_index(drop=True)
    ids = pd.to_numeric(ordered["id"], errors="coerce").to_numpy(dtype=int)
    if not np.array_equal(ids, np.arange(len(ordered))):
        raise ValueError("Behavior join requires chronological presentation ids")
    common = ordered.loc[np.isclose(ordered["spatial_frequency"], 0.04)].copy()
    common = common.loc[
        np.isclose(common["contrast"], 0.8)
        & common["orientation"].isin([0.0, 45.0, 90.0, 135.0])
        & common["temporal_frequency"].isin([1.0, 2.0, 4.0, 8.0, 15.0])
    ]
    eye_valid = (
        np.isfinite(signals["pupil_area"])
        & np.isfinite(signals["pupil_x"])
        & np.isfinite(signals["pupil_y"])
        & (signals["pupil_area"] > 10)
        & (signals["pupil_x"] >= 0)
        & (signals["pupil_y"] >= 0)
        & (signals["blink"] < 0.5)
    )
    rows = []
    for row in common.itertuples():
        start = float(row.start_time)
        stop = start + 1.0
        running = _window_values(
            np.abs(signals["running"]), signals["running_time"], start, stop
        )
        area = _window_values(
            signals["pupil_area"], signals["eye_time"], start, stop, eye_valid
        )
        pupil_x = _window_values(
            signals["pupil_x"], signals["eye_time"], start, stop, eye_valid
        )
        pupil_y = _window_values(
            signals["pupil_y"], signals["eye_time"], start, stop, eye_valid
        )
        blink = _window_values(
            signals["blink"], signals["eye_time"], start, stop
        )
        eye_samples = len(blink)
        valid_eye_fraction = len(area) / eye_samples if eye_samples else 0.0
        eye_window_usable = valid_eye_fraction >= MIN_VALID_EYE_FRACTION
        parameters = (
            float(row.orientation),
            float(row.temporal_frequency),
            float(row.spatial_frequency),
            float(row.contrast),
        )
        rows.append(
            {
                "presentation_ordinal": int(row.id),
                "condition_id": condition_id(parameters),
                "start_time": start,
                "orientation_deg": parameters[0],
                "temporal_frequency_hz": parameters[1],
                "running_abs_median_stim": float(np.median(running)) if len(running) else np.nan,
                "log_pupil_area_median_stim": (
                    float(np.log(np.median(area))) if eye_window_usable else np.nan
                ),
                "pupil_x_median_stim": (
                    float(np.median(pupil_x)) if eye_window_usable else np.nan
                ),
                "pupil_y_median_stim": (
                    float(np.median(pupil_y)) if eye_window_usable else np.nan
                ),
                "blink_fraction_stim": float(np.mean(blink > 0.5)) if len(blink) else np.nan,
                "running_samples": len(running),
                "eye_samples": eye_samples,
                "valid_eye_samples": len(area),
                "valid_eye_fraction": valid_eye_fraction,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 300 or result["presentation_ordinal"].duplicated().any():
        raise ValueError("Expected 300 unique common-support behavior presentations")
    return result


def complex_phase_association(
    table: pd.DataFrame,
    metric: str,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray, int]:
    """Condition-stratified complex correlation with a within-condition null."""
    selected = table.dropna(
        subset=[metric, "population_phase_real", "population_phase_imag"]
    ).copy()
    x_parts = []
    y_parts = []
    slices = []
    offset = 0
    for _, group in selected.groupby("condition_id", sort=True):
        x = group[metric].to_numpy(dtype=float)
        if len(x) < 5 or np.std(x) == 0:
            continue
        x = (x - np.mean(x)) / np.std(x)
        y = group["population_phase_real"].to_numpy(dtype=float) + 1j * group[
            "population_phase_imag"
        ].to_numpy(dtype=float)
        magnitude = np.abs(y)
        phase_valid = magnitude > 0
        if np.sum(phase_valid) < 5:
            continue
        x = x[phase_valid]
        y = y[phase_valid] / magnitude[phase_valid]
        x_parts.append(x)
        y_parts.append(y)
        slices.append(slice(offset, offset + len(x)))
        offset += len(x)
    if not x_parts:
        return np.nan, np.full(permutations, np.nan), 0
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)

    def association(values: np.ndarray) -> float:
        denominator = float(np.sqrt(np.sum(values**2) * np.sum(np.abs(y) ** 2)))
        return float(np.abs(np.sum(values * y)) / denominator) if denominator > 0 else np.nan

    observed = association(x)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for permutation in range(permutations):
        shuffled = x.copy()
        for current_slice in slices:
            shuffled[current_slice] = rng.permutation(shuffled[current_slice])
        null[permutation] = association(shuffled)
    return observed, null, len(x)


def residualize_metric_against_block_time(
    table: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    """Remove condition means and linear/quadratic grating-block time trends."""
    result = np.full(len(table), np.nan, dtype=float)
    columns = table[["condition_id", "presentation_ordinal", metric]]
    selected_positions = np.flatnonzero(columns.notna().all(axis=1).to_numpy())
    selected = columns.iloc[selected_positions]
    if selected.empty:
        return result
    centered = selected[metric] - selected.groupby("condition_id")[metric].transform("mean")
    time = selected["presentation_ordinal"].to_numpy(dtype=float)
    time = (time - np.mean(time)) / np.std(time)
    design = np.column_stack([np.ones(len(time)), time, time**2])
    coefficients, *_ = np.linalg.lstsq(design, centered.to_numpy(dtype=float), rcond=None)
    residual = centered.to_numpy(dtype=float) - design @ coefficients
    result[selected_positions] = residual
    return result


def summarize_shared_sessions(
    conditions: pd.DataFrame,
    permutations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (site, session_id), group in conditions.groupby(["site", "session_id"], sort=True):
        weights = group["n_units"].to_numpy(dtype=float)

        def weighted(column: str) -> float:
            return float(np.average(group[column], weights=weights))

        null = permutations.loc[
            permutations["site"].eq(site) & permutations["session_id"].eq(session_id)
        ]
        null_rows = []
        for permutation, current in null.groupby("permutation", sort=True):
            by_condition = current.set_index("condition_id").loc[group["condition_id"]]
            null_rows.append(
                {
                    "permutation": permutation,
                    "cross_probe_alignment": float(
                        np.average(by_condition["cross_probe_alignment"], weights=weights)
                    ),
                    "cross_probe_adjusted_coherence": float(
                        np.average(
                            by_condition["mean_cross_probe_adjusted_coherence"],
                            weights=weights,
                        )
                    ),
                }
            )
        null_summary = pd.DataFrame(null_rows)
        observed_alignment = weighted("cross_probe_alignment")
        observed_adjusted = weighted("mean_cross_probe_adjusted_coherence")
        source = weighted("mean_source_corrected_coherence")
        rows.append(
            {
                "site": site,
                "session_id": int(session_id),
                "conditions": len(group),
                "units": int(group["n_units"].sum()),
                "cross_probe_alignment": observed_alignment,
                "permutation_cross_probe_alignment_mean": null_summary[
                    "cross_probe_alignment"
                ].mean(),
                "permutation_cross_probe_alignment_025": null_summary[
                    "cross_probe_alignment"
                ].quantile(0.025),
                "permutation_cross_probe_alignment_975": null_summary[
                    "cross_probe_alignment"
                ].quantile(0.975),
                "cross_probe_alignment_p": (
                    1
                    + np.sum(
                        null_summary["cross_probe_alignment"] >= observed_alignment
                    )
                )
                / (len(null_summary) + 1),
                "mean_source_corrected_coherence": source,
                "mean_cross_probe_adjusted_coherence": observed_adjusted,
                "mean_cross_probe_coherence_gain": observed_adjusted - source,
                "permutation_adjusted_coherence_mean": null_summary[
                    "cross_probe_adjusted_coherence"
                ].mean(),
                "permutation_adjusted_coherence_025": null_summary[
                    "cross_probe_adjusted_coherence"
                ].quantile(0.025),
                "permutation_adjusted_coherence_975": null_summary[
                    "cross_probe_adjusted_coherence"
                ].quantile(0.975),
                "cross_probe_adjusted_coherence_p": (
                    1
                    + np.sum(
                        null_summary["cross_probe_adjusted_coherence"]
                        >= observed_adjusted
                    )
                )
                / (len(null_summary) + 1),
            }
        )
    return pd.DataFrame(rows)


def summarize_shared_center(
    sessions: pd.DataFrame,
    conditions: pd.DataFrame,
    permutations: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the matched-trial null with sessions as the equal-weight unit."""
    null_session_rows = []
    for (site, session_id), group in conditions.groupby(
        ["site", "session_id"], sort=True
    ):
        weights = group["n_units"].to_numpy(dtype=float)
        current_null = permutations.loc[
            permutations["site"].eq(site)
            & permutations["session_id"].eq(session_id)
        ]
        for permutation, current in current_null.groupby("permutation", sort=True):
            by_condition = current.set_index("condition_id").loc[group["condition_id"]]
            null_session_rows.append(
                {
                    "site": site,
                    "session_id": int(session_id),
                    "permutation": int(permutation),
                    "cross_probe_alignment": float(
                        np.average(
                            by_condition["cross_probe_alignment"], weights=weights
                        )
                    ),
                    "cross_probe_adjusted_coherence": float(
                        np.average(
                            by_condition["mean_cross_probe_adjusted_coherence"],
                            weights=weights,
                        )
                    ),
                }
            )
    null_sessions = pd.DataFrame(null_session_rows)
    aggregate_null = null_sessions.groupby("permutation")[
        ["cross_probe_alignment", "cross_probe_adjusted_coherence"]
    ].mean()
    observed_alignment = float(sessions["cross_probe_alignment"].mean())
    observed_adjusted = float(
        sessions["mean_cross_probe_adjusted_coherence"].mean()
    )
    return pd.DataFrame(
        [
            {
                "sessions": sessions["session_id"].nunique(),
                "phase_permutations": aggregate_null.index.nunique(),
                "equal_session_source_corrected_coherence": sessions[
                    "mean_source_corrected_coherence"
                ].mean(),
                "equal_session_cross_probe_adjusted_coherence": observed_adjusted,
                "equal_session_cross_probe_coherence_gain": sessions[
                    "mean_cross_probe_coherence_gain"
                ].mean(),
                "equal_session_cross_probe_alignment": observed_alignment,
                "permutation_cross_probe_alignment_mean": aggregate_null[
                    "cross_probe_alignment"
                ].mean(),
                "permutation_cross_probe_alignment_025": aggregate_null[
                    "cross_probe_alignment"
                ].quantile(0.025),
                "permutation_cross_probe_alignment_975": aggregate_null[
                    "cross_probe_alignment"
                ].quantile(0.975),
                "cross_probe_alignment_p": (
                    1
                    + np.sum(
                        aggregate_null["cross_probe_alignment"]
                        >= observed_alignment
                    )
                )
                / (len(aggregate_null) + 1),
                "permutation_adjusted_coherence_mean": aggregate_null[
                    "cross_probe_adjusted_coherence"
                ].mean(),
                "permutation_adjusted_coherence_025": aggregate_null[
                    "cross_probe_adjusted_coherence"
                ].quantile(0.025),
                "permutation_adjusted_coherence_975": aggregate_null[
                    "cross_probe_adjusted_coherence"
                ].quantile(0.975),
                "cross_probe_adjusted_coherence_p": (
                    1
                    + np.sum(
                        aggregate_null["cross_probe_adjusted_coherence"]
                        >= observed_adjusted
                    )
                )
                / (len(aggregate_null) + 1),
                "sessions_alignment_p_le_0_05": int(
                    (sessions["cross_probe_alignment_p"] <= 0.05).sum()
                ),
            }
        ]
    )


def summarize_behavior_centers(
    sessions: pd.DataFrame,
    permutations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (metric, analysis, label), observed in sessions.groupby(
        ["behavior_metric", "analysis", "label"], sort=False
    ):
        null = permutations.loc[
            permutations["behavior_metric"].eq(metric)
            & permutations["analysis"].eq(analysis)
        ]
        aggregate_null = null.groupby("permutation")["association"].mean()
        center = observed["association"].mean()
        rows.append(
            {
                "behavior_metric": metric,
                "analysis": analysis,
                "label": label,
                "sessions": observed["session_id"].nunique(),
                "equal_session_association": center,
                "permutation_mean": aggregate_null.mean(),
                "permutation_025": aggregate_null.quantile(0.025),
                "permutation_975": aggregate_null.quantile(0.975),
                "permutation_p": (1 + np.sum(aggregate_null >= center))
                / (len(aggregate_null) + 1),
                "sessions_above_null_mean": int(
                    (
                        observed["association"]
                        > observed["permutation_mean"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    sessions: pd.DataFrame,
    conditions: pd.DataFrame,
    behavior_sessions: pd.DataFrame,
    behavior_centers: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    ax = axes[0, 0]
    for row in sessions.itertuples():
        ax.plot(
            [0, 1],
            [row.mean_source_corrected_coherence, row.mean_cross_probe_adjusted_coherence],
            color="#888888",
            alpha=0.8,
        )
    ax.scatter(
        np.zeros(len(sessions)),
        sessions["mean_source_corrected_coherence"],
        color="#1B9E77",
        zorder=3,
    )
    ax.scatter(
        np.ones(len(sessions)),
        sessions["mean_cross_probe_adjusted_coherence"],
        color="#377EB8",
        zorder=3,
    )
    ax.set(
        xticks=[0, 1],
        xticklabels=["source-phase\nadjusted", "other-probe phase\nadjusted"],
        ylabel="session mean weighted coherence",
        title="A. Cross-probe phase prediction",
    )

    ax = axes[0, 1]
    x = np.arange(len(sessions))
    ax.scatter(x, sessions["cross_probe_alignment"], color="#377EB8", label="observed")
    ax.scatter(
        x,
        sessions["permutation_cross_probe_alignment_mean"],
        color="#777777",
        label="trial-shuffled null",
    )
    for index, row in enumerate(sessions.itertuples()):
        ax.plot(
            [index, index],
            [row.permutation_cross_probe_alignment_025, row.permutation_cross_probe_alignment_975],
            color="#777777",
        )
    ax.set(
        xticks=x,
        xticklabels=sessions["site"],
        ylabel="cross-probe residual-phase alignment",
        title="B. Matched trials versus shuffled trials",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    grouped = conditions.groupby("temporal_frequency_hz").apply(
        lambda frame: pd.Series(
            {
                "observed": np.average(frame["cross_probe_alignment"], weights=frame["n_units"]),
                "null": np.average(
                    frame["permutation_cross_probe_alignment_mean"],
                    weights=frame["n_units"],
                ),
            }
        ),
    )
    ax.plot(grouped.index, grouped["observed"], "o-", color="#377EB8", label="observed")
    ax.plot(grouped.index, grouped["null"], "o--", color="#777777", label="shuffled")
    ax.set(
        xscale="log",
        xticks=[1, 2, 4, 8, 15],
        xticklabels=["1", "2", "4", "8", "15"],
        xlabel="temporal frequency (Hz)",
        ylabel="unit-weighted cross-probe alignment",
        title="C. Residual sharing by temporal frequency",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    order = list(BEHAVIOR_METRICS)
    raw = behavior_centers.loc[
        behavior_centers["analysis"].eq("within_condition")
        & behavior_centers["behavior_metric"].isin(order)
    ].set_index("behavior_metric").loc[order]
    controlled = behavior_centers.loc[
        behavior_centers["analysis"].eq("time_residualized")
        & behavior_centers["behavior_metric"].isin(order)
    ].set_index("behavior_metric").loc[order]
    positions = np.arange(len(order))
    ax.bar(
        positions - 0.18,
        raw["equal_session_association"] - raw["permutation_mean"],
        width=0.36,
        color="#E69F00",
        label="condition-controlled",
    )
    ax.bar(
        positions + 0.18,
        controlled["equal_session_association"] - controlled["permutation_mean"],
        width=0.36,
        color="#009E73",
        label="+ block-time control",
    )
    for position, metric in zip(positions, order):
        current = behavior_sessions.loc[
            behavior_sessions["behavior_metric"].eq(metric)
            & behavior_sessions["analysis"].eq("time_residualized")
        ]
        ax.scatter(
            np.full(len(current), position + 0.18),
            current["association"] - current["permutation_mean"],
            s=13,
            color="black",
            alpha=0.55,
        )
    ax.set(
        xticks=positions,
        xticklabels=["running", "pupil\narea", "pupil x", "pupil y"],
        ylabel="phase–behavior association above shuffle",
        title="D. Behavioral covariance and time control",
    )
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("MouseV2 residual grating phase: shared state and behavior")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    sessions: pd.DataFrame,
    shared_center: pd.DataFrame,
    behavior_centers: pd.DataFrame,
) -> None:
    center = shared_center.iloc[0]
    source = center.equal_session_source_corrected_coherence
    adjusted = center.equal_session_cross_probe_adjusted_coherence
    null_adjusted = center.permutation_adjusted_coherence_mean
    alignment = center.equal_session_cross_probe_alignment
    null_alignment = center.permutation_cross_probe_alignment_mean
    alignment_p = center.cross_probe_alignment_p
    significant_alignment = int((sessions["cross_probe_alignment_p"] <= 0.05).sum())
    behavior_lines = []
    for metric, label in BEHAVIOR_METRICS.items():
        raw = behavior_centers.loc[
            behavior_centers["behavior_metric"].eq(metric)
            & behavior_centers["analysis"].eq("within_condition")
        ].iloc[0]
        controlled = behavior_centers.loc[
            behavior_centers["behavior_metric"].eq(metric)
            & behavior_centers["analysis"].eq("time_residualized")
        ].iloc[0]
        behavior_lines.append(
            f"- {label}: condition-controlled {raw.equal_session_association:.3f} "
            f"versus shuffle {raw.permutation_mean:.3f} (p = {raw.permutation_p:.3f}); "
            f"after linear/quadratic block-time control {controlled.equal_session_association:.3f} "
            f"versus {controlled.permutation_mean:.3f} (p = {controlled.permutation_p:.3f}; "
            f"{controlled.sessions_above_null_mean}/{controlled.sessions} sessions above null mean)."
        )
    block_time = behavior_centers.loc[
        behavior_centers["behavior_metric"].eq(BLOCK_POSITION_METRIC)
    ].iloc[0]
    lines = [
        "# MouseV2 shared residual-phase and behavior bridge",
        "",
        "Starting phase was first removed using the frozen acquisition schedule. For each",
        "unit and trial, intrinsic unit phase was then estimated without that trial. The",
        "primary shared-state test predicts each unit only from units on the other three",
        "simultaneously recorded probes. Trial labels are independently shuffled within",
        "unit and condition for the null.",
        "",
        "## Cross-probe phase sharing",
        "",
        f"- Equal-session source-phase-adjusted coherence is {source:.3f}; removing the",
        f"  phase predicted from other probes changes it to {adjusted:.3f}, versus",
        f"  {null_adjusted:.3f} with shuffled trial correspondence.",
        f"- Cross-probe residual-phase alignment is {alignment:+.3f}, versus",
        f"  {null_alignment:+.3f} under the trial-shuffled null",
        f"  (equal-session aggregate p = {alignment_p:.4f}).",
        f"- {significant_alignment}/{len(sessions)} sessions individually exceed the",
        "  one-sided 0.05 permutation threshold.",
        "",
        "A positive matched-trial excess means that residual phase displacement is shared",
        "across physically separate probes; it cannot be produced by a target unit predicting",
        "itself. The phase adjustment remains a mechanism diagnostic, not a replacement for",
        "the released modulation index.",
        "",
        "## Behavioral covariance",
        "",
        *behavior_lines,
        f"- Grating-block position itself: observed {block_time.equal_session_association:.3f}, "
        f"shuffle {block_time.permutation_mean:.3f} "
        f"(aggregate permutation p = {block_time.permutation_p:.3f}).",
        "",
        "Behavior values are summarized during each 1-s grating and standardized within",
        f"orientation × TF condition. Eye summaries require at least {MIN_VALID_EYE_FRACTION:.0%} valid",
        "samples in the stimulus window. The primary sensitivity removes linear and quadratic",
        "grating-block time trends. Associations use only the angle of the probe-balanced",
        "population phase vector and are calibrated by within-condition behavior shuffles. These tests are",
        "descriptive across eight sessions and should not be interpreted from uncorrected",
        "per-session p-values alone.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_manifest(output_dir: Path) -> None:
    path = output_dir / "import_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    upstream = {
        record["site"]: record
        for record in json.loads(START_PHASE_MANIFEST.read_text(encoding="utf-8"))["inputs"]
    }
    for record in manifest.get("inputs", []):
        source = upstream[record["site"]]
        if int(record["bytes"]) != int(source["bytes"]):
            raise ValueError(f"{record['site']}: input byte-count drift")
        record["sha256"] = source["sha256"]
        record["sha256_source"] = str(START_PHASE_MANIFEST.relative_to(ROOT))
    manifest["outputs"] = []
    for name in (
        "README.md",
        "behavior_center_summary.csv",
        "behavior_permutation_association.csv",
        "behavior_session_association.csv",
        "condition_permutation_metrics.csv",
        "condition_shared_phase_summary.csv",
        "shared_phase_center_summary.csv",
        "session_shared_phase_summary.csv",
        "shared_phase_behavior_diagnostic.png",
        "trial_population_phase_behavior.csv",
        "unit_shared_phase_metrics.csv",
    ):
        output = output_dir / name
        if output.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": output.stat().st_size, "sha256": sha256(output)}
            )
    manifest["code"]["render_script_sha256"] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    sessions = pd.read_csv(output_dir / "session_shared_phase_summary.csv")
    conditions = pd.read_csv(output_dir / "condition_shared_phase_summary.csv")
    condition_permutations = pd.read_csv(
        output_dir / "condition_permutation_metrics.csv"
    )
    behavior_sessions = pd.read_csv(output_dir / "behavior_session_association.csv")
    behavior_permutations = pd.read_csv(
        output_dir / "behavior_permutation_association.csv"
    )
    behavior_centers = summarize_behavior_centers(
        behavior_sessions, behavior_permutations
    )
    shared_center = summarize_shared_center(
        sessions, conditions, condition_permutations
    )
    behavior_centers.to_csv(output_dir / "behavior_center_summary.csv", index=False)
    shared_center.to_csv(
        output_dir / "shared_phase_center_summary.csv", index=False
    )
    make_figure(
        sessions,
        conditions,
        behavior_sessions,
        behavior_centers,
        output_dir / "shared_phase_behavior_diagnostic.png",
    )
    write_report(output_dir, sessions, shared_center, behavior_centers)
    refresh_manifest(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        render_existing(output_dir)
        print(f"Rendered shared-phase behavior bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    stimulus_manifest_path = args.stimulus_manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schedule = phase_schedule(stimulus_manifest_path)
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    requested = set(args.sites) if args.sites else None
    sessions_config = [
        session
        for session in config["sessions"]
        if requested is None or str(session["site"]) in requested
    ]
    if requested is not None and {str(item["site"]) for item in sessions_config} != requested:
        raise ValueError("One or more requested sites are absent from the config")

    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    all_units = []
    all_conditions = []
    all_condition_permutations = []
    all_trials = []
    all_behavior_sessions = []
    all_behavior_permutations = []
    input_rows = []
    upstream_manifest = json.loads(START_PHASE_MANIFEST.read_text(encoding="utf-8"))
    upstream_inputs = {record["site"]: record for record in upstream_manifest["inputs"]}

    for session in sessions_config:
        site = str(session["site"])
        session_id = int(session["site_number"])
        offset = int(session["id_offset"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file() or nwb_path.stat().st_size != int(
            session["expected_nwb_bytes"]
        ):
            raise FileNotFoundError(f"Missing or size-mismatched MouseV2 NWB: {nwb_path}")
        print(f"[{site}] loading spikes, stimulus, running, and eye tracking", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        _, grating_table = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        conditions = phase_aware_conditions(grating_table, schedule)
        for condition in conditions:
            condition["condition_id"] = condition_id(tuple(condition["parameters"]))
        behavior = trial_behavior_table(grating_table, _read_behavior_signals(nwb_path))

        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")
        quality = quality.loc[quality["default_qc"].eq(True), ["unit_id"]]
        layer = pd.read_csv(ROOT / "data" / f"{site}_processed" / "layer_info.csv")
        layer = layer[["unit_id", "ecephys_structure_acronym"]].copy()
        layer["probe"] = layer["ecephys_structure_acronym"].str.rsplit("_").str[-1]
        quality = quality.merge(layer[["unit_id", "probe"]], on="unit_id", validate="one_to_one")
        if not set(quality["probe"]).issubset({"A", "B", "C", "E"}):
            raise ValueError(f"{site}: unexpected probe label")
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        grouped: dict[str, list[dict[str, object]]] = {}
        for index, row in enumerate(quality.itertuples(), start=1):
            local_id = int(row.unit_id) - offset
            selected, coefficients = trial_coefficients(
                extracted.spikes_by_unit[row_by_id[local_id]], conditions
            )
            key = str(selected["condition_id"])
            grouped.setdefault(key, []).append(
                {
                    "unit_id": int(row.unit_id),
                    "probe": str(row.probe),
                    "coefficients": coefficients,
                    "condition": selected,
                }
            )
            if index % 400 == 0:
                print(f"[{site}] decomposed {index}/{len(quality)} units", flush=True)

        session_trials = []
        for condition_index, (key, records) in enumerate(sorted(grouped.items())):
            probes = np.array([record["probe"] for record in records])
            if len(np.unique(probes)) < 2:
                continue
            coefficients = np.stack([record["coefficients"] for record in records])
            result = shared_phase_condition(
                coefficients,
                probes,
                permutations=args.phase_permutations,
                seed=int(
                    np.random.SeedSequence(
                        [args.permutation_seed, session_id, condition_index]
                    ).generate_state(1)[0]
                ),
            )
            condition = records[0]["condition"]
            parameters = tuple(condition["parameters"])
            source_coherence = np.asarray(result["source_coherence"])
            cross_adjusted = np.asarray(result["cross_probe_adjusted_coherence"])
            all_adjusted = np.asarray(result["all_other_adjusted_coherence"])
            unit_cross_alignment = np.asarray(result["unit_cross_probe_alignment"])
            unit_all_alignment = np.asarray(result["unit_all_other_alignment"])
            for unit_index, record in enumerate(records):
                all_units.append(
                    {
                        "site": site,
                        "session_id": session_id,
                        "condition_id": key,
                        "unit_id": record["unit_id"],
                        "probe": record["probe"],
                        "preferred_orientation_deg": parameters[0],
                        "preferred_tf_hz": parameters[1],
                        "condition_units": len(records),
                        "condition_probes": len(np.unique(probes)),
                        "source_corrected_weighted_phase_coherence": source_coherence[unit_index],
                        "all_other_adjusted_weighted_phase_coherence": all_adjusted[unit_index],
                        "cross_probe_adjusted_weighted_phase_coherence": cross_adjusted[unit_index],
                        "all_other_alignment": unit_all_alignment[unit_index],
                        "cross_probe_alignment": unit_cross_alignment[unit_index],
                        "cross_probe_coherence_gain": cross_adjusted[unit_index]
                        - source_coherence[unit_index],
                    }
                )
            permutation_table = result["permutations"].copy()
            permutation_table.insert(0, "condition_id", key)
            permutation_table.insert(0, "session_id", session_id)
            permutation_table.insert(0, "site", site)
            all_condition_permutations.append(permutation_table)
            cross_null = permutation_table["cross_probe_alignment"]
            adjusted_null = permutation_table["mean_cross_probe_adjusted_coherence"]
            condition_row = {
                "site": site,
                "session_id": session_id,
                "condition_id": key,
                "orientation_deg": parameters[0],
                "temporal_frequency_hz": parameters[1],
                "n_units": len(records),
                "n_probes": len(np.unique(probes)),
                "all_other_alignment": result["all_other_alignment"],
                "cross_probe_alignment": result["cross_probe_alignment"],
                "permutation_cross_probe_alignment_mean": cross_null.mean(),
                "permutation_cross_probe_alignment_025": cross_null.quantile(0.025),
                "permutation_cross_probe_alignment_975": cross_null.quantile(0.975),
                "cross_probe_alignment_p": (1 + np.sum(cross_null >= result["cross_probe_alignment"]))
                / (len(cross_null) + 1),
                "mean_source_corrected_coherence": float(np.nanmean(source_coherence)),
                "mean_all_other_adjusted_coherence": float(np.nanmean(all_adjusted)),
                "mean_cross_probe_adjusted_coherence": float(np.nanmean(cross_adjusted)),
                "mean_cross_probe_coherence_gain": float(
                    np.nanmean(cross_adjusted - source_coherence)
                ),
                "permutation_cross_probe_adjusted_mean": adjusted_null.mean(),
                "cross_probe_adjusted_p": (
                    1
                    + np.sum(
                        adjusted_null >= float(np.nanmean(cross_adjusted))
                    )
                )
                / (len(adjusted_null) + 1),
            }
            all_conditions.append(condition_row)

            trial_consensus = np.asarray(result["trial_consensus"])
            for trial_index, ordinal in enumerate(condition["ordinals"]):
                value = trial_consensus[trial_index]
                session_trials.append(
                    {
                        "site": site,
                        "session_id": session_id,
                        "condition_id": key,
                        "presentation_ordinal": int(ordinal),
                        "orientation_deg": parameters[0],
                        "temporal_frequency_hz": parameters[1],
                        "condition_units": len(records),
                        "condition_probes": len(np.unique(probes)),
                        "population_phase_real": float(np.real(value)),
                        "population_phase_imag": float(np.imag(value)),
                        "population_phase_magnitude": float(np.abs(value)),
                        "population_phase_cycles": float(np.angle(value) / (2 * np.pi)),
                        "valid_phase_units": int(result["trial_valid_units"][trial_index]),
                    }
                )
        session_trials = pd.DataFrame(session_trials).merge(
            behavior,
            on=["condition_id", "presentation_ordinal"],
            how="left",
            validate="one_to_one",
        )
        if len(session_trials) != 300:
            raise ValueError(f"{site}: expected 300 population-phase trials")
        behavior_analyses = []
        for metric, label in BEHAVIOR_METRICS.items():
            residual_column = f"{metric}_time_residualized"
            session_trials[residual_column] = residualize_metric_against_block_time(
                session_trials, metric
            )
            behavior_analyses.extend(
                [
                    (metric, label, "within_condition", metric),
                    (metric, label, "time_residualized", residual_column),
                ]
            )
        behavior_analyses.append(
            (
                BLOCK_POSITION_METRIC,
                "grating-block position",
                "within_condition",
                BLOCK_POSITION_METRIC,
            )
        )
        all_trials.append(session_trials)

        for metric_index, (metric, label, analysis, input_column) in enumerate(
            behavior_analyses
        ):
            observed, null, n_presentations = complex_phase_association(
                session_trials,
                input_column,
                permutations=args.behavior_permutations,
                seed=int(
                    np.random.SeedSequence(
                        [args.permutation_seed, session_id, 100 + metric_index]
                    ).generate_state(1)[0]
                ),
            )
            all_behavior_sessions.append(
                {
                    "site": site,
                    "session_id": session_id,
                    "behavior_metric": metric,
                    "analysis": analysis,
                    "label": label,
                    "presentations": n_presentations,
                    "association": observed,
                    "permutation_mean": float(np.nanmean(null)),
                    "permutation_025": float(np.nanquantile(null, 0.025)),
                    "permutation_975": float(np.nanquantile(null, 0.975)),
                    "permutation_p": (1 + np.sum(null >= observed)) / (len(null) + 1),
                }
            )
            for permutation, value in enumerate(null):
                all_behavior_permutations.append(
                    {
                        "site": site,
                        "session_id": session_id,
                        "behavior_metric": metric,
                        "analysis": analysis,
                        "permutation": permutation,
                        "association": value,
                    }
                )
        source = upstream_inputs[site]
        input_rows.append(
            {
                "site": site,
                "path": str(nwb_path),
                "bytes": nwb_path.stat().st_size,
                "sha256": source["sha256"],
                "sha256_source": str(START_PHASE_MANIFEST.relative_to(ROOT)),
            }
        )

    units = pd.DataFrame(all_units)
    conditions = pd.DataFrame(all_conditions)
    condition_permutations = pd.concat(all_condition_permutations, ignore_index=True)
    trials = pd.concat(all_trials, ignore_index=True)
    behavior_sessions = pd.DataFrame(all_behavior_sessions)
    behavior_permutations = pd.DataFrame(all_behavior_permutations)
    shared_sessions = summarize_shared_sessions(conditions, condition_permutations)
    shared_center = summarize_shared_center(
        shared_sessions, conditions, condition_permutations
    )
    behavior_centers = summarize_behavior_centers(
        behavior_sessions, behavior_permutations
    )

    units.to_csv(output_dir / "unit_shared_phase_metrics.csv", index=False)
    conditions.to_csv(output_dir / "condition_shared_phase_summary.csv", index=False)
    condition_permutations.to_csv(
        output_dir / "condition_permutation_metrics.csv", index=False
    )
    shared_sessions.to_csv(output_dir / "session_shared_phase_summary.csv", index=False)
    shared_center.to_csv(
        output_dir / "shared_phase_center_summary.csv", index=False
    )
    trials.to_csv(output_dir / "trial_population_phase_behavior.csv", index=False)
    behavior_sessions.to_csv(
        output_dir / "behavior_session_association.csv", index=False
    )
    behavior_permutations.to_csv(
        output_dir / "behavior_permutation_association.csv", index=False
    )
    behavior_centers.to_csv(output_dir / "behavior_center_summary.csv", index=False)
    write_report(output_dir, shared_sessions, shared_center, behavior_centers)
    if not args.skip_figure:
        make_figure(
            shared_sessions,
            conditions,
            behavior_sessions,
            behavior_centers,
            output_dir / "shared_phase_behavior_diagnostic.png",
        )

    manifest = {
        "schema_version": 1,
        "phase_permutations": args.phase_permutations,
        "behavior_permutations": args.behavior_permutations,
        "permutation_seed": args.permutation_seed,
        "population_phase_method": {
            "starting_phase_removed": True,
            "intrinsic_unit_phase": "leave-one-trial-out",
            "primary_prediction": "other-probe consensus",
            "trial_consensus": "probe-balanced residual phase",
            "null": "independent within-unit, within-condition trial shuffle",
        },
        "behavior_method": {
            "window_s": [0.0, 1.0],
            "condition_standardization": True,
            "association": "phase-angle-only normalized complex association",
            "time_control": "condition-centered behavior residualized on linear and quadratic presentation ordinal",
            "null": "within-condition behavior shuffle",
            "metrics": BEHAVIOR_METRICS,
            "minimum_valid_eye_fraction": MIN_VALID_EYE_FRACTION,
        },
        "inputs": input_rows,
        "upstream_source_provenance": upstream_manifest["source_provenance"],
        "code": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "render_script_sha256": sha256(Path(__file__).resolve()),
            "start_phase_script_sha256": sha256(
                ROOT / "scripts" / "mousev2_grating_start_phase_bridge.py"
            ),
            "config_sha256": sha256(config_path),
            "stimulus_manifest_sha256": sha256(stimulus_manifest_path),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir)
    print(f"Shared-phase behavior bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
