#!/usr/bin/env python3
"""Freeze provenance and trace concrete grating cases for the V1 gap audit.

This is an exploratory checkpoint, not a population analysis.  Cases are
selected with explicit roles from frozen upstream tables, then traced from raw
NWB spike/event data through trial binning and several alignment constructions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import (  # noqa: E402
    _bin_trial_spike_counts,
    f1_f0_from_trial_counts,
)
from scripts.extract_allen_v1_bridge import (  # noqa: E402
    _preferred_condition,
    _subsample_conditions,
    condition_starts,
)
from scripts.mousev2_grating_corrected_welch_bridge import (  # noqa: E402
    corrected_welch_metrics,
    target_coefficients,
    welch_spectral_metrics,
)
from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    phase_aware_conditions,
    phase_schedule,
    preferred_condition_counts,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "v1_systemic_gap_audit_v1"
MOUSE_UNITS = (
    ROOT
    / "data/imports/mousev2_grating_corrected_welch_bridge_v1"
    / "unit_corrected_welch_metrics.csv"
)
MOUSE_MANIFEST = MOUSE_UNITS.with_name("import_manifest.json")
ALLEN_TRIALS = (
    ROOT / "data/imports/allen_v1_raw_bridge_v2/trial_subsample_metrics.csv"
)
ALLEN_MANIFEST = ALLEN_TRIALS.with_name("import_manifest.json")
MOUSE_CONFIG = ROOT / "config/figure3_mousev2.json"
ALLEN_CONFIG = ROOT / "config/allen_v1_bridge.json"
STIMULUS_MANIFEST = ROOT / "config/mousev2_stimulus_manifest.json"
FRAME_MS = 1000.0 / 60.0


@dataclass
class TargetedExtract:
    spikes_by_id: dict[int, np.ndarray]
    interval_table: pd.DataFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-table-only", action="store_true")
    return parser.parse_args()


def read_targeted_nwb(
    nwb_path: Path, *, unit_ids: list[int], interval_name: str
) -> TargetedExtract:
    """Read selected ragged spike slices and one interval table without expansion."""
    import h5py

    with h5py.File(nwb_path, "r") as handle:
        ids = np.asarray(handle["/units/id"][:], dtype=np.int64)
        ends = np.asarray(handle["/units/spike_times_index"][:], dtype=np.int64)
        row_by_id = {int(unit_id): row for row, unit_id in enumerate(ids)}
        spikes_by_id: dict[int, np.ndarray] = {}
        spike_data = handle["/units/spike_times"]
        for unit_id in unit_ids:
            if int(unit_id) not in row_by_id:
                raise KeyError(f"Unit {unit_id} is absent from {nwb_path}")
            row = row_by_id[int(unit_id)]
            start = 0 if row == 0 else int(ends[row - 1])
            stop = int(ends[row])
            spikes_by_id[int(unit_id)] = np.sort(
                np.asarray(spike_data[start:stop], dtype=float)
            )

        group = handle[f"/intervals/{interval_name}"]
        columns: dict[str, np.ndarray] = {}
        skip = {"timeseries", "timeseries_index"}
        for name, dataset in group.items():
            if name in skip or not isinstance(dataset, h5py.Dataset):
                continue
            raw = dataset[:]
            if raw.dtype.kind in "SUO":
                raw = np.asarray(
                    [value.decode() if isinstance(value, bytes) else str(value) for value in raw]
                )
                if name in {"orientation", "temporal_frequency", "spatial_frequency", "contrast"}:
                    raw = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy()
            columns[name] = raw
        table = pd.DataFrame(columns)
    return TargetedExtract(spikes_by_id=spikes_by_id, interval_table=table)


def closest_index(values: pd.Series, target: float) -> int:
    finite = pd.to_numeric(values, errors="coerce")
    return int((finite - target).abs().idxmin())


def select_mouse_cases(units: pd.DataFrame) -> pd.DataFrame:
    """Select auditable roles deterministically; outcome-based roles are labeled."""
    affected = units.loc[units["source_phase_expected_to_vary"].eq(True)].copy()
    stable = units.loc[units["preferred_tf_hz"].eq(4.0)].copy()
    rows: list[dict[str, object]] = []

    median_gain = float(affected["source_log10_mod_idx_gain"].median())
    typical = affected.loc[closest_index(affected["source_log10_mod_idx_gain"], median_gain)]
    rows.append(
        dict(
            dataset="MouseV2",
            session_id=int(typical.session_id),
            site=str(typical.site),
            unit_id=int(typical.unit_id),
            selection_role="typical_affected",
            selection_basis="closest to median source-phase gain among affected units",
            criterion_value=float(typical.source_log10_mod_idx_gain),
            selection_timing="post_hoc_outcome_based",
        )
    )

    dissociation = affected.loc[
        closest_index(affected["source_log10_mod_idx_gain"], 0.0)
    ]
    rows.append(
        dict(
            dataset="MouseV2",
            session_id=int(dissociation.session_id),
            site=str(dissociation.site),
            unit_id=int(dissociation.unit_id),
            selection_role="prediction_without_response",
            selection_basis="affected phase schedule but source-phase gain closest to zero",
            criterion_value=float(dissociation.source_log10_mod_idx_gain),
            selection_timing="post_hoc_outcome_based",
        )
    )

    high = affected.loc[affected["source_log10_mod_idx_gain"].idxmax()]
    rows.append(
        dict(
            dataset="MouseV2",
            session_id=int(high.session_id),
            site=str(high.site),
            unit_id=int(high.unit_id),
            selection_role="largest_observed_gain",
            selection_basis="largest source-phase log10 modulation-index gain",
            criterion_value=float(high.source_log10_mod_idx_gain),
            selection_timing="post_hoc_outcome_based",
        )
    )

    stable_target = float(stable["log10_raw_mod_idx"].median())
    control = stable.loc[closest_index(stable["log10_raw_mod_idx"], stable_target)]
    rows.append(
        dict(
            dataset="MouseV2",
            session_id=int(control.session_id),
            site=str(control.site),
            unit_id=int(control.unit_id),
            selection_role="phase_stable_negative_control",
            selection_basis="4-Hz unit closest to the 4-Hz median raw score",
            criterion_value=float(control.log10_raw_mod_idx),
            selection_timing="predeclared_control_role",
        )
    )
    return pd.DataFrame(rows)


def select_allen_cases(trials: pd.DataFrame) -> pd.DataFrame:
    selected = trials.loc[
        trials["common_qc"].eq(True)
        & trials["selection_role"].eq("representative")
        & trials["subsample"].eq(0)
    ].copy()
    rows = []
    for cohort, group in selected.groupby("cohort", sort=True):
        values = np.log10(pd.to_numeric(group["common_mod_idx_dg"], errors="coerce"))
        target = float(values.median())
        row = group.loc[closest_index(values, target)]
        rows.append(
            dict(
                dataset=str(cohort),
                session_id=int(row.session_id),
                site="",
                unit_id=int(row.ecephys_unit_id),
                selection_role="representative_session_typical_unit",
                selection_basis="common-QC unit closest to within-session median harmonized score",
                criterion_value=float(np.log10(row.common_mod_idx_dg)),
                selection_timing="post_hoc_within_preselected_session",
            )
        )
    return pd.DataFrame(rows)


def circular_fractional_shift(rows: np.ndarray, shifts_samples: np.ndarray) -> np.ndarray:
    """Circularly delay each trial by a possibly fractional number of samples."""
    values = np.asarray(rows, dtype=float)
    shifts = np.asarray(shifts_samples, dtype=float)
    frequencies = np.fft.fftfreq(values.shape[1])
    phase = np.exp(-2j * np.pi * shifts[:, None] * frequencies[None, :])
    return np.fft.ifft(np.fft.fft(values, axis=1) * phase, axis=1).real


def spectral_row(
    *, case: pd.Series, view: str, psth: np.ndarray, tf_hz: float, f1_f0: float
) -> dict[str, object]:
    metrics = welch_spectral_metrics(psth, tf_hz)
    return {
        "dataset": case.dataset,
        "session_id": int(case.session_id),
        "site": case.site,
        "unit_id": int(case.unit_id),
        "selection_role": case.selection_role,
        "view": view,
        "preferred_tf_hz": tf_hz,
        "f1_f0": f1_f0,
        "coherent_carrier_amplitude": float(
            np.abs(target_coefficients(np.asarray(psth)[None, :], tf_hz)[0])
        ),
        **{key: float(value) for key, value in metrics.items()},
    }


def trace_mouse_case(
    case: pd.Series,
    spikes: np.ndarray,
    conditions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    parameters, counts, phases = preferred_condition_counts(spikes, conditions)
    tf_hz = float(parameters[1])
    raw_psth = counts.mean(axis=0)
    corrected = corrected_welch_metrics(
        counts, tf_hz, phases, permutations=20, seed=20260824 + int(case.unit_id)
    )
    phase_delay_samples = phases / tf_hz * 1000.0
    fully_aligned_counts = circular_fractional_shift(counts, phase_delay_samples)
    full_psth = fully_aligned_counts.mean(axis=0)
    f1_raw = f1_f0_from_trial_counts(counts, tf_hz, 1.0)
    f1_full = f1_f0_from_trial_counts(fully_aligned_counts, tf_hz, 1.0)

    # A bounded timestamp sensitivity: deterministic alternating +/- one frame.
    frame_offsets_ms = np.resize(np.array([-FRAME_MS, FRAME_MS]), len(counts))
    jitter_counts = circular_fractional_shift(counts, frame_offsets_ms)
    jitter_psth = jitter_counts.mean(axis=0)
    rows = [
        spectral_row(case=case, view="raw", psth=raw_psth, tf_hz=tf_hz, f1_f0=f1_raw),
        spectral_row(
            case=case,
            view="full_trial_source_phase_alignment",
            psth=full_psth,
            tf_hz=tf_hz,
            f1_f0=f1_full,
        ),
        spectral_row(
            case=case,
            view="alternating_plus_minus_one_frame",
            psth=jitter_psth,
            tf_hz=tf_hz,
            f1_f0=f1_f0_from_trial_counts(jitter_counts, tf_hz, 1.0),
        ),
    ]
    rows.append(
        {
            **rows[0],
            "view": "carrier_only_source_phase_alignment",
            "mod_idx": float(corrected["source_corrected_mod_idx"]),
            "target_psd": float(corrected["source_corrected_target_psd"]),
            "off_target_mean_psd": float(
                corrected["source_corrected_off_target_mean_psd"]
            ),
            "psd_mean": float(corrected["source_corrected_psd_mean"]),
            "psd_sd": float(corrected["source_corrected_psd_sd"]),
            "target_to_offtarget_psd": float(
                corrected["source_corrected_target_to_offtarget_psd"]
            ),
            "coherent_carrier_amplitude": float(
                corrected["source_corrected_coherent_f1_hz"]
            ),
        }
    )
    trial_table = pd.DataFrame(
        {
            "dataset": case.dataset,
            "session_id": int(case.session_id),
            "site": case.site,
            "unit_id": int(case.unit_id),
            "selection_role": case.selection_role,
            "trial": np.arange(len(counts)),
            "source_start_phase_cycles": phases,
            "source_phase_delay_ms": phase_delay_samples,
            "spike_count": counts.sum(axis=1),
            "carrier_real": np.real(
                np.fft.fft(counts, axis=1)[:, int(round(tf_hz))]
            ),
            "carrier_imag": np.imag(
                np.fft.fft(counts, axis=1)[:, int(round(tf_hz))]
            ),
        }
    )
    psth_table = pd.DataFrame(
        {
            "dataset": case.dataset,
            "session_id": int(case.session_id),
            "site": case.site,
            "unit_id": int(case.unit_id),
            "selection_role": case.selection_role,
            "time_ms": np.arange(1000),
            "raw_psth": raw_psth,
            "full_trial_source_phase_alignment_psth": full_psth,
            "alternating_plus_minus_one_frame_psth": jitter_psth,
        }
    )
    return rows, trial_table, psth_table


def trace_allen_case(
    case: pd.Series, spikes: np.ndarray, table: pd.DataFrame, support: dict[str, object]
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    conditions = condition_starts(table, common_support=support)
    conditions = _subsample_conditions(
        conditions,
        trials_per_condition=int(support["trials_per_condition"]),
        seed=int(case.session_id),
    )
    preferred = _preferred_condition(spikes, conditions, duration_s=1.0)
    parameters, starts = conditions[preferred]
    counts = _bin_trial_spike_counts(spikes, starts, duration_ms=1000)
    tf_hz = float(parameters[1])
    psth = counts.mean(axis=0)
    row = spectral_row(
        case=case,
        view="raw_harmonized",
        psth=psth,
        tf_hz=tf_hz,
        f1_f0=f1_f0_from_trial_counts(counts, tf_hz, 1.0),
    )
    psth_table = pd.DataFrame(
        {
            "dataset": case.dataset,
            "session_id": int(case.session_id),
            "site": "",
            "unit_id": int(case.unit_id),
            "selection_role": case.selection_role,
            "time_ms": np.arange(1000),
            "raw_psth": psth,
        }
    )
    return [row], psth_table


def write_frozen_manifest(output: Path, cases: pd.DataFrame) -> None:
    mouse_manifest = json.loads(MOUSE_MANIFEST.read_text())
    allen_manifest = json.loads(ALLEN_MANIFEST.read_text())
    files = [
        Path(__file__).resolve(),
        ROOT / "common/drifting_gratings.py",
        ROOT / "scripts/mousev2_grating_corrected_welch_bridge.py",
        ROOT / "scripts/mousev2_grating_start_phase_bridge.py",
        ROOT / "scripts/extract_allen_v1_bridge.py",
        MOUSE_CONFIG,
        ALLEN_CONFIG,
        STIMULUS_MANIFEST,
        MOUSE_UNITS,
        ALLEN_TRIALS,
    ]
    manifest = {
        "schema_version": 1,
        "status": "exploratory_concrete_case_checkpoint",
        "question": "Can systemic event timing or estimator construction explain the residual MouseV2-Allen V1 grating gap?",
        "frozen_support": {
            "duration_s": 1.0,
            "trials_per_condition": 15,
            "spatial_frequency_cpd": 0.04,
            "contrast": 0.8,
            "orientations_deg": [0.0, 45.0, 90.0, 135.0],
            "temporal_frequencies_hz": [1.0, 2.0, 4.0, 8.0, 15.0],
        },
        "hypotheses": [
            {
                "name": "whole_trial_phase_misalignment",
                "prediction": "full-trial phase alignment changes off-carrier structure and the Welch score beyond carrier-only correction",
                "against": "full-trial and carrier-only corrections agree closely in affected and control cases",
            },
            {
                "name": "one_frame_event_jitter",
                "prediction": "a +/- one-frame trial pattern materially moves the Welch score while leaving single-trial F1/F0 comparatively stable",
                "against": "the perturbation is negligible relative to the observed residual",
            },
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "mouse_inputs": mouse_manifest["inputs"],
        "allen_inputs": allen_manifest["inputs"],
        "case_count": len(cases),
        "selection_table": "case_selection.csv",
    }
    (output / "frozen_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def render(metrics: pd.DataFrame, psths: pd.DataFrame, output: Path) -> None:
    mouse = metrics.loc[metrics.dataset.eq("MouseV2")].copy()
    roles = list(mouse.selection_role.drop_duplicates())
    fig, axes = plt.subplots(len(roles), 2, figsize=(12, 3.1 * len(roles)), squeeze=False)
    for row_index, role in enumerate(roles):
        group = mouse.loc[mouse.selection_role.eq(role)]
        case_psth = psths.loc[
            psths.dataset.eq("MouseV2") & psths.selection_role.eq(role)
        ]
        axes[row_index, 0].plot(case_psth.time_ms, case_psth.raw_psth, label="raw", lw=1)
        axes[row_index, 0].plot(
            case_psth.time_ms,
            case_psth.full_trial_source_phase_alignment_psth,
            label="full-trial aligned",
            lw=1,
        )
        axes[row_index, 0].set(xlabel="time from logged onset (ms)", ylabel="spikes / 1-ms bin", title=role)
        views = ["raw", "carrier_only_source_phase_alignment", "full_trial_source_phase_alignment", "alternating_plus_minus_one_frame"]
        plotted = group.set_index("view").reindex(views)
        axes[row_index, 1].bar(np.arange(len(views)), np.log10(plotted.mod_idx), color=["#777777", "#4C78A8", "#F58518", "#E45756"])
        axes[row_index, 1].set_xticks(np.arange(len(views)), ["raw", "carrier", "full trial", "+/- frame"], rotation=20)
        axes[row_index, 1].set(ylabel="log10 Welch modulation index")
        if row_index == 0:
            axes[row_index, 0].legend(frameon=False)
    fig.suptitle("Concrete MouseV2 grating traces: alignment and frame-timing sensitivity")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def refresh_output_manifest(output: Path) -> None:
    path = output / "frozen_comparison_manifest.json"
    manifest = json.loads(path.read_text())
    names = [
        "case_selection.csv",
        "case_metric_trace.csv",
        "mouse_trial_trace.csv",
        "case_psth_trace.csv",
        "grating_initial_case_traces.png",
        "INITIAL_GRATING_CHECKPOINT.md",
    ]
    manifest["outputs"] = [
        {"path": name, "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
        for name in names
        if (output / name).is_file()
    ]
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mouse_units = pd.read_csv(MOUSE_UNITS)
    allen_trials = pd.read_csv(ALLEN_TRIALS)
    cases = pd.concat(
        [select_mouse_cases(mouse_units), select_allen_cases(allen_trials)],
        ignore_index=True,
    )
    cases.to_csv(output / "case_selection.csv", index=False)
    write_frozen_manifest(output, cases)
    if args.case_table_only:
        refresh_output_manifest(output)
        print(f"Wrote frozen manifest and case table to {output}")
        return

    mouse_config = json.loads(MOUSE_CONFIG.read_text())
    mouse_by_site = {str(item["site"]): item for item in mouse_config["sessions"]}
    schedule = phase_schedule(STIMULUS_MANIFEST)
    metric_rows: list[dict[str, object]] = []
    trial_frames = []
    psth_frames = []
    for site, group in cases.loc[cases.dataset.eq("MouseV2")].groupby("site", sort=True):
        session = mouse_by_site[site]
        nwb_path = Path(mouse_config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        offset = int(session["id_offset"])
        local_ids = [int(value) - offset for value in group.unit_id]
        extracted = read_targeted_nwb(
            nwb_path,
            unit_ids=local_ids,
            interval_name="drifting_gratings_field_block_presentations",
        )
        conditions = phase_aware_conditions(extracted.interval_table, schedule)
        for _, case in group.iterrows():
            local_id = int(case.unit_id) - offset
            rows, trials, psths = trace_mouse_case(
                case, extracted.spikes_by_id[local_id], conditions
            )
            metric_rows.extend(rows)
            trial_frames.append(trials)
            psth_frames.append(psths)

    allen_config = json.loads(ALLEN_CONFIG.read_text())
    allen_assets = {int(item["session_id"]): item for item in allen_config["assets"]}
    for session_id, group in cases.loc[cases.dataset.ne("MouseV2")].groupby("session_id"):
        asset = allen_assets[int(session_id)]
        nwb_path = Path(allen_config["download_root"]) / asset["relative_path"]
        unit_ids = [int(value) for value in group.unit_id]
        extracted = read_targeted_nwb(
            nwb_path, unit_ids=unit_ids, interval_name=asset["grating_table"]
        )
        for _, case in group.iterrows():
            rows, psths = trace_allen_case(
                case,
                extracted.spikes_by_id[int(case.unit_id)],
                extracted.interval_table,
                allen_config["common_support"],
            )
            metric_rows.extend(rows)
            psth_frames.append(psths)

    metrics = pd.DataFrame(metric_rows)
    trials = pd.concat(trial_frames, ignore_index=True)
    psths = pd.concat(psth_frames, ignore_index=True)
    metrics.to_csv(output / "case_metric_trace.csv", index=False)
    trials.to_csv(output / "mouse_trial_trace.csv", index=False)
    psths.to_csv(output / "case_psth_trace.csv", index=False)
    render(metrics, psths, output / "grating_initial_case_traces.png")
    refresh_output_manifest(output)
    print(f"Wrote systemic V1 gap checkpoint to {output}")


if __name__ == "__main__":
    main()
