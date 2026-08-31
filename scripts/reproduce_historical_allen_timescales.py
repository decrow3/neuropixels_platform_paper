#!/usr/bin/env python3
"""Reproduce historical Allen V1 response-timescale values from raw flashes.

The raw audit uses the checksum-pinned sessions in ``config/allen_v1_bridge.json``.
It also regenerates the full historical session/cohort centers from the frozen
``data/unit_table.csv`` so the raw numerical audit and the reported summaries
remain distinct and traceable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.flashes import (  # noqa: E402
    bin_trial_spike_counts,
    fit_response_timescale,
    prepare_flash_presentations,
)


DEFAULT_CONFIG = ROOT / "config" / "allen_v1_bridge.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "allen_timescale_reproduction_v1"
COHORTS = {
    "brain_observatory_1.1": "Allen Brain Observatory 1.1",
    "functional_connectivity": "Allen Functional Connectivity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def historical_valid(table: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(table["timescale_ac"], errors="coerce").between(1, 300)
        & pd.to_numeric(table["spike_count_ac"], errors="coerce").gt(50)
        & pd.to_numeric(table["err_ac"], errors="coerce").lt(20)
    )


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


def read_invalid_intervals(path: Path) -> np.ndarray:
    """Read start/stop pairs without requiring ragged ``tags`` decoding."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle.get("intervals/invalid_times")
        if group is None or "start_time" not in group or "stop_time" not in group:
            return np.empty((0, 2), dtype=float)
        return np.column_stack(
            [
                np.asarray(group["start_time"], dtype=float),
                np.asarray(group["stop_time"], dtype=float),
            ]
        )


def exclude_invalid_spikes(spikes: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Match AllenSDK's removal of spikes falling in declared invalid times."""
    result = np.asarray(spikes, dtype=float)
    if not len(intervals):
        return result
    keep = np.ones(len(result), dtype=bool)
    for start, stop in intervals:
        keep &= ~((result >= start) & (result < stop))
    return result[keep]


def full_historical_summary(release: pd.DataFrame) -> pd.DataFrame:
    selected = release.loc[
        release["ecephys_structure_acronym"].eq("VISp")
        & release["session_type"].isin(COHORTS)
        & release["timescale_ac"].notna()
    ].copy()
    selected["valid_timescale"] = historical_valid(selected)
    selected["valid_value"] = selected["timescale_ac"].where(
        selected["valid_timescale"]
    )
    sessions = (
        selected.groupby(["session_type", "ecephys_session_id"], sort=True)
        .agg(
            historical_fit_units=("ecephys_unit_id", "size"),
            valid_units=("valid_timescale", "sum"),
            mean_valid_timescale_ms=("valid_value", "mean"),
            median_valid_timescale_ms=("valid_value", "median"),
        )
        .reset_index()
    )
    sessions["cohort"] = sessions["session_type"].map(COHORTS)
    return sessions


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    release = pd.read_csv(ROOT / "data" / "unit_table.csv", low_memory=False)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    from generate_retinotopic_csvs import read_nwb_tables

    rows: list[dict[str, object]] = []
    for asset in config["assets"]:
        session_id = int(asset["session_id"])
        path = Path(config["download_root"]) / asset["relative_path"]
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched input: {path}")
        print(f"[{session_id}] reading raw NWB", flush=True)
        extracted = read_nwb_tables(str(path))
        flashes = prepare_flash_presentations(
            extracted.intervals_tables["flashes_presentations"]
        )
        if len(flashes) != int(asset["expected_flash_presentations"]):
            raise ValueError(f"{session_id}: unexpected flash count")
        starts = flashes["start_time"].to_numpy(dtype=float)
        invalid_intervals = read_invalid_intervals(path)
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        historical = release.loc[
            release["ecephys_session_id"].eq(session_id)
            & release["ecephys_structure_acronym"].eq("VISp")
            & release["timescale_ac"].notna()
        ]
        for old in historical.itertuples(index=False):
            unit_id = int(old.ecephys_unit_id)
            if unit_id not in row_by_id:
                raise ValueError(f"{session_id}: historical unit {unit_id} absent")
            new = reproduce_unit(
                exclude_invalid_spikes(
                    extracted.spikes_by_unit[row_by_id[unit_id]], invalid_intervals
                ),
                starts,
            )
            rows.append(
                {
                    "session_id": session_id,
                    "session_type": asset["session_type"],
                    "selection_role": asset["selection_role"],
                    "ecephys_unit_id": unit_id,
                    "flash_trials": len(flashes),
                    "historical_timescale_ac": old.timescale_ac,
                    "historical_err_ac": old.err_ac,
                    "historical_spike_count_ac": old.spike_count_ac,
                    **new,
                }
            )

    audit = pd.DataFrame(rows)
    audit["timescale_difference_ms"] = (
        audit["reproduced_timescale_ac"] - audit["historical_timescale_ac"]
    )
    audit["error_difference_ms"] = (
        audit["reproduced_err_ac"] - audit["historical_err_ac"]
    )
    audit["spike_count_difference"] = (
        audit["reproduced_spike_count_ac"] - audit["historical_spike_count_ac"]
    )
    old_valid = historical_valid(
        audit.rename(
            columns={
                "historical_timescale_ac": "timescale_ac",
                "historical_err_ac": "err_ac",
                "historical_spike_count_ac": "spike_count_ac",
            }
        )
    )
    new_valid = historical_valid(
        audit.rename(
            columns={
                "reproduced_timescale_ac": "timescale_ac",
                "reproduced_err_ac": "err_ac",
                "reproduced_spike_count_ac": "spike_count_ac",
            }
        )
    )
    audit["historical_valid"] = old_valid
    audit["reproduced_valid"] = new_valid
    audit.to_csv(output / "raw_unit_validation.csv", index=False)

    raw_summary = (
        audit.assign(
            historical_valid_value=audit["historical_timescale_ac"].where(old_valid),
            reproduced_valid_value=audit["reproduced_timescale_ac"].where(new_valid),
        ).groupby(["session_type", "session_id", "selection_role"])
        .agg(
            units=("ecephys_unit_id", "size"),
            exact_spike_counts=("spike_count_difference", lambda x: int((x == 0).sum())),
            max_abs_timescale_difference_ms=("timescale_difference_ms", lambda x: np.nanmax(np.abs(x))),
            max_abs_error_difference_ms=("error_difference_ms", lambda x: np.nanmax(np.abs(x))),
            validity_disagreements=("historical_valid", lambda x: 0),
            historical_mean_valid_timescale_ms=("historical_valid_value", "mean"),
            reproduced_mean_valid_timescale_ms=("reproduced_valid_value", "mean"),
        )
        .reset_index()
    )
    raw_summary["mean_valid_timescale_difference_ms"] = (
        raw_summary["reproduced_mean_valid_timescale_ms"]
        - raw_summary["historical_mean_valid_timescale_ms"]
    )
    disagreements = (
        audit.assign(disagree=audit["historical_valid"] != audit["reproduced_valid"])
        .groupby("session_id")["disagree"]
        .sum()
    )
    raw_summary["validity_disagreements"] = raw_summary["session_id"].map(disagreements)
    raw_summary.to_csv(output / "raw_session_summary.csv", index=False)

    historical_sessions = full_historical_summary(release)
    historical_sessions.to_csv(output / "historical_full_cohort_sessions.csv", index=False)
    cohort = (
        historical_sessions.groupby("cohort")
        .agg(
            sessions=("ecephys_session_id", "nunique"),
            historical_fit_units=("historical_fit_units", "sum"),
            valid_units=("valid_units", "sum"),
            equal_session_mean_timescale_ms=("mean_valid_timescale_ms", "mean"),
        )
        .reset_index()
    )
    cohort.to_csv(output / "historical_full_cohort_summary.csv", index=False)

    passed = (
        audit["spike_count_difference"].eq(0).all()
        and (audit["historical_valid"] == audit["reproduced_valid"]).all()
        and np.nanmax(np.abs(raw_summary["mean_valid_timescale_difference_ms"])) < 0.1
    )
    report = [
        "# Historical Allen V1 response-timescale reproduction",
        "",
        f"**Raw reproduction gate: {'PASS' if passed else 'FAIL'}**",
        "",
        f"- Raw sessions: {audit['session_id'].nunique()} ({len(audit):,} historical VISp fits).",
        f"- Exact spike counts: {int(audit['spike_count_difference'].eq(0).sum()):,}/{len(audit):,} units.",
        f"- Maximum absolute timescale difference: {np.nanmax(np.abs(audit['timescale_difference_ms'])):.6g} ms.",
        f"- Maximum absolute valid-session mean difference: {np.nanmax(np.abs(raw_summary['mean_valid_timescale_difference_ms'])):.6g} ms.",
        f"- Validity-gate disagreements: {int((audit['historical_valid'] != audit['reproduced_valid']).sum())}.",
        f"- Full frozen-table support: {int(historical_sessions['historical_fit_units'].sum()):,} fits, {int(historical_sessions['valid_units'].sum()):,} valid units, {historical_sessions['ecephys_session_id'].nunique()} sessions.",
        "",
        "The raw gate requires exact spike counts, identical historical validity decisions,",
        "and <0.1-ms drift in every valid-session mean. Per-unit rejected fits are not",
        "constrained because bounded nonlinear solutions are unstable at the fit limits.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("PASS" if passed else "FAIL", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
