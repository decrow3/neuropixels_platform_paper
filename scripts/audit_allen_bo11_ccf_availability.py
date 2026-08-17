#!/usr/bin/env python3
"""Audit released CCF-coordinate availability across Allen BO 1.1 sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "allen_multisession_rf_validation_v1"
    / "09_ccf_availability"
)
CCF_COLUMNS = (
    "anterior_posterior_ccf_coordinate",
    "dorsal_ventral_ccf_coordinate",
    "left_right_ccf_coordinate",
)
HVA5 = ("VISal", "VISrl", "VISam", "VISl", "VISpm")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def status(rate):
    if not np.isfinite(rate):
        return "not_recorded"
    if rate >= 0.95:
        return "complete"
    if rate < 0.05:
        return "missing"
    return "partial"


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "ecephys_unit_id",
        "ecephys_session_id",
        "session_type",
        "ecephys_structure_acronym",
        *CCF_COLUMNS,
    ]
    units = pd.read_csv(args.units.resolve(), usecols=columns, low_memory=False)
    units = units.loc[units.session_type.eq("brain_observatory_1.1")].copy()
    units["ccf_complete"] = units[list(CCF_COLUMNS)].notna().all(axis=1)
    units["ccf_any"] = units[list(CCF_COLUMNS)].notna().any(axis=1)
    if (units.ccf_any & ~units.ccf_complete).any():
        raise ValueError("Found units with only a subset of the three CCF coordinates")
    if units.ecephys_unit_id.duplicated().any():
        raise ValueError("ecephys_unit_id is not unique")

    populations = {
        "all_units": lambda frame: pd.Series(True, index=frame.index),
        "visual_all": lambda frame: frame.ecephys_structure_acronym.fillna("").str.startswith("VIS"),
        "V1": lambda frame: frame.ecephys_structure_acronym.eq("VISp"),
        "HVA5": lambda frame: frame.ecephys_structure_acronym.isin(HVA5),
    }
    rows = []
    for session_id, local in units.groupby("ecephys_session_id", observed=True):
        for population, selector in populations.items():
            selected = local.loc[selector(local)]
            rate = selected.ccf_complete.mean() if len(selected) else np.nan
            rows.append(
                {
                    "ecephys_session_id": int(session_id),
                    "population": population,
                    "units": len(selected),
                    "ccf_complete_units": int(selected.ccf_complete.sum()),
                    "ccf_complete_fraction": rate,
                    "status": status(rate),
                }
            )
    session = pd.DataFrame(rows)
    session.to_csv(output / "bo11_session_ccf_availability.csv", index=False, float_format="%.9g")

    area = (
        units.loc[units.ecephys_structure_acronym.isin(("VISp", *HVA5))]
        .groupby(["ecephys_session_id", "ecephys_structure_acronym"], observed=True)
        .agg(units=("ecephys_unit_id", "size"), ccf_complete_units=("ccf_complete", "sum"))
        .reset_index()
    )
    area["ccf_complete_fraction"] = area.ccf_complete_units / area.units
    area["status"] = area.ccf_complete_fraction.map(status)
    area.to_csv(output / "bo11_session_area_ccf_availability.csv", index=False, float_format="%.9g")

    summary = (
        session.groupby(["population", "status"], observed=True)
        .size()
        .rename("sessions")
        .reset_index()
    )
    summary.to_csv(output / "bo11_ccf_availability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
