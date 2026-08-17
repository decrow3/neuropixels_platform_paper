"""Focused tests for the Iteration 6C Allen RF-targeting audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.allen_rf_matching import (
    add_population_flags,
    paired_hva_v1_offsets,
    points_in_convex_hull,
)


def _unit_row(**updates) -> dict[str, object]:
    row = {
        "ecephys_unit_id": 1,
        "ecephys_session_id": 10,
        "ecephys_probe_id": 100,
        "specimen_id": 1000,
        "session_type": "brain_observatory_1.1",
        "ecephys_structure_acronym": "VISp",
        "azimuth_rf": 20.0,
        "elevation_rf": 5.0,
        "area_rf": 400.0,
        "p_value_rf": 0.001,
        "snr": 2.0,
        "firing_rate_dg": 1.0,
        "amplitude_cutoff": 0.01,
        "presence_ratio": 0.95,
        "isi_violations": 0.01,
    }
    row.update(updates)
    return row


def test_population_flags_are_explicit_and_nested():
    table = pd.DataFrame(
        [
            _unit_row(ecephys_unit_id=1),
            _unit_row(ecephys_unit_id=2, firing_rate_dg=0.0),
            _unit_row(ecephys_unit_id=3, amplitude_cutoff=0.2),
            _unit_row(ecephys_unit_id=4, p_value_rf=0.2),
        ]
    )
    flagged = add_population_flags(table)

    assert flagged["rf_only"].tolist() == [True, True, True, False]
    assert flagged["published_like"].tolist() == [True, False, True, False]
    assert flagged["intersection"].tolist() == [True, False, False, False]


def test_hva_offsets_are_paired_to_v1_within_session():
    summary = pd.DataFrame(
        [
            {
                "ecephys_session_id": 10,
                "specimen_id": 1000,
                "cohort": "Brain Observatory 1.1",
                "area": "V1",
                "n_units": 20,
                "azimuth_median_deg": 30.0,
                "elevation_median_deg": 5.0,
            },
            {
                "ecephys_session_id": 10,
                "specimen_id": 1000,
                "cohort": "Brain Observatory 1.1",
                "area": "AL",
                "n_units": 15,
                "azimuth_median_deg": 42.0,
                "elevation_median_deg": -4.0,
            },
            {
                "ecephys_session_id": 11,
                "specimen_id": 1001,
                "cohort": "Brain Observatory 1.1",
                "area": "PM",
                "n_units": 12,
                "azimuth_median_deg": 50.0,
                "elevation_median_deg": 20.0,
            },
        ]
    )
    paired = paired_hva_v1_offsets(summary)

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["area"] == "AL"
    assert row["delta_azimuth_deg"] == 12.0
    assert row["delta_elevation_deg"] == -9.0
    assert row["distance_from_v1_deg"] == 15.0


def test_convex_hull_includes_edges_and_rejects_extrapolation():
    reference = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    query = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [np.nan, 1.0]])

    assert points_in_convex_hull(reference, query).tolist() == [
        True,
        True,
        False,
        False,
    ]
