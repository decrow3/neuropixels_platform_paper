"""Focused tests for the RF-adjusted Allen response checkpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.allen_rf_adjusted_response import (
    bootstrap_matched_summary,
    matched_session_contrasts,
    nearest_v1_matches,
    prepare_outcomes,
)


def test_prepare_outcomes_applies_metric_specific_validity_rules():
    table = pd.DataFrame(
        {
            "time_to_first_spike_fl": [0.05, 0.12],
            "mod_idx_dg": [2.0, 0.0],
            "f1_f0_dg": [1.0, -1.0],
            "timescale_ac": [50.0, 50.0],
            "err_ac": [10.0, 10.0],
            "spike_count_ac": [100.0, 20.0],
            "area_rf": [400.0, 500.0],
            "cortical_depth": [0.25, 0.75],
            "cortical_layer": [2, 5],
            "ecephys_session_id": [1, 1],
            "area": ["V1", "LM"],
        }
    )
    result = prepare_outcomes(table)

    assert result.loc[0, "ttfs_ms"] == 50.0
    assert np.isnan(result.loc[1, "ttfs_ms"])
    assert result.loc[0, "log10_mod_idx"] == np.log10(2.0)
    assert np.isnan(result.loc[1, "log10_mod_idx"])
    assert result.loc[0, "timescale_ms"] == 50.0
    assert np.isnan(result.loc[1, "timescale_ms"])


def test_nearest_v1_matching_uses_two_dimensional_rf_distance():
    v1 = pd.DataFrame(
        {
            "ecephys_unit_id": [1, 2],
            "relative_azimuth_deg": [0.0, 10.0],
            "relative_elevation_deg": [0.0, 10.0],
        }
    )
    hva = pd.DataFrame(
        {
            "relative_azimuth_deg": [1.0, 9.0],
            "relative_elevation_deg": [1.0, 8.0],
        }
    )
    matched, distances = nearest_v1_matches(hva, v1)

    assert matched["ecephys_unit_id"].tolist() == [1, 2]
    assert distances.tolist() == [np.sqrt(2), np.sqrt(5)]


def test_matched_summary_bootstraps_sessions_not_units():
    contrasts = pd.DataFrame(
        {
            "outcome": ["ttfs_ms"] * 3,
            "area": ["LM"] * 3,
            "session_id": ["1", "2", "3"],
            "matched_difference_hva_minus_v1": [1.0, 2.0, 3.0],
        }
    )
    summary = bootstrap_matched_summary(contrasts, n_bootstrap=2000, seed=7)

    assert len(summary) == 1
    assert summary.loc[0, "session_pairs"] == 3
    assert summary.loc[0, "equal_session_mean_difference"] == 2.0
    assert summary.loc[0, "bootstrap_ci_low"] <= 2.0
    assert summary.loc[0, "bootstrap_ci_high"] >= 2.0


def test_caliper_reports_discarded_hva_support():
    rows = []
    for unit_id in range(5):
        rows.append(
            {
                "ecephys_unit_id": unit_id,
                "session_id": "1",
                "cohort": "Brain Observatory 1.1",
                "area": "V1",
                "inside_v1_robust_box": True,
                "relative_azimuth_deg": float(unit_id),
                "relative_elevation_deg": 0.0,
            }
        )
    for unit_id, azimuth in enumerate([0.1, 1.1, 2.1, 3.1, 4.1, 20.0], start=10):
        rows.append(
            {
                "ecephys_unit_id": unit_id,
                "session_id": "1",
                "cohort": "Brain Observatory 1.1",
                "area": "LM",
                "inside_v1_robust_box": True,
                "relative_azimuth_deg": azimuth,
                "relative_elevation_deg": 0.0,
            }
        )
    table = pd.DataFrame(rows)
    for outcome in ("ttfs_ms", "log10_mod_idx", "log10_f1_f0", "timescale_ms", "rf_area_deg2"):
        table[outcome] = 1.0

    contrasts, balance = matched_session_contrasts(table, match_caliper_deg=5.0)

    assert len(contrasts) == 5
    assert balance["n_hva_available"].eq(6).all()
    assert balance["n_hva_matched"].eq(5).all()
    assert np.allclose(balance["hva_discarded_fraction"], 1 / 6)
