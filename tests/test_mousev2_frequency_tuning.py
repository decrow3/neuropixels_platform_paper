"""Tests for trial-derived MouseV2 SF x TF support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.drifting_gratings import (
    benjamini_hochberg,
    compute_frequency_tuning_surfaces,
)
from scripts.extract_mousev2_frequency_tuning import apply_support_contract


def _synthetic_frequency_trials() -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    tuned_spikes = []
    tf_only_spikes = []
    start = 10.0
    for sf in (0.02, 0.04, 0.08, 0.16, 0.32):
        for tf in (1.0, 2.0, 4.0, 8.0, 15.0):
            for orientation in (0.0, 45.0, 90.0, 135.0):
                for repeat in range(6):
                    rows.append(
                        {
                            "start_time": start,
                            "stop_time": start + 1.00084,
                            "spatial_frequency": sf,
                            "temporal_frequency": tf,
                            "orientation": orientation,
                        }
                    )
                    # A repeat-stable nonlinear peak after orientation marginalization.
                    tuned_count = 1 + (8 if (sf, tf) == (0.08, 4.0) else 0)
                    tf_count = 1 + (6 if tf == 8.0 else 0)
                    tuned_spikes.extend(start + np.linspace(0.05, 0.9, tuned_count))
                    tf_only_spikes.extend(start + np.linspace(0.05, 0.9, tf_count))
                    start += 1.5
    return pd.DataFrame(rows), {
        1: np.asarray(tuned_spikes),
        2: np.asarray(tf_only_spikes),
        3: np.asarray([row["start_time"] + 0.5 for row in rows]),
    }


def test_trial_surfaces_recover_supported_axis_preferences():
    presentations, spikes = _synthetic_frequency_trials()
    metrics, surfaces = compute_frequency_tuning_surfaces(
        [1, 2, 3], spikes, presentations, min_trials_per_sf_tf=10
    )
    assert len(surfaces) == 3 * 25
    assert surfaces["trials"].eq(24).all()
    assert metrics.loc[0, "surface_peak_sf_cpd"] == 0.08
    assert metrics.loc[0, "surface_peak_tf_hz"] == 4.0
    assert metrics.loc[0, "surface_split_half_spearman_brown"] > 0.99

    supported = apply_support_contract(metrics, fdr_alpha=0.05, minimum_reliability=0.3)
    assert bool(supported.loc[0, "sf_preference_supported"])
    assert bool(supported.loc[0, "tf_preference_supported"])
    assert supported.loc[0, "pref_sf_supported_dg"] == 0.08
    assert supported.loc[0, "pref_tf_supported_dg"] == 4.0
    assert not bool(supported.loc[1, "sf_preference_supported"])
    assert bool(supported.loc[1, "tf_preference_supported"])
    assert np.isnan(supported.loc[1, "pref_sf_supported_dg"])
    assert np.isnan(supported.loc[2, "pref_sf_supported_dg"])
    assert np.isnan(supported.loc[2, "pref_tf_supported_dg"])


def test_bh_adjustment_is_monotone_and_preserves_missing_values():
    adjusted = benjamini_hochberg(np.array([0.04, 0.001, np.nan, 0.03]))
    assert np.isnan(adjusted[2])
    assert adjusted[1] <= adjusted[3] <= adjusted[0]
