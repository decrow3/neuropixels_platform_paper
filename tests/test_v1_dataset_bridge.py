"""Focused regression tests for the V1 cross-dataset bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.extract_mousev2_grating_common_support import common_presentations
from scripts.v1_dataset_bridge import _bootstrap_difference, _welch_lookup


def test_welch_lookup_exposes_duration_dependent_frequency_grid():
    lookup = _welch_lookup()
    mouse_2hz = lookup[
        lookup["dataset"].eq("MouseV2") & lookup["requested_tf_hz"].eq(2.0)
    ].iloc[0]
    allen_2hz = lookup[
        lookup["dataset"].eq("Allen") & lookup["requested_tf_hz"].eq(2.0)
    ].iloc[0]

    assert mouse_2hz["welch_nperseg_effective"] == 1000
    assert mouse_2hz["searchsorted_frequency_hz"] == 2.0
    assert allen_2hz["welch_nperseg_effective"] == 1024
    assert allen_2hz["searchsorted_frequency_hz"] == pytest.approx(2.9296875)
    assert allen_2hz["nearest_frequency_hz"] == pytest.approx(1.953125)


def test_session_bootstrap_reports_mouse_minus_allen():
    observed, low, high = _bootstrap_difference(
        np.array([0.0, 0.1, 0.2]),
        np.array([1.0, 1.1, 1.2]),
        n_bootstrap=2000,
        seed=7,
    )
    assert observed == pytest.approx(-1.0)
    assert low < observed < high


def test_mouse_common_support_selects_twenty_conditions():
    rows = []
    start = 0.0
    for orientation in (0.0, 45.0, 90.0, 135.0):
        for tf in (1.0, 2.0, 4.0, 8.0, 15.0):
            for sf in (0.04, 0.08):
                for _ in range(15):
                    rows.append(
                        {
                            "orientation": orientation,
                            "temporal_frequency": tf,
                            "spatial_frequency": sf,
                            "contrast": 0.8,
                            "start_time": start,
                            "stop_time": start + 1.0,
                        }
                    )
                    start += 2.0
    selected = common_presentations(pd.DataFrame(rows))
    assert len(selected) == 300
    assert selected["spatial_frequency"].eq(0.04).all()
