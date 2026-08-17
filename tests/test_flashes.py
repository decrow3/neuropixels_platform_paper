"""Tests for raw, polarity-specific full-field flash metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.flashes import (
    TIMESCALE_BIN_EDGES_S,
    compute_flash_metrics,
    first_spike_latency_seconds,
    prepare_flash_presentations,
    timescale_bin_mask,
)


def synthetic_flashes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": [1.0, 2.0, 3.0, 4.0],
            "stop_time": [1.25, 2.25, 3.25, 4.25],
            "contrast": [1.0, -1.0, 1.0, -1.0],
        }
    )


def test_allen_color_encodes_flash_polarity():
    flashes = synthetic_flashes().assign(contrast=0.8, color=[1.0, -1.0, 1.0, -1.0])
    prepared = prepare_flash_presentations(flashes)
    assert prepared["flash_polarity"].tolist() == ["bright", "dark", "bright", "dark"]
    assert prepared["flash_polarity_source"].eq("color").all()


def test_flash_polarity_is_explicit_and_balanced():
    table = prepare_flash_presentations(synthetic_flashes())
    assert table["flash_polarity"].tolist() == ["bright", "dark", "bright", "dark"]
    with pytest.raises(ValueError, match="Expected flash polarity"):
        prepare_flash_presentations(table.assign(contrast=0.5))


def test_ttfs_matches_absolute_legacy_millisecond_bins():
    starts = np.array([1.0, 2.0, 3.0, 4.0])
    spikes = np.array(
        [
            1.0300,  # included at bin 30
            2.0509,  # truncated to bin 50
            3.1999,  # included at bin 199
            4.2000,  # excluded at the right edge
        ]
    )
    latency, valid_trials = first_spike_latency_seconds(spikes, starts)
    assert valid_trials == 3
    assert latency == pytest.approx(0.050)


def test_timescale_window_uses_allensdk_bin_centers():
    mask = timescale_bin_mask(TIMESCALE_BIN_EDGES_S)
    centers = TIMESCALE_BIN_EDGES_S[:-1] + np.diff(TIMESCALE_BIN_EDGES_S) / 2
    assert mask.sum() == 25
    assert centers[mask][0] == pytest.approx(0.045)
    assert centers[mask][-1] == pytest.approx(0.285)
    assert not mask[29]  # 0.295-s center was included by the prior left-edge mask.


def test_full_metric_table_keeps_pooled_and_both_polarities():
    flashes = synthetic_flashes()
    spikes = {
        7: np.array([1.0504, 2.0904, 3.0704, 4.1104]),
        8: np.array([], dtype=float),
    }
    metrics = compute_flash_metrics(spikes, flashes).set_index("unit_id")
    assert metrics.loc[7, "time_to_first_spike_bright"] == pytest.approx(0.060)
    assert metrics.loc[7, "time_to_first_spike_dark"] == pytest.approx(0.100)
    assert metrics.loc[7, "time_to_first_spike_pooled"] == pytest.approx(0.080)
    assert metrics.loc[7, "ttfs_valid_trials_pooled"] == 4
    assert metrics.loc[7, "flash_trials_pooled"] == 4
    assert metrics.loc[7, "flash_trials_bright"] == 2
    assert metrics.loc[7, "flash_trials_dark"] == 2
    assert np.isnan(metrics.loc[8, "time_to_first_spike_pooled"])
