"""Tests for Allen-matched MouseV2 flash trial support."""

import numpy as np
import pandas as pd

from scripts.mousev2_timescale_trial_bridge import (
    balanced_subsample_masks,
    is_valid_timescale,
)


def test_balanced_subsamples_have_allen_trial_support():
    flashes = pd.DataFrame(
        {"flash_polarity": ["bright"] * 150 + ["dark"] * 150}
    )
    first = balanced_subsample_masks(flashes, n_subsamples=3, seed=17)
    second = balanced_subsample_masks(flashes, n_subsamples=3, seed=17)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
        assert left.sum() == 150
        assert left[:150].sum() == 75
        assert left[150:].sum() == 75


def test_timescale_validity_matches_figure_rule():
    table = pd.DataFrame(
        {
            "timescale_ms": [50.0, 301.0, 50.0, 50.0],
            "fit_error_ms": [10.0, 10.0, 20.0, 10.0],
            "spike_count": [51.0, 51.0, 51.0, 50.0],
        }
    )
    assert is_valid_timescale(table).tolist() == [True, False, False, False]
