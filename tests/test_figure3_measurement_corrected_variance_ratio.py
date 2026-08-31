import numpy as np
import pandas as pd

from scripts.figure3_measurement_corrected_variance_ratio import (
    corrected_session_variances,
    matched_timescale_cluster_bootstrap,
)


def test_corrected_variance_subtracts_mean_squared_se_without_clipping():
    cells = pd.DataFrame(
        {
            "dataset": ["d"] * 3,
            "metric": ["m"] * 3,
            "session_id": ["1"] * 3,
            "group": ["a", "b", "c"],
            "mean": [1.0, 2.0, 3.0],
            "group_mean_se": [0.5, 0.5, 0.5],
        }
    )
    result = corrected_session_variances(cells).iloc[0]
    assert np.isclose(result.observed_variance, 1.0)
    assert np.isclose(result.estimated_measurement_variance, 0.25)
    assert np.isclose(result.corrected_variance_unclipped, 0.75)

    cells["mean"] = 2.0
    negative = corrected_session_variances(cells).iloc[0]
    assert negative.corrected_variance_unclipped < 0


def test_timescale_cluster_bootstrap_reproduces_mean_of_draw_means():
    part = pd.DataFrame(
        {
            "unit_id": np.repeat([1, 2, 3], 10),
            "subsample": np.tile(np.arange(10), 3),
            "timescale_ms": np.repeat([10.0, 20.0, 30.0], 10),
        }
    )
    mean, se, mean_n, unique_n, rejection_fraction = matched_timescale_cluster_bootstrap(
        part, repetitions=2000, rng=np.random.default_rng(7)
    )
    assert np.isclose(mean, 20.0)
    assert se > 0
    assert mean_n == 3
    assert unique_n == 3
    assert rejection_fraction == 0
