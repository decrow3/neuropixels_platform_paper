import numpy as np

from scripts.compare_v1_f1_f0_datasets import bootstrap_session_difference


def test_bootstrap_session_difference_preserves_observed_center():
    observed, low, high = bootstrap_session_difference(
        np.array([2.0, 2.0]), np.array([1.0, 1.0]), draws=20, seed=1
    )
    assert observed == 1.0
    assert low == 1.0
    assert high == 1.0
