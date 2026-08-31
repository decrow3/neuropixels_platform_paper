import numpy as np

from scripts.diagnose_f1_f0_steady_state_history import (
    steady_window,
    window_trial_components,
)


def test_steady_windows_exclude_at_least_half_second_and_use_complete_cycles():
    assert steady_window(1.0) is None
    for tf in (2.0, 4.0, 8.0, 15.0):
        start, end, cycles = steady_window(tf)
        assert start >= 0.5
        assert np.isclose((end - start) * tf, cycles)


def test_one_steady_state_spike_has_ratio_two():
    starts = np.array([1.0])
    _, ratio, _ = window_trial_components(
        np.array([1.75]), starts, 2.0, steady_window(2.0)
    )
    assert np.isclose(ratio[0], 2.0)
