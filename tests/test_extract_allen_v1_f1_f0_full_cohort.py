import numpy as np

from common.drifting_gratings import _bin_trial_spike_counts, f1_f0_from_trial_counts
from scripts.extract_allen_v1_f1_f0_full_cohort import trial_f1_f0_components


def test_direct_trial_components_match_dense_cycle_fold():
    rng = np.random.default_rng(4)
    starts = np.array([1.0, 3.0, 5.0])
    spikes = np.sort(np.concatenate([start + rng.uniform(0, 1, 20) for start in starts]))
    for tf in (1.0, 2.0, 4.0, 8.0, 15.0):
        _, ratios = trial_f1_f0_components(spikes, starts, tf)
        dense = _bin_trial_spike_counts(spikes, starts, duration_ms=1000)
        expected = f1_f0_from_trial_counts(dense, tf, 1.0)
        assert np.isclose(np.nanmean(ratios), expected, atol=1e-12)


def test_one_spike_trial_has_ratio_two():
    _, ratios = trial_f1_f0_components(np.array([1.125]), np.array([1.0]), 4.0)
    assert np.isclose(ratios[0], 2.0)
