import numpy as np

from scripts.evaluate_f1_f0_candidate_cases import trial_components


def test_trial_components_recovers_known_f1_f0_and_zero_f0_missingness():
    time = np.arange(1000) / 1000
    rate = 3 + 2 * np.cos(2 * np.pi * 4 * time)
    counts = np.vstack([rate, np.zeros(1000)])
    result = trial_components(counts, 4.0)
    assert np.isclose(result.loc[0, "f0_hz"], 3000.0)
    assert np.isclose(result.loc[0, "f1_hz"], 2000.0)
    assert np.isclose(result.loc[0, "f1_f0"], 2 / 3)
    assert np.isnan(result.loc[1, "f1_f0"])
