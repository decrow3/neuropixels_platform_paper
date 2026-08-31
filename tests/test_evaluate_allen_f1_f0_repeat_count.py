import numpy as np

from scripts.evaluate_allen_f1_f0_repeat_count import (
    allen_trial_components,
    summarize_condition,
)


def test_allen_trial_components_uses_released_cycle_fold_convention():
    counts = np.ones((2, 1999), dtype=float)
    result = allen_trial_components(counts, 2.0)
    assert np.allclose(result.f0_hz, 1000.0)
    assert np.allclose(result.f1_hz, 0.0, atol=1e-12)
    assert np.allclose(result.f1_f0, 0.0, atol=1e-12)


def test_fixed_condition_does_not_reselect():
    prepared = [
        {
            "parameters": (0.0, 2.0),
            "selection_counts": np.array([10, 10]),
            "components": allen_trial_components(np.ones((2, 1999)), 2.0),
        },
        {
            "parameters": (90.0, 2.0),
            "selection_counts": np.array([100, 100]),
            "components": allen_trial_components(np.ones((2, 1999)) * 2, 2.0),
        },
    ]
    result = summarize_condition(prepared, [np.array([0]), np.array([0])], 0)
    assert result["condition_index"] == 0
