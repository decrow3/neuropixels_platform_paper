import numpy as np
import pandas as pd

from scripts.trace_v1_systemic_gap_cases import (
    circular_fractional_shift,
    select_mouse_cases,
)


def test_circular_fractional_shift_has_requested_fourier_phase():
    time = np.arange(1000)
    values = np.sin(2 * np.pi * 4 * time / 1000)[None, :]
    shifted = circular_fractional_shift(values, np.array([62.5]))
    expected = np.sin(2 * np.pi * 4 * (time - 62.5) / 1000)
    assert np.allclose(shifted[0], expected, atol=1e-12)


def test_mouse_case_roles_are_unique_and_include_negative_control():
    rows = []
    for unit_id, tf, gain, raw in [
        (1, 1.0, -0.1, -0.4),
        (2, 2.0, 0.0, -0.2),
        (3, 15.0, 0.8, 0.1),
        (4, 4.0, 0.0, -0.3),
        (5, 4.0, 0.0, 0.3),
    ]:
        rows.append(
            {
                "site": "site2",
                "session_id": 2,
                "unit_id": unit_id,
                "preferred_tf_hz": tf,
                "source_phase_expected_to_vary": tf in (1.0, 2.0, 15.0),
                "source_log10_mod_idx_gain": gain,
                "log10_raw_mod_idx": raw,
            }
        )
    selected = select_mouse_cases(pd.DataFrame(rows))
    assert set(selected.selection_role) == {
        "typical_affected",
        "prediction_without_response",
        "largest_observed_gain",
        "phase_stable_negative_control",
    }
    assert selected.loc[
        selected.selection_role.eq("phase_stable_negative_control"), "unit_id"
    ].item() == 4
