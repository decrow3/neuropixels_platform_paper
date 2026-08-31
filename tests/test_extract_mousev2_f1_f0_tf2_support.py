import numpy as np
import pandas as pd

from scripts.extract_mousev2_f1_f0_tf2_support import mouse_conditions, session_bootstrap


def test_session_bootstrap_uses_session_centers():
    allen = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2],
            "mean_log10_f1_f0": [1.0, 1.0, 1.0, 1.0],
        }
    )
    observed, low, high = session_bootstrap(
        np.array([2.0, 2.0]), allen, draws=20, seed=1
    )
    assert observed == 1.0
    assert low == 1.0
    assert high == 1.0


def test_mouse_conditions_are_lexicographically_ordered():
    rows = []
    for orientation in (90.0, 0.0, 45.0, 135.0):
        for tf in (2.0, 1.0, 4.0, 8.0, 15.0):
            for repeat in range(15):
                rows.append(
                    {
                        "orientation": orientation,
                        "temporal_frequency": tf,
                        "spatial_frequency": 0.04,
                        "contrast": 0.8,
                        "start_time": len(rows) * 2.0,
                        "stop_time": len(rows) * 2.0 + 1.0,
                    }
                )
    conditions = mouse_conditions(pd.DataFrame(rows), temporal_frequency_hz=2.0)
    assert [condition[0][0] for condition in conditions] == [0.0, 45.0, 90.0, 135.0]
