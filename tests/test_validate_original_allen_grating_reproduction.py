import numpy as np
import pandas as pd

from scripts.extract_allen_v1_bridge import _preferred_condition_actual_stops
from scripts.validate_original_allen_grating_reproduction import validate_rows


def test_preference_selection_uses_actual_presentation_stops_and_first_tie():
    spikes = np.array([0.5, 1.0005, 2.5, 3.0005])
    conditions = [
        ((0.0, 2.0, 0.04, 0.8), np.array([0.0]), np.array([1.001])),
        ((45.0, 2.0, 0.04, 0.8), np.array([2.0]), np.array([3.001])),
    ]
    # Both conditions contain two spikes under their actual recorded stops;
    # numpy argmax/Allen idxmax resolves the tie to first condition order.
    assert _preferred_condition_actual_stops(spikes, conditions) == 0


def test_reproduction_gate_accepts_matched_missing_f1_only():
    rows = pd.DataFrame(
        {
            "raw_f1_f0_dg": [np.nan, np.nan],
            "released_f1_f0_dg": [np.nan, 1.0],
            "raw_mod_idx_dg": [2.0, 2.0],
            "released_mod_idx_dg": [2.0, 2.0],
        }
    )
    result = validate_rows(rows)
    assert result["rows"].passes.tolist() == [True, False]
