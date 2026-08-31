import numpy as np
import pandas as pd

from scripts.trace_v1_15hz_frame_phase_case import carrier_frame_table, phase_summary, select_case


def test_carrier_frame_table_has_rotated_four_frame_cycles():
    table = carrier_frame_table(np.array([0.0, 0.25, 0.5, 0.75]), frames=4)
    assert len(table) == 16
    phase_zero = table.loc[table.start_phase_cycles.eq(0), "carrier_cosine"].to_numpy()
    assert np.allclose(phase_zero, [1, 0, -1, 0], atol=1e-12)
    phase_half = table.loc[table.start_phase_cycles.eq(0.5), "carrier_cosine"].to_numpy()
    assert np.allclose(phase_half, [-1, 0, 1, 0], atol=1e-12)


def test_select_case_uses_support_and_median_not_phase_outcome():
    rows = []
    for unit_id, ratio, valid, spikes in (
        (1, 0.5, 15, 4),
        (2, 1.0, 15, 4),
        (3, 2.0, 15, 4),
        (4, 1.05, 14, 4),
    ):
        rows.append(
            {
                "dataset": "MouseV2",
                "cohort": "BO_support",
                "session_id": 2,
                "unit_id": unit_id,
                "preferred_tf_hz": 15.0,
                "full_f1_f0": ratio,
                "full_valid_trials": valid,
                "steady_valid_trials": valid,
                "steady_mean_spikes": spikes,
            }
        )
    selected = select_case(pd.DataFrame(rows), 2)
    assert selected.unit_id == 2
    assert "phase outcomes not used" in selected.selection_rule


def test_phase_summary_retains_counts_and_validity():
    trials = pd.DataFrame(
        {
            "trial": [0, 1, 2, 3],
            "start_phase_cycles": [0.0, 0.0, 0.25, 0.25],
            "full_spikes": [2, 4, 6, 8],
            "canonical_1ms_f1_f0": [0.8, 1.2, 0.4, 1.6],
            "full_15_cycles_f1_f0": [1.0, np.nan, 0.5, 1.5],
        }
    )
    result = phase_summary(trials).set_index("start_phase_cycles")
    assert result.loc[0.0, "trials"] == 2
    assert result.loc[0.0, "valid_full_15_cycles_f1_f0"] == 1
    assert result.loc[0.25, "mean_full_spikes"] == 7
    assert result.loc[0.25, "mean_full_15_cycles_f1_f0"] == 1
    assert result.loc[0.0, "mean_canonical_1ms_f1_f0"] == 1
