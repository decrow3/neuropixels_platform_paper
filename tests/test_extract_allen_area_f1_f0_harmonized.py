import numpy as np
import pandas as pd

from scripts.extract_allen_area_f1_f0_harmonized import complete_support, eligible_units


def test_complete_support_requires_twenty_conditions_with_fifteen_trials(monkeypatch):
    conditions = [((0.0, float(index), 0.04, 0.8), np.arange(15)) for index in range(20)]
    monkeypatch.setattr(
        "scripts.extract_allen_area_f1_f0_harmonized.condition_starts",
        lambda table, common_support: conditions,
    )
    result, complete, minimum, maximum = complete_support(pd.DataFrame(), {})
    assert result == conditions
    assert complete
    assert minimum == maximum == 15
    conditions[-1] = (conditions[-1][0], np.arange(14))
    result, complete, minimum, maximum = complete_support(pd.DataFrame(), {})
    assert not complete
    assert (minimum, maximum) == (14, 15)


def test_eligible_units_maps_figure_groups_and_applies_qc():
    table = pd.DataFrame(
        {
            "session_type": ["brain_observatory_1.1"] * 3,
            "ecephys_structure_acronym": ["VISp", "VISl", "LP"],
            "amplitude_cutoff": [0.01, 0.2, 0.01],
            "presence_ratio": [0.9, 0.9, 0.9],
            "isi_violations": [0.1, 0.1, 0.1],
        }
    )
    result = eligible_units(table)
    assert result.area_coarse.tolist() == ["V1", "LP"]
