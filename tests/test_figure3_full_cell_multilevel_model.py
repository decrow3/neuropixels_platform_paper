import numpy as np
import pandas as pd

from scripts.figure3_full_cell_multilevel_model import (
    prepare_model_table,
    resample_sessions,
)


def test_prepare_model_table_builds_crossed_identifiers_and_unit_scale():
    units = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "session_id": ["s1", "s1", "s2", "s2"],
            "group": ["a", "b", "a", "b"],
        }
    )
    table, scale = prepare_model_table(units)
    assert np.isclose(table["z"].std(ddof=1), 1)
    assert np.isclose(scale, units["value"].std(ddof=1))
    assert set(table["session_group"]) == {"s1:a", "s1:b", "s2:a", "s2:b"}


def test_session_bootstrap_keeps_whole_blocks_and_relabels_duplicates():
    units = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "group": ["a", "b", "a", "b"],
            "value": [1, 2, 3, 4],
        }
    )
    result = resample_sessions(units, np.random.default_rng(4))
    assert result["session_id"].nunique() == 2
    assert len(result) == 4
    assert result.groupby("session_id")["group"].nunique().eq(2).all()
