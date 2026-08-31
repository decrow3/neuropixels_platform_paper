import numpy as np
import pandas as pd

from scripts.figure3_full_cell_multilevel_extension import (
    aggregate_point_fits,
    resample_timescale_v1,
    summarize_timescale_draw_sensitivity,
)


def test_timescale_bootstrap_uses_one_draw_per_neuron_in_each_block():
    rows = []
    for session in ["s1", "s2"]:
        for draw in [0, 1, 2]:
            for unit in ["u1", "u2"]:
                rows.append(
                    {
                        "session_id": session,
                        "draw_id": draw,
                        "unit_id": f"{session}:{unit}",
                        "group": "A",
                        "value": float(draw),
                    }
                )
    result = resample_timescale_v1(pd.DataFrame(rows), np.random.default_rng(3))
    assert result["session_id"].nunique() == 2
    assert not result.duplicated(["session_id", "unit_id"]).any()
    assert result.groupby("session_id")["selected_draw_id"].nunique().eq(1).all()


def test_point_fit_aggregation_averages_matched_draw_components():
    fits = pd.DataFrame(
        {
            "metric": ["m", "m"],
            "dataset": ["Within-V1", "Within-V1"],
            "optimizer": ["nm", "nm"],
            "converged": [True, True],
            "n_cells": [10, 12],
            "n_sessions": [2, 2],
            "n_groups": [4, 4],
            "n_session_groups": [8, 8],
            "variance_group": [1.0, 3.0],
            "variance_interaction": [2.0, 4.0],
            "variance_session": [1.0, 1.0],
            "variance_residual": [10.0, 14.0],
            "variance_total_structured": [3.0, 7.0],
            "fraction_structured_stable": [1 / 3, 3 / 7],
        }
    )
    result = aggregate_point_fits(fits).iloc[0]
    assert result.point_fits == 2
    assert np.isclose(result.variance_group, 2)
    assert np.isclose(result.variance_total_structured, 5)


def test_timescale_sensitivity_flags_only_near_zero_v1_draw():
    base = {
        "metric": "Response timescale (ms)",
        "optimizer": "nm",
        "reml_log_likelihood": -10.0,
        "n_cells": 10,
        "variance_group": 0.0,
        "variance_interaction": 0.0,
        "variance_residual": 100.0,
    }
    fits = pd.DataFrame(
        [
            {**base, "dataset": "Within-V1", "draw_id": 0,
             "variance_total_structured": 1e-8},
            {**base, "dataset": "Within-V1", "draw_id": 1,
             "variance_total_structured": 2.0},
            {**base, "dataset": "Post-V1", "draw_id": -1,
             "variance_total_structured": 20.0},
        ]
    )
    result = summarize_timescale_draw_sensitivity(fits)
    assert result["v1_total_near_boundary"].tolist() == [True, False]
    assert np.isclose(result.iloc[1]["hva_to_v1_total_ratio"], 10.0)
