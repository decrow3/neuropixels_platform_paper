"""Checks for the cell-level denominator and clustered randomization."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_figure4_cell_location_controls import (
    analyze, collapse_cells, omega_moments, sufficient_blocks,
)
from scripts.figure3_robust_spread_comparison import omega_squared
from scripts.build_figure4_variant_review import METRICS


def test_moments_match_expanded_cells_with_unequal_cluster_sizes():
    table = pd.DataFrame({"session_id": ["1"] * 5 + ["2"] * 4,
                          "group": ["a", "a", "b", "b", "b", "a", "a", "a", "b"],
                          "value": [1., 4., 2., 9., 5., 3., 8., 6., 12.]})
    blocks = sufficient_blocks(table)
    np.testing.assert_allclose(omega_moments(blocks.sum(axis=0)),
                               omega_squared(table.group.to_numpy(), table.value.to_numpy()))
    # Repeated sessions must duplicate every cell, including the unequal counts.
    expanded = pd.concat([table.loc[table.session_id.eq("1")]] * 2 + [table.loc[table.session_id.eq("2")]])
    np.testing.assert_allclose(omega_moments(blocks[[0, 0, 1]].sum(axis=0)),
                               omega_squared(expanded.group.to_numpy(), expanded.value.to_numpy()))


def test_multiple_fits_count_as_one_neuron_and_duplicates_fail():
    raw = pd.DataFrame(dict(dataset=["Within-V1"] * 3, metric=[METRICS[2]] * 3,
                            session_id=["1"] * 3, group=["A"] * 3,
                            unit_id=["u1", "u1", "u2"], draw_id=[0, 1, 0], value=[10., 30., 60.]))
    cells = collapse_cells(raw)
    np.testing.assert_allclose(cells.value, [20., 60.])
    np.testing.assert_array_equal(cells.n_valid_fits, [2, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        collapse_cells(pd.concat([raw, raw.iloc[[0]]]))


def test_single_location_sessions_do_not_change_under_block_shuffle():
    rows = []
    for metric in METRICS:
        for comparison in ["v1", "hva", "control"]:
            for session in range(12):
                for value in [0., 1., 2.]:
                    rows.append(dict(metric=metric, comparison=comparison, session_id=str(session),
                                     group="a" if session % 2 else "b", value=value + (session % 2) * 10))
    stats, draws = analyze(pd.DataFrame(rows), 30, 42)
    for row in stats.loc[stats.comparison.ne("delta")].itertuples():
        null = draws.loc[draws.metric.eq(row.metric) & draws.comparison.eq(row.comparison), "shuffle"]
        np.testing.assert_allclose(null, row.observed)


def test_paired_location_shuffle_moves_complete_cell_populations():
    rows = []
    for metric in METRICS:
        for comparison in ["v1", "hva", "control"]:
            for session in range(15):
                for group, values in [("a", [0., 0.]), ("b", [10., 10., 10.])]:
                    rows.extend(dict(metric=metric, comparison=comparison, session_id=str(session),
                                     group=group, value=value) for value in values)
    stats, draws = analyze(pd.DataFrame(rows), 100, 42)
    assert stats.loc[stats.comparison.ne("delta"), "observed"].eq(1.).all()
    assert draws.loc[draws.comparison.ne("delta"), "shuffle"].quantile(.975) < .4
