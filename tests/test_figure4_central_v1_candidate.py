"""Verify dependence and the restricted null for the added Allen V1 cohort."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_figure4_central_v1_candidate import joint_bootstraps, conditional_shuffle
from scripts.figure3_robust_spread_comparison import omega_squared


def test_identical_comparisons_share_allen_bootstrap_draws():
    rows = []
    for comparison in ["v1", "hva", "control"]:
        for session in range(8):
            for group, offset in [("a", 0), ("b", session + 1)]:
                for value in [0., 1., 3.]:
                    rows.append(dict(source="Allen", session_id=str(session), comparison=comparison,
                                     group=group, value=value + offset))
    draws = joint_bootstraps(pd.DataFrame(rows), 100, np.random.default_rng(42))
    assert np.std(draws["v1"]) > 0
    np.testing.assert_array_equal(draws["v1"], draws["hva"])
    np.testing.assert_array_equal(draws["v1"], draws["control"])
    np.testing.assert_array_equal(draws["hva"] - draws["v1"], np.zeros(100))


def test_central_cohort_effect_is_preserved_in_conditional_shuffle():
    rows = []
    for source, groups in [("Allen", ["Central"]), ("MouseV2", ["A", "B", "C", "E"])]:
        for session in range(8):
            for group in groups:
                for value in [0., 1., 2.]:
                    rows.append(dict(source=source, session_id=str(session), group=group,
                                     value=value + (100 if source == "Allen" else 0)))
    table = pd.DataFrame(rows)
    point = omega_squared(table.group.to_numpy(), table.value.to_numpy())
    null = conditional_shuffle(table, 100, np.random.default_rng(42))
    assert point > .99
    # A large Central-vs-MouseV2 shift is not tested by peripheral label shuffling.
    np.testing.assert_allclose(null, point)
