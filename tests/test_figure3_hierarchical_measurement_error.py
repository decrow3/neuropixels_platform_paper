import numpy as np
import pandas as pd

from scripts.figure3_hierarchical_measurement_error import (
    bootstrap_mean_uncertainty,
    build_measurement_table,
    select_diagnostic_cases,
)


def test_bootstrap_mean_uncertainty_is_reproducible_and_matches_mean_se():
    values = np.arange(1.0, 21.0)
    first = bootstrap_mean_uncertainty(
        values, repetitions=2000, batch_size=137, rng=np.random.default_rng(8)
    )
    second = bootstrap_mean_uncertainty(
        values, repetitions=2000, batch_size=137, rng=np.random.default_rng(8)
    )
    assert first == second
    assert first["mean"] == np.mean(values)
    assert 0.90 < first["bootstrap_se"] / first["analytic_se"] < 1.10
    assert first["bootstrap_ci_low"] < first["mean"] < first["bootstrap_ci_high"]


def test_measurement_table_uses_metric_filter_and_session_centering():
    frame = pd.DataFrame(
        {
            "session": [1] * 12,
            "group": ["A"] * 6 + ["B"] * 6,
            "subject": [10] * 12,
            "ttfs": [0.05] * 5 + [0.15] + [0.07] * 6,
        }
    )
    table = build_measurement_table(
        frame, dataset="d", session_column="session", group_column="group",
        groups=["A", "B"], metric_specs=[("ttfs", "TTFS")], min_units=5,
        repetitions=100, batch_size=20, rng=np.random.default_rng(3),
        subject_column="subject",
    )
    assert table.set_index("group").loc["A", "n_units"] == 5
    assert np.isclose(table.groupby("session_id")["centered_mean"].sum().iloc[0], 0)
    assert set(table["subject_id"]) == {"10"}


def test_diagnostic_cases_have_three_predeclared_roles():
    table = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "metric": ["m"] * 4,
            "session_id": ["1", "2", "3", "4"],
            "group": ["A"] * 4,
            "n_units": [10, 20, 30, 40],
            "mean": [1, 2, 3, 4],
            "bootstrap_se": [0.4, 0.1, 0.3, 0.2],
        }
    )
    selected = select_diagnostic_cases(table)
    assert set(selected["selection_role"]) == {
        "lowest measurement error", "typical measurement error", "highest measurement error"
    }
    assert selected.loc[selected["selection_role"] == "lowest measurement error", "session_id"].iloc[0] == "2"
