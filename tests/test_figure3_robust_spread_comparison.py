import numpy as np
import pandas as pd

from scripts.figure3_robust_spread_comparison import (
    HARMONIZED_F1_COLUMN,
    apply_harmonized_f1_f0,
    bootstrap_mean_difference,
    clustered_omega_squared,
    hierarchy_fit_summary,
    historical_proxy_full20_population,
    leave_one_out_differences,
    matched_mouse_timescale_session_group_table,
    omega_squared,
    session_group_table,
    session_spreads,
)


def test_full20_rf_proxy_applies_frozen_gates_and_records_flow(tmp_path):
    mouse = pd.DataFrame({
        "unit_id": [1, 2, 3], "session_num": [2, 2, 2], "probe_letter": ["A"] * 3,
        "snr": [2.0, 2.0, 2.0], "firing_rate_dg": [0.2, 0.2, 0.2],
    })
    rf = pd.DataFrame({
        "unit_id": [1, 2, 3], "rf_fit_success": [True, True, True],
        "rf_lrt_p": [0.001, 0.02, 0.001],
        "rf_sigma_major_deg": [10.0, 10.0, 40.0],
        "rf_sigma_minor_deg": [10.0, 10.0, 40.0],
    })
    path = tmp_path / "rf.csv"
    rf.to_csv(path, index=False)
    selected, flow, audit = historical_proxy_full20_population(mouse, rf_path=path)
    assert selected["unit_id"].tolist() == [1]
    assert audit["unit_id"].tolist() == [1]
    assert flow.iloc[-1]["retained_units"] == 1


def test_matched_timescale_table_drops_partial_draw_cells(tmp_path):
    rows = []
    for draw in range(10):
        for unit_id in range(1, 6):
            rows.append({"view": "mouse_matched_150", "session_id": 2, "unit_id": unit_id,
                         "subsample": draw, "timescale_ms": 40.0, "valid_timescale": True})
        if draw < 9:
            for unit_id in range(6, 11):
                rows.append({"view": "mouse_matched_150", "session_id": 2, "unit_id": unit_id,
                             "subsample": draw, "timescale_ms": 60.0, "valid_timescale": True})
    path = tmp_path / "bridge.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata = pd.DataFrame({"unit_id": range(1, 11), "session_num": [2] * 10,
                             "probe_letter": ["A"] * 5 + ["E"] * 5})
    result, coverage = matched_mouse_timescale_session_group_table(
        metadata, bridge_path=path, groups=["A", "E"], min_units=5, return_coverage=True
    )
    assert result["group"].tolist() == ["A"]
    assert not coverage.set_index("probe_letter").loc["E", "retained"]


def test_matched_timescale_table_averages_complete_trial_draws(tmp_path):
    rows = []
    for draw in range(10):
        for unit_id, value in ((1, 40.0 + draw), (2, 60.0 + draw)):
            rows.append(
                {
                    "view": "mouse_matched_150",
                    "session_id": 2,
                    "site": "site2",
                    "unit_id": unit_id,
                    "subsample": draw,
                    "timescale_ms": value,
                    "valid_timescale": True,
                }
            )
    path = tmp_path / "bridge.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata = pd.DataFrame(
        {"unit_id": [1, 2], "session_num": [2, 2], "probe_letter": ["A", "A"]}
    )
    result = matched_mouse_timescale_session_group_table(
        metadata, bridge_path=path, groups=["A"], min_units=2
    )
    assert len(result) == 1
    assert result.iloc[0]["mean"] == 54.5
    assert result.iloc[0]["n_units"] == 2


def test_hierarchy_fit_summary_separates_visp_step_from_post_v1_slope():
    rows = []
    scores = {
        "Visual Coding VISp": -0.357,
        "LM": -0.093,
        "RL": -0.059,
        "LP": 0.105,
        "AL": 0.152,
        "PM": 0.327,
        "AM": 0.441,
    }
    for group, score in scores.items():
        values = [0.0, 0.01] if group == "Visual Coding VISp" else [-0.1, -0.09]
        target = rows if group != "Visual Coding VISp" else None
        if target is not None:
            for session, value in enumerate(values):
                rows.append({"dataset": "Post-V1", "metric": "m", "session_id": session, "group": group, "mean": value})
    v1 = pd.DataFrame(
        {"dataset": ["Allen-V1", "Allen-V1"], "metric": ["m", "m"], "session_id": [0, 1], "group": ["Visual Coding VISp"] * 2, "mean": [0.0, 0.01]}
    )
    result = hierarchy_fit_summary(
        pd.DataFrame(rows), v1, n_bootstrap=50, rng=np.random.default_rng(4)
    ).set_index("scope")
    assert result.loc["VISp_plus_post_V1", "slope_per_hierarchy_score"] < 0
    assert np.isclose(result.loc["post_V1_only", "slope_per_hierarchy_score"], 0, atol=1e-12)


def test_apply_harmonized_f1_f0_maps_only_frozen_source_rows(tmp_path):
    mouse_path = tmp_path / "mouse.csv"
    allen_path = tmp_path / "allen.csv"
    pd.DataFrame(
        {
            "unit_id": [1, 2],
            "f1_f0_dg_common_support": [0.8, 1.2],
            "default_qc": [True, True],
        }
    ).to_csv(mouse_path, index=False)
    pd.DataFrame(
        {
            "ecephys_unit_id": [10],
            "f1_f0_dg_harmonized": [0.9],
            "population_profile": ["common_qc"],
        }
    ).to_csv(allen_path, index=False)
    mouse, allen = apply_harmonized_f1_f0(
        pd.DataFrame({"unit_id": [1, 2]}),
        pd.DataFrame({"ecephys_unit_id": [10, 11]}),
        mouse_path=mouse_path,
        allen_path=allen_path,
    )
    assert mouse[HARMONIZED_F1_COLUMN].tolist() == [0.8, 1.2]
    assert allen[HARMONIZED_F1_COLUMN].iloc[0] == 0.9
    assert np.isnan(allen[HARMONIZED_F1_COLUMN].iloc[1])


def test_session_group_table_centers_within_session():
    frame = pd.DataFrame(
        {
            "session": [1] * 10 + [2] * 10,
            "group": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
            "metric": [1.0] * 5 + [3.0] * 5 + [10.0] * 5 + [14.0] * 5,
        }
    )
    result = session_group_table(
        frame,
        dataset="test",
        session_column="session",
        group_column="group",
        groups=["A", "B"],
        metric="metric",
        metric_label="metric",
        metric_index=2,
        min_units=5,
    )
    assert np.allclose(result.groupby("session_id")["centered_mean"].sum(), 0)
    assert result.set_index(["session_id", "group"]).loc[("1", "A"), "centered_mean"] == -1
    assert result.set_index(["session_id", "group"]).loc[("2", "B"), "centered_mean"] == 2


def test_session_spreads_require_complete_v1_and_minimum_hva_groups():
    rows = []
    for group, mean in zip(["A", "B", "C", "E"], [1, 2, 3, 4]):
        rows.append({"dataset": "Within-V1", "metric": "m", "session_id": "v1-full", "group": group, "mean": mean})
    for group, mean in zip(["A", "B", "C"], [1, 2, 3]):
        rows.append({"dataset": "Within-V1", "metric": "m", "session_id": "v1-missing", "group": group, "mean": mean})
    for group, mean in zip(["LM", "RL", "AL"], [4, 6, 8]):
        rows.append({"dataset": "Post-V1", "metric": "m", "session_id": "hva-three", "group": group, "mean": mean})
    for group, mean in zip(["LM", "RL"], [4, 6]):
        rows.append({"dataset": "Post-V1", "metric": "m", "session_id": "hva-two", "group": group, "mean": mean})
    result = session_spreads(pd.DataFrame(rows), min_hva_areas=3)
    assert set(result["session_id"]) == {"v1-full", "hva-three"}
    assert result.set_index("session_id").loc["v1-full", "n_groups"] == 4
    assert np.isclose(result.set_index("session_id").loc["hva-three", "spread_sd"], 2.0)


def test_bootstrap_and_leave_one_out_are_deterministic_and_session_level():
    v1 = np.array([1.0, 2.0, 3.0])
    hva = np.array([4.0, 5.0, 6.0, 7.0])
    first = bootstrap_mean_difference(v1, hva, n_bootstrap=250, rng=np.random.default_rng(7))
    second = bootstrap_mean_difference(v1, hva, n_bootstrap=250, rng=np.random.default_rng(7))
    assert first == second
    assert first[0] == np.mean(hva) - np.mean(v1)
    loo = leave_one_out_differences(v1, hva)
    assert len(loo) == len(v1) + len(hva)
    assert np.all(np.isfinite(loo))


def test_clustered_omega_resamples_complete_session_vectors():
    table = pd.DataFrame(
        {
            "session_id": ["1", "1", "2", "2", "3", "3"],
            "group": ["A", "B"] * 3,
            "mean": [0.0, 2.0, 1.0, 3.0, 2.0, 4.0],
        }
    )
    observed, first = clustered_omega_squared(
        table, n_bootstrap=100, rng=np.random.default_rng(11)
    )
    _, second = clustered_omega_squared(
        table, n_bootstrap=100, rng=np.random.default_rng(11)
    )
    expected = omega_squared(table["group"].to_numpy(str), table["mean"].to_numpy(float))
    assert observed == expected
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
