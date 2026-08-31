from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.figure3_multivariate_area_fingerprints import (
    METRICS,
    balanced_accuracy,
    build_fingerprint_table,
    fast_loso_summary,
    permute_labels_within_sessions,
    run_loso_centroid,
)


def synthetic_table(*, stable: bool, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = ["A", "B", "C", "E"]
    rows = []
    effects = {
        "A": np.array([-1.2, -0.7, 0.8]),
        "B": np.array([-0.4, 1.1, -0.5]),
        "C": np.array([0.6, -1.0, -0.9]),
        "E": np.array([1.0, 0.6, 0.7]),
    }
    for session in range(12):
        noise = rng.normal(0, 0.18, size=(len(groups), len(METRICS)))
        for index, group in enumerate(groups):
            values = effects[group] + noise[index] if stable else noise[index]
            row = {
                "dataset": "Within-V1",
                "session_id": str(session),
                "group": group,
                "available_metrics": 3,
            }
            row.update({f"value::{metric}": value for metric, value in zip(METRICS, values)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_loso_centroid_recovers_stable_identity() -> None:
    table = synthetic_table(stable=True)
    predictions, _, summary = run_loso_centroid(table, "Within-V1")
    assert balanced_accuracy(predictions) > 0.95
    assert summary["identity_r2"] > 0.8
    fast = fast_loso_summary(table, "Within-V1")
    assert np.isclose(fast["balanced_accuracy"], summary["balanced_accuracy"])
    assert np.isclose(fast["identity_r2"], summary["identity_r2"])


def test_loso_centroid_does_not_invent_null_identity() -> None:
    _, _, summary = run_loso_centroid(
        synthetic_table(stable=False), "Within-V1"
    )
    assert summary["balanced_accuracy"] < 0.5
    assert summary["identity_r2"] < 0.2


def test_within_session_permutation_preserves_session_label_sets() -> None:
    table = synthetic_table(stable=True)
    permuted = permute_labels_within_sessions(table, np.random.default_rng(11))
    for session_id, part in permuted.groupby("session_id"):
        original = table.loc[table["session_id"].eq(session_id), "group"]
        assert sorted(original) == sorted(part["permuted_group"])


def test_missing_metric_is_not_used_as_a_feature() -> None:
    table = synthetic_table(stable=True)
    table.loc[table.index[0], f"value::{METRICS[1]}"] = np.nan
    predictions, _, _ = run_loso_centroid(table, "Within-V1")
    row = predictions.loc[
        predictions["session_id"].eq("0") & predictions["true_group"].eq("A")
    ].iloc[0]
    assert row["eligible"]
    assert row["n_features_used"] == 2


def test_cortical_hva_center_excludes_thalamic_lp() -> None:
    rows = []
    for metric in METRICS[:2]:
        for group, value in [("LM", 10.0), ("RL", 14.0), ("LP", 100.0)]:
            session_mean = (10.0 + 14.0 + 100.0) / 3.0
            rows.append(
                {
                    "dataset": "Post-V1",
                    "metric": metric,
                    "session_id": "s1",
                    "group": group,
                    "mean": value,
                    "centered_mean": value - session_mean,
                    "session_mean": session_mean,
                    "n_groups_in_session": 3,
                    "n_units": 10,
                }
            )
    table, _ = build_fingerprint_table(pd.DataFrame(rows), "Post-V1")
    lm = table.loc[table["group"].eq("LM")].iloc[0]
    rl = table.loc[table["group"].eq("RL")].iloc[0]
    for metric in METRICS[:2]:
        assert np.isclose(lm[f"value::{metric}"], -2.0)
        assert np.isclose(rl[f"value::{metric}"], 2.0)
