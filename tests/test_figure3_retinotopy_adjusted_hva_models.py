import numpy as np
import pandas as pd

from scripts.figure3_retinotopy_adjusted_hva_models import (
    HIERARCHY_SCORE,
    HVA_ORDER,
    add_common_box_flag,
    loso_predictions,
    permute_labels_within_sessions,
    robust_common_box,
    summarize_predictions,
)


def synthetic_cells(*, categorical_signal: bool) -> pd.DataFrame:
    rows = []
    effects = {"LM": -2.0, "RL": 1.0, "AL": -1.0, "PM": 2.0, "AM": 0.0}
    for session in range(12):
        raw = []
        for index, area in enumerate(HVA_ORDER):
            azimuth = -15 + 7 * index + (session % 3)
            elevation = -8 + 3 * ((index + session) % 5)
            outcome = 0.04 * azimuth - 0.03 * elevation
            if categorical_signal:
                outcome += effects[area]
            raw.append((area, azimuth, elevation, outcome))
        center = np.mean([row[3] for row in raw])
        for area, azimuth, elevation, outcome in raw:
            rows.append(
                {
                    "ecephys_session_id": str(session),
                    "area": area,
                    "rf_azimuth_deg": azimuth,
                    "rf_elevation_deg": elevation,
                    "centered_outcome": outcome - center,
                    "outcome": outcome,
                    "session_mean": center,
                    "hierarchy_score": HIERARCHY_SCORE[area],
                    "metric": "TTFS (ms)",
                    "population": "common_box",
                    "n_units": 10,
                }
            )
    return pd.DataFrame(rows)


def test_common_box_is_intersection_of_area_quantiles():
    rows = []
    for index, area in enumerate(HVA_ORDER):
        for value in range(10):
            rows.append(
                {
                    "area": area,
                    "relative_azimuth_deg": value + index,
                    "relative_elevation_deg": value - index,
                }
            )
    table = pd.DataFrame(rows)
    box = robust_common_box(table, 0.0)
    assert box == {
        "azimuth_low_deg": 4.0,
        "azimuth_high_deg": 9.0,
        "elevation_low_deg": 0.0,
        "elevation_high_deg": 5.0,
    }
    flagged = add_common_box_flag(table, box)
    assert flagged["inside_hva_common_box"].any()


def test_categorical_signal_improves_held_out_prediction_beyond_rf():
    cells = synthetic_cells(categorical_signal=True)
    predictions = loso_predictions(cells, fixed_alphas={
        "retinotopy": 0.01, "hierarchy": 0.01, "categorical_hva": 0.01,
    })
    summary = summarize_predictions(predictions).set_index("model")
    assert summary.loc["categorical_hva", "incremental_r2_vs_retinotopy"] > 0.4


def test_rf_only_signal_does_not_require_categorical_identity():
    cells = synthetic_cells(categorical_signal=False)
    predictions = loso_predictions(cells, fixed_alphas={
        "retinotopy": 0.01, "hierarchy": 0.01, "categorical_hva": 0.01,
    })
    summary = summarize_predictions(predictions).set_index("model")
    assert summary.loc["retinotopy", "cv_r2_vs_pooled"] > 0.95
    assert summary.loc["categorical_hva", "incremental_r2_vs_retinotopy"] < 0.01


def test_within_session_permutation_preserves_label_sets():
    cells = synthetic_cells(categorical_signal=True)
    shuffled = permute_labels_within_sessions(cells, np.random.default_rng(4))
    for session, part in cells.groupby("ecephys_session_id"):
        before = sorted(part["area"].tolist())
        after = sorted(shuffled.loc[shuffled.ecephys_session_id.eq(session), "permuted_area"].tolist())
        assert before == after
