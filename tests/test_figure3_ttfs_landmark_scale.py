import numpy as np
import pandas as pd

from scripts.figure3_ttfs_landmark_scale import (
    bootstrap_landmarks,
    compute_pair_distances,
    landmark_point_summary,
    session_landmarks,
    unbiased_squared_mean_distance,
)


def test_unbiased_distance_removes_finite_cell_mean_noise():
    x = np.array([0.0, 2.0])
    y = np.array([2.0, 4.0])
    result = unbiased_squared_mean_distance(x, y)
    assert np.isclose(result["naive_squared_mean_difference"], 4.0)
    assert np.isclose(result["sampling_noise_correction"], 2.0)
    assert np.isclose(result["corrected_squared_mean_distance"], 2.0)


def test_pair_construction_uses_expected_within_and_between_pairs():
    mouse_rows = []
    for group, values in {"A": [1, 2], "B": [3, 4], "C": [5, 6]}.items():
        for index, value in enumerate(values):
            mouse_rows.append(
                {"session_id": "m1", "group": group, "unit_id": f"{group}{index}", "value": value}
            )
    allen_rows = []
    for group, values in {"V1": [1, 2], "LM": [3, 4], "AL": [5, 6]}.items():
        for index, value in enumerate(values):
            allen_rows.append(
                {"session_id": "a1", "group": group, "unit_id": f"{group}{index}", "value": value}
            )
    result = compute_pair_distances(pd.DataFrame(mouse_rows), pd.DataFrame(allen_rows))
    assert result.landmark.value_counts().to_dict() == {
        "within_v1": 3,
        "v1_to_hva": 2,
        "hva_to_hva": 1,
    }


def test_scale_places_midpoint_hva_distance_at_half():
    sessions = pd.DataFrame(
        {
            "landmark": ["within_v1", "within_v1", "hva_to_hva", "hva_to_hva", "v1_to_hva", "v1_to_hva"],
            "session_id": ["m1", "m2", "a1", "a2", "a1", "a2"],
            "distance": [1.0, 1.0, 3.0, 3.0, 5.0, 5.0],
            "naive_distance": [1.0, 1.0, 3.0, 3.0, 5.0, 5.0],
            "sampling_noise_correction": [0.0] * 6,
            "n_pairs": [1] * 6,
            "cells_in_pairs": [4] * 6,
        }
    )
    point, derived = landmark_point_summary(sessions)
    assert np.isclose(derived["hva_landmark_position"], 0.5)
    bootstrap = bootstrap_landmarks(sessions, repetitions=20, seed=4)
    assert np.allclose(bootstrap["hva_landmark_position"], 0.5)


def test_session_landmarks_equal_weight_pairs_within_session():
    pairs = pd.DataFrame(
        {
            "landmark": ["within_v1", "within_v1", "within_v1"],
            "session_id": ["s1", "s1", "s2"],
            "group_a": ["A", "A", "A"],
            "group_b": ["B", "C", "B"],
            "n_a": [2, 2, 2],
            "cells_in_pair": [4, 4, 4],
            "naive_squared_mean_difference": [1.0, 3.0, 10.0],
            "sampling_noise_correction": [0.0, 0.0, 0.0],
            "corrected_squared_mean_distance": [1.0, 3.0, 10.0],
        }
    )
    result = session_landmarks(pairs).set_index("session_id")
    assert np.isclose(result.loc["s1", "distance"], 2.0)
    assert np.isclose(result.loc["s2", "distance"], 10.0)
