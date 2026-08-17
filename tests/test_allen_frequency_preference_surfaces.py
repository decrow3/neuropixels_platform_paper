"""Focused tests for nonlinear Allen SF/TF preference surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.allen_frequency_preference_surfaces import (
    add_v1_differences,
    polar_coordinates,
    rf_occupancy_counts,
    session_balanced_gaussian_surface,
    validate_bandwidths,
)
from scripts.render_allen_bo11_simultaneous_v1_hva_session_maps import (
    simultaneous_v1_hva_sessions,
)
from scripts.allen_bo11_affine_session_alignment import (
    apply_affine,
    fit_regularized_affine,
)
from scripts.allen_bo11_tuning_driven_limited_affine import affine_matrix
from scripts.allen_bo11_tuning_weighted_session_surfaces import (
    normalize_unit_weights,
    tuning_quality_components,
    weighted_gaussian_surface,
)
from scripts.render_allen_bo11_registration_comparison import (
    v1_rf_center_translation_parameters,
)
from scripts.allen_bo11_noncenter_similarity_alignment import (
    pack_parameters,
    robust_area_standardize,
)
from scripts.allen_bo11_ccf_retinotopy_alignment import (
    leave_one_session_out_predictions,
    session_balanced_weights,
    session_translations,
)


def test_bandwidth_validation_requires_positive_primary_member():
    assert validate_bandwidths([16, 8, 12, 12], 12) == (8.0, 12.0, 16.0)
    with pytest.raises(ValueError, match="positive"):
        validate_bandwidths([0, 12], 12)
    with pytest.raises(ValueError, match="included"):
        validate_bandwidths([8, 16], 12)


def test_session_balancing_prevents_unit_rich_session_domination():
    points = np.zeros((101, 2))
    values = np.r_[np.zeros(100), 2.0]
    sessions = np.concatenate([np.repeat("many", 100), np.array(["one"])])
    result = session_balanced_gaussian_surface(
        points,
        values,
        sessions,
        np.array([[0.0, 0.0]]),
        bandwidth_deg=10,
        minimum_effective_sessions=1,
        minimum_local_units=1,
    )
    assert np.isclose(result["estimate_log2"][0], 1.0)
    assert np.isclose(result["effective_sessions"][0], 2.0)


def test_ccf_model_balances_total_weight_across_sessions():
    sessions = np.array([1, 1, 1, 2])
    weights = session_balanced_weights(sessions)
    assert np.isclose(weights[sessions == 1].sum(), weights[sessions == 2].sum())


def test_ccf_to_rf_heldout_residual_recovers_session_translation():
    rng = np.random.default_rng(4)
    rows = []
    offsets = {1: (0.0, 0.0), 2: (3.0, -2.0), 3: (-4.0, 5.0), 4: (6.0, 1.0)}
    for session_id, (az_offset, el_offset) in offsets.items():
        for unit_id in range(40):
            ccf = rng.normal(size=3)
            rows.append(
                {
                    "ecephys_session_id": session_id,
                    "anterior_posterior_ccf_coordinate": ccf[0],
                    "dorsal_ventral_ccf_coordinate": ccf[1],
                    "left_right_ccf_coordinate": ccf[2],
                    "azimuth_rf": 20 + 8 * ccf[0] - 3 * ccf[2] + az_offset,
                    "elevation_rf": -5 + 4 * ccf[1] + 2 * ccf[2] + el_offset,
                }
            )
    table = pd.DataFrame(rows)
    predictions = leave_one_session_out_predictions(table, degree=1, ridge=1e-6)
    transforms = session_translations(table, predictions, bound_deg=15).set_index("ecephys_session_id")
    for session_id, (az_offset, el_offset) in offsets.items():
        training_offsets = np.array([value for key, value in offsets.items() if key != session_id])
        expected = training_offsets.mean(axis=0) - np.array([az_offset, el_offset])
        assert np.isclose(transforms.loc[session_id, "translation_azimuth_deg"], expected[0], atol=0.4)
        assert np.isclose(transforms.loc[session_id, "translation_elevation_deg"], expected[1], atol=0.4)


def test_surface_masks_cells_without_enough_local_support():
    result = session_balanced_gaussian_surface(
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([0.0, 1.0]),
        np.array([1, 2]),
        np.array([[0.0, 0.0], [100.0, 100.0]]),
        bandwidth_deg=5,
        minimum_effective_sessions=2,
        minimum_local_units=2,
    )
    assert result["supported"].tolist() == [True, False]
    assert np.isfinite(result["estimate_log2"][0])
    assert np.isnan(result["estimate_log2"][1])


def test_v1_difference_requires_shared_support():
    rows = [
        {"preference": "sf", "bandwidth_deg": 12.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "area": "V1", "estimate_log2": -4.0, "supported": True},
        {"preference": "sf", "bandwidth_deg": 12.0, "azimuth_deg": 0.0, "elevation_deg": 0.0, "area": "LM", "estimate_log2": -3.0, "supported": True},
        {"preference": "sf", "bandwidth_deg": 12.0, "azimuth_deg": 1.0, "elevation_deg": 0.0, "area": "V1", "estimate_log2": -4.0, "supported": False},
        {"preference": "sf", "bandwidth_deg": 12.0, "azimuth_deg": 1.0, "elevation_deg": 0.0, "area": "LM", "estimate_log2": -3.0, "supported": True},
    ]
    result = add_v1_differences(pd.DataFrame(rows))
    lm = result.loc[result["area"].eq("LM")].sort_values("azimuth_deg")
    assert lm["shared_v1_support"].tolist() == [True, False]
    assert lm["delta_from_v1_log2"].iloc[0] == 1.0
    assert np.isnan(lm["delta_from_v1_log2"].iloc[1])


def test_polar_coordinates_use_requested_visual_field_center():
    theta, radius = polar_coordinates(
        np.array([0.0, 10.0, 0.0]),
        np.array([20.0, 20.0, 30.0]),
        center_azimuth_deg=0.0,
        center_elevation_deg=20.0,
    )
    assert np.allclose(radius, [0.0, 10.0, 10.0])
    assert np.allclose(theta, [0.0, 0.0, np.pi / 2])


def test_rf_occupancy_counts_units_in_requested_bins():
    units = pd.DataFrame(
        {
            "azimuth_rf": [10.0, 14.0, 20.0, 90.0],
            "elevation_rf": [-30.0, -26.0, -20.0, 50.0],
        }
    )
    counts = rf_occupancy_counts(
        units,
        np.array([5.0, 15.0, 25.0, 95.0]),
        np.array([-35.0, -25.0, -15.0, 55.0]),
    )
    assert counts.sum() == 4
    assert counts[0, 0] == 2
    assert counts[1, 1] == 1
    assert counts[2, 2] == 1


def test_simultaneous_v1_hva_gate_requires_both_tuning_populations():
    units = pd.DataFrame(
        {
            "ecephys_session_id": [1, 1, 2, 2],
            "area": ["V1", "LM", "V1", "LM"],
            "tuning_eligible_sf": [True, True, True, True],
            "tuning_eligible_tf": [True, True, True, False],
        }
    )
    assert simultaneous_v1_hva_sessions(units) == [1]


def test_regularized_affine_keeps_identical_coordinates_fixed():
    points = np.array([[10.0, -20.0], [50.0, 0.0], [80.0, 30.0]])
    linear, translation = fit_regularized_affine(
        points,
        points,
        np.ones(len(points)),
        ridge_lambda=0.1,
    )
    assert np.allclose(linear, np.eye(2))
    assert np.allclose(translation, 0.0)
    assert np.allclose(apply_affine(points, linear, translation), points)


def test_limited_affine_parameterization_prevents_reflection():
    matrix, translation = affine_matrix(
        np.array([15.0, -15.0, 12.0, np.log(0.85), np.log(1.15), 0.12])
    )
    assert np.linalg.det(matrix) > 0
    assert np.allclose(translation, [15.0, -15.0])


def test_tuning_quality_weight_increases_with_strength_rate_and_stability():
    _, _, _, weights = tuning_quality_components(
        np.array([0.2, 0.8, 0.8, 0.8]),
        np.array([2.0, 2.0, 8.0, 8.0]),
        np.array([2.0, 2.0, 2.0, 0.5]),
    )
    assert weights[1] > weights[0]
    assert weights[2] > weights[1]
    assert weights[3] > weights[2]


def test_normalized_tuning_weights_preserve_mean_contribution():
    weights = normalize_unit_weights(np.array([0.01, 0.2, 1.0, 20.0]))
    assert np.isclose(np.mean(weights), 1.0)
    assert weights.min() > 0
    assert weights.max() / weights.min() <= 16.0


def test_weighted_surface_uses_quality_and_effective_local_support():
    result = weighted_gaussian_surface(
        np.zeros((3, 2)),
        np.array([0.0, 0.0, 2.0]),
        np.array([1.0, 1.0, 4.0]),
        np.array([[0.0, 0.0]]),
        bandwidth_deg=5.0,
        minimum_effective_local_units=2.0,
    )
    assert np.isclose(result["estimate_log2"][0], 4.0 / 3.0)
    assert np.isclose(result["effective_local_units"][0], 36.0 / 18.0)
    assert result["supported"][0]


def test_v1_rf_center_registration_is_translation_only_and_session_balanced():
    units = pd.DataFrame(
        {
            "cohort": ["Brain Observatory 1.1"] * 5,
            "area": ["V1"] * 5,
            "ecephys_session_id": [1, 1, 1, 2, 2],
            "ecephys_unit_id": [10, 11, 12, 20, 21],
            "ecephys_probe_id": [100, 100, 100, 200, 200],
            "azimuth_rf": [0.0, 10.0, 20.0, 30.0, 40.0],
            "elevation_rf": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    parameters, audit = v1_rf_center_translation_parameters(units, [1, 2])
    assert np.allclose(parameters[1], [12.5, 12.5, 0, 0, 0, 0])
    assert np.allclose(parameters[2], [-12.5, -12.5, 0, 0, 0, 0])
    aligned_azimuth = audit["v1_rf_center_azimuth_deg"] + audit["translation_azimuth_deg"]
    assert np.allclose(aligned_azimuth, 22.5)


def test_noncenter_similarity_is_isotropic_and_orientation_preserving():
    packed = pack_parameters(np.array([3.0, -2.0, 5.0, np.log(1.05)]), "similarity")
    assert np.allclose(packed, [3.0, -2.0, 5.0, np.log(1.05), np.log(1.05), 0.0])
    matrix, _ = affine_matrix(packed)
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert np.allclose(singular, 1.05)
    assert np.linalg.det(matrix) > 0


def test_noncenter_features_are_robustly_standardized_within_area():
    table = pd.DataFrame({"area": ["V1"] * 4 + ["LM"] * 4, "value": [1, 2, 3, 4, 10, 20, 30, 40]})
    standardized = robust_area_standardize(table, "value", "identity")
    assert np.isclose(standardized.iloc[:4].median(), 0.0)
    assert np.isclose(standardized.iloc[4:].median(), 0.0)
    assert np.isclose(standardized.iloc[:4].quantile(0.75) - standardized.iloc[:4].quantile(0.25), 1.0)
