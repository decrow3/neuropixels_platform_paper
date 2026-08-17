"""Synthetic recovery tests for MouseV2 parametric response models."""

from __future__ import annotations

import numpy as np
import pytest

from common.parametric_models import (
    elliptical_gaussian_rate,
    fit_grating_cell_model,
    fit_rf_cell_model,
    grating_rate,
)


def test_joint_grating_model_recovers_continuous_preferences():
    sf, tf, ori = np.meshgrid(
        np.array([0.02, 0.04, 0.08, 0.16, 0.32]),
        np.array([1.0, 2.0, 4.0, 8.0, 15.0]),
        np.array([0.0, 45.0, 90.0, 135.0]),
        indexing="ij",
    )
    parameters = np.array([0.5, 8.0, np.log2(0.1), 0.7, np.log2(5.0), 0.9, np.deg2rad(35), 2.0])
    rate = grating_rate(parameters, np.log2(sf.ravel()), np.log2(tf.ravel()), np.deg2rad(ori.ravel()))
    trials = np.full(rate.shape, 60.0)
    result = fit_grating_cell_model(sf.ravel(), tf.ravel(), ori.ravel(), rate * trials, trials)
    assert result["parametric_fit_success"]
    assert result["parametric_pref_sf_cpd"] == pytest.approx(0.1, rel=0.03)
    assert result["parametric_pref_tf_hz"] == pytest.approx(5.0, rel=0.03)
    assert result["parametric_pref_ori_deg"] == pytest.approx(35.0, abs=1.0)
    assert result["parametric_pseudo_r2"] > 0.99


def test_joint_grating_model_can_recover_peak_beyond_tested_support():
    sf, tf, ori = np.meshgrid(
        np.array([0.02, 0.04, 0.08, 0.16, 0.32]),
        np.array([1.0, 2.0, 4.0, 8.0, 15.0]),
        np.array([0.0, 45.0, 90.0, 135.0]),
        indexing="ij",
    )
    parameters = np.array([0.4, 7.0, np.log2(0.5), 0.8, np.log2(4.0), 0.9, 0.4, 1.5])
    rate = grating_rate(parameters, np.log2(sf.ravel()), np.log2(tf.ravel()), np.deg2rad(ori.ravel()))
    trials = np.full(rate.shape, 60.0)
    result = fit_grating_cell_model(sf.ravel(), tf.ravel(), ori.ravel(), rate * trials, trials)
    assert result["parametric_pref_sf_cpd"] == pytest.approx(0.5, rel=0.05)
    assert not result["parametric_sf_in_tested_range"]
    assert not result["parametric_sf_at_extrapolation_bound"]


def test_elliptical_rf_model_recovers_subgrid_center():
    x, y = np.meshgrid(np.arange(-40.0, 41.0, 10.0), np.arange(-40.0, 41.0, 10.0))
    parameters = np.array([0.3, 6.0, 7.5, -12.0, 18.0, 9.0, np.deg2rad(25.0)])
    rate = elliptical_gaussian_rate(parameters, x.ravel(), y.ravel())
    trials = np.full(rate.shape, 60.0)
    result = fit_rf_cell_model(x.ravel(), y.ravel(), rate * trials, trials)
    assert result["rf_fit_success"]
    assert result["rf_center_x_deg"] == pytest.approx(7.5, abs=0.3)
    assert result["rf_center_y_deg"] == pytest.approx(-12.0, abs=0.3)
    assert result["rf_sigma_major_deg"] == pytest.approx(18.0, rel=0.05)
    assert result["rf_sigma_minor_deg"] == pytest.approx(9.0, rel=0.05)
    assert result["rf_pseudo_r2"] > 0.99
