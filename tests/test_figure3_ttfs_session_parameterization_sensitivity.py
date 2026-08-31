import numpy as np

from scripts.figure3_ttfs_session_parameterization_sensitivity import (
    log_noncentered_session_density,
    sample_coefficients,
)


def test_noncentered_session_density_is_finite_and_prefers_matching_scale():
    known = np.array([0.1, 0.1])
    signal = np.array([1.0, -1.0])
    target = np.array([0.5, -0.5])
    matching = log_noncentered_session_density(np.log(0.5), target, signal, known, 0.1)
    too_large = log_noncentered_session_density(np.log(2.0), target, signal, known, 0.1)
    assert np.isfinite(matching)
    assert matching > too_large


def test_coefficient_sampler_returns_expected_shape():
    y = np.array([1.0, 2.0, 3.0])
    se = np.array([0.2, 0.2, 0.2])
    design = np.column_stack([np.ones(3), np.array([0.0, 1.0, 0.0])])
    draw = sample_coefficients(y, se, design, np.array([0.04, 1.0]), 0.3, np.random.default_rng(3))
    assert draw.shape == (2,)
    assert np.all(np.isfinite(draw))
