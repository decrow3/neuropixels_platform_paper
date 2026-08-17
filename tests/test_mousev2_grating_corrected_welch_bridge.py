import numpy as np
from scipy import signal

from common.drifting_gratings import welch_modulation_index
from scripts.mousev2_grating_corrected_welch_bridge import (
    corrected_welch_metrics,
    target_coefficients,
    target_component,
    welch_spectral_metrics,
)


def test_vectorized_welch_matches_released_scalar_metric():
    rng = np.random.default_rng(123)
    responses = rng.normal(size=(4, 1000))

    observed = welch_spectral_metrics(responses, 8.0)["mod_idx"]
    expected = np.array(
        [welch_modulation_index(response, 8.0) for response in responses]
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
    assert welch_spectral_metrics(np.zeros(1000), 8.0)["mod_idx"] == 0.0


def test_target_component_reconstructs_only_the_requested_dft_bin():
    rng = np.random.default_rng(456)
    response = rng.normal(size=1000)
    coefficient = target_coefficients(response[None, :], 4.0)[0]
    component = target_component(coefficient, 4.0)
    residual = response - component

    response_dft = np.fft.fft(response)
    residual_dft = np.fft.fft(residual)
    changed = np.flatnonzero(np.abs(response_dft - residual_dft) > 1e-10)

    np.testing.assert_array_equal(changed, np.array([4, 996]))
    np.testing.assert_allclose(residual_dft[[4, 996]], 0.0, atol=1e-10)
    np.testing.assert_allclose(component.mean(), 0.0, atol=1e-15)


def test_source_phase_correction_restores_cancelled_carrier():
    samples = 1000
    temporal_frequency_hz = 4.0
    time_s = np.arange(samples) / 1000.0
    phases = np.tile(np.array([0.0, 0.25, 0.5, 0.75]), 4)
    noncarrier = 0.01 + 0.002 * np.cos(2 * np.pi * 3.0 * time_s + 0.2)
    counts = np.stack(
        [
            noncarrier
            + 0.006
            * np.cos(2 * np.pi * temporal_frequency_hz * time_s + 2 * np.pi * phase)
            for phase in phases
        ]
    )

    metrics = corrected_welch_metrics(
        counts,
        temporal_frequency_hz,
        phases,
        permutations=200,
        seed=10,
    )

    assert metrics["source_corrected_target_psd"] > metrics["raw_target_psd"]
    assert metrics["source_corrected_mod_idx"] > metrics["raw_mod_idx"]
    assert (
        metrics["source_corrected_mod_idx"]
        > metrics["phase_permutation_mod_idx_975"]
    )
    assert metrics["source_corrected_mod_idx"] > metrics["opposite_sign_mod_idx"]
    assert metrics["raw_reconstruction_max_abs_error"] < 1e-15
    assert abs(metrics["source_mean_rate_change"]) < 1e-15


def test_stable_source_phase_is_an_exact_negative_control():
    rng = np.random.default_rng(789)
    counts = rng.poisson(0.01, size=(15, 1000)).astype(float)
    # The 135-frame MouseV2 stride advances 4/8-Hz gratings by an integer
    # number of cycles, so the protocol-derived stable phase is exactly zero.
    phases = np.zeros(15)

    metrics = corrected_welch_metrics(
        counts,
        8.0,
        phases,
        permutations=20,
        seed=11,
    )

    for prefix in ("source_corrected", "opposite_sign"):
        np.testing.assert_allclose(
            metrics[f"{prefix}_mod_idx"], metrics["raw_mod_idx"], atol=1e-12
        )
        np.testing.assert_allclose(
            metrics[f"{prefix}_target_psd"], metrics["raw_target_psd"], atol=1e-12
        )
    np.testing.assert_allclose(
        metrics["phase_permutation_mod_idx_mean"],
        metrics["raw_mod_idx"],
        atol=1e-12,
    )
    assert metrics["source_start_phase_count"] == 1
