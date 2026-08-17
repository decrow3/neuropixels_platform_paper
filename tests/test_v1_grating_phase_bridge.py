"""Tests for trial-amplitude and phase-coherence decomposition."""

import numpy as np
import pytest

from scripts.v1_grating_phase_bridge import fourier_decomposition


def sinusoid_trials(phases: list[float], frequency_hz: float = 4.0) -> np.ndarray:
    time = np.arange(1000) / 1000.0
    return np.stack(
        [2.0 + np.cos(2 * np.pi * frequency_hz * time + phase) for phase in phases]
    )


def test_aligned_trials_preserve_coherent_amplitude():
    result = fourier_decomposition(sinusoid_trials([0.0] * 15), 4.0)
    assert result["weighted_phase_coherence"] == pytest.approx(1.0)
    assert result["coherent_f1_hz"] == pytest.approx(result["mean_trial_f1_hz"])
    assert result["unweighted_phase_ppc"] == pytest.approx(1.0)


def test_phase_scrambling_preserves_trial_amplitude_but_cancels_coherence():
    aligned = fourier_decomposition(sinusoid_trials([0.0] * 16), 4.0)
    scrambled = fourier_decomposition(
        sinusoid_trials(list(np.linspace(0, 2 * np.pi, 16, endpoint=False))), 4.0
    )
    assert scrambled["mean_trial_f1_hz"] == pytest.approx(
        aligned["mean_trial_f1_hz"]
    )
    assert scrambled["weighted_phase_coherence"] < 1e-10
    assert scrambled["coherent_f1_hz"] < 1e-10
    assert scrambled["unweighted_phase_ppc"] == pytest.approx(-1 / 15)
