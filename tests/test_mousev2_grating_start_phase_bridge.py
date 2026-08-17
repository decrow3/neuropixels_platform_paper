"""Tests for the source-derived MouseV2 grating start-phase bridge."""

import numpy as np
import pytest

from scripts.mousev2_grating_start_phase_bridge import (
    phase_adjusted_metrics,
    presentation_start_phase_cycles,
)


def test_source_schedule_has_expected_temporal_frequency_phase_support():
    ordinal = np.arange(20)
    expected = {1.0: 4, 2.0: 2, 4.0: 1, 8.0: 1, 15.0: 4}
    for temporal_frequency_hz, unique_phases in expected.items():
        phases = presentation_start_phase_cycles(
            ordinal,
            temporal_frequency_hz,
            fps=60.0,
            presentation_stride_frames=135,
        )
        assert len(np.unique(np.round(phases, decimals=9))) == unique_phases


def test_source_phase_rotation_recovers_scrambled_periodic_response():
    temporal_frequency_hz = 1.0
    ordinal = np.arange(16)
    phases = presentation_start_phase_cycles(
        ordinal,
        temporal_frequency_hz,
        fps=60.0,
        presentation_stride_frames=135,
    )
    time_s = np.arange(1000) / 1000.0
    trials = np.stack(
        [
            (
                2.0
                + np.cos(
                    2 * np.pi * temporal_frequency_hz * time_s + 2 * np.pi * phase
                )
            )
            / 1000.0
            for phase in phases
        ]
    )
    result = phase_adjusted_metrics(
        trials,
        temporal_frequency_hz,
        phases,
        permutations=25,
        seed=7,
    )
    assert result["mean_trial_f1_hz"] == pytest.approx(1.0)
    assert result["raw_weighted_phase_coherence"] < 1e-10
    assert result["source_corrected_weighted_phase_coherence"] == pytest.approx(1.0)
    assert result["source_corrected_coherent_f1_hz"] == pytest.approx(1.0)
    assert result["source_gain_over_permutation"] > 0.5


def test_phase_stable_temporal_frequency_is_unchanged():
    temporal_frequency_hz = 4.0
    ordinal = np.arange(15)
    phases = presentation_start_phase_cycles(
        ordinal,
        temporal_frequency_hz,
        fps=60.0,
        presentation_stride_frames=135,
    )
    time_s = np.arange(1000) / 1000.0
    trials = np.stack(
        [
            (2.0 + np.cos(2 * np.pi * temporal_frequency_hz * time_s)) / 1000.0
        ]
        * len(phases)
    )
    result = phase_adjusted_metrics(
        trials,
        temporal_frequency_hz,
        phases,
        permutations=10,
        seed=9,
    )
    assert result["source_start_phase_count"] == 1
    assert result["raw_weighted_phase_coherence"] == pytest.approx(1.0)
    assert result["source_corrected_weighted_phase_coherence"] == pytest.approx(1.0)
    assert result["phase_permutation_weighted_phase_coherence"] == pytest.approx(1.0)
