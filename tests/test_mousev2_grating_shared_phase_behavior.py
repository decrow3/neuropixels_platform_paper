"""Tests for cross-probe residual phase and behavior association."""

import json

import numpy as np
import pandas as pd

from scripts.mousev2_grating_shared_phase_behavior import (
    complex_phase_association,
    refresh_manifest,
    residualize_metric_against_block_time,
    shared_phase_condition,
    summarize_shared_center,
    trial_behavior_table,
)


def synthetic_shared_coefficients() -> tuple[np.ndarray, np.ndarray]:
    probes = np.array(list("ABCE") * 4)
    intrinsic = np.linspace(-np.pi, np.pi, len(probes), endpoint=False)
    shared = 1.1 * np.sin(np.linspace(0, 2 * np.pi, 15, endpoint=False))
    coefficients = np.exp(1j * (intrinsic[:, None] + shared[None, :]))
    return coefficients, probes


def test_other_probe_consensus_recovers_shared_trial_phase():
    coefficients, probes = synthetic_shared_coefficients()
    result = shared_phase_condition(coefficients, probes, permutations=40, seed=4)
    source = np.mean(result["source_coherence"])
    adjusted = np.mean(result["cross_probe_adjusted_coherence"])
    null = result["permutations"]["mean_cross_probe_adjusted_coherence"].mean()
    assert result["cross_probe_alignment"] > 0.8
    assert result["cross_probe_alignment"] > result["permutations"][
        "cross_probe_alignment"
    ].max()
    assert adjusted > source + 0.15
    assert adjusted > null + 0.15


def test_cross_probe_prediction_does_not_require_same_probe_units():
    coefficients, probes = synthetic_shared_coefficients()
    result = shared_phase_condition(coefficients, probes, permutations=10, seed=8)
    assert np.isfinite(result["unit_cross_probe_alignment"]).all()
    assert np.min(result["unit_cross_probe_alignment"]) > 0.75


def test_condition_stratified_behavior_phase_association_exceeds_shuffle():
    rows = []
    for condition_index in range(3):
        x = np.linspace(-1, 1, 15)
        phase = 1.2 * x + 0.4 * condition_index
        for value, angle in zip(x, phase):
            rows.append(
                {
                    "condition_id": f"condition_{condition_index}",
                    "behavior": value,
                    "population_phase_real": np.cos(angle),
                    "population_phase_imag": np.sin(angle),
                }
            )
    observed, null, presentations = complex_phase_association(
        pd.DataFrame(rows),
        "behavior",
        permutations=100,
        seed=12,
    )
    assert presentations == 45
    assert observed > 0.5
    assert observed > np.quantile(null, 0.99)


def test_behavior_phase_association_uses_angle_not_population_magnitude():
    angle = np.linspace(-1.0, 1.0, 15)
    magnitude = np.geomspace(0.01, 100.0, len(angle))
    table = pd.DataFrame(
        {
            "condition_id": "condition_0",
            "behavior": angle,
            "population_phase_real": magnitude * np.cos(angle),
            "population_phase_imag": magnitude * np.sin(angle),
        }
    )
    scaled = table.copy()
    scaled[["population_phase_real", "population_phase_imag"]] /= magnitude[:, None]
    observed, _, _ = complex_phase_association(
        table, "behavior", permutations=10, seed=2
    )
    scaled_observed, _, _ = complex_phase_association(
        scaled, "behavior", permutations=10, seed=2
    )
    assert np.isclose(observed, scaled_observed)


def test_block_time_residualization_removes_quadratic_drift():
    ordinal = np.tile(np.arange(20, dtype=float), 2)
    condition = np.repeat(["condition_0", "condition_1"], 20)
    standardized_time = (ordinal - ordinal.mean()) / ordinal.std()
    behavior = (
        np.repeat([3.0, -2.0], 20)
        + 1.2 * standardized_time
        - 0.7 * standardized_time**2
    )
    table = pd.DataFrame(
        {
            "condition_id": condition,
            "presentation_ordinal": ordinal,
            "behavior": behavior,
        },
        index=np.arange(100, 140),
    )
    residual = residualize_metric_against_block_time(table, "behavior")
    assert np.nanmax(np.abs(residual)) < 1e-12


def test_eye_summary_requires_half_window_coverage():
    starts = np.arange(300, dtype=float) * 2.0
    table = pd.DataFrame(
        {
            "id": np.arange(300),
            "start_time": starts,
            "orientation": np.tile([0.0, 45.0, 90.0, 135.0], 75),
            "temporal_frequency": np.tile([1.0, 2.0, 4.0, 8.0, 15.0], 60),
            "spatial_frequency": 0.04,
            "contrast": 0.8,
        }
    )
    eye_time = np.concatenate([start + np.array([0.1, 0.3, 0.5, 0.7]) for start in starts])
    blink = np.zeros(len(eye_time))
    blink[:3] = 1.0
    blink[4:6] = 1.0
    signals = {
        "running": np.ones(len(eye_time)),
        "running_time": eye_time,
        "pupil_area": np.full(len(eye_time), 100.0),
        "pupil_x": np.full(len(eye_time), 20.0),
        "pupil_y": np.full(len(eye_time), 30.0),
        "eye_time": eye_time,
        "blink": blink,
    }
    result = trial_behavior_table(table, signals)
    assert result.loc[0, "valid_eye_fraction"] == 0.25
    assert np.isnan(result.loc[0, "pupil_x_median_stim"])
    assert result.loc[1, "valid_eye_fraction"] == 0.5
    assert result.loc[1, "pupil_x_median_stim"] == 20.0


def test_shared_center_uses_equal_session_permutation_null():
    sessions = pd.DataFrame(
        {
            "site": ["site2", "site3"],
            "session_id": [2, 3],
            "cross_probe_alignment": [0.5, 0.7],
            "cross_probe_alignment_p": [0.01, 0.02],
            "mean_source_corrected_coherence": [0.4, 0.6],
            "mean_cross_probe_adjusted_coherence": [0.45, 0.55],
            "mean_cross_probe_coherence_gain": [0.05, -0.05],
        }
    )
    conditions = pd.DataFrame(
        {
            "site": ["site2", "site3"],
            "session_id": [2, 3],
            "condition_id": ["c", "c"],
            "n_units": [10, 20],
        }
    )
    permutations = pd.DataFrame(
        {
            "site": ["site2", "site2", "site3", "site3"],
            "session_id": [2, 2, 3, 3],
            "condition_id": ["c", "c", "c", "c"],
            "permutation": [0, 1, 0, 1],
            "cross_probe_alignment": [0.1, 0.2, 0.3, 0.4],
            "mean_cross_probe_adjusted_coherence": [0.2, 0.3, 0.4, 0.5],
        }
    )
    center = summarize_shared_center(sessions, conditions, permutations).iloc[0]
    assert np.isclose(center.equal_session_cross_probe_alignment, 0.6)
    assert np.isclose(center.permutation_cross_probe_alignment_mean, 0.25)
    assert np.isclose(center.cross_probe_alignment_p, 1 / 3)


def test_render_refresh_preserves_analysis_script_hash(tmp_path):
    manifest = {
        "inputs": [],
        "code": {
            "script_sha256": "original-analysis-hash",
            "analysis_script_sha256": "original-analysis-hash",
        },
    }
    path = tmp_path / "import_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_manifest(tmp_path)
    refreshed = json.loads(path.read_text(encoding="utf-8"))
    assert refreshed["code"]["script_sha256"] == "original-analysis-hash"
    assert refreshed["code"]["analysis_script_sha256"] == "original-analysis-hash"
    assert refreshed["code"]["render_script_sha256"] != "original-analysis-hash"
