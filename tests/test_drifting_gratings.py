"""Tests for the full-condition MouseV2 drifting-grating metrics."""

from __future__ import annotations

import ast
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import signal
from scipy.fftpack import fft

from common.drifting_gratings import (
    _bin_trial_spike_counts,
    compute_drifting_grating_metrics,
    f1_f0_from_trial_counts,
    welch_modulation_index,
)


ROOT = Path(__file__).resolve().parents[1]
ALLENSDK_SOURCE = Path(
    "/home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/lib/"
    "python3.10/site-packages/allensdk/brain_observatory/ecephys/"
    "stimulus_analysis/drifting_gratings.py"
)


def synthetic_presentations() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    target_starts = []
    start = 10.0
    for orientation in (0.0, 90.0):
        for spatial_frequency in (0.02, 0.08):
            for contrast in (0.2, 0.8):
                for _ in range(5):
                    rows.append(
                        {
                            "start_time": start,
                            "stop_time": start + 1.00084,
                            "orientation": str(orientation),
                            "temporal_frequency": "2.0",
                            "spatial_frequency": str(spatial_frequency),
                            "contrast": str(contrast),
                        }
                    )
                    if (orientation, spatial_frequency, contrast) == (90.0, 0.08, 0.8):
                        target_starts.append(start)
                    start += 2.0
    return pd.DataFrame(rows), np.asarray(target_starts)


def test_full_condition_includes_sf_and_other_varying_dimensions():
    presentations, target_starts = synthetic_presentations()
    spikes = []
    for start in presentations["start_time"]:
        spikes.append(start + 0.25)
    for start in target_starts:
        # Two additional, phase-locked spikes make this ori x TF x SF x
        # contrast condition uniquely preferred.
        spikes.extend((start + 0.125, start + 0.625))

    metrics = compute_drifting_grating_metrics(
        [7], {7: np.sort(spikes)}, presentations
    ).iloc[0]
    assert metrics["pref_ori_dg"] == 90.0
    assert metrics["pref_tf_dg"] == 2.0
    assert metrics["pref_sf_dg"] == 0.08
    assert metrics["pref_contrast_dg"] == 0.8
    assert metrics["preferred_trials_dg"] == 5
    assert metrics["preferred_condition_ties_dg"] == 1
    assert metrics["analysis_duration_s_dg"] == 1.0
    assert metrics["firing_rate_dg"] == pytest.approx(
        metrics["preferred_mean_spikes_dg"] / metrics["analysis_duration_s_dg"]
    )
    assert np.isfinite(metrics["f1_f0_dg"])
    assert np.isfinite(metrics["mod_idx_dg"])


def test_one_ms_bins_are_counts_not_binary():
    counts = _bin_trial_spike_counts(
        np.array([1.0001, 1.0002, 1.002]), np.array([1.0]), duration_ms=5
    )
    assert counts.shape == (1, 5)
    assert counts[0, 0] == 2
    assert counts.sum() == 3


def test_missing_spatial_frequency_fails_explicitly():
    presentations, _ = synthetic_presentations()
    with pytest.raises(ValueError, match="spatial_frequency"):
        compute_drifting_grating_metrics(
            [1], {1: np.array([10.1])}, presentations.drop(columns="spatial_frequency")
        )


@pytest.mark.skipif(not ALLENSDK_SOURCE.is_file(), reason="AllenSDK source unavailable")
def test_low_level_math_matches_installed_allensdk():
    """Compare both formulas directly with installed AllenSDK 2.16.2 source.

    The environment's full AllenSDK import is currently broken by an unrelated
    marshmallow/dataclasses dependency conflict, so the test executes only the
    two unmodified function definitions from that installed source file.
    """
    tree = ast.parse(ALLENSDK_SOURCE.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"f1_f0", "modulation_index"}
    ]
    assert {node.name for node in definitions} == {"f1_f0", "modulation_index"}
    namespace = {"np": np, "signal": signal, "fft": fft, "warnings": warnings}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(ALLENSDK_SOURCE), "exec"),
        namespace,
    )
    rng = np.random.default_rng(8128)
    arr = rng.poisson(0.015, size=(7, 1000)).astype(float)
    psth = arr.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        allen_f1_f0 = namespace["f1_f0"](arr, 4.0, 1.0)
        local_f1_f0 = f1_f0_from_trial_counts(arr, 4.0, 1.0)
        allen_mod_idx = namespace["modulation_index"](psth, 4.0, 1000.0)
        local_mod_idx = welch_modulation_index(psth, 4.0, 1000.0)
    assert local_f1_f0 == pytest.approx(allen_f1_f0, abs=1e-12)
    assert local_mod_idx == pytest.approx(allen_mod_idx, abs=1e-12)
