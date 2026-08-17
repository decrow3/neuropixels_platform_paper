"""Focused tests for MouseV2 RF/preference surface inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.mousev2_frequency_preference_surfaces import (
    load_mousev2_units,
    to_allen_display_coordinates,
)
from scripts.render_mousev2_simultaneous_probe_maps import (
    align_simultaneous_rf_translation,
    complete_simultaneous_sessions,
)


def _write_inputs(tmp_path: Path, duplicate_grating: bool = False) -> tuple[Path, Path]:
    rf = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "site": ["site2", "site2"],
            "probe": ["A", "B"],
            "rf_center_x_deg": [0.0, 10.0],
            "rf_center_y_deg": [20.0, 30.0],
            "pilot_qc": [True, True],
            "default_qc": [True, True],
        }
    )
    gratings = pd.DataFrame(
        {
            "unit_id": [1, 1] if duplicate_grating else [1, 2],
            "pref_sf_dg": [0.04, 0.08],
            "pref_tf_dg": [2.0, 4.0],
            "preferred_condition_ties_dg": [1, 2],
        }
    )
    rf_path = tmp_path / "rf.csv"
    grating_dir = tmp_path / "gratings" / "site2"
    grating_dir.mkdir(parents=True)
    rf.to_csv(rf_path, index=False)
    gratings.to_csv(grating_dir / "grating_metrics.csv", index=False)
    return rf_path, grating_dir.parent


def test_mousev2_join_and_unique_preference_population(tmp_path):
    rf_path, grating_dir = _write_inputs(tmp_path)
    units, paths = load_mousev2_units(
        rf_path,
        grating_dir,
        qc_profile="pilot_qc",
        require_unique_preference=True,
    )
    assert len(paths) == 1
    assert units["analysis_eligible"].tolist() == [True, False]
    assert units.loc[0, "stimulus_x_deg"] == 0.0
    assert units.loc[0, "stimulus_y_deg"] == 20.0
    assert units.loc[0, "azimuth_rf"] == 50.0
    assert units.loc[0, "elevation_rf"] == 30.0


def test_mousev2_grid_maps_to_allen_released_rf_axes():
    azimuth, elevation = to_allen_display_coordinates(
        pd.Series([-40.0, 0.0, 40.0]),
        pd.Series([-40.0, 0.0, 40.0]),
    )
    assert azimuth.tolist() == [10.0, 50.0, 90.0]
    assert elevation.tolist() == [-30.0, 10.0, 50.0]


def test_mousev2_join_rejects_duplicate_unit_ids(tmp_path):
    rf_path, grating_dir = _write_inputs(tmp_path, duplicate_grating=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_mousev2_units(
            rf_path,
            grating_dir,
            qc_profile="pilot_qc",
            require_unique_preference=True,
        )


def test_mousev2_mapping_uses_only_trial_supported_preferences(tmp_path):
    rf_path, grating_dir = _write_inputs(tmp_path)
    tuning = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "sf_preference_supported": [True, False],
            "tf_preference_supported": [False, True],
            "pref_sf_supported_dg": [0.08, float("nan")],
            "pref_tf_supported_dg": [float("nan"), 8.0],
        }
    )
    tuning_path = tmp_path / "frequency_tuning_support.csv"
    tuning.to_csv(tuning_path, index=False)
    units, _ = load_mousev2_units(
        rf_path,
        grating_dir,
        qc_profile="pilot_qc",
        require_unique_preference=True,
        tuning_support_path=tuning_path,
    )
    assert units["analysis_eligible"].tolist() == [True, True]
    assert units["tuning_eligible_sf"].tolist() == [True, False]
    assert units["tuning_eligible_tf"].tolist() == [False, True]


def test_simultaneous_gate_requires_all_probes_for_every_population():
    rows = []
    for site in ("site1", "site2"):
        for probe in ("A", "B", "C", "E"):
            rows.append(
                {
                    "site": site,
                    "probe": probe,
                    "analysis_eligible": True,
                    "tuning_eligible_sf": True,
                    "tuning_eligible_tf": not (site == "site2" and probe == "E"),
                }
            )
    assert complete_simultaneous_sessions(pd.DataFrame(rows)) == ["site1"]


def test_translation_alignment_preserves_within_session_probe_offsets():
    rows = []
    for site, translation in (("site1", 0.0), ("site2", 10.0)):
        for index, probe in enumerate(("A", "B", "C", "E")):
            rows.append(
                {
                    "site": site,
                    "probe": probe,
                    "analysis_eligible": True,
                    "stimulus_x_deg": translation + index,
                    "stimulus_y_deg": translation - index,
                }
            )
    aligned, audit = align_simultaneous_rf_translation(
        pd.DataFrame(rows), ["site1", "site2"]
    )
    for site in ("site1", "site2"):
        selected = aligned.loc[aligned["site"].eq(site)].set_index("probe")
        assert selected.loc["E", "aligned_stimulus_x_deg"] - selected.loc[
            "A", "aligned_stimulus_x_deg"
        ] == pytest.approx(3.0)
        assert selected.loc["E", "aligned_stimulus_y_deg"] - selected.loc[
            "A", "aligned_stimulus_y_deg"
        ] == pytest.approx(-3.0)
    assert audit["aligned_reference_x_deg"].nunique() == 1
    assert audit["aligned_reference_y_deg"].nunique() == 1
