"""Schema, coverage, and frozen-protocol checks for the MouseV2 analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.figure3_mousev2 import (
    DEFAULT_CONFIG_PATH,
    ROOT,
    load_config,
    load_mousev2_units,
    within_v1_x_positions,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def raw_units():
    return load_mousev2_units(apply_qc=False)


@pytest.fixture(scope="module")
def qc_units():
    return load_mousev2_units(apply_qc=True)


def test_session_manifest_is_complete(config):
    assert DEFAULT_CONFIG_PATH.is_file()
    assert [session["site"] for session in config["sessions"]] == [
        "site2",
        "site3",
        "site4",
        "site5",
        "site6",
        "site7",
        "site8",
        "site9",
    ]
    assert config["probe_labels"] == ["A", "B", "C", "E"]
    assert config["display_probe_order"] == ["B", "C", "A", "E"]
    assert len({session["subject_id"] for session in config["sessions"]}) == 8
    assert len({session["id_offset"] for session in config["sessions"]}) == 8
    assert config["nwb_input"]["dandiset_id"] == "DANDI:001568"
    assert config["nwb_input"]["dandiset_version"] == "draft"
    assert all(session["nwb_relative_path"].endswith(".nwb") for session in config["sessions"])
    assert all(session["expected_nwb_bytes"] > 8_000_000_000 for session in config["sessions"])


def test_raw_table_schema_counts_and_coverage(config, raw_units):
    required_columns = {
        "unit_id",
        "cortical_depth",
        "cortical_layer",
        "ecephys_structure_acronym",
        "f1_f0_dg",
        "timescale_ac",
        "err_ac",
        "spike_count_ac",
        "time_to_first_spike_fl",
        "site",
        "session_num",
        "subject_id",
        "probe_letter",
    }
    assert required_columns.issubset(raw_units.columns)
    assert len(raw_units) == 20_374
    assert raw_units["unit_id"].is_unique
    assert set(raw_units["probe_letter"]) == {"A", "B", "C", "E"}

    observed_counts = raw_units.groupby("site").size().to_dict()
    expected_counts = {
        session["site"]: session["expected_units"] for session in config["sessions"]
    }
    assert observed_counts == expected_counts
    assert raw_units.groupby("site")["probe_letter"].nunique().eq(4).all()


def test_default_qc_profile_is_stable(qc_units):
    assert len(qc_units) == 11_242
    assert qc_units["unit_id"].is_unique
    assert qc_units.groupby("site")["probe_letter"].nunique().eq(4).all()


def test_stimulus_manifest_combinatorics():
    path = ROOT / "config" / "mousev2_stimulus_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    blocks = manifest["blocks"]

    rf = blocks["receptive_field_gabors"]
    assert (
        len(rf["position_x_deg"])
        * len(rf["position_y_deg"])
        * len(rf["orientation_deg"])
        * rf["repeats"]
        == rf["expected_presentations"]
        == 4860
    )

    gratings = blocks["drifting_gratings"]
    assert (
        len(gratings["orientation_deg"])
        * len(gratings["temporal_frequency_hz"])
        * len(gratings["spatial_frequency_cpd"])
        * gratings["repeats"]
        == gratings["expected_presentations"]
        == 1500
    )

    flashes = blocks["full_field_flashes"]
    assert len(flashes["contrast"]) * flashes["repeats"] == 300
    assert flashes["expected_presentations"] == 300


def test_versioned_flash_variants_preserve_pooled_ttfs_and_select_named_columns():
    flash_dir = "data/imports/mousev2_flash_metrics_v1"
    pooled = load_mousev2_units(
        apply_qc=False,
        flash_metrics_dir=flash_dir,
        flash_variant="pooled",
    )
    assert len(pooled) == 20_374
    assert pooled["flash_variant"].eq("pooled").all()
    assert np.allclose(
        pooled["time_to_first_spike_fl"],
        pooled["time_to_first_spike_fl_pooled_legacy"],
        equal_nan=True,
        rtol=0.0,
        atol=1e-12,
    )

    bright = load_mousev2_units(
        apply_qc=False,
        flash_metrics_dir=flash_dir,
        flash_variant="bright",
    )
    assert bright["flash_variant"].eq("bright").all()
    assert np.allclose(
        bright["time_to_first_spike_fl"],
        bright["time_to_first_spike_bright"],
        equal_nan=True,
    )


def test_display_only_v1_positions_are_centered_on_visp_and_not_legacy_scores():
    labels = ["B", "C", "A", "E"]
    positions = within_v1_x_positions(
        labels,
        "display_only",
        visp_score=-0.357,
        legacy_bounds=(-0.32, -0.12),
        display_half_span=0.06,
    )
    values = np.array(list(positions.values()))
    assert list(positions) == labels
    assert np.mean(values) == pytest.approx(-0.357)
    assert values.min() == pytest.approx(-0.417)
    assert values.max() == pytest.approx(-0.297)
    assert not np.allclose(values, np.linspace(-0.32, -0.12, 4))

    with pytest.raises(ValueError, match="Unknown within-V1 x mode"):
        within_v1_x_positions(
            labels,
            "hierarchy_score",
            visp_score=-0.357,
            legacy_bounds=(-0.32, -0.12),
            display_half_span=0.06,
        )
