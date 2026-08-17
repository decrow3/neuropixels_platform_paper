"""Integration checks for the provisional PilotAnalysis RF snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from common.figure3_mousev2 import load_config, load_mousev2_units
from common.figure3_rf import DEFAULT_RF_IMPORT_DIR, load_rf_import


def import_directory() -> Path:
    return Path(os.environ.get("FIGURE3_RF_IMPORT_DIR", DEFAULT_RF_IMPORT_DIR))


@pytest.fixture(scope="module")
def imported():
    directory = import_directory()
    if not directory.is_dir():
        pytest.skip(f"RF import snapshot not present: {directory}")
    return load_rf_import(directory)


def test_rf_unit_mapping_and_qc(imported):
    peaks, _, _, manifest = imported
    current = load_mousev2_units(apply_qc=False)
    assert len(peaks) == len(current) == 20_374
    assert peaks["unit_id"].is_unique
    assert set(peaks["unit_id"]) == set(current["unit_id"])
    assert int(peaks["pilot_qc"].sum()) == 4_807
    assert int(peaks["default_qc"].sum()) == 11_242
    assert peaks.loc[peaks["pilot_qc"], "default_qc"].all()
    assert manifest["validation"]["mapped_units"] == 20_374


def test_rf_grid_and_probe_coverage(imported):
    peaks, summary, ordering, _ = imported
    config = load_config()
    expected_grid = set(float(value) for value in range(-40, 41, 10))
    assert set(peaks["rf_center_x_deg"].dropna().unique()) == expected_grid
    assert set(peaks["rf_center_y_deg"].dropna().unique()) == expected_grid
    assert len(summary) == 32
    assert summary.groupby("site_number")["probe"].nunique().eq(4).all()
    assert set(summary["probe"]) == set(config["probe_labels"])
    assert len(ordering) == 8


def test_declared_probe_order_is_not_universal(imported):
    _, _, ordering, manifest = imported
    assert int(ordering["declared_order_strictly_descending_x"].sum()) == 3
    assert int(ordering["declared_order_descending_x_allowing_ties"].sum()) == 5
    assert manifest["validation"]["strict_declared_order_sessions"] == 3
    assert manifest["validation"]["declared_order_sessions_allowing_ties"] == 5
