"""Tests for named cross-dataset unit-population profiles."""

from __future__ import annotations

import pytest

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from common.population_masks import PopulationUnavailableError


def test_mouse_common_qc_exactly_matches_declared_default_qc():
    legacy = load_mousev2_units(apply_qc=True)
    named = load_mousev2_units(apply_qc=False, population_profile="common_qc")
    assert len(legacy) == len(named) == 11_242
    assert set(legacy["unit_id"]) == set(named["unit_id"])
    assert named["population_profile"].eq("common_qc").all()


def test_allen_common_qc_uses_homologous_thresholds():
    all_units = load_allen_units()
    common = load_allen_units(population_profile="common_qc")
    assert len(all_units) == 99_180
    assert len(common) == 43_496
    assert common["population_profile"].eq("common_qc").all()


def test_mouse_published_like_fails_instead_of_weakening_rule():
    with pytest.raises(PopulationUnavailableError, match="missing columns") as exc:
        load_mousev2_units(
            apply_qc=False,
            grating_metrics_dir="data/imports/mousev2_grating_metrics_v1",
            population_profile="published_like",
        )
    message = str(exc.value)
    assert "area_rf" in message
    assert "p_value_rf" in message
    assert "firing_rate_dg" not in message


def test_legacy_and_named_profiles_cannot_be_combined_implicitly():
    with pytest.raises(ValueError, match="either legacy apply_qc"):
        load_mousev2_units(apply_qc=True, population_profile="common_qc")
