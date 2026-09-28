from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.figure4_rerun_core import category_recording_weights, fit_standardized_dispersion
from scripts.figure4_rf_support import classify_local_support, reference_weights
from scripts.validate_figure4_rerun_models import representative_design, resample_animals
from scripts.run_figure4_matched_control import control_weights
from scripts.run_figure4_central_companion import CATEGORIES, identifiability
from scripts.build_figure4_rerun_outputs import display_cells


def test_declared_weights_balance_categories_animals_recordings_and_cells():
    rows = []
    layout = {"a": {"m1": {"r1": 2, "r2": 4}, "m2": {"r3": 3}}, "b": {"m3": {"r4": 7}}}
    for category, animals in layout.items():
        for animal, recordings in animals.items():
            for recording, count in recordings.items():
                rows += [dict(category=category, animal_id=animal, recording_id=recording, unit_id=f"{recording}:{i}") for i in range(count)]
    table = pd.DataFrame(rows)
    table["weight"] = category_recording_weights(table)
    np.testing.assert_allclose(table.groupby("category").weight.sum(), [.5, .5])
    np.testing.assert_allclose(table.loc[table.category.eq("a")].groupby("animal_id").weight.sum(), [.25, .25])
    np.testing.assert_allclose(table.loc[table.animal_id.eq("m1")].groupby("recording_id").weight.sum(), [.125, .125])


def test_dispersion_uses_k_denominator_and_preserves_response_units():
    rows = []
    for session in ["s1", "s2", "s3"]:
        for category, mean in [("a", 0.0), ("b", 2.0), ("c", 4.0)]:
            for cell in range(5):
                rows.append(dict(
                    category=category, animal_id=session, recording_id=f"{session}:{category}",
                    session_id=session, unit_id=f"{session}:{category}:{cell}", value=mean,
                ))
    fit = fit_standardized_dispersion(pd.DataFrame(rows))
    np.testing.assert_allclose(list(fit.category_means.values()), [0, 2, 4], atol=1e-12)
    np.testing.assert_allclose(fit.dispersion, 8 / 3)
    np.testing.assert_allclose(fit.root_dispersion, np.sqrt(8 / 3))


def test_adjustment_recovers_category_shift_under_rf_imbalance():
    rng = np.random.default_rng(8)
    rows = []
    for session in ["s1", "s2", "s3", "s4"]:
        for category, azimuth, shift in [("a", 25.0, 0.0), ("b", 70.0, 3.0)]:
            for cell in range(40):
                az = azimuth + rng.normal(0, 4)
                el = 10 + rng.normal(0, 5)
                rows.append(dict(
                    category=category, animal_id=session, recording_id=f"{session}:{category}",
                    session_id=session, unit_id=f"{session}:{category}:{cell}",
                    azimuth=az, elevation=el, value=shift + .4 * az + rng.normal(0, .2),
                ))
    table = pd.DataFrame(rows)
    adjusted = fit_standardized_dispersion(table, adjusted=True, reference_rf=table)
    difference = adjusted.category_means["b"] - adjusted.category_means["a"]
    np.testing.assert_allclose(difference, 3, atol=.25)


def test_local_support_requires_hull_cells_and_independent_recordings():
    rows = []
    for category, shift in [("a", 0), ("b", 1)]:
        for recording in range(2):
            for cell, (x, y) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]):
                rows.append(dict(
                    support_category=category, azimuth=x + shift, elevation=y,
                    recording_id=f"{category}:{recording}", unit_id=f"{category}:{recording}:{cell}",
                ))
    table = pd.DataFrame(rows)
    supported, audit = classify_local_support(table, ["a", "b"], radius_deg=5, min_cells=5, min_recordings=2)
    assert supported.any()
    assert audit.loc[audit.category_supported, "local_recordings"].ge(2).all()
    far = table.copy()
    far.loc[len(far)] = dict(support_category="a", azimuth=100, elevation=100, recording_id="a:3", unit_id="far")
    supported, _ = classify_local_support(far, ["a", "b"], radius_deg=5, min_cells=5, min_recordings=2)
    assert not supported.iloc[-1]


def test_reference_weights_give_each_population_half():
    rows = []
    for population, categories in [("V1", ["a", "b"]), ("HVA", ["x", "y", "z"])]:
        for category in categories:
            for cell in range(1 + (category == categories[-1])):
                rows.append(dict(population=population, category=category, animal_id=f"{population}:animal", recording_id=f"{population}:{category}", unit_id=f"{population}:{category}:{cell}"))
    table = pd.DataFrame(rows)
    weights = reference_weights(table)
    np.testing.assert_allclose(weights.groupby(table.population).sum(), [.5, .5])
    expected = {"HVA": 1/6, "V1": 1/4}
    for (population, _), value in weights.groupby([table.population, table.category]).sum().items():
        np.testing.assert_allclose(value, expected[population])


def test_animal_bootstrap_carries_nested_recordings_and_relabels_copies():
    table = pd.DataFrame([
        dict(animal_id="a", session_id="s1", recording_id="r1", unit_id="u1", category="x", value=1.),
        dict(animal_id="a", session_id="s2", recording_id="r2", unit_id="u2", category="y", value=2.),
        dict(animal_id="b", session_id="s3", recording_id="r3", unit_id="u3", category="x", value=3.),
    ])
    class DuplicateRng:
        def choice(self, values, size, replace):
            return np.asarray(["a"] * size)
    result = resample_animals(table, DuplicateRng())
    assert len(result) == 4
    assert result.animal_id.nunique() == 2
    assert result.recording_id.nunique() == 4
    assert not result.unit_id.duplicated().any()


def test_timescale_representative_draw_is_source_specific():
    rows = []
    for population, draw in [("V1", 0), ("HVA", -1)]:
        for cell in range(5):
            rows.append(dict(metric="Response timescale (ms)", population=population, draw_id=draw,
                             category="x", recording_id=f"{population}:r", unit_id=f"{population}:{cell}"))
    table = pd.DataFrame(rows)
    assert len(representative_design(table, "Response timescale (ms)", "V1")) == 5
    assert len(representative_design(table, "Response timescale (ms)", "HVA")) == 5


def test_figure_timescale_display_uses_one_coherent_draw_and_allen_estimate():
    rows = []
    for draw, count in [(0, 5), (1, 7)]:
        for cell in range(count):
            rows.append(dict(
                source="MouseV2", population="V1", metric="Response timescale (ms)",
                draw_id=draw, category="A", recording_id="mouse:r",
                unit_id=f"mouse:{draw}:{cell}", value=float(cell),
            ))
    for cell in range(5):
        rows.append(dict(
            source="Allen", population="Central", metric="Response timescale (ms)",
            draw_id=-1, category="Central", recording_id="allen:r",
            unit_id=f"allen:{cell}", value=float(cell),
        ))
    displayed = display_cells(pd.DataFrame(rows))
    assert set(displayed.draw_id) == {-1, 0}
    assert len(displayed.loc[displayed.source.eq("MouseV2")]) == 5
    assert len(displayed.loc[displayed.source.eq("Allen")]) == 5


def test_matched_control_balances_groups_and_hva_areas():
    rows = []
    for group, categories in [("V1", ["V1"]), ("HVA", ["LM", "AL"])]:
        for category in categories:
            for animal in ["a", "b"]:
                for cell in range(2 + (category == "AL")):
                    rows.append(dict(control_group=group, category=category, animal_id=animal,
                                     recording_id=f"{animal}:{category}", unit_id=f"{animal}:{category}:{cell}"))
    table = pd.DataFrame(rows)
    table["weight"] = control_weights(table)
    np.testing.assert_allclose(table.groupby("control_group").weight.sum(), [.5, .5])
    np.testing.assert_allclose(table.loc[table.control_group.eq("HVA")].groupby("category").weight.sum(), [.25, .25])


def test_central_companion_records_exact_source_session_confounding():
    rows = []
    for category in [value for value in CATEGORIES if value != "Central"]:
        for session in ["m1", "m2"]:
            rows.append(dict(category=category, session_id=session))
    for session in ["a1", "a2"]:
        rows.append(dict(category="Central", session_id=session))
    rank, parameters = identifiability(pd.DataFrame(rows))
    assert rank == parameters - 1
