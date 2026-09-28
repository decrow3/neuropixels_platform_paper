"""Protect against stale manuscript counts and conclusions after rerunning analysis."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_figure4_variant_review as figure
from scripts.figure4_location_controls_legend import generate_legend


def inputs():
    entries = pd.DataFrame({'subject_id': [1, 2, 3], 'probe': ['A']*3,
                            'surface_agrees_visp': [True, False, True]})
    records = [dict(metric=m, comparison=k, n_sessions=11+i, observed=.1,
                    bootstrap_low=-.1, bootstrap_high=.2, permutation_p_greater=.04)
               for i, m in enumerate(figure.METRICS) for k in ['v1', 'hva', 'control']]
    identity = {m: dict(delta=.05, delta_low=-.1, delta_high=.2) for m in figure.METRICS}
    return dict(entries=entries, displayed_entries=entries.iloc[:2], records=records,
                identity=identity, n_draws=1234, seed=99, timescale_limits=(0, 125))


def test_generated_legend_uses_run_counts_settings_and_results():
    args = inputs()
    text = generate_legend(**args)
    assert 'across 2 animals' in text
    assert 'All 2 positions' in text
    assert 'including 1 outside' in text
    assert '11, 12, 13 session pairs' in text
    assert '1,234 whole-session bootstrap' in text
    assert '0–125 ms' in text
    assert 'seed: 99' in text
    assert 'All three difference intervals include zero' in text
    assert '0.040000' in text

    args['identity'][figure.METRICS[0]]['delta_low'] = .01
    text = generate_legend(**args)
    assert 'All three difference intervals include zero' not in text
    assert 'remaining difference intervals exclude zero' in text
    for values in args['identity'].values():
        values['delta_low'] = .01
    assert 'All three difference intervals exclude zero.' in generate_legend(**args)


def test_legend_row_order_tracks_plot_configuration(monkeypatch):
    monkeypatch.setattr(figure, 'LOCATION_EFFECT_ROWS', figure.LOCATION_EFFECT_ROWS[::-1])
    text = generate_legend(**inputs())
    assert 'From top to bottom, rows show: Across HVAs vs within V1; Across HVAs; Within V1; V1 group vs HVA group.' in text
