from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_figure4_location_controls import matched_control, shuffled_effects
from scripts.figure3_robust_spread_comparison import omega_squared


def test_paired_control_excludes_unmatched_sessions_and_weights_areas_equally():
    groups = pd.DataFrame({'metric': ['x']*4, 'dataset': ['Post-V1']*4,
                           'session_id': ['1', '1', '2', '3'], 'mean': [10., 30., 40., 99.]})
    v1 = pd.DataFrame({'metric': ['x']*3, 'session_id': [1, 2, 4], 'mean': [1., 2., 4.]})
    control = matched_control(groups, v1, 'x').pivot(index='session_id', columns='group', values='mean')
    assert list(control.index) == ['1', '2']
    np.testing.assert_allclose(control['hva'], [20., 40.])
    np.testing.assert_allclose(control['v1'], [1., 2.])


def test_shuffle_breaks_stable_location_effect_but_preserves_single_location_sessions():
    table = pd.DataFrame({'session_id': np.repeat(np.arange(20), 2),
                          'group': ['a', 'b']*20, 'mean': [0., 10.]*20})
    null = shuffled_effects(table, 200, np.random.default_rng(42))
    assert omega_squared(table.group.to_numpy(), table['mean'].to_numpy()) == 1.
    assert np.quantile(null, .975) < .3
    table['session_id'] = np.arange(40)
    np.testing.assert_allclose(shuffled_effects(table, 10, np.random.default_rng(42)), 1.)
