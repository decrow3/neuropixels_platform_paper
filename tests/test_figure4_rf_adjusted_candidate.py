from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_figure4_rf_adjusted_candidate import fit_moments, regression_cube, rf_features
from scripts.figure3_robust_spread_comparison import omega_squared


def synthetic(group_shift=0.):
    rng = np.random.default_rng(18)
    group = np.repeat(["a", "b"], 600)
    az = rng.normal(np.where(group == "a", 35., 65.), 12.)
    el = rng.normal(10., 15., len(group))
    values = .7 * az + .004 * az**2 - .2 * el + rng.normal(0, 2., len(group)) + group_shift*(group == "b")
    return pd.DataFrame(dict(group=group, session_id=np.tile(np.repeat(np.arange(12).astype(str), 50), 2),
                             value=values, azimuth=az, elevation=el))


def test_partial_omega_matches_direct_nested_least_squares():
    table = synthetic(5.)
    cube = regression_cube(table, sorted(table.session_id.unique()), ["a", "b"])
    fit = fit_moments(cube.sum(axis=0))
    r = rf_features(table)
    y = table.value.to_numpy()
    reduced = np.column_stack([np.ones(len(table)), r])
    full = np.column_stack([table.group.eq("a"), table.group.eq("b"), r])
    bf = np.linalg.lstsq(full, y, rcond=None)[0]
    br = np.linalg.lstsq(reduced, y, rcond=None)[0]
    sf = np.sum((y-full@bf)**2)
    sr = np.sum((y-reduced@br)**2)
    mse = sf/(len(y)-7)
    np.testing.assert_allclose(fit["omega"], (sr-sf-mse)/(sr+mse), atol=1e-10)
    np.testing.assert_allclose(fit["beta"], bf, atol=1e-10)


def test_rf_only_group_separation_is_removed_but_extra_group_shift_remains():
    pure_rf = synthetic()
    assert omega_squared(pure_rf.group.to_numpy(), pure_rf.value.to_numpy()) > .5
    cube = regression_cube(pure_rf, sorted(pure_rf.session_id.unique()), ["a", "b"])
    assert abs(fit_moments(cube.sum(axis=0))["omega"]) < .01
    extra = synthetic(10.)
    cube = regression_cube(extra, sorted(extra.session_id.unique()), ["a", "b"])
    assert fit_moments(cube.sum(axis=0))["omega"] > .4


def test_session_bootstrap_moments_refit_like_expanded_rows():
    table = synthetic(4.)
    sessions = sorted(table.session_id.unique())
    cube = regression_cube(table, sessions, ["a", "b"])
    selected = [0, 0, 3, 4, 5, 7, 7, 8, 9, 10, 10, 11]
    batched = fit_moments(np.stack([cube.sum(axis=0), cube[selected].sum(axis=0)]))["omega"]
    expanded = pd.concat([table.loc[table.session_id.eq(sessions[i])] for i in selected])
    full = np.column_stack([expanded.group.eq("a"), expanded.group.eq("b"), rf_features(expanded)])
    reduced = np.column_stack([np.ones(len(expanded)), rf_features(expanded)])
    y = expanded.value.to_numpy()
    sf = np.sum((y - full@np.linalg.lstsq(full, y, rcond=None)[0])**2)
    sr = np.sum((y - reduced@np.linalg.lstsq(reduced, y, rcond=None)[0])**2)
    mse = sf/(len(y)-7)
    np.testing.assert_allclose(batched[1], (sr-sf-mse)/(sr+mse), atol=1e-10)
