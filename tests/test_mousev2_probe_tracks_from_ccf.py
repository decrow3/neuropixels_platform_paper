from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_mousev2_probe_tracks_from_ccf import fit_track  # noqa: E402


def test_fit_track_recovers_and_orients_straight_trajectory():
    rel_y = np.linspace(0.0, 3800.0, 192)
    expected = np.array([0.2, -0.95, 0.24])
    expected /= np.linalg.norm(expected)
    points = np.array([9000.0, 3800.0, 3200.0]) + rel_y[:, None] * expected

    fit = fit_track(points, rel_y)

    np.testing.assert_allclose(fit["direction"], expected, atol=1e-12)
    assert fit["r2_colinearity"] == 1.0
    assert np.corrcoef(fit["projection"], rel_y)[0, 1] > 0.999999
    assert np.isclose(
        fit["angle_from_ccf_y_deg"],
        np.degrees(np.arccos(abs(expected[1]))),
    )


def test_fit_track_is_stable_to_small_perpendicular_noise():
    rng = np.random.default_rng(7)
    rel_y = np.linspace(0.0, 3000.0, 150)
    expected = np.array([-0.15, -0.97, 0.19])
    expected /= np.linalg.norm(expected)
    points = np.array([8500.0, 3600.0, 3300.0]) + rel_y[:, None] * expected
    points += rng.normal(scale=8.0, size=points.shape)

    fit = fit_track(points, rel_y)

    assert np.dot(fit["direction"], expected) > 0.999
    assert fit["r2_colinearity"] > 0.999
