import numpy as np
import pandas as pd

from scripts.figure3_hierarchical_identity_model import (
    build_design,
    effective_sample_size,
    log_interaction_density,
    log_scale_density,
    slice_sample,
    split_rhat,
)


def test_design_contains_intercept_group_and_session_columns():
    table = pd.DataFrame({"group": ["A", "B", "A"], "session_id": ["1", "1", "2"]})
    design, group_columns, groups, sessions = build_design(table)
    assert design.shape == (3, 1 + 2 + 2)
    assert np.all(design[:, 0] == 1)
    assert np.all(design[:, group_columns].sum(axis=1) == 1)
    assert groups == ["A", "B"]
    assert sessions == ["1", "2"]


def test_slice_sampler_returns_finite_positive_scale():
    rng = np.random.default_rng(4)
    current = 0.0
    for _ in range(100):
        current = slice_sample(current, lambda x: log_scale_density(x, np.array([0.2, -0.1])), rng)
    assert np.isfinite(current)
    assert np.exp(current) > 0
    assert np.isfinite(log_interaction_density(current, np.array([0.1, -0.2]), np.array([0.05, 0.1])))


def test_chain_diagnostics_for_independent_matching_chains():
    rng = np.random.default_rng(2)
    chains = rng.normal(size=(4, 2000))
    assert 0.99 < split_rhat(chains) < 1.01
    assert effective_sample_size(chains) > 5000
