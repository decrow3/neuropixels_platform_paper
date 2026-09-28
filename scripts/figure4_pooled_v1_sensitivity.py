"""Compare the matched Allen control with the addition of MouseV2 V1 sessions.

Uses the anatomically filtered, frozen inputs exported by
build_figure4_location_controls.py. No cross-project permutation test is assumed.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figure3_robust_spread_comparison import omega_squared


def pool_control(inputs, metric):
    local = inputs.loc[inputs.metric.eq(metric)]
    allen = local.loc[local.comparison.eq('control'), ['session_id', 'group', 'mean']].copy()
    allen['source'] = 'Allen'
    # Equal weight per eligible probe in a MouseV2 session, then one V1 mean.
    mouse = local.loc[local.comparison.eq('v1')].groupby('session_id', as_index=False)['mean'].mean()
    mouse['group'] = 'v1'
    mouse['source'] = 'MouseV2'
    pooled = pd.concat([allen, mouse], ignore_index=True)
    pooled['session_id'] = pooled.source + ':' + pooled.session_id.astype(str)
    return pooled


def stratified_bootstrap(table, rng, n_draws=5000):
    # Retain source counts; resample Allen pairs jointly and MouseV2 sessions
    # independently. No fabricated HVA measurements for MouseV2 sessions.
    strata = [[(p.group.to_numpy(str), p['mean'].to_numpy(float))
               for _, p in source.groupby('session_id', sort=True)]
              for _, source in table.groupby('source', sort=True)]
    draws = []
    for _ in range(n_draws):
        sampled = [blocks[i] for blocks in strata
                   for i in rng.integers(0, len(blocks), len(blocks))]
        draws.append(omega_squared(np.concatenate([b[0] for b in sampled]),
                                   np.concatenate([b[1] for b in sampled])))
    return np.asarray(draws)


def main():
    out = ROOT / 'Figure3'
    inputs = pd.read_csv(out / 'Figure4_location_controls_inputs.csv', dtype={'session_id': str})
    original = pd.read_csv(out / 'Figure4_location_controls_statistics.csv')
    rng = np.random.default_rng(44)
    records, tables = [], []
    for metric in inputs.metric.unique():
        pooled = pool_control(inputs, metric)
        draws = stratified_bootstrap(pooled, rng)
        estimate = omega_squared(pooled.group.to_numpy(str), pooled['mean'].to_numpy(float))
        old = original.loc[original.metric.eq(metric) & original.comparison.eq('control')].iloc[0]
        records.append(dict(metric=metric, allen_only=old.observed,
                            allen_only_low=old.bootstrap_low, allen_only_high=old.bootstrap_high,
                            pooled=estimate, pooled_low=np.quantile(draws, .025), pooled_high=np.quantile(draws, .975),
                            n_allen_pairs=pooled.loc[pooled.source.eq('Allen'), 'session_id'].nunique(),
                            n_mousev2_sessions=pooled.loc[pooled.source.eq('MouseV2'), 'session_id'].nunique()))
        tables.append(pooled.assign(metric=metric))
    result = pd.DataFrame(records)
    result.to_csv(out / 'Figure4_pooled_v1_sensitivity.csv', index=False)
    pd.concat(tables).to_csv(out / 'Figure4_pooled_v1_sensitivity_inputs.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
