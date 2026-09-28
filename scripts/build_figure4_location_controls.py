"""Review candidate with session-block label shuffles and a matched Allen control.

Run: MPLBACKEND=Agg python scripts/build_figure4_location_controls.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_figure4_variant_review as figure
from scripts.figure3_robust_spread_comparison import omega_squared, clustered_omega_squared
from scripts.figure4_location_controls_legend import generate_legend


def shuffled_effects(table, n_draws, rng):
    """Permute available location labels independently within each session."""
    blocks = [(p['group'].to_numpy(str), p['mean'].to_numpy(float))
              for _, p in table.groupby('session_id', sort=True)]
    values = np.concatenate([v for _, v in blocks])
    return np.array([omega_squared(np.concatenate([rng.permutation(g) for g, _ in blocks]), values)
                     for _ in range(n_draws)])


def matched_control(groups, allen_v1, metric):
    """One V1 and one equal-area HVA mean per shared Allen session."""
    v1 = allen_v1.loc[allen_v1.metric.eq(metric), ['session_id', 'mean']].copy()
    v1['session_id'] = v1.session_id.astype(str)
    hva = groups.loc[groups.metric.eq(metric) & groups.dataset.eq('Post-V1')]
    hva = hva.groupby('session_id', as_index=False)['mean'].mean()
    pairs = v1.merge(hva, on='session_id', suffixes=('_v1', '_hva'), validate='one_to_one')
    return pd.concat([pairs[['session_id', f'mean_{g}']].rename(columns={f'mean_{g}': 'mean'}).assign(group=g)
                      for g in ['v1', 'hva']], ignore_index=True)


def main():
    n_draws, seed = 5000, 42
    rng = np.random.default_rng(seed + 1)
    groups, allen_v1, hierarchy = figure.load_inputs()
    groups, raw_cells, _ = figure.apply_anatomical_v1_filter(groups, figure.load_figure_cells())
    identity = figure.identity_statistics(groups, n_bootstrap=n_draws, seed=seed)
    records, inputs = [], []
    for metric in figure.METRICS:
        local = groups.loc[groups.metric.eq(metric)]
        tables = {'v1': local.loc[local.dataset.eq('Within-V1')],
                  'hva': local.loc[local.dataset.eq('Post-V1')],
                  'control': matched_control(groups, allen_v1, metric)}
        nulls = {}
        for key, table in tables.items():
            inputs.append(table[['session_id', 'group', 'mean']].assign(metric=metric, comparison=key))
            if key == 'control':
                observed, draws = clustered_omega_squared(table, n_bootstrap=n_draws, rng=rng)
                identity[metric].update(control=observed, control_low=np.quantile(draws, .025), control_high=np.quantile(draws, .975))
            null = shuffled_effects(table, n_draws, rng)
            nulls[key] = null
            identity[metric][f'{key}_p'] = (1 + np.sum(null >= identity[metric][key])) / (n_draws + 1)
            identity[metric].update({f'{key}_null': np.median(null), f'{key}_null_low': np.quantile(null, .025), f'{key}_null_high': np.quantile(null, .975)})
            records.append(dict(metric=metric, comparison=key, observed=identity[metric][key],
                                bootstrap_low=identity[metric][key+'_low'], bootstrap_high=identity[metric][key+'_high'],
                                shuffle_median=np.median(null), shuffle_low=np.quantile(null, .025), shuffle_high=np.quantile(null, .975),
                                n_sessions=table.session_id.nunique(), n_rows=len(table),
                                permutation_p_greater=identity[metric][f'{key}_p']))
        delta_null = nulls['hva'] - nulls['v1']
        identity[metric].update(delta_null=np.median(delta_null), delta_null_low=np.quantile(delta_null, .025), delta_null_high=np.quantile(delta_null, .975))
    out = ROOT / 'Figure3'
    pd.DataFrame(records).to_csv(out / 'Figure4_location_controls_statistics.csv', index=False)
    pd.concat(inputs, ignore_index=True).to_csv(out / 'Figure4_location_controls_inputs.csv', index=False)
    entries, atlas, labels = figure.load_mousev2_ccf_surface()
    displayed_entries = entries.copy()
    if displayed_entries.duplicated(['subject_id', 'probe']).any():
        raise ValueError('Mean locations require one entry per animal and probe')
    position_means = displayed_entries.groupby('probe').agg(
        n_animals=('subject_id', 'nunique'), ccf_ml_um=('ccf_ml_um', 'mean'),
        ccf_ap_um=('ccf_ap_um', 'mean'))
    if not position_means.n_animals.eq(8).all():
        raise ValueError('Expected eight animals for each mean probe position')
    position_means.to_csv(out / 'Figure4_mean_probe_positions.csv')
    schematic = figure.recolor_open_scope_squares(plt.imread(figure.DEFAULT_OPEN_SCOPE_SCHEMATIC))
    fig = figure.variant_j(groups, figure.weight_cells(groups, raw_cells), hierarchy, identity, entries, atlas, labels, schematic)
    # Requested anatomical framing, with anterior at the top.
    fig.axes[0].set_xlim(1.25, 4.5)
    fig.axes[0].set_ylim(10.5, 6.75)
    anatomy = fig.axes[0]
    box = anatomy.get_position(original=True)
    anatomy.set_position([box.x0, box.y0 + .04, box.width, box.height - .06])
    anatomy.set_title('A  V1 probe locations', loc='left', fontweight='bold', fontsize=11, pad=22)
    for label in anatomy.texts:
        if label.get_text().startswith('Allen dorsal surface'):
            label.set_text(f'{displayed_entries.subject_id.nunique()} animals · large: mean; small: individual')
            label.set_position((0, 1.02))
            label.set_verticalalignment('bottom')
    legend = anatomy.get_legend()
    handles = legend.legend_handles
    labels = [label.get_text() for label in legend.get_texts()]
    legend.remove()
    anatomy.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.14),
                   frameon=False, fontsize=7, columnspacing=.7, handletextpad=.2)
    fig.axes[1].set_title('B  OpenScope imaging schematic', loc='left', fontweight='bold', fontsize=10.5, pad=8)
    fig.text(.975, .008, '* p < .05   ** p < .01   *** p < .001; n.s.: p ≥ .05 (one-sided shuffle tests, uncorrected)\n'
             '‡ Effect difference: n.s. indicates that the 95% bootstrap interval includes zero.',
             ha='right', fontsize=7, color=figure.MUTED, linespacing=1.4)
    # Keep SVG labels as text and embed TrueType fonts in the vector PDF.
    with plt.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        for suffix in ['pdf', 'svg', 'png']:
            fig.savefig(out / f'Figure4_location_controls_candidate.{suffix}', dpi=200, facecolor='white')
    legend_text = generate_legend(
        entries=entries, displayed_entries=displayed_entries, records=records,
        identity=identity, n_draws=n_draws, seed=seed,
        timescale_limits=fig.axes[-2].get_xlim(),
    )
    (out / 'Figure4_location_controls_candidate_legend.md').write_text(legend_text, encoding='utf-8')
    plt.close(fig)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
