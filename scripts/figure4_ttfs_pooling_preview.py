"""Temporary TTFS distribution comparison; leaves the integrated figure untouched."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scripts import build_figure4_rerun_outputs as r


def main():
    out = ROOT / 'artifacts/figure4_rerun/temporary_ttfs_pooling'
    out.mkdir(exist_ok=True)
    registry = pd.read_csv(ROOT / 'artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz',
                           dtype={'unit_id': str, 'session_id': str})
    metric = r.METRICS[0]
    table = r.response_cells(registry, 'HVA', metric)
    point = pd.read_csv(r.PRIMARY / 'draw_specific_point_fits.csv')
    means = r.averaged_means(point.loc[point.metric.eq(metric) & point.population.eq('HVA'),
                                    'category_means'], r.HVA_CATEGORIES)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True,
                                     gridspec_kw={'width_ratios': [3, 1.6]})
    r.violin(left, table, r.HVA_CATEGORIES, r.HVA_COLORS, means)
    for annotation in list(left.texts):
        annotation.remove()
    left.set_title('Across HVAs', fontsize=17, pad=16)
    left.set_ylabel('TTFS (ms)', fontsize=16)
    values = table.value.to_numpy(float)
    assert np.isfinite(values).all() and not table.unit_id.duplicated().any()
    grid = np.linspace(values.min(), values.max(), 400)
    pooled = gaussian_kde(values)(grid)
    densities = {}
    # Use the pooled absolute bandwidth for all outlines for a fair shape comparison.
    bandwidth = np.sqrt(gaussian_kde(values).covariance[0, 0])
    for cat in r.HVA_CATEGORIES:
        v = table.loc[table.category.eq(cat), 'value'].to_numpy(float)
        density = gaussian_kde(v, bw_method=bandwidth / np.std(v, ddof=1))(grid)
        density[(grid < v.min()) | (grid > v.max())] = np.nan
        densities[cat] = density
    scale = .42 / max(pooled.max(), *(np.nanmax(d) for d in densities.values()))
    right.fill_betweenx(grid, 0, scale * pooled,
                        facecolor='#d6d8db', edgecolor='#555555', linewidth=1.5)
    for cat, density in densities.items():
        right.plot(scale * density, grid, color=r.HVA_COLORS[cat],
                   lw=1.0, alpha=.50)
    # One pooled subsample preserves category prevalence among the displayed dots.
    rng = np.random.default_rng(20260918)
    chosen = rng.choice(len(table), min(800, len(table)), replace=False)
    dots = table.iloc[chosen]
    right.scatter(rng.uniform(-.24, -.03, len(dots)), dots.value,
                  c=[r.HVA_COLORS[c] for c in dots.category], s=4, alpha=.16,
                  linewidth=0, rasterized=True)
    for offset, cat in zip(np.linspace(-.055, .055, len(r.HVA_CATEGORIES)), r.HVA_CATEGORIES):
        right.plot(offset, means[cat], marker='D', ms=5.2,
                   color=r.HVA_COLORS[cat], mec='white', mew=.55, zorder=5)
    mean_center = float(np.mean(list(means.values())))
    mean_sd = float(np.std(list(means.values()), ddof=0))
    cell_center = float(np.mean(values))
    cell_sd = float(np.std(values, ddof=0))
    for x, center, sd, direction, label in [
        (-.36, cell_center, cell_sd, 1, 'All cells'),
        (.51, mean_center, mean_sd, -1, 'Area means'),
    ]:
        tip = x + direction * .035
        right.plot([tip, x, x, tip], [center - sd, center - sd, center + sd, center + sd],
                   color='#252525', lw=1.8, zorder=6)
        right.plot([x, tip], [center, center], color='#252525', lw=1.8, zorder=6)
        right.text(x, center + sd + 2, f'{label}\n±1 SD\n({sd:.1f} ms)',
                   ha='center', va='bottom', fontsize=10.5, color='#252525')
    right.set_xlim(-.58, .76)
    right.set_xticks([0], ['All HVA cells'])
    right.set_title('Combined distribution', fontsize=17, pad=16)
    r.clean(right)
    v1 = r.response_cells(registry, 'V1', metric)
    limits = [t.value.quantile([.002, .998]).to_numpy() for t in [table, v1]]
    lo, hi = min(v[0] for v in limits), max(v[1] for v in limits)
    pad = .08 * (hi - lo)
    left.set_ylim(lo - pad, hi + pad)
    for ax in [left, right]:
        ax.tick_params(labelsize=14)
        ax.set_yticks([40, 60, 80, 100])
    left.set_xticks(range(5), r.HVA_CATEGORIES, rotation=0, ha='center')
    fig.subplots_adjust(left=.10, right=.97, top=.86, bottom=.16, wspace=.25)
    stem = out / 'TTFS_area_and_combined_violin'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=180, facecolor='white')
    table.groupby('category').agg(cells=('value', 'size'), raw_mean_ms=('value', 'mean'),
                                raw_sd_ms=('value', 'std')).to_csv(out / 'counts_and_raw_summaries.csv')
    (out / 'README.md').write_text(
        '# Temporary TTFS pooling preview\n\n'
        'Left: same eligible HVA cells, half-violins, deterministic dot subsample, and fitted '
        'category-mean diamonds as Figure 4 E. Right: gray half-violin pools all eligible '
        'HVA cells with equal per-cell weight (larger categories contribute more). Colored '
        'cell dots on the left are a deterministic pooled subsample of up to 800 cells. '
        'Diamonds repeat the fitted category means from the left panel, with small '
        'horizontal offsets solely to distinguish overlapping markers. '
        'Colored '
        'outlines are separately normalized area densities, using the pooled absolute KDE '
        'bandwidth and a common density-to-width scale; they are not additive contributions. '
        'No recentering or session adjustment is applied to the distributions. '
        'The right bracket spans the equally weighted fitted area-mean center ±1 finite-category '
        'SD (ddof=0). The left bracket spans the raw pooled cell mean ±1 population SD '
        '(ddof=0), weighting cells equally. These are descriptive spreads, not confidence '
        'intervals. Pooled cell spread includes both within-area and between-area variation; '
        'it is not a session-adjusted within-area residual SD. '
        'Both panels retain the current TTFS response limits. '
        'Original figure files are unchanged.\n\n'
        'For TTFS across the five HVAs, a descriptive one-way ANOVA of all 1,858 eligible cells gave F(4, 1853) = 9.17, p = 2.48 × 10⁻⁷, and η² = 0.0194 (between-area sum of squares / total sum of squares). Area membership thus accounted for 1.94% of the raw cell-level TTFS variance. This analysis weights cells equally and treats them as independent; it does not account for animal/recording clustering or session effects and is distinct from the adjusted category means and whole-animal bootstrap comparisons.\n')
    omega_methods = ROOT / 'artifacts/figure4_rerun/cell_omega/caption.txt'
    if omega_methods.exists():
        with (out / 'README.md').open('a') as handle:
            handle.write('\n' + omega_methods.read_text() + '\n')
    pd.DataFrame([
        {'quantity': 'fitted_area_means', 'center_ms': mean_center, 'sd_ms': mean_sd},
        {'quantity': 'raw_pooled_cells', 'center_ms': cell_center, 'sd_ms': cell_sd},
    ]).to_csv(out / 'bracket_summaries.csv', index=False)
    print(f'Saved {stem}; {len(table)} unique cells; counts: {table.category.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
