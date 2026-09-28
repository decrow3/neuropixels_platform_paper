#!/usr/bin/env python3
"""Compose the verified schematics and saved Figure 4 results on one shared grid."""
from pathlib import Path
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scripts import build_hierarchy_schematic_row as schematic
from scripts import build_figure4_rerun_outputs as results

OUT = ROOT / 'artifacts/figure4_rerun/v8_final'
STEM = OUT / 'Figure4_CDE_rerun_with_schematic'
REGISTRY = ROOT / 'artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz'


def heading(fig, x, letter, title):
    fig.text(x - .045, .985, letter, fontsize=20, va='top', gid='panel-letter')
    fig.text(x, .975, title, fontsize=9.5, va='top', gid='panel-heading')


def spread_effects(effects, draws):
    """Transform paired population variances before forming the SD contrast."""
    paired = draws.loc[draws.effect.isin(['v1', 'hva'])].pivot(
        index=['metric', 'replicate'], columns='effect', values='estimate')
    if paired.isna().any().any() or (paired < 0).any().any():
        raise ValueError('Root-spread inference requires complete nonnegative paired variances')
    paired['hva_minus_v1_sd'] = np.sqrt(paired.hva) - np.sqrt(paired.v1)
    rows = []
    for metric, group in paired.groupby(level='metric'):
        point = effects.loc[effects.metric.eq(metric)].set_index('effect')
        estimate = np.sqrt(point.loc['hva', 'estimate']) - np.sqrt(point.loc['v1', 'estimate'])
        low, high = group.hva_minus_v1_sd.quantile([.025, .975])
        rows.append(dict(metric=metric, effect='hva_minus_v1_sd', estimate=estimate,
                         low=low, high=high, n_bootstrap=len(group)))
    return pd.DataFrame(rows), paired.reset_index()


def pooled_distribution(ax, table, means, order, colors, response_ylim, label_bars=False):
    """Pooled cells and area densities with descriptive cellular/mean SD brackets."""
    values = table.value.to_numpy(float)
    grid = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    pooled = kde(grid)
    scale = .34 / pooled.max()
    ax.fill_betweenx(grid, 0, scale * pooled, facecolor='#d6d8db',
                     edgecolor='#555555', linewidth=1.1)
    rng = np.random.default_rng(20260918)
    dots = table.iloc[rng.choice(len(table), min(800, len(table)), replace=False)]
    ax.scatter(rng.uniform(-.24, -.03, len(dots)), dots.value,
               color='#777777', s=4, alpha=.20,
               linewidth=0, rasterized=True)
    for x, cat in zip(np.linspace(-.13, .13, len(order)), order):
        ax.plot(x, means[cat], marker='D', ms=5.2, color=colors[cat],
                mec='white', mew=.55, zorder=5)
    mean_values = np.array([means[c] for c in order])
    center, sd = float(mean_values.mean()), float(mean_values.std(ddof=0))
    cell_center, cell_sd = float(values.mean()), float(values.std(ddof=0))
    for x, mid, spread, direction in [(-.44, cell_center, cell_sd, 1),
                                       (.53, center, sd, -1)]:
        ax.plot([x, x], [mid-spread, mid+spread], color='#242424', lw=2.4, zorder=6)
        ax.plot([x-.025, x+.025], [mid-spread]*2, color='#242424', lw=1.2, zorder=6)
        ax.plot([x-.025, x+.025], [mid+spread]*2, color='#242424', lw=1.2, zorder=6)
    ax.set_xlim(-.59, .68)
    ax.set_ylim(response_ylim)
    ax.set_xticks([.08], ['Pooled'])
    if label_bars:
        lo, hi = response_ylim
        for x, label, mid, spread in [(-.44, 'cells', cell_center, cell_sd),
                                      (.53, 'means', center, sd)]:
            at_top = hi - (mid + spread) >= (mid - spread) - lo
            text = ax.text(x, .98 if at_top else .02, label,
                           transform=ax.get_xaxis_transform(), rotation=90,
                           ha='center', va='top' if at_top else 'bottom', fontsize=11,
                           zorder=8)
            text._sd_span = (x, mid-spread, mid+spread)
    results.clean(ax)
    ax.tick_params(labelsize=12, labelleft=False, left=False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return center, sd, cell_center, cell_sd


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    registry = pd.read_csv(REGISTRY, dtype={'unit_id': str, 'session_id': str})
    effects = pd.read_csv(results.PRIMARY / 'primary_effects.csv')
    point = pd.read_csv(results.PRIMARY / 'draw_specific_point_fits.csv')
    control = pd.read_csv(results.CONTROL / 'matched_control.csv')
    bootstrap = pd.read_csv(results.PRIMARY / 'bootstrap_effects.csv.gz')
    sd_effects, sd_draws = spread_effects(effects, bootstrap)
    sd_draws.to_csv(STEM.with_name(STEM.name + '_spread_bootstrap.csv.gz'), index=False)
    results.V1_COLORS.update({'A': '#B2182B', 'E': '#B59B19', 'C': '#008C95', 'B': '#5263A8'})
    results.DISPLAY.update({'A': 'Ant.', 'E': 'Lat.', 'C': 'Post.', 'B': 'Med.'})
    # Share categorical identity colors across all maps, nodes, and distributions.
    schematic.AREA_COLORS.update(results.HVA_COLORS)
    schematic.PROBE_COLORS.update({k: results.V1_COLORS[k] for k in schematic.PROBE_COLORS})
    for atlas_name, area in [('VISl', 'LM'), ('VISrl', 'RL'), ('VISal', 'AL'), ('VISpm', 'PM'), ('VISam', 'AM')]:
        schematic.ATLAS_FIGURE4_COLORS[atlas_name] = results.HVA_COLORS[area]

    fig = plt.figure(figsize=(7.0, 5.3 * 7.0 / 7.4), facecolor='white')
    atlas = schematic.read_nrrd(schematic.ATLAS_PATH)
    labels = schematic.read_labels(schematic.ATLAS_LABELS_PATH)
    # Explicit gutters keep the distributions left and reserve label space for E.
    outer = fig.add_gridspec(3, 5, width_ratios=[.270, .060, .252, .098, .225],
                            left=.075, right=.98, top=.635, bottom=.06, wspace=0, hspace=.26)
    a = fig.add_axes([.060, .725, .175, .22])
    b = fig.add_axes([.250, .725, .14, .22])
    c = fig.add_axes([.445, .725, .17, .22])
    d = fig.add_axes([.665, .725, .315, .22])
    schematic.draw_atlas_area_map(a, atlas, labels)
    schematic.draw_published_hierarchy(b)
    b.set_aspect('auto')
    b.set_xlim(-1.65, 1.10)
    for label in b.texts:
        if label.get_text() == 'anatomical\nhierarchy score':
            label.set_text('Anatomical score')
            label.set_position((-1.90, .02))
        elif label.get_text() in ['-0.4', '0.0', '0.4']:
            label.set_x(-1.0)
            label.set_ha('right')

    schematic.draw_atlas_area_map(c, atlas, labels, merged=True)
    for label in a.texts:
        if label.get_text() == 'LM':
            x, y = label.get_position()
            label.remove()
            a.annotate('LM', xy=(x, y), xytext=(-16, -18), textcoords='offset points',
                       ha='right', va='top', fontsize=12,
                       arrowprops=dict(arrowstyle='-', color='#333333', lw=.6))
        elif label.get_text() == 'P':
            x, y = label.get_position()
            label.remove()
            a.annotate('P', xy=(x, y), xytext=(-22, -8), textcoords='offset points',
                       ha='right', va='top', fontsize=12,
                       arrowprops=dict(arrowstyle='-', color='#333333', lw=.6))
    probe_positions = schematic.draw_probe_locations(
        c, atlas, labels, location_names=True, subjects=registry.loc[registry.population.eq('V1'), 'animal_id'].unique())
    for annotation in c.texts:
        name = annotation.get_text()
        if name in ['Anterior', 'Lateral', 'Medial', 'Posterior']:
            # Offsets are halved by the final print-size pass.
            offsets = {'Anterior': (-38, 20), 'Lateral': (-46, 12),
                       'Medial': (30, -9), 'Posterior': (0, -28)}
            annotation.set_position(offsets[name])
            if name == 'Anterior':
                annotation.set_ha('right')
                annotation.set_va('center')
            annotation.set_text({'Anterior': 'Ant.', 'Lateral': 'Lat.',
                                 'Posterior': 'Post.', 'Medial': 'Med.'}[name])
    assert all(row['n_animals'] == 8 for row in probe_positions)
    pd.DataFrame(probe_positions).to_csv(STEM.with_name(STEM.name + '_probe_positions.csv'), index=False)
    # Retain the existing categorical order; its original selection rationale
    # has not been established. All heights here are illustrative.
    schematic.clean_axis(d, (-.42, 6.9), (.40, 3.00))
    d.set_aspect('auto')
    d.annotate('', xy=(-.28, 2.87), xytext=(-.28, .62),
               arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#333333'))
    d.text(-.52, 1.73, 'Functional metric', rotation=90, ha='center', va='center', fontsize=12)
    groups = [(['PM', 'LM', 'AM', 'RL', 'AL'], [2.40, 2.62, 2.45, 2.56, 2.36],
               np.arange(5) + .26, results.HVA_COLORS, 'Across HVA'),
              (['A', 'E', 'C', 'B'], [1.09, 1.31, 1.17, 1.37],
               np.arange(4) * 1.2 + .55, results.V1_COLORS, 'Within V1')]
    for cats, heights, xs, colors, spread_label in groups:
        for cat, height, x in zip(cats, heights, xs):
            d.scatter(x, height, s=165, color=colors[cat], edgecolor='#333333', linewidth=.85, zorder=3)
            if spread_label == 'Across HVA':
                d.text(x, height + .17, cat, fontsize=12, ha='center', va='bottom')
            else:
                d.text(x + (.24 if cat == 'A' else 0), .90, results.DISPLAY[cat], fontsize=11, ha='right', va='top', rotation=15)
        center, sd = np.mean(heights), np.std(heights)
        d.plot([.05, 4.85], [center, center], color='#62666b', lw=1.5, ls=(0, (3, 2)), zorder=0)
        d.plot([4.85, 4.85], [center-sd, center+sd], color='#242424', lw=3)
        d.text(5.0, center, spread_label + '\nSD of means', fontsize=11, ha='left', va='center')
    # Conceptual difference between the two illustrative group centers.
    control_color = results.POP_COLORS['Control']
    hva_center = np.mean(groups[0][1])
    v1_center = np.mean(groups[1][1])
    d.annotate('', xy=(2.75, hva_center), xytext=(2.75, v1_center),
               arrowprops=dict(arrowstyle='->', lw=1.4, color=control_color,
                               shrinkA=0, shrinkB=0))
    d.text(3.12, (hva_center + v1_center) / 2, 'Mean shift', color=control_color,
           fontsize=11, ha='left', va='center')
    heading(fig, .085, 'A', 'Visual areas hierarchy')
    heading(fig, .445, 'B', 'Within-area benchmark')
    records = []
    estimate_records = []
    spread_records = []
    for row, metric in enumerate(results.METRICS):
        me = effects.loc[effects.metric.eq(metric)].set_index('effect')
        cr = control.loc[control.metric.eq(metric)].iloc[0]
        dr = sd_effects.loc[sd_effects.metric.eq(metric)].iloc[0]
        bounds = [0, cr.hva_minus_v1_low, cr.hva_minus_v1_high, dr.low, dr.high]
        span = max(bounds) - min(bounds)
        effect_ylim = (min(bounds) - .22 * span, max(bounds) + .17 * span)
        tables = {p: results.response_cells(registry, p, metric) for p in ['V1', 'HVA']}
        limits = [tables[p].value.quantile([.002, .998]).to_numpy() for p in ['V1', 'HVA']]
        lo, hi = min(v[0] for v in limits), max(v[1] for v in limits)
        pad = .08 * (hi - lo)
        effect_axes = []
        population_grids = [outer[row, 2 * i].subgridspec(1, 2, width_ratios=[n_categories, 1], wspace=.18)
                            for i, n_categories in enumerate([len(results.HVA_CATEGORIES), len(results.V1_CATEGORIES)])]
        for col, pop, cats, colors, key, title, panel in [
            (0, 'HVA', results.HVA_CATEGORIES, results.HVA_COLORS, 'hva', 'Across HVAs', 'C'),
            (1, 'V1', results.V1_CATEGORIES, results.V1_COLORS, 'v1', 'Within V1', 'D'),
        ]:
            ax = fig.add_subplot(population_grids[col][0, 0])
            means_rows = point.loc[point.metric.eq(metric) & point.population.eq(pop)]
            if metric == results.METRICS[2]:
                means_rows = means_rows.loc[means_rows.draw.eq(results.TIMESCALE_DISPLAY_DRAW)]
            adjusted_means = results.averaged_means(means_rows.category_means, cats)
            raw_means = tables[pop].groupby('category').value.mean().reindex(cats)
            # Anchor the display to the observed equal-category centre. A single
            # translation preserves all fitted category contrasts and their SD.
            adjusted_center = np.mean(list(adjusted_means.values()))
            display_shift = float(raw_means.mean() - adjusted_center)
            means = {cat: adjusted_means[cat] + display_shift for cat in cats}
            original_values = np.array([adjusted_means[cat] for cat in cats])
            display_values = np.array([means[cat] for cat in cats])
            assert np.allclose(display_values - display_values.mean(),
                               original_values - original_values.mean(), rtol=1e-12, atol=1e-12)
            results.violin(ax, tables[pop], cats, colors, means)
            for annotation in list(ax.texts):
                annotation.remove()  # Category sample sizes are in the caption table.
            ax.tick_params(labelsize=12)
            ax.set_xticklabels([results.DISPLAY.get(cat, cat) for cat in cats], rotation=0, ha='center')
            ax.set_ylim(lo - pad, hi + pad)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if row < 2:
                ax.tick_params(axis='x', labelbottom=False)
            if col == 0:
                ax.set_ylabel('Timescale (ms)' if metric == results.METRICS[2] else metric, fontsize=14)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                pos = ax.get_position()
                fig.text(pos.x0 - .03, .705, panel, fontsize=20, va='top', gid='panel-letter')
                ax.set_title(title, fontsize=12, pad=10)
            ea = fig.add_subplot(population_grids[col][0, 1])
            center, display_sd, cell_center, cell_sd = pooled_distribution(ea, tables[pop], means, cats, colors, ax.get_ylim(), label_bars=True)
            if row < 2:
                ea.tick_params(axis='x', labelbottom=False)
            if row == 0:
                ea.set_title('Pooled SD', fontsize=11, pad=10)
            spread_records.append(dict(metric=metric, population=pop, center=center,
                                       model_reference_center=adjusted_center, display_shift=display_shift,
                                       display_sd=display_sd, pooled_cell_center=cell_center, pooled_cell_sd=cell_sd,
                                       display_draw=results.TIMESCALE_DISPLAY_DRAW if metric == results.METRICS[2] else -1,
                                       inference_sd=np.sqrt(me.loc[key, 'estimate'])))
            for cat in cats:
                records.append({'panel': panel, 'population': pop, 'metric': metric, 'category': cat,
                                'n_cells': len(tables[pop].loc[tables[pop].category.eq(cat)]),
                                'category_mean': means[cat], 'model_reference_mean': adjusted_means[cat],
                                'raw_category_mean': raw_means[cat], 'display_shift': display_shift})
        sub = outer[row, 4].subgridspec(1, 2, wspace=.22)
        ca = fig.add_subplot(sub[0, 0])
        pos = ca.get_position()
        ca.set_position([pos.x0 + .25 * pos.width, pos.y0, .5 * pos.width, pos.height])
        results.effect(ca, cr.hva_minus_v1, cr.hva_minus_v1_low,
                       cr.hva_minus_v1_high, results.POP_COLORS['Control'], 'HVA − V1')
        ca.set_ylabel('Δ log10 F1/F0' if metric == results.METRICS[1] else 'Δ ms', fontsize=12)
        # Preserve the physical label-to-axis distance as the plot narrows.
        ca.yaxis.set_label_coords(-1.28, .5)
        da = fig.add_subplot(sub[0, 1])
        pos = da.get_position()
        da.set_position([pos.x0 + .25 * pos.width, pos.y0, .5 * pos.width, pos.height])
        results.effect(da, dr.estimate, dr.low, dr.high,
                       results.POP_COLORS['Difference'], 'Across HVA\n− within V1')
        da.set_ylabel('')
        effect_axes = [ca, da]
        if row == 0:
            ca.set_title('Mean shift', fontsize=12, pad=10)
            da.set_title('SD contrast', fontsize=11, pad=10)
        if row == 0:
            fig.text(outer[0, 4].get_position(fig).x0 - .045, .705, 'E', ha='left',
                     fontsize=20, va='top', gid='panel-letter')
        for axis in effect_axes:
            # Exact values live in the accompanying table; leave marks unobstructed.
            for annotation in list(axis.texts):
                annotation.remove()
            axis.set_ylim(effect_ylim)
            axis.tick_params(labelsize=12)
            if row < 2:
                axis.tick_params(axis='x', labelbottom=False)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        da.tick_params(labelleft=False, left=False)
        da.spines['left'].set_visible(False)
        estimate_records.append(dict(metric=metric, effect='hva_minus_v1_sd',
                                     estimate=dr.estimate, low=dr.low, high=dr.high))
        estimate_records.append(dict(metric=metric, effect='matched_hva_minus_v1_mean',
                                     estimate=cr.hva_minus_v1, low=cr.hva_minus_v1_low,
                                     high=cr.hva_minus_v1_high))
    pd.DataFrame(spread_records).to_csv(STEM.with_name(STEM.name + '_mean_spreads.csv'), index=False)
    counts = pd.DataFrame(records)
    old = pd.read_csv(OUT / 'figure4_display_counts.csv')
    old = old.loc[old.figure.eq('Figure4_CDE_rerun')]
    joined = counts.merge(old, on=['metric', 'population', 'category'], suffixes=('_new', '_original'), validate='one_to_one')
    assert len(joined) == len(counts) == len(old)
    assert joined.n_cells_new.eq(joined.n_cells_original).all(), 'Display population changed'
    counts.to_csv(STEM.with_name(STEM.name + '_display.csv'), index=False)
    estimates = pd.DataFrame(estimate_records)
    estimates.to_csv(STEM.with_name(STEM.name + '_estimates.csv'), index=False)
    # Print-size typography is deliberately larger than proportional reduction.
    from matplotlib.text import Text, Annotation
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, LineCollection
    for artist in fig.findobj(Text):
        if artist.get_gid() not in {'panel-letter', 'panel-heading'}:
            artist.set_fontsize(max(7.0, artist.get_fontsize() * .60))
        if isinstance(artist, Annotation) and artist.anncoords == 'offset points':
            artist.set_position(tuple(v * .5 for v in artist.get_position()))
    for artist in fig.findobj(Line2D):
        artist.set_linewidth(artist.get_linewidth() * .55)
        artist.set_markersize(artist.get_markersize() * .55)
        artist.set_markeredgewidth(artist.get_markeredgewidth() * .55)
    for artist in fig.findobj(PathCollection):
        artist.set_sizes(artist.get_sizes() * .25)
        artist.set_linewidths(artist.get_linewidths() * .55)
    for artist in fig.findobj(LineCollection):
        artist.set_linewidths(artist.get_linewidths() * .55)
    for axis in fig.axes:
        axis.tick_params(pad=1.5, length=2, width=.5)
        axis.xaxis.labelpad = 2
        axis.yaxis.labelpad = 2
        axis.set_title(axis.get_title(), pad=4, fontsize=9)
        for spine in axis.spines.values():
            spine.set_linewidth(.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if hasattr(text, '_sd_span'):
            x, lo, hi = text._sd_span
            y0, y1 = text.axes.transData.transform([(x, lo), (x, hi)])[:, 1]
            box = text.get_window_extent(renderer)
            assert box.y1 < y0 or box.y0 > y1, 'SD label overlaps its bar'
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(STEM.with_suffix('.' + suffix), dpi=400, facecolor='white')
    plt.close(fig)
    legend = """# Figure 4 — Functional variation within V1 and across higher visual areas

**A–B, anatomy and schematic.** A combines the Allen CCFv3 dorsal visual-area map with the anatomical-hierarchy schematic. The anatomical-hierarchy schematic shows the published scores from Siegle et al. (2021) for V1 and five higher visual areas, without renormalization. Curved links are schematic visual context, not a measured or exhaustive connectivity graph; their widths do not encode connection strengths. B uses the same atlas geometry with non-V1 masks merged. V1 locations use a distinct warm-to-cool palette (anterior crimson, lateral ochre, posterior turquoise, medial slate blue) consistently in B and D, and their mean diamonds; HVA colors are reserved for HVA identity. Probe markers show the mean AP/ML coordinates across eight animals of the most superficial VISp-labelled contact on each probe. B and D use Ant., Lat., Post., and Med. for anterior, lateral, posterior, and medial V1 sampling locations. B (Within-area benchmark) illustrates functional-metric spread with deliberately unordered, uneven category means. V1 locations are displayed in the fixed categorical order anterior, lateral, posterior, medial; horizontal position does not indicate a hierarchy score or correspondence to an HVA. Its vertical bars show illustrative ±1 SD of means, labelled Across HVA and Within V1, with dashed lines marking the group centers. The orange mean-shift arrow connects the two illustrative group centers to explain a difference in means. This is a conceptual schematic, not an estimate of the difference between the populations in C and D; the measured positive control in E compares matched Allen V1 and HVA recordings. These are not measured hierarchy scores or empirical SD estimates.

**C–D, cellular distributions and pooled spread.** Half-violins show eligible single-neuron distributions; dots are deterministic display subsamples. Colored diamonds show model-standardized category means translated by one common constant per population and metric: displayed mean = standardized mean − equal-category standardized center + equal-category raw center. Thus their center matches the equally weighted mean of the raw category means, while every fitted category contrast and category-mean SD is preserved. Each adjacent pooled panel combines all eligible cells with equal per-cell weight, showing a gray half-violin and up to 800 gray cell dots. Pooled distributions are raw, without recentering or session adjustment. Repeated diamonds use the same translated adjusted category means as the individual plots, with small horizontal offsets solely for visibility. The left vertical bar spans the raw pooled cell mean ±1 population SD; the right vertical bar spans the translated, equally weighted adjusted category-mean center ±1 finite-category SD (both ddof=0). “Means” refers to area means in C and location means in D. Vertical bars with small end caps show these ±1 SD spans; they describe spread, not uncertainty. Pooled cellular SD includes within- and between-category variation; these two bars do not constitute a formal variance decomposition because they have different weighting and adjustment. Categories, animals within categories, recordings within animal/category, and cells within recordings receive equal nested fitting weight for the adjusted diamonds and E spread contrasts. Session fixed effects are nuisance terms omitted from standardized predictions. The common display shift anchors the adjusted means to the observed category center without changing their contrasts or SD. The cells bar retains its raw pooled, equal-cell center, which need not equal the equal-category center of the means bar. The diamonds remain adjusted estimates, not raw category means. C/D absolute centers are not the matched control comparison. Across-HVA distributions come from Allen and within-V1 distributions from MouseV2; shared response axes show common units, not a controlled comparison of their absolute levels, which may reflect differences in rigs, stimuli, and animals.

For response timescale, the violins, diamonds, and descriptive SD bracket all use fixed display draw 1 of 10 and its draw-specific five-cell floor. The timescale SD contrast in E uses all ten draws and can therefore differ from the contrast of the displayed SDs. Other metrics have one accepted value per neuron. Response axes show the central 99.6% with display padding; all eligible values enter the fits. Category sample sizes and displayed means are listed in the accompanying tables.

**E, signed control and direct spread contrast.** Both E columns show point estimates and 95% whole-animal bootstrap confidence intervals. The mean-shift control is the fitted mean difference, equal-area pooled HVA minus matched Allen V1, with a 95% whole-animal bootstrap interval. It uses matched Allen sessions, not the MouseV2 V1 population in D. Lower phase modulation is associated with higher hierarchical levels; the negative HVA-minus-V1 phase-modulation mean shift is therefore in the expected direction. This directional interpretation concerns the mean shift, not the SD contrast. E response-difference units are shown on the first column for both columns in each row. The second column, labelled “Across HVA − within V1,” contrasts across-HVA with within-V1 category-mean spread: sqrt(V_HVA) − sqrt(V_V1), in response units. For timescale, V for each population is the mean of its ten complete draw-specific variances; the square root is applied after averaging. In each bootstrap replicate, the paired V1 and HVA variances are separately square-rooted before taking their difference. The 95% interval is the 2.5th–97.5th percentile of those paired SD contrasts. The bootstrap samples whole animals and selects a coherent matched-trial draw per MouseV2 session occurrence. Intervals are pointwise and do not establish equivalence. C/D use a shared absolute-response scale within each row; the two E columns share a difference scale within each row. All summaries use ms for TTFS/timescale and log10 F1/F0 units for phase modulation.

Descriptive cell-level variance associated with category identity was estimated using one-way ω² = [SS_between − (K−1) MS_within] / [SS_total + MS_within], with equal cell weights. TTFS (ms): V1 ω² = -0.0010 (95% CI -0.0024 to 0.0259); HVA ω² = 0.0173 (95% CI 0.0086 to 0.0369). log10 F1/F0: V1 ω² = 0.0086 (95% CI 0.0024 to 0.0229); HVA ω² = 0.0021 (95% CI 0.0002 to 0.0089). Response timescale (ms): V1 ω² = 0.0055 (95% CI -0.0008 to 0.0281); HVA ω² = 0.0223 (95% CI 0.0134 to 0.0390). Intervals use 5,000 whole-animal bootstrap replicates. Raw category means enter this calculation, not the session-adjusted diamonds. SS_total equals N times the squared pooled-cell population SD, linking the denominator to the full cellular spread. Negative estimates are retained as a consequence of bias correction; they do not imply negative population variance. This is an unadjusted association measure, not a multilevel variance decomposition. V1 timescale estimates average ten draw-specific ω² values; bootstrap samples select one coherent draw per sampled session occurrence.

Analyses use harmonized metric definitions, common waveform quality criteria, metric-specific validity filters, an anatomical VISp restriction for MouseV2, and at least five eligible cells per recording/category. TTFS is response-selected preferred-polarity latency below 100 ms; phase modulation is log10 harmonized F1/F0; timescale requires valid 1–300 ms fits with >50 spikes and fit error <20 ms. These are plug-in spreads of estimated category means, without an explicit correction for estimation-noise inflation.

Anatomical hierarchy scores: Siegle et al. (2021), https://pmc.ncbi.nlm.nih.gov/articles/PMC10399640/#F2.
"""
    legend += '\n\nAn exploratory sensitivity analysis assesses estimation-noise inflation of category-mean variance. It subtracts tr(P C)/K from the plug-in variance of fitted category means, where P centers the K means and C is their animal-clustered covariance with a CR1 correction. It accounts approximately for unequal precision and correlated mean estimates but is not guaranteed unbiased, particularly with only eight V1 animals. Corrected point variances are negative for V1 TTFS and HVA phase modulation; these are retained and not square-rooted or truncated to zero. Consequently a corrected SD contrast is undefined for those metrics. This sensitivity analysis provides corrected point estimates, not corrected confidence intervals; E shows the uncorrected SD contrast. Neither this correction nor the cell-level omega-squared analysis removes between-dataset confounding.\n'
    legend += '\n## Displayed pooled SD values\n\n| Metric | Population | Cell SD | Category-mean SD |\n|---|---|---:|---:|\n'
    for r in spread_records:
        legend += f"| {r['metric']} | {r['population']} | {r['pooled_cell_sd']:.4g} | {r['display_sd']:.4g} |\n"
    legend += '\n## Category sample sizes and displayed means\n\n| Panel | Metric | Category | Cells | Mean |\n|---|---|---|---:|---:|\n'
    for r in counts.itertuples():
        legend += f'| {r.panel} | {r.metric} | {r.category} | {r.n_cells} | {r.category_mean:.6g} |\n'
    legend += '\n## Exact effect estimates\n\n| Metric | Effect | Estimate | 95% interval |\n|---|---|---:|---:|\n'
    for r in estimates.itertuples():
        legend += f'| {r.metric} | {r.effect} | {r.estimate:.6g} | [{r.low:.6g}, {r.high:.6g}] |\n'
    STEM.with_name(STEM.name + '_legend.md').write_text(legend)
    sources = [REGISTRY, results.PRIMARY / 'primary_effects.csv', results.PRIMARY / 'draw_specific_point_fits.csv',
               results.CONTROL / 'matched_control.csv', results.PRIMARY / 'bootstrap_effects.csv.gz', schematic.PROBE_ENTRIES_PATH, schematic.ATLAS_PATH, schematic.ATLAS_LABELS_PATH]
    manifest = {'description': 'Pooled distribution presentation: C/D pooled half-violins, cells, adjusted means, and cellular/mean SD brackets; E signed control and paired SD contrast. Source populations retained.',
                'probe_location_overlay': {'excluded_subjects': [], 'n_animals': 8,
                                           'exclusion_scope': 'Anatomical overlay only; response populations unchanged'},
                'renderer': str(Path(__file__).relative_to(ROOT)), 'display_count_check': 'All 27 category counts match original.',
                'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    STEM.with_name(STEM.name + '_layout_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(STEM.with_suffix('.pdf'))
    print('Verified all 27 display counts against the original figure.')


if __name__ == '__main__':
    main()
