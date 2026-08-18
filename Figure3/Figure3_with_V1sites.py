"""
Figure3_with_V1sites.py

Extends Figure3.py to overlay all configured V1 multi-site (MouseV2) data.
The grating panel can show either matched F1/F0 or matched Allen modulation
index from a versioned full-condition import. Site sessions can retain the
legacy pseudo-hierarchy geometry for regression checks or use explicitly
non-metric display offsets around VISp. Measured RF position is shown in a
companion view.
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress, pearsonr, spearmanr, ranksums
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ── Paths ────────────────────────────────────────────────────────────────────
code_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if code_directory not in sys.path:
    sys.path.insert(0, code_directory)

from common.figure3_mousev2 import (  # noqa: E402
    load_allen_units,
    load_config,
    load_mousev2_units,
    site_coarse_label,
    within_v1_x_positions,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--output-dir',
    type=Path,
    default=Path(code_directory) / 'Figure3',
    help='Directory for the generated PNG (default: Figure3).',
)
parser.add_argument('--config', type=Path, default=None)
parser.add_argument('--grating-metrics-dir', type=Path, default=None)
parser.add_argument('--flash-metrics-dir', type=Path, default=None)
parser.add_argument(
    '--flash-variant', choices=('pooled', 'bright', 'dark'), default='pooled'
)
parser.add_argument(
    '--ttfs-display', choices=('mean_matched', 'raw_nwb'), default='mean_matched'
)
parser.add_argument(
    '--within-v1-x-mode',
    choices=('legacy_pseudo_hierarchy', 'display_only'),
    default='legacy_pseudo_hierarchy',
)
parser.add_argument(
    '--grating-metric', choices=('f1_f0_dg', 'mod_idx_dg'), default='f1_f0_dg'
)
parser.add_argument('--population-profile', default=None)
args = parser.parse_args()
output_dir = args.output_dir.resolve()
config = load_config(args.config)

# ── Colour palette ───────────────────────────────────────────────────────────
_V1_BASE = np.array([129, 116, 177]) / 255
_SITE_LABELS = [f"V1_s{s['site_number']}" for s in config['sessions']]

def _build_color_palette():
    colors = [[217,141,194],[129,116,177],[78,115,174],[101,178,201],
              [88,167,106],[202,183,120],[219,132,87],[194,79,84]]
    to_f = lambda c: tuple(v/255 for v in c)
    palette = dict(zip(('LGd','V1','LM','RL','LP','AL','PM','AM'),
                        [to_f(c) for c in colors]))
    # 8 site sessions: tints from very light to full V1 purple
    tints = np.linspace(0.35, 1.0, len(_SITE_LABELS))
    for label, t in zip(_SITE_LABELS, tints):
        palette[label] = tuple(1 - t * (1 - _V1_BASE))
    return palette

_COLOR_PALETTE = _build_color_palette()

def get_color_palette(area):
    return _COLOR_PALETTE.get(area, '#C9C9C9')

# ── Load original data (unit_table.csv — Allen SDK F1/F0, TTFS, timescale) ──
df_orig = load_allen_units(
    args.config, population_profile=args.population_profile
)

# ── Load all V1 site data (site2_processed, site3_processed, …) ──────────────
print(f"Configured {len(config['sessions'])} site directories: "
      f"{[s['site'] + '_processed' for s in config['sessions']]}")
df_site2 = load_mousev2_units(
    apply_qc=False,
    config_path=args.config,
    grating_metrics_dir=args.grating_metrics_dir,
    flash_metrics_dir=args.flash_metrics_dir,
    flash_variant=args.flash_variant,
    population_profile=args.population_profile,
)

# Map each V1_siteN_X probe label → per-session coarse label V1_sN
if not df_site2.empty:
    df_site2['area_coarse'] = df_site2['ecephys_structure_acronym'].map(site_coarse_label)
    df_site2 = df_site2[df_site2['area_coarse'].notna()]
    found_sites = sorted(df_site2['area_coarse'].unique())
    print(f'Site sessions loaded: {found_sites}  '
          f'({len(df_site2)} units total)')

df = pd.concat([df_orig, df_site2], ignore_index=True, sort=False)

# ── Areas & hierarchy scores ─────────────────────────────────────────────────
orig_areas  = ('LGd','V1','LM','RL','LP','AL','PM','AM')
# Only include sessions that are actually present in the data
site2_areas = tuple(s for s in _SITE_LABELS if not df_site2.empty and
                    s in df_site2['area_coarse'].values)
areas       = orig_areas + site2_areas

_visp_hs = -0.357
hierarchy_score = {
    'LGd':-0.515, 'V1':-0.357, 'LM':-0.093, 'RL':-0.059,
    'LP':0.105,   'AL':0.152,  'PM':0.327,   'AM':0.441,
}
# Historical geometry spread sessions over a broad pseudo-hierarchy range.
# The reviewed mode uses only small, explicitly non-metric display offsets.
hierarchy_score.update(
    within_v1_x_positions(
        _SITE_LABELS,
        args.within_v1_x_mode,
        visp_score=_visp_hs,
        legacy_bounds=(_visp_hs - 0.16, _visp_hs + 0.16),
        display_half_span=0.035,
    )
)
HS     = [hierarchy_score[a] for a in areas]
n_areas = len(areas)

# ── Metric configuration ─────────────────────────────────────────────────────
_grating_label = 'F1/F0' if args.grating_metric == 'f1_f0_dg' else 'modulation index'
metrics = ['time_to_first_spike_fl', args.grating_metric, 'timescale_ac']
labels  = ['Time to first spike (ms)',
           f'$log_{{10}}$ {_grating_label}',
           'Response decay timescale (ms)']
bins    = [np.linspace(15, 120, 30),
           np.linspace(-1.5, 2.0, 50),    # log10(F1/F0)
           np.linspace(0, 150, 50)]
function_to_apply = [
    lambda v: v * 1000,                   # s → ms
    lambda v: np.log10(np.clip(v, 1e-6, None)),  # F1/F0 → log10
    lambda v: v,
]
y_vals = [60, 0.3, 35]   # y-position for correlation text in each panel

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_bootstrap_95ci(M, f, N=500):
    if len(M) < 5: return np.nan
    n = max(1, int(len(M) / 2))
    est = np.array([f(M[np.random.permutation(len(M))[:n]]) for _ in range(N)])
    return np.percentile(est, 97.5) - np.nanmean(est)

measure = np.nanmean
np.random.seed(10)

# ── Diagnostics ──────────────────────────────────────────────────────────────
print('Metric n per area (coarse):')
for m in metrics:
    if m in df.columns:
        counts = df.groupby('area_coarse')[m].apply(
            lambda x: int(np.sum(np.isfinite(x.astype(float)))))
        line = ', '.join(f'{a}:{counts[a]}' for a in list(orig_areas)+list(site2_areas) if a in counts)
        print(f'  {m}: {line}')

# ── Main loop ────────────────────────────────────────────────────────────────
centers    = np.full((n_areas, len(metrics)), np.nan)
errorbars  = np.full((n_areas, len(metrics)), np.nan)
max_values = np.zeros(len(metrics))
all_values = {mi: {} for mi in range(len(metrics))}

plt.figure(14782, figsize=(22, 12))
plt.clf()

for area_idx, area in enumerate(areas):
    is_site2 = area in site2_areas
    lw    = 1.5 if is_site2 else 2.0
    alpha = 0.70 if is_site2 else 0.85
    ls    = '--'  if is_site2 else '-'

    base_sel = (df.area_coarse == area)

    for metric_idx, metric in enumerate(metrics):
        sel = base_sel.copy()

        if metric_idx == 0:   # TTFS
            sel &= (df.time_to_first_spike_fl < 0.1)
        elif metric_idx == 1: # f1_f0_dg — exclude zeros
            sel &= (df[metric].astype(float) > 0)
        elif metric_idx == 2: # timescale
            sel &= (df[metric].astype(float) < 300) & (df[metric].astype(float) > 1)
            sel &= (df.spike_count_ac.astype(float) > 50) & (df.err_ac.astype(float) < 20)

        if metric not in df.columns:
            M = np.array([])
        else:
            M = function_to_apply[metric_idx](df.loc[sel, metric].astype(float).values)

        M = M[np.isfinite(M)]
        all_values[metric_idx][area] = M

        if len(M) < 5:
            continue

        h, b = np.histogram(M, bins=bins[metric_idx], density=True)
        h_filt = gaussian_filter1d(h, 1.5)
        max_values[metric_idx] = max(np.max(h_filt), max_values[metric_idx])

        plt.subplot(len(metrics), 4, metric_idx*4+1)
        plt.plot(b[:-1], h_filt, color=get_color_palette(area),
                 lw=lw, alpha=alpha, ls=ls)
        if area_idx == n_areas - 1:
            plt.xlabel(labels[metric_idx])

        plt.subplot(len(metrics), 4, metric_idx*4+2)
        cdf = np.cumsum(h_filt)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
        plt.plot(b[:-1], cdf, color=get_color_palette(area),
                 lw=lw, alpha=alpha, ls=ls)
        if area_idx == n_areas - 1:
            plt.xlabel(labels[metric_idx])

        centers[area_idx, metric_idx]   = measure(M)
        errorbars[area_idx, metric_idx] = get_bootstrap_95ci(M, measure)

print('TOTAL original: ' + str(int(df[df.area_coarse.isin(orig_areas)].shape[0])))
print('TOTAL site2:    ' + str(int(df[df.area_coarse.isin(site2_areas)].shape[0])))

# ── Scatter: metric vs. hierarchy score ──────────────────────────────────────
for i in range(len(metrics)):
    plt.subplot(len(metrics), 4, i*4+3)

    orig_mask = np.array([a in orig_areas for a in areas])
    x_orig    = np.array(HS)[orig_mask]
    y_orig    = centers[orig_mask, i]
    fin       = np.isfinite(y_orig)

    # Original area dots
    for k, area in enumerate(areas):
        if area not in orig_areas: continue
        plt.plot(HS[k], centers[k,i], 'o', ms=7,
                 color=get_color_palette(area), zorder=3)
        plt.errorbar(HS[k], centers[k,i], yerr=errorbars[k,i],
                     fmt='none', ecolor=get_color_palette(area), lw=1.2, zorder=2)

    # Regression line (original 8 areas)
    if np.sum(fin) >= 3:
        slope, intercept, r, p, _ = linregress(x_orig[fin], y_orig[fin])
        x2 = np.linspace(-0.75, 0.5, 10)
        plt.plot(x2, x2*slope+intercept, '--k', alpha=0.5, lw=1.5, zorder=1)
        r_s, p_s = spearmanr(x_orig[fin], y_orig[fin])
        r_p, p_p = pearsonr(x_orig[fin], y_orig[fin])
        plt.text(-0.30, y_vals[i],
                 f'$r_P$={r_p:.2f}; $P_P$={p_p:.4f}\n$r_S$={r_s:.2f}; $P_S$={p_s:.4f}',
                 fontsize=7)

    # Site session stars (raw)
    for k, area in enumerate(areas):
        if area not in site2_areas: continue
        if not np.isfinite(centers[k, i]): continue
        plt.plot(HS[k], centers[k, i], '*', ms=11,
                 color=get_color_palette(area),
                 markeredgecolor='k', markeredgewidth=0.5, zorder=4)
        plt.errorbar(HS[k], centers[k, i], yerr=errorbars[k, i],
                     fmt='none', ecolor=get_color_palette(area), lw=1.2, zorder=3)

    # TTFS only: add mean-matched (offset-corrected) copies as hollow stars
    if i == 0 and args.ttfs_display == 'mean_matched':
        v1_idx   = list(areas).index('V1')
        site_idxs = [list(areas).index(a) for a in site2_areas if a in areas]
        site_vals = [centers[k, 0] for k in site_idxs if np.isfinite(centers[k, 0])]
        if site_vals and np.isfinite(centers[v1_idx, 0]):
            ttfs_shift = centers[v1_idx, 0] - np.mean(site_vals)
            for k, area in enumerate(areas):
                if area not in site2_areas: continue
                raw_y = centers[k, i]
                if not np.isfinite(raw_y): continue
                corr_y = raw_y + ttfs_shift
                # Thin dotted line connecting raw to corrected
                plt.plot([HS[k], HS[k]], [raw_y, corr_y], ':',
                         color=get_color_palette(area), lw=1.0, zorder=3)
                # Hollow star at corrected position
                plt.plot(HS[k], corr_y, '*', ms=11,
                         color='white',
                         markeredgecolor=get_color_palette(area),
                         markeredgewidth=1.2, zorder=5)

    plt.ylabel(labels[i])

# Legend in the first scatter panel
ax_leg = plt.subplot(len(metrics), 4, 3)
handles = [plt.Line2D([0],[0], marker='*', ms=10, linestyle='none',
                       color=get_color_palette(a), markeredgecolor='k',
                       markeredgewidth=0.5, label=a)
           for a in site2_areas]
if args.ttfs_display == 'mean_matched':
    handles += [
        plt.Line2D([0],[0], marker='*', ms=10, linestyle='none',
                   color='white', markeredgecolor='grey', markeredgewidth=1.2,
                   label='TTFS corrected\n(mean-matched to V1)'),
    ]
ax_leg.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.6,
               title='V1 multi-site sessions', title_fontsize=7)
if args.within_v1_x_mode == 'display_only':
    ax_leg.text(
        0.02,
        0.02,
        'MouseV2 horizontal offsets are display-only\n'
        '(all recordings in V1; no within-V1 hierarchy score)',
        transform=ax_leg.transAxes,
        fontsize=7,
        color='#666',
        style='italic',
        ha='left',
        va='bottom',
    )

# ── P-value matrix (original 8 areas) ────────────────────────────────────────
common_names = ['LGN','V1','LM','RL','LP','AL','PM','AM']

for mi, metric in enumerate(metrics):
    n8   = len(orig_areas)
    comp = np.zeros((n8, n8))
    for i1, a1 in enumerate(orig_areas):
        for i2, a2 in enumerate(orig_areas):
            if i2 <= i1: continue
            v1 = all_values[mi].get(a1, np.array([]))
            v2 = all_values[mi].get(a2, np.array([]))
            if len(v1) < 5 or len(v2) < 5: continue
            _, p = ranksums(v1, v2)
            comp[i1, i2] = max(p, 1e-10)

    flat = comp.flatten()
    ok   = np.where(flat > 0)[0]
    if len(ok) == 0: continue
    _, corr, _, _ = multipletests(flat[ok], alpha=0.05, method='fdr_bh')
    corr2 = np.zeros_like(flat); corr2[ok] = corr
    safe  = np.reshape(corr2, comp.shape)
    safe[safe <= 0] = 1e-10

    plt.subplot(len(metrics), 4, mi*4+4)
    plt.imshow(np.log10(safe), cmap='bone', vmin=-5, vmax=np.log10(0.05))
    plt.colorbar(fraction=0.026, pad=0.04)
    plt.xticks(np.arange(n8), common_names, fontsize=7, rotation=45)
    plt.yticks(np.arange(n8), common_names, fontsize=7)
    plt.ylim([-0.5, n8-0.5])
    plt.xlim([-0.5, n8-0.5])

# ── Cosmetics ─────────────────────────────────────────────────────────────────
for mi in range(len(metrics)):
    ax1 = plt.subplot(len(metrics), 4, mi*4+1)
    [ax1.spines[s].set_visible(False) for s in ('top','right')]
    if max_values[mi] > 0:
        ax1.set_ylim(0, max_values[mi] * 1.05)

    ax2 = plt.subplot(len(metrics), 4, mi*4+2)
    [ax2.spines[s].set_visible(False) for s in ('top','right')]
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if mi == 0:
        ax2.set_ylabel('Cumulative fraction')

    ax3 = plt.subplot(len(metrics), 4, mi*4+3)
    [ax3.spines[s].set_visible(False) for s in ('top','right')]
    ax3.set_xlim(-0.85, 0.65)
    if mi == len(metrics) - 1 and args.within_v1_x_mode == 'display_only':
        ax3.set_xlabel('Published inter-area hierarchy score')

plt.tight_layout()
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / 'Figure3_with_V1sites.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
