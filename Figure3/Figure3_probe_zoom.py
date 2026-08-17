"""
Figure3/Figure3_probe_zoom.py

Zoomed scatter: each metric vs. position, showing:
  - Original 8 areas (circles) at their hierarchy scores
  - New V1 sessions split by probe letter (A/B/C/E), each session as a dot,
    placed between V1 and LM on the x-axis
Purpose: compare within-V1 probe-to-probe variance to the inter-area gradient.
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

code_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if code_directory not in sys.path:
    sys.path.insert(0, code_directory)

from common.figure3_mousev2 import (  # noqa: E402
    load_allen_units,
    load_config,
    load_mousev2_units,
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

# ── Load original data (Allen SDK metrics) ────────────────────────────────────
df_orig = load_allen_units(
    args.config, population_profile=args.population_profile
)

# ── Load all MouseV2 site data ────────────────────────────────────────────────
df_site = load_mousev2_units(
    apply_qc=False,
    config_path=args.config,
    grating_metrics_dir=args.grating_metrics_dir,
    flash_metrics_dir=args.flash_metrics_dir,
    flash_variant=args.flash_variant,
    population_profile=args.population_profile,
)
print(f'Site data: {len(df_site)} units, '
      f'{len(df_site["session_num"].unique())} sessions, '
      f'probes {sorted(df_site["probe_letter"].unique())}')

# ── Colour palettes ───────────────────────────────────────────────────────────
def orig_color(area):
    _pal = {'LGd': (217, 141, 194), 'V1': (129, 116, 177), 'LM': (78, 115, 174),
            'RL': (101, 178, 201),  'LP':  (88, 167, 106), 'AL': (202, 183, 120),
            'PM': (219, 132,  87),  'AM': (194,  79,  84)}
    return tuple(v / 255 for v in _pal.get(area, (180, 180, 180)))

probe_color = {'A': '#d73027', 'B': '#4575b4', 'C': '#1a9850', 'E': '#8073ac'}

# ── Hierarchy scores & x-axis positions ──────────────────────────────────────
orig_areas = ('LGd', 'V1', 'LM', 'RL', 'LP', 'AL', 'PM', 'AM')
_hs = {'LGd': -0.515, 'V1': -0.357, 'LM': -0.093, 'RL': -0.059,
       'LP':   0.105, 'AL':  0.152, 'PM':  0.327, 'AM':  0.441}

# The legacy view inserts probe labels between V1 and LM as if they had numeric
# hierarchy positions. The reviewed display-only view centers categorical
# offsets on VISp and never fits a trend through them.
probe_letters = tuple(config['display_probe_order'])
probe_x = within_v1_x_positions(
    probe_letters,
    args.within_v1_x_mode,
    visp_score=_hs['V1'],
    legacy_bounds=(-0.32, -0.12),
    display_half_span=0.06,
)
_gap_l, _gap_r = min(probe_x.values()), max(probe_x.values())

sessions = sorted(df_site['session_num'].unique()) if not df_site.empty else []
n_sess = len(sessions)
_session_half_span = 0.006 if args.within_v1_x_mode == 'display_only' else 0.018
_jitter = (
    np.linspace(-_session_half_span, _session_half_span, n_sess)
    if n_sess > 1
    else np.array([0.0])
)
sess_jit = {s: _jitter[i] for i, s in enumerate(sessions)}

# ── Metrics ───────────────────────────────────────────────────────────────────
_grating_label = 'F1/F0' if args.grating_metric == 'f1_f0_dg' else 'modulation index'
_metrics = ['time_to_first_spike_fl', args.grating_metric, 'timescale_ac']
_labels  = ['Time to first spike (ms)',
            f'$\\log_{{10}}$ {_grating_label}',
            'Response decay timescale (ms)']
_fns     = [lambda v: v * 1000,
            lambda v: np.log10(np.clip(v, 1e-6, None)),
            lambda v: v]

def _filt(df_sub, mi):
    sel = pd.Series(True, index=df_sub.index)
    if mi == 0:
        sel &= df_sub['time_to_first_spike_fl'].astype(float) < 0.1
    elif mi == 1:
        sel &= df_sub[args.grating_metric].astype(float) > 0
    elif mi == 2:
        sel &= df_sub['timescale_ac'].astype(float).between(1, 300)
        sel &= df_sub['spike_count_ac'].astype(float) > 50
        sel &= df_sub['err_ac'].astype(float) < 20
    return sel

def _get(df_sub, mi):
    if _metrics[mi] not in df_sub.columns:
        return np.array([])
    sel = _filt(df_sub, mi)
    v = _fns[mi](df_sub.loc[sel, _metrics[mi]].astype(float).values)
    return v[np.isfinite(v)]

np.random.seed(42)

def _bootstrap_ci(v, N=500):
    if len(v) < 5:
        return np.nan
    n = max(1, len(v) // 2)
    est = [np.nanmean(v[np.random.permutation(len(v))[:n]]) for _ in range(N)]
    return np.percentile(est, 97.5) - np.nanmean(est)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
fig.subplots_adjust(hspace=0.3, left=0.11, right=0.97, top=0.93, bottom=0.1)

for mi in range(3):
    ax = axes[mi]

    # Shaded band marking the within-V1 probe region
    ax.axvspan(_gap_l - 0.025, _gap_r + 0.025,
               alpha=0.07, color='mediumpurple', zorder=0)

    # ── Original areas: dot + error bar ──────────────────────────────────
    orig_xs, orig_ys = [], []
    for area in orig_areas:
        v = _get(df_orig[df_orig['area_coarse'] == area], mi)
        if len(v) < 5:
            continue
        mu = np.nanmean(v)
        ci = _bootstrap_ci(v)
        x  = _hs[area]
        ax.errorbar(x, mu, yerr=ci, fmt='none', ecolor=orig_color(area), lw=1.5, zorder=2)
        ax.plot(x, mu, 'o', ms=9, color=orig_color(area), zorder=3)
        orig_xs.append(x)
        orig_ys.append(mu)

    # Regression line through original 8 areas
    if len(orig_xs) >= 3:
        slope, intercept, *_ = linregress(orig_xs, orig_ys)
        x2 = np.linspace(-0.75, 0.55, 100)
        ax.plot(x2, slope * x2 + intercept, '--k', alpha=0.30, lw=1.2, zorder=1)

    # ── Probe × session dots ──────────────────────────────────────────────
    if not df_site.empty:
        # For TTFS: compute correction shift so probe means align to original V1
        ttfs_shift = 0.0
        if mi == 0 and args.ttfs_display == 'mean_matched':
            v1_v = _get(df_orig[df_orig['area_coarse'] == 'V1'], 0)
            v1_mean = np.nanmean(v1_v) if len(v1_v) >= 5 else np.nan
            all_probe_means = []
            for probe in probe_letters:
                for sess in sessions:
                    sub = df_site[(df_site['probe_letter'] == probe) &
                                  (df_site['session_num'] == sess)]
                    v = _get(sub, 0)
                    if len(v) >= 5:
                        all_probe_means.append(np.nanmean(v))
            if all_probe_means and np.isfinite(v1_mean):
                ttfs_shift = v1_mean - np.mean(all_probe_means)

        probe_grand_means = {}
        for probe in probe_letters:
            sess_means = []
            for sess in sessions:
                sub = df_site[(df_site['probe_letter'] == probe) &
                              (df_site['session_num'] == sess)]
                v = _get(sub, mi)
                if len(v) < 5:
                    continue
                mu = np.nanmean(v) + ttfs_shift
                xp = probe_x[probe] + sess_jit[sess]
                ax.plot(xp, mu, 'o', ms=6.5,
                        color=probe_color[probe], alpha=0.75, zorder=4,
                        markeredgewidth=0.4, markeredgecolor='k')
                sess_means.append(mu)

            # Grand mean bar across sessions
            if sess_means:
                gm = np.mean(sess_means)
                sem = np.std(sess_means) / np.sqrt(len(sess_means))
                probe_grand_means[probe] = (probe_x[probe], gm)
                ax.hlines(gm, probe_x[probe] - 0.022, probe_x[probe] + 0.022,
                          colors=probe_color[probe], lw=3.0, zorder=5)
                ax.errorbar(probe_x[probe], gm, yerr=sem,
                            fmt='none', ecolor=probe_color[probe], lw=1.5, zorder=5,
                            capsize=3, capthick=1.5)

        # A regression is meaningful only for the historical pseudo-hierarchy
        # geometry. Display-only categorical offsets have no numeric scale.
        if (
            args.within_v1_x_mode == 'legacy_pseudo_hierarchy'
            and len(probe_grand_means) >= 3
        ):
            px = np.array([probe_grand_means[p][0] for p in probe_letters
                           if p in probe_grand_means])
            py = np.array([probe_grand_means[p][1] for p in probe_letters
                           if p in probe_grand_means])
            slope_p, intercept_p, *_ = linregress(px, py)
            xp2 = np.linspace(_gap_l - 0.03, _gap_r + 0.03, 50)
            ax.plot(xp2, slope_p * xp2 + intercept_p,
                    '-', color='#555', alpha=0.6, lw=1.5, zorder=3)

    ax.set_ylabel(_labels[mi], fontsize=10)
    [ax.spines[s].set_visible(False) for s in ('top', 'right')]

# ── X-axis labels ─────────────────────────────────────────────────────────────
common_names = ['LGN', 'V1', 'LM', 'RL', 'LP', 'AL', 'PM', 'AM']
ax = axes[-1]
if args.within_v1_x_mode == 'display_only':
    tick_x = [_hs[a] for a in orig_areas]
    tick_lab = common_names
else:
    tick_x = [_hs[a] for a in orig_areas] + [probe_x[p] for p in probe_letters]
    tick_lab = common_names + [f'V1_{p}' for p in probe_letters]
ax.set_xticks(tick_x)
ax.set_xticklabels(tick_lab, rotation=40, ha='right', fontsize=9)
ax.set_xlim(-0.75, 0.60)
if args.within_v1_x_mode == 'display_only':
    ax.set_xlabel(
        'Published inter-area hierarchy score '
        '(MouseV2 probe offsets at VISp are display-only)',
        fontsize=9,
    )
else:
    ax.set_xlabel('← Hierarchy score →   (probe groups placed near VISp)',
                  fontsize=9)

# Region annotation above top panel
_position_note = (
    'within-V1 probes (display-only x offsets; no hierarchy scores)'
    if args.within_v1_x_mode == 'display_only'
    else 'within-V1 probes (8 sessions)'
)
axes[0].annotate(_position_note,
                 xy=(np.mean([_gap_l, _gap_r]), axes[0].get_ylim()[1]),
                 fontsize=8.5, ha='center', va='bottom', color='#666',
                 xytext=(0, 4), textcoords='offset points')
if args.ttfs_display == 'raw_nwb':
    axes[0].text(
        0.01,
        0.03,
        '* MouseV2 raw relative to NWB start_time; physical light onset uncalibrated',
        transform=axes[0].transAxes,
        fontsize=7.5,
        color='#777',
        style='italic',
        ha='left',
        va='bottom',
    )

# ── Legend ────────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
probe_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=probe_color[p],
           markeredgecolor='k', markeredgewidth=0.4, ms=8, label=f'Probe {p}')
    for p in probe_letters
]
probe_handles.append(
    Line2D([0], [0], color='gray', lw=3.0, label='Probe mean ± SEM\n(across sessions)')
)
axes[0].legend(handles=probe_handles, loc='upper right', fontsize=8,
               framealpha=0.75, title='V1 multi-site sessions', title_fontsize=8)

# ── Save ──────────────────────────────────────────────────────────────────────
output_dir.mkdir(parents=True, exist_ok=True)
out = output_dir / 'Figure3_probe_zoom.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
