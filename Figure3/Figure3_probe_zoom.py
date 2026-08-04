"""
Figure3/Figure3_probe_zoom.py

Zoomed scatter: each metric vs. position, showing:
  - Original 8 areas (circles) at their hierarchy scores
  - New V1 sessions split by probe letter (A/B/C/E), each session as a dot,
    placed between V1 and LM on the x-axis
Purpose: compare within-V1 probe-to-probe variance to the inter-area gradient.
"""

import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
import glob
import re
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

code_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(code_directory, 'data')

# ── Load original data (Allen SDK metrics) ────────────────────────────────────
df_orig = pd.read_csv(os.path.join(data_dir, 'unit_table.csv'), low_memory=False)
fine_to_coarse = {'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL',
                  'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'}
df_orig['area_coarse'] = df_orig['ecephys_structure_acronym'].map(
    lambda a: fine_to_coarse.get(a, a))

# ── Load all MouseV2 site data ────────────────────────────────────────────────
def _load_one_site(site_dir):
    needed = ['layer_info.csv', 'change_modulation_data.csv',
              'timescale_metrics.csv', 'time_to_first_spike.csv']
    if not all(os.path.isfile(os.path.join(site_dir, f)) for f in needed):
        return pd.DataFrame()
    lay  = pd.read_csv(os.path.join(site_dir, 'layer_info.csv'))
    mod  = pd.read_csv(os.path.join(site_dir, 'change_modulation_data.csv'))
    ts   = pd.read_csv(os.path.join(site_dir, 'timescale_metrics.csv'))
    ttfs = pd.read_csv(os.path.join(site_dir, 'time_to_first_spike.csv'))
    df = lay.merge(mod, on='unit_id').merge(ts, on='unit_id').merge(ttfs, on='unit_id')
    return df.rename(columns={
        'time_to_first_spike': 'time_to_first_spike_fl',
        'modulation_index':    'f1_f0_dg',
        'autocorr_tau':        'timescale_ac',
    })

_site_dirs = sorted(glob.glob(os.path.join(data_dir, 'site*_processed')))
_frames = [_load_one_site(d) for d in _site_dirs]
_frames = [f for f in _frames if not f.empty]
df_site = pd.concat(_frames, ignore_index=True, sort=False) if _frames else pd.DataFrame()

if not df_site.empty:
    def _parse(acronym):
        m = re.match(r'V1_site(\d+)_([ABCE])', str(acronym))
        return (int(m.group(1)), m.group(2)) if m else (None, None)
    parsed = [_parse(a) for a in df_site['ecephys_structure_acronym']]
    df_site['session_num']  = [x[0] for x in parsed]
    df_site['probe_letter'] = [x[1] for x in parsed]
    df_site = df_site[df_site['probe_letter'].notna()].copy()
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

# Probe groups inserted in the gap between V1 (-0.357) and LM (-0.093)
# Order: B, C, A, E (retinotopic/anatomical order)
probe_letters = ('B', 'C', 'A', 'E')
_gap_l, _gap_r = -0.32, -0.12
probe_x = dict(zip(probe_letters, np.linspace(_gap_l, _gap_r, 4)))

sessions = sorted(df_site['session_num'].unique()) if not df_site.empty else []
n_sess = len(sessions)
_jitter = np.linspace(-0.018, 0.018, n_sess) if n_sess > 1 else np.array([0.0])
sess_jit = {s: _jitter[i] for i, s in enumerate(sessions)}

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics = ['time_to_first_spike_fl', 'f1_f0_dg', 'timescale_ac']
_labels  = ['Time to first spike (ms)',
            '$\\log_{10}$ F1/F0',
            'Response decay timescale (ms)']
_fns     = [lambda v: v * 1000,
            lambda v: np.log10(np.clip(v, 1e-6, None)),
            lambda v: v]

def _filt(df_sub, mi):
    sel = pd.Series(True, index=df_sub.index)
    if mi == 0:
        sel &= df_sub['time_to_first_spike_fl'].astype(float) < 0.1
    elif mi == 1:
        sel &= df_sub['f1_f0_dg'].astype(float) > 0
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
        if mi == 0:
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

        # Regression line through the 4 probe grand means
        if len(probe_grand_means) >= 3:
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
tick_x   = [_hs[a] for a in orig_areas] + [probe_x[p] for p in probe_letters]
tick_lab = common_names + [f'V1_{p}' for p in probe_letters]
ax.set_xticks(tick_x)
ax.set_xticklabels(tick_lab, rotation=40, ha='right', fontsize=9)
ax.set_xlim(-0.75, 0.60)
ax.set_xlabel('← Hierarchy score →   (probe groups placed near VISp)',
              fontsize=9)

# Region annotation above top panel
axes[0].annotate('within-V1 probes (8 sessions)',
                 xy=(np.mean([_gap_l, _gap_r]), axes[0].get_ylim()[1]),
                 fontsize=8.5, ha='center', va='bottom', color='#666',
                 xytext=(0, 4), textcoords='offset points')

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
out = os.path.join(code_directory, 'Figure3', 'Figure3_probe_zoom.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
