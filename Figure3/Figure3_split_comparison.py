"""
Figure3/Figure3_split_comparison.py

Split comparison figure:
  Col 0  — V1 probe scatter (B, C, A, E; 8 session means each)
  Col 1  — Rotated KDE of all V1 probe unit values (pooled)
  Col 2  — LM→AM hierarchy scatter (post-V1 areas only)
  Col 3  — Rotated KDE of all LM→AM unit values (pooled)

Same y-axis tick increment across all 4 columns per row.
sharey between col 0–1 and between col 2–3.
"""

import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress, gaussian_kde
import glob
import re
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(42)

code_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(code_directory, 'data')

# ── Load original data ────────────────────────────────────────────────────────
df_orig = pd.read_csv(os.path.join(data_dir, 'unit_table.csv'), low_memory=False)
fine_to_coarse = {'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL',
                  'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'}
df_orig['area_coarse'] = df_orig['ecephys_structure_acronym'].map(
    lambda a: fine_to_coarse.get(a, a))

# ── Load all site data ────────────────────────────────────────────────────────
APPLY_QC = True   # set False to revert to unfiltered units

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
    df = df.rename(columns={
        'time_to_first_spike': 'time_to_first_spike_fl',
        'modulation_index':    'f1_f0_dg',
        'autocorr_tau':        'timescale_ac',
    })
    qc_path = os.path.join(site_dir, 'unit_quality.csv')
    if APPLY_QC and os.path.isfile(qc_path):
        qc = pd.read_csv(qc_path)[['unit_id', 'default_qc']]
        df = df.merge(qc, on='unit_id', how='left')
        df = df[df['default_qc'] == True].drop(columns=['default_qc'])
    return df

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

# ── Colours ───────────────────────────────────────────────────────────────────
def orig_color(area):
    _pal = {'LGd': (217,141,194), 'V1': (129,116,177), 'LM': (78,115,174),
            'RL': (101,178,201), 'LP': (88,167,106), 'AL': (202,183,120),
            'PM': (219,132,87),  'AM': (194,79,84)}
    return tuple(v/255 for v in _pal.get(area, (180,180,180)))

probe_color = {'A': '#d73027', 'B': '#4575b4', 'C': '#1a9850', 'E': '#8073ac'}
V1_PURPLE   = (129/255, 116/255, 177/255)

# ── Areas ─────────────────────────────────────────────────────────────────────
post_v1_areas  = ('LM', 'RL', 'LP', 'AL', 'PM', 'AM')
post_v1_labels = ['LM', 'RL', 'LP', 'AL', 'PM', 'AM']
_hs = {'LGd':-0.515, 'V1':-0.357, 'LM':-0.093, 'RL':-0.059,
       'LP':0.105, 'AL':0.152, 'PM':0.327, 'AM':0.441}

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics    = ['time_to_first_spike_fl', 'f1_f0_dg', 'timescale_ac']
_row_labels = ['TTFS (ms)', '$\\log_{10}$ F1/F0', 'Response timescale (ms)']
_fns        = [lambda v: v * 1000,
               lambda v: np.log10(np.clip(v, 1e-6, None)),
               lambda v: v]
_tick_inc   = [5, 0.1, 10]

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
    metric = _metrics[mi]
    if metric not in df_sub.columns:
        return np.array([])
    sel = _filt(df_sub, mi)
    v = _fns[mi](df_sub.loc[sel, metric].astype(float).values)
    return v[np.isfinite(v)]

def _ci(v, N=500):
    if len(v) < 5: return np.nan
    n = max(1, len(v) // 2)
    est = [np.nanmean(v[np.random.permutation(len(v))[:n]]) for _ in range(N)]
    return np.percentile(est, 97.5) - np.nanmean(est)

sessions    = sorted(df_site['session_num'].unique()) if not df_site.empty else []
probe_order = ['B', 'C', 'A', 'E']
probe_xi    = {p: i for i, p in enumerate(probe_order)}
n_sess      = len(sessions)
_jit        = np.linspace(-0.18, 0.18, n_sess) if n_sess > 1 else np.array([0.0])
sess_jit    = {s: _jit[i] for i, s in enumerate(sessions)}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 4, figsize=(16, 10),
    gridspec_kw={
        'width_ratios': [1.2, 0.4, 1.5, 0.4],
        'wspace': 0.08,
        'hspace': 0.40,
    }
)

# Share y-axes: col 0 ↔ col 1  and  col 2 ↔ col 3
for mi in range(3):
    axes[mi, 1].sharey(axes[mi, 0])
    axes[mi, 3].sharey(axes[mi, 2])

for mi in range(3):
    ax_s   = axes[mi, 0]   # V1 probe scatter
    ax_kl  = axes[mi, 1]   # KDE for V1 probes
    ax_r   = axes[mi, 2]   # LM→AM scatter
    ax_kr  = axes[mi, 3]   # KDE for LM→AM

    inc = _tick_inc[mi]

    # ── Col 0: V1 probe scatter ───────────────────────────────────────────────
    all_probe_means = []

    for probe in probe_order:
        xi = float(probe_xi[probe])
        col = probe_color[probe]
        sess_means = []

        # pool all units for this probe across sessions
        all_probe_units = _get(
            df_site[df_site['probe_letter'] == probe], mi)

        if len(all_probe_units) >= 10:
            p10, q25, q75, p90 = [float(p) for p in
                                   np.percentile(all_probe_units, [10, 25, 75, 90])]
            hw = 0.28  # half-width of IQR rectangle
            # P10–P90 whisker (thin line)
            ax_s.plot([xi, xi], [p10, p90], color=col, lw=1.2, zorder=1, alpha=0.5)
            # IQR rectangle (translucent fill + border)
            ax_s.fill_between([xi - hw, xi + hw], q25, q75,
                              color=col, alpha=0.18, zorder=1)
            ax_s.plot([xi - hw, xi + hw, xi + hw, xi - hw, xi - hw],
                      [q25, q25, q75, q75, q25],
                      color=col, lw=0.8, alpha=0.45, zorder=1)

        for sess in sessions:
            sub = df_site[(df_site['probe_letter'] == probe) &
                          (df_site['session_num']  == sess)]
            v = _get(sub, mi)
            if len(v) < 5:
                continue
            mu = float(np.nanmean(v))
            xp = xi + sess_jit[sess]
            ax_s.plot(xp, mu, 'o', ms=6.5, color=col,
                      alpha=0.72, zorder=4,
                      markeredgewidth=0.35, markeredgecolor='k')
            sess_means.append(mu)
            all_probe_means.append(mu)

        if sess_means:
            gm  = np.mean(sess_means)
            sem = np.std(sess_means) / np.sqrt(len(sess_means))
            ax_s.hlines(gm, xi - 0.30, xi + 0.30,
                        colors=col, lw=3.0, zorder=5)
            ax_s.errorbar(xi, gm, yerr=sem, fmt='none',
                          ecolor=col, lw=1.5,
                          capsize=3, capthick=1.5, zorder=5)

    ax_s.set_xticks(range(len(probe_order)))
    ax_s.set_xticklabels(probe_order, fontsize=10)
    ax_s.set_xlim(-0.65, len(probe_order) - 0.35)
    ax_s.set_ylabel(_row_labels[mi], fontsize=10)
    if mi == 2:
        ax_s.set_xlabel('Probe', fontsize=10)
    [ax_s.spines[s].set_visible(False) for s in ('top', 'right')]
    ax_s.yaxis.set_major_locator(MultipleLocator(inc))
    if mi == 0:
        ax_s.set_title('Within-V1 (8 sessions)', fontsize=10, pad=6)

    # ── Col 1: KDE of pooled V1 probe units (rotated) ────────────────────────
    v1_units = _get(df_site, mi)

    if len(v1_units) > 20:
        lo = np.percentile(v1_units, 1)
        hi = np.percentile(v1_units, 99)
        y_grid = np.linspace(lo - (hi-lo)*0.15, hi + (hi-lo)*0.15, 400)
        bw = max(0.2, (hi - lo) / 20)   # data-adaptive bandwidth
        kde = gaussian_kde(v1_units, bw_method=bw / np.std(v1_units))
        dens = kde(y_grid)
        dens /= dens.max()
        ax_kl.fill_betweenx(y_grid, 0, dens, alpha=0.55, color=V1_PURPLE)
        ax_kl.plot(dens, y_grid, lw=1.2, color=V1_PURPLE)
        # IQR annotation
        q1, q3 = np.percentile(v1_units, [25, 75])
        ax_kl.hlines([q1, q3], 0, 0.3, colors=V1_PURPLE, lw=1.5, ls='--')
        ax_kl.text(0.35, (q1+q3)/2,
                   f'IQR\n{q3-q1:.2g}', fontsize=6.5, color=V1_PURPLE,
                   va='center', ha='left')

    ax_kl.set_xlim(0, 1.35)
    ax_kl.set_xticks([])
    ax_kl.set_xlabel('density', fontsize=8)
    [ax_kl.spines[s].set_visible(False) for s in ('top', 'right', 'bottom')]
    plt.setp(ax_kl.get_yticklabels(), visible=False)
    if mi == 0:
        ax_kl.set_title('dist.', fontsize=9, pad=6, color='#555')

    # Tight y-limits on left pair
    if all_probe_means and len(v1_units) > 0:
        lo = min(np.percentile(v1_units, 1), min(all_probe_means))
        hi = max(np.percentile(v1_units, 99), max(all_probe_means))
        pad = 1.5 * inc
        ax_s.set_ylim(np.floor(lo / inc) * inc - pad,
                      np.ceil(hi  / inc) * inc + pad)

    # ── Col 2: LM→AM scatter ─────────────────────────────────────────────────
    orig_xs, orig_ys = [], []
    for area in post_v1_areas:
        v = _get(df_orig[df_orig['area_coarse'] == area], mi)
        if len(v) < 5:
            continue
        mu   = float(np.nanmean(v))
        q10, q25, q75, q90 = [float(p) for p in np.percentile(v, [10, 25, 75, 90])]
        x    = float(_hs[area])
        col  = orig_color(area)
        # thin whisker: 10th–90th percentile
        ax_r.plot([x, x], [q10, q90], color=col, lw=1.2, zorder=2)
        # thick bar: IQR (25th–75th)
        ax_r.plot([x, x], [q25, q75], color=col, lw=4.0, zorder=3)
        # mean dot on top
        ax_r.plot(x, mu, 'o', ms=7, color='white', zorder=4,
                  markeredgecolor=col, markeredgewidth=1.5)
        orig_xs.append(x)
        orig_ys.append(mu)

    if len(orig_xs) >= 3:
        slope, intercept, r, p, _ = linregress(orig_xs, orig_ys)
        x2 = np.linspace(min(orig_xs)-0.05, max(orig_xs)+0.05, 100)
        ax_r.plot(x2, slope*x2 + intercept, '--k', alpha=0.30, lw=1.2, zorder=1)
        ax_r.text(max(orig_xs) + 0.02, slope*max(orig_xs) + intercept,
                  f'r={r:.2f}\np={p:.3f}', fontsize=7, va='center', color='#555')

    ax_r.set_xticks([_hs[a] for a in post_v1_areas])
    ax_r.set_xticklabels(post_v1_labels, rotation=35, ha='right', fontsize=9)
    ax_r.set_xlim(min([_hs[a] for a in post_v1_areas]) - 0.08,
                  max([_hs[a] for a in post_v1_areas]) + 0.08)
    ax_r.yaxis.set_major_locator(MultipleLocator(inc))
    ax_r.set_ylabel(_row_labels[mi], fontsize=10)
    [ax_r.spines[s].set_visible(False) for s in ('top', 'right')]
    if mi == 2:
        ax_r.set_xlabel('Hierarchy score', fontsize=10)
    if mi == 0:
        ax_r.set_title('Post-V1 areas (Allen SDK)', fontsize=10, pad=6)

    # ── Col 3: KDE of pooled LM→AM unit values (rotated) ─────────────────────
    lmam_df   = df_orig[df_orig['area_coarse'].isin(post_v1_areas)]
    lmam_units = _get(lmam_df, mi)
    lmam_color = '#888'

    if len(lmam_units) > 20:
        lo = np.percentile(lmam_units, 1)
        hi = np.percentile(lmam_units, 99)
        y_grid = np.linspace(lo - (hi-lo)*0.15, hi + (hi-lo)*0.15, 400)
        bw = max(0.2, (hi - lo) / 20)
        kde = gaussian_kde(lmam_units, bw_method=bw / np.std(lmam_units))
        dens = kde(y_grid)
        dens /= dens.max()
        ax_kr.fill_betweenx(y_grid, 0, dens, alpha=0.45, color=lmam_color)
        ax_kr.plot(dens, y_grid, lw=1.2, color=lmam_color)
        q1, q3 = np.percentile(lmam_units, [25, 75])
        ax_kr.hlines([q1, q3], 0, 0.3, colors=lmam_color, lw=1.5, ls='--')
        ax_kr.text(0.35, (q1+q3)/2,
                   f'IQR\n{q3-q1:.2g}', fontsize=6.5, color='#555',
                   va='center', ha='left')

    ax_kr.set_xlim(0, 1.35)
    ax_kr.set_xticks([])
    ax_kr.set_xlabel('density', fontsize=8)
    [ax_kr.spines[s].set_visible(False) for s in ('top', 'right', 'bottom')]
    plt.setp(ax_kr.get_yticklabels(), visible=False)
    if mi == 0:
        ax_kr.set_title('dist.', fontsize=9, pad=6, color='#555')

    # Tight y-limits on right pair
    if orig_ys and len(lmam_units) > 0:
        lo = min(np.percentile(lmam_units, 1), min(orig_ys))
        hi = max(np.percentile(lmam_units, 99), max(orig_ys))
        pad = 1.5 * inc
        ax_r.set_ylim(np.floor(lo / inc) * inc - pad,
                      np.ceil(hi  / inc) * inc + pad)

# ── TTFS offset note ──────────────────────────────────────────────────────────
axes[0, 0].text(0.02, 0.03,
    '* ~10 ms display-timing offset\n  relative to original (see other figure)',
    transform=axes[0, 0].transAxes, fontsize=6.5, color='#888',
    style='italic', va='bottom')

# ── Probe legend ──────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=probe_color[p], markeredgecolor='k',
                  markeredgewidth=0.4, ms=8, label=f'Probe {p}')
           for p in probe_order]
handles.append(Line2D([0],[0], color='gray', lw=3.0, label='Mean ± SEM'))
axes[0, 0].legend(handles=handles, fontsize=7, loc='upper right',
                  framealpha=0.75, title='V1 sessions', title_fontsize=7)

# KDE legend on top-right
axes[0, 1].text(0.5, 1.01, 'V1\nprobes', transform=axes[0,1].transAxes,
                ha='center', va='bottom', fontsize=8, color=V1_PURPLE)
axes[0, 3].text(0.5, 1.01, 'LM→AM\nunits', transform=axes[0,3].transAxes,
                ha='center', va='bottom', fontsize=8, color='#555')

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(code_directory, 'Figure3', 'Figure3_split_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
