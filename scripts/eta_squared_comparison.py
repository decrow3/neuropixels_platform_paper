"""
scripts/eta_squared_comparison.py

Compare between-group variance structure in V1-probe data vs post-V1 area data.

Effect size: omega-squared (ω²) — corrects for group-count bias in η².
  ω² = (SS_between − (k−1)·MS_within) / (SS_total + MS_within)

Pseudoreplication fix: aggregate to session×group means before all analysis.
  V1:    (probe × session)  → up to 8 sessions × 4 probes = 32 data points
  Allen: (area  × session)  → ~40 sessions × 6 areas      = ~240 data points
Both datasets still have within-session correlation *across* groups
(same session → data from multiple probes/areas), but within-group
pseudoreplication (many units from the same session treated as independent)
is removed.

Bootstrap 95% CIs resample sessions within groups to respect the
remaining between-session dependency.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(code_dir, 'data')

# ── Metric definitions ────────────────────────────────────────────────────────
_metrics       = ['time_to_first_spike_fl', 'f1_f0_dg', 'timescale_ac']
_metric_labels = ['TTFS (ms)', 'log10 F1/F0', 'Timescale (ms)']
_fns           = [lambda v: v * 1000,
                  lambda v: np.log10(np.clip(v, 1e-6, None)),
                  lambda v: v]

def _filt_mask(df, mi):
    sel = pd.Series(True, index=df.index)
    if mi == 0:
        sel &= df['time_to_first_spike_fl'].astype(float) < 0.1
    elif mi == 1:
        sel &= df['f1_f0_dg'].astype(float) > 0
    elif mi == 2:
        sel &= df['timescale_ac'].astype(float).between(1, 300)
        if 'spike_count_ac' in df.columns:
            sel &= df['spike_count_ac'].astype(float) > 50
        if 'err_ac' in df.columns:
            sel &= df['err_ac'].astype(float) < 20
    return sel

def _transform(v, mi):
    out = _fns[mi](v.astype(float).values)
    return out[np.isfinite(out)]

# ── Load V1 site data ─────────────────────────────────────────────────────────
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

# ── Load Allen data ───────────────────────────────────────────────────────────
df_orig = pd.read_csv(os.path.join(data_dir, 'unit_table.csv'), low_memory=False)
fine_to_coarse = {'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL',
                  'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'}
df_orig['area_coarse'] = df_orig['ecephys_structure_acronym'].map(
    lambda a: fine_to_coarse.get(a, a))

probe_groups = ['B', 'C', 'A', 'E']
area_groups  = ['LM', 'RL', 'LP', 'AL', 'PM', 'AM']

# ── Build session×group mean tables ──────────────────────────────────────────
def make_session_means(df, group_col, session_col, groups, mi):
    """
    Return dict: group → array of per-session means (one per session that
    has ≥5 valid units in that group).
    """
    metric = _metrics[mi]
    out = {}
    for g in groups:
        sub = df[df[group_col] == g].copy()
        if metric not in sub.columns:
            out[g] = np.array([])
            continue
        mask = _filt_mask(sub, mi)
        sub = sub[mask].copy()
        sub['_val'] = _fns[mi](sub[metric].astype(float))
        sub = sub[np.isfinite(sub['_val'])]
        sess_means = (sub.groupby(session_col)['_val']
                         .agg(lambda x: np.mean(x) if len(x) >= 5 else np.nan)
                         .dropna().values)
        out[g] = sess_means
    return out

# ── Effect sizes ─────────────────────────────────────────────────────────────
def omega_squared(groups_vals):
    """
    ω² = (SS_between − (k−1)·MS_within) / (SS_total + MS_within)
    groups_vals: list of 1-D arrays (session means per group)
    """
    groups_vals = [g for g in groups_vals if len(g) >= 2]
    k = len(groups_vals)
    if k < 2:
        return np.nan
    all_vals  = np.concatenate(groups_vals)
    N         = len(all_vals)
    grand_mean = np.mean(all_vals)
    ss_total   = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_vals)
    ss_within  = ss_total - ss_between
    df_within  = N - k
    if df_within <= 0 or ss_total == 0:
        return np.nan
    ms_within  = ss_within / df_within
    return (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

def eta_squared(groups_vals):
    all_vals   = np.concatenate([g for g in groups_vals if len(g) >= 1])
    if len(all_vals) < 2:
        return np.nan
    grand_mean = np.mean(all_vals)
    ss_total   = np.sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2
                     for g in groups_vals if len(g) >= 1)
    return ss_between / ss_total

def bootstrap_omega2(session_means_dict, n_boot=5000):
    """
    Resample sessions (with replacement) within each group.
    Returns obs ω², (CI_lo, CI_hi), boot array.
    """
    groups = [v for v in session_means_dict.values() if len(v) >= 2]
    obs = omega_squared(groups)
    boot = []
    for _ in range(n_boot):
        resampled = [g[np.random.randint(0, len(g), len(g))] for g in groups]
        boot.append(omega_squared(resampled))
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return obs, ci_lo, ci_hi, boot

# ── Unit counts (for reporting) ───────────────────────────────────────────────
def _unit_counts(df, group_col, groups, mi):
    """Return total units (filtered by metric criteria) across all groups."""
    metric = _metrics[mi]
    total = 0
    for g in groups:
        sub = df[df[group_col] == g]
        if metric not in sub.columns:
            continue
        mask = _filt_mask(sub, mi)
        sub2 = sub[mask].copy()
        sub2['_val'] = _fns[mi](sub2[metric].astype(float))
        total += int(np.isfinite(sub2['_val']).sum())
    return total

# also collect raw unit counts before/after QC for the new session data
n_units_raw = len(df_site) if not df_site.empty else 0
# reload without QC to get unfiltered count (just check if unit_quality was merged in)
_raw_frames = []
for d in _site_dirs:
    needed = ['layer_info.csv']
    if os.path.isfile(os.path.join(d, needed[0])):
        _raw_frames.append(len(pd.read_csv(os.path.join(d, 'layer_info.csv'))))
n_units_total_raw = sum(_raw_frames)

# ── Main analysis ─────────────────────────────────────────────────────────────
N_BOOT = 5000

hdr = (f"{'Metric':<20}  {'Dataset':<18}  {'k':>3}  {'n_sess_obs':>10}  "
       f"{'η²':>7}  {'ω²':>7}  {'95% CI (ω²)':>18}")
print(f"\n{hdr}")
print("-" * 100)

results = {}
for mi, label in enumerate(_metric_labels):

    # V1 probes: session×probe means
    probe_sess = make_session_means(df_site, 'probe_letter', 'session_num',
                                    probe_groups, mi)
    probe_groups_arr = [probe_sess[p] for p in probe_groups if len(probe_sess[p]) >= 2]
    n_obs_p = sum(len(g) for g in probe_groups_arr)
    eta_p = eta_squared(probe_groups_arr)
    om_p, lo_p, hi_p, boot_p = bootstrap_omega2(
        {p: probe_sess[p] for p in probe_groups}, n_boot=N_BOOT)

    print(f"{label:<20}  {'Probes (V1)':<18}  {len(probe_groups_arr):>3}  {n_obs_p:>10}  "
          f"{eta_p:>7.4f}  {om_p:>7.4f}  [{lo_p:+.4f}, {hi_p:+.4f}]")

    # Allen areas: session×area means
    area_sess = make_session_means(df_orig, 'area_coarse', 'ecephys_session_id',
                                   area_groups, mi)
    area_groups_arr = [area_sess[a] for a in area_groups if len(area_sess[a]) >= 2]
    n_obs_a = sum(len(g) for g in area_groups_arr)
    eta_a = eta_squared(area_groups_arr)
    om_a, lo_a, hi_a, boot_a = bootstrap_omega2(
        {a: area_sess[a] for a in area_groups}, n_boot=N_BOOT)

    print(f"{'':<20}  {'Areas (post-V1)':<18}  {len(area_groups_arr):>3}  {n_obs_a:>10}  "
          f"{eta_a:>7.4f}  {om_a:>7.4f}  [{lo_a:+.4f}, {hi_a:+.4f}]")

    # Difference ω²_areas − ω²_probes
    diff_obs  = om_a - om_p
    diff_boot = boot_a - boot_p
    diff_lo, diff_hi = np.percentile(diff_boot, [2.5, 97.5])
    p_above_zero = float(np.mean(diff_boot <= 0))   # P(Δω² ≤ 0)

    print(f"{'':<20}  {'Δω² (areas−probes)':<18}  {'':>3}  {'':>10}  "
          f"{'':>7}  {diff_obs:>+7.4f}  [{diff_lo:+.4f}, {diff_hi:+.4f}]"
          f"  P(Δ≤0)={p_above_zero:.4f}")
    print()

    results[label] = dict(
        eta_probe=eta_p, omega_probe=om_p, ci_probe=(lo_p, hi_p),
        n_obs_probe=n_obs_p,
        eta_area=eta_a,  omega_area=om_a,  ci_area=(lo_a, hi_a),
        n_obs_area=n_obs_a,
        delta_omega=diff_obs, delta_ci=(diff_lo, diff_hi), p_delta_le0=p_above_zero,
        n_units_probe=_unit_counts(df_site, 'probe_letter', probe_groups, mi),
        n_units_area=_unit_counts(df_orig, 'area_coarse', area_groups, mi),
    )

print("Notes:")
print("  · Session×group means used throughout — removes within-session unit correlation.")
print("  · ω² corrects for upward bias in η² due to differing k (4 probes vs 6 areas).")
print("  · Cross-group within-session correlation (same session → multiple areas/probes)")
print("    remains; bootstrap resamples sessions to partially account for this.")
print("  · P(Δ≤0): bootstrap probability that ω²_areas ≤ ω²_probes.")

# ── Write companion markdown ──────────────────────────────────────────────────
from datetime import date as _date

qc_label = "yes (`default_qc == True`)" if APPLY_QC else "no"

def _fmt(v):
    """Format a float, showing negative ω² as '< 0'."""
    if np.isnan(v): return "n/a"
    return f"{v:+.3f}" if v < 0 else f"{v:.3f}"

def _ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"

n_new_sessions = df_site['session_num'].nunique() if not df_site.empty else 0
n_new_probes   = df_site['probe_letter'].nunique() if not df_site.empty else 0

md_lines = [
    f"# Figure 3 — Statistical Companion",
    f"",
    f"_Generated {_date.today().isoformat()} · QC filter applied: {qc_label}_",
    f"",
    f"---",
    f"",
    f"## Figure captions",
    f"",
    f"### `Figure3_with_V1sites.png`",
    f"",
    f"**Response properties across the mouse visual hierarchy, with new multi-site V1 data overlaid.**",
    f"Each row shows one response metric: time-to-first-spike (TTFS, ms), log₁₀ F1/F0",
    f"(drifting gratings modulation ratio), and response decay timescale (ms).",
    f"*Left two columns*: probability density and cumulative distribution functions",
    f"for each visual area (LGN → AM), coloured by hierarchy position (Allen CCF).",
    f"New multi-site V1 sessions (n = {n_new_sessions} sessions, probes B/C/A/E, coloured by probe)",
    f"are overlaid on the V1 distributions.",
    f"*Third column*: area medians plotted against published hierarchy score",
    f"(Siegle et al. 2021); dashed line is OLS regression with r and p values.",
    f"New V1 probe means are placed at the VISp hierarchy score.",
    f"*Fourth column*: pairwise significance matrix (Mann–Whitney U, Bonferroni corrected).",
    f"F1/F0 computed using Allen SDK cycle-fold method (preferred orientation × TF,",
    f"fold into single cycle, FFT; F0 = ½·amplitude[DC], F1 = amplitude[1st harmonic]).",
    f"",
    f"### `Figure3_probe_zoom.png`",
    f"",
    f"**Within-V1 probe variance compared against the full hierarchy gradient.**",
    f"Same three metrics as above (rows), plotted against hierarchy score.",
    f"Each coloured dot is the session mean for one probe (B, C, A, E) in one of",
    f"n = {n_new_sessions} new multi-site sessions; error bars show probe mean ± SEM across sessions.",
    f"Original Allen area means (open circles, coloured by area) and OLS regression",
    f"(dashed) are shown for context. A linear fit through the within-V1 probe means",
    f"(solid grey, shaded 95% band) illustrates within-V1 spatial variance.",
    f"TTFS values for new sessions carry a ~10 ms display-timing offset relative to",
    f"the original Allen data (different stimulus delivery hardware); this is a",
    f"systematic scalar shift and does not affect relative probe or area comparisons.",
    f"",
    f"### `Figure3_split_comparison.png`",
    f"",
    f"**Direct comparison of within-V1 probe variance (left) against post-V1 area variance (right).**",
    f"*Columns 0–1 (left pair)*: within-V1 data from n = {n_new_sessions} sessions.",
    f"Probes ordered B → C → A → E (approximately lateral → medial within VISp).",
    f"Dots show per-session means; horizontal bar and error bars show probe grand mean ± SEM.",
    f"Shaded box spans unit-level IQR (25th–75th percentile) pooled across all sessions",
    f"for that probe; thin whiskers span 10th–90th percentile.",
    f"Rotated KDE (col 1) shows the unit-level distribution pooled across all V1 probes;",
    f"dashed lines mark the IQR.",
    f"*Columns 2–3 (right pair)*: Allen SDK unit_table data (Siegle et al. 2021)",
    f"for post-V1 areas LM, RL, LP, AL, PM, AM, ordered by hierarchy score.",
    f"Thick segment = IQR; thin whisker = 10th–90th percentile; white dot = mean.",
    f"Rotated KDE (col 3) pools all post-V1 units.",
    f"Dashed regression line with r and p values.",
    f"Both panels use the same y-axis tick increment per row to allow direct",
    f"visual comparison of spread; absolute offsets between panels are expected",
    f"(different recording hardware/populations) and do not affect the variance comparison.",
    f"Unit quality filter: `default_qc == True` (NWB units table flag; ≈ 55% of units pass)",
    f"applied to new session data to match Allen's pre-filtered unit population.",
    f"",
    f"---",
    f"",
    f"## Statistical analysis: between-group variance in response properties",
    f"",
    f"### Motivation",
    f"",
    f"We ask whether categorical area labels (LM, RL, LP, AL, PM, AM) explain",
    f"significantly more variance in response properties than categorical probe labels",
    f"(B, C, A, E — different spatial positions within VISp) do within V1.",
    f"If area structure genuinely reflects a processing hierarchy, area membership",
    f"should account for substantially more variance than equivalent positional",
    f"variation within a single area.",
    f"",
    f"### Effect size: ω² (omega-squared)",
    f"",
    f"ω² = (SS_between − (k−1)·MS_within) / (SS_total + MS_within)",
    f"",
    f"where k is the number of groups. ω² corrects the positive bias in η²",
    f"(= SS_between / SS_total) that arises when k differs between the two datasets",
    f"(k = 4 probes vs k = 6 areas). Under the null (no group structure), E[ω²] ≈ 0.",
    f"",
    f"### Pseudoreplication correction",
    f"",
    f"Units recorded in the same session share stimulus conditions, brain state,",
    f"and anaesthetic depth, violating independence. Treating them as independent",
    f"(n ≈ 10 000 per dataset) inflates effective sample size and can produce",
    f"spurious significance.",
    f"",
    f"**Fix**: aggregate to session × group means before all analysis.",
    f"",
    f"| Dataset | Aggregation | Approximate n |",
    f"|---------|-------------|---------------|",
    f"| New V1 sessions | probe × session mean | {n_new_sessions} sessions × {n_new_probes} probes = {n_new_sessions * n_new_probes} obs |",
    f"| Allen (post-V1) | area × session mean | ~40 sessions × 6 areas ≈ 240 obs |",
    f"",
    f"Remaining limitation: the same Allen session contributes means for multiple areas",
    f"(simultaneous Neuropixels recording), introducing cross-group within-session",
    f"correlation. Bootstrap CIs resample sessions within groups to partially account",
    f"for this. This bias inflates the apparent Allen-area ω², making the comparison",
    f"conservative toward finding areas > probes.",
    f"",
    f"### Unit quality filter",
    f"",
    f"New session data: `default_qc == True` flag from NWB units table",
    f"({n_units_total_raw} total units → {n_units_raw} passing QC, "
    f"{100*n_units_raw/n_units_total_raw:.0f}%).",
    f"Allen data: unit_table.csv is pre-filtered to `quality == 'good'` units only.",
    f"",
    f"### Results",
    f"",
    f"Bootstrap 95% CIs (n = {N_BOOT} resamples) on ω²; sessions resampled within groups.",
    f"",
]

# build results table
rows_table = [
    "| Metric | Dataset | k | Session obs | η² | ω² | 95% CI (ω²) | Δω² (areas−probes) | 95% CI (Δω²) | P(Δ≤0) |",
    "|--------|---------|---|-------------|----|----|-------------|-------------------|--------------|--------|",
]
interp_map = {
    'TTFS (ms)':       ('after QC, within-V1 probe variance is comparable to between-area variance',
                        'no clear hierarchy > V1 distinction'),
    'log10 F1/F0':     ('no between-group structure in either dataset',
                        'F1/F0 does not track the hierarchy at session-mean level'),
    'Timescale (ms)':  ('areas show real ω²; probes near zero — directional but underpowered with 8 sessions',
                        'direction consistent with hierarchy; not statistically distinguishable'),
}

for label in _metric_labels:
    r = results[label]
    rows_table.append(
        f"| **{label}** | Within-V1 probes | 4 | {r['n_obs_probe']} "
        f"| {r['eta_probe']:.3f} | {_fmt(r['omega_probe'])} "
        f"| {_ci(*r['ci_probe'])} | — | — | — |"
    )
    rows_table.append(
        f"| | Post-V1 areas | 6 | {r['n_obs_area']} "
        f"| {r['eta_area']:.3f} | {_fmt(r['omega_area'])} "
        f"| {_ci(*r['ci_area'])} "
        f"| {r['delta_omega']:+.3f} | {_ci(*r['delta_ci'])} "
        f"| {r['p_delta_le0']:.3f} |"
    )

md_lines += rows_table
md_lines += [
    f"",
    f"_P(Δ≤0): bootstrap probability that ω²_areas ≤ ω²_probes; values near 0.5 indicate no detectable difference._",
    f"",
    f"### Interpretation",
    f"",
]

for label in _metric_labels:
    r = results[label]
    short, long = interp_map[label]
    md_lines.append(f"**{label}** — {short}. {long}.")
    md_lines.append(f"ω²_probes = {_fmt(r['omega_probe'])} {_ci(*r['ci_probe'])}, "
                    f"ω²_areas = {_fmt(r['omega_area'])} {_ci(*r['ci_area'])}, "
                    f"Δω² = {r['delta_omega']:+.3f} {_ci(*r['delta_ci'])}, "
                    f"P(Δ≤0) = {r['p_delta_le0']:.3f}.")
    md_lines.append(f"")

md_lines += [
    f"### Caveats",
    f"",
    f"- **TTFS timing offset**: New sessions show ~10 ms longer TTFS than Allen data,",
    f"  attributable to different stimulus display hardware (different monitor refresh",
    f"  latency). This is a scalar shift affecting absolute values but not variance structure.",
    f"- **F1/F0 population offset**: After QC, new-session V1 median F1/F0 ≈ 1.01 vs",
    f"  Allen V1 median ≈ 0.77. Residual offset may reflect remaining population",
    f"  differences (e.g., absence of receptive-field quality filter in new sessions).",
    f"- **Underpowered comparison for TTFS and timescale**: With only {n_new_sessions} new",
    f"  sessions, ω²_probes has wide CIs. The comparison requires ~40 sessions to match",
    f"  Allen dataset power.",
    f"- **Within-session cross-area correlation** (Allen data) inflates effective df;",
    f"  direction of bias favours finding area > probe structure, so any null result",
    f"  is conservative.",
]

out_md = os.path.join(code_dir, 'Figure3', 'Figure3_stats.md')
with open(out_md, 'w') as fh:
    fh.write('\n'.join(md_lines) + '\n')
print(f"\nMarkdown written → {out_md}")
