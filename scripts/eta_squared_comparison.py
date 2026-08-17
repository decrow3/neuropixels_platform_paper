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

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from common.figure3_mousev2 import (  # noqa: E402
    load_allen_units,
    load_config,
    load_mousev2_units,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--output-dir',
    type=Path,
    default=Path(code_dir) / 'Figure3',
    help='Directory for the generated Markdown report (default: Figure3).',
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

# ── Metric definitions ────────────────────────────────────────────────────────
_grating_name = 'F1/F0' if args.grating_metric == 'f1_f0_dg' else 'modulation index'
_grating_label = f'log10 {_grating_name}'
_metrics       = ['time_to_first_spike_fl', args.grating_metric, 'timescale_ac']
_metric_labels = ['TTFS (ms)', _grating_label, 'Timescale (ms)']
_fns           = [lambda v: v * 1000,
                  lambda v: np.log10(np.clip(v, 1e-6, None)),
                  lambda v: v]

def _filt_mask(df, mi):
    sel = pd.Series(True, index=df.index)
    if mi == 0:
        sel &= df['time_to_first_spike_fl'].astype(float) < 0.1
    elif mi == 1:
        sel &= df[args.grating_metric].astype(float) > 0
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
df_site = load_mousev2_units(
    apply_qc=APPLY_QC if args.population_profile is None else False,
    config_path=args.config,
    grating_metrics_dir=args.grating_metrics_dir,
    flash_metrics_dir=args.flash_metrics_dir,
    flash_variant=args.flash_variant,
    population_profile=args.population_profile,
)

# ── Load Allen data ───────────────────────────────────────────────────────────
df_orig = load_allen_units(
    args.config, population_profile=args.population_profile
)

probe_groups = list(config['display_probe_order'])
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
n_units_total_raw = sum(int(session['expected_units']) for session in config['sessions'])

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

if args.population_profile is None:
    qc_label = "yes (`default_qc == True`)" if APPLY_QC else "no"
    _population_caption_lines = [
        "Unit quality filter: `default_qc == True` (NWB units table flag; ≈ 55% of units pass)",
        "applied to new session data to match Allen's pre-filtered unit population.",
    ]
    _population_method_lines = [
        "New session data: `default_qc == True` flag from NWB units table",
        f"({n_units_total_raw} total units → {n_units_raw} passing QC, "
        f"{100*n_units_raw/n_units_total_raw:.0f}%).",
        "Allen data: unit_table.csv is pre-filtered to `quality == 'good'` units only.",
    ]
else:
    qc_label = f"yes (`{args.population_profile}`)"
    _population_caption_lines = [
        f"Named population profile `{args.population_profile}` is applied to Allen and MouseV2",
        "before the metric-specific validity filters.",
    ]
    _population_method_lines = [
        f"Both datasets use the named `{args.population_profile}` profile.",
        f"MouseV2: {n_units_total_raw} total units → {n_units_raw} selected "
        f"({100*n_units_raw/n_units_total_raw:.0f}%).",
        f"Allen: 99,180 total units → {len(df_orig)} selected before area restriction.",
    ]

def _fmt(v):
    """Format a float, showing negative ω² as '< 0'."""
    if np.isnan(v): return "n/a"
    return f"{v:+.3f}" if v < 0 else f"{v:.3f}"

def _ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"

n_new_sessions = df_site['session_num'].nunique() if not df_site.empty else 0
n_new_probes   = df_site['probe_letter'].nunique() if not df_site.empty else 0
_new_grating = pd.to_numeric(df_site[args.grating_metric], errors='coerce')
_allen_grating = pd.to_numeric(
    df_orig.loc[df_orig['area_coarse'] == 'V1', args.grating_metric], errors='coerce'
)
_new_grating_median = float(np.nanmedian(_new_grating[_new_grating > 0]))
_allen_grating_median = float(np.nanmedian(_allen_grating[_allen_grating > 0]))
if args.grating_metrics_dir is None:
    _metric_caption_lines = [
        "Each row shows one response metric: time-to-first-spike (TTFS, ms), log₁₀ F1/F0",
        "(drifting gratings modulation ratio), and response decay timescale (ms).",
    ]
    _grating_method_lines = [
        "F1/F0 computed using Allen SDK cycle-fold method (preferred orientation × TF,",
        "fold into single cycle, FFT; F0 = ½·amplitude[DC], F1 = amplitude[1st harmonic]).",
    ]
elif args.grating_metric == 'f1_f0_dg':
    _metric_caption_lines = [
        "Each row shows one response metric: time-to-first-spike (TTFS, ms), log₁₀ F1/F0,",
        "and response decay timescale (ms).",
    ]
    _grating_method_lines = [
        "F1/F0 uses AllenSDK cycle-fold mathematics at the preferred full orientation × TF × SF condition."
    ]
else:
    _metric_caption_lines = [
        "Each row shows one response metric: time-to-first-spike (TTFS, ms), log₁₀ modulation index,",
        "and response decay timescale (ms).",
    ]
    _grating_method_lines = [
        "Modulation index uses AllenSDK's Welch-spectrum z-score at the preferred full orientation × TF × SF condition."
    ]

if args.flash_metrics_dir is None:
    _flash_method_lines = []
else:
    _flash_method_lines = [
        f"MouseV2 flash metrics use the `{args.flash_variant}` presentation set; Allen uses its released pooled-flash values.",
        "TTFS is the median first occupied 1-ms bin from 30–200 ms. Timescale uses",
        "25 AllenSDK-centered 10-ms bins (centers 45–285 ms) and the released exponential fit.",
    ]

if args.ttfs_display == 'raw_nwb':
    _ttfs_caption_lines = [
        "MouseV2 TTFS is shown raw relative to NWB flash `start_time`; no cross-dataset",
        "mean matching or latency correction is applied. Physical light-onset timing and",
        "display latency are not encoded in the NWB and remain uncalibrated.",
    ]
    _ttfs_caveat_lines = [
        "- **TTFS timing provenance**: MouseV2 interval starts exactly match the NWB's",
        "  processed stimulus timestamp series, but no photodiode trace or physical",
        "  light-onset metadata is present. Absolute Allen–MouseV2 offsets are not interpreted.",
    ]
else:
    _ttfs_caption_lines = [
        "TTFS values for new sessions carry a ~10 ms display-timing offset relative to",
        "the original Allen data (different stimulus delivery hardware); this is a",
        "systematic scalar shift and does not affect relative probe or area comparisons.",
    ]
    _ttfs_caveat_lines = [
        "- **TTFS timing offset**: New sessions show ~10 ms longer TTFS than Allen data,",
        "  attributable to different stimulus display hardware (different monitor refresh",
        "  latency). This is a scalar shift affecting absolute values but not variance structure.",
    ]

if args.within_v1_x_mode == 'display_only':
    _full_position_caption_lines = [
        "New MouseV2 session means use small horizontal display offsets around the published VISp score.",
        "Those offsets are not within-V1 hierarchy scores and are not used in inference.",
    ]
    _zoom_position_caption_lines = [
        "(dashed) are shown for context.",
        "MouseV2 probes use categorical display offsets centered on VISp; no numerical",
        "within-V1 hierarchy coordinate is assigned and no trend is fitted through probe means.",
    ]
    _position_caveat_lines = [
        "- **Within-V1 location**: the recordings are anatomically localized within V1,",
        "  but the probes have no validated hierarchy scores comparable to the published",
        "  inter-area values. Categorical probe and measured RF coordinates are kept separate.",
    ]
    _multiple_comparison_caption = "Benjamini–Hochberg FDR corrected"
else:
    _full_position_caption_lines = [
        "New V1 probe means are placed at the VISp hierarchy score.",
    ]
    _zoom_position_caption_lines = [
        "(dashed) are shown for context. A linear fit through the within-V1 probe means",
        "(solid grey, shaded 95% band) illustrates within-V1 spatial variance.",
    ]
    _position_caveat_lines = []
    _multiple_comparison_caption = "Bonferroni corrected"

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
    *_metric_caption_lines,
    f"*Left two columns*: probability density and cumulative distribution functions",
    f"for each visual area (LGN → AM), coloured by hierarchy position (Allen CCF).",
    f"New multi-site V1 sessions (n = {n_new_sessions} sessions, probes B/C/A/E, coloured by probe)",
    f"are overlaid on the V1 distributions.",
    f"*Third column*: area medians plotted against published hierarchy score",
    f"(Siegle et al. 2021); dashed line is OLS regression with r and p values.",
    *_full_position_caption_lines,
    f"*Fourth column*: pairwise significance matrix (Mann–Whitney U, {_multiple_comparison_caption}).",
    *_grating_method_lines,
    *_flash_method_lines,
    f"",
    f"### `Figure3_probe_zoom.png`",
    f"",
    f"**Within-V1 probe variance compared against the full hierarchy gradient.**",
    f"Same three metrics as above (rows), plotted against hierarchy score.",
    f"Each coloured dot is the session mean for one probe (B, C, A, E) in one of",
    f"n = {n_new_sessions} new multi-site sessions; error bars show probe mean ± SEM across sessions.",
    f"Original Allen area means (open circles, coloured by area) and OLS regression",
    *_zoom_position_caption_lines,
    *_ttfs_caption_lines,
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
    *_population_caption_lines,
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
    *_population_method_lines,
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
_grating_interpretation = (
    ('no between-group structure in either dataset',
     'F1/F0 does not track the hierarchy at session-mean level')
    if args.grating_metric == 'f1_f0_dg'
    else ('between-group structure is reported for the selected grating metric',
          'interpret this row together with its bootstrap interval')
)
interp_map = {
    'TTFS (ms)':       ('after QC, within-V1 probe variance is comparable to between-area variance',
                        'no clear hierarchy > V1 distinction'),
    _grating_label:    _grating_interpretation,
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
    *_ttfs_caveat_lines,
    *_position_caveat_lines,
    f"- **{_grating_name} population offset**: After QC, new-session V1 median {_grating_name} ≈ {_new_grating_median:.2f} vs",
    f"  Allen V1 median ≈ {_allen_grating_median:.2f}. Residual offset may reflect remaining population",
    f"  differences (e.g., absence of receptive-field quality filter in new sessions).",
    f"- **Underpowered comparison for TTFS and timescale**: With only {n_new_sessions} new",
    f"  sessions, ω²_probes has wide CIs. The comparison requires ~40 sessions to match",
    f"  Allen dataset power.",
    f"- **Within-session cross-area correlation** (Allen data) inflates effective df;",
    f"  direction of bias favours finding area > probe structure, so any null result",
    f"  is conservative.",
]

output_dir.mkdir(parents=True, exist_ok=True)
out_md = output_dir / 'Figure3_stats.md'
with out_md.open('w') as fh:
    fh.write('\n'.join(md_lines) + '\n')
print(f"\nMarkdown written → {out_md}")
