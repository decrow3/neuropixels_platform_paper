# Figure 3 — Statistical Companion

_Generated 2026-08-04 · QC filter applied: yes (`default_qc == True`)_

---

## Figure captions

### `Figure3_with_V1sites.png`

**Response properties across the mouse visual hierarchy, with new multi-site V1 data overlaid.**
Each row shows one response metric: time-to-first-spike (TTFS, ms), log₁₀ modulation index,
and response decay timescale (ms).
*Left two columns*: probability density and cumulative distribution functions
for each visual area (LGN → AM), coloured by hierarchy position (Allen CCF).
New multi-site V1 sessions (n = 8 sessions, probes B/C/A/E, coloured by probe)
are overlaid on the V1 distributions.
*Third column*: area medians plotted against published hierarchy score
(Siegle et al. 2021); dashed line is OLS regression with r and p values.
New V1 probe means are placed at the VISp hierarchy score.
*Fourth column*: pairwise significance matrix (Mann–Whitney U, Bonferroni corrected).
Modulation index uses AllenSDK's Welch-spectrum z-score at the preferred full orientation × TF × SF condition.

### `Figure3_probe_zoom.png`

**Within-V1 probe variance compared against the full hierarchy gradient.**
Same three metrics as above (rows), plotted against hierarchy score.
Each coloured dot is the session mean for one probe (B, C, A, E) in one of
n = 8 new multi-site sessions; error bars show probe mean ± SEM across sessions.
Original Allen area means (open circles, coloured by area) and OLS regression
(dashed) are shown for context. A linear fit through the within-V1 probe means
(solid grey, shaded 95% band) illustrates within-V1 spatial variance.
TTFS values for new sessions carry a ~10 ms display-timing offset relative to
the original Allen data (different stimulus delivery hardware); this is a
systematic scalar shift and does not affect relative probe or area comparisons.

### `Figure3_split_comparison.png`

**Direct comparison of within-V1 probe variance (left) against post-V1 area variance (right).**
*Columns 0–1 (left pair)*: within-V1 data from n = 8 sessions.
Probes ordered B → C → A → E (approximately lateral → medial within VISp).
Dots show per-session means; horizontal bar and error bars show probe grand mean ± SEM.
Shaded box spans unit-level IQR (25th–75th percentile) pooled across all sessions
for that probe; thin whiskers span 10th–90th percentile.
Rotated KDE (col 1) shows the unit-level distribution pooled across all V1 probes;
dashed lines mark the IQR.
*Columns 2–3 (right pair)*: Allen SDK unit_table data (Siegle et al. 2021)
for post-V1 areas LM, RL, LP, AL, PM, AM, ordered by hierarchy score.
Thick segment = IQR; thin whisker = 10th–90th percentile; white dot = mean.
Rotated KDE (col 3) pools all post-V1 units.
Dashed regression line with r and p values.
Both panels use the same y-axis tick increment per row to allow direct
visual comparison of spread; absolute offsets between panels are expected
(different recording hardware/populations) and do not affect the variance comparison.
Unit quality filter: `default_qc == True` (NWB units table flag; ≈ 55% of units pass)
applied to new session data to match Allen's pre-filtered unit population.

---

## Statistical analysis: between-group variance in response properties

### Motivation

We ask whether categorical area labels (LM, RL, LP, AL, PM, AM) explain
significantly more variance in response properties than categorical probe labels
(B, C, A, E — different spatial positions within VISp) do within V1.
If area structure genuinely reflects a processing hierarchy, area membership
should account for substantially more variance than equivalent positional
variation within a single area.

### Effect size: ω² (omega-squared)

ω² = (SS_between − (k−1)·MS_within) / (SS_total + MS_within)

where k is the number of groups. ω² corrects the positive bias in η²
(= SS_between / SS_total) that arises when k differs between the two datasets
(k = 4 probes vs k = 6 areas). Under the null (no group structure), E[ω²] ≈ 0.

### Pseudoreplication correction

Units recorded in the same session share stimulus conditions, brain state,
and anaesthetic depth, violating independence. Treating them as independent
(n ≈ 10 000 per dataset) inflates effective sample size and can produce
spurious significance.

**Fix**: aggregate to session × group means before all analysis.

| Dataset | Aggregation | Approximate n |
|---------|-------------|---------------|
| New V1 sessions | probe × session mean | 8 sessions × 4 probes = 32 obs |
| Allen (post-V1) | area × session mean | ~40 sessions × 6 areas ≈ 240 obs |

Remaining limitation: the same Allen session contributes means for multiple areas
(simultaneous Neuropixels recording), introducing cross-group within-session
correlation. Bootstrap CIs resample sessions within groups to partially account
for this. This bias inflates the apparent Allen-area ω², making the comparison
conservative toward finding areas > probes.

### Unit quality filter

New session data: `default_qc == True` flag from NWB units table
(20374 total units → 11242 passing QC, 55%).
Allen data: unit_table.csv is pre-filtered to `quality == 'good'` units only.

### Results

Bootstrap 95% CIs (n = 5000 resamples) on ω²; sessions resampled within groups.

| Metric | Dataset | k | Session obs | η² | ω² | 95% CI (ω²) | Δω² (areas−probes) | 95% CI (Δω²) | P(Δ≤0) |
|--------|---------|---|-------------|----|----|-------------|-------------------|--------------|--------|
| **TTFS (ms)** | Within-V1 probes | 4 | 32 | 0.246 | 0.160 | [+0.050, +0.485] | — | — | — |
| | Post-V1 areas | 6 | 260 | 0.252 | 0.237 | [+0.161, +0.346] | +0.076 | [-0.249, +0.220] | 0.467 |
| **log10 modulation index** | Within-V1 probes | 4 | 32 | 0.255 | 0.171 | [+0.036, +0.487] | — | — | — |
| | Post-V1 areas | 6 | 264 | 0.144 | 0.127 | [+0.074, +0.216] | -0.044 | [-0.352, +0.117] | 0.790 |
| **Timescale (ms)** | Within-V1 probes | 4 | 32 | 0.098 | 0.001 | [-0.090, +0.402] | — | — | — |
| | Post-V1 areas | 6 | 243 | 0.198 | 0.180 | [+0.114, +0.282] | +0.179 | [-0.217, +0.321] | 0.182 |

_P(Δ≤0): bootstrap probability that ω²_areas ≤ ω²_probes; values near 0.5 indicate no detectable difference._

### Interpretation

**TTFS (ms)** — after QC, within-V1 probe variance is comparable to between-area variance. no clear hierarchy > V1 distinction.
ω²_probes = 0.160 [+0.050, +0.485], ω²_areas = 0.237 [+0.161, +0.346], Δω² = +0.076 [-0.249, +0.220], P(Δ≤0) = 0.467.

**log10 modulation index** — between-group structure is reported for the selected grating metric. interpret this row together with its bootstrap interval.
ω²_probes = 0.171 [+0.036, +0.487], ω²_areas = 0.127 [+0.074, +0.216], Δω² = -0.044 [-0.352, +0.117], P(Δ≤0) = 0.790.

**Timescale (ms)** — areas show real ω²; probes near zero — directional but underpowered with 8 sessions. direction consistent with hierarchy; not statistically distinguishable.
ω²_probes = 0.001 [-0.090, +0.402], ω²_areas = 0.180 [+0.114, +0.282], Δω² = +0.179 [-0.217, +0.321], P(Δ≤0) = 0.182.

### Caveats

- **TTFS timing offset**: New sessions show ~10 ms longer TTFS than Allen data,
  attributable to different stimulus display hardware (different monitor refresh
  latency). This is a scalar shift affecting absolute values but not variance structure.
- **modulation index population offset**: After QC, new-session V1 median modulation index ≈ 0.78 vs
  Allen V1 median ≈ 1.27. Residual offset may reflect remaining population
  differences (e.g., absence of receptive-field quality filter in new sessions).
- **Underpowered comparison for TTFS and timescale**: With only 8 new
  sessions, ω²_probes has wide CIs. The comparison requires ~40 sessions to match
  Allen dataset power.
- **Within-session cross-area correlation** (Allen data) inflates effective df;
  direction of bias favours finding area > probe structure, so any null result
  is conservative.
