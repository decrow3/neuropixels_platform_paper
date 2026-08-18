# MouseV2 RF-inverted V1 registration, per-probe properties, and shank-geometry correction

_Iterations 6E-6H, completed 2026-08-17_

## Motivation

MouseV2 NWBs have no CCF or anatomical coordinates at all (`/general/extracellular_ephys/electrodes/location`
is uniformly `unknown` in all eight sessions). The recordings are experimentally known to be
within V1, but nothing machine-readable places a given probe or unit at a specific point in V1's
retinotopic map. Iterations 6E-6H build that missing spatial axis from RF values themselves (an
inversion of every other registration in this project, which goes anatomy -> predicted RF value)
and use it to map MouseV2's improved parametric RF fits (size, and previously SF/TF in 6D) onto a
real position, then correct a geometric artifact the first version of that inversion produced.

## 6E — RF-inverted registration to the Zhuang V1 compartment

**Script:** `scripts/register_mousev2_rf_to_zhuang_v1.py`. **Output:** `artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1/`.

Since the probes are known to be in V1, the search space for inversion is restricted to Zhuang's
V1 (VISp) compartment only -- this removes almost all inversion ambiguity a full multi-area
inversion would have (no area-boundary confusion, no folded/reversed gradients). Each probe's
position is found by nearest-value match to the compartment's span-matched azimuth/elevation
field; a session-level RF-value offset (delta), shared across that session's probes, is fit
jointly via alternation (find positions given delta -> refit delta given positions -> repeat).

**Calibration bug found and fixed:** the harmonization constants borrowed from
`mousev2_frequency_preference_surfaces.py` (`+50` azimuth, `+10` elevation, chosen to match
Allen's *declared grid range*, not validated against true retinotopic correspondence) put
azimuth in good agreement with V1's own atlas median but put elevation ~11.5 deg off. Recalibrated
empirically by matching pooled medians against V1's own atlas distribution: azimuth offset +47.8
(vs. borrowed +50, confirms azimuth was fine), elevation offset -1.5 (vs. borrowed +10 -- this was
the bug).

**Validation (sign-unambiguous, never used in fitting):** probe LETTER (A/B/C/E) explains ~80% of
the variance in inferred position across independently-registered sessions (omega-squared:
row=0.82, col=0.84, both p<0.0005 against a 2000-shuffle label-permutation null) -- despite each
session's registration using no information about any other session, the same probe letter lands
in a consistent V1 region every time. A secondary check (declared probe order B>C>A>E vs.
inferred-position PC1, sign-corrected) gives median |rho|=0.50 with 6/7 sessions sharing the same
sign, consistent with (not stronger than) the original PilotAnalysis finding that this order held
in only 3/8 sessions.

## 6F — RF size and dispersion mapped to the registration

**Script:** `scripts/mousev2_rf_size_dispersion_surfaces.py`. **Output:** `artifacts/figure3/06f_mousev2_rf_size_dispersion_surfaces/`.

Extends 6E's per-probe registration to per-UNIT resolution (each unit's own RF value matched
independently, reusing its session's already-fitted delta), then:

- **RF size**: `pi * rf_sigma_major_deg * rf_sigma_minor_deg`, log2-scaled to match Allen's
  `log2_rf_area` convention. MouseV2 median log2 area 8.44 vs. Allen V1 7.83 (~1.5x larger) --
  descriptive only, different stimulus family/estimator (same caveat already on record for 6D's
  SF preference offset).
- **RF dispersion**: within-PROBE RF-center scatter (trace of covariance around the probe's own
  Huber-location center) -- the non-circular analog to Allen's anatomy-residual dispersion
  (MouseV2 has no independent anatomy to residualize against, since position here is derived from
  RF value). MouseV2 median much higher than Allen's per-unit-250um-neighborhood dispersion
  (log2 trace 8.35 vs. 6.27) -- expected given the very different spatial scale of "probe" vs.
  "250um neighborhood," not a comparable claim.

**This is where the geometric problem was first spotted**: the per-unit RF-size figure showed
units on a single probe scattered independently rather than forming the single linear shank they
physically must, since real probes are straight and units on one probe sit along a known relative
depth order.

## 6G — Depth-constrained shank-line registration

**Script:** `scripts/register_mousev2_units_along_probe_shank.py`. **Output:** `artifacts/figure3/06g_mousev2_rf_units_along_probe_shank/`.

Replaces independent per-unit nearest-match with a physically constrained model: a probe's units
lie along one straight line segment through V1 (`position(depth) = p0 + t(depth) * (p1 - p0)`),
fit per probe (Huber loss, session delta re-fit jointly by the same alternation as 6E) rather than
independently per unit.

An unconstrained version of this fit is underdetermined (a 4-parameter line easily overfits 15+
per-probe units): fitted shank lengths ranged 0.03x-6.2x an Allen-calibrated population-average
expectation with no regularization. This is resolved by 6H below.

## 6H — Per-probe insertion angle: what worked and what didn't

**Scripts:** `scripts/compute_mousev2_csd_insertion_angle.py`,
`scripts/detect_mousev2_csd_reversal_fixed_window.py`,
`scripts/measure_mousev2_cortical_thickness_vs_allen.py`,
`scripts/compare_rf_depth_span_mousev2_vs_allen.py`,
`scripts/render_mousev2_csd_batch_for_visual_read.py`.
**Output:** `artifacts/figure3/06h_mousev2_csd_insertion_angle/`.

MouseV2's `cortical_depth` field is raw `probe_vertical_position` (along-shank distance from a
device-specific reference point; `generate_retinotopic_csvs.py` L419-424), not a laminar/pia-
normal depth, and the insertion angle of these probes is unknown. Four CSD-based automated
landmark detectors were tried, in order, each validated against a real Allen ground-truth probe
(session 756029989, probeD, DANDI:001568's local ecephys cache, known L4 depth 2640-2700 um and
known VISl histological extent 2240-2980 um from `channels.csv`):

1. Earliest single-channel threshold crossing -- found pre-response noise (t=10-15ms, before any
   plausible visual latency).
2. Per-channel adaptive threshold -- a near-silent channel trivially crosses its own tiny noise
   floor and wins the earliest-onset comparison.
3. Global-threshold spatially-coherent sink detector -- worked well visually on MouseV2 (lands on
   the obvious sink by eye) but found nothing on the Allen validation probe (different noise/gain
   scale breaks fixed absolute thresholds).
4. Fixed-time-window source/sink reversal boundary -- landed within ~130-190 um of the known
   Allen L4 depth (better, not exact), but converting the resulting MouseV2 landmark depths into
   angles required assuming MouseV2 and Allen share the same `probe_vertical_position`
   zero-point convention. They very likely do not (most MouseV2 reversal depths were *shallower*
   than the Allen reference, which makes the angle formula undefined for a majority of probes) --
   this is a real, unresolved gap, not a tuning failure.
5. Responsive-band thickness (spatially-smoothed response power, largest contiguous run,
   1500-3500 um search window) -- validated reasonably well against the same probe's known VISl
   extent (920 vs. 740 um true, ~24% over-estimate) but, once restricted to the same search
   window as the MouseV2 application, gave a smaller Allen reference (680 um) than most MouseV2
   probes' detected bands, again failing to produce a consistent, trustworthy angle set (many
   probes clamped at 0 deg, others hit a suspicious repeated exact value).

**What worked: along-probe depth span of RF-significant units.** This reuses already-computed
unit-level RF fits with no new raw-signal processing, and gives a real multi-session Allen
reference rather than the single locally-cached LFP probe every CSD attempt was stuck with.
Allen (V1, `quality=='good'`, published-like RF significance: `p_value_rf<0.01`,
`on_screen_rf<0.01`, `area_rf<2500`, `snr>1`, `firing_rate_dg>0.1`): 24 probes with >=15
significant-RF units, median depth span (5th-95th percentile) 481 um (IQR 388-547). MouseV2
(`pilot_qc & rf_model_supported`): 27 probes, median span 844 um (IQR 697-1187) -- significantly
larger (Mann-Whitney p=2.2e-08), and consistently so across all 27 probes (every probe's span
exceeds the Allen median, unlike every CSD attempt's mix of zeros and outliers). Per-probe
estimated angle from vertical: median 55.3 deg (IQR 46.4-66.1), ranging 21-78 deg with a smooth,
plausible distribution (no boundary pile-up). A deliberately angled multi-probe insertion
strategy -- reaching different retinotopic positions from a shared craniotomy -- is consistent
with this project's explicit multi-probe-dispersal design, so a substantial median angle is
plausible rather than surprising. This is a **relative (span) measurement**: any fixed
per-probe/per-dataset zero-point offset in `probe_vertical_position` cancels out, which is why it
succeeds where the absolute-depth-landmark approaches (1-4 above) could not be trusted.

**Outputs:** `mousev2_rf_depth_span.csv`, `allen_rf_depth_span.csv`,
`Figure_rf_depth_span_mousev2_vs_allen.png`, plus the full CSD investigation trail
(`mousev2_csd_reversal_fixed_window.csv`, `mousev2_cortical_thickness_vs_allen.csv`,
per-probe CSD figures) kept on record for provenance even though superseded.

## Final integration

6G's shank-line fit was re-run using 6H's per-probe angle estimates (converted to an expected
tangential/depth ratio via `sin(angle)`, in Zhuang pixels via the project's fixed 104.6 px/mm
scale) as a soft log-ratio regularization penalty on fitted shank length, replacing the earlier
population-average-only regularization. Fitted/expected length ratios are now mostly 0.48-1.6x
(vs. 0.03-6.2x unregularized) -- well-behaved, not noise-dominated. One probe, `site2/C`, is a
clear outlier (ratio 1.59, fit loss 61 deg -- more than double any other probe's loss) and should
be treated as lower-confidence if used downstream.

## Known limitations

- The RF-depth-span angle estimate is itself approximate (a median-based population comparison,
  not a per-probe histological measurement) and inherits whatever bias is common to both the
  MouseV2 and Allen RF-significance criteria; it is trusted here because it is *relative* and
  because the resulting distribution is well-behaved, not because it has been independently
  validated the way the CSD approach's ground-truth checks attempted (and partly achieved) for
  absolute depth.
- The CSD investigation's absolute-depth findings (fixed-window reversal, responsive-band
  thickness) are retained on record because the underlying extraction pipeline and Allen
  ground-truth validation are sound, but they should NOT be used for absolute angle claims given
  the unresolved reference-point-convention gap.
- `site2/C`'s shank-line fit is flagged low-confidence (see above).
- No gaze correction is available for any MouseV2 iteration to date.
