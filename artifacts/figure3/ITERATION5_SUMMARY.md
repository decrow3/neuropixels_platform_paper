# Iteration 5 — flash timing and polarity sensitivity

## Outcome

The eight MouseV2 sessions now have a versioned, all-unit flash-metric import
with pooled, bright-only, and dark-only measurements. The primary checkpoint
uses pooled flashes, matching the released analysis. Bright and dark are
retained as sensitivity analyses for the protocol difference in MouseV2.

MouseV2 TTFS is now shown raw relative to NWB interval `start_time` everywhere.
The former plot-only shift that forced the MouseV2 mean toward Allen V1 is
disabled in these checkpoints because it is not an independent display-timing
calibration. The available NWBs establish synchronized stimulus timestamps but
do not contain a photodiode trace or physical light-onset provenance, so
absolute Allen–MouseV2 latency offsets remain uninterpretable.

No flash variant supports the claim that post-V1 area labels explain more
variation than V1 probe labels. TTFS point estimates are larger for probes than
areas in all three variants. Timescale has an areas-greater-than-probes point
estimate for pooled and bright flashes, but all difference intervals cross zero
and the dark-flash result is nearly equal.

## Method validation and accommodation

- Each NWB has exactly 300 flashes: 150 bright (`contrast == +1`) and 150 dark
  (`contrast == -1`). The polarity labels are validated explicitly.
- TTFS is the median first occupied 1-ms bin across trials with a spike in the
  released 30–200-ms window. Pooled TTFS matches the frozen legacy value
  exactly for every shared unit in every session.
- Timescale uses the released 10-ms bin edges and AllenSDK xarray bin-center
  selection: 25 centers from 45 through 285 ms. The earlier MouseV2 path
  selected left edges and included a 26th bin centered at 295 ms.
- The corrected-versus-legacy timescale rank correlation ranges from 0.882 to
  0.952 across sessions. The extra-bin error therefore changes unit ordering
  materially even though many fitted values remain stable or hit fit bounds.
- The released exponential fit and figure validity rules are preserved:
  1–300 ms, flash spike count greater than 50, and fit error less than 20.

## Coverage and polarity differences

All 20,374 units were extracted; the declared `common_qc` population contains
11,242 units.

| MouseV2 flash set | Finite TTFS | Median TTFS | Figure-valid timescale | Median valid timescale |
| --- | ---: | ---: | ---: | ---: |
| Pooled | 11,063 | 87.0 ms | 2,375 | 44.07 ms |
| Bright | 10,870 | 87.0 ms | 1,902 | 37.26 ms |
| Dark | 10,750 | 84.5 ms | 2,246 | 37.08 ms |

Among common-QC units with paired values, the median bright-minus-dark TTFS is
+1.0 ms. Polarity affects timescale fit eligibility and the probe effect-size
point estimate more than it affects median TTFS, so the sensitivity views must
remain visible rather than being collapsed into an assumed equivalence.

## Effect-size sensitivity

The table reports session-aggregated omega squared. Allen values are the
released pooled-flash measurements in all rows; bright and dark rows change
only the MouseV2 side and are therefore sensitivity analyses, not matched
polarity comparisons.

| Metric | MouseV2 flash set | ω² probes | ω² post-V1 areas | Δω² areas−probes | 95% CI for Δω² | P(Δ≤0) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFS | Pooled | 0.160 | 0.125 | −0.036 | [−0.356, +0.112] | 0.811 |
| TTFS | Bright | 0.215 | 0.125 | −0.090 | [−0.445, +0.075] | 0.891 |
| TTFS | Dark | 0.172 | 0.125 | −0.047 | [−0.407, +0.145] | 0.763 |
| Timescale | Pooled | 0.033 | 0.172 | +0.140 | [−0.230, +0.304] | 0.252 |
| Timescale | Bright | 0.013 | 0.172 | +0.160 | [−0.247, +0.302] | 0.276 |
| Timescale | Dark | 0.129 | 0.172 | +0.043 | [−0.340, +0.247] | 0.521 |

The grating modulation-index row is unchanged across these checkpoints
(`ω²_probes = 0.171`, `ω²_areas = 0.151`, `Δ = −0.019`, 95% CI
`[−0.334, +0.147]`). None of the three response metrics currently supports a
resolved areas-greater-than-probes difference.

## Claim status

Iteration 5 strengthens the interim wording: **within-V1 variation can be
comparable to between-area variation, and the present analysis does not show
that area identity explains more variation.** It does not establish
equivalence, because the MouseV2 side has only eight sessions and the current
bootstrap does not yet preserve matched probes within session. The timescale
direction also remains sensitive to flash polarity.

The paper should not claim an absolute MouseV2-versus-Allen TTFS shift until
the independent calibration acceptance condition in
[`TIMING_CALIBRATION_BLOCKER.md`](../../data/imports/mousev2_flash_metrics_v1/TIMING_CALIBRATION_BLOCKER.md)
is met.

## Review artifacts

- Primary pooled checkpoint: [`05a_flash_pooled`](05a_flash_pooled/Figure3_stats.md)
- Bright-only sensitivity: [`05b_flash_bright_sensitivity`](05b_flash_bright_sensitivity/Figure3_stats.md)
- Dark-only sensitivity: [`05c_flash_dark_sensitivity`](05c_flash_dark_sensitivity/Figure3_stats.md)
- Versioned per-unit import and timing audit:
  [`mousev2_flash_metrics_v1`](../../data/imports/mousev2_flash_metrics_v1/README.md)

Each checkpoint contains all main comparison figures, the statistical report,
population and RF diagnostics, run logs, input/code hashes, and a manifest.

## Next executable step

Physical-onset calibration can proceed upstream without blocking the next local
iteration. The recordings are already anatomically localized within V1;
Iteration 6 should represent those known V1 locations without assigning them an
unsupported anatomical hierarchy score. It should compare categorical probe,
anatomical-coordinate, and measured-RF-coordinate views. In parallel,
PilotAnalysis can recover or validate raw sync/photodiode timing if those source
signals exist.
