# Figure 5 methods and reproduction contract

## What is reproduced

The default `reproduce.py` run starts from bundled metric-level observations,
released Allen unit metadata/timescales, and scalar anatomical source tables.
It reconstructs the anatomical inputs, the eligible-unit registry, all
statistical tables, and the figure. It checks the regenerated results against
the locked publication version (v17, corrected tests).

`extract.py` additionally computes the response metrics from the specified
public or original NWBs. Large recordings remain external. Public DANDI
001568 downloads need no API key or account; `extract.py download-mouse`
uses pinned public asset IDs and verifies their SHA-256 checksums. The public
files have updated metadata, while anatomical inclusion remains fixed to the
bundled snapshots below. Their identifiers, paths,
file sizes, and available hashes are in `raw_sources.json`. Original source
hashes and the exact columns retained when packaging metric tables are in
`source_manifest.json`. Compression and column projection do not change the
analysis values. `method_provenance.json` identifies the research-branch
functions retained here; draft plots and alternative analyses are omitted.

The published Allen timescale fits and recording metadata remain explicit
reference-data inputs. `audit-allen-timescales` exposes the raw reconstruction
and invalid-time removal for comparison; differences in historical fitting
backends are not silently substituted into the canonical analysis. Original
Allen scripts elsewhere in the repository are archival and are not runtime
dependencies of this extension.

## Populations and anatomical sources

MouseV2 contributes eight animals, each recorded with four V1 probes: anterior
(A), lateral (E), posterior (C), and medial (B). Allen higher visual areas are
LM, RL, AL, PM, and AM. Allen V1 from sessions shared with the HVA population is
the positive control for the mean-difference analysis.

The common waveform-quality rule is amplitude cutoff < 0.1, presence ratio
> 0.8, and ISI-violations ratio < 0.5. MouseV2's `default_qc` must agree with
that rule. Metric eligibility is applied independently, followed by at least
five unique cells per metric × recording × category × timescale draw.

The anatomical sources have two roles and dates:

- **Unit inclusion: September 2, 2026 snapshot.** Units join to AP electrodes
  using subject/session, probe, and extremum-channel index. AP channel numbers
  are converted from one-based names to zero-based extremum indices. Only
  locations beginning `VISp` enter the response registry. Full unit metadata
  are supplied, so inclusion is computed rather than encoded as an admitted-ID
  list. The later snapshot would change some units' eligibility.
- **Surface illustration: September 25, 2026 snapshot.** Each probe entry is
  represented by the most superficial VISp-labelled AP contact (maximum
  `rel_y`). AP/ML coordinates are projected onto the 10-µm cortical atlas.
  Several contacts share the same depth. The original NumPy quicksort tie
  choices are recorded explicitly in `raw_sources.json`; regeneration checks
  that each chosen contact is still at the maximal eligible depth. This avoids
  library-version-dependent changes to the locked illustration.

The bundled scalar tables preserve those snapshots without distributing whole
NWB recordings. `anatomy.py` reconstructs both products from them. Its optional
`--nwb-root` reloads the September 2 unit-inclusion snapshot; the surface
illustration still uses its separately declared September 25 scalar source.

## Time to first spike

Preferred flash polarity is the polarity with the larger mean spike count in
[0, 250) ms; an exact tie selects bright. Latency is the median first occupied
1-ms bin from 30 through 200 ms, excluding trials without a spike in that
window. Preference and latency use all trials of the relevant polarity.

Response selection uses alternating presentations within each polarity,
starting with the first presentation. Mean rate in 30–180 ms is compared with
−100–0 ms using a one-sided paired Wilcoxon test. Selection requires an effect
of at least 2 Hz and Benjamini–Hochberg q < 0.01. The FDR family is the whole
session/polarity for MouseV2 and session/area/polarity for Allen. Thus this is
latency-independent amplitude selection, not a claim that the final latency
estimate uses held-out trials. The preferred polarity must pass selection,
and its TTFS must be < 100 ms. `flash_metrics.py` retains the original bin and
floating-point boundary conventions.

## Phase modulation

The common stimulus support is orientations 0°, 45°, 90°, and 135°, temporal
frequencies 1, 2, 4, 8, and 15 Hz, spatial frequency 0.04 cycles/degree, and
contrast 0.8. There are 15 trials per condition; the analysis window is one
second. Allen sessions lacking this complete support do not contribute to
harmonized F1/F0. This yields the accepted Brain Observatory cohort rather
than pooling different stimulus protocols.

Preferred condition maximizes mean spike count in the analysis window.
MouseV2 resolves ties by lexicographic condition order; Allen preserves first
presentation order, matching its condition ordering. Each trial's 1-ms spike
counts are folded into whole cycles. F1/F0 is the first harmonic amplitude
divided by DC, averaged across trials with nonzero F0. At 15 Hz, the integer
cycle length uses 990 of the 1,000 bins for the harmonic calculation, while
condition preference uses the full second. Only positive ratios enter the
analysis; values are transformed with log10. The code does not substitute a
coherent trial-average spectrum or Welch modulation index for this metric.

## Response timescale

MouseV2 uses ten balanced random draws of 75 bright and 75 dark flashes from
150 of each polarity. The session seed is `20260805 + site_number`. Each draw
uses 10-ms spike-count bins spanning 40–290 ms. The original 2-D correlation
reduction removes the central trial-offset row, averages remaining rows, and
fits the positive-lag curve with a bounded exponential plus offset. Its time
axis and optimizer settings are retained in `flash_metrics.py`.

A fit is attempted above 50 spikes. Eligible estimates are 1–300 ms inclusive,
with > 50 spikes and fitted tau standard error < 20 ms. Allen uses the
published `timescale_ac`, `err_ac`, and `spike_count_ac` fields with the same
eligibility thresholds. The optional raw Allen audit bins the released flash
trials and removes spikes in declared invalid intervals before reconstruction.

## Category means and dispersion (panel E)

`prepare.py` fits weighted category indicators plus session nuisance effects.
Weights give equal mass to categories, then animals within category,
recordings within category/animal, and cells within recording. The category
coefficient variance (population divisor K) is the dispersion estimand.
Square-root dispersion is displayed as the SD of category means.

The HVA-minus-V1 SD difference is calculated within each paired bootstrap
replicate after taking square roots; it is not the square root of a variance
difference. The Allen central-V1 control uses sessions containing both V1 and
HVA observations, balanced weights, and a shared-animal resample. Its mean
contrast preserves the within-session comparison.

Whole animals are resampled with replacement, retaining their sessions and
recordings together. Repeated animals receive distinct bootstrap identifiers.
For MouseV2 timescales, each resampled session selects one of the ten draw
registries. Point estimates average the draw-specific results. There are
5,000 replicates; primary dispersion uses seed 20260918. The matched control's
seed is specified in `prepare.py`. Rank-deficient fits fail rather than
silently changing the model.

## Location-associated variance (panel F)

Panel F uses bias-corrected one-way omega-squared on neuronal responses with
equal cell weights, without session adjustment. Negative estimates are
retained. It therefore answers a different question from panel E's balanced,
session-adjusted mean dispersion.

Intervals are individual 95% percentile intervals from 5,000 whole-animal
bootstrap replicates (seed 20260921). The direct HVA-minus-V1 contrast is the
difference between independent-population bootstrap estimates, reported in
percentage points. It has no separate shuffle-test significance symbol.

Component tests use 5,000 within-session location-block label shuffles (seed
20260929). All neurons in a location move together; MouseV2 timescale draws
share a coherent permutation within each session. Available labels are
permuted; single-category sessions remain fixed. One-sided p-values use
`(1 + exceedances) / (5000 + 1)`. Bonferroni adjustment covers six component
tests: three metrics × two populations. Confidence intervals are not
simultaneous. Nonsignificance does not establish equivalence or absence of an
effect.

## Validation and output isolation

The quick figure build checks the locked CSV hashes and decompressed bootstrap
hash. Full regeneration compares all seven analysis tables, allowing numerical
roundoff of `rtol=1e-10`, `atol=1e-12` when serialization differs. It also checks
shapes, order, and nonnumeric labels. The final renderer reads the regenerated
tables and applies the same numerical-equivalence thresholds to figure outputs.
Raw extraction uses `rtol=1e-7`, `atol=1e-8` for metric comparisons across the
historical extraction backend; IDs, selected populations, and validity flags
must still agree. A subset extraction cannot pass the full-reproduction gate.

All calculation seeds and source hashes are recorded. The pinned rendering
environment is separate from the historical extraction environment. Fonts and
numerical-library backends can affect image pixels or low-order numerical
serialization on another system; numerical validation is the acceptance gate.
The regression suite checks the reconstructed registry, anatomical derivation,
source completeness, incomplete-run rejection, and metric boundary cases.
