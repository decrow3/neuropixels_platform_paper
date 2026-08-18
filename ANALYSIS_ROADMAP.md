# Retinotopic V1 hierarchy analysis roadmap

_Living plan, initialized 2026-08-04_

## Purpose

The scientific question is whether functional differences among labeled post-V1
visual areas are larger than the variation found across retinotopic locations
within a single established area, V1. The analysis uses the published hierarchy
metrics as a reference and asks whether area identity explains more variation
than V1 position does after accounting for the receptive-field locations
actually sampled. Allen's insertion plan was retinotopically targeted, but
intended targeting and achieved unit-level RF matching are separate quantities
and must not be treated as interchangeable controls.

The current MouseV2 pass is a successful pipeline validation, not the final
paper analysis. All eight sessions have TTFS, F1/F0, response-timescale,
probe-label, and unit-quality tables, and the comparison figures and
omega-squared analysis run end to end. Several metric definitions, population
filters, and experimental-drift accommodations still require validation.

The working principle for the next phase is **improve and rerun incrementally**.
Every material change should produce a new, reviewable set of figures and a
short comparison with the previous iteration. RF, within-V1 location, and CCG work should
not prevent us from improving and viewing the main three-metric analysis now.

## Execution status

| Iteration | Status | Review artifact |
| --- | --- | --- |
| 0 — pipeline baseline | Completed 2026-08-04 | [`00_pipeline_baseline`](artifacts/figure3/00_pipeline_baseline/BASELINE_SUMMARY.md) |
| 1 — reproducible baseline | Figure stage completed 2026-08-04 | [`01_reproducible_baseline`](artifacts/figure3/01_reproducible_baseline/delta_from_previous.md) |
| 2 — measured RF centers | Completed provisionally 2026-08-04 | [`02_measured_retinotopy`](artifacts/figure3/02_measured_retinotopy/rf_position_report.md) |
| 3 — full-condition grating metric | Completed 2026-08-04 | [`ITERATION3_SUMMARY`](artifacts/figure3/ITERATION3_SUMMARY.md) |
| 4A — harmonized common QC | Completed 2026-08-05; 4B/4C blocked upstream | [`ITERATION4A_SUMMARY`](artifacts/figure3/ITERATION4A_SUMMARY.md) |
| 5 — flash timing and polarity | Completed provisionally 2026-08-05; physical onset blocked upstream | [`ITERATION5_SUMMARY`](artifacts/figure3/ITERATION5_SUMMARY.md) |
| 6A — known within-V1 locations | Completed 2026-08-05; exact anatomical coordinates optional | [`ITERATION6_SUMMARY`](artifacts/figure3/ITERATION6_SUMMARY.md) |
| 6B — V1 cross-dataset bridge | Source-corrected Welch diagnostic completed 2026-08-09; claim gate remains closed pending multi-session Allen and population matching | [`V1_DATASET_BRIDGE`](artifacts/figure3/06b_v1_dataset_bridge/V1_DATASET_BRIDGE.md) |
| 6C — achieved Allen retinotopy | Targeting audit and first RF-adjusted response checkpoint completed 2026-08-11; balance/model sensitivity and MouseV2 bridge remain | [`ALLEN_RF_MATCHING`](artifacts/figure3/06c_allen_rf_matching/ALLEN_RF_MATCHING.md), [`RF_ADJUSTED_RESPONSE`](artifacts/figure3/06c_allen_rf_matching/response_adjustment/ALLEN_RF_ADJUSTED_RESPONSE.md) |
| 6D — MouseV2 frequency-preference surfaces | Parametric trial-derived RF and SF/TF/orientation models completed 2026-08-11; gaze correction remains unavailable | [`MOUSEV2_FREQUENCY_PREFERENCE_SURFACES`](artifacts/figure3/06d_mousev2_frequency_preference_surfaces/MOUSEV2_FREQUENCY_PREFERENCE_SURFACES.md) |
| 6E-6H — RF-inverted V1 registration, size/dispersion mapping, shank-geometry correction | Completed 2026-08-17; per-probe insertion angle estimated via RF-significant-unit depth span vs. a 24-probe Allen reference (CSD-based absolute-depth landmark detection tried first, retained on record but not trusted for angle claims -- see summary) | [`MOUSEV2_PROBE_SHANK_REGISTRATION`](artifacts/figure3/06g_mousev2_rf_units_along_probe_shank/MOUSEV2_PROBE_SHANK_REGISTRATION.md) |

Iteration 0 reran all four current entry points successfully. The regenerated
figures are pixel-identical to the preserved inputs, and the statistical report
differs only by its generated date. The baseline bundle records per-site,
per-probe, and per-metric counts plus code, data, environment, and repository
provenance.

Iteration 1 now provides a central eight-session configuration, frozen stimulus
manifest, validated shared loader, per-entry-point output directory, automated
schema/coverage tests, and a one-command figure/statistics runner. Its products
are scientifically equivalent to Iteration 0. Raw NWB-to-metric extraction is
not rerun by that command yet because the source NWB locations and identities
are not versioned in this repository; the existing extractor already opens
NWB files read-only. Iteration 3 completed the provenance bridge by recording
the eight DANDI:001568 relative paths, expected sizes, and full SHA-256 hashes.

Iteration 2 maps all 20,374 PilotAnalysis RF peak rows one-to-one onto the paper
tables and summarizes positions from 4,807 units passing Pilot's declared
stricter QC. The measured centers show that B>C>A>E is strictly descending in
RF azimuth in only 3/8 sessions (5/8 allowing grid ties), so a universal
one-dimensional probe order is not supported. The new analysis retains both RF
coordinates and keeps the categorical view as a sensitivity analysis. These
positions remain provisional: no gaze correction or RF-significance filter is
available, and the per-unit raw grid argmax lands on a stimulus-grid edge for a
median 46% of units within a session × probe group.

Iteration 6D supersedes those grid argmaxes for frequency-preference mapping
with supported trial-level elliptical Gaussian RF fits; the historical
Iteration 2 artifact remains frozen for provenance.

Iteration 3 recomputes drifting-grating metrics for all 20,374 units at each
unit's preferred orientation × TF × SF condition. It keeps corrected `f1_f0_dg`,
Allen's separate Welch-spectrum `mod_idx_dg`, and the former pooled-SF values
under the explicit name `f1_f0_dg_pooled_sf_legacy`. The raw-to-metric formulas
match the installed AllenSDK 2.16.2 source in synthetic tests; independent
PilotAnalysis preferred-condition triplets agree for a mean 97.0% of shared
units across sessions. Both metric-specific figure checkpoints are retained.

Iteration 4A applies an explicit homologous waveform-quality profile to every
Figure 3 and statistics path. It retains 11,242/20,374 MouseV2 units (exactly
the existing `default_qc` set) and 43,496/99,180 Allen units before area
restriction. Both grating-metric checkpoints were regenerated with population
flows and per-group counts. The released population filter is computable for
Allen, but the MouseV2 `published_like` and RF-area checkpoints remain blocked
because PilotAnalysis has no validated all-session RF significance or area
export. The grating firing-rate criterion is now derived explicitly from the
already validated preferred-condition response window.

Iteration 5 recomputes pooled, bright, and dark flash metrics for all 20,374
units. Pooled TTFS is exactly equal to the legacy extractor for every shared
unit. Timescale now matches AllenSDK's bin-center selection (25 centers at
45–285 ms), correcting the earlier extra 295-ms bin. The figures use raw
NWB-relative TTFS rather than mean matching to Allen V1. All three polarity
checkpoints leave the areas-minus-probes contrast unresolved; physical
light-onset calibration remains unavailable in the NWBs.

Iteration 6A records that the recordings and sampled subregions are known to be
within V1; what is unavailable is a hierarchy score for those locations. The
reviewed figures now use display-only offsets centered on VISp and do not fit a
line through arbitrary probe x positions. The categorical effect sizes are
unchanged, and the measured RF coordinates remain a separate retinotopic view.

Iteration 6B makes the absolute V1 discrepancy an explicit prerequisite rather
than a later sensitivity. Equal-session MouseV2 log10 `mod_idx_dg` is −0.107,
compared with +0.040 for Allen Brain Observatory V1 and +0.199 for Allen
Functional Connectivity V1. F1/F0 does not share that downward offset, which
identifies a metric/protocol interaction rather than a generic absence of
grating modulation. The released Welch estimator also evaluates different
frequency bins solely because the Allen PSTH is 2 s and MouseV2 is 1 s. The
timescale offset is smaller (+3.65 ms versus Allen Brain Observatory) but remains
unmatched for flash trial structure and population support. The main claim is
therefore gated on a common-window, common-condition, common-trial-count raw
reanalysis; no mean matching is permitted.

Iteration 6C separates Allen's intended targeting from the RF locations
achieved by the recorded units. Allen used ISI-derived retinotopic maps to
target a common V1-aligned region in V1, LM, AL, AM, and PM; RL received a
documented geometric-center accommodation. The first executable audit now
summarizes `azimuth_rf` and `elevation_rf` by unit, probe, session, and area and
estimates paired HVA-minus-V1 offsets within Allen sessions. Median paired
session-center distances are 25.9° for LM, 31.7° for RL, 33.3° for AL, 33.7°
for PM, and 19.8° for AM in the published-like population. These achieved
offsets do not invalidate the targeting method, but they rule out assuming
tight neural RF matching. RF-center dispersion and individual RF size remain
separate quantities. The first response checkpoint now carries those
coordinates into within-session models and same-session V1 matching. The
result is metric-specific: modulation-index and RF-area differences persist;
TTFS and timescale differences attenuate strongly for several areas; and F1/F0
area coefficients shrink toward zero in the covariate-adjusted model. This is
not yet a pass because RF-position interactions are detected for 3/5 outcomes
and no matching caliper simultaneously provides tight balance, low attrition,
and broad session support.

The first nonlinear tuning-preference view is also implemented for Allen Brain
Observatory sessions. Session-balanced Gaussian surfaces map released preferred
SF and TF bins over achieved RF azimuth/elevation at 8°, 12°, and 16° spatial
bandwidths. Each HVA-minus-V1 surface uses V1 only from the same session set and
masks cells without local multi-session support. At the primary 12° bandwidth,
the HVA surfaces generally prefer lower SF and higher TF than paired-session V1,
with spatially varying effect sizes. These are preference surfaces rather than
full tuning curves because Allen SF and TF were measured in separate static-
and drifting-grating stimulus classes.

The first raw bridge leg is complete for all eight MouseV2 sessions. Restricting
preference to Allen's fixed SF = 0.04 cycles/degree changes the equal-site mean
log10 modulation index by only +0.009 (site range −0.019 to +0.036), while
log10 F1/F0 increases by +0.061. The SF/condition-space difference therefore
does not explain the large modulation-index offset. The remaining executable
bridge requires raw Allen Brain Observatory and Functional Connectivity NWBs.

Iterations 6E-6H build the spatial axis MouseV2 otherwise lacks entirely (no CCF
in these NWBs) by inverting the registration direction used everywhere else in
this project: RF value → inferred V1 position, restricted to Zhuang's V1
compartment (the recordings are known to be in V1, which removes almost all
inversion ambiguity). A harmonization-offset bug was found and fixed in the
process (elevation was off by ~11.5 deg; azimuth was fine). The registration is
validated by a sign-unambiguous, never-fit-on check: probe letter explains ~80%
of inferred-position variance across independently-registered sessions
(omega-squared, p<0.0005). Mapping RF size and dispersion onto that registration
(6F) exposed a geometric problem: independent per-unit matching does not respect
the fact that a probe is a straight physical shank, so units scattered rather
than forming a line. Iteration 6G fixes this with a depth-constrained line model
per probe. That fit is underdetermined without knowing insertion angle, which is
not recorded in these NWBs; iteration 6H tried five methods to estimate it,
including CSD source/sink landmark detection validated against a real Allen
ground-truth probe -- informative about the extraction pipeline's correctness
but ultimately not trustworthy for absolute-depth angle claims because MouseV2
and Allen almost certainly do not share a `probe_vertical_position` zero-point
convention. What worked instead: the along-probe depth SPAN of RF-significant
units (a relative measure, immune to that zero-point problem), compared against
a real 24-probe Allen V1 reference -- MouseV2 spans are significantly larger
(Mann-Whitney p=2.2e-08) consistently across all 27 probes, giving a plausible
median 55.3 deg estimated insertion angle from vertical. Full narrative,
failure modes, and caveats in
[`MOUSEV2_PROBE_SHANK_REGISTRATION.md`](artifacts/figure3/06g_mousev2_rf_units_along_probe_shank/MOUSEV2_PROBE_SHANK_REGISTRATION.md).

## Experiment-to-analysis chain

```text
openscope_v2species
    authoritative stimulus and acquisition definitions
                 |
                 v
MouseV2 NWB files
                 |
                 v
PilotAnalysis
    NWB/RF/gaze/tuning/anatomy exploration and validation
                 |
        versioned export contract
                 |
                 v
neuropixels_platform_paper
    paper-compatible metrics, comparisons, statistics, and figures
```

### Repository responsibilities

| Repository | Owns | Should not own |
| --- | --- | --- |
| [`openscope_v2species`](../../openscope_v2species/README.md) | Authoritative ephys stimulus sequence, trial structure, parameter values, presentation counts, and acquisition-era timing assumptions. | Paper statistics or derived neural metrics. |
| [`PilotAnalysis`](../../PilotAnalysis/PilotAnalysis/README.md) | NWB validation; RF and gaze analyses; probe retinotopy; SF/TF and orientation-tuning exploration; anatomy/CSD checks; diagnostic figures. | Final hierarchy figures or a second independent implementation of paper metrics. |
| This repository | Stable raw-to-metric extraction for the hierarchy analysis; compatibility with the published definitions; cross-dataset population rules; statistical comparisons; iteration manifests; final figures and methods. | Acquisition code or open-ended exploratory notebooks duplicated from PilotAnalysis. |

The boundary is not “each calculation may exist in only one place.” PilotAnalysis
can retain exploratory or approximate versions for diagnosis. The paper-facing
implementation and the definition used in a claim must live here, with tests
and provenance. Conversely, RF/gaze/anatomical calculations should mature in
PilotAnalysis and cross into this repository through an explicit, validated
table rather than by copying a large interactive script.

## Authoritative stimulus facts to preserve

The acquisition repository and the eight NWBs agree on the following ephys
design:

| Block | Design | Presentations per session | Analysis consequence |
| --- | --- | ---: | --- |
| Receptive-field Gabors | 9 × 9 positions, three orientations, fixed TF/SF, 20 repeats | 4,860 | RF significance, center, and area must account for position and orientation. |
| Drifting gratings | 4 orientations × 5 TFs × 5 SFs, 15 repeats, approximately 1 s | 1,500 | Preferred condition must include spatial frequency; the published dataset used a different duration/condition space. |
| Full-field flashes | bright and dark, 150 repeats each, approximately 250 ms | 300 | TTFS and timescale can pool polarities only after confirming that this matches the published method; polarity-specific results are a useful sensitivity check. |

Iteration 1 froze a machine-readable snapshot of these facts in
[`mousev2_stimulus_manifest.json`](config/mousev2_stimulus_manifest.json),
including the source repository commit and the sessions checked. Subsequent
code should consume that snapshot rather than repeatedly inferring the protocol
from loose filename matching.

## Current state and known gaps

### Completed baseline

- Eight sessions (`site2` through `site9`), each with probes A, B, C, and E.
- 20,374 units before filtering and 11,242 units passing `default_qc`.
- Per-unit TTFS, current F1/F0, response-timescale, probe/depth, and QC tables.
- Three Figure 3 comparison views and a session-aggregated omega-squared report.
- A single-session cross-probe CCG pipeline test (`site2`).
- PilotAnalysis RF peak tables and RF/tuning diagnostic figures for all eight
  sessions.

### Gaps that affect the main claim

1. The original base filter is named and computable for Allen, and its
   `snr`/`firing_rate_dg` components are available for MouseV2, but MouseV2
   still lacks validated all-session `p_value_rf` and `area_rf` fields.
2. Common waveform QC is now applied consistently across every Figure 3 and
   statistics entry point, but the primary paper population cannot be selected
   until the RF filter is available and session balance is reviewed.
3. The reviewed categorical figures now label MouseV2 x offsets as display-only
   and fit no within-V1 hierarchy trend; the companion view uses provisional
   measured RF coordinates. Exact anatomical coordinates are not yet versioned.
4. The recording locations are anatomically established within V1, including
   which part of V1 was sampled, but those locations do not yet have a validated
   anatomical hierarchy score comparable to the inter-area score.
5. Iteration 5 figures consistently use raw NWB-relative TTFS, but independent
   physical display-onset calibration is still unavailable; absolute
   Allen–MouseV2 offsets are therefore not interpreted.
6. The current session bootstrap does not preserve matched probes within a
   MouseV2 session.
7. LP is grouped with cortical post-V1 areas even though it is thalamic.
8. The CCG result is only a smoke test and currently uses a PSTH predictor,
    not the original jitter predictor.
9. The released Allen and MouseV2 absolute V1 levels are not calibrated. In
   particular, `mod_idx_dg` changes meaning with the 2-s versus 1-s Welch grid,
   and Allen V1 mixes two stimulus sets with different modulation-index centers.
   The paper claim remains closed until Iteration 6B's raw bridge acceptance
   analysis passes.
10. Allen's intended V1/HVA targets were retinotopically aligned, with RL as a
    documented exception, but achieved RF-center matching has not been
    quantified at the session level. The paper must not use “retinotopically
    targeted” and “RF-matched neural populations” interchangeably. Iteration 6C
    audits the public RF metrics and carries achieved coordinates into the
    primary comparison.

PilotAnalysis currently has useful per-unit RF peak exports for all eight
subjects. Its `rf_metrics.csv`/`rf_probe_summary.csv` batch product exists only
for subject 810531, and `compute_stimulus_metrics.py` labels its RF significance
as a placeholder. Those files are suitable for an early probe-position figure,
but not yet for the final RF filter or RF-area panel.

## Cross-repository data contract

### Stable session map

| Site | Subject | Probe labels |
| --- | --- | --- |
| site2 | 816305 | A, B, C, E |
| site3 | 810531 | A, B, C, E |
| site4 | 810532 | A, B, C, E |
| site5 | 813810 | A, B, C, E |
| site6 | 815152 | A, B, C, E |
| site7 | 816308 | A, B, C, E |
| site8 | 817334 | A, B, C, E |
| site9 | 817335 | A, B, C, E |

This mapping should move from duplicated dictionaries into one versioned
configuration file in this repository.

### PilotAnalysis export

PilotAnalysis should eventually emit one versioned table per session with at
least:

```text
subject_id
local_unit_id
probe
rf_center_x_deg
rf_center_y_deg
area_rf_deg2
p_value_rf
has_significant_rf
rf_method
gaze_correction
anatomical_structure (when available)
cortical_layer (when available)
```

It should also emit a probe summary containing the session × probe RF center,
dispersion, number of valid units, and uncertainty. Every export needs a small
JSON/YAML provenance record with the PilotAnalysis commit, input NWB identity,
parameters, and creation time.

### Import into this repository

This repository should own a small importer/validator that:

1. maps `subject_id + local_unit_id` to the existing offset `unit_id`;
2. rejects duplicate or unmapped units;
3. validates probe labels and units;
4. records the source export and checksum;
5. writes stable columns into `data/siteN_processed/` without recalculating RFs;
6. produces a coverage report before those columns affect filtering.

The final analysis must not read arbitrary files directly from a neighboring
working tree. Imports should be explicit snapshots so a result remains
reproducible if PilotAnalysis changes.

## Iterative execution rule

Each iteration follows the same short loop:

1. State the one methodological change being tested.
2. Add or update a focused equivalence/unit test.
3. Run one representative session first (site3 is useful because PilotAnalysis
   already has its fuller RF export).
4. Inspect per-unit agreement, counts, exclusions, and one diagnostic figure.
5. Run all eight sessions if the smoke test passes.
6. Regenerate all main comparison figures and the statistics report.
7. Write a one-page delta summary: what changed, why, expected effect, observed
   effect, and any newly discovered issue.
8. Freeze the iteration artifacts before starting the next improvement.

An iteration can reveal a problem without being discarded. Preserve it and
label it clearly; the history is useful evidence that the final result is not
dependent on silent pipeline changes.

### Iteration artifacts

Add an output option so generated products are written beneath a run ID rather
than silently overwriting the canonical figures:

```text
artifacts/figure3/<run_id>/
    run_manifest.json
    metric_counts.csv
    exclusions.csv
    Figure3_with_V1sites.png
    Figure3_probe_zoom.png
    Figure3_split_comparison.png
    Figure3_stats.md
    delta_from_previous.md
```

Large per-unit tables may remain ignored, but the run manifest should include
their checksums, source NWBs, code commits from all contributing repositories,
environment/package versions, arguments, and random seeds. Reviewed iterations
can be copied to the canonical Figure 3 filenames; exploratory runs should not
replace them automatically.

## Step-by-step analysis plan

### Iteration 0 — Freeze the current pipeline baseline

**Location:** this repository.

**Work:**

- Preserve the existing three figures and `Figure3_stats.md` as
  `00_pipeline_baseline`.
- Record that F1/F0 pools spatial frequencies, RF filters are absent, two
  figures omit MouseV2 QC, and TTFS is raw in some views and mean-matched in
  others.
- Save counts by session, probe, metric, and exclusion reason.

**Figure checkpoint:** no scientific change. The frozen outputs must match the
current files pixel-for-pixel or numerically within plotting tolerance.

**Value:** establishes an auditable reference before corrections begin.

### Iteration 1 — Make reruns reproducible without changing the science

**Location:** primarily this repository; stimulus facts sourced from
`openscope_v2species`.

**Work:**

- Add a single session/config manifest and remove duplicated site maps and
  machine-specific paths from analysis scripts.
- Open NWBs explicitly read-only.
- Add one command that extracts metrics, validates tables, regenerates figures,
  and writes a run manifest.
- Add `--output-dir`/`--run-id` support to figures and statistics.
- Add table-schema, row-count, unique-ID, and probe-coverage tests.
- Snapshot the stimulus manifest with acquisition-repository provenance.

**Figure checkpoint:** `01_reproducible_baseline` should reproduce Iteration 0.
Any difference is a pipeline regression and is resolved before proceeding.

### Iteration 2 — Use measured RF centers for the first retinotopic view

**Location:** PilotAnalysis validates/exports RF positions; this repository
imports and plots them.

**Work in PilotAnalysis:**

- Validate existing per-unit RF peak tables across all sessions.
- Produce a versioned session × probe RF-center summary using a declared QC
  rule and uncertainty estimate.
- Compare uncorrected and gaze-corrected centers where eye data support it.
- Do not yet treat the existing approximate RF area or placeholder p-value as
  the final Allen-compatible RF metric.

**Work here:**

- Implement and test the RF import contract.
- Replace arbitrary probe order/x positions with measured azimuth/elevation.
- Add a two-dimensional RF-position diagnostic and retain the original
  categorical probe view alongside it.

**Figure checkpoint:** `02_measured_retinotopy`. The three response metrics are
still the baseline versions, but probe positions now reflect data. Report how
often probe ordering agrees across sessions and whether B→C→A→E is justified.

**Completed checkpoint:** the exact RF input snapshot and new figure can be
regenerated with:

```bash
python scripts/import_pilot_rf_peaks.py
python scripts/run_figure3_iteration.py \
    --run-id 02_measured_retinotopy \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
```

The original three figures and statistical report are regression-identical to
Iteration 0. The measured-position figure uses raw TTFS, the current baseline
F1/F0 and timescale definitions, and `default_qc` for response-property means;
RF centers use Pilot's stricter `is_qc` population. All response-coordinate
correlations in this checkpoint are descriptive session × probe summaries,
not final matched-session inference.

### Iteration 3 — Correct and validate the drifting-grating metric

**Location:** final implementation and tests here; PilotAnalysis tuning outputs
serve as diagnostics; `openscope_v2species` supplies protocol truth.

**Work:**

- Define conditions as orientation × TF × SF × any other varying stimulus
  dimension.
- Select the preferred full condition per unit.
- Apply the AllenSDK cycle-fold F1/F0 mathematics with an explicitly documented
  accommodation for the approximately 1-s MouseV2 trials.
- Compute the original `mod_idx_dg` separately; never store F1/F0 under the
  `mod_idx_dg` name.
- Add synthetic tests and a gold-standard test that processes an original Allen
  session through the new low-level path and compares unit metrics with the
  published AllenSDK table.
- First rerun site3, compare corrected results with PilotAnalysis SF/TF
  preferences, then rerun all sessions.

**Figure checkpoints:**

- `03a_f1f0_full_condition` — corrected F1/F0.
- `03b_original_modulation_index` — original Figure 3 metric.

Keep both views. Choose the primary metric only after examining equivalence,
missingness, and sensitivity of the scientific conclusion.

**Completed checkpoint:** raw metrics and both figure views can be regenerated
with:

```bash
python scripts/extract_mousev2_grating_metrics.py
python scripts/run_figure3_iteration.py \
    --run-id 03a_f1f0_full_condition \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric f1_f0_dg \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
python scripts/run_figure3_iteration.py \
    --run-id 03b_original_modulation_index \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric mod_idx_dg \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
```

The formulas match the installed AllenSDK source, the all-session import is
fully hashed, and the independent Pilot tuning preferences agree at 97.0% on
average. A whole-session original Allen NWB gold-standard run remains
outstanding because no original Allen NWB is available on the local mounts.
See [`ITERATION3_SUMMARY.md`](artifacts/figure3/ITERATION3_SUMMARY.md) for the
metric deltas, tie sensitivity, statistical results, and claim status.

### Iteration 4 — Harmonize unit populations and add a preliminary RF filter

**Location:** RF significance/area method developed and validated in
PilotAnalysis; import, masks, comparison, and figures here.

**Work in PilotAnalysis:**

- Implement a defensible significance test for the Gabor RF data, preferably a
  trial-label permutation matched to the stimulus structure.
- Define RF area in degrees squared and benchmark it against the Allen method on
  compatible data.
- Export per-unit p-value, area, center, method, and gaze status for all eight
  sessions.

**Work here:**

- Define named population masks rather than embedding filters inside plots:
  `published_like`, `default_qc`, `intersection`, and diagnostic alternatives.
- Apply the same declared mask consistently to every figure and statistical
  script.
- Restore RF area as a fourth metric when equivalence is adequate.
- Produce a flow diagram/table of unit counts at each filtering step.

**Figure checkpoints:**

- `04a_common_qc` — consistent `default_qc` comparison.
- `04b_published_like` — closest achievable published population.
- `04c_rf_area` — restored RF-area comparison.

The primary claim should not be updated from this iteration until filter
coverage and session balance are reviewed.

**Iteration 4A completed:** both grating views can be regenerated with:

```bash
python scripts/run_figure3_iteration.py \
    --run-id 04a_common_qc \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric mod_idx_dg \
    --population-profile common_qc \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
python scripts/run_figure3_iteration.py \
    --run-id 04a_common_qc_f1f0_sensitivity \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric f1_f0_dg \
    --population-profile common_qc \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
```

The common-QC checkpoint includes a population flow, per-area and per-session ×
probe counts, and both metric sensitivities. `04b_published_like` and
`04c_rf_area` were not approximated: they are explicitly blocked on validated
all-session RF significance and area fields from PilotAnalysis. See
[`ITERATION4A_SUMMARY.md`](artifacts/figure3/ITERATION4A_SUMMARY.md) and the
cross-repository [`RF_FILTER_BLOCKER.md`](data/imports/RF_FILTER_BLOCKER.md).

### Iteration 5 — Validate flash timing and timescale accommodations

**Location:** timing provenance may require acquisition/PilotAnalysis work;
paper metrics and sensitivity figures remain here.

**Work:**

- Determine whether NWB `start_time` values are command, frame, photodiode, or
  display-corrected times.
- Locate or derive an independent display-latency calibration. Do not estimate
  the final correction by forcing MouseV2 to equal Allen V1.
- Keep raw TTFS as the invariant primary within-MouseV2 comparison and show
  calibrated absolute TTFS only when the correction is independently known.
- Compare bright, dark, and pooled flash results.
- Verify 1-ms TTFS bin edges and the 10-ms, 40–290-ms timescale window against
  the original analysis on Allen data.
- Confirm that timescale differences are not driven by flash polarity,
  fit failures, or spike-count thresholds.

**Figure checkpoint:** `05_flash_validated`, containing raw and calibrated TTFS
where available plus flash-polarity sensitivity panels.

**Iteration 5 completed provisionally:** the raw per-unit import and three full
figure checkpoints can be regenerated with:

```bash
python scripts/extract_mousev2_flash_metrics.py
python scripts/run_figure3_iteration.py \
    --run-id 05a_flash_pooled \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric mod_idx_dg \
    --flash-metrics-dir data/imports/mousev2_flash_metrics_v1 \
    --flash-variant pooled --ttfs-display raw_nwb \
    --population-profile common_qc \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
```

Repeat the figure command with `--flash-variant bright` and `dark` for the two
sensitivity checkpoints. The acquisition-timestamp audit is complete, but a
calibrated TTFS view was deliberately not manufactured from neural-response
mean matching. See [`ITERATION5_SUMMARY.md`](artifacts/figure3/ITERATION5_SUMMARY.md)
and [`TIMING_CALIBRATION_BLOCKER.md`](data/imports/mousev2_flash_metrics_v1/TIMING_CALIBRATION_BLOCKER.md).

### Iteration 6 — Represent known within-V1 anatomy without inventing a hierarchy score

**Location:** existing anatomical localization and PilotAnalysis provide the
within-V1 locations; this repository defines and tests their paper-facing
representation.

**Work:**

- Document the existing evidence that all four recording locations are in V1
  and identify the anatomical V1 subregion sampled by each probe.
- Keep anatomical V1 location separate from hierarchy score: do not place
  probe locations on the published inter-area hierarchy axis without an
  independently justified mapping.
- Test measured RF azimuth/elevation, anatomical coordinates, and categorical
  probe identity as complementary within-V1 position representations.
- If a within-V1 hierarchy score is proposed, define it from independent
  connectivity/anatomical evidence and validate its direction and uncertainty;
  do not derive it from the response metrics being tested.
- Add cortical layer as a covariate or sensitivity analysis where available,
  rather than treating it as evidence for V1 identity.

**Figure checkpoint:** `06_within_v1_location`, with the known anatomical
locations, RF coordinates, and any independently supported hierarchy mapping
shown as distinct quantities.

**Iteration 6A completed:** the current paper-facing accommodation can be
regenerated with:

```bash
python scripts/run_figure3_iteration.py \
    --run-id 06a_known_v1_locations \
    --grating-metrics-dir data/imports/mousev2_grating_metrics_v1 \
    --grating-metric mod_idx_dg \
    --flash-metrics-dir data/imports/mousev2_flash_metrics_v1 \
    --flash-variant pooled --ttfs-display raw_nwb \
    --within-v1-x-mode display_only \
    --population-profile common_qc \
    --rf-import-dir data/imports/pilot_rf_peaks_v1
```

The checkpoint does not invent coordinates absent from the current inputs. An
optional exact anatomical-coordinate export is specified in
[`WITHIN_V1_LOCATION_AUDIT.md`](data/imports/WITHIN_V1_LOCATION_AUDIT.md).
See [`ITERATION6_SUMMARY.md`](artifacts/figure3/ITERATION6_SUMMARY.md) for the
unchanged effect sizes and updated fairness assessment.

### Iteration 6B — Reconcile the Allen and MouseV2 V1 metric levels

**Location:** matched response extraction and final diagnostics here; exact
anatomical layer/RF exports mature in PilotAnalysis; acquisition definitions
remain in `openscope_v2species`.

**Why this gates the claim:** the Allen V1 reference and MouseV2 V1 values have
different absolute centers. This is not explained by pooled-unit versus
equal-session aggregation. The modulation discrepancy is metric-specific:
equal-session log10 `mod_idx_dg` is −0.107 for MouseV2, +0.040 for Allen Brain
Observatory, and +0.199 for Allen Functional Connectivity, whereas MouseV2
log10 F1/F0 is +0.086 above Allen Brain Observatory. Valid pooled-flash
timescale is 47.53 ms for MouseV2 and 43.88 ms for Allen Brain Observatory.
After matching MouseV2 to Allen's balanced 150-flash support, the MouseV2
timescale center is 45.92 ms; trial count therefore explains about 1.61 ms of
the original 3.65-ms offset, leaving about 2.04 ms.

**Identified protocol/implementation differences:**

- Allen drifting gratings use a 2-s PSTH; MouseV2 uses 1 s.
- Allen's requested Welch `nperseg=1024` is retained, whereas SciPy reduces it
  to 1,000 samples for MouseV2.
- The released `np.searchsorted` lookup evaluates the next-higher Welch bin in
  Allen (for example, requested 2 Hz maps to 2.930 Hz), but lands exactly on
  the requested TF in MouseV2. Running identical source code therefore does
  not define an identical spectral measurement.
- Allen V1 combines Brain Observatory and higher-repeat Functional Connectivity
  stimulus sets; the latter has a much higher `mod_idx_dg` center but almost
  the same F1/F0 center.
- Allen uses fixed SF = 0.04 cycles/degree; MouseV2 varies SF and chooses the
  preferred orientation × TF × SF condition.
- MouseV2's currently exported `firing_rate_dg` is preferred-condition rate,
  while the released Allen field is overall block rate. It is not a homologous
  population-matching variable.

**Required raw bridge:**

1. **MouseV2 leg completed:** all eight sessions were recomputed at SF = 0.04
   cycles/degree on the shared 1-s/15-trial condition support. The modulation
   center changed by only +0.009 log units; see
   [`mousev2_grating_common_support_v1`](data/imports/mousev2_grating_common_support_v1/README.md).
2. **Raw Allen reproduction completed for the claim-gate population:** four
   checksum-verified NWBs (a representative and an independent reproduction
   control per cohort) reproduce released common-QC F1/F0 to at most 2e-9 and
   modulation index to at most 1.5e-7; see
   [`allen_v1_raw_bridge_v2`](data/imports/allen_v1_raw_bridge_v2/README.md).
3. **Representative matched-window diagnostic completed:** Allen was
   recomputed on the first 1 s, 15 trials per condition, the
   shared orientation/TF support, SF = 0.04 cycles/degree, and contrast = 0.8.
   The modulation gap remains −0.186 log10 versus Brain Observatory and −0.220
   versus Functional Connectivity, while the F1/F0 gap shrinks to −0.032 and
   −0.035. Functional Connectivity trial subsampling is propagated, but more
   Allen sessions are still required for a population coefficient.
4. **Harmonized spectral diagnostic completed:** the common 1-s grid and exact
   TF evaluation are primary bridge views; released `mod_idx_dg` remains a
   historical sensitivity and F1/F0 is the co-primary diagnostic.
5. **Flash trial/polarity match completed:** ten balanced trial draws per
   MouseV2 session reduce 150 bright + 150 dark flashes to Allen's 75 + 75.
   The timescale center moves from 47.53 to 45.92 ms and the mean validity
   fraction from 0.207 to 0.168; see
   [`mousev2_timescale_trial_bridge_v1`](data/imports/mousev2_timescale_trial_bridge_v1/README.md).
6. **Phase-coherence mechanism diagnostic completed:** MouseV2 has comparable
   or higher single-trial F1 amplitude but lower weighted phase coherence
   (0.387 versus 0.539/0.516 in representative Allen BO/FC). It loses −0.494
   log10 amplitude during trial averaging versus −0.321/−0.345 in Allen; see
   [`v1_grating_phase_bridge_v1`](data/imports/v1_grating_phase_bridge_v1/README.md).
7. **Acquisition-source start-phase diagnostic completed:** camstim advances
   phase as `TF * current_frame / fps` without resetting at presentation onset.
   MouseV2's 135-frame stimulus-plus-blank stride therefore varies starting
   phase at 1, 2, and 15 Hz but not 4 or 8 Hz. Source-derived adjustment raises
   equal-session coherence from 0.387 to 0.433, while leaving residual gaps of
   0.106/0.083 versus representative Allen BO/FC; see
   [`mousev2_grating_start_phase_bridge_v1`](data/imports/mousev2_grating_start_phase_bridge_v1/README.md).
8. **Residual shared-phase/behavior diagnostic completed:** matched-trial
   residual phase is weakly shared across probes (alignment 0.173 versus 0.141
   shuffled; 1,000-permutation equal-session p = 0.001), but other-probe
   adjustment changes coherence from 0.433 to 0.429 rather than closing the
   residual Allen gap. After condition, 50% valid-eye-coverage, and
   linear/quadratic block-time control, residual population phase covaries with
   running and pupil x/y, but not pupil area. These descriptive associations
   identify candidate state variables without establishing causality; see
   [`mousev2_grating_shared_phase_behavior_v1`](data/imports/mousev2_grating_shared_phase_behavior_v1/README.md).
9. **Source-corrected Welch diagnostic completed:** rotating only the
   source-phase-dependent target-frequency component and preserving the mean
   and all non-carrier PSTH structure raises equal-session log10 modulation
   index from −0.098 to +0.019. All eight sessions increase (exact two-sided
   sign-test p = 0.0078); phase permutation gives −0.123, the opposite-sign
   rotation gives −0.087, and the predicted phase-stable 4/8-Hz control is
   unchanged. The remaining representative Allen gaps are −0.069/−0.104 log10;
   see [`mousev2_grating_corrected_welch_bridge_v1`](data/imports/mousev2_grating_corrected_welch_bridge_v1/README.md).
10. Match homologous waveform QC, validated RF significance/area, normalized
   cortical depth/layer, and a genuinely homologous firing-rate field.

**Pass criterion:** the dataset coefficient and the areas-minus-probes effect
size contrast must remain stable across released, common-window,
common-condition, common-trial-count, and common-population views. Otherwise,
the paper must restrict inference to within-dataset effects and cannot use the
absolute Allen V1 point as a calibrated reference.

**Current decision:** this criterion has not passed for modulation index. A
source-defined start-phase mismatch explains a substantial and reproducible
part of the gap when propagated through the unchanged Welch estimator, but the
corrected MouseV2 center remains below both representative Allen sessions. The
residual shared-phase analysis finds a real but weak cross-probe signal that
does not provide an additional repair. Until the Allen raw bridge is expanded
across sessions and homologous RF/layer/population support is applied, the
defensible claim is restricted to within-dataset spatial effects; the absolute
Allen V1 point is context, not a calibrated baseline.

**Completed diagnostic checkpoint:**

```bash
python scripts/v1_dataset_bridge.py
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/extract_mousev2_grating_common_support.py --skip-figure
python scripts/extract_mousev2_grating_common_support.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/extract_allen_v1_bridge.py --skip-figure
python scripts/extract_allen_v1_bridge.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/mousev2_timescale_trial_bridge.py --skip-figure
python scripts/mousev2_timescale_trial_bridge.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/v1_grating_phase_bridge.py --skip-figure
python scripts/v1_grating_phase_bridge.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/mousev2_grating_start_phase_bridge.py --skip-figure
python scripts/mousev2_grating_start_phase_bridge.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/mousev2_grating_shared_phase_behavior.py --skip-figure
python scripts/mousev2_grating_shared_phase_behavior.py --render-existing
PYTHONNOUSERSITE=1 \
  /home/huklaban5/anaconda3/envs/neuropixels_platform_paper_py310/bin/python \
  scripts/mousev2_grating_corrected_welch_bridge.py
```

This writes the session-level figure, center/contrast tables, Welch frequency
audit, controlled protocol simulation, timescale selection flow, and the
claim-gate report to
[`06b_v1_dataset_bridge`](artifacts/figure3/06b_v1_dataset_bridge/V1_DATASET_BRIDGE.md).
The second and third commands write the raw MouseV2 common-support import and
its figure to
[`mousev2_grating_common_support_v1`](data/imports/mousev2_grating_common_support_v1/README.md).
The Allen and timescale commands write versioned raw diagnostic figures and
tables to [`allen_v1_raw_bridge_v2`](data/imports/allen_v1_raw_bridge_v2/README.md)
and [`mousev2_timescale_trial_bridge_v1`](data/imports/mousev2_timescale_trial_bridge_v1/README.md).
The phase command writes its diagnostic figure, TF-stratified table, and
descriptive rate/TF-adjusted model to
[`v1_grating_phase_bridge_v1`](data/imports/v1_grating_phase_bridge_v1/README.md).
The start-phase command writes the source-derived schedule audit, unit-level
phase adjustment, permutation controls, and residual-gap figure to
[`mousev2_grating_start_phase_bridge_v1`](data/imports/mousev2_grating_start_phase_bridge_v1/README.md).
The shared-phase/behavior command writes cross-probe leave-one-trial-out phase
controls, condition-stratified behavior permutations, and the time-controlled
diagnostic to
[`mousev2_grating_shared_phase_behavior_v1`](data/imports/mousev2_grating_shared_phase_behavior_v1/README.md).
The corrected-Welch command writes unit/session/TF tables, paired-session
controls, the target-power/denominator decomposition, and the residual-gap
figure to
[`mousev2_grating_corrected_welch_bridge_v1`](data/imports/mousev2_grating_corrected_welch_bridge_v1/README.md).

### Iteration 6C — Verify achieved Allen retinotopic matching and define the RF-adjusted comparison

**Location:** this repository, using the public Allen unit metrics already
represented in `data/unit_table.csv`; validated MouseV2 RF exports continue to
mature in PilotAnalysis and enter through the versioned import contract.

**Why this gates the comparison:** Allen did not aim probes at the geometric
centers of V1, LM, AL, AM, and PM. The insertion plan used ISI-derived altitude
and azimuth maps to target a common V1-aligned retinotopic region. RL was an
explicit exception because its retinotopic center often lies near the RL–S1
boundary. This establishes a careful intended-targeting control, but it does
not establish that the units ultimately recorded in each area had closely
matched RF centers. Probe trajectory, local retinotopic gradients, RF scatter,
eye position, unit selection, and the coarse 20° Gabor/10° grid can all widen or
shift the achieved distributions.

The original paper measured per-unit RF center, RF area, and RF significance,
and showed area-level mean RF outlines and RF-area distributions. It did not
report the session-level distribution of achieved RF centers, paired HVA–V1
offsets, or targeting-error covariance. Those quantities are required before
the Allen areas can be described as an RF-matched comparison.

**Targeting audit:**

1. Record the targeting protocol and its exception without converting intended
   coordinates into achieved neural coordinates: V1, LM, AL, AM, and PM share
   the V1-aligned target rule; RL is predeclared as a separate sensitivity.
2. Reproduce the published RF-quality population from explicit named masks.
   At minimum, report the effects of `p_value_rf < 0.01`, `area_rf < 2500`,
   `snr > 1`, and `firing_rate_dg > 0.1`, plus an RF-only view that does not
   condition on the response metric under study.
3. Summarize `azimuth_rf`, `elevation_rf`, and `area_rf` at four levels: unit,
   probe, session × area, and area. Report valid-unit counts and screen-edge
   truncation/`on_screen_rf` sensitivity where available.
4. For every Allen session containing V1 and an HVA, compute the paired
   difference between robust session × area RF centers. Report signed azimuth
   and elevation offsets, Euclidean distance, covariance, and uncertainty—not
   only pooled unit distributions.
5. Separate three concepts throughout: intended cortical target, dispersion of
   achieved RF centers, and size of individual RFs. Larger RF area must not be
   misdescribed as less accurate targeting.
6. Visualize RF-center common support by area and stimulus set. Identify areas
   and sessions for which V1 matching would require extrapolation rather than
   interpolation.

**RF-adjusted response analysis:**

- For TTFS, grating modulation, and response-decay timescale, fit a
  session-aware model containing visual area, RF azimuth, RF elevation, and a
  prespecified flexible spatial term where supported. Include RF area,
  normalized cortical depth/layer, stimulus set, and homologous population
  variables as covariates or declared sensitivities rather than silently
  changing the primary population.
- Preserve the correlation among areas recorded in the same Allen session and
  among probes recorded in the same MouseV2 session. Mouse/specimen and session
  are grouping variables; the exact random-effects structure must be selected
  before inspecting the area coefficient.
- Test area × RF-position interactions. A common RF gradient across areas is a
  stronger assumption than simply adjusting for RF position and must be checked.
- For RF area as an outcome, adjust for RF-center coordinates but do not include
  `area_rf` as a predictor of itself.
- Estimate whether the visual-area effect and its association with anatomical
  hierarchy remain after RF adjustment. Report the adjusted coefficient and
  uncertainty alongside the unadjusted result; do not interpret coefficient
  shrinkage as all-or-none confounding.
- Add an assumption-light sensitivity that matches or weights V1 observations
  to each HVA's achieved two-dimensional RF-center distribution on common
  support. Matching is performed within stimulus set and, where possible,
  within session; balance diagnostics and discarded support are reported.

**V1 reference interpretations:**

| Comparison | Role |
| --- | --- |
| Repeated sampling near one central V1 location | Lower-bound null for repeated measurements at a nominally fixed retinotopic location |
| MouseV2 locations spanning V1 | Estimate within-area retinotopic gradients and the range available for common-support matching |
| V1 matched/weighted to each HVA's achieved RF centers | Primary retinotopically fair comparison of area identity beyond sampled visual-field location |
| Unadjusted Allen area comparison | Historical reproduction and sensitivity, not evidence of RF matching |

**Outputs:**

- a targeting-protocol note with the RL exception;
- unit- and session-level RF-center coverage/count tables;
- paired HVA–V1 offset and common-support tables;
- an RF-center scatter/ellipse figure separated from the RF-area panel;
- unadjusted, covariate-adjusted, and matched/weighted response estimates; and
- a short claim-gate report stating whether the hierarchy result survives
  achieved-RF adjustment.

**Figure checkpoint:** `06c_allen_rf_matching`. This checkpoint is descriptive
and model-defining until the MouseV2 RF export is validated. It may establish
that Allen's achieved samples were close, dispersed, or systematically offset;
none of those outcomes is assumed in advance.

**Targeting audit completed:** regenerate the versioned summaries and figure
with:

```bash
python scripts/allen_rf_matching.py --overwrite
```

The checkpoint uses the `published_like` population by default, refuses to
overwrite a non-empty artifact directory without `--overwrite`, records the
Allen table checksum, and writes probe/session-area summaries, paired offsets,
session-centered common-support flags, population flows, population-mask
sensitivity, and the diagnostic figure. Focused tests cover nested population
masks, within-session pairing, and convex-hull support classification. See
[`ALLEN_RF_MATCHING.md`](artifacts/figure3/06c_allen_rf_matching/ALLEN_RF_MATCHING.md).

**First response-adjustment checkpoint completed:** regenerate the
session-fixed-effect models, clustered intervals, matched session contrasts,
caliper sensitivities, and figure with:

```bash
python scripts/allen_rf_adjusted_response.py --overwrite
```

The primary adjusted model is restricted to the conservative session-centered
V1 support box and uses flexible azimuth/elevation terms. The matching
sensitivity uses same-session nearest-neighbor V1 RF centers with replacement
and a 10° caliper. It reduces mean RF-coordinate imbalance but discards part of
the eligible HVA population; stricter calipers improve balance at the cost of
substantial attrition and sparse session support. See
[`ALLEN_RF_ADJUSTED_RESPONSE.md`](artifacts/figure3/06c_allen_rf_matching/response_adjustment/ALLEN_RF_ADJUSTED_RESPONSE.md).

**Nonlinear SF/TF preference checkpoint completed:** regenerate the
session-balanced preference surfaces, paired-session V1 differences, bandwidth
sensitivities, and support diagnostics with:

```bash
python scripts/allen_frequency_preference_surfaces.py --overwrite
```

The checkpoint is restricted to Brain Observatory sessions, where multiple SF
and TF values were presented. It smooths the released preferred bins on a log2
scale and does not claim to reconstruct response-amplitude tuning curves. See
[`ALLEN_FREQUENCY_PREFERENCE_SURFACES.md`](artifacts/figure3/06c_allen_rf_matching/frequency_preference_surfaces/ALLEN_FREQUENCY_PREFERENCE_SURFACES.md).

A separate tuning-enriched sensitivity excludes low-selectivity and ambiguous
preferences using metric-specific lifetime sparseness > 0.1, stimulus firing
rate > 0.1 Hz, and a unique released preferred bin. It is versioned at
[`frequency_preference_surfaces_tuning_enriched`](artifacts/figure3/06c_allen_rf_matching/frequency_preference_surfaces_tuning_enriched/ALLEN_FREQUENCY_PREFERENCE_SURFACES.md); the inclusive checkpoint remains unchanged for comparison. These criteria enrich for condition selectivity but are not substitutes for unavailable SF-/TF-specific curve-fit significance.

**Pass criterion:** the primary area/hierarchy conclusion is stable on RF common
support, paired Allen session structure is preserved, and balance diagnostics
show that the adjusted estimate is not driven by extrapolation. If this fails,
the paper reports RF-dependent and residual area effects separately rather than
describing the Allen comparison as retinotopically matched.

### Iteration 6D — Map MouseV2 frequency preference over multi-probe V1 retinotopy

The analysis returns to all 1,500 drifting-grating presentations per session.
For every independently Pilot-QC-selected unit it fits a joint Poisson model
with log-Gaussian SF, log-Gaussian TF, and orientation-periodic von Mises terms
over all 100 conditions. The empirical orientation-marginal 5 x 5 surface is
retained as a diagnostic. It also fits a Poisson baseline-plus-rotated-
elliptical-Gaussian RF over all 4,860 RF presentations and 81 positions.

Dataset-wide Benjamini-Hochberg FDR, split-half reliability, pseudo-R2 >= 0.1,
and parameter-identifiability checks gate both model families. SF/TF preferences
also require the corresponding nonparametric axis-specific test. Peaks may be
estimated up to one octave beyond the sampled range from the observed flank;
only peaks or widths pinned to the wider optimizer bounds are treated as
unidentified and excluded. Extrapolated preferences are explicitly flagged.
Regenerate the parametric tuning and RF fits, then the maps, with:

```bash
python scripts/extract_mousev2_frequency_tuning.py --overwrite
python scripts/extract_mousev2_parametric_rf.py --overwrite
python scripts/mousev2_frequency_preference_surfaces.py --overwrite
```

Of 4,807 independently selected Pilot-QC units, 1,110 have supported parametric
RF models, 2,634 have supported SF preferences, and 1,636 have supported TF
preferences. Their intersections yield 843 SF-mapped and 528 TF-mapped units
across all eight sessions. Of those mapped preferences, 74 SF and 117 TF peaks
are extrapolated beyond the sampled range and explicitly flagged. The remaining positional limitation is the lack of
gaze correction: fitted centers are display-centered rather than eye-centered.
The result must not yet be described as a gaze-calibrated Allen–MouseV2 surface
contrast. See
[`MOUSEV2_FREQUENCY_PREFERENCE_SURFACES.md`](artifacts/figure3/06d_mousev2_frequency_preference_surfaces/MOUSEV2_FREQUENCY_PREFERENCE_SURFACES.md).

The descriptive MouseV2-minus-Allen preference offset is larger for SF
(median 1.35x) than TF (1.07x). Candidate explanations include stimulus and
estimator differences, RF/unit sampling, and retained identifiable extrapolated
MouseV2 peaks; these are recorded as hypotheses rather than resolved effects.
The next within-MouseV2 sensitivity view restricts to complete simultaneous
A/B/C/E probe quartets, estimates RF density and SF/TF surfaces from their
independent supported populations, weights sessions equally, and removes only
a shared per-session RF translation. This alignment is not gaze correction.
Before that aligned probe-specific view, an intermediate map pools A/B/C/E
within each session and keeps the original display coordinates. It directly
shows the RF, SF, and TF fields under the screen and unmeasured eye-position
state shared by the four simultaneously recorded probes.

The corresponding Allen Brain Observatory 1.1 session atlas compares each V1
sample with the simultaneously recorded pooled HVAs. RF, static-grating SF, and
drifting-grating TF blocks remain separate in time, but all probes recorded
simultaneously within each block. Area-specific HVA grids are retained even
when the displayed HVA row is pooled for support.

An RF-only global-affine sensitivity does not provide a defensible correction.
Although density/evidence-weighted SF/TF RMSE decreases, leave-one-session-out
pattern correlations worsen for both V1 maps, remain approximately unchanged
for pooled-HVA SF, and improve only for pooled-HVA TF. RF-optimal transforms
often compress the field severely (median determinant about 0.32, five reflected
sessions), so a physically constrained rigid or limited-scale model is required
before aligned maps can be treated as primary.

More fundamentally, that diagnostic used area-wise RF-center consensus as the
registration landmark, which assumes the retinotopic correspondence being
tested. It is therefore retained only as a rejected failure mode. Any revised
registration must fit transforms from non-center scalar fields such as RF size,
CCF coordinates, cortical depth/probe position, and RF response properties,
then evaluate SF/TF out of sample. After computing Allen eccentricity from its
released `(0°, 0°)` origin, RF area is only weakly radial (rho = -0.069; 55%
gradient-sign agreement). The stronger landmarks are dorsal–ventral CCF position,
probe position, and cortical depth (maximum |rho| = 0.116–0.172). Flash latency
and modulation metrics are weaker still (generally |rho| < 0.06). Identity and low-dimensional
transforms must therefore remain explicit competitors, and failure to identify
a transform is an acceptable result.

The first non-center registration pilot combines log RF area, dorsal–ventral
CCF position, probe-horizontal position, RF response time-to-peak, and flash
first-spike latency without consulting SF or TF. Translation-only and tightly
bounded similarity models were fitted to V1 and pooled-HVA feature surfaces.
With corrected Allen-origin feature weights, similarity passes the predeclared
selection threshold: its median regularized objective improves by 0.026 versus
translation, exceeding the required 0.020. The selected similarity gives
independent median paired tuning-correlation changes of V1 SF +0.031, V1 TF
-0.116, pooled-HVA SF -0.170, and pooled-HVA TF -0.013. These landmarks therefore
do not reproduce the tuning-fitted SF gain;
the identity transform remains a serious competitor.

That heterogeneous RF-size/latency feature model is now superseded as the
fourth-row visualization by a direct anatomical retinotopy pilot. In the 23 of
31 simultaneous sessions with reconstructed V1 CCF coordinates, a robust,
session-balanced CCF→V1-RF mapping is trained while holding out the entire
target session. Linear and quadratic mappings are compared using RF prediction
only; the linear model has slightly lower median centered session error (10.81°
versus 10.95°). The held-out session's robust median prediction residual then
defines one translation, bounded at ±15°, that is shared by its simultaneous
V1 and HVA maps. RF size and HVA units do not enter this fit. Five of 23 sessions
reach a bound, so the anatomical registration remains a sensitivity analysis.
On the same 23-session support, its independent median SF/TF correlation changes
are V1 SF -0.087, V1 TF +0.033, pooled-HVA SF -0.019, and pooled-HVA TF -0.057.
Thus CCF explains a modest amount of within-session V1 RF structure but does not
produce broad SF/TF agreement; the fourth row is diagnostic rather than a
preferred correction.

The corrected Allen-origin RF-area diagnostic shows a weak, non-monotonic radial
pattern rather than a strong eccentricity gradient. Area falls at the far mapped
periphery, which may reflect finite-display truncation of partially off-screen
RFs and consequent underestimation of released RF area. V1 spatial support is also narrow (median
99 units/session and 29% supported grid) relative to pooled HVA (247 units and
88%). RF area can remain an exploratory scalar registration field, but it is
not an uncensored biological calibration of RF size versus eccentricity.

Distance from the nearest RF-grid boundary exposes an area-specific pattern:
V1 has a negative within-session association with RF area (rho = -0.168; 81%
gradient-sign agreement), whereas pooled HVAs are strongly positive (rho =
+0.537; 94%). LM, RL, AL, PM, and AM are all positive individually. A single
screen-edge correction is therefore inappropriate. CCF coordinates remain
valuable for the separate anatomical question—mapping reconstructed V1 tissue
location onto achieved RF location—even though they should not be described as
a direct estimate of eye displacement.

As an explicitly in-sample visibility pilot, a separate orientation-preserving
limited affine was fit directly to the four stacked tuning maps (V1/HVA ×
SF/TF), with equal map weight and local-density evidence. Median paired spatial
correlation increases for V1 SF (+0.178), pooled-HVA SF (+0.360), and pooled-HVA
TF (+0.086), but decreases for V1 TF (-0.081). Several sessions reach transform
bounds. This suggests some potentially alignable SF structure but not a single
uniformly improved SF/TF map; it is exploratory rather than validation.

A tuning-quality-weighted sensitivity now gives each eligible unit a tempered
combination of lifetime-sparseness tuning strength, saturating stimulus response
rate, and inverse Fano factor. The Fano term is explicitly a trial-variability
proxy, not split-half tuning reliability. Weights are clipped and renormalized
within session × area group × SF/TF, and spatial support uses weighted Kish
effective unit count. Under the same affine settings, median paired correlation
changes are V1 SF +0.080, V1 TF +0.014, pooled-HVA SF +0.333, and pooled-HVA TF
+0.044. Thus the previous negative V1 TF result becomes slightly positive, but
the V1 SF gain is less than half as large. The alignment signal is sensitive to
unit-quality weighting and remains an exploratory visibility result.

For direct visualization, the quality-weighted Allen stacks now include an
intermediate V1-RF-center registration. Each session's median supported V1 RF
center is translated to the cross-session median V1 center, and the same
translation is applied to all simultaneous V1 and HVA SF/TF maps. This middle
row permits no rotation, scale, or shear and does not consult tuning. Relative
to raw coordinates, its median paired correlation changes are V1 SF +0.017,
V1 TF +0.021, pooled-HVA SF -0.035, and pooled-HVA TF -0.028. Center alignment
alone therefore provides little general improvement and slightly worsens both
pooled-HVA tuning maps.

The displayed four-row comparison is now restricted to the 23 CCF-available
sessions for all rows. On that common subset, V1-RF-center changes are V1 SF
+0.022, V1 TF +0.057, pooled-HVA SF -0.003, and pooled-HVA TF -0.039.

### Iteration 7 — Finalize the inferential comparison

**Location:** this repository.

**Work:**

- Preserve session × group aggregation.
- Replace independent within-group resampling with a session-block bootstrap or
  a hierarchical/mixed-effects model that preserves within-session probe
  dependence.
- Predeclare the primary effect size and contrast.
- Report uncertainty on the difference between area and probe effects, not only
  separate confidence intervals.
- Run cortical-area-only results with LP excluded, then show LP-inclusive results
  separately.
- Use the Iteration 6C model specification and common-support diagnostics;
  retain categorical probe/area estimates as unadjusted historical
  reproductions rather than the sole primary comparison.
- Report the eight-session power limitation explicitly; distinguish “no
  evidence of a larger area effect” from evidence of equivalence.
- Correct figure captions so their named tests and multiple-comparison methods
  match the code.

**Figure checkpoint:** `07_primary_statistics`. This becomes the candidate
paper figure set and the basis for claim wording.

Iteration 7 must not become paper-facing until the Iteration 6B and 6C pass
criteria are met or the analysis is explicitly reframed as within-dataset-only.

### Iteration 8 — Revisit CCG/feedforwardness as a separate analysis

**Location:** original CCG method and final result here; PilotAnalysis may be
used for performance experiments and diagnostics.

**Work:**

- Restore or validate the original 25-ms jitter predictor rather than labeling
  a PSTH predictor as jitter.
- Include spatial frequency in stimulus-condition construction.
- Reproduce the original significant/nonzero peak-selection rule; do not assign
  a direction to every noisy pair.
- Validate lag sign, boundary handling, firing-rate selection, and peak window on
  synthetic spike trains and an original Allen session.
- Run one MouseV2 session, inspect pair-level CCGs, then run all eight.
- Aggregate and infer at the session level.

**Figure checkpoint:** `08_ccg_secondary`. CCG should strengthen or qualify the
main response-property analysis, not block it.

### Iteration 9 — Freeze the paper-facing pipeline

**Location:** this repository, consuming versioned exports from neighbors.

**Work:**

- Lock the modern environment.
- Run the automated equivalence and schema tests from a clean checkout.
- Ensure the all-eight-site unit-table path is current; the existing helper only
  enumerates sites 2–5.
- Generate the final figures, tables, methods, captions, exclusions, and
  provenance manifest with one command.
- Archive the exact PilotAnalysis export and stimulus manifest used.
- Separate primary, sensitivity, exploratory, and failed/obsolete iterations.

**Figure checkpoint:** `09_paper_candidate`, promoted to canonical filenames
only after scientific review.

## Figure set regenerated at every main iteration

At minimum, every iteration that changes data, filtering, or statistics should
regenerate:

1. the full hierarchy overlay;
2. the probe/session zoom;
3. the within-V1 versus post-V1 split comparison;
4. the session-level effect-size/statistical report;
5. a diagnostic page with unit counts, missingness, per-session metric means,
   and change from the preceding iteration.

During development, run the existing entry points from the repository root:

```bash
python Figure3/Figure3_with_V1sites.py
python Figure3/Figure3_probe_zoom.py
python Figure3/Figure3_split_comparison.py
python scripts/eta_squared_comparison.py
```

The Iteration 1 runner now replaces those separate calls for reviewable runs,
while leaving every underlying script independently runnable:

```bash
python scripts/run_figure3_iteration.py --run-id <iteration_name>
```

The runner refuses to overwrite a non-empty iteration directory, runs the
schema/coverage tests, captures logs and provenance, and compares the result to
the configured baseline.

## Decision log required before the paper claim

Record explicit decisions for:

- F1/F0 versus the original modulation index as the primary grating metric;
- the full-condition definition for MouseV2 gratings;
- raw versus independently calibrated TTFS;
- pooled versus polarity-specific flash trials;
- RF significance and area definitions;
- primary unit population/QC mask;
- categorical probe versus continuous RF position;
- the primary form of the RF-position adjustment, common-support rule, and
  treatment of areas/sessions that require extrapolation;
- whether RL is excluded from the primary matched-target analysis or retained
  only as a predeclared targeting-exception sensitivity;
- inclusion or exclusion of LP from “post-V1 cortical areas”;
- primary statistical model and resampling unit;
- whether CCG is primary, supporting, or exploratory evidence.

## Definition of done

The main analysis is ready for a strong claim when:

- all paper-facing metrics either reproduce AllenSDK on reference data or have a
  documented, tested accommodation for a known protocol difference;
- preferred grating conditions include all varying dimensions;
- RF position and RF quality have quantified coverage, and known within-V1
  anatomy is represented without conflating location with hierarchy score;
- intended Allen targeting is distinguished from achieved RF-center matching,
  paired HVA–V1 offsets and common support are reported, and the primary area
  effect is evaluated after RF adjustment;
- every figure and model uses an explicit, consistent population definition;
- session dependence is preserved in inference;
- raw and calibrated quantities are never conflated;
- the conclusion is stable across the declared sensitivity analyses;
- one clean command reproduces the paper candidate with provenance.

Until then, each iteration remains scientifically useful. The appropriate
interim conclusion is that the current data provide a working estimate of
within-V1 variation and identify which methodological choices materially affect
its comparison with post-V1 area variation.
