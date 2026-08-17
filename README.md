# Neuropixels Platform Paper — Retinotopic V1 Analysis

This repository contains the figure-generation code from the Allen Institute
Neuropixels platform paper and an ongoing extension that compares response
properties across retinotopic recording locations within mouse V1.

The active analysis asks whether variation between higher visual areas exceeds
the variation observed across spatially separated probes within V1, after
accounting for the visual-field locations actually sampled in both datasets.
Allen targeted V1 and most higher visual areas near a common retinotopic
location, but intended targeting is not treated as evidence that the achieved
unit populations were RF-matched. The public Allen RF centers will therefore
be audited and incorporated into the primary model and matching sensitivities.
The three response metrics currently compared are:

- time to first spike (TTFS) after a flash;
- drifting-grating modulation, using released log10 `mod_idx_dg` with log10
  F1/F0 as a co-primary bridge diagnostic; and
- flash-evoked response decay timescale.

The scientific motivation is described in [problemstatement.md](problemstatement.md).
The original dataset and paper are described by the
[Allen Visual Coding Neuropixels resource](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels).

> **Current claim status:** exploratory and within-dataset only. Raw bridge
> analyses show that matching grating duration, condition support, and trial
> count does not remove the Allen–multi-site V1 modulation-index gap, although it
> nearly closes the F1/F0 gap. The frozen *MouseV2* project acquisition source further
> shows that unreset grating start phase causes a material part of the
> trial-average coherence loss. Carrying that source-defined correction through
> the unchanged Welch estimator raises the multi-site V1 equal-session center from
> −0.098 to +0.019 log10 in all eight sessions, but leaves −0.069/−0.104 gaps
> to representative Allen BO/FC sessions. Residual phase is weakly shared
> across probes but does not provide an additional repair. The absolute Allen
> V1 modulation point is therefore not yet a calibrated multi-site V1 reference. See
> the reproducible
> [V1 cross-dataset bridge](artifacts/figure3/06b_v1_dataset_bridge/V1_DATASET_BRIDGE.md)
> and its raw-data acceptance analysis before using the area-versus-probe result.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ANALYSIS_ROADMAP.md` | Living, checkpointed analysis plan, including the V1 calibration and achieved-RF matching claim gates. |
| `generate_retinotopic_csvs.py` | Extract the three response metrics and probe metadata from a MouseV2 NWB file. |
| `data/site*_processed/` | Per-session metric and unit-quality tables for the new V1 recordings. |
| `Figure3/Figure3_with_V1sites.py` | Reproduce the original Figure 3 layout with the new V1 sessions overlaid. |
| `Figure3/Figure3_probe_zoom.py` | Show session-level V1 probe measurements in the context of the published hierarchy. |
| `Figure3/Figure3_split_comparison.py` | Directly compare within-V1 probe variation with post-V1 area variation. |
| `scripts/eta_squared_comparison.py` | Session-level effect-size analysis using eta-squared and bias-corrected omega-squared. |
| `scripts/v1_dataset_bridge.py` | Diagnose Allen/MouseV2 V1 offsets, Allen stimulus-set heterogeneity, Welch-grid non-equivalence, and the claim gate. |
| `scripts/allen_rf_matching.py` | Audit achieved Allen V1/HVA RF centers, paired session offsets, and RF common support without altering the released table. |
| `scripts/allen_rf_adjusted_response.py` | Fit within-session RF-adjusted Allen area models and same-session V1 matching sensitivities with balance/attrition reporting. |
| `scripts/allen_frequency_preference_surfaces.py` | Estimate session-balanced nonlinear SF/TF preference surfaces over achieved Allen RF azimuth/elevation, with optional metric-specific tuning enrichment and paired-session V1 differences. |
| `scripts/extract_mousev2_frequency_tuning.py` | Fit joint Poisson log-Gaussian SF × log-Gaussian TF × von-Mises orientation models, retain empirical diagnostics, and gate continuous preferences by tuning, reliability, fit quality, and identifiability. |
| `scripts/extract_mousev2_parametric_rf.py` | Fit trial-level Poisson rotated-elliptical-Gaussian RF models and gate centers by significance, reliability, fit quality, and identifiability. |
| `scripts/mousev2_frequency_preference_surfaces.py` | Reproduce pooled and probe-resolved maps using only supported parametric RF centers and SF/TF preferences. |
| `scripts/extract_mousev2_grating_common_support.py` | Recompute all MouseV2 units on the Allen SF = 0.04 condition subset and render the first raw bridge diagnostic. |
| `scripts/extract_allen_v1_bridge.py` | Reproduce released Allen grating metrics from verified raw NWBs and recompute representative sessions on the common 1-s/15-trial support. |
| `scripts/mousev2_timescale_trial_bridge.py` | Downsample MouseV2 to Allen's 75 bright + 75 dark flashes and quantify timescale and fit-selection sensitivity. |
| `scripts/v1_grating_phase_bridge.py` | Decompose harmonized grating responses into single-trial amplitude, coherent amplitude, phase consistency, and target/off-target spectral power. |
| `scripts/mousev2_grating_start_phase_bridge.py` | Reconstruct MouseV2 grating start phase from the frozen acquisition code and test source-phase adjustment against TF-specific and permutation controls. |
| `scripts/mousev2_grating_corrected_welch_bridge.py` | Replace only the source-phase-dependent carrier component and rerun the unchanged Welch modulation index with TF, sign, and permutation controls. |
| `scripts/mousev2_grating_shared_phase_behavior.py` | Test whether source-corrected residual phase is shared across probes or covaries with running and eye state. |
| `reports/multisite_v1_allen_v1_differences/MULTISITE_V1_ALLEN_V1_DIFFERENCES.pdf` | Plain-language, illustrated synthesis of the observed multi-site V1–Allen V1 differences, resolved protocol effects, and remaining comparison limits. |
| `scripts/render_multisite_v1_allen_v1_difference_report.py` | Regenerate the synthesis report, its summary figure, HTML, and PDF from the versioned Markdown source. |
| `scripts/compute_probe_ccg.py` | Compute jitter-corrected cross-probe CCG feedforwardness scores. |
| `scripts/extract_unit_quality.py` | Extract quality metrics for the new V1 units. |
| `Figure1/`–`Figure4/`, `ExtDataFigure*/` | Historical scripts and outputs from the original platform paper. |
| `reference/` | Exploratory modern-NWB code; not part of the main batch pipeline. |

## Current processed dataset

The repository currently contains processed tables for eight multi-site V1
sessions from the *MouseV2* project, `site2` through `site9`. Each session
contains probes A, B, C, and E. Together, the tables contain 20,374 units before
metric-specific and quality filters.

Each `data/siteN_processed/` directory is expected to contain:

| File | Key outputs |
| --- | --- |
| `change_modulation_data.csv` | `unit_id`, `modulation_index` (F1/F0) |
| `time_to_first_spike.csv` | `unit_id`, `time_to_first_spike` in seconds |
| `timescale_metrics.csv` | `unit_id`, `autocorr_tau`, fit error, spike count |
| `layer_info.csv` | depth, layer, and `V1_siteN_<probe>` label |
| `unit_quality.csv` | NWB quality metrics and `default_qc` flag |

`data/unit_table.csv` is the original AllenSDK unit table used for the published
areas. The current V1 comparison scripts merge the per-site tables at runtime;
they do not require the new sites to be appended to `unit_table.csv`.

## Environment

`environment.yml` preserves the original paper-era environment (Python 3.7 and
AllenSDK 2.2). It is useful for historical scripts but is not a complete
environment specification for the MouseV2 extension.

The current scripts require Python 3 plus NumPy, pandas, SciPy, Matplotlib,
h5py, statsmodels, and scikit-learn. AllenSDK is only required for scripts that
access the original Allen cache or reconstruct the original unit table.

For a modern environment, install the dependencies with your preferred Conda
or virtual-environment workflow. Do not mix an old AllenSDK environment with
user-site PyNWB/HDMF packages; incompatible NWB schema versions can produce
misleading reader errors.

## Process a MouseV2 session

The generator reads the NWB file directly with h5py, avoiding the older
AllenSDK NWB reader. Run it from the repository root:

```bash
python generate_retinotopic_csvs.py \
  --nwb /path/to/session.nwb \
  --out_dir data/site10_processed \
  --site_name V1_site10 \
  --id_offset 10000000
```

The equivalent wrapper is:

```bash
scripts/run_retinotopic_site.sh \
  --nwb /path/to/session.nwb \
  --out_dir data/site10_processed \
  --site_name V1_site10 \
  --id_offset 10000000
```

Use a unique `id_offset` for every session. By convention, site N uses
`N * 1,000,000`.

The NWB file must contain:

- a units table with indexed spike times;
- a flash interval table with `start_time` for TTFS and timescale estimation;
- a drifting-gratings interval table with `start_time`, `stop_time`,
  orientation, and temporal frequency; and
- preferably `device_name` in the units table so units can be assigned to
  probes A, B, C, or E.

The generator does not currently compute receptive-field metrics or assign
cortical layers. `cortical_layer` is therefore left missing unless a separate
assignment step supplies it.

## Generate the current figures

Run these commands from the repository root:

```bash
python Figure3/Figure3_with_V1sites.py
python Figure3/Figure3_probe_zoom.py
python Figure3/Figure3_split_comparison.py
```

They produce:

- `Figure3/Figure3_with_V1sites.png` — distributions, CDFs, hierarchy
  correlations, and pairwise significance matrices;
- `Figure3/Figure3_probe_zoom.png` — session means for each V1 probe overlaid
  on the published hierarchy; and
- `Figure3/Figure3_split_comparison.png` — within-V1 versus post-V1 spreads.

The scripts discover all complete `data/site*_processed/` directories. Missing
or incomplete site directories are skipped.

## Statistical comparison

Run:

```bash
python scripts/eta_squared_comparison.py
```

The analysis first aggregates units to session-by-group means to reduce
pseudoreplication. It then compares four V1 probe groups with six post-V1 area
groups using omega-squared, with session-resampled bootstrap confidence
intervals. It writes the figure captions, methods, results, and caveats to
`Figure3/Figure3_stats.md`.

The default analysis applies `default_qc == True` to MouseV2 units when the
quality table is present. Original Allen units are already drawn from the
published unit table and its established filtering pipeline.

## Cross-probe CCG analysis

`scripts/compute_probe_ccg.py` computes jitter-corrected CCG peak offsets and a
feedforwardness score for all cross-probe unit pairs. It currently contains a
machine-specific `NWB_BASE` and a fixed site-to-subject map, so update those
values before running on another system.

```bash
# One-session smoke test
python scripts/compute_probe_ccg.py --test

# All configured sessions
python scripts/compute_probe_ccg.py
```

The result is saved as `data/processed_data/probe_ccg_results.npz`.

## Metric definitions

TTFS uses 1 ms binary bins and takes the first spike between 30 and 200 ms after
flash onset. Values are stored in seconds in the per-site CSV and converted to
milliseconds for plotting.

F1/F0 is calculated at each unit's preferred drifting-grating condition. Spike
trains are folded into stimulus cycles, and the DC and first-harmonic FFT
components provide F0 and F1.

Response timescale uses 10 ms flash-locked spike-count bins over 0–2 seconds.
The trial autocorrelation from the 40–290 ms post-flash response window is
averaged and fit with an exponential. `autocorr_tau` is stored in milliseconds.

## Known limitations

- The new V1 probe groups are placed near the published VISp hierarchy score
  for display. Their x-positions are not RF-derived hierarchy estimates.
- Receptive-field metrics are not yet generated for the MouseV2 sessions.
- Allen's ISI-guided insertion targets were retinotopically aligned for V1, LM,
  AL, AM, and PM, with an explicit geometric-center accommodation for RL. This
  controls intended targeting, not necessarily the distribution of achieved
  unit RF centers. The paper reports RF-size distributions and area-level mean
  RF outlines, but not session-level RF-center offsets or dispersion. The
  Iteration 6C now estimates those quantities from `azimuth_rf` and
  `elevation_rf`; its paired audit finds substantial residual offsets. The
  primary inferential comparison must preserve Allen session dependence and
  test whether visual-area effects survive RF adjustment. The first adjusted
  checkpoint finds metric-specific attenuation rather than a uniform result;
  its matching balance/attrition trade-off keeps the claim gate open. See
  Iteration 6C in
  [ANALYSIS_ROADMAP.md](ANALYSIS_ROADMAP.md#iteration-6c--verify-achieved-allen-retinotopic-matching-and-define-the-rf-adjusted-comparison).
- The released Allen table supports nonlinear preferred-SF and preferred-TF
  surfaces over RF azimuth/elevation for Brain Observatory sessions. These are
  preference-bin surfaces, not full tuning curves: Allen measured SF with
  static gratings and TF with drifting gratings, while Functional Connectivity
  presented only one TF. HVA-minus-V1 maps use V1 from the same session set and
  mask cells without local multi-session support.
- New-data TTFS may include a display-timing offset relative to the original
  Allen recordings. Within-dataset comparisons are unaffected, but absolute
  cross-dataset latency differences require care.
- Allen `mod_idx_dg` is not an equivalent absolute cross-dataset metric:
  Allen uses 2-s gratings and a 1,024-sample Welch segment, whereas MouseV2
  uses 1-s gratings and a 1,000-sample segment. The released frequency lookup
  therefore evaluates different physical bins. Raw common-window recomputation
  leaves MouseV2 about 0.19–0.22 log10 below representative Allen sessions,
  while harmonized F1/F0 differs by only about 0.03–0.04 log10. Allen V1 also
  pools two stimulus sets with different modulation-index centers. The raw
  phase bridge shows that MouseV2 has comparable or higher single-trial F1
  amplitude but lower weighted phase coherence (0.387 versus 0.539/0.516), so
  the discrepancy emerges during coherent trial averaging. MouseV2's acquisition
  code advances phase from the absolute frame without resetting at presentation
  onset; reconstructing this schedule raises coherence from 0.387 to 0.433, but
  leaves residual gaps of 0.106/0.083 versus representative Allen BO/FC.
  Matched-trial residual phase is weakly shared across probes (alignment 0.173
  versus a 0.141 shuffled null; 1,000-permutation equal-session p = 0.001), but
  applying the other-probe estimate changes
  coherence from 0.433 to 0.429 rather than restoring it. Condition- and
  block-time-controlled associations with a 50% valid-eye-coverage requirement
  remain for running and pupil x/y, not pupil area; these are descriptive
  candidate-state signals, not evidence that behavior causes the cross-dataset
  offset.
- Matching MouseV2 from 300 to Allen's balanced 150 flashes lowers its
  equal-session timescale mean from 47.53 to 45.92 ms and the validity fraction
  from 0.207 to 0.168. A roughly 2.04-ms offset from Allen Brain Observatory
  remains and should not be interpreted biologically until RF/layer and other
  population support are matched.
- Some historical figure scripts and the CCG pipeline contain developer-local
  paths. The three current Figure 3 scripts resolve repository paths
  automatically.
- The bootstrap in `eta_squared_comparison.py` resamples sessions within each
  group. Correlation between areas recorded in the same Allen session is not
  modeled fully.
- The active bridge code has automated tests, but a fully locked modern
  extraction environment is still needed.

## Historical paper code

Most original figures can be generated from publicly available NWB files and
unit tables through AllenSDK. The historical scripts are retained as released
and may assume the original cache layout, dependency versions, or local paths.
They should be treated as archival reproduction code rather than as the entry
point for the MouseV2 analysis.

## Support and license

This code is provided as-is without active support. Contributions are welcome;
see [CONTRIBUTING.md](CONTRIBUTING.md). Licensing terms are in
[LICENSE.txt](LICENSE.txt), and author information is in [AUTHORS.rst](AUTHORS.rst).

© 2019 Allen Institute for Brain Science
