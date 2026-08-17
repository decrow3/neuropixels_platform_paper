# Registering visual-field coordinates across animals with V1, HVA, and LGd RF dispersion

## Technical summary

The project asks whether a shared two-dimensional visual-coordinate translation can be estimated for each animal from several independent relationships between receptive-field (RF) location and local RF-center dispersion. The proposed latent translation represents static eye position, screen alignment, or another session-wide offset that should affect V1, the higher visual areas (HVAs), and LGd in the same direction.

There is reproducible spatial organization in RF-center dispersion, and the available Allen corpus is large enough to construct matched V1, combined-HVA, and LGd dispersion fields in 16 simultaneous sessions. The present implementation, however, has **not validated a shared translation**. Structure-specific optima usually disagree, and the strongest apparent V1–LGd agreement disappeared when the translation range was expanded from ±30° to ±60°. In session `760345702`, V1 moved to `(22°, −24°)`, LGd continued to the new boundary at `(18°, −60°)`, and HVA preferred `(18°, +18°)`. The earlier common `(30°, −30°)` solution was therefore a boundary artifact.

The current evidence supports three narrower conclusions:

1. anatomy-corrected RF dispersion contains real retinotopic organization;
2. its ability to identify a unique translation is highly session- and structure-dependent;
3. incompatible component likelihoods must not be multiplied or averaged into a falsely precise registration.

The next model should retain the full V1/HVA anatomical topology and use dispersion as one likelihood term, but it needs stronger direct anchors—most plausibly the raw Gabor screen-boundary likelihood and carefully regularized LGd anatomy—before estimating a production screen/eye translation.

## The scientific objective

Widefield intrinsic-signal imaging (ISI) maps were used to plan Allen electrode penetrations. The original goal was therefore not to invent a new cortical coordinate system, but to reconstruct, per animal, the relationship

\[
\text{CCF anatomy} \longleftrightarrow \text{retinal coordinates}.
\]

The fixed observations are the anatomical locations of recorded cells and their measured RFs. A population retinotopic atlas supplies a prior over how visual coordinates should vary across cortex, but the atlas may warp from animal to animal. This led to the working principle used throughout the later analyses:

> Plot cells at their measured anatomical coordinates and warp the retinotopic template or contour field around them. Do not move anatomy to make it resemble an average map.

The desired registration must account for two different sources of between-animal variation:

- **map geometry:** local rotation, scale, shear, and nonlinear distortion of retinotopy over anatomical coordinates;
- **shared visual translation:** a constant shift in azimuth and elevation caused by eye/screen alignment or another session-wide visual origin.

These are not interchangeable. RF gradients, field sign, map reversals, and CCF-to-RF progression constrain geometry, but all remain unchanged if the same translation is added to every RF in a session. A screen/eye offset therefore needs a variable whose value changes with absolute RF location.

## Template and coordinate conventions

### Historical Zhuang/Han registration

The prior Han/Bonin workflow, described in the conversation but not available as executable code on this machine, used the mean retinotopic maps from Zhuang et al. (2017), Figure 9:

- panel C for altitude/elevation;
- panel D for azimuth.

Han/Bonin two-photon fields were first registered to each animal's one-photon map and then to the Han/Bonin group common-map image. Ten landmarks in the Zhuang mean maps and Han common map were matched, and a single global affine transform was retained:

```matlab
T = [-1.0   0.5   0;
      0.5   1.05  0;
    370   -180    1];
```

That transform was used to place the Zhuang borders and smooth azimuth/elevation fields into the Han common-map pixel frame. This history motivated using Zhuang Figure 9 as a population-level prior rather than a single-animal example.

### Allen plotting convention adopted here

For the Allen analyses, cells remain in their CCF-defined anatomical positions while the template contours are warped. The displayed AP/ML axes were swapped and both signs were flipped so that values run high-to-low, matching the coarse orientation of the Zhuang figure. Axis scaling is equal and the medial display extent is cropped rather than forcing the entire contour template into view.

This convention changes only presentation. It must not introduce an image-space vertical flip into the numerical elevation field; explicit `imshow(..., origin=...)` checks were added during the interpolated azimuth/elevation and field-sign QA.

## Data and improved RF measurements

The current analysis uses the improved parametric RF fits rather than only Allen's released RF centers.

| Population | Coverage used in this project |
|---|---:|
| Improved cortical aperture fits | 5,529 V1 units and 15,350 HVA units |
| V1 sessions in the absolute-size/dispersion checkpoint | 54 |
| Raw LGd units refit from Gabor responses | 2,582 units in 33 sessions |
| Sessions meeting joint dispersion minimums | 16 |

For LGd, spike counts were extracted for all Gabor presentations and an analytic-aperture Gaussian RF was fitted with deterministic train/test splits. A unit entered the LGd coordinate set only if its held-out spatial gain was positive. Centers exactly at the ±60° optimizer limit were excluded; extrapolated centers between 40° and 60° were retained when the response flank generalized to held-out trials.

The raw refits recovered useful boundary evidence that the released clean-center set omitted. In the three original concrete sessions, 82 of 341 LGd units had positive held-out spatial gain, 22 had boundary centers, and 53 of the 82 useful fits were absent from the released significant/on-screen subset.

## How the analysis evolved

### 1. Anatomy-constrained affine and warped cortical maps

The first diagnostic registered RF centers and CCF anatomy to a Zhuang-derived cortical template. A global affine warp was useful for visualizing broad geometry, but it was too restrictive for the biological question. Because individual maps can distort locally, the analysis moved toward holding the anatomy and RF observations fixed while allowing the population contour map to warp.

This established an important distinction: the penetration targets were aimed at corresponding area centers, but a probe labeled LM or RL can sample cells whose RF progression suggests that the track crossed an anatomical or retinotopic boundary. Such cases should be diagnostic evidence about the warp, not automatically removed.

### 2. RF size as a translation signal

The initial hypothesis was that V1 RF size changes with visual-field location. If a session-specific RF-size surface could be translated onto a leave-one-session-out population surface, that translation could be applied unchanged to V1 and HVA maps.

Early implementations excluded RFs near the stimulus edge and normalized RF size within animal. Both choices were reconsidered:

- improved aperture fits can recover information near and beyond the stimulus boundary, so blanket edge exclusion can remove useful observations;
- within-animal RF-size normalization removes the absolute between-animal signal that could carry the translation.

The corrected analyses retained absolute `log2(RF area)` and did not exclude centers merely because they were near the screen edge. Nevertheless, RF-size translation remained unstable. Across bounds from ±10° to ±30°, median held-out correlation gains were only +0.022 to +0.068, only 52–58% of sessions improved, and no bound reached conventional significance. Training improvements grew as the bound increased without comparable gains in reproducibility, indicating overfitting to broad, heterogeneous surfaces.

RF size was then kept out of the covariance estimator and used as an independent validator. In six preregistered cases, covariance and size optima differed by roughly 30–59°, and no session improved held-out RF-size prediction in both cell-split directions. RF size therefore has not earned registration weight, although its biological relationship with retinal location remains a separate question worth studying in V1 and HVAs.

### 3. RF-center dispersion and the sampling-control problem

The next hypothesis was that local cell-to-cell RF scatter varies systematically with RF location. Let anatomical position be \(x\), the local mean retinotopic map be \(\mu(x)\), and an individual RF be

\[
r_i = \mu(x_i) + \eta_i.
\]

For a sampled neighborhood,

\[
\operatorname{Var}(r)
=
\operatorname{Var}[\mu(x)]
+
E[\operatorname{Var}(r\mid x)].
\]

The first term is apparent dispersion caused by sampling different anatomical locations along a retinotopic gradient. The second is the local RF scatter of interest. A large anatomical jump is not itself a confound when it produces the RF jump predicted by the map. The nuisance is extra dispersion caused by discontinuous or unusually broad anatomical pooling beyond the mean-map prediction.

An early control residualized covariance against position along the probe. That was too aggressive: real retinotopic variation is expected to be predictable from anatomy. The refined approach instead estimated a smooth CCF-to-RF mean map and calculated RF residuals around it. The mean-map smoothness scale (roughly 120–250 µm in diagnostics, 250 µm in the matched multi-structure descriptor) determines what is classified as map progression versus local scatter and is therefore a biological/modeling assumption, not a harmless tuning parameter.

The physical-sampling audit showed that covariance placement contains nonrandom organization, but translation stability often came from the smooth component predictable along the shank. After residualization, the strongest apparent raw-trace successes became substantially less stable. This means dispersion is not simply noise, but it also cannot yet be treated as a sampling-independent translation anchor.

### 4. Trace, tensor anisotropy, and pairwise directional estimators

The most stable covariance summary was the trace,

\[
\operatorname{tr}(\Sigma)=\sigma^2_{az}+\sigma^2_{el},
\]

which measures total local RF scatter without requiring a reliable ellipse orientation. In an independently quality-selected five-session V1 discovery set, all five sessions showed descriptor-to-location organization beyond exact-support shuffles, but only `760345702` had a compact basin stable across random cells and physical sections. Nested uncertainty for that session was approximately 14° at the 90th percentile.

Directional information performed worse. The normalized covariance tensor components and pooled pairwise axial moment were sensitive to a few large residuals and anatomical pair-distance bands. For `760345702`, the full pairwise anisotropy magnitude looked nonzero, yet the median cell-half axis disagreement was 47.9°—approximately chance for an unoriented axis—and removing two large-residual cells rotated the full-data axis by 42°. Adding tensor anisotropy increased rather than reduced localization uncertainty. Anisotropy has therefore not earned registration weight.

### 5. Gaze, screen censoring, and thalamic corroboration

The internally estimated V1 CCF-to-RF residual is extremely reproducible across cell halves (azimuth \(r=0.995\), elevation \(r=0.992\), median two-dimensional discrepancy 1.22°), but that does not prove it is a pure screen/eye translation. It can also contain animal-specific cortical geometry.

Absolute filtered gaze during the Gabor block did not predict that V1 residual: matched-axis correlations were essentially zero and leave-one-session-out prediction had negative \(R^2\) for both axes. Allen also cautions that absolute gaze has no accuracy guarantee because of rig degrees of freedom.

A held-out `on_screen_rf` likelihood supplied weak edge information but did not identify a stable two-dimensional shift. This motivated returning to the raw Gabor response matrix, whose rising edge flanks can constrain an off-screen RF center more directly than a binary released label.

Released LGd/LP RF centers also failed to corroborate V1 offsets. Their anatomical maps were too weak for precise session localization: typical split-half offset discrepancies were 12–15°, and combined thalamic offsets did not predict the V1 residual. This failure did not rule out a common screen offset; it showed that thalamic anatomy and released RF measurements were too noisy. The subsequent raw LGd Gabor fitting materially improved the observation model and enabled the current joint-dispersion test.

## Current matched multi-structure descriptor

The same descriptor is now computed in V1, combined HVAs, and LGd:

1. estimate a leave-one-cell-out mean RF from 3-D CCF position with a 250 µm Gaussian kernel;
2. for HVAs, estimate the anatomical mean separately within each named cortical area;
3. subtract the anatomical mean to obtain each cell's RF residual;
4. calculate the covariance trace of residual RF vectors among cells within a 15° RF-space neighborhood;
5. attach `log2(trace)` to the cell's observed RF location;
6. pool the already anatomy-corrected HVA descriptors across areas;
7. smooth each session's descriptor over RF location with a 12° kernel.

The descriptor value is invariant to adding a common visual translation, but its location in the population field changes. For a structure \(g\), session \(s\), and candidate shift \(\delta_s\), the component objective is conceptually

\[
L_{s,g}(\delta_s)
=
D\!\left[C_{s,g}(r),\,C_{-s,g}(r+\delta_s)\right].
\]

The three structures share one candidate \(\delta_s\). Because a population template learned from the same unregistered animals has an arbitrary global intercept, only relative animal offsets are identifiable; the alternating model imposes a zero-mean translation constraint.

Sixteen sessions passed fixed minimums of 30 valid V1, 50 valid HVA, and 10 valid LGd descriptors.

## Joint-model results

### Fixed raw-coordinate templates showed incompatible optima

The first joint checkpoint used raw-coordinate leave-one-session-out templates and exact-support shuffles. V1 and HVA often contained strong nonrandom organization, but their preferred shifts disagreed; LGd was sometimes unsupported or boundary-dominated. For example, in `715093703`, V1 preferred `(-14°, +20°)` and HVA preferred `(+30°, −8°)`, both with shuffle \(p=0.020\), while LGd was unsupported (\(p=0.725\)). A weighted average of those surfaces would be a compromise, not evidence for a shared translation.

### Alternating registration improved numerical stability but not biological agreement

The next implementation alternated between translating sessions, rebuilding structure-specific templates, and updating each session against leave-one-session-out fields. A coordinate-order bug in the accelerated interpolation path was found and corrected before interpretation. Simultaneous hard updates oscillated, so updates were damped to 25%.

With damping, the five initializations became more similar: median per-session distances from the best solution were approximately 2.6–5.9°. However, none reached the strict maximum-update convergence threshold by 60 iterations, and the structure-specific optima remained far apart. Across the 16 sessions, the median component-to-joint discrepancy was about 21°.

### Cell halves exposed heterogeneous reliability

Descriptors were recomputed independently in cell halves stratified by HVA area and probe while training templates were frozen and the target session was excluded.

| Session | Structure | Median half-to-half optimum distance |
|---|---|---:|
| `715093703` | V1 | 51.4° |
| `715093703` | HVA | 7.0° |
| `715093703` | LGd | 22.4° |
| `754829445` | V1 | 0.0° |
| `754829445` | HVA | 35.2° |
| `754829445` | LGd | 22.3° |
| `760345702` | V1 | 2.0° |
| `760345702` | HVA | 6.8° |
| `760345702` | LGd | 0.0° in 8 evaluable repeats |

The apparent V1–LGd agreement in `760345702` was within 10° in 89% of usable half comparisons. This looked like the first candidate for a two-structure constraint, but both full-data optima were exactly at the original `(30°, −30°)` boundary.

### The expanded-bound test rejects the apparent V1–LGd success

The decisive test expanded candidate translations to ±60° and added an absolute template-support rule: a session could contribute at a template pixel only when its third-nearest observed cell was within 24° (twice the smoothing bandwidth). At least five training sessions had to support a template pixel. This prevents Gaussian-tail extrapolation from manufacturing an optimum outside observed RF support.

| Session | V1 optimum | HVA optimum | LGd optimum | V1+LGd optimum | All-component optimum |
|---|---:|---:|---:|---:|---:|
| `715093703` | (−10°, +2°) | (+34°, +6°) | (+28°, −32°) | (+24°, −8°) | (+32°, −6°) |
| `754829445` | (−48°, +12°) | (−10°, −30°) | (−28°, +16°) | (−28°, +16°) | (−28°, +16°) |
| `760345702` | (+22°, −24°) | (+18°, +18°) | (+18°, −60°) | (+26°, −36°) | (+20°, −34°) |

For `760345702`, LGd continued to the new −60° elevation boundary while V1 and HVA selected different interior regions. The earlier exact V1–LGd match was therefore a clipping artifact. The V1+LGd and all-component optima are compromises between incompatible landscapes and must not be interpreted as recovered screen/eye translations.

## What is established and what is not

### Established

- Improved cortical and raw LGd RF fits provide enough simultaneous data to construct matched dispersion fields in 16 sessions.
- V1, HVA, and LGd dispersion fields can each show nonrandom organization over RF location.
- V1 trace is more reproducible than covariance anisotropy.
- The objective shape—not only its minimum—reveals rings, ridges, boundaries, and unsupported regions that make a translation non-identifiable.
- Search-bound expansion and absolute support masks are necessary safeguards.
- HVA area identity must be respected when estimating anatomy-to-RF residuals, even when the resulting descriptors are pooled across HVAs.

### Not established

- No session currently has a shared V1/HVA/LGd translation that is interior, sharp, split-half reproducible, physically reproducible, and supported by held-out animals.
- The stable V1 CCF-to-RF residual has not been shown to be a pure eye/screen offset.
- RF size, covariance anisotropy, absolute gaze, released on-screen status, or released thalamic centers do not presently validate the translation.
- The mean visual-field origin remains unidentified without an external retinal-coordinate anchor.

## Limitations and robustness requirements

1. **Training translations and templates are coupled.** Multiple local fixed points can exist even with a zero-mean constraint.
2. **LGd support is sparse.** Many sessions barely exceed ten valid descriptors; apparent split-half precision may be based on few evaluable splits.
3. **Scalar dispersion fields can be non-injective.** Similar dispersion values may occur along rings or broad ridges, so a significant shuffle test does not imply unique localization.
4. **Physical sampling and retinotopy are intrinsically correlated.** Controls must remove only excess dispersion due to discontinuous sampling, not the real anatomical substrate of the retinotopic map.
5. **Smoothing choices define the estimand.** CCF and RF bandwidths determine which variation is called mean-map geometry versus local scatter.
6. **Boundary and support effects are powerful.** Optimizer bounds, stimulus bounds, and extrapolated template tails must be audited separately.
7. **The present work is exploratory.** Method choices were refined after inspecting concrete cases; confirmatory performance requires a frozen model and untouched evaluation sessions.

## Recommended next steps

1. Replace point-estimated off-screen LGd centers with the full held-out raw Gabor response likelihood, so screen-boundary evidence enters without treating extrapolated centers as exact coordinates.
2. Fit the V1/HVA mean retinotopic geometry and the shared translation hierarchically, allowing animal-specific smooth warps while fixing observed CCF anatomy.
3. Use dispersion trace only after reliability calibration; omit anisotropy unless a robust estimator passes cell, physical-section, and pair-distance tests.
4. Treat V1, combined HVA, and LGd as separate likelihoods and require their credible regions to overlap. Do not report a joint point estimate when component posteriors are incompatible.
5. Add explicit uncertainty and rejection states: `localized`, `one-axis/ridge`, `boundary-limited`, `component-conflict`, and `insufficient-support`.
6. Freeze the model using the current 16-session discovery cohort, then validate on held-out animals and held-out physical probe sections.
7. If the original per-animal Allen ISI maps and their screen coordinates become available, use them as the preferred absolute anchor and treat unit-derived registration as validation rather than reconstruction.

## Reproducibility and artifact index

The primary scripts added for the current multi-structure stage are:

- `scripts/pilot_lgd_gabor_boundary_maps.py`
- `scripts/checkpoint_multistructure_dispersion_fields.py`
- `scripts/checkpoint_joint_multistructure_dispersion_likelihood.py`
- `scripts/fit_joint_multistructure_dispersion_em.py`
- `scripts/validate_joint_dispersion_cell_halves.py`
- `scripts/test_expanded_joint_dispersion_bounds.py`

Key saved artifacts are:

- [joint descriptor checkpoint](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/joint_multistructure_dispersion_checkpoint/README.md)
- [alternating-model convergence figure](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/joint_multistructure_dispersion_em/Figure_em_convergence_and_component_agreement.png)
- [cell-half reproducibility figure](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/joint_multistructure_cell_half_validation/Figure_cell_half_reproducibility.png)
- [expanded-bound loss surfaces](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/expanded_bound_support_limited_test/Figure_expanded_bound_landscapes.png)
- [expanded-bound numerical optima](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/expanded_bound_support_limited_test/expanded_bound_component_optima.csv)
- [V1/screen-offset signal stocktake](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/V1_SCREEN_OFFSET_SIGNAL_STOCKTAKE.md)
- [physical-sampling control](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/physical_sampling_control/PHYSICAL_SAMPLING_CONTROL_README.md)
- [covariance discovery set](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/covariance_discovery_set/COVARIANCE_DISCOVERY_SET_README.md)
- [RF-size corroboration](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/rf_size_corroboration_cases/RF_SIZE_CORROBORATION_README.md)
- [LGd sufficiency audit](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/lgd_map_sufficiency/README.md)
- [raw LGd Gabor checkpoint](../artifacts/v1_absolute_size_dispersion_translation_checkpoint/lgd_gabor_boundary_pilot/README.md)

Reproduce the newest boundary test from the repository root with:

```bash
env MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdgcache \
python -m scripts.test_expanded_joint_dispersion_bounds
```

## Further questions

- Does a raw-response LGd likelihood yield an interior translation when point-center extrapolation does not?
- Can V1/HVA map junctions and field-sign reversals constrain the animal warp tightly enough that a weak screen-boundary likelihood only needs to resolve the global translation?
- Are HVA dispersion fields genuinely shared after area-specific anatomical correction, or should each HVA retain a separate dispersion field with partial pooling?
- How much of the extremely reproducible V1 CCF-to-RF residual is animal-specific cortical geometry rather than a common visual-coordinate offset?
- What rejection threshold gives adequate coverage when component likelihoods are broad or incompatible?

