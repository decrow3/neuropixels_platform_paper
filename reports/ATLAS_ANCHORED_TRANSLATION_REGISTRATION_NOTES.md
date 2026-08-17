# Atlas-anchored translation registration: working notes

## Relationship to the earlier report

`reports/V1_HVA_LGD_RF_DISPERSION_REGISTRATION.md` documents the RF-dispersion-trace approach
to estimating a per-session visual-field translation and concludes that approach is not yet
validated: structure-specific optima disagree, and the alternating EM used to jointly fit
per-session translations against a coupled template is unstable (median ~21° component-to-joint
discrepancy across 16 sessions; one apparent exact V1-LGd match at `760345702` turned out to be
a shared optimizer-boundary artifact once the search bound was widened from ±30° to ±60°).

This document picks up from there with a different strategy: decouple shape estimation from
translation estimation, then ground shape against external retinotopic atlases (Zhuang et al.
2017, Garrett et al. 2014) rather than relying only on the coupled template. It's written as a
chronological account of what was tried, what broke, what was learned from each break, and
where things currently stand — several dead ends are included deliberately because the reason
they failed is informative for anyone extending this work.

## Part 1 — Decoupled shape/translation via a local-linear Jacobian field

### The core idea

A local (kernel-weighted) **linear** regression of RF on CCF position, with the response
demeaned **within session** before fitting, has a slope estimate that is exactly invariant to
each session's own additive translation — a standard OLS/WLS fact (adding a constant to the
response only shifts the fitted intercept, never the fitted slope). This means the local
Jacobian (retinotopic gradient) can be estimated by pooling cells across many sessions with
*zero* risk of a session's own unknown translation leaking into the shape estimate — unlike
the old EM, where the template's absolute placement depended on other sessions' currently-
estimated translations, creating real coupling.

Given a pooled Jacobian field, translation for one session becomes a simple, decoupled
robust-location fit: integrate the Jacobian along that session's own sampled anatomy to get a
shape-only predicted RF field (up to one unknown additive constant), then find the constant
that best matches observed RF. No grid search, no damped alternation.

### Implementation: `scripts/fit_multistructure_fixed_effect_translation.py`

- V1+HVA fit as **one connected domain** (single shared translation, no per-area intercept) —
  Allen targeted most probes to be retinotopically matched in eccentricity across areas, so a
  per-area offset would impose an artificial break at every area boundary the real map doesn't
  have. LGd is a separate modality/frame, fit and weighted separately, not fused into the
  cortex shape fit.
- `local_linear_jacobian_field()`: within-session-weighted-demeaned kernel-weighted local-linear
  regression, Huber-IRLS reweighted on top (order-1 fits are far more leverage-sensitive than
  the existing order-0 Nadaraya-Watson averaging elsewhere in this codebase).
- `mst_path_integrate_session()`: per-session minimum-spanning-tree walk over that session's own
  cells, trapezoidal integration of the Jacobian along tree edges.
- Combination: cortex and LGd per-session deltas combined by `clip(shuffle_z,0,3)/3` reliability
  weight (same convention as `checkpoint_joint_multistructure_dispersion_likelihood.py`), then
  `recenter()` to zero mean.

### Two real bugs caught before trusting the results

1. **Artificial 30° clip.** `recenter()` (reused from `fit_joint_multistructure_dispersion_em.py`)
   defaults to a 30° clip bound tuned for the *old* grid-searched EM. This new fit is never
   bounded by a search window, so reusing that default silently piled several sessions onto an
   identical clipped value. Fixed by widening to 90°.
2. **Path-integration drift at fold/reversal regions.** A few grid nodes had well-supported,
   well-conditioned but extremely steep fitted slopes (up to ~22 deg/µm) — almost certainly
   retinotopic fold/reversal regions where a local-linear model is structurally wrong regardless
   of how much data supports it. Left unguarded, this turned ordinary ~140 µm integration steps
   into >1000° jumps. Fixed with a magnitude cap on the fitted Jacobian
   (`MAX_JACOBIAN_NORM_DEG_PER_UM`) and by switching the *display* "common map" reconstruction
   from a single MST path (one arbitrary route, errors compound additively along it) to a
   least-squares reconstruction over every grid edge simultaneously (redundant connectivity
   averages noise down instead of accumulating it). This only affects the display panel, not the
   per-session delta fit, which still uses the per-session MST on that session's own (much
   smaller, more local) cell cluster.

### Results

Cortex: 92% of sessions pass a shuffle-null test on the local Jacobian; median split-half
translation reproducibility 3.4° (vs. ~21° for the old EM). LGd: 38% pass, split-half 8.6° —
consistent with the report's standing recommendation to replace point-estimated LGd centers
with a full raw-Gabor-response likelihood before trusting LGd as a strong anchor.

Artifacts: `artifacts/v1_absolute_size_dispersion_translation_checkpoint/multistructure_fixed_effect_translation/`
(`session_translations.csv`, `domain_shuffle_reliability.csv`, `split_half_reproducibility.csv`,
`cross_check_against_existing_estimates.csv`, `Figure_per_session_registration.pdf` — one page
per session, RF-space observed-vs-predicted panel plus anatomical map panels).

## Part 2 — The reconstruction is underdetermined without an external anchor

Reconstructing an *absolute* shape-only map by integrating a *local* gradient field, with only
one node pinned to an arbitrary zero, has a large null space: any smooth warp that preserves
local edge-to-edge differences equally well is an equally valid optimum, and nothing in that
objective favors the real cortical topology over a self-consistent fiction. In practice the
least-squares "common map" reconstruction drifted into an inflated, physically implausible
azimuth range (~350° across the domain, vs. a real ~60° screen) — a textbook gauge-freedom
symptom, not a coding bug in the usual sense.

Fix direction agreed on: anchor the reconstruction against an external population retinotopic
atlas, as a **soft** shape prior (breaking the null-space degeneracy in directions the ephys
data itself can't constrain) rather than as ground truth to match exactly — individual-animal
deviation from a population atlas is real biology this project is trying to characterize, not
noise to be fit away.

## Part 3 — Which atlas, and does it actually agree with this data?

### Zhuang et al. (2017) infrastructure already in the repo

`artifacts/retinotopy_template/zhuang2017_figure9/` (digitized Figure 9C/D, sparse 5°-spaced
contours) plus a substantial amount of prior registration work:
`scripts/register_allen_session_to_zhuang.py` (single-session, 6-parameter affine,
`differential_evolution`, penetration-median landmarks), `scripts/build_14animal_retinotopy_registration.py`
(14-animal shared-frame cohort), `scripts/build_cross_animal_retinotopy_registration.py`.

### Is the elevation-gradient disagreement a sign bug?

The 14-animal cohort's own gradient diagnostic showed azimuth-gradient sign agreement 62%
(barely above chance) and elevation 44% (*below* chance) across 68 probes. The code never
tried an independent elevation-sign convention (only azimuth has an alternate `100-azimuth`
convention; `target_rf()` never touches elevation). Hypothesis: hidden sign bug.

Tested directly (`scripts/test_elevation_convention_zhuang_registration.py`): added an
elevation-flip candidate axis to the shared-frame search. Result: **not a sign bug**. At the
level of individual (session, azimuth-convention, reflection) combinations, flipping elevation
improved the fit objective in only 55.4% of 56 comparisons — indistinguishable from chance.
Different sessions preferred different signs. A genuine global bug should show flipped-better
in the large majority of sessions.

### Then why is elevation gradient agreement so weak?

Range mismatch, not sign. Zhuang's published elevation range is −25° to +30° (55° span);
this project's own recorded elevation spans roughly −60° to +60° raw (120° span) — after the
`+10` shift used for the shared-frame fit, 15.2% of cells have a target elevation entirely
outside what Zhuang's map can represent (10.4% for azimuth, a smaller mismatch, consistent
with azimuth being the comparatively more reliable axis in every diagnostic). Compounding
causes identified: (1) widefield-imaging spatial blur compresses true local peaks; (2) Zhuang's
map is itself a **cross-animal average** (per its own template README) — averaging many
individually-jittered single-animal peaks mechanically rounds off the group peak, on top of any
per-animal blur; (3) a genuine digitization artifact — `build_template()`'s
linear-interpolation-plus-nearest-fill construction floods everything beyond the outermost
drawn 5° contour with that contour's own value, creating an artificial flat plateau exactly
where a true peak should be (confirmed in the raw stored digitization: it explicitly does *not*
interpolate unobserved pixels; the plateau is introduced downstream by the registration code's
own surface-building step, not present in the source data).

Implication for anchoring: use Zhuang's local **gradient direction**, not absolute predicted
degree values — direction survives blur/cross-animal-averaging much better than magnitude, and
is undefined only right at a real-or-artifactual flat plateau, which can be down-weighted via
distance from the nearest real digitized contour pixel.

### Session-level pilot: does the atlas track one session's real gradient?

Session `781842082` chosen by combining highest ephys-side local-Jacobian reliability
(shuffle z=10.3, this project's own pipeline) with existing Zhuang landmark coverage.
`scripts/compare_atlas_gradient_to_ephys_pilot_781842082.py`: sampled Zhuang's
support-masked, Gaussian-smoothed gradient (`render_zhuang_interpolated_field_sign_qa.py`'s
`reconstruct()`/`normalized_smooth()`, sigma=2px) at this session's own cells, converted to
deg/mm via the session's exact (OLS-recovered, not refit — residual ~1e-13 px) CCF↔pixel
affine, and compared to this project's ephys-derived local Jacobian, also in deg/mm.

Result: azimuth median angle between gradients 108° (worse than the 90° expected under pure
noise), only 11% of cells within 45°. Elevation: median angle 52°, 38% within 45°, and
ephys elevation-gradient magnitude (34.2 deg/mm) genuinely steeper than Zhuang's (24.3 deg/mm)
— the "atlas peaks are blunted" prediction, directly confirmed for this session.

Per-area breakdown (not simply "azimuth bad, elevation ok" — genuinely area-dependent):
VISp/VISal/VISrl show poor-to-terrible azimuth agreement (82-121° median angle) uniformly;
elevation is much more area-dependent, with VISam showing excellent 10.9° median angle
agreement but a red flag (Zhuang's own elevation gradient magnitude there is nearly flat,
0.6 deg/mm — a near-zero-magnitude gradient makes direction numerically unstable, so that
"good" agreement may be a near-plateau coincidence, not real signal).

### Three-way comparison: Zhuang vs. Garrett vs. a naive atlas-free baseline

Built a **naive V1-centered cross-session pooled map** (`scripts/build_naive_v1_centered_cross_session_rf_map.py`):
every cell's RF, minus that session's own median V1 RF, pooled across 45 sessions (≥5 V1 cells
required) into one CCF-indexed table — no atlas, no fitted affine, translation cancels because
it's a same-session subtraction. `scripts/compare_three_way_gradient_781842082.py` built an
analogous smoothed local field from this naive set and reran the same gradient comparison.

| source | azimuth median angle | azimuth within 45° | elevation median angle | elevation within 45° |
|---|---:|---:|---:|---:|
| Zhuang | 108.1° | 11% | 52.5° | 38% |
| Garrett | 82.2° | 26% | 55.3° | 41% |
| **naive (atlas-free)** | **25.0°** | **94%** | 52.3° | 48% |

The atlas-free naive map beat both published atlases decisively on azimuth direction, and tied
them on elevation. Since the naive set shares this project's own species/strain/rig/RF-fitting
methodology (no cross-lab or cross-modality mismatch), this is evidence the elevation problem
is not really "the atlases are wrong" so much as elevation gradients being intrinsically harder
to pin down from local data by any method, while azimuth's problem really is specific to the
(widefield-imaging-derived) atlases.

## Part 4 — Population-level registration of the naive map to the atlases

Individual-session landmark counts (4-6 probe/area medians) are too sparse to fit a full
6-parameter affine reliably — a standing concern in every prior single-session Zhuang fit's own
documentation. Pooling the naive map's full spatial coverage into many more, much
less noisy landmarks should do better: `scripts/register_naive_map_to_atlases.py` bins the
naive pooled cells onto a 150 µm CCF grid (≥8 cells/landmark), giving 173 landmarks (vs. 4-6 per
session), and fits a CCF↔atlas-frame affine plus a free 2-parameter RF offset (needed because
the naive map's values are V1-relative, not absolute).

### Two real bugs found via visual inspection, not just numbers

1. **Wrong reflection, chosen by a coin-flip-thin margin.** The population fit's RF-only
   objective (no anatomical area-membership penalty) picked `reflection=-1` at objective 1.038
   vs. `reflection=+1` at 1.111 — a narrow, spurious margin. The properly anatomy-constrained
   14-animal single-session fit is unambiguous on this: `reflection=+1` scores 1.08 vs. `-1`'s
   9.68, a ~9x gap. RF-value agreement alone barely distinguishes true handedness from its
   mirror image; anatomical compartment membership does. Confirmed only by rendering the
   result and recognizing a left-right-flipped map by eye.
2. **Forcing the correct reflection alone was not enough.** Without an area-membership penalty
   at all, the optimizer found a *different*, wildly over-elongated degenerate affine at a
   similarly "good" RF-only objective — visually obvious as a thin sliver instead of a cortex
   shape once rendered. Properly fixed by adding a vectorized area-membership penalty
   (`build_zhuang_area_penalty()`, using `template["area_distance"]`), matching
   `register_allen_session_to_zhuang.py::fit_candidate`'s own established convention. A
   first version of this penalty used a **per-landmark Python loop calling the interpolator
   173 times per objective evaluation**, which combined with `differential_evolution`'s
   thousands of evaluations timed out at the 300s wall-clock limit; vectorized by grouping
   landmarks by area and calling each area's interpolator once per group (~5 calls instead of
   173), a ~1250x speedup (0.8 ms/eval afterward).

This population-level full-affine line of work was set aside (not fully re-run to convergence)
in favor of a simpler, more directly inspectable approach — see Part 5 — but the two bugs above
are exactly the kind of failure mode ("looks plausible on a summary number, wrong on inspection")
this project has repeatedly guarded against, so they're recorded here rather than discarded.

## Part 5 — The pragmatic "default rough registration"

Rather than a black-box multi-parameter optimizer, build placement from a small number of
independently-checkable pieces, refined one at a time by direct visual inspection:

1. **True physical scale**, not a bounding-box stretch. Zhuang: 104.6 px/mm, from the Figure 3
   scale bar calibration already established in `render_anatomy_constrained_cell_mapping.py`
   (`ZHUANG_FIG3_SCALE_BAR_PX=62.0`, `ZHUANG_FIG3_SCALE_BAR_MM=0.5`,
   `FIG3_TO_FIG9_SIMILARITY_SCALE=0.8432313316638625`). Garrett has no independent scale bar
   (its own README: "absolute millimetre scale remain[s] to be established") — first attempt
   matched Garrett's V1-mask size against the naive map's own scattered V1 cells directly
   (0.49 panel-units/mm) and came out visibly ~2x too small (a filled 2D mask isn't a fair size
   comparison against sparse sampled points). Fixed by cross-calibrating Garrett's V1 mask
   against **Zhuang's own V1 compartment** (which does have a true scale bar): 0.259
   panel-units/mm, implying a ~3.85 mm domain width, consistent with Zhuang's own ~4.4 mm
   calibrated width.
2. **V1 anchor**, not a fitted center. Zhuang: `AREA_SEEDS_XY["VISp"]` pixel seed matched to
   the naive map's own median V1 CCF position. Garrett: its coordinate origin *is* the V1
   centroid by construction (per its own template README), so no seed lookup is needed at all.
3. **One open reflection flag**, locked by direct visual comparison of mirrored vs. unmirrored
   placements against the source figure and the rendered domain — not re-derived by an
   under-constrained optimizer (see Part 4).
4. **Color-scale calibration** via one direct anchor point (Zhuang's/Garrett's own predicted
   value at the V1 anchor position added to the naive map's V1-relative values), not a fitted
   intercept.

This became the checked-in "default registration" between the ephys data and each atlas.
Scripts: `scripts/render_naive_map_over_zhuang_rough_bbox.py` (writes the locked reference
`Figure_naive_pooled_cells_over_zhuang_true_scale_mirrored.png`),
`scripts/render_naive_map_over_garrett_rough_bbox.py`.

### A real Zhuang extraction bug, found and fixed via this overlay

Rendering the naive map over Zhuang at true scale (rather than a bounding-box stretch that had
been silently masking the problem) revealed a white gap in the domain, plus cell clusters
falling outside the atlas outline entirely — both symptoms of (a) a real digitization defect
and (b) the still-missing rotation (Part 6).

Diagnosis path (`scripts/patch_zhuang_domain_narrow_waist.py`, iterated three times):
- The raw boundary-derived domain (`ndimage.binary_fill_holes` of the digitized boundary line)
  was already topologically **one connected component** (101,780 px) — not a broken extraction
  in the strict sense. But eroding it by just 1 px split it into 86 fragments; a column scan
  showed the only link across a ~24-column span was two threads only 1-3 px wide — too thin to
  survive any real smoothing (`normalized_smooth` uses sigma=2px, comparable to the thread
  width itself), so it behaved as disconnected downstream even though it technically wasn't.
- v1 fix (rectangular dilation over a guessed pinch window) and v2 (a hand-drawn line between
  user-specified pixel coordinates, read directly off a rendered figure's axes) each targeted
  the wrong specific location, corrected through two rounds of user-guided coordinate
  refinement — confirmed correct once it filled exactly the V-notch pocket the user intended,
  deliberately *not* chasing the full shape of a separate thin lobe further out ("we don't need
  to include the full extended lobe").
- v3 (final, principled fix): rather than guessing a line, checked whether the raw sparse
  digitized contour points themselves (`azimuth_deg`/`altitude_deg` in
  `retinotopy_contour_grid.npz`, as opposed to the boundary-*line*-derived domain) already cover
  the gap region. They did — 1852 of 2405 sparse azimuth points and 495 of 793 sparse altitude
  points in the affected region fell **outside** the boundary-derived domain, direct evidence
  Zhuang's figure has real mapped cortex there that the boundary-line extraction simply failed
  to enclose. Domain redefined as
  `fill_holes(boundary_domain OR closing(dilation(has_any_contour_point, r=5), r=2))`. Result:
  domain grew from 101,780 to 124,004 px and, critically, stayed as **1 component even after
  1px erosion** (vs. 86 fragile fragments before) — a categorically more robust fix than either
  manual line.

Patched field saved to `artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa/interpolated_fields_and_field_sign_domain_patched.npz`
(original unpatched version untouched alongside it). The locked reference figure
(`Figure_naive_pooled_cells_over_zhuang_true_scale_mirrored.png`) now uses the patched field.

## Part 6 — Does the rough registration actually place areas correctly?

Direct check, not just a visual impression: `scripts/check_probe_area_labels_vs_zhuang_registration.py`
transforms every probe's own median CCF position into Zhuang pixel space under the default
(translation + true scale + mirror, **no rotation**) placement, and checks whether the nearest
named Zhuang compartment matches that probe's own recorded Allen area label.

| area | probes | agreement |
|---|---:|---:|
| VISp (V1) | 46 | 97.8% |
| VISrl (RL) | 41 | 90.2% |
| VISam (AM) | 43 | 74.4% |
| VISl (LM) | 28 | **21.4%** |
| VISal (AL) | 40 | **20.0%** |
| overall | 198 | 64.6% |

Not random noise: the rendered overlay shows VISl-labeled probes clustering almost entirely on
top of the VISal cluster, near the "AL" label rather than "LM" — a systematic swap between two
*neighboring* areas, exactly the signature of a missing rotation (V1/RL/AM sit in directions
where translation+scale+mirror alone still lands close enough; LM/AL sit in a direction where
the angular error is large enough to cross into the wrong neighboring compartment).

## Part 7 — Fitting translation + rotation only (scale and reflection fixed)

`scripts/fit_translation_rotation_naive_to_zhuang.py`: 5 free parameters only (rotation angle,
a small translation on top of the V1-anchor starting point, and the 2-parameter RF offset) —
scale fixed at Zhuang's true 104.6 px/mm, reflection fixed at the already-locked mirror. Uses
the same 173-landmark naive-map binning as Part 4, plus the vectorized area-membership penalty.

### A real axis-order bug, caught by the optimizer running away to the bound

First attempt: unconstrained rotation bound (±90°) converged to −60.2° with the translation
parameter pinned exactly at its −80 px bound, objective=118.96 (vs. ~1-3 for every reasonable
fit elsewhere in this work), and area agreement **worse** than the no-rotation baseline (21%
vs. 65%). Narrowing the rotation bound to ±30° didn't fix it — both rotation and translation
still pinned at their new bounds, objective got *worse* (133). Two parameters simultaneously
pinned at bounds while the objective degrades under a *tighter* search space is a strong signal
of a bug, not a hard local optimum.

Found it: `ccf` columns are ordered `[ap, ml]`, but `pixel_center` (and everything downstream in
`sample_template`) is ordered `[col, row]` = `[ml-ish, ap-ish]`. The scale/reflection matrix
preserved input order (ap→dim0, ml→dim1) instead of being built for `[ml, ap]` input — an
accidental full AP↔ML axis swap. The optimizer's repeated push toward large rotation angles was
it trying (unsuccessfully, within the bound) to claw back toward the ~90° rotation that would
approximately undo an axis swap. Fixed by reordering the CCF delta to `[delta_ml, delta_ap]`
before applying the scale/reflection matrix. Sanity check: with the fix and θ=0 (pure
translation), area agreement reproduces ~61%, matching the independently-computed baseline
(64.6%) closely — confirms the transform is now equivalent to the known-good placement at the
no-rotation point, as it must be.

### Result, after the fix

Rotation converges to a small, plausible **−8.1°** (no bounds pinned), objective 2.94 (in the
same range as every other well-behaved fit in this project), median vector error 18.4°.

| area | translation-only | translation+rotation |
|---|---:|---:|
| VISp | 97.8% | 95.5% |
| VISrl | 90.2% | 77.1% |
| VISam | 74.4% | 51.9% |
| **VISl** | **21.4%** | **61.5%** |
| **VISal** | **20.0%** | **39.3%** |
| overall | 64.6% | 68.8% |

The LM/AL confusion improved substantially (LM roughly tripled, AL roughly doubled) at the cost
of some decline in RL/AM — a real trade-off, not a universal win, but concentrated exactly where
the no-rotation diagnostic predicted it should be, from a rotation small enough to be a
plausible genuine CCF-axis-vs-figure-orientation mismatch rather than a fitting artifact.
Overlay: `Figure_naive_pooled_cells_over_zhuang_rotation_fit.png`. Fit parameters and full
per-area breakdown: `translation_rotation_fit_manifest.json`.

## What's established and what isn't, right now

**Established**
- The local-linear-Jacobian decoupled shape/translation approach is far more internally
  reproducible than the old coupled EM (3.4° vs. ~21° median split-half distance).
- Zhuang's weak elevation-gradient agreement is a real range/blur/digitization-plateau effect,
  not a sign-convention bug (tested directly, not just argued).
- An atlas-free, purely empirical V1-anchored cross-session pooling of this project's own data
  tracks local azimuth gradients far better than either published atlas, and ties them on
  elevation — evidence the naive map is a *better* azimuth shape prior than Zhuang or Garrett,
  not merely a fallback.
- The default (translation + true scale + fixed mirror, no rotation) registration measurably
  mis-locates LM/AL into each other's compartment (~20% agreement); adding a small fitted
  rotation (−8.1°) recovers most of that (LM→61.5%, AL→39.3%) at a real but smaller cost to
  RL/AM (overall 64.6%→68.8%).
- The Zhuang domain extraction had a real, now-fixed defect: a fragile 1-3px-wide connection
  that behaved as disconnected under any real smoothing, and — more substantially — a
  boundary-line-derived domain that excluded a region the raw digitized contour points
  themselves show is real mapped cortex.

**Not established / open**
- Elevation still has no atlas or atlas-free method that agrees well with local ephys gradients
  at the single-session level (~50-55° median angle everywhere tested); this looks like a
  harder, more fundamental problem than atlas quality.
- The translation+rotation fit (Part 7) has not been extended to Garrett, and has not been
  re-run with a wider/joint search including scale as a sanity check on the "fixed true scale"
  assumption.
- Per-area agreement after the rotation fit is still well short of 100% even for the previously-
  good areas (RL 77%, AM 52%) — whether further gains need shear, a spatially-varying warp, or
  are limited by genuine animal-to-animal variability (the actual scientific question) is unresolved.
- The population-level full-affine fit (Part 4) was never re-run to a trustworthy convergence
  after its two bugs were fixed; the constrained translation+rotation fit (Part 7) superseded it
  for now but a properly-constrained full-affine (or affine decomposed into scale/shear/rotation
  with sensible independent bounds on each) is still open.

## Reproducibility index

Scripts, in the order referenced above (all under `scripts/`):
`fit_multistructure_fixed_effect_translation.py`,
`test_elevation_convention_zhuang_registration.py`,
`compare_atlas_gradient_to_ephys_pilot_781842082.py`,
`build_naive_v1_centered_cross_session_rf_map.py`,
`compare_three_way_gradient_781842082.py`,
`build_garrett2014_smoothed_field_and_ccf_affine.py`,
`register_naive_map_to_atlases.py`,
`render_naive_map_over_zhuang_rough_bbox.py`,
`render_naive_map_over_garrett_rough_bbox.py`,
`patch_zhuang_domain_narrow_waist.py`,
`check_probe_area_labels_vs_zhuang_registration.py`,
`fit_translation_rotation_naive_to_zhuang.py`,
`render_naive_map_over_zhuang_rotation_fit.py`.

Key artifacts:
`artifacts/v1_absolute_size_dispersion_translation_checkpoint/multistructure_fixed_effect_translation/`,
`artifacts/v1_absolute_size_dispersion_translation_checkpoint/naive_v1_centered_cross_session_map/`,
`artifacts/retinotopy_template/atlas_gradient_vs_ephys_pilot_781842082/`,
`artifacts/retinotopy_template/naive_map_registered_to_atlases/` (the main working directory for
Parts 4-7: landmark tables, fit manifests, all overlay figures),
`artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa/` (smoothed field,
domain-patch manifest, both patched and original npz).
