# Figure 4 rerun: supporting methods and audit detail

Updated planning checkpoint: 17 September 2026. Read [the execution plan](Figure4_rerun_analysis_plan.md) first; it fixes scope, the primary estimand, and the resampling algorithm. This appendix supplies implementation detail, not an additional mandatory work program. This document specifies proposed work; it does not report a completed rerun. Preserve the existing candidates as exploratory checkpoints. Resolve the explicitly listed methodological decisions before inspecting new effect estimates or significance labels.

## 1. Scientific questions and permitted conclusions

The central question is whether differentiation among the five cortical HVAs is greater than differentiation among recording locations within V1, for TTFS, phase modulation, and response timescale. The new hierarchy question is whether the previously reported ordered relationship remains after accounting for receptive-field position. The original paper already tested the unadjusted hierarchy relationship; reproducing it with our current metrics is a verification step, not a separate headline discovery or a requirement to re-establish the original paper.

Separate the new comparisons from the necessary reference checks:

1. **Coarse cortical contrast:** Allen V1 versus pooled cortical HVAs in eligible shared recordings. This is a reference/control comparison, not a prerequisite that must be significant for every metric.
2. **Categorical differentiation:** variation associated with the four MouseV2 V1 location labels, and separately with LM, RL, AL, PM, and AM. The primary comparison is HVA minus V1 differentiation.
3. **Published-hierarchy verification:** check the previously reported relationship using our current metric definitions and population. Document any changes from the original paper. This supplies a reference for RF adjustment, not a new primary hypothesis.
4. **RF-conditional ordered relationship:** the contribution of hierarchy after accounting for RF position, and its change relative to the same-cell unadjusted result.

These questions are not interchangeable. An area effect can be strong without following hierarchy; a weak area effect can nevertheless be consistently ordered. Comparable categorical effects in V1 and HVAs do not establish the absence of hierarchy. A significant V1–HVA contrast does not establish an ordered relationship among HVAs.

We will not select methods to obtain nonsignificance. Report a precisely small effect differently from an imprecisely estimated effect. The intended paper claims concern these measured response properties and sampled populations, not every possible form of cortical functional hierarchy.

## 2. Freeze the evidence and reproduce the current discrepancies

**Work.** Archive candidate figures, analysis inputs, statistics, seeds, code revision, and source paths. Run the existing harmonization and TTFS RF-attrition audits. Save machine-readable counts and a short discrepancy ledger. Add a run manifest tying every output to its source tables, eligibility rules, transformations, and model configuration.

**Why.** Changes to source metrics, sampling, RF eligibility, modeling, and plotting otherwise become indistinguishable. A reproducible baseline lets us attribute each change rather than explaining an improved-looking final figure after the fact.

Known discrepancies to reproduce:

- Central used native Allen metrics and broader QC instead of the harmonized Allen cohort. In 1,948 identical neurons, mean log10 F1/F0 changed from −0.200 native to −0.022 harmonized.
- All 1,090 original MouseV2 TTFS cells have raw RF records; only 515 have entries in the older parametric-fit table. Only 189 pass its support rules and 158 survive population floors. Missing fits are not failed fits.
- Current single-cell timescale figures average available valid matched draws and omit the earlier draw-level uncertainty handling.
- The updated control measurements still inherit session eligibility from an older figure.

**Output/checkpoint.** Frozen manifest and discrepancy ledger, with a source-level example for each problem. Existing audit assertions must reproduce before changing loaders.

## 3. Establish the experimental structure and population target

**Work.** Build a registry containing source dataset, animal/specimen, session, physical probe, anatomical area/location, unit ID, and metric availability. Verify unique joins and distinguish one probe traversing several areas from genuinely independent probes. Identify repeated recordings from the same animal. Tabulate recordings in which each pair of categories co-occurs.

For each metric and category, report animals, sessions, probes, neurons, and neurons per recording. Keep anatomical assignment and RF eligibility as separate columns. Reapply the accepted MouseV2 VISp anatomical restriction consistently.

**Why.** A cell is the response observation, but cells from the same recording are not independent experimental replications. The model and resampling unit must follow the actual design. A probe-specific effect cannot always be separated from a session-by-area effect; we must not fit two indistinguishable components merely because both have names.

**Population target.** Use the four MouseV2 locations as the primary within-V1 comparator. Retain the Central-inclusive five-location analysis as an explicit complementary result, as requested. Keep Central in the middle of the V1 display, but identify its Allen origin. Confirm the provenance of the name “Central”; do not imply anatomical or RF centrality more precisely than the metadata justify.

Central and dataset coincide within the V1 comparison. Harmonization reduces processing differences but does not identify separate Central and cohort effects. A model with both unrestricted Central and dataset indicators cannot resolve this confounding. Shared Allen V1/HVA observations help estimate within-Allen contrasts but do not remove it without additional assumptions.

**Design checkpoint.** Record a diagram of the actual nesting/crossing. The existing audit already shows multiple locations in a MouseV2 session (session 2 contains A/E/C/B); do not assume one location per session. Assess retained within-session contrasts after filtering. Apply the execution plan's support gates rather than inventing a universal sessions-per-location threshold. The prior Allen RF audit found one specimen per session in its population; verify this again for the reconstructed population and independently for MouseV2.

## 4. Repair and unify metric construction

**Work.** Create one provenance-checked loader used by the four-location, Central-inclusive, control, RF-qualified, and RF-adjusted analyses. Do not let each figure script independently choose a source or QC profile.

- **Common waveform QC:** preserve the accepted amplitude-cutoff, presence-ratio, and ISI rules, including exact inequalities and missing-value behavior. The prior contract uses amplitude cutoff <0.1, presence ratio >0.8, and ISI violations <0.5; confirm the source fields before encoding them.
- **TTFS:** use the accepted response-selected preferred-polarity estimator and identical response-selection/validity definitions. Retain the documented <100-ms population initially, and label the inference as conditional on that truncation. Check counts near the cutoff. Independently calibrated physical light onset remains unresolved; do not interpret a cross-dataset absolute offset as physiology.
- **F1/F0:** retain conventional harmonized processing, the 1-s window, 15 trials per orientation×TF condition, orientations 0/45/90/135°, TFs 1/2/4/8/15 Hz, SF 0.04 cycles/degree, and contrast 0.8. Use the eligible complete Allen Brain Observatory sessions. Apply log10 only to valid positive values, and report excluded zeros/nonfinite values rather than adding an arbitrary offset.
- **Timescale:** preserve the matched 75-bright/75-dark MouseV2 flash fits and the accepted Allen estimator. Retain the documented 1–300-ms validity range, >50-spike rule, and fit-error <20-ms rule initially. Verify that any additional source-specific validity decisions are exposed in the manifest.
- **Central:** use harmonized Allen V1 metric sources and common QC. Include every eligible metric-specific Central session, not just the subset shared with the pooled-HVA control.

Recompute all control/session eligibility from current harmonized tables. Do not inherit the legacy figure's session list. Preserve metric-specific populations rather than requiring every neuron to have all three response metrics.

**Why.** The same-cell F1/F0 discrepancy demonstrates that processing alone can create an apparent spatial effect. Common code prevents the same regression in the next figure variant.

**Output/checkpoint.** A metric contract and per-neuron audit table. Require unique IDs, explicit reasons for exclusion, and numerical agreement with accepted source values. Explain discrepancies before fitting new models.

## 5. Restore draw-aware timescale analysis

**Work.** Reproduce the earlier approach in which the ten matched MouseV2 trial draws are analyzed separately rather than treated as extra neurons or collapsed silently. During resampling, select a matched draw coherently for a resampled MouseV2 session and carry its associated cell-validity mask. Maintain shared draws wherever comparisons reuse the same observations.

Use the execution plan's explicit algorithm: average ten draw-specific effect estimates for the point estimate; for each outer independent-unit bootstrap, select a matched draw for each sampled MouseV2 session occurrence and carry its entire validity mask and all dependent comparisons together. Compute the effect before averaging, including nonlinear dispersion and normalized summaries. Use no additional Rubin-rule variance term: these are trial subsamples, not proper multiple imputations. The ten draws are Monte Carlo versions of one recording, not ten independent biological observations. Their spread captures the trial-subsampling uncertainty represented by this procedure, not every source of measurement error. Check whether comparable trial-level uncertainty can be estimated for Allen; otherwise state the asymmetry.

Keep the previously informative sensitivity populations: cells valid in all ten draws and the matched <10-ms fit-error gate. Do not promote the tighter error gate to the primary analysis simply because its result changes: fit error and timescale are coupled, so that gate selectively removes longer estimates. Report draw-specific retention and distribution changes.

**Why.** Of the current 773 MouseV2 timescale neurons, 351 have fewer than ten valid fits and 82 have only one. Equal weighting of per-cell averages changes the target and omits a previously important source of uncertainty.

**Output/checkpoint.** Draw-level counts, distribution overlays, and reproducible estimation rules. If mean-per-neuron dots are retained for readability, label them as a display summary rather than implying they exactly reproduce the inferential weighting.

## 6. Complete and validate the RF population

**Work.** Fit the missing MouseV2 RFs for the current response-analysis population using the documented RF pipeline. Audit lateral and medial probes explicitly. Distinguish no raw RF data, no fit attempted, failed fit, unsupported fit, successful eligible fit, and removal by a population-count rule.

Inspect representative accepted and rejected fits before accepting aggregate coverage. Select examples transparently: missing legacy fits, newly accepted fits, borderline reliability, off-screen/edge fits, and disagreement between a raw peak and a fitted center. RF coordinates remain cell-specific; do not substitute probe-average RFs without making that a separate analysis.

For Allen, document the released estimator and quality gates. Determine whether an equivalent refit is possible from available data. If not, validate comparability of the two RF pipelines and report their different error/selection properties. Identical threshold numbers on different quality statistics would not establish equivalence.

Validate axis signs, coordinate origins, screen geometry, and the current MouseV2 +50°/+10° conversion. A coordinate translation is not gaze correction. Treat absolute display coordinates and session-V1-relative coordinates as distinct analysis choices; the latter changes the spatial question and requires reliable V1 anchors.

**Why.** The two-session lateral/medial result partly reflected incomplete fit coverage, not absence of RF measurements. RF selection can change the response population before any adjustment is performed.

**Output/checkpoint.** Cell-level RF provenance and recording-level attrition tables. Do not silently relax quality gates to restore sample size. If comparable RF measurements cannot be established, keep the cross-dataset RF comparison explicitly exploratory.

## 7. Separate selection, support restriction, and RF adjustment

Construct a population ladder for each metric:

1. All harmonized response-eligible cells.
2. RF-qualified cells, still using unadjusted responses.
3. Cells on the prespecified shared RF support, still unadjusted.
4. Exactly the cells in step 3 analyzed with RF adjustment.

The difference between 1 and 2 reflects RF selection; 2 to 3 reflects support restriction; only 3 to 4 isolates the statistical adjustment on unchanged observations. Use paired resampling for every before/after comparison.

Map RF coverage by category and recording. Review the earlier intersection of 2.5th–97.5th percentile RF boxes as a concrete starting point, but check actual two-dimensional occupancy: overlapping boxes do not ensure local data overlap. Choose a support rule before examining the new hierarchy result. A restriction chosen for model evaluation must be learned inside training folds, or otherwise be fixed independently of the evaluation data.

Use the same support definition for category effects that will be directly compared. If no adequate common support exists, say the common-support comparison is not estimable. Report broader-support or pairwise-overlap analyses separately; they answer different questions and must not replace an unidentifiable primary result.

Review minimum-cell floors symmetrically. Existing RF TTFS floors differ between V1 and HVAs (5 versus 10). Propose a common rule based on estimator stability and inspect prespecified alternative floors, rather than selecting a rule by significance. Report the independent-recording support as well as cell counts.

## 8. Specify the single-cell model and the weighting target

**Starting model.** For each metric, represent an individual response as a category-associated mean plus RF terms when applicable, animal/session effects supported by the registry, a recording-population deviation where identifiable, and cell residual variation. Treat the named locations/areas as fixed categories for the primary question about these specific categories; do not automatically interpret five areas as a random sample of all possible cortical areas.

Use separate cohort-specific nuisance and residual structures where needed. Do not force MouseV2 and Allen to have identical residual variance simply because response estimators are harmonized. Check heteroscedasticity, residual shapes, singular fits, and instability in session/probe components. Avoid unsupported random-effect complexity.

**Primary weighting.** Target equal category weights, with equal independent-recording contribution within each category and equal cell weights within each eligible recording population. This asks about the typical sampled category/recording rather than the typical neuron in the pooled yield. Implement the target through explicit weighting or standardized model summaries; do not assume a random intercept automatically creates equal recording weights. Retain the existing equal-cell target as a sensitivity analysis. This target is fixed in the execution plan; serialize its implementation before reviewing effects.

**Why.** Cell yield otherwise changes the influence of sessions and areas. The model must separate stable category differences from recording-specific shifts while keeping the observational resolution at the cell level.

Before the full rerun, fit a minimal model to a few inspected recordings and validate it on simulations with the actual missing-category and cluster-size structure. Use one small simulation harness with two conditions: no category effect and a known category effect, both with the actual cluster-size/missing-category structure and recording shifts. Broaden validation only if a concrete failure warrants it. These checks test implementation and recoverability; they do not prove all real-data assumptions.

## 9. Define an effect measure that remains comparable after RF adjustment

Do not carry forward the current switch from total-variance omega-squared to partial omega-squared without distinction. The current RF-adjusted denominator is variance left after RF, so its percentage can increase when that denominator shrinks.

The execution plan fixes Figure E to equal-category dispersion of standardized category means in squared response units, and its signed HVA-minus-V1 difference. A total-variance-normalized version is a companion, not an alternative chosen after seeing results. Report these separately from the optional predictive diagnostic:

- **Category contrasts and their dispersion in response units:** standardized category means and the weighted dispersion of those means, with uncertainty. Use a common RF reference distribution for adjusted means and a fixed weighting target. Preserve this quantity in squared response units as well as any normalized version.
- **Incremental predictive contribution on a fixed total-variance scale:** for matched observations and weights, compare reduced-model and full-model prediction errors and divide their difference by the same raw-response total sum of squares. For example, the RF-conditional category increment is `(SSE_RF − SSE_RF+category) / SST_raw`. The unadjusted increment uses the corresponding nuisance-only versus nuisance-plus-category models. All models retain the same recording/nuisance design.

These are not classical bias-corrected omega-squared. The primary dispersion and its normalization are model-standardized descriptive effects, not automatically unbiased variance components or predictive variance explained. Check finite-sample inflation in the small validation harness. Held-out prediction is supplementary and conditional on adequate recording support; do not change the headline estimand if predictive estimates are noisy. An in-sample increment favors the larger model. Held-out increments can be negative and must not be clipped.

Use the same definition and weighting for V1 and HVA effects. Each population's fraction has its own total response variance, so compare the raw-unit effects and denominators as well: comparable fractions alone do not imply identical absolute differentiation. For before/after RF comparisons within a population, hold the denominator and cells fixed.

## 10. Estimate the V1–HVA categorical comparison and the control

Fit the four-location V1 and five-area HVA models using the shared specification. Compute `Delta_category = effect_HVA − effect_V1` within every joint resample. Report signed differences and intervals; avoid relying on ratios when the V1 effect is near zero.

Run the Central-inclusive analysis alongside it, clearly identifying cohort confounding. Preserve dependence when the same Allen session contributes Central, HVAs, and the coarse control. Do not independently resample reused Allen observations.

Rebuild the coarse control from eligible matched Allen recordings. Specify the pooled-HVA weights explicitly; use the chosen target rather than accidentally giving high-yield areas more influence. Estimate V1 versus pooled HVAs with recording structure accounted for. A nonsignificant control is a result with uncertainty, not an automatic analysis failure or proof that the metric lacks spatial structure.

Remove the gray difference-of-shuffles reference from the main contrast. Subtracting two no-label-effect shuffle distributions does not test equality of two potentially nonzero effects. Within-session label shuffling can remain a secondary restricted check, with its null stated accurately; it is not the primary general-category test and cannot provide a five-group Central test.

## 11. Verify the published baseline, then estimate what remains after RF adjustment

First identify the original paper's exact population, metrics, aggregation, and hierarchy analysis. Reuse an existing validated reproduction if those inputs and methods are unchanged. Where our harmonization, outcome definitions, or population differ, verify the relationship using the current inputs and explicitly document the differences; do not describe a changed-population check as an exact replication. Investigate discrepancies rather than requiring the original significance label to recur.

For the new RF-conditional HVA-only analysis, use LM, RL, AL, PM, and AM. Verify and cite the original provenance of the fixed scores already used in the repository: LM −0.093, RL −0.059, AL 0.152, PM 0.327, AM 0.441. Do not reorder areas, refit scores to these responses, or add V1 to strengthen an HVA-only association. The hierarchy predictor varies at area level; thousands of neurons do not create thousands of independent hierarchy-score values.

The same-cell unadjusted fit is still necessary even when the published result has been reproduced: RF selection may change the baseline relationship. On the identical RF-qualified/common-support population, fit the following models to quantify adjustment and its interpretation, not to launch another standalone test of the published hierarchy:

- nuisance structure only;
- nuisance + hierarchy score;
- nuisance + RF position;
- nuisance + RF position + hierarchy score;
- nuisance + categorical area;
- nuisance + RF position + categorical area.

Use an intercept and signed hierarchy slope; do not constrain its direction to the hoped-for result. Model hierarchy and RF jointly. Regressing RF out of the outcome and correlating the residuals with the original hierarchy score is not the intended conditional model when score and RF position covary.

The categorical model contains the linear hierarchy pattern as one possible configuration of area means. Use categorical versus hierarchy model performance as a secondary diagnostic of whether additional non-ordered area differences improve prediction; this is not an additional headline aim. Do not include unrestricted area fixed effects and the area-level hierarchy score as independently identifiable terms in the same model.

The shared RF surface assumes the relationship supported by within-category RF variation applies when standardizing between-category differences. Inspect within-area-centered RF features and their area means, including the quadratic features if used. Compare within- and between-area patterns as a diagnostic, recognizing that five area means cannot support a rich between-area model. Five areas alone do not prove non-identifiability, and many cells alone do not resolve area–RF confounding. Apply the execution plan's overlap and rank/stability gates. Start RF adjustment with the documented low-complexity surface, after checking its suitability. If tuning or a more flexible surface is necessary, perform selection only in training data, and apply the same selection procedure across relevant models. Report the degree of score–RF confounding and the uncertainty it creates. A weakly identified conditional slope is not evidence for zero hierarchy.

Report the unadjusted and adjusted slopes, standardized changes across the observed HVA score range, and hierarchy predictive increments. Estimate slope change and change in predictive contribution directly within paired resamples. A transition from significant to nonsignificant does not establish attenuation. RF-associated attenuation is statistical sharing of explanatory information, not proof that RF position causally mediates hierarchy.

## 12. Uncertainty, validation, and sensitivity analyses

Resample the highest independent unit supported by the registry, preserving all nested recordings and cells. If each session is a different animal, a session bootstrap is appropriate to that structure; if animals repeat, preserve their sessions together. Shared observations and timescale draws must be handled jointly across contrasts.

Hold out complete independent recording units for predictive validation. Predictions for unseen recordings must not use response-derived random effects from the held-out unit. Any centering, RF anchoring, scaling, support estimation, or tuning that uses held-out responses must be eliminated or explicitly recognized as a different transductive estimand. Use matched folds for nested-model comparisons. Keep training and test category support checks explicit.

Core sensitivities are items 1, 2, 4, and 5 below. Item 3 is limited to the predefined count-floor alternative. Items 6 and 7 are targeted follow-up if the primary diagnostics identify a concrete problem, not mandatory new analysis programs:

1. Equal-cell versus category/recording-balanced target.
2. Leave-one-independent-recording-out influence, especially for MouseV2 and depleted RF populations.
3. Common minimum-count rules and their predefined alternatives.
4. Full response population versus RF-qualified versus common-support populations.
5. Timescale complete-draw and tighter-error populations, with the selection consequences shown.
6. Plausible RF-coordinate/model alternatives and RF-fit uncertainty where estimable.
7. Measured depth/layer or cell-class composition differences, if metadata support a meaningful check. These are additional conditional analyses, not silent changes to the primary population.

Do not interpret five HVA categories as evidence about arbitrary unseen areas. A leave-one-area-out diagnostic may reveal leverage by PM or AM, but changes the target and is not interchangeable with leaving out recordings.

## 13. Decide the inferential rules before the rerun

The primary output is numerical effects and 95% pointwise uncertainty intervals, clearly labeled as such. The three categorical contrasts are reported together; RF-conditional hierarchy and attenuation estimates are a distinct set. Published-hierarchy reproduction is a reference check. Do not imply simultaneous coverage or use interval crossing as an equivalence test.

Formal equivalence testing is out of scope: there is no agreed margin. Report the compatible effect range in interpretable units. Do not add stars by converting bootstrap sign fractions into p-values. If stars are retained for the requested display, first freeze and validate a test and its multiplicity treatment; they are secondary and cannot delay or replace the interval-based result. Variance/dispersion estimates near zero are nonregular: document the null-case validation and do not equate an ordinary percentile interval for a nonnegative plug-in dispersion with evidence for a nonzero effect.

## 14. Reconcile earlier analyses explicitly

The prior RF validation report found a reproducible timescale hierarchy signal on its restricted support and did not find an advantage of the categorical model over the hierarchy model. The later timescale population-sensitivity checkpoint did not support a robust systematic HVA hierarchy across fit-population definitions. Neither result should be ignored.

Create a comparison ledger listing population, anatomical filter, RF support, response aggregation, timescale draw handling, weighting, model, evaluation unit, and estimand for each checkpoint and the rerun. Reuse saved validated results where possible; rerun an old analysis only if needed to resolve a specific discrepancy. Change one substantive component at a time where feasible. The old session-mean held-out R² and the new single-cell categorical effect are different quantities; their numerical values need not agree.

**Why.** This prevents the new narrative from overwriting evidence that supports an ordered effect under some conditions, or from treating sensitivity-dependent absence of evidence as universal absence.

## 15. Figure and reporting plan

Preserve C/D as neuron-level distribution panels, with the appropriate inferential summaries beside each population. Keep vertical response axes and column-oriented effect displays. Retain the extra whitespace between D and E. Keep the coarse control and main HVA-minus-V1 comparison together in E.

Show the primary four-location comparison and the complementary Central-inclusive version without silently substituting one for the other. Show RF-selection and same-cell adjusted/unadjusted comparisons in a companion figure or clearly separated panels. If displayed, the hierarchy panel should focus on the RF adjustment: show the same-cell unadjusted relationship as the reference, the adjusted effect and paired change, and the RF-conditional predictive contribution. Put detailed reproduction checks in the methods or supplement rather than presenting unadjusted hierarchy as a new headline result.

Display cells and independent-recording counts, analysis population, weighting, units, and uncertainty type. Put the explanation of bias correction in the legend rather than a long panel title if omega-squared remains in any companion. If the new primary metric is predictive R² or a model-standardized variance measure, use that name and explain its negative values correctly; do not retain an omega-squared caption from the old figure.

Describe allowed outcomes explicitly:

- Positive, well-supported HVA-minus-V1 contrast: more differentiation across these HVAs than across sampled V1 locations under this metric/target.
- Narrow interval near zero: report the numerical bound on the difference; do not claim formal equivalence.
- Broad interval spanning zero: comparison unresolved.
- Hierarchy persists after RF adjustment: ordered information remains conditional on measured RF position.
- Demonstrated attenuation with a tightly small adjusted effect: hierarchy-associated variation is largely shared with RF position under the specified model/population.
- Adjusted nonsignificance with broad uncertainty: residual hierarchy unresolved.

## 16. Execution order and completion criteria

1. Freeze current evidence and produce the recording registry.
2. Repair shared metric loaders, Central, and current eligibility; verify source parity.
3. Complete and inspect missing RF fits; establish coordinate and selection comparability.
4. Present population/support counts and model feasibility; implement the fixed weighting and estimand and resolve only genuinely data-dependent design details.
5. Validate the minimal model and draw-aware resampling on concrete recordings and synthetic cases.
6. Verify the published hierarchy baseline where inputs changed; serialize the full configuration and source checksums and record its SHA-256 before the primary run.
7. Run primary models, paired contrasts, and core sensitivities. Grouped predictive validation and extra covariate analyses are optional follow-up triggered by a specific concern.
8. Reconcile historical results and document remaining limitations.
9. Generate revised figures, source tables, model diagnostics, and a methods/legend draft.

Completion requires traceable sources, no unexplained join losses, verified recording structure, explicit denominator/weighting definitions, coherent shared resampling, successful computational checks, and conclusions that match the interval precision and sensitivity results. It does not require any particular significance pattern. If RF support or recording replication is inadequate, report that limitation as an outcome rather than progressively weakening filters until a result appears.

## Local evidence and implementation starting points

- [Harmonization audit](Figure4_harmonization_audit.md)
- [TTFS RF attrition audit](Figure4_RF_TTFS_probe_audit.md)
- [Current single-cell builder](../scripts/build_figure4_cell_location_controls.py)
- [Central builder](../scripts/build_figure4_central_v1_candidate.py)
- [RF-adjusted builder](../scripts/build_figure4_rf_adjusted_candidate.py)
- [Accepted F1/F0 checkpoint](../artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0/FIGURE3_HARMONIZED_F1_F0_CHECKPOINT.md)
- [Trial-matched timescale bridge](../data/imports/mousev2_timescale_trial_bridge_v1/README.md)
- [Timescale sensitivity checkpoint](../artifacts/figure3/07_big_picture_concrete_first/timescale_population_sensitivity/TIMESCALE_INTERIM_CONCLUSION.md)
- [Earlier RF-model validation](../artifacts/figure3/07_big_picture_concrete_first/retinotopy_adjusted_hva_models/VALIDATION_REPORT.md)
