# Figure 4 rerun: execution plan

Updated 17 September 2026 after review. This is the operative plan; the [methods appendix](Figure4_rerun_methods_appendix.md) retains detailed source contracts and implementation explanations. No rerun is reported here. Existing candidates remain exploratory checkpoints.

## 1. Scope and questions

The new primary question is: **how large is differentiation among LM, RL, AL, PM, and AM compared with differentiation among the four MouseV2 V1 locations, for each response metric?** The new RF question is: **how much of the previously reported hierarchy relationship remains after conditioning on measured RF position?**

The original paper already tested unadjusted hierarchy. Reuse validated reproductions where inputs are unchanged; check that result with our own metrics where processing or population changed. This is a baseline verification, not another headline hypothesis. Do not require the original significance label to recur.

Keep Central in a complementary five-location comparison, with its Allen origin explicit. The primary V1 comparison uses the four MouseV2 locations. Central versus periphery is also Allen versus MouseV2 and cannot separately identify location and cohort effects. Harmonizing methods does not remove this design limitation.

Keep the matched Allen V1–pooled-HVA comparison as the coarse control. It need not be significant for every metric. Neither that control nor a categorical area effect establishes an ordered hierarchy.

Continuous anatomical/retinotopic placement and categorical differentiation answer different questions. Retain the categorical comparison requested here and continuous RF terms for adjustment. The external review cites a manuscript todo favoring continuous placement and an unresolved OpenScope scope decision; that file was not available in the searched MouseV2 tree. These are unverified external planning context, not reasons to cancel this authorized work. Reconcile them if the source document becomes available; do not silently replace the estimand.

## 2. Fixed primary effect and weighting

**Figure E will show dispersion of standardized category means in squared response units and the HVA-minus-V1 difference.** This choice is made before fitting the revised effects. Held-out predictive increments are optional supplementary diagnostics, not competing headline metrics.

For population P with K categories, let m_g be its model-standardized mean for category g. Give each category weight 1/K:

- grand mean: `m_bar = sum(m_g) / K`;
- category dispersion: `V_P = sum((m_g - m_bar)^2) / K`;
- main contrast: `Delta_V = V_HVA - V_V1`.

This is dispersion across the named finite set of categories, so use K rather than K−1. Units are ms² for TTFS/timescale and (log10 F1/F0)² for phase modulation. Also supply `sqrt(V_P)` in response units for interpretation, without replacing the primary contrast. Use the identical definition in the Central-inclusive companion, with its five V1 categories.

The sampling target gives categories equal weight, independent recording units equal weight within each category, and eligible cells equal weight within each category/recording unit. If an animal has repeated sessions, split its within-category weight across its eligible sessions before splitting across cells. Implement this explicitly in fitting/standardization; a random intercept alone does not enforce the target. Pooled equal-cell weighting is a sensitivity analysis.

For adjusted means, average predictions over one shared RF reference distribution on supported overlap, with recording deviations set to their population reference (zero for mean-zero random effects). Use the same RF reference for V1 and HVA comparisons. Construct that reference from eligible RF coordinates with the fixed category/recording weights and equal V1/HVA population contribution. Freeze it for the conditional analysis and bootstrap; intervals are then conditional on that reference/support definition. The primary four-location and Central-inclusive analyses use the same reference when supported. If adding Central requires a narrower reference, show a four-location result on that narrower reference too, rather than attributing reference changes to Central.

Provide a companion normalization `F_P = 100 * V_P / T_P`, where T_P is the weighted variance of raw cell outcomes in the same population. Its contrast is in percentage points. Hold cells, weights, and T_P fixed between adjusted and unadjusted fits within each draw/resample; recompute them consistently across resamples. V1 and HVA have distinct T_P values, so show those denominators alongside the raw-unit contrast. This is a standardized category-dispersion fraction, not automatically predictive variance explained or classical omega-squared; do not clip values or mislabel the axis.

These plug-in dispersions can be inflated by uncertainty in estimated category means. The small validation exercise below must quantify that issue. No claim of unbiased variance components is made. If instability makes the contrast uninterpretable, report the category contrasts and uncertainty without silently choosing a more favorable effect measure.

## 3. Repair the inputs and verify the recording design

**What to do.** Reuse the existing harmonization and RF-attrition audits. Build one shared metric loader and a registry of source, animal, session, physical probe, anatomical category, unit ID, metric availability, and RF status. Reconstruct eligibility from current data, not legacy figure session lists.

Preserve accepted common waveform QC; harmonized conventional F1/F0; response-selected preferred-polarity TTFS; and trial-matched timescale fits. Replace Central's native Allen metrics/broad QC with harmonized Allen V1 sources. Keep metric-specific eligibility. Retain the documented anatomical VISp restriction and expose all validity filters. The appendix lists exact source contracts.

**Why.** The current Central processing difference is demonstrable on identical cells. Updated measurements with inherited eligibility are also not a fully updated population. Shared loaders prevent further divergence among figure versions.

**Known design fact.** MouseV2 is not one location per session: the audit includes A/E/C/B in session 2 and lateral/medial populations across the same sessions. Use the actual within-session contrasts. Audit which remain after each filter; do not use the review's hypothetical one-location-per-session degrees of freedom.

**Feasibility gate.** For every metric/population, report category co-occurrence, independent recordings, model rank, and the influence of removing each recording. If the specified contrasts are rank-deficient, a category is represented in only one independent unit, or resampling routinely loses required categories, do not give a population-level inferential contrast for that population. Report descriptive results and the support failure. Two or more units is not a guarantee of adequacy: report sparse replication and interval instability explicitly. Do not reject an estimable contrast merely because its interval is wide, and do not invent a universal N cutoff to guarantee credibility.

## 4. Restore timescale uncertainty with a specified algorithm

Use the existing ten 75-bright/75-dark MouseV2 fits as matched trial-subsampling versions, never as independent neurons or multiple imputations.

**Point estimate:** for each draw index 1–10, construct its valid cell population, apply the fixed eligibility rules, fit the model, and calculate the complete effect (including dispersion, normalization, and HVA-minus-V1 contrast). Average these ten effect estimates. Do not first average individual neurons' fits and then estimate the effect. Document any draw with an unestimable contrast; do not silently drop it from the mean.

**Uncertainty:** use 5,000 outer bootstrap replicates with a fixed seed. Resample the highest independent units within source, carrying all nested recordings. For each sampled MouseV2 session occurrence, select one of its ten draws uniformly and carry the whole session's validity mask and all probe populations together. Different copies of a resampled session may receive different draw indices. Reuse each occurrence and draw across every comparison and adjusted/unadjusted fit that shares those data. Reuse Allen cluster multiplicities across Central, HVA, and control fits. Calculate contrasts inside each replicate; report percentile intervals, subject to the finite-sample/boundary checks below. Use analogous cluster resampling without trial-draw selection for the other metrics.

This distribution represents recording resampling plus the available matched-trial draw variability; do not shrink its trial component by treating ten draws as biological replication. Do not add Rubin-rule pooling on top: these are not proper missing-data imputations. Allen currently has one estimate per neuron, so state that the trial-uncertainty treatment remains asymmetric. Do not expand this revision into a new Allen estimator unless a concrete source problem requires it.

The estimator targets the accepted draw-dependent fit population. Retain complete-ten-draw and tighter-fit-error sensitivities because validity selection changes the population. The tighter error gate preferentially changes timescale composition and is not automatically a better primary filter.

## 5. Complete RF coverage, then separate selection from adjustment

Fit the current MouseV2 response population missing from the legacy RF-fit table. Inspect concrete accepted/rejected examples, especially lateral and medial probes. Distinguish unavailable raw data, unattempted fits, failed fits, unsupported fits, and later population exclusions. Preserve cell-specific RFs.

Verify the MouseV2/Allen coordinate conversion, estimator definitions, and quality selection. Different quality statistics cannot be harmonized by assigning them the same numerical cutoff. Record any remaining pipeline asymmetry; do not interpret a display-coordinate translation as gaze correction.

For each metric retain this ladder:

1. Harmonized response-eligible population.
2. RF-qualified population, unadjusted responses.
3. Shared-RF-support population, unadjusted responses.
4. Exactly population 3 with RF adjustment.

Only 3 versus 4 isolates adjustment. Compare changes at the earlier steps as selection effects. Inspect two-dimensional RF occupancy, not only overlapping bounding boxes. Determine support using RF/design information before reviewing response-effect results, and freeze its exact algorithm and reference distribution. Use the existing percentile-box approach only if the occupancy check supports it.

Use a common starting floor of five valid cells per recording/category for all three metrics and both populations; use ten as the single predefined floor sensitivity. Apply it consistently after anatomical/RF selection and per timescale draw. This resolves the existing five-versus-ten TTFS asymmetry without lowering RF-quality standards. If the validation/design checks show the five-cell rule cannot support estimation, record the problem before the primary fit rather than selecting the floor by effect size.

**RF feasibility gate:** if a shared reference would require unsupported extrapolation, report the cross-population RF-adjusted contrast as unsupported. Preserve the unadjusted comparison and attrition results. HVA-only RF analysis may still proceed on its own adequate common support, with its different population explicit.

## 6. Fit the minimum adequate recording-aware model

Use single-cell outcomes, fixed named categories, recording/animal effects supported by the registry, and a session-by-category or physical-probe deviation only where identifiable. Do not include duplicate nuisance components for the same recording population. Allow separate cohort residual structures rather than imposing identical variance. Fit unadjusted and RF-adjusted models with the same nuisance specification.

Resolve the exact identifiable random-effects formula from the registry, then freeze it before primary effect estimation. For RF adjustment start with the documented five-term quadratic surface (azimuth, elevation, their squares, and interaction), on supported coordinates; do not launch a broad search over smoothers.

Use one small simulation harness under the observed cluster sizes and missing-category pattern, with two conditions: no category effect and a known category effect, both including recording shifts. Check implementation, finite-sample dispersion inflation, and interval behavior. This is a targeted validation step, not a claim of universal calibration. Broaden only to diagnose an observed failure.

If variance-component fitting is singular or the requested category contrast is not identifiable, document the failure and simplify only redundant nuisance terms according to the recorded design. Do not switch estimands or discard recordings to achieve a favorable result. Near-zero dispersion is a boundary problem; a percentile interval on a nonnegative estimated dispersion is not itself a valid nonzero-effect test.

## 7. Verify the published baseline and estimate RF-conditional hierarchy

Verify the original hierarchy-score source/variant and reuse the fixed scores. Reproduce the published relationship only where our metrics or population changed. The new analysis remains HVA-only; do not add V1 to strengthen the relationship.

On exactly the same supported cells, estimate a signed hierarchy slope without RF and with RF jointly in the model. Report both slopes, the response change across the observed score range, and their paired attenuation contrast. A change from significant to nonsignificant is not evidence that slopes differ. Do not correlate RF-corrected responses with an unadjusted hierarchy score as a substitute for the joint model.

**Explicit assumption:** the shared RF surface uses within-area RF variation to help standardize between-area differences. Inspect within-area-centered RF features and area mean features, including the quadratic terms, and compare within/between patterns where estimable. With five areas, a rich between-area model is not supported; this diagnostic cannot manufacture identification. Do not include unrestricted area fixed effects and the area-level hierarchy score as separately identifiable terms.

**Hierarchy feasibility gate:** check RF overlap, design rank, collinearity, recording influence, and sensitivity to the within/between RF assumption. Five areas alone does not imply non-identifiability, but many cells alone does not resolve it. If the conditional slope is estimable but imprecise, report the interval. If the separation relies on unsupported extrapolation or is rank-deficient, report that RF and hierarchy cannot be separated under the available design; do not present an adjusted null result. This gate does not prevent reporting the categorical comparison if that remains supported.

Categorical-versus-hierarchy predictive comparisons, more flexible RF models, and additional layer/cell-class adjustment are optional targeted diagnostics, not mandatory new headline analyses.

## 8. Reporting, sensitivity checks, and configuration freeze

Primary outputs are effect estimates and 95% pointwise uncertainty intervals for all three metrics, not an equivalence verdict. Report the compatible effect range. Formal equivalence testing is out of scope because no justified margin has been agreed. Do not import the earlier exploratory R² margin.

Do not automatically generate stars from bootstrap sign fractions or dispersion intervals. If significance annotations are retained, validate and freeze their testing/multiplicity convention before running them; use interval estimates in the meantime. Remove the gray difference-of-shuffles reference. A within-session shuffle may remain a restricted diagnostic but is not an omnibus five-location Central test or a test that two nonzero effects are equal.

Core sensitivities are: equal-cell weighting, leave-one-independent-unit-out influence, the ten-cell floor, and the two existing timescale validity sensitivities. The population ladder is mandatory. Add other analyses only for a concrete diagnostic problem. Preserve shared cluster draws for all paired differences.

Serialize source paths and content checksums, code revision, populations, weights, model formulas, RF support/reference, metric transformations, floors, draw handling, effect definitions, interval method, and random seed in one YAML/JSON file. Record its SHA-256 before the primary run. Any later change gets a new version/hash and a stated reason; it is not retrospectively called prespecified.

Keep C/D as cell-distribution panels with category effects beside them, vertical axes, and the extra gap before E. Keep the control and main contrast together in E. Label the new dispersion measure correctly; retain normalized fractions in a companion. Central remains in the middle of its complementary display. Explain any surviving bias-corrected omega-squared only in the relevant legacy/companion legend, not as the definition of the new measure.

Reconcile the earlier restricted-support timescale hierarchy signal and later population-sensitive conclusion with a compact ledger. Reuse validated saved outputs; rerun historical pipelines only when needed to explain a specific discrepancy. Neither checkpoint predetermines the new outcome.

## 9. Deliverables and execution order

1. Current source audit and animal/session/probe registry, including post-filter co-occurrence.
2. Unified corrected cell table and explicit metric/eligibility contract.
3. Completed RF-fit coverage audit and population/support ladder.
4. Design feasibility decisions, minimal model validation, and frozen hashed configuration.
5. Main dispersion contrasts, complementary Central/control results, and RF-conditional hierarchy results where supported.
6. Core sensitivities, concise historical reconciliation, revised figures, and methods/legend text.

Progress through these stages without expanding the task into unrelated model development. Show the recording/support checkpoint before expensive final inference. Completion means traceable estimates and defensible uncertainty or an explicit support limitation, not a requested significance pattern. This plan authorizes no claim that “little hierarchy” follows solely from a nonsignificant category contrast.

## Evidence and portability

The detailed [methods appendix](Figure4_rerun_methods_appendix.md) contains nine verified repository-relative source links. They resolve in this analysis repository; a standalone copy in the manuscript repository must bundle those files or rewrite links with an explicit analysis-repository root. Do not treat a copied document's missing relative targets as missing analysis evidence.

Key local audits: [harmonization](Figure4_harmonization_audit.md) and [RF TTFS attrition](Figure4_RF_TTFS_probe_audit.md). No figure or analysis data were changed by this planning revision.
