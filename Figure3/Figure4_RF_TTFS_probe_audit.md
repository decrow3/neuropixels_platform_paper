# Lateral and medial TTFS RF attrition audit

All current lateral/medial TTFS cells match raw per-neuron RF-peak records. No session/probe-ID mismatch was found. The parametric RF fit table covers only the older Pilot-QC-selected population; fit availability matches that legacy eligibility flag exactly. Thus absence from the fit table is not a failed RF fit and not missing RF stimulation.

The current plot applied supported-parametric-RF eligibility, followed by at least five supported TTFS cells per session/probe. The reconstructed retained neuron IDs match the plotted population exactly.

| Session | Animal | Location | Original TTFS | Raw RF records | Parametric fits | Supported fits | Final cells |
|---|---|---|---:|---:|---:|---:|---:|
| site2 | 816305 | Lateral | 19 | 19 | 6 | 5 | 5 |
| site2 | 816305 | Medial | 21 | 21 | 9 | 3 | 0 |
| site3 | 810531 | Lateral | 34 | 34 | 10 | 0 | 0 |
| site3 | 810531 | Medial | 48 | 48 | 19 | 11 | 11 |
| site4 | 810532 | Lateral | 21 | 21 | 5 | 2 | 0 |
| site4 | 810532 | Medial | 25 | 25 | 12 | 2 | 0 |
| site5 | 813810 | Lateral | 32 | 32 | 21 | 15 | 15 |
| site5 | 813810 | Medial | 60 | 60 | 36 | 4 | 0 |
| site6 | 815152 | Lateral | 26 | 26 | 8 | 3 | 0 |
| site6 | 815152 | Medial | 43 | 43 | 22 | 6 | 6 |
| site7 | 816308 | Lateral | 22 | 22 | 8 | 0 | 0 |
| site7 | 816308 | Medial | 0 | 0 | 0 | 0 | 0 |
| site8 | 817334 | Lateral | 23 | 23 | 10 | 2 | 0 |
| site8 | 817334 | Medial | 39 | 39 | 19 | 2 | 0 |
| site9 | 817335 | Lateral | 48 | 48 | 21 | 4 | 0 |
| site9 | 817335 | Medial | 58 | 58 | 27 | 2 | 0 |

Site7 medial already had only two anatomically eligible TTFS neurons before RF selection and failed the pre-existing five-cell floor. The pre-anatomy session mean had 38 units. Its zero in this audit means absent from the incoming figure population, not absence of RF data at that probe.

## Where cells disappear

| Location | No parametric fit (legacy Pilot-QC) | Fitted but unsupported | Supported but below session/probe floor | Retained |
|---|---:|---:|---:|---:|
| Lateral | 136 | 58 | 11 | 20 |
| Medial | 150 | 114 | 13 | 17 |

## RF support criteria

The saved criteria require fit success, an on-screen center, model and reliability BH-FDR q≤.05, Spearman–Brown split-half reliability≥.3, pseudo-R²≥.1, and both width parameters in [3.05,79.5] degrees. The support-failure CSV counts each failed gate separately; failures overlap and must not be added as exclusive causes. The cell CSV retains every gate and original fit values.

## Interpretation and next step

The two-session result is a property of the figure's inherited fit coverage and selection pipeline, not a statement that only two sessions recorded RFs. In particular, more than half of the lateral/medial TTFS neurons were never included in this parametric-fit table. Their RF validity has not been assessed by these models. The current figure cannot establish that those cells lack reliable RFs.

Before interpreting or relaxing the RF-adjusted TTFS analysis, the targeted next step is to fit/validate parametric RFs for the current TTFS population missing from the legacy fit table, retaining explicit quality criteria. Existing raw argmax RF centers are not automatically interchangeable with supported Gaussian centers. This audit does not refit RFs, relax thresholds, or change the figures.

Provenance: `scripts/extract_mousev2_parametric_rf.py` selects `pilot_qc` before fitting; `data/imports/mousev2_parametric_rf_v1/run_manifest.json` records the fit population and thresholds; `scripts/build_figure4_rf_adjusted_candidate.py:attach_rf` applies supported-fit selection and the five-cell V1 floor.
