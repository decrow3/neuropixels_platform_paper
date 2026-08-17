# Adaptation patterns

Use these patterns to translate the evidence ladder into the domain at hand. They are examples, not mandatory templates.

## Data and metrics

Start with individual records or matched entities and the exact transformations that produce the metric. Compare a few representative, extreme, contradictory, and missing-data cases. Drill into denominators, filters, joins, cohort assignment, and temporal alignment before producing segment or population summaries.

Useful artifacts:

- source rows beside transformed rows;
- numerator and denominator decompositions;
- matched before/after entities;
- cohort membership audits;
- missingness and exclusion tables;
- contribution waterfalls or record-level traces.

## Software and system debugging

Begin with the smallest reproducible behavior. Compare successful and failing executions under one controlled difference. Select cases such as deterministic failure, intermittent failure, false alarm, slow success, and boundary input. Trace logs, state transitions, requests, data transformations, and resource use before summarizing failure rates or performance distributions.

Useful artifacts:

- minimal reproduction and expected/observed output;
- side-by-side logs or traces;
- input and configuration diffs;
- state-transition timelines;
- request or dependency graphs;
- benchmark details for selected cases.

## Model and algorithm behavior

Start with concrete inputs, outputs, targets, and intermediate evidence when available. Compare cases across one controlled perturbation. Select confident success, confident failure, prediction-response dissociation, ambiguous case, distribution edge, and control. Trace preprocessing, representations, scores, decision thresholds, and postprocessing before reporting aggregate metrics.

Useful artifacts:

- input-output-target panels or tables;
- paired perturbation results;
- score and calibration traces;
- feature or intermediate-state diagnostics;
- error taxonomies with auditable case IDs;
- per-case contributions to aggregate metrics.

## Scientific or experimental investigation

Begin with the smallest interpretable treatment-control contrast and raw observations. Compare a small number of subjects, trials, samples, or time courses. Select expected responders, nonresponders, paradoxical responders, typical cases, and controls. Trace preprocessing and measurement construction before estimating group effects.

Useful artifacts:

- raw measurements beside processed values;
- paired treatment-control observations;
- subject or sample timelines;
- instrument and quality-control traces;
- exclusion and selection tables;
- within-case changes before between-group summaries.

## Qualitative or document-based inquiry

Start with complete, source-linked excerpts or records rather than themes alone. Compare multiple cases with the same coding questions. Select typical examples, counterexamples, boundary interpretations, and conflicting sources. Trace how evidence supports each code or claim before reporting theme prevalence or a synthesized narrative.

Useful artifacts:

- source-linked excerpts with concise annotations;
- side-by-side case matrices;
- coding decisions and disagreements;
- claim-to-evidence tables;
- counterexample and uncertainty logs;
- provenance for every summarized theme.

## Operational or process analysis

Start with one work item moving through the process. Compare normal, delayed, failed, reworked, and exceptional cases. Trace handoffs, queue time, decisions, and state changes before reporting throughput or service-level summaries.

Useful artifacts:

- case timelines;
- event histories;
- handoff and ownership traces;
- normal-versus-exception comparisons;
- reason-code audits;
- case-level decomposition of cycle time.

## Choosing a concrete representation

Prefer the representation closest to the phenomenon:

- use examples when semantics matter;
- use rows when transformations matter;
- use traces or timelines when sequence matters;
- use diffs when a controlled change matters;
- use images or spatial views when geometry matters;
- use tables when exact cross-case comparison matters;
- use charts only when relationships across several observations become clearer than in the underlying cases.

The goal is inspectability, not visualization for its own sake.
