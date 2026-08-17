---
name: concrete-first-analysis
description: Use an iterative, human-guided, concrete-cases-before-summaries workflow for new or poorly understood analyses, investigations, diagnostics, hypotheses, metrics, model behaviors, experiments, or system failures. Use when aggregate results could hide mechanism, heterogeneity, failure modes, selection bias, or metric artifacts, and the work should progress from a minimal contrast through inspectable examples and auditable case selection to detailed follow-up and only then aggregate conclusions. Do not use for routine reproduction of a validated workflow or a narrowly specified calculation unless the user asks to re-examine its assumptions or examples.
---

# Concrete-first analysis

Establish what a result means in inspectable cases before building aggregate evidence. Adapt the evidence form to the domain: examples, records, inputs and outputs, logs, traces, diffs, timelines, tables, images, or other primary artifacts.

## Preserve the human-AI loop

- Default to checkpoints after the initial evidence view, the multi-case comparison, and the selected-case drill-down.
- At each checkpoint, present the artifacts, visible observations, surprises, current uncertainty, and smallest useful next step.
- Pause for human interpretation unless the user explicitly requests autonomous execution. Do not treat permission for one stage as permission to silently complete later stages.
- Keep expected outcomes labeled as hypotheses. Let concrete evidence revise or reject them.

## Follow the evidence ladder

### 1. Make the question smaller

- Identify the smallest contrast that could change the interpretation.
- Define the unit of analysis and hold nuisance dimensions fixed where practical.
- State the proposed mechanism, its concrete prediction, and what observation would count against it.
- Defer broad grids, large model suites, and population claims.

### 2. Inspect the primary evidence

- Show the inputs, intervention, process, and immediate outputs needed to understand one real case.
- Separate observed quantities from derived metrics, predictions, and annotations.
- Preserve identifiers and source context so the case can be reproduced.
- Stop at the initial-evidence checkpoint.

### 3. Compare several cases directly

- Show matched cases side by side across the smallest useful condition or time sweep.
- Include enough variation to expose heterogeneity without turning the first pass into a population study.
- Use direct differences, before/after views, paired records, or aligned traces when they clarify the change.
- Keep scales, definitions, and context comparable; disclose any case-specific normalization.
- Describe what differs without generalizing beyond the displayed cases.
- Stop at the multi-case checkpoint.

### 4. Select follow-up cases transparently

- Define selection roles before choosing cases when practical.
- Include expected successes, informative failures or dissociations, and controls when available.
- Useful roles include largest observed change, strongest predicted change, prediction without response, response without prediction, typical case, boundary case, negative control, and suspected measurement artifact.
- Save a table with case identifier, selection role, criterion, criterion value, reference contrast, and provenance.
- Distinguish algorithmic selection, user-requested examples, and post hoc judgment. Never present hand-picking as automatic selection.

### 5. Trace selected cases deeply

- Follow each selected case through the relevant intermediate states, transformations, decisions, or timepoints.
- Keep derived metrics adjacent to the primary evidence from which they arise.
- Test whether averaging, normalization, filtering, missingness, clipping, denominator instability, or data leakage changes the interpretation.
- Refine, qualify, or reject the mechanism based on these cases.
- Stop at the drill-down checkpoint.

### 6. Summarize last

- Compute cohort, group, benchmark, or population summaries only after the case-level behavior is understood.
- Test whether the proposed mechanism explains both the immediate evidence and the aggregate outcome.
- Report dissociations and failures alongside positive results.
- Prefer paired differences and absolute quantities over unstable ratios.
- Make aggregate claims traceable to the inspected cases and the saved selection table.

## Guard against narrative overfitting

- Do not let a compelling example stand in for prevalence.
- Do not let an aggregate association stand in for mechanism.
- Use selection roles to seek disconfirming cases deliberately.
- Separate exploratory observations from confirmatory tests.
- When the exploration changes the hypothesis, define a fresh evaluation set or clearly label the result as exploratory.

## Preserve an audit trail

- Save commands, configuration, source identifiers, intermediate artifacts, selection criteria, and output paths.
- Label quick probes, targeted diagnostics, and production analyses distinctly.
- Preserve raw and normalized quantities needed to diagnose apparent effects.
- Keep earlier interpretable checkpoints instead of overwriting them with a polished final artifact.

## Report each checkpoint

Report:

1. artifacts produced;
2. what the primary evidence shows;
3. surprising, contradictory, or ambiguous cases;
4. what remains unsupported;
5. the smallest useful next step;
6. the judgment requested from the human, if any.

## Adapt the workflow to the domain

Read [references/adaptation-patterns.md](references/adaptation-patterns.md) when choosing concrete artifacts, comparison structures, or selection roles for a particular kind of investigation. Use only the relevant pattern; do not force every domain into a visual or statistical workflow.
