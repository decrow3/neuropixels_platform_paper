#!/usr/bin/env python3
"""Assemble the auditable Figure 4 rerun report, legend, ledger, and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import source_manifest

BASE = ROOT / "artifacts/figure4_rerun"
OUT = BASE / "v8_final"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def interval(row: pd.Series, name: str = "estimate") -> str:
    return f"{row[name]:.5g} [{row['interval_low']:.5g}, {row['interval_high']:.5g}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(BASE / "v4_primary_dispersion/primary_effects.csv")
    normalized = pd.read_csv(BASE / "v4_primary_dispersion/normalized_effects.csv")
    control = pd.read_csv(BASE / "v6_matched_control/matched_control.csv")
    hierarchy = pd.read_csv(BASE / "v5_hva_hierarchy/hierarchy_slopes.csv")
    central = pd.read_csv(BASE / "v7_central_companion/central_inclusive_effects.csv")
    ladder = pd.read_csv(BASE / "v5_hva_hierarchy/rf_effect_ladder.csv")
    historical = pd.DataFrame([
        {
            "checkpoint": "Earlier strict common-RF predictive analysis",
            "population_or_estimand": "153 HVA session-area timescale cells; within-session centered held-out prediction",
            "result": "Hierarchy incremental R2 over RF-only = 0.11223; permutation p=0.000999",
            "interpretation": "Restricted-support ordered timescale signal; categorical model did not outperform hierarchy",
            "relationship_to_rerun": "Different prediction estimand and support rule; retained as corroborating context, not substituted for the new slope",
        },
        {
            "checkpoint": "Earlier timescale population sensitivity",
            "population_or_estimand": "Full-cell structured variance HVA minus V1 across fit-validity populations",
            "result": "Primary difference positive; complete-ten and error<10 ms intervals included zero",
            "interpretation": "Magnitude was fit-population sensitive; no blanket robust-hierarchy claim",
            "relationship_to_rerun": "Different HVA-vs-V1 dispersion estimand; motivates complete-ten/error-gate sensitivities retained here",
        },
        {
            "checkpoint": "Current response-eligible HVA slope",
            "population_or_estimand": "Harmonized response-eligible cells; weighted signed hierarchy slope",
            "result": "; ".join(f"{r.metric}: {r.hierarchy_slope:.4g}" for r in ladder.loc[ladder.stage.eq("1_response_eligible")].itertuples()),
            "interpretation": "Shows the ordered point trend before RF selection",
            "relationship_to_rerun": "Directly separates population selection from adjustment",
        },
        {
            "checkpoint": "Current shared-support paired RF analysis",
            "population_or_estimand": "HVA-only local shared support; same cells before/after quadratic RF adjustment",
            "result": "; ".join(f"{r.metric}: {r.unadjusted_slope:.4g} to {r.adjusted_slope:.4g} (change {r.adjusted_minus_unadjusted:+.4g})" for r in hierarchy.itertuples()),
            "interpretation": "RF adjustment is evaluated by the paired slope-change interval, not significance transitions",
            "relationship_to_rerun": "Current inferential result under the frozen support/model contract",
        },
    ])
    historical.to_csv(OUT / "historical_reconciliation.csv", index=False)

    lines = [
        "# Figure 4 rerun — final analysis report", "",
        "## Primary category-dispersion result", "",
        "The primary estimand is the finite-category dispersion of model-standardized means, V, with equal category weight. The contrast is V(HVA) − V(V1). These are squared response units; intervals are 95% pointwise whole-animal bootstrap intervals.", "",
        "| Metric | V(V1) | V(HVA) | HVA − V1 | Normalized V1 / HVA; difference (pp) |", "|---|---:|---:|---:|---:|",
    ]
    for metric in primary.metric.unique():
        part = primary.loc[primary.metric.eq(metric)].set_index("effect")
        norm = normalized.loc[normalized.metric.eq(metric)].set_index("effect")
        nd = norm.loc["hva_minus_v1"]
        lines.append(f"| {metric} | {interval(part.loc['v1'])} | {interval(part.loc['hva'])} | {interval(part.loc['hva_minus_v1'])} | {norm.loc['v1', 'estimate']:.3g}% / {norm.loc['hva', 'estimate']:.3g}%; {nd.estimate:+.3g} [{nd.interval_low:+.3g}, {nd.interval_high:+.3g}] |")
    lines += [
        "", "All three primary contrast intervals include zero. This leaves both positive and negative differences compatible with the data; it is not an equivalence result. Null simulations showed upward plug-in dispersion bias at the boundary, so nonnegative dispersion intervals are not treated as tests against zero.", "",
        "## Matched Allen control", "",
        "| Metric | HVA − V1 response mean | Two-group dispersion | Sessions |", "|---|---:|---:|---:|",
    ]
    for row in control.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.hva_minus_v1:.5g} [{row.hva_minus_v1_low:.5g}, {row.hva_minus_v1_high:.5g}] | {row.two_group_dispersion:.5g} [{row.two_group_dispersion_low:.5g}, {row.two_group_dispersion_high:.5g}] | {row.n_sessions} |")
    lines += [
        "", "The control is a coarse matched-session comparison. It does not establish an ordered hierarchy and is not used to validate the sign of the primary dispersion contrast.", "",
        "## HVA-only RF-conditional hierarchy", "",
        "| Metric | Unadjusted slope | RF-adjusted slope | Paired adjusted − unadjusted |", "|---|---:|---:|---:|",
    ]
    for row in hierarchy.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.unadjusted_slope:.5g} [{row.unadjusted_slope_low:.5g}, {row.unadjusted_slope_high:.5g}] | {row.adjusted_slope:.5g} [{row.adjusted_slope_low:.5g}, {row.adjusted_slope_high:.5g}] | {row.adjusted_minus_unadjusted:.5g} [{row.adjusted_minus_unadjusted_low:.5g}, {row.adjusted_minus_unadjusted_high:.5g}] |")
    lines += [
        "", "Cross-population RF adjustment is unsupported for every metric under the frozen local-support rule, so no adjusted V1-versus-HVA null estimate is reported. The hierarchy fit is HVA-only on adequate local shared support. Adjustment is conditional statistical sharing, not causal mediation. Three of 15,000 hierarchy bootstrap attempts lost an HVA category (all TTFS); they remain in the failure ledger and were not silently replaced.", "",
        "## Central-inclusive companion", "",
        "Central (Allen V1) is placed between E and C in the display. Its location is completely nested in cohort/session, making the category-plus-session design rank deficient by one parameter for every metric. The companion is therefore descriptive and cannot separate Central location from cohort effects.", "",
        "| Metric | Five-location descriptive dispersion | Session-model rank |", "|---|---:|---:|",
    ]
    for row in central.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.dispersion:.5g} [{row.dispersion_low:.5g}, {row.dispersion_high:.5g}] | {int(row.fixed_session_rank)}/{int(row.fixed_session_parameters)} |")
    lines += [
        "", "## Robustness and limitations", "",
        "Equal-cell weights, a ten-cell floor, leave-one-animal-out influence, complete-ten timescale cells, and the existing tighter timescale error gate are retained as sensitivities. Point conclusions are directionally similar, but V1 TTFS and especially V1 timescale are sensitive to individual animals; the tighter timescale error gate materially reduces the HVA-minus-V1 point difference and is phenotype-selective rather than a neutral quality improvement.", "",
        "The targeted simulation recovered a known dispersion with small average bias but only 83.3% coverage for the sparse V1 TTFS design. Its interval should therefore be read cautiously. In the hierarchy diagnostic, some five-area RF feature means correlate strongly with hierarchy (absolute correlation up to 0.934), although each feature also has substantial within-area spread and the adjusted designs remain full rank (condition numbers 56–90). The conditional slope therefore depends on the stated within-area RF-surface assumption. No formal equivalence margin or multiplicity-adjusted significance family was frozen, and the final figures contain no significance stars.", "",
        "## Historical reconciliation", "",
        "The prior strict-support predictive result and later fit-population sensitivity concern different populations and estimands. The current ladder makes that explicit: selection into RF-qualified/shared support can change the slope (especially TTFS), while stage 3 versus stage 4 isolates RF adjustment on identical cells. The current result neither erases the restricted-support signal nor licenses a population-invariant hierarchy claim.", "",
        "Detailed reconciliation is saved in `historical_reconciliation.csv`; source/design checkpoints, RF cases, simulations, failure ledgers, and all bootstrap draws remain in the versioned artifact directories.",
    ]
    (OUT / "Figure4_rerun_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    legend = [
        "# Figure 4 rerun — methods and legend", "",
        "**C–D, cell distributions and category effects.** Half-violins show eligible cell distributions. MouseV2 response-timescale violins, counts, and diamonds use fixed display draw 1 of 10, chosen by index rather than response result, after applying its draw-specific five-cell floor. Points are a deterministic display subsample. Response axes show the central 99.6% for legibility; all eligible values enter the estimates. The adjacent timescale effect estimates retain the primary estimator, which averages ten complete draw-specific effects. Other metrics have one accepted value per neuron. Category dispersion is V = mean_g[(m_g − mean(m))²], using K rather than K−1. Categories, animals within categories, physical recordings within animal/category, and cells within recordings receive equal nested weight. Session fixed effects are nuisance terms and are set to their population reference for standardized means. Exact displayed counts are saved in `figure4_display_counts.csv`.", "",
        "**E, control and primary contrast.** The control is matched Allen V1 versus equal-area pooled HVA, summarized with the analogous two-group dispersion. The main contrast is V(HVA) − V(V1). Whiskers are 95% percentile intervals from 5,000 whole-animal bootstrap replicates. MouseV2 timescale resampling selects one coherent matched-trial draw per sampled session occurrence; its point estimate averages ten complete draw-specific effects. Allen has one accepted estimate per neuron, so trial-estimation uncertainty remains asymmetric across cohorts. Intervals are pointwise compatibility intervals, not equivalence tests. No stars or shuffle reference are shown.", "",
        "All metrics use harmonized accepted sources, common waveform QC, metric-specific validity, anatomical VISp restriction, and a common floor of five eligible cells per recording/category. TTFS is response-selected preferred-polarity latency below 100 ms; phase modulation is log10 harmonized F1/F0; timescale is restricted to valid 1–300 ms fits with >50 spikes and fit error <20 ms. Raw total variance and normalized dispersion fractions are supplied as companions, not labeled as predictive variance explained or classical omega-squared.", "",
        "**RF/hierarchy companion.** On HVA-only local shared RF support, the signed hierarchy slope is fit without and with a frozen quadratic surface in RF azimuth, elevation, their squares, and interaction. Both fits use identical cells and nuisance structure. The paired adjusted-minus-unadjusted interval evaluates attenuation; a significance transition would not. Cross-population RF adjustment was not supported, so no adjusted V1-versus-HVA contrast is displayed. Coordinate conversion and estimator-specific quality rules are retained; a display-coordinate translation is not described as gaze correction.", "",
        "**Central companion.** Central is harmonized Allen V1 and appears between lateral and posterior MouseV2 locations. Because Central is nested in the Allen cohort/sessions, the five-location session-adjusted design is rank deficient. Its equal-category dispersion and bootstrap interval are descriptive mixed-cohort summaries, not an identified anatomical-location effect.",
    ]
    (OUT / "Figure4_rerun_methods_legend.md").write_text("\n".join(legend) + "\n", encoding="utf-8")

    code_files = [
        ROOT / "scripts/figure4_rerun_core.py", ROOT / "scripts/run_figure4_rerun.py",
        ROOT / "scripts/extract_mousev2_parametric_rf.py", ROOT / "scripts/audit_figure4_rf_completion.py",
        ROOT / "scripts/figure4_rf_support.py", ROOT / "scripts/validate_figure4_rerun_models.py",
        ROOT / "scripts/run_figure4_primary_dispersion.py", ROOT / "scripts/audit_figure4_primary_sensitivities.py",
        ROOT / "scripts/run_figure4_hva_hierarchy.py", ROOT / "scripts/run_figure4_matched_control.py",
        ROOT / "scripts/run_figure4_central_companion.py", ROOT / "scripts/build_figure4_rerun_outputs.py",
        ROOT / "scripts/audit_figure4_rf_effect_ladder.py",
        ROOT / "scripts/summarize_figure4_normalized_dispersion.py",
        ROOT / "scripts/validate_figure4_final_package.py",
    ]
    configs = sorted((ROOT / "config").glob("figure4_rerun_v*.json"))
    outputs = sorted(path for path in OUT.iterdir() if path.is_file() and path.name not in {"final_analysis_manifest.json", "FINAL_MANIFEST_SHA256.txt"})
    analysis_artifacts = sorted(
        path for version in range(1, 8)
        for path in (BASE / f"v{version}" if version <= 1 else BASE / {
            2: "v2_support", 3: "v3_model_validation", 4: "v4_primary_dispersion",
            5: "v5_hva_hierarchy", 6: "v6_matched_control", 7: "v7_central_companion",
        }[version]).rglob("*") if path.is_file()
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout)
    manifest = {
        "schema_version": 5, "created_for": "Figure4 rerun final package",
        "git_revision": revision, "worktree_dirty": dirty,
        "prespecification_note": "Component v1-v4 hashes were frozen before their corresponding inferential stages. This v5 manifest is a post-run reporting manifest and is not retrospectively labeled prespecified.",
        "random_seed": 20260918, "bootstrap_replicates": 5000,
        "interval_method": "pointwise 2.5th and 97.5th percentiles of whole-animal bootstrap draws",
        "metric_transformations": {"TTFS (ms)": "preferred-polarity response-selected milliseconds <100", "log10 F1/F0": "log10 harmonized positive F1/F0", "Response timescale (ms)": "valid 1-300 ms fit, >50 spikes, error <20 ms"},
        "weights": "equal categories; equal animals/category; equal physical recordings/category/animal; equal cells/recording",
        "populations": {"primary_v1": ["A", "E", "C", "B"], "primary_hva": ["LM", "RL", "AL", "PM", "AM"], "central_companion": ["A", "E", "Central (Allen V1)", "C", "B"]},
        "models": {"category_dispersion": "weighted category one-hot coefficients plus session fixed nuisance effects", "hierarchy_unadjusted": "weighted intercept plus signed hierarchy score plus session fixed nuisance effects", "hierarchy_adjusted": "hierarchy_unadjusted plus azimuth, elevation, squared terms, and interaction"},
        "cell_floor": 5,
        "effect": "K-denominator dispersion of standardized named-category means; HVA minus V1",
        "rf_support": "response-blind category convex hull plus 15-degree local radius with >=5 cells from >=2 recordings in every category",
        "rf_reference": "HVA-only supported cells for hierarchy; cross-population adjustment unsupported",
        "timescale_draws": "point averages ten complete effects; bootstrap chooses one coherent draw per sampled MouseV2 session occurrence",
        "sources": source_manifest(),
        "component_configs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in configs],
        "code": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in code_files],
        "analysis_artifacts": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in analysis_artifacts],
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in outputs],
    }
    manifest_path = OUT / "final_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "FINAL_MANIFEST_SHA256.txt").write_text(digest(manifest_path) + "  final_analysis_manifest.json\n", encoding="utf-8")
    print(OUT / "Figure4_rerun_final_report.md")


if __name__ == "__main__":
    main()
