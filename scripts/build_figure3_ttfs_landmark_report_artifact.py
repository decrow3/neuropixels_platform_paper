#!/usr/bin/env python3
"""Package the TTFS landmark-scale pilot as a Data Analytics report artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/figure3/07_big_picture_concrete_first/landmark_scale/ttfs_pilot"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def source(source_id: str, label: str, path: str, description: str, metrics: list[str]) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "description": description,
            "language": "sql",
            "engine": "DuckDB",
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "tables_used": [path],
            "filters": [
                "Preferred-polarity positive responders",
                "TTFS < 100 ms",
                "At least 10 neurons per population",
                "Allen sessions require VISp and at least two HVAs",
            ],
            "metric_definitions": metrics,
        },
    }


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    point = pd.read_csv(BUNDLE / "ttfs_landmark_point_estimates.csv")
    boot = pd.read_csv(BUNDLE / "ttfs_landmark_bootstrap_summary.csv")
    selected = pd.read_csv(BUNDLE / "ttfs_landmark_selected_sessions.csv")
    selected["session_label"] = selected["session_id"].astype(str)
    selected = selected.drop(columns="session_id")
    validation = pd.read_csv(BUNDLE / "ttfs_landmark_input_validation.csv")
    loo = pd.read_csv(BUNDLE / "ttfs_landmark_leave_one_session_out.csv")

    labels = {
        "within_v1": "V1 location ↔ V1 location",
        "hva_to_hva": "HVA ↔ HVA",
        "v1_to_hva": "V1 ↔ HVA",
    }
    point["landmark_label"] = point["landmark"].map(labels)
    interval = boot.set_index("estimand")
    point["ci_low"] = point["landmark"].map(interval["ci_low"])
    point["ci_high"] = point["landmark"].map(interval["ci_high"])
    point["distance_ms2"] = point["distance"]
    point["interval"] = point.apply(
        lambda row: f"[{row.ci_low:.2f}, {row.ci_high:.2f}]", axis=1
    )
    order = {"within_v1": 0, "hva_to_hva": 1, "v1_to_hva": 2}
    point["display_order"] = point["landmark"].map(order)
    point = point.sort_values("display_order")

    score = float(interval.loc["hva_landmark_position", "point_estimate"])
    naive_score = float(interval.loc["naive_hva_landmark_position", "point_estimate"])
    hva_difference = interval.loc["hva_minus_within_v1"]
    anchor_difference = interval.loc["hva_minus_v1_to_hva"]
    denominator_positive = 1 - float(interval.loc["between_minus_within_v1", "fraction_le_zero"])
    loo_low = float(loo["hva_landmark_position"].min())
    loo_high = float(loo["hva_landmark_position"].max())
    maximum_error = float(validation["absolute_mean_error"].max())

    result_path = "artifacts/figure3/07_big_picture_concrete_first/landmark_scale/ttfs_pilot/ttfs_landmark_point_estimates.csv"
    audit_path = "artifacts/figure3/07_big_picture_concrete_first/landmark_scale/ttfs_pilot/ttfs_landmark_selected_sessions.csv"
    validation_path = "artifacts/figure3/07_big_picture_concrete_first/landmark_scale/ttfs_pilot/ttfs_landmark_input_validation.csv"
    sources = [
        source(
            "landmarks",
            "TTFS landmark point estimates",
            result_path,
            "Full-cell U-statistic population distances with equal session weighting and whole-session bootstrap.",
            [
                "Corrected distance = squared difference of cell means - s1²/n1 - s2²/n2",
                "Landmark position = (HVA-HVA - within-V1) / (V1-HVA - within-V1)",
            ],
        ),
        source(
            "audit",
            "Algorithmically selected concrete sessions",
            audit_path,
            "Smallest, median-like, and largest session mean for each landmark.",
            ["Session distance = equal-weight mean of eligible population-pair distances"],
        ),
        source(
            "validation",
            "TTFS input reconstruction",
            validation_path,
            "Reconstruction of current Figure 3 session-by-population means and counts from retained neurons.",
            ["Mean error = absolute reconstructed minus released session-population mean"],
        ),
    ]

    charts = [
        {
            "id": "landmark_chart",
            "title": "TTFS population-mean distance landmarks",
            "subtitle": "Corrected squared distances; exact bootstrap intervals are in the adjacent table",
            "type": "bar",
            "dataset": "landmarks",
            "sourceId": "landmarks",
            "intent": "comparison",
            "encodings": {
                "x": {"field": "landmark_label", "type": "nominal", "label": "Comparison"},
                "y": {
                    "field": "distance_ms2",
                    "type": "quantitative",
                    "label": "Corrected squared distance",
                    "unit": "ms²",
                },
            },
            "referenceLines": [
                {"axis": "y", "value": 0, "label": "No latent separation", "color": "neutral", "lineStyle": "dashed"}
            ],
            "layout": "full",
        }
    ]
    tables = [
        {
            "id": "landmark_table",
            "title": "Landmark estimates and uncertainty",
            "subtitle": "Equal-weight session means; 10,000 whole-session bootstrap draws",
            "dataset": "landmarks",
            "sourceId": "landmarks",
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "distance_ms2", "direction": "asc"},
            "columns": [
                {"field": "landmark_label", "label": "Landmark", "type": "text"},
                {"field": "n_sessions", "label": "Sessions", "type": "number"},
                {"field": "distance_ms2", "label": "Distance", "type": "number", "unit": "ms²"},
                {"field": "interval", "label": "95% interval", "type": "text"},
                {"field": "naive_distance", "label": "Uncorrected", "type": "number", "unit": "ms²"},
            ],
        },
        {
            "id": "audit_table",
            "title": "Concrete-session sensitivity cases",
            "subtitle": "Algorithmic smallest, median-like, and largest sessions for each landmark",
            "dataset": "selected",
            "sourceId": "audit",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "landmark", "direction": "asc"},
            "columns": [
                {"field": "landmark", "label": "Landmark", "type": "text"},
                {"field": "selection_role", "label": "Role", "type": "text"},
                {"field": "session_label", "label": "Session", "type": "text"},
                {"field": "distance", "label": "Distance", "type": "number", "unit": "ms²"},
                {"field": "n_pairs", "label": "Pairs", "type": "number"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# TTFS Area-Separation Landmark Pilot", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                f"TTFS differences among HVAs are between-area-sized, not within-V1-sized. "
                f"The corrected HVA↔HVA distance is 50.67 ms², compared with 7.52 ms² across "
                f"V1 locations and 43.46 ms² for simultaneously recorded Allen V1↔HVA populations. "
                f"The affine landmark position is {score:.2f}, where 0 is within V1 and 1 is V1↔HVA. "
                "Its ratio confidence interval is denominator-unstable, so the raw distances and "
                "their paired differences are the inferential evidence."
            ),
            "sourceId": "landmarks",
            "layout": "full",
        },
        {
            "id": "finding",
            "type": "markdown",
            "body": (
                "## HVA separation is far above the within-V1 baseline\n\n"
                f"HVA↔HVA exceeds within-V1 by {hva_difference.point_estimate:.2f} ms² "
                f"(95% interval {hva_difference.ci_low:.2f} to {hva_difference.ci_high:.2f}). "
                "The V1↔HVA anchor and HVA↔HVA landmark use exactly the same 16 Allen sessions, "
                "each containing VISp and at least two retained HVAs. The within-V1 baseline uses "
                "all four locations in eight MouseV2 sessions."
            ),
            "sourceId": "landmarks",
            "layout": "full",
        },
        {"id": "landmark_chart_block", "type": "chart", "chartId": "landmark_chart", "layout": "full"},
        {
            "id": "placement",
            "type": "markdown",
            "body": (
                "## HVA↔HVA is statistically indistinguishable from the V1↔HVA anchor\n\n"
                f"The corrected HVA↔HVA minus V1↔HVA difference is {anchor_difference.point_estimate:.2f} ms² "
                f"(95% interval {anchor_difference.ci_low:.2f} to {anchor_difference.ci_high:.2f}). "
                f"The uncorrected landmark position is {naive_score:.2f}, while the corrected position "
                f"is {score:.2f}. Leave-one-session-out corrected positions range from {loo_low:.2f} to "
                f"{loo_high:.2f}. The result therefore supports 'approximately at the between-area anchor,' "
                "not a precise claim that HVA separation exceeds V1↔HVA separation."
            ),
            "sourceId": "landmarks",
            "layout": "full",
        },
        {"id": "landmark_table_block", "type": "table", "tableId": "landmark_table", "layout": "full"},
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                "**TTFS population.** Preferred-polarity positive responders with latency below 100 ms "
                "and at least 10 neurons per session-population. **Within-area landmark.** Every distinct "
                "pair among four MouseV2 V1 recording locations in one session. **Between-area anchor.** "
                "Allen VISp paired with every retained HVA in the same session. **HVA landmark.** Every "
                "distinct retained HVA pair in that same Allen session cohort. **Distance.** Squared "
                "difference of population means after subtracting finite-cell mean-sampling variance."
            ),
            "layout": "full",
        },
        {
            "id": "methods",
            "type": "markdown",
            "body": (
                "## Full-cell estimator and session bootstrap\n\n"
                "For cell populations X and Y, the estimator is (x̄−ȳ)² − sₓ²/nₓ − sᵧ²/nᵧ, "
                "the two-sample U-statistic estimate of the squared latent population-mean difference. "
                "Pair distances are averaged within session before sessions receive equal weight. "
                "MouseV2 and Allen sessions are resampled independently; the same Allen draw supplies "
                "both Allen landmarks, preserving their covariance. Negative corrected pair and session "
                "distances are retained rather than clipped."
            ),
            "layout": "full",
        },
        {
            "id": "robustness",
            "type": "markdown",
            "body": (
                "## The raw landmarks are more stable than the normalized ratio\n\n"
                f"The between-minus-within denominator is positive in {denominator_positive:.1%} of "
                "bootstrap draws. Near-zero denominators produce an unhelpfully wide ratio interval, "
                "so the score is a descriptive placement. The raw HVA-minus-within contrast remains "
                "positive at the 95% bootstrap level. Individual corrected pair distances can be negative "
                "because unbiased noise correction is not constrained; their session-averaged population "
                "estimands are not clipped."
            ),
            "sourceId": "landmarks",
            "layout": "full",
        },
        {
            "id": "validation",
            "type": "markdown",
            "body": (
                "## Validation status: share with caveats\n\n"
                f"All {len(validation)} retained session×population means and counts reproduce the current "
                f"Figure 3 TTFS input; maximum absolute mean error is {maximum_error:.3g}. Four unit tests "
                "cover the U-statistic correction, pair construction, scale calculation, and equal pair "
                "weighting within session. The main remaining biological caveat is that populations are "
                "not yet matched for RF position, layer, cell type, depth, or firing rate."
            ),
            "sourceId": "validation",
            "layout": "full",
        },
        {"id": "audit_table_block", "type": "table", "tableId": "audit_table", "layout": "full"},
        {
            "id": "next",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Treat the three raw distances as primary and the 0→1 score as a descriptive visual coordinate.\n"
                "2. Extend the same full-cell estimator to harmonized F1/F0 and trial-matched timescale, preserving the existing one-draw-per-neuron rule.\n"
                "3. Add RF/layer matching as a sensitivity analysis before making an anatomical-area claim in the paper."
            ),
            "layout": "full",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does the between-area-sized HVA separation replicate for timescale and harmonized F1/F0?\n"
                "- Which named HVA pairs contribute reproducibly across sessions rather than through a few large contrasts?\n"
                "- How much does RF-, layer-, and cell-type matching move each raw landmark?"
            ),
            "layout": "full",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "TTFS Area-Separation Landmark Pilot",
            "description": "Full-cell comparison of within-V1, V1-to-HVA, and HVA-to-HVA TTFS population distances.",
            "generatedAt": generated,
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "landmarks": records(point),
                "selected": records(selected),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (BUNDLE / "artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    chart_map = pd.DataFrame(
        [
            {
                "section": "raw_landmarks",
                "question": "How do the three TTFS distance landmarks compare?",
                "family": "Comparison & Ranking",
                "type": "bar",
                "dataset": "landmarks",
                "source": "landmarks",
                "palette_policy": "single-root preferred with neutral benchmark",
                "supported_claim": "HVA-HVA and V1-HVA distances are far above within-V1.",
            }
        ]
    )
    chart_map.to_csv(BUNDLE / "report_chart_map.csv", index=False)
    print(BUNDLE / "artifact.json")


if __name__ == "__main__":
    main()
