#!/usr/bin/env python3
"""Fail-closed validation of the completed Figure 4 rerun package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/figure4_rerun"
OUT = BASE / "v8_final"


def main() -> None:
    checks: dict[str, object] = {}
    primary = pd.read_csv(BASE / "v4_primary_dispersion/bootstrap_effects.csv.gz")
    primary_fail = pd.read_csv(BASE / "v4_primary_dispersion/bootstrap_failures_and_draw_ledger.csv.gz")
    checks["primary_effect_draws"] = len(primary)
    checks["primary_expected_effect_draws"] = 3 * 5000 * 6
    checks["primary_failures"] = int(primary_fail.failure.fillna("").ne("").sum())
    assert len(primary) == checks["primary_expected_effect_draws"]
    assert checks["primary_failures"] == 0

    hierarchy = pd.read_csv(BASE / "v5_hva_hierarchy/hierarchy_bootstrap.csv.gz")
    hierarchy_fail = pd.read_csv(BASE / "v5_hva_hierarchy/hierarchy_bootstrap_failures.csv")
    checks["hierarchy_draws"] = len(hierarchy)
    checks["hierarchy_failures"] = int(hierarchy_fail.failure.fillna("").ne("").sum())
    checks["hierarchy_attempts_accounted_for"] = len(hierarchy) + checks["hierarchy_failures"] == 3 * 5000
    assert checks["hierarchy_attempts_accounted_for"] and checks["hierarchy_failures"] / (3 * 5000) < .01

    control = pd.read_csv(BASE / "v6_matched_control/matched_control_bootstrap.csv.gz")
    control_fail = pd.read_csv(BASE / "v6_matched_control/matched_control_failures.csv")
    checks["control_draws"] = len(control)
    checks["control_failures"] = int(control_fail.failure.fillna("").ne("").sum())
    checks["control_attempts_accounted_for"] = len(control) + checks["control_failures"] == 3 * 5000
    assert checks["control_attempts_accounted_for"] and checks["control_failures"] / (3 * 5000) < .01

    central = pd.read_csv(BASE / "v7_central_companion/central_inclusive_bootstrap.csv.gz")
    central_fail = pd.read_csv(BASE / "v7_central_companion/central_inclusive_failures.csv")
    checks["central_draws"] = len(central)
    checks["central_failures"] = int(central_fail.failure.fillna("").ne("").sum())
    checks["central_attempts_accounted_for"] = len(central) + checks["central_failures"] == 3 * 5000
    assert checks["central_attempts_accounted_for"] and checks["central_failures"] / (3 * 5000) < .01

    primary_effects = pd.read_csv(BASE / "v4_primary_dispersion/primary_effects.csv")
    contrast = primary_effects.loc[primary_effects.effect.eq("hva_minus_v1")]
    checks["primary_contrast_intervals_contain_zero"] = bool(
        ((contrast.interval_low <= 0) & (contrast.interval_high >= 0)).all())
    assert checks["primary_contrast_intervals_contain_zero"]

    central_effects = pd.read_csv(BASE / "v7_central_companion/central_inclusive_effects.csv")
    checks["central_models_rank_deficient"] = bool(
        (~central_effects.identifiable_with_session_effects.astype(bool)).all())
    assert checks["central_models_rank_deficient"]

    hierarchy_effects = pd.read_csv(BASE / "v5_hva_hierarchy/hierarchy_slopes.csv")
    checks["hierarchy_designs_full_rank"] = bool(
        (hierarchy_effects.adjusted_rank == hierarchy_effects.adjusted_parameters).all()
        and (hierarchy_effects.unadjusted_rank == hierarchy_effects.unadjusted_parameters).all())
    checks["hierarchy_values_finite"] = bool(np.isfinite(hierarchy_effects.select_dtypes("number")).all().all())
    assert checks["hierarchy_designs_full_rank"] and checks["hierarchy_values_finite"]

    display_counts = pd.read_csv(OUT / "figure4_display_counts.csv")
    point_fits = pd.read_csv(BASE / "v4_primary_dispersion/draw_specific_point_fits.csv")
    central_fits = pd.read_csv(BASE / "v7_central_companion/central_inclusive_draw_fits.csv")
    counts_match = True
    for metric in primary_effects.metric.unique():
        fit_draw = 0 if metric == "Response timescale (ms)" else -1
        for panel, population in [("C", "V1"), ("D", "HVA")]:
            shown = display_counts.loc[
                display_counts.figure.eq("Figure4_CDE_rerun")
                & display_counts.panel.eq(panel)
                & display_counts.metric.eq(metric), "n_cells"
            ].sum()
            expected = point_fits.loc[
                point_fits.metric.eq(metric) & point_fits.population.eq(population)
                & point_fits.draw.eq(fit_draw), "n_cells"
            ]
            counts_match &= len(expected) == 1 and shown == expected.iloc[0]
        shown_central = display_counts.loc[
            display_counts.figure.eq("Figure4_Central_inclusive_companion")
            & display_counts.metric.eq(metric), "n_cells"
        ].sum()
        expected_central = central_fits.loc[
            central_fits.metric.eq(metric) & central_fits.draw.eq(fit_draw), "n_cells"
        ]
        counts_match &= len(expected_central) == 1 and shown_central == expected_central.iloc[0]
    checks["display_counts_match_fitted_population"] = bool(counts_match)
    assert checks["display_counts_match_fitted_population"]

    figures = [
        "Figure4_CDE_rerun.png", "Figure4_Central_inclusive_companion.png",
        "Figure4_hierarchy_RF_conditional.png", "Figure4_normalized_dispersion_companion.png",
    ]
    checks["figures_present"] = all((OUT / name).stat().st_size > 10_000 for name in figures)
    assert checks["figures_present"]

    (OUT / "final_package_validation.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final package validation", ""] + [f"- {key}: `{value}`" for key, value in checks.items()]
    (OUT / "FINAL_PACKAGE_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
