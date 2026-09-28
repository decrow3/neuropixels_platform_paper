#!/usr/bin/env python3
"""Quantify HVA response selection separately from same-cell RF adjustment."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import METRICS, apply_cell_floor, build_response_registry, fit_standardized_dispersion
from scripts.run_figure4_hva_hierarchy import SUPPORT, fit_slope

OUT = ROOT / "artifacts/figure4_rerun/v5_hva_hierarchy"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = apply_cell_floor(build_response_registry(), 5)
    registry = registry.loc[registry.population.eq("HVA")].copy()
    support = pd.read_csv(SUPPORT, dtype={"unit_id": str})[[
        "metric", "source", "unit_id", "azimuth", "elevation", "rf_qualified", "hva_shared_support",
    ]]
    joined = registry.merge(support, on=["metric", "source", "unit_id"], how="left", validate="one_to_one")
    joined["hierarchy_score"] = joined.category.map(
        {"LM": -.093, "RL": -.059, "AL": .152, "PM": .327, "AM": .441})
    rows = []
    for metric in METRICS:
        source = joined.loc[joined.metric.eq(metric)]
        stages = [
            ("1_response_eligible", source),
            ("2_rf_qualified", source.loc[source.rf_qualified.eq(True)]),
            ("3_shared_support_unadjusted", source.loc[source.hva_shared_support.eq(True)]),
        ]
        for stage, table in stages:
            table = apply_cell_floor(table, 5)
            slope = fit_slope(table, adjusted=False)
            dispersion = fit_standardized_dispersion(table)
            rows.append({
                "metric": metric, "stage": stage, "n_cells": len(table),
                "n_recordings": table.recording_id.nunique(),
                "hierarchy_slope": slope["slope"],
                "hierarchy_score_range_change": slope["score_range_change"],
                "category_dispersion": dispersion.dispersion,
                "root_category_dispersion": dispersion.root_dispersion,
            })
        supported = apply_cell_floor(source.loc[source.hva_shared_support.eq(True)], 5)
        adjusted = fit_slope(supported, adjusted=True)
        rows.append({
            "metric": metric, "stage": "4_same_cells_rf_adjusted", "n_cells": len(supported),
            "n_recordings": supported.recording_id.nunique(),
            "hierarchy_slope": adjusted["slope"],
            "hierarchy_score_range_change": adjusted["score_range_change"],
            "category_dispersion": float("nan"), "root_category_dispersion": float("nan"),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "rf_effect_ladder.csv", index=False)
    lines = [
        "# HVA RF population/effect ladder", "",
        "Stages 1→2→3 change the cell population and therefore quantify selection. Stages 3→4 use exactly the same cells and isolate the specified quadratic RF adjustment. Category dispersion is not shown at stage 4 because cross-population adjusted dispersion failed the prespecified support gate; the supported HVA-only conditional target is the hierarchy slope.", "",
        "| Metric | Stage | Cells | Recordings | Hierarchy slope | Score-range change | Category dispersion |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        dispersion = "—" if pd.isna(row.category_dispersion) else f"{row.category_dispersion:.5g}"
        lines.append(f"| {row.metric} | {row.stage} | {row.n_cells} | {row.n_recordings} | {row.hierarchy_slope:.5g} | {row.hierarchy_score_range_change:.5g} | {dispersion} |")
    (OUT / "RF_EFFECT_LADDER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
