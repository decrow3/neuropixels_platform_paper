#!/usr/bin/env python3
"""Create the prespecified normalized-dispersion companion from saved primary draws."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure4_rerun/v4_primary_dispersion"


def main() -> None:
    primary = pd.read_csv(OUT / "primary_effects.csv")
    draws = pd.read_csv(OUT / "bootstrap_effects.csv.gz")
    rows = []
    mapping = {
        "v1_fraction_percent": "v1",
        "hva_fraction_percent": "hva",
        "fraction_difference_pp": "hva_minus_v1",
    }
    for metric in primary.metric.unique():
        point = primary.loc[primary.metric.eq(metric)].set_index("effect")
        estimates = {
            "v1_fraction_percent": point.loc["v1", "fraction_percent"],
            "hva_fraction_percent": point.loc["hva", "fraction_percent"],
            "fraction_difference_pp": point.loc["hva", "fraction_percent"] - point.loc["v1", "fraction_percent"],
        }
        for effect, source_effect in mapping.items():
            values = draws.loc[draws.metric.eq(metric) & draws.effect.eq(effect), "estimate"]
            low, high = np.quantile(values, [.025, .975])
            rows.append({"metric": metric, "effect": source_effect, "estimate": estimates[effect],
                         "interval_low": low, "interval_high": high})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "normalized_effects.csv", index=False)
    lines = [
        "# Normalized category-dispersion companion", "",
        "Values are 100 × category dispersion / weighted raw-cell outcome variance in the same population and draw. Differences are HVA minus V1 in percentage points. This is not predictive variance explained or classical omega-squared.", "",
        "| Metric | Effect | Estimate [95% interval] |", "|---|---|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.effect} | {row.estimate:.5g} [{row.interval_low:.5g}, {row.interval_high:.5g}] |")
    (OUT / "NORMALIZED_DISPERSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
