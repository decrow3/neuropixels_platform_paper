#!/usr/bin/env python3
"""Core point and leave-one-animal-out sensitivities for Figure 4 dispersion."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units
from scripts.figure3_robust_spread_comparison import MOUSE_TIMESCALE_TRIAL_BRIDGE
from scripts.figure4_rerun_core import METRICS, apply_cell_floor, build_response_registry, fit_standardized_dispersion
from scripts.run_figure4_primary_dispersion import point_table

OUT = ROOT / "artifacts/figure4_rerun/v4_primary_dispersion/sensitivities"


def averaged_effect(
    registry: pd.DataFrame,
    metric: str,
    population: str,
    *,
    equal_cells: bool = False,
) -> float:
    draws = range(10) if metric == METRICS[2] else [None]
    values = []
    for draw in draws:
        table = point_table(registry, metric, population, draw)
        values.append(fit_standardized_dispersion(table, equal_cells=equal_cells).dispersion)
    return float(np.mean(values))


def timescale_scenarios(registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary = registry.copy()
    mouse = primary.loc[primary.metric.eq(METRICS[2]) & primary.population.eq("V1")]
    counts = mouse.groupby("unit_id").draw_id.nunique()
    complete_ids = set(counts.loc[counts.eq(10)].index)
    complete = primary.loc[
        primary.metric.ne(METRICS[2])
        | primary.population.ne("V1")
        | primary.unit_id.isin(complete_ids)
    ].copy()
    complete = apply_cell_floor(complete, 5)

    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE, dtype={"unit_id": str})
    tight_mouse = bridge.loc[
        bridge.view.eq("mouse_matched_150") & bridge.valid_timescale.astype(bool)
        & pd.to_numeric(bridge.fit_error_ms, errors="coerce").lt(10)
    ][["unit_id", "subsample"]].rename(columns={"subsample": "draw_id"})
    tight_mouse_keys = pd.MultiIndex.from_frame(tight_mouse)
    allen = load_allen_units(population_profile="common_qc")
    tight_allen_ids = set(allen.loc[
        pd.to_numeric(allen.err_ac, errors="coerce").lt(10), "ecephys_unit_id"
    ].astype(str))
    is_tau = primary.metric.eq(METRICS[2])
    mouse_tau = is_tau & primary.population.eq("V1")
    hva_tau = is_tau & primary.population.eq("HVA")
    keep_mouse = pd.MultiIndex.from_frame(primary.loc[mouse_tau, ["unit_id", "draw_id"]]).isin(tight_mouse_keys)
    keep = ~is_tau
    keep.loc[mouse_tau] = keep_mouse
    keep.loc[hva_tau] = primary.loc[hva_tau, "unit_id"].isin(tight_allen_ids)
    tight = apply_cell_floor(primary.loc[keep].copy(), 5)
    return {"primary": primary, "ten_complete_draws": complete, "fit_error_lt_10_ms": tight}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry5 = apply_cell_floor(build_response_registry(), 5)
    registry10 = apply_cell_floor(build_response_registry(), 10)
    rows = []
    for floor, registry in [(5, registry5), (10, registry10)]:
        for weighting, equal_cells in [("balanced", False), ("equal_cells", True)]:
            for metric in METRICS:
                v1 = averaged_effect(registry, metric, "V1", equal_cells=equal_cells)
                hva = averaged_effect(registry, metric, "HVA", equal_cells=equal_cells)
                rows.append({
                    "analysis": "floor_weighting", "metric": metric,
                    "cell_floor": floor, "weighting": weighting,
                    "v1_dispersion": v1, "hva_dispersion": hva, "hva_minus_v1": hva - v1,
                })
    for scenario, registry in timescale_scenarios(registry5).items():
        v1 = averaged_effect(registry, METRICS[2], "V1")
        hva = averaged_effect(registry, METRICS[2], "HVA")
        rows.append({
            "analysis": "timescale_population", "metric": METRICS[2],
            "scenario": scenario, "cell_floor": 5, "weighting": "balanced",
            "v1_dispersion": v1, "hva_dispersion": hva, "hva_minus_v1": hva - v1,
        })
    sensitivities = pd.DataFrame(rows)
    sensitivities.to_csv(OUT / "point_sensitivities.csv", index=False)

    influence = []
    baseline = {
        (metric, population): averaged_effect(registry5, metric, population)
        for metric in METRICS for population in ["V1", "HVA"]
    }
    for metric in METRICS:
        for population in ["V1", "HVA"]:
            animals = sorted(registry5.loc[
                registry5.metric.eq(metric) & registry5.population.eq(population), "animal_id"
            ].unique())
            for animal in animals:
                reduced = registry5.loc[
                    ~(registry5.metric.eq(metric) & registry5.population.eq(population) & registry5.animal_id.eq(animal))
                ].copy()
                try:
                    value = averaged_effect(reduced, metric, population)
                    failure = ""
                except (ValueError, np.linalg.LinAlgError) as exc:
                    value, failure = np.nan, f"{type(exc).__name__}: {exc}"
                influence.append({
                    "metric": metric, "population": population, "animal_id": animal,
                    "baseline_dispersion": baseline[(metric, population)],
                    "leave_one_out_dispersion": value,
                    "change": value - baseline[(metric, population)], "failure": failure,
                })
    influence = pd.DataFrame(influence)
    influence.to_csv(OUT / "leave_one_animal_out.csv", index=False)
    print(sensitivities.to_string(index=False))
    print("\nLargest absolute influence")
    indices = influence.groupby(["metric", "population"]).change.apply(lambda x: x.abs().idxmax()).to_numpy()
    print(influence.loc[indices].to_string(index=False))


if __name__ == "__main__":
    main()
