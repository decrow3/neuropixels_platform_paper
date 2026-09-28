#!/usr/bin/env python3
"""Matched Allen V1 versus equal-area pooled-HVA control for Figure 4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import HVA_CATEGORIES, METRICS, apply_cell_floor, build_response_registry
from scripts.validate_figure4_rerun_models import resample_animals

OUT = ROOT / "artifacts/figure4_rerun/v6_matched_control"


def control_population(registry: pd.DataFrame, metric: str) -> pd.DataFrame:
    local = registry.loc[registry.metric.eq(metric) & registry.population.isin(["Central", "HVA"])].copy()
    central_sessions = set(local.loc[local.population.eq("Central"), "session_id"])
    hva_sessions = set(local.loc[local.population.eq("HVA"), "session_id"])
    local = local.loc[local.session_id.isin(central_sessions & hva_sessions)].copy()
    local["control_group"] = np.where(local.population.eq("Central"), "V1", "HVA")
    if local.duplicated(["control_group", "unit_id"]).any():
        raise ValueError("Matched control duplicates cells")
    paired = local.groupby("session_id").control_group.nunique()
    local = local.loc[local.session_id.isin(paired.loc[paired.eq(2)].index)].copy()
    return local


def control_weights(table: pd.DataFrame) -> pd.Series:
    weights = pd.Series(0.0, index=table.index)
    v1 = table.control_group.eq("V1")
    hva = ~v1
    for mask, is_hva, group_total in [(v1, False, .5), (hva, True, .5)]:
        part = table.loc[mask]
        if is_hva:
            areas = part.category.nunique()
            area_factor = part.category.map(part.groupby("category").size()).astype(float) * 0 + areas
        else:
            area_factor = pd.Series(1.0, index=part.index)
        stratum = "category" if is_hva else "control_group"
        animals = part.groupby(stratum).animal_id.transform("nunique")
        recordings = part.groupby([stratum, "animal_id"]).recording_id.transform("nunique")
        cells = part.groupby([stratum, "animal_id", "recording_id"]).unit_id.transform("nunique")
        weights.loc[part.index] = group_total / (area_factor * animals * recordings * cells)
    np.testing.assert_allclose(weights.groupby(table.control_group).sum(), [.5, .5], atol=1e-12)
    return weights


def fit_control(table: pd.DataFrame) -> dict[str, float | int]:
    groups = ["V1", "HVA"]
    group_x = np.column_stack([table.control_group.eq(group).to_numpy(float) for group in groups])
    sessions = sorted(table.session_id.unique())
    nuisance = np.column_stack([table.session_id.eq(value).to_numpy(float) for value in sessions[1:]])
    x = np.column_stack([group_x, nuisance])
    weights = control_weights(table).to_numpy()
    root = np.sqrt(weights)
    beta, _, rank, singular = np.linalg.lstsq(x * root[:, None], table.value.to_numpy(float) * root, rcond=1e-10)
    if rank != x.shape[1]:
        raise np.linalg.LinAlgError(f"Matched-control rank deficient ({rank}/{x.shape[1]})")
    difference = float(beta[1] - beta[0])
    return {
        "v1_mean": float(beta[0]), "pooled_hva_mean": float(beta[1]),
        "hva_minus_v1": difference, "two_group_dispersion": difference**2 / 4,
        "rank": int(rank), "parameters": x.shape[1],
        "condition_number": float(singular[0] / singular[-1]),
        "n_cells": len(table), "n_sessions": table.session_id.nunique(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = apply_cell_floor(build_response_registry(), 5)
    rng = np.random.default_rng(args.seed)
    points, draws, failures = [], [], []
    for metric in METRICS:
        table = control_population(registry, metric)
        point = fit_control(table)
        point["metric"] = metric
        points.append(point)
        for repetition in range(args.bootstrap):
            failure = ""
            try:
                sample = resample_animals(table, rng)
                if set(sample.control_group) != {"V1", "HVA"}:
                    raise np.linalg.LinAlgError("lost control group")
                fit = fit_control(sample)
                draws.append({"metric": metric, "replicate": repetition, **fit})
            except (ValueError, np.linalg.LinAlgError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failures.append({"metric": metric, "replicate": repetition, "failure": failure})
        print(f"bootstrapped control {metric}", flush=True)
    points = pd.DataFrame(points)
    draws = pd.DataFrame(draws)
    for column in ["hva_minus_v1", "two_group_dispersion"]:
        points[column + "_low"] = points.metric.map(lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.025))
        points[column + "_high"] = points.metric.map(lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.975))
    points.to_csv(output / "matched_control.csv", index=False)
    draws.to_csv(output / "matched_control_bootstrap.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(failures).to_csv(output / "matched_control_failures.csv", index=False)
    print(points.to_string(index=False))


if __name__ == "__main__":
    main()
