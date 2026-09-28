#!/usr/bin/env python3
"""Estimate same-cell unadjusted and RF-conditional HVA hierarchy slopes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import (
    HIERARCHY_SCORES, METRICS, apply_cell_floor, build_response_registry,
    category_recording_weights, rf_features,
)
from scripts.validate_figure4_rerun_models import resample_animals

SUPPORT = ROOT / "artifacts/figure4_rerun/v2_support/hva_rf_support_cells.csv.gz"
OUT = ROOT / "artifacts/figure4_rerun/v5_hva_hierarchy"
SCORE_RANGE = max(HIERARCHY_SCORES.values()) - min(HIERARCHY_SCORES.values())


def load_populations() -> dict[str, pd.DataFrame]:
    registry = apply_cell_floor(build_response_registry(), 5)
    registry = registry.loc[registry.population.eq("HVA")].copy()
    support = pd.read_csv(SUPPORT, dtype={"unit_id": str})[[
        "metric", "source", "unit_id", "azimuth", "elevation", "hva_shared_support",
    ]]
    table = registry.merge(support, on=["metric", "source", "unit_id"], how="left", validate="many_to_one")
    table = table.loc[table.hva_shared_support.eq(True)].copy()
    table = apply_cell_floor(table, 5)
    table["hierarchy_score"] = table.category.map(HIERARCHY_SCORES)
    populations = {}
    for metric in METRICS:
        local = table.loc[table.metric.eq(metric)].copy()
        if local.duplicated("unit_id").any():
            raise ValueError(f"HVA hierarchy population duplicates cells for {metric}")
        if set(local.category) != set(HIERARCHY_SCORES):
            raise ValueError(f"HVA hierarchy population lost an area for {metric}")
        populations[metric] = local
    return populations


def fit_slope(table: pd.DataFrame, *, adjusted: bool) -> dict[str, float | int]:
    sessions = sorted(table.session_id.unique())
    nuisance = np.column_stack([
        table.session_id.eq(value).to_numpy(float) for value in sessions[1:]
    ]) if len(sessions) > 1 else np.empty((len(table), 0))
    columns = [np.ones(len(table)), table.hierarchy_score.to_numpy(float), nuisance]
    if adjusted:
        columns.append(rf_features(table))
    x = np.column_stack(columns)
    weights = category_recording_weights(table).to_numpy()
    root = np.sqrt(weights)
    xw, yw = x * root[:, None], table.value.to_numpy(float) * root
    beta, _, rank, singular = np.linalg.lstsq(xw, yw, rcond=1e-10)
    if rank != x.shape[1]:
        raise np.linalg.LinAlgError(f"Hierarchy design rank deficient ({rank}/{x.shape[1]})")
    residual = table.value.to_numpy(float) - x @ beta
    return {
        "slope": float(beta[1]), "score_range_change": float(beta[1] * SCORE_RANGE),
        "rank": int(rank), "parameters": x.shape[1],
        "condition_number": float(singular[0] / singular[-1]),
        "weighted_residual_variance": float(np.sum(weights * residual**2)),
        "n_cells": len(table), "n_animals": table.animal_id.nunique(),
        "n_recordings": table.recording_id.nunique(),
    }


def within_between_diagnostic(table: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = rf_features(table)
    names = ["azimuth", "elevation", "azimuth_squared", "elevation_squared", "azimuth_by_elevation"]
    frame = pd.DataFrame(features, columns=names, index=table.index)
    frame["category"] = table.category
    means = frame.groupby("category")[names].mean()
    means["hierarchy_score"] = means.index.map(HIERARCHY_SCORES)
    rows = []
    for name in names:
        rows.append({
            "metric": metric, "feature": name,
            "between_area_correlation_with_hierarchy": means[name].corr(means.hierarchy_score),
            "between_area_range": means[name].max() - means[name].min(),
            "pooled_within_area_sd": float(frame[name].sub(frame.groupby("category")[name].transform("mean")).std()),
        })
    means = means.reset_index().assign(metric=metric)
    return pd.DataFrame(rows), means


def point_and_influence(populations: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points, influence, diagnostics, area_means = [], [], [], []
    for metric, table in populations.items():
        unadjusted, adjusted = fit_slope(table, adjusted=False), fit_slope(table, adjusted=True)
        points.append({
            "metric": metric, "unadjusted_slope": unadjusted["slope"],
            "adjusted_slope": adjusted["slope"],
            "adjusted_minus_unadjusted": adjusted["slope"] - unadjusted["slope"],
            "unadjusted_score_range_change": unadjusted["score_range_change"],
            "adjusted_score_range_change": adjusted["score_range_change"],
            **{f"unadjusted_{key}": value for key, value in unadjusted.items() if key not in {"slope", "score_range_change"}},
            **{f"adjusted_{key}": value for key, value in adjusted.items() if key not in {"slope", "score_range_change"}},
        })
        for animal in sorted(table.animal_id.unique()):
            reduced = table.loc[table.animal_id.ne(animal)]
            try:
                u, a = fit_slope(reduced, adjusted=False), fit_slope(reduced, adjusted=True)
                influence.append({
                    "metric": metric, "animal_id": animal,
                    "unadjusted_slope": u["slope"], "adjusted_slope": a["slope"],
                    "adjusted_minus_unadjusted": a["slope"] - u["slope"], "failure": "",
                })
            except (ValueError, np.linalg.LinAlgError) as exc:
                influence.append({"metric": metric, "animal_id": animal, "failure": f"{type(exc).__name__}: {exc}"})
        diag, means = within_between_diagnostic(table, metric)
        diagnostics.append(diag)
        area_means.append(means)
    return pd.DataFrame(points), pd.DataFrame(influence), pd.concat(diagnostics), pd.concat(area_means)


def bootstrap(populations: dict[str, pd.DataFrame], repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, failures = [], []
    for metric, table in populations.items():
        for repetition in range(repetitions):
            failure = ""
            try:
                sample = resample_animals(table, rng)
                if set(sample.category) != set(HIERARCHY_SCORES):
                    raise np.linalg.LinAlgError("lost HVA category")
                unadjusted = fit_slope(sample, adjusted=False)
                adjusted = fit_slope(sample, adjusted=True)
                rows.append({
                    "metric": metric, "replicate": repetition,
                    "unadjusted_slope": unadjusted["slope"],
                    "adjusted_slope": adjusted["slope"],
                    "adjusted_minus_unadjusted": adjusted["slope"] - unadjusted["slope"],
                    "unadjusted_score_range_change": unadjusted["score_range_change"],
                    "adjusted_score_range_change": adjusted["score_range_change"],
                })
            except (ValueError, np.linalg.LinAlgError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failures.append({"metric": metric, "replicate": repetition, "failure": failure})
        print(f"bootstrapped hierarchy {metric}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(failures)


def add_intervals(points: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    result = points.copy()
    for column in [
        "unadjusted_slope", "adjusted_slope", "adjusted_minus_unadjusted",
        "unadjusted_score_range_change", "adjusted_score_range_change",
    ]:
        result[column + "_low"] = np.nan
        result[column + "_high"] = np.nan
        for index, row in result.iterrows():
            values = draws.loc[draws.metric.eq(row.metric), column]
            result.loc[index, [column + "_low", column + "_high"]] = np.quantile(values, [.025, .975])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    populations = load_populations()
    points, influence, diagnostics, area_means = point_and_influence(populations)
    draws, failures = bootstrap(populations, args.bootstrap, args.seed)
    points = add_intervals(points, draws)
    points.to_csv(output / "hierarchy_slopes.csv", index=False)
    draws.to_csv(output / "hierarchy_bootstrap.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    failures.to_csv(output / "hierarchy_bootstrap_failures.csv", index=False)
    influence.to_csv(output / "hierarchy_leave_one_animal_out.csv", index=False)
    diagnostics.to_csv(output / "rf_within_between_diagnostic.csv", index=False)
    area_means.to_csv(output / "rf_area_feature_means.csv", index=False)
    lines = [
        "# HVA-only RF-conditional hierarchy", "",
        f"Same HVA-supported cells, {args.bootstrap:,} paired whole-animal bootstrap replicates, seed {args.seed}. Slopes are signed response units per hierarchy-score unit; score-range changes span {SCORE_RANGE:.3f}.", "",
        "| Metric | Unadjusted slope [95% interval] | Adjusted slope [95% interval] | Adjusted − unadjusted [95% interval] |", "|---|---:|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.unadjusted_slope:.4g} [{row.unadjusted_slope_low:.4g}, {row.unadjusted_slope_high:.4g}] | {row.adjusted_slope:.4g} [{row.adjusted_slope_low:.4g}, {row.adjusted_slope_high:.4g}] | {row.adjusted_minus_unadjusted:.4g} [{row.adjusted_minus_unadjusted_low:.4g}, {row.adjusted_minus_unadjusted_high:.4g}] |")
    failed = failures.failure.ne("").groupby(failures.metric).sum()
    lines += ["", "Adjustment uses the fixed five-term quadratic RF surface jointly with hierarchy and session nuisance effects. It is conditional sharing of information, not causal mediation. A change in interval significance is not used as evidence of slope attenuation; the paired adjusted-minus-unadjusted interval is reported directly.", "", "Bootstrap failures: " + ", ".join(f"{metric}={int(value)}" for metric, value in failed.items()) + "."]
    (output / "HVA_HIERARCHY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(points.to_string(index=False))


if __name__ == "__main__":
    main()
