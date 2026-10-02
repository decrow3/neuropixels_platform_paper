#!/usr/bin/env python3
"""Descriptive Central-inclusive five-location V1 companion for Figure 4."""

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
    METRICS, V1_CATEGORIES, apply_cell_floor, build_response_registry,
    category_recording_weights,
)

OUT = ROOT / "artifacts/figure4_rerun/v7_central_companion"
CATEGORIES = (*V1_CATEGORIES[:2], "Central", *V1_CATEGORIES[2:])


def population(registry: pd.DataFrame, metric: str, draw: int | None) -> pd.DataFrame:
    local = registry.loc[
        registry.metric.eq(metric) & registry.population.isin(["V1", "Central"])
    ].copy()
    local.loc[local.population.eq("Central"), "category"] = "Central"
    if metric == METRICS[2]:
        mouse = local.source.eq("MouseV2")
        local = local.loc[(mouse & local.draw_id.eq(int(draw))) | (~mouse & local.draw_id.eq(-1))].copy()
        local = apply_cell_floor(local, 5)
    if local.duplicated("unit_id").any():
        raise ValueError("Central-inclusive point population duplicates cells")
    return local


def descriptive_fit(table: pd.DataFrame) -> dict[str, object]:
    if set(table.category) != set(CATEGORIES):
        raise ValueError("Central-inclusive population lost a category")
    weights = category_recording_weights(table)
    means = {
        category: float(np.sum(weights.loc[table.category.eq(category)] * table.loc[table.category.eq(category), "value"]))
        * len(CATEGORIES)
        for category in CATEGORIES
    }
    values = np.asarray(list(means.values()))
    dispersion = float(np.mean((values - values.mean()) ** 2))
    y = table.value.to_numpy(float)
    overall = float(np.sum(weights * y))
    total = float(np.sum(weights * (y - overall) ** 2))
    return {
        "dispersion": dispersion,
        "root_dispersion": float(np.sqrt(dispersion)),
        "total_variance": total,
        "fraction_percent": 100 * dispersion / total,
        "category_means": ";".join(f"{key}={value:.12g}" for key, value in means.items()),
        "n_cells": len(table), "n_animals": table.animal_id.nunique(),
        "n_recordings": table.recording_id.nunique(),
    }


def identifiability(table: pd.DataFrame) -> tuple[int, int]:
    categories = list(CATEGORIES)
    sessions = sorted(table.session_id.unique())
    x = np.column_stack(
        [table.category.eq(value).to_numpy(float) for value in categories]
        + [table.session_id.eq(value).to_numpy(float) for value in sessions[1:]]
    )
    return int(np.linalg.matrix_rank(x)), int(x.shape[1])


def resample(registry: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    source = registry.loc[
        registry.metric.eq(metric) & registry.population.isin(["V1", "Central"])
    ].copy()
    source.loc[source.population.eq("Central"), "category"] = "Central"
    blocks = []
    for cohort in ["MouseV2", "Allen"]:
        part = source.loc[source.source.eq(cohort)]
        animals = np.asarray(sorted(part.animal_id.unique()), dtype=object)
        sampled = rng.choice(animals, len(animals), replace=True)
        for occurrence, animal in enumerate(sampled):
            animal_block = part.loc[part.animal_id.eq(animal)]
            for session, session_block in animal_block.groupby("session_id", sort=False):
                session_block = session_block.copy()
                if metric == METRICS[2] and cohort == "MouseV2":
                    draw = int(rng.integers(0, 10))
                    session_block = session_block.loc[session_block.draw_id.eq(draw)].copy()
                prefix = f"bootstrap_{cohort}_{occurrence}:{animal}:"
                for column in ["animal_id", "session_id", "recording_id", "unit_id"]:
                    session_block[column] = prefix + session_block[column].astype(str)
                blocks.append(session_block)
    return apply_cell_floor(pd.concat(blocks, ignore_index=True), 5)


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
    points, draw_rows, failure_rows, mean_rows = [], [], [], []
    for metric in METRICS:
        fits = []
        for draw in (range(10) if metric == METRICS[2] else [None]):
            table = population(registry, metric, draw)
            fit = descriptive_fit(table)
            rank, parameters = identifiability(table)
            fits.append(fit)
            mean_rows.append({"metric": metric, "draw": -1 if draw is None else draw, **fit,
                              "fixed_session_rank": rank, "fixed_session_parameters": parameters})
        points.append({
            "metric": metric,
            **{key: float(np.mean([fit[key] for fit in fits])) for key in
               ["dispersion", "root_dispersion", "total_variance", "fraction_percent"]},
            "fixed_session_rank": rank, "fixed_session_parameters": parameters,
            "identifiable_with_session_effects": rank == parameters,
        })
        for repetition in range(args.bootstrap):
            failure = ""
            try:
                sample = resample(registry, metric, rng)
                fit = descriptive_fit(sample)
                draw_rows.append({"metric": metric, "replicate": repetition, **fit})
            except (ValueError, np.linalg.LinAlgError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failure_rows.append({"metric": metric, "replicate": repetition, "failure": failure})
        print(f"bootstrapped Central companion {metric}", flush=True)
    points = pd.DataFrame(points)
    draws = pd.DataFrame(draw_rows)
    for column in ["dispersion", "fraction_percent"]:
        points[column + "_low"] = points.metric.map(
            lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.025))
        points[column + "_high"] = points.metric.map(
            lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.975))
    points.to_csv(output / "central_inclusive_effects.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(output / "central_inclusive_draw_fits.csv", index=False)
    draws.to_csv(output / "central_inclusive_bootstrap.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})
    failures = pd.DataFrame(failure_rows)
    failures.to_csv(output / "central_inclusive_failures.csv", index=False)
    lines = [
        "# Central-inclusive five-location V1 companion", "",
        f"Descriptive equal-category estimates with {args.bootstrap:,} independently source-stratified whole-animal bootstrap replicates (seed {args.seed}). Central is Allen V1; A/E/C/B are MouseV2. Central is displayed between E and C.", "",
        "A category-plus-session model is rank deficient for every metric because Central is nested entirely in the Allen cohort/sessions. The table therefore reports descriptive weighted category means and dispersion, not an inferential location effect adjusted for session.", "",
        "| Metric | Dispersion [95% interval] | Root dispersion | Raw total variance | Normalized fraction | Fixed-session rank |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in points.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.dispersion:.5g} [{row.dispersion_low:.5g}, {row.dispersion_high:.5g}] | {row.root_dispersion:.4g} | {row.total_variance:.4g} | {row.fraction_percent:.3g}% | {row.fixed_session_rank}/{row.fixed_session_parameters} |")
    lines += ["", "Intervals describe resampling variability of the mixed-cohort descriptive quantity; they do not separate anatomical location from cohort or session effects.", "", "Bootstrap failures: " + ", ".join(f"{metric}={int(value)}" for metric, value in failures.failure.ne("").groupby(failures.metric).sum().items()) + "."]
    (output / "CENTRAL_COMPANION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(points.to_string(index=False))


if __name__ == "__main__":
    main()
