#!/usr/bin/env python3
"""Estimate prespecified unadjusted Figure 4 category dispersions and contrasts."""

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
    HVA_CATEGORIES, METRICS, V1_CATEGORIES, apply_cell_floor,
    build_response_registry, fit_standardized_dispersion,
)

OUT = ROOT / "artifacts/figure4_rerun/v4_primary_dispersion"


def point_table(registry: pd.DataFrame, metric: str, population: str, draw: int | None) -> pd.DataFrame:
    local = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)].copy()
    if metric == METRICS[2]:
        selected_draw = int(draw) if population == "V1" else -1
        local = local.loc[local.draw_id.eq(selected_draw)].copy()
        local = apply_cell_floor(local, 5)
    if local.duplicated("unit_id").any():
        raise ValueError(f"Point population has duplicate cells: {metric}/{population}/{draw}")
    return local


def resample_population(
    registry: pd.DataFrame,
    metric: str,
    population: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str]:
    source = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)].copy()
    animals = np.asarray(sorted(source.animal_id.unique()), dtype=object)
    sampled = rng.choice(animals, size=len(animals), replace=True)
    blocks, draw_ledger = [], []
    for occurrence, animal in enumerate(sampled):
        animal_block = source.loc[source.animal_id.eq(animal)].copy()
        for session, session_block in animal_block.groupby("session_id", sort=False):
            if metric == METRICS[2]:
                draw = int(rng.integers(0, 10)) if population == "V1" else -1
                session_block = session_block.loc[session_block.draw_id.eq(draw)].copy()
                draw_ledger.append(f"{occurrence}:{animal}:{session}:{draw}")
            prefix = f"bootstrap_{occurrence}:{animal}:"
            session_block["animal_id"] = prefix + session_block.animal_id.astype(str)
            session_block["session_id"] = prefix + session_block.session_id.astype(str)
            session_block["recording_id"] = prefix + session_block.recording_id.astype(str)
            session_block["unit_id"] = prefix + session_block.unit_id.astype(str)
            blocks.append(session_block)
    result = pd.concat(blocks, ignore_index=True)
    result = apply_cell_floor(result, 5)
    return result, ";".join(draw_ledger)


def fit_row(fit, *, metric: str, population: str, draw: int | str) -> dict[str, object]:
    return {
        "metric": metric, "population": population, "draw": draw,
        "dispersion": fit.dispersion, "root_dispersion": fit.root_dispersion,
        "total_variance": fit.total_variance, "fraction_percent": fit.fraction_percent,
        "n_cells": fit.n_cells, "n_recordings": fit.n_recordings,
        "n_animals": fit.n_animals, "rank": fit.rank, "parameters": fit.parameters,
        "condition_number": fit.condition_number,
        "category_means": ";".join(f"{key}={value:.12g}" for key, value in fit.category_means.items()),
    }


def point_estimates(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows, effect_rows = [], []
    for metric in METRICS:
        draws = range(10) if metric == METRICS[2] else [None]
        per_population = {"V1": [], "HVA": []}
        for draw in draws:
            for population in ["V1", "HVA"]:
                table = point_table(registry, metric, population, draw)
                expected = set(V1_CATEGORIES if population == "V1" else HVA_CATEGORIES)
                if set(table.category) != expected:
                    raise ValueError(f"Point draw lost categories: {metric}/{population}/{draw}")
                fit = fit_standardized_dispersion(table)
                fit_rows.append(fit_row(fit, metric=metric, population=population, draw=-1 if draw is None else draw))
                per_population[population].append(fit)
        v1 = float(np.mean([fit.dispersion for fit in per_population["V1"]]))
        hva = float(np.mean([fit.dispersion for fit in per_population["HVA"]]))
        for population, fits in per_population.items():
            effect_rows.append({
                "metric": metric, "effect": population.lower(),
                "estimate": float(np.mean([fit.dispersion for fit in fits])),
                "root_response_units": float(np.mean([fit.root_dispersion for fit in fits])),
                "fraction_percent": float(np.mean([fit.fraction_percent for fit in fits])),
                "total_variance": float(np.mean([fit.total_variance for fit in fits])),
            })
        effect_rows.append({"metric": metric, "effect": "hva_minus_v1", "estimate": hva - v1})
    return pd.DataFrame(fit_rows), pd.DataFrame(effect_rows)


def bootstrap(
    registry: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, failures = [], []
    for metric in METRICS:
        expected = {"V1": set(V1_CATEGORIES), "HVA": set(HVA_CATEGORIES)}
        for repetition in range(repetitions):
            fits, ledgers = {}, {}
            failure = ""
            try:
                for population in ["V1", "HVA"]:
                    table, ledger = resample_population(registry, metric, population, rng)
                    ledgers[population] = ledger
                    if set(table.category) != expected[population]:
                        raise np.linalg.LinAlgError(f"lost {population} category")
                    fits[population] = fit_standardized_dispersion(table)
                rows += [
                    {"metric": metric, "replicate": repetition, "effect": "v1", "estimate": fits["V1"].dispersion},
                    {"metric": metric, "replicate": repetition, "effect": "hva", "estimate": fits["HVA"].dispersion},
                    {"metric": metric, "replicate": repetition, "effect": "hva_minus_v1", "estimate": fits["HVA"].dispersion - fits["V1"].dispersion},
                    {"metric": metric, "replicate": repetition, "effect": "v1_fraction_percent", "estimate": fits["V1"].fraction_percent},
                    {"metric": metric, "replicate": repetition, "effect": "hva_fraction_percent", "estimate": fits["HVA"].fraction_percent},
                    {"metric": metric, "replicate": repetition, "effect": "fraction_difference_pp", "estimate": fits["HVA"].fraction_percent - fits["V1"].fraction_percent},
                ]
            except (np.linalg.LinAlgError, ValueError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failures.append({
                "metric": metric, "replicate": repetition, "failure": failure,
                "v1_draw_ledger": ledgers.get("V1", ""), "hva_draw_ledger": ledgers.get("HVA", ""),
            })
        print(f"bootstrapped {metric}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(failures)


def add_intervals(effects: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    result = effects.copy()
    result["interval_low"] = np.nan
    result["interval_high"] = np.nan
    mapping = {"v1": "v1", "hva": "hva", "hva_minus_v1": "hva_minus_v1"}
    for index, row in result.iterrows():
        values = draws.loc[draws.metric.eq(row.metric) & draws.effect.eq(mapping[row.effect]), "estimate"]
        result.loc[index, ["interval_low", "interval_high"]] = np.quantile(values, [.025, .975])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = apply_cell_floor(build_response_registry(), 5)
    point_fits, effects = point_estimates(registry)
    draws, failures = bootstrap(registry, repetitions=args.bootstrap, seed=args.seed)
    effects = add_intervals(effects, draws)
    point_fits.to_csv(output / "draw_specific_point_fits.csv", index=False)
    effects.to_csv(output / "primary_effects.csv", index=False)
    draws.to_csv(output / "bootstrap_effects.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    failures.to_csv(output / "bootstrap_failures_and_draw_ledger.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    failed = failures.failure.ne("").groupby(failures.metric).sum()
    lines = [
        "# Figure 4 primary unadjusted category dispersion", "",
        f"Whole-animal bootstrap with {args.bootstrap:,} replicates and seed {args.seed}. MouseV2 timescale selects one coherent matched-trial draw per sampled session occurrence. Point estimates average ten complete draw-specific effects.", "",
        "| Metric | Effect | Estimate | 95% percentile interval | Root dispersion | Raw total variance | Normalized fraction |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        root = f"{row.root_response_units:.4g}" if np.isfinite(row.root_response_units) else "—"
        total = f"{row.total_variance:.4g}" if np.isfinite(row.total_variance) else "—"
        fraction = f"{row.fraction_percent:.3g}%" if np.isfinite(row.fraction_percent) else "—"
        lines.append(f"| {row.metric} | {row.effect} | {row.estimate:.5g} | [{row.interval_low:.5g}, {row.interval_high:.5g}] | {root} | {total} | {fraction} |")
    lines += ["", "Intervals are pointwise compatibility intervals, not equivalence tests. Near-zero nonnegative dispersion intervals are not treated as significance tests because the simulation checkpoint demonstrated null plug-in inflation.", "", "Bootstrap failures by metric: " + ", ".join(f"{metric}={int(value)}" for metric, value in failed.items()) + "."]
    (output / "PRIMARY_DISPERSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(effects.to_string(index=False))
    print(failures.groupby("metric").failure.apply(lambda x: x.ne("").sum()).to_string())


if __name__ == "__main__":
    main()
