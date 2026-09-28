#!/usr/bin/env python3
"""Observed-design simulation and bootstrap validation for Figure 4 rerun models."""

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
    METRICS, apply_cell_floor, build_response_registry, fit_standardized_dispersion,
)

OUT = ROOT / "artifacts/figure4_rerun/v3_model_validation"


def representative_design(registry: pd.DataFrame, metric: str, population: str) -> pd.DataFrame:
    local = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)].copy()
    if metric == METRICS[2]:
        draw = 0 if population == "V1" else -1
        local = local.loc[local.draw_id.eq(draw)].copy()
        local = apply_cell_floor(local, 5)
    if local.empty:
        raise ValueError(f"Empty representative design for {metric}/{population}")
    if local.duplicated("unit_id").any():
        raise ValueError("Simulation design must have one row per unit")
    return local


def resample_animals(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample highest independent units and relabel repeated occurrences."""
    animals = np.asarray(sorted(table.animal_id.unique()), dtype=object)
    selected = rng.choice(animals, size=len(animals), replace=True)
    blocks = []
    for occurrence, animal in enumerate(selected):
        block = table.loc[table.animal_id.eq(animal)].copy()
        prefix = f"bootstrap_{occurrence}:{animal}:"
        block["animal_id"] = prefix + block.animal_id.astype(str)
        block["session_id"] = prefix + block.session_id.astype(str)
        block["recording_id"] = prefix + block.recording_id.astype(str)
        block["unit_id"] = prefix + block.unit_id.astype(str)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def category_signal(categories: list[str], root_dispersion: float) -> dict[str, float]:
    values = np.linspace(-1, 1, len(categories))
    values -= values.mean()
    values *= root_dispersion / np.sqrt(np.mean(values**2))
    return dict(zip(categories, values))


def simulate_outcomes(
    design: pd.DataFrame,
    rng: np.random.Generator,
    category_means: dict[str, float],
) -> pd.DataFrame:
    result = design.copy()
    session_shift = {value: rng.normal(0, .5) for value in result.session_id.unique()}
    recording_shift = {value: rng.normal(0, .35) for value in result.recording_id.unique()}
    result["value"] = (
        result.category.map(category_means).to_numpy(float)
        + result.session_id.map(session_shift).to_numpy(float)
        + result.recording_id.map(recording_shift).to_numpy(float)
        + rng.normal(0, 1, len(result))
    )
    return result


def validate_design(
    design: pd.DataFrame,
    *,
    simulations: int,
    bootstraps: int,
    rng: np.random.Generator,
    metric: str,
    population: str,
) -> pd.DataFrame:
    rows = []
    categories = sorted(design.category.unique())
    for condition, root_dispersion in [("null", 0.0), ("known", .5)]:
        means = category_signal(categories, root_dispersion)
        truth = root_dispersion**2
        for simulation in range(simulations):
            sample = simulate_outcomes(design, rng, means)
            point = fit_standardized_dispersion(sample)
            draws = []
            failures = 0
            for _ in range(bootstraps):
                try:
                    resampled = resample_animals(sample, rng)
                    if set(resampled.category) != set(categories):
                        raise np.linalg.LinAlgError("lost category")
                    draws.append(fit_standardized_dispersion(resampled).dispersion)
                except (np.linalg.LinAlgError, ValueError):
                    failures += 1
            low, high = (np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan))
            rows.append({
                "metric": metric, "population": population, "condition": condition,
                "simulation": simulation, "true_dispersion": truth,
                "estimated_dispersion": point.dispersion,
                "bootstrap_low": low, "bootstrap_high": high,
                "interval_covers_truth": bool(low <= truth <= high) if np.isfinite(low) else False,
                "bootstrap_failures": failures, "bootstrap_replicates": bootstraps,
                "n_cells": len(design), "n_animals": design.animal_id.nunique(),
                "n_recordings": design.recording_id.nunique(), "n_categories": len(categories),
                "rank": point.rank, "parameters": point.parameters,
                "condition_number": point.condition_number,
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["metric", "population", "condition"], as_index=False).agg(
        true_dispersion=("true_dispersion", "first"),
        mean_estimate=("estimated_dispersion", "mean"),
        median_estimate=("estimated_dispersion", "median"),
        mean_bias=("estimated_dispersion", lambda x: x.mean() - results.loc[x.index, "true_dispersion"].iloc[0]),
        interval_coverage=("interval_covers_truth", "mean"),
        bootstrap_failure_fraction=("bootstrap_failures", lambda x: x.sum() / results.loc[x.index, "bootstrap_replicates"].sum()),
        maximum_condition_number=("condition_number", "max"),
        simulations=("simulation", "size"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--metrics", nargs="*", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--populations", nargs="*", choices=["V1", "HVA"], default=["V1", "HVA"])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = apply_cell_floor(build_response_registry(), 5)
    rng = np.random.default_rng(args.seed)
    frames = []
    for metric in args.metrics:
        for population in args.populations:
            design = representative_design(registry, metric, population)
            frames.append(validate_design(
                design, simulations=args.simulations, bootstraps=args.bootstraps,
                rng=rng, metric=metric, population=population,
            ))
            print(f"validated {metric}/{population}", flush=True)
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(output / "simulation_results.csv", index=False)
    summary.to_csv(output / "simulation_summary.csv", index=False)
    lines = [
        "# Figure 4 observed-design model validation", "",
        f"Synthetic outcomes only; {args.simulations} simulations per condition and {args.bootstraps} whole-animal bootstrap replicates per simulation (seed {args.seed}). The observed cluster sizes and category/session missingness are retained. Each outcome contains session shifts, recording shifts, and cell noise.", "",
        "The known-effect condition has true category dispersion 0.25 in squared standardized response units. The null condition has zero category dispersion.", "",
        "| Metric | Population | Condition | Truth | Mean estimate | Bias | 95% interval coverage | Bootstrap failures |", "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.population} | {row.condition} | {row.true_dispersion:.3f} | {row.mean_estimate:.3f} | {row.mean_bias:+.3f} | {row.interval_coverage:.1%} | {row.bootstrap_failure_fraction:.2%} |")
    lines += ["", "Null plug-in dispersion is necessarily nonnegative and is inflated by category-mean estimation noise. Its ordinary percentile interval therefore is not interpreted as a valid nonzero-effect test. The known-effect condition evaluates recovery and interval behavior away from the boundary. Bootstrap category loss and rank failures are reported rather than silently dropped."]
    (output / "MODEL_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
