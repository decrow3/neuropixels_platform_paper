#!/usr/bin/env python3
"""Freeze response-blind two-dimensional RF support for the Figure 4 rerun."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import (
    HVA_CATEGORIES, METRICS, V1_CATEGORIES, apply_cell_floor,
    build_response_registry, sha256,
)

MOUSE_RF = ROOT / "data/imports/mousev2_parametric_rf_figure4_v1/rf_unit_fits.csv"
ALLEN_RF = ROOT / "artifacts/figure3/06c_allen_rf_matching/rf_unit_common_support.csv"
OUT = ROOT / "artifacts/figure4_rerun/v2_support"
RADIUS_DEG = 15.0
MIN_LOCAL_CELLS = 5
MIN_LOCAL_RECORDINGS = 2


def points_in_hull(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.unique(np.asarray(reference, dtype=float), axis=0)
    query = np.asarray(query, dtype=float)
    result = np.zeros(len(query), dtype=bool)
    if len(reference) < 3:
        return result
    try:
        hull = ConvexHull(reference)
    except QhullError:
        return result
    finite = np.isfinite(query).all(axis=1)
    result[finite] = np.all(
        query[finite] @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-9,
        axis=1,
    )
    return result


def attach_rf(registry: pd.DataFrame) -> pd.DataFrame:
    mouse = pd.read_csv(MOUSE_RF, dtype={"unit_id": str})
    mouse = mouse.loc[mouse.rf_model_supported].copy()
    mouse["azimuth"] = mouse.supported_rf_center_x_deg + 50.0
    mouse["elevation"] = mouse.supported_rf_center_y_deg + 10.0
    mouse["source"] = "MouseV2"
    allen = pd.read_csv(ALLEN_RF, dtype={"ecephys_unit_id": str})
    allen = allen.rename(columns={
        "ecephys_unit_id": "unit_id", "azimuth_rf": "azimuth", "elevation_rf": "elevation",
    })
    allen["source"] = "Allen"
    coordinates = pd.concat([
        mouse[["source", "unit_id", "azimuth", "elevation"]],
        allen[["source", "unit_id", "azimuth", "elevation"]],
    ], ignore_index=True)
    if coordinates.duplicated(["source", "unit_id"]).any():
        raise ValueError("RF coordinate source contains duplicate unit IDs")
    result = registry.merge(coordinates, on=["source", "unit_id"], how="left", validate="many_to_one")
    result["rf_qualified"] = np.isfinite(result[["azimuth", "elevation"]]).all(axis=1)
    result["support_category"] = np.where(result.population.eq("Central"), "Central", result.category)
    return result


def classify_local_support(
    table: pd.DataFrame,
    categories: list[str],
    *,
    radius_deg: float = RADIUS_DEG,
    min_cells: int = MIN_LOCAL_CELLS,
    min_recordings: int = MIN_LOCAL_RECORDINGS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Require hull membership and local cells/recordings for every category."""
    query = table[["azimuth", "elevation"]].to_numpy(float)
    supported = np.ones(len(table), dtype=bool)
    audits = []
    for category in categories:
        group = table.loc[table.support_category.eq(category)].copy()
        points = group[["azimuth", "elevation"]].to_numpy(float)
        in_hull = points_in_hull(points, query)
        distance = np.sum((query[:, None, :] - points[None, :, :]) ** 2, axis=2)
        local = distance <= radius_deg**2
        local_cells = local.sum(axis=1)
        local_recordings = np.asarray([
            group.loc[mask, "recording_id"].nunique() for mask in local
        ])
        category_supported = in_hull & (local_cells >= min_cells) & (local_recordings >= min_recordings)
        supported &= category_supported
        audits.append(pd.DataFrame({
            "candidate_index": table.index,
            "required_category": category,
            "inside_hull": in_hull,
            "local_cells": local_cells,
            "local_recordings": local_recordings,
            "category_supported": category_supported,
        }))
    return pd.Series(supported, index=table.index), pd.concat(audits, ignore_index=True)


def reference_weights(table: pd.DataFrame) -> pd.Series:
    """Equal population/category/animal/recording/cell RF-reference weights."""
    if set(table.population) != {"V1", "HVA"}:
        raise ValueError("Reference requires both V1 and HVA candidates")
    populations = table.population.map({"V1": .5, "HVA": .5}).astype(float)
    categories = table.groupby("population").category.transform("nunique")
    animals = table.groupby(["population", "category"]).animal_id.transform("nunique")
    recordings = table.groupby(["population", "category", "animal_id"]).recording_id.transform("nunique")
    cells = table.groupby(["population", "category", "animal_id", "recording_id"]).unit_id.transform("nunique")
    weights = populations / (categories * animals * recordings * cells)
    np.testing.assert_allclose(weights.groupby(table.population).sum(), [.5, .5], atol=1e-12)
    return weights


def analyze(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique = registry.drop_duplicates(["metric", "source", "unit_id"]).copy()
    cell_frames, audit_frames, reference_frames, gate_rows = [], [], [], []
    primary_categories = [*V1_CATEGORIES, *HVA_CATEGORIES]
    central_categories = [*primary_categories, "Central"]
    for metric in METRICS:
        qualified = unique.loc[unique.metric.eq(metric) & unique.rf_qualified].copy()
        primary, audit = classify_local_support(qualified, primary_categories)
        inclusive, audit_central = classify_local_support(qualified, central_categories)
        qualified["primary_shared_support"] = primary
        qualified["central_shared_support"] = inclusive
        cell_frames.append(qualified)
        audit["metric"], audit["analysis"] = metric, "primary_four_location"
        audit_central["metric"], audit_central["analysis"] = metric, "central_inclusive"
        audit_frames += [audit, audit_central]
        prefloor_reference = qualified.loc[primary & qualified.population.isin(["V1", "HVA"])].copy()
        reference = apply_cell_floor(prefloor_reference, 5).drop_duplicates(["metric", "source", "unit_id"])
        populations = set(reference.population)
        category_recordings = reference.groupby(["population", "category"]).recording_id.nunique()
        expected_pairs = len(V1_CATEGORIES) + len(HVA_CATEGORIES)
        supported = (
            populations == {"V1", "HVA"}
            and len(category_recordings) == expected_pairs
            and category_recordings.ge(2).all()
        )
        if supported:
            reference["reference_weight"] = reference_weights(reference)
            reference_frames.append(reference)
        gate_rows.append({
            "metric": metric,
            "rf_qualified_cells": len(qualified),
            "primary_support_cells": int(primary.sum()),
            "primary_v1_reference_cells": int((primary & qualified.population.eq("V1")).sum()),
            "primary_hva_reference_cells": int((primary & qualified.population.eq("HVA")).sum()),
            "postfloor_reference_cells": len(reference),
            "minimum_postfloor_category_recordings": int(category_recordings.min()) if len(category_recordings) else 0,
            "central_inclusive_support_cells": int(inclusive.sum()),
            "cross_population_adjustment_supported": supported,
            "gate_reason": "supported" if supported else "post-support population lacks both populations and at least two recordings per named category",
        })
    references = pd.concat(reference_frames, ignore_index=True) if reference_frames else pd.DataFrame()
    return pd.concat(cell_frames, ignore_index=True), pd.concat(audit_frames, ignore_index=True), references, pd.DataFrame(gate_rows)


def analyze_hva_only(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = registry.drop_duplicates(["metric", "source", "unit_id"])
    cells, gates = [], []
    for metric in METRICS:
        local = unique.loc[unique.metric.eq(metric) & unique.population.eq("HVA")].copy()
        supported, _ = classify_local_support(local, list(HVA_CATEGORIES))
        local["hva_shared_support"] = supported
        cells.append(local)
        final = apply_cell_floor(local.loc[supported], 5)
        recordings = final.groupby("category").recording_id.nunique()
        adequate = len(recordings) == len(HVA_CATEGORIES) and recordings.ge(2).all()
        gates.append({
            "metric": metric, "rf_qualified_cells": len(local),
            "shared_support_cells": int(supported.sum()), "postfloor_cells": len(final),
            "minimum_category_recordings": int(recordings.min()) if len(recordings) else 0,
            "hva_only_adjustment_supported": adequate,
        })
    return pd.concat(cells, ignore_index=True), pd.DataFrame(gates)


def population_ladder(registry: pd.DataFrame, support_cells: pd.DataFrame) -> pd.DataFrame:
    flags = support_cells[["metric", "source", "unit_id", "primary_shared_support"]]
    table = registry.merge(flags, on=["metric", "source", "unit_id"], how="left", validate="many_to_one")
    table["primary_shared_support"] = table.primary_shared_support.eq(True)
    rows = []
    for stage, subset in [
        ("1_response_eligible", table),
        ("2_rf_qualified", table.loc[table.rf_qualified]),
        ("2_rf_qualified_floor5", apply_cell_floor(table.loc[table.rf_qualified], 5)),
        ("3_shared_support_prefloor", table.loc[table.primary_shared_support]),
        ("3_shared_support_floor5", apply_cell_floor(table.loc[table.primary_shared_support], 5)),
    ]:
        for (metric, population, category), part in subset.groupby(["metric", "population", "category"], sort=False):
            rows.append({
                "stage": stage, "metric": metric, "population": population, "category": category,
                "cells": part.unit_id.nunique(), "recordings": part.recording_id.nunique(),
                "animals": part.animal_id.nunique(), "draw_rows": len(part),
            })
    return pd.DataFrame(rows)


def render(support: pd.DataFrame, output: Path) -> None:
    colors = {"V1": "#7567a8", "HVA": "#54575b", "Central": "#b07a2a"}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 12), sharex=True, sharey=True)
    for row, metric in enumerate(METRICS):
        local = support.loc[support.metric.eq(metric)]
        for column, population in enumerate(["V1", "HVA"]):
            ax = axes[row, column]
            part = local.loc[local.population.eq(population)]
            ax.scatter(part.azimuth, part.elevation, s=5, alpha=.12, color=colors[population], rasterized=True)
            selected = part.loc[part.primary_shared_support]
            ax.scatter(selected.azimuth, selected.elevation, s=12, alpha=.65, color="#d62728", label="shared support")
            ax.set_title(f"{metric}: {population} ({len(selected)}/{len(part)})")
            ax.set_xlabel("RF azimuth (display deg)")
            ax.set_ylabel("RF elevation (display deg)")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Response-blind local RF support: all named primary categories required", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    response = apply_cell_floor(build_response_registry(), 5)
    registry = attach_rf(response)
    rf_floor = apply_cell_floor(registry.loc[registry.rf_qualified], 5)
    support, audit, reference, gates = analyze(rf_floor)
    hva_support, hva_gates = analyze_hva_only(rf_floor)
    ladder = population_ladder(registry, support)
    support.to_csv(output / "rf_support_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    audit.to_csv(output / "rf_local_support_audit.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    reference.to_csv(output / "rf_reference_distribution.csv", index=False)
    gates.to_csv(output / "rf_support_gates.csv", index=False)
    hva_support.to_csv(output / "hva_rf_support_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    hva_gates.to_csv(output / "hva_rf_support_gates.csv", index=False)
    ladder.to_csv(output / "population_ladder.csv", index=False)
    render(support, output / "rf_support_occupancy")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in [MOUSE_RF, ALLEN_RF]}
    report = [
        "# Figure 4 shared-RF-support checkpoint", "",
        "The support rule was fixed using RF coordinates, category labels, and recording IDs only; response values and category-effect estimates were not inspected.", "",
        "A candidate RF coordinate is supported only when it lies inside every primary category's convex hull and, for every category, has at least five RF-qualified cells from at least two physical recordings within 15°. The 15° radius spans one diagonal neighborhood of the 10° MouseV2 RF stimulus grid. The empirical reference is then formed from supported observed coordinates with equal V1/HVA population weight, equal categories within population, equal animals and recordings within category, and equal cells within recording.", "",
        "| Metric | RF-qualified after floor | Shared support | V1 reference, prefloor | HVA reference, prefloor | Final reference | Min category recordings | Cross-population adjustment |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gates.itertuples(index=False):
        report.append(f"| {row.metric} | {row.rf_qualified_cells:,} | {row.primary_support_cells:,} | {row.primary_v1_reference_cells:,} | {row.primary_hva_reference_cells:,} | {row.postfloor_reference_cells:,} | {row.minimum_postfloor_category_recordings} | {'supported' if row.cross_population_adjustment_supported else 'unsupported'} |")
    report += ["", "No metric passes the final cross-population RF-adjustment gate. TTFS has no empirical coordinate satisfying the local all-category rule. Timescale has supported Allen coordinates but no supported MouseV2 coordinate. F1/F0 has prefloor overlap, but after the common five-cell recording/category floor RL is represented by only one independent recording. Under the prespecified feasibility gate this is descriptive-only, not a population-level inferential contrast. These are support failures, not adjusted null results.", "", "## HVA-only support", "", "The same rule restricted to LM, RL, AL, PM, and AM supports the hierarchy analysis for all three metrics:", "", "| Metric | RF-qualified | Shared support | Postfloor | Minimum category recordings | Gate |", "|---|---:|---:|---:|---:|---|:"]
    for row in hva_gates.itertuples(index=False):
        report.append(f"| {row.metric} | {row.rf_qualified_cells:,} | {row.shared_support_cells:,} | {row.postfloor_cells:,} | {row.minimum_category_recordings} | {'supported' if row.hva_only_adjustment_supported else 'unsupported'} |")
    report += ["", "Adding Central does not rescue any failed cross-population metric and cannot be used to broaden the primary reference. HVA-only RF/hierarchy fits remain conditional on this explicitly different HVA-only support.", "", "Source checksums:"]
    report += [f"- `{path}`: `{digest}`" for path, digest in sources.items()]
    (output / "RF_SUPPORT_CHECKPOINT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(gates.to_string(index=False))
    print("\nHVA-only gates")
    print(hva_gates.to_string(index=False))
    print(output / "RF_SUPPORT_CHECKPOINT.md")


if __name__ == "__main__":
    main()
