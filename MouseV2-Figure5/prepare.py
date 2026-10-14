#!/usr/bin/env python3
"""Validate sources and regenerate every derived input to canonical Figure 5.

The default ``check`` command is read-only. ``regenerate`` writes a parallel
artifact tree (never the canonical inputs) and verifies its files against the
locked hashes in ``source_manifest.json``. Metric-level source tables are bundled
with this folder; their original provenance, retained columns, and hashes are
part of that manifest.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

# Small least-squares fits are substantially slower with many BLAS threads.
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("FIGURE5_BLAS_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = os.environ.get("FIGURE5_BLAS_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "MouseV2-Figure5/source_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "MouseV2-Figure5/work/reproduction"
METRIC_DIRECTORY = None

METRICS = ("TTFS (ms)", "log10 F1/F0", "Response timescale (ms)")
V1_CATEGORIES = ("A", "E", "C", "B")
HVA_CATEGORIES = ("LM", "RL", "AL", "PM", "AM")
ANALYSIS_COLUMNS = (
    "source", "animal_id", "session_id", "physical_probe_id", "recording_id",
    "population", "category", "unit_id", "metric", "draw_id", "value",
)
FINE_TO_COARSE = {
    "VISp": "V1", "VISl": "LM", "VISrl": "RL", "VISal": "AL",
    "VISpm": "PM", "VISam": "AM",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text())


def sources() -> dict[str, dict[str, object]]:
    return {str(item["label"]): item for item in manifest()["sources"]}


def source_path(label: str) -> Path:
    if METRIC_DIRECTORY is not None and label in {
        "mouse_ttfs", "allen_ttfs", "mouse_f1_f0", "allen_f1_f0", "mouse_timescale"
    }:
        return METRIC_DIRECTORY / f"{label}.csv.gz"
    return ROOT / str(sources()[label]["path"])


def check_sources(*, quiet: bool = False) -> bool:
    failures = []
    for item in manifest()["sources"]:
        path = ROOT / str(item["path"])
        if not path.is_file():
            failures.append(f"MISSING  {item['label']}: {item['path']}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(
                f"DRIFT    {item['label']}: {item['path']}\n"
                f"         expected {item['sha256']}\n         actual   {actual}"
            )
        elif not quiet:
            print(f"OK       {item['label']}: {item['path']}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return False
    if not quiet:
        print("All Figure 5 preparation sources match their locked hashes.")
    return True


def download_public() -> None:
    """Download only missing public Allen inputs and verify before installing."""
    downloadable = [item for item in manifest()["sources"] if item.get("url")]
    for item in downloadable:
        target = ROOT / str(item["path"])
        if target.exists():
            actual = sha256(target)
            if actual != item["sha256"]:
                raise RuntimeError(f"Refusing to overwrite drifted source: {target}")
            print(f"present  {target.relative_to(ROOT)}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(str(item["url"])) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            actual = sha256(temporary)
            if actual != item["sha256"]:
                raise RuntimeError(
                    f"Downloaded hash mismatch for {item['label']}: expected "
                    f"{item['sha256']}, got {actual}"
                )
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"downloaded {target.relative_to(ROOT)}")


def _mouse_and_allen_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    mouse = pd.read_csv(source_path("mouse_analysis_units"), dtype={"unit_id": str})
    mouse = mouse[["unit_id", "site_number", "subject_id", "probe", "location"]].copy()
    mouse["session_id"] = mouse.site_number.astype(str)
    mouse["animal_id"] = mouse.subject_id.astype(str)
    mouse["physical_probe_id"] = mouse.probe.str.removeprefix("Probe")
    if mouse.unit_id.duplicated().any():
        raise ValueError("MouseV2 CCF metadata contain duplicate unit IDs")

    allen = pd.read_csv(source_path("allen_units"), low_memory=False)
    allen["area_coarse"] = allen.ecephys_structure_acronym.map(
        lambda value: FINE_TO_COARSE.get(value, value)
    )
    allen = allen.rename(columns={
        "ecephys_unit_id": "unit_id", "specimen_id": "animal_id",
        "ecephys_session_id": "session_id", "ecephys_probe_id": "physical_probe_id",
    })
    for column in ("unit_id", "animal_id", "session_id", "physical_probe_id"):
        allen[column] = allen[column].astype(str)
    allen["common_qc"] = (
        pd.to_numeric(allen.amplitude_cutoff, errors="coerce").lt(.1)
        & pd.to_numeric(allen.presence_ratio, errors="coerce").gt(.8)
        & pd.to_numeric(allen.isi_violations, errors="coerce").lt(.5)
    )
    if allen.unit_id.duplicated().any():
        raise ValueError("Allen metadata contain duplicate unit IDs")
    return mouse, allen


def _finish_registry_rows(
    table: pd.DataFrame, *, source: str, population: str,
    mouse: pd.DataFrame, allen: pd.DataFrame,
) -> pd.DataFrame:
    result = table.copy()
    result["unit_id"] = result.unit_id.astype(str)
    result["session_id"] = result.session_id.astype(str)
    result["source"] = source
    result["population"] = population
    metadata = mouse if source == "MouseV2" else allen
    fields = ["unit_id", "animal_id", "session_id", "physical_probe_id"]
    fields += ["location"] if source == "MouseV2" else ["common_qc"]
    result = result.merge(
        metadata[fields], on=["unit_id", "session_id"], how="left", validate="many_to_one"
    )
    if source == "MouseV2":
        # Historical unit-to-electrode metadata preserve the original anatomical
        # population. The later DANDI draft must not redefine eligibility.
        result = result.loc[result.location.fillna("").str.startswith("VISp")].copy()
    else:
        result = result.loc[result.common_qc].copy()
    if result[["animal_id", "physical_probe_id"]].isna().any().any():
        raise ValueError(f"{source}/{population} rows are missing recording metadata")
    result["recording_id"] = (
        result.source + ":" + result.animal_id + ":" + result.session_id
        + ":" + result.physical_probe_id
    )
    result["draw_id"] = result.get("draw_id", -1)
    result["draw_id"] = result.draw_id.astype(int)
    result["value"] = pd.to_numeric(result.value, errors="coerce")
    if not np.isfinite(result.value).all():
        raise ValueError(f"Nonfinite values survived in {source}/{population}")
    if result.duplicated(["source", "metric", "unit_id", "draw_id"]).any():
        raise ValueError(f"Duplicate unit/draw rows in {source}/{population}")
    return result[list(ANALYSIS_COLUMNS)]


def build_response_registry() -> pd.DataFrame:
    mouse, allen = _mouse_and_allen_metadata()
    frames: list[pd.DataFrame] = []

    table = pd.read_csv(source_path("mouse_ttfs"), dtype={"session_id": str, "unit_id": str})
    table = table.loc[
        table.cohort.eq("MouseV2") & table.selected.astype(bool)
        & pd.to_numeric(table.preferred_0_250_ttfs_ms, errors="coerce").lt(100)
    ].rename(columns={"location": "category", "preferred_0_250_ttfs_ms": "value"})
    table["metric"] = METRICS[0]
    frames.append(_finish_registry_rows(table, source="MouseV2", population="V1", mouse=mouse, allen=allen))

    table = pd.read_csv(source_path("allen_ttfs"), dtype={"session_id": str, "unit_id": str})
    table = table.loc[
        table.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & table.selected_positive_responder_area.astype(bool)
        & pd.to_numeric(table.preferred_0_250_ttfs_ms, errors="coerce").lt(100)
    ].rename(columns={"area_coarse": "category", "preferred_0_250_ttfs_ms": "value"})
    table["metric"] = METRICS[0]
    frames += [
        _finish_registry_rows(table.loc[table.category.eq("V1")], source="Allen", population="Central", mouse=mouse, allen=allen),
        _finish_registry_rows(table.loc[table.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse=mouse, allen=allen),
    ]

    mouse_f1 = pd.read_csv(source_path("mouse_f1_f0"), dtype={"unit_id": str})
    table = mouse_f1.loc[
        mouse_f1.default_qc.astype(bool)
        & pd.to_numeric(mouse_f1.f1_f0_dg_common_support, errors="coerce").gt(0)
    ].rename(columns={"site_number": "session_id", "probe_letter": "category"})
    table["value"] = np.log10(table.f1_f0_dg_common_support.astype(float))
    table["metric"] = METRICS[1]
    frames.append(_finish_registry_rows(table, source="MouseV2", population="V1", mouse=mouse, allen=allen))

    table = pd.read_csv(
        source_path("allen_f1_f0"),
        dtype={"ecephys_session_id": str, "ecephys_unit_id": str},
    )
    table = table.loc[
        table.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & pd.to_numeric(table.f1_f0_dg_harmonized, errors="coerce").gt(0)
    ].rename(columns={
        "ecephys_session_id": "session_id", "ecephys_unit_id": "unit_id",
        "area_coarse": "category",
    })
    table["value"] = np.log10(table.f1_f0_dg_harmonized.astype(float))
    table["metric"] = METRICS[1]
    frames += [
        _finish_registry_rows(table.loc[table.category.eq("V1")], source="Allen", population="Central", mouse=mouse, allen=allen),
        _finish_registry_rows(table.loc[table.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse=mouse, allen=allen),
    ]

    unit_map = mouse_f1[["unit_id", "site_number", "probe_letter"]].drop_duplicates("unit_id")
    table = pd.read_csv(source_path("mouse_timescale"), dtype={"unit_id": str})
    table = table.loc[table.view.eq("mouse_matched_150") & table.valid_timescale.astype(bool)]
    table = table.merge(unit_map, on="unit_id", validate="many_to_one")
    table = table.rename(columns={
        "session_id": "bridge_session_id", "site_number": "session_id",
        "probe_letter": "category", "subsample": "draw_id", "timescale_ms": "value",
    })
    if not table.bridge_session_id.astype(int).eq(table.session_id.astype(int)).all():
        raise ValueError("MouseV2 timescale session metadata disagree")
    table["metric"] = METRICS[2]
    frames.append(_finish_registry_rows(table, source="MouseV2", population="V1", mouse=mouse, allen=allen))

    table = allen.loc[
        allen.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & pd.to_numeric(allen.timescale_ac, errors="coerce").between(1, 300)
        & pd.to_numeric(allen.spike_count_ac, errors="coerce").gt(50)
        & pd.to_numeric(allen.err_ac, errors="coerce").lt(20)
    ].rename(columns={"area_coarse": "category", "timescale_ac": "value"})
    table["metric"] = METRICS[2]
    table = table[["session_id", "unit_id", "category", "value", "metric"]]
    frames += [
        _finish_registry_rows(table.loc[table.category.eq("V1")], source="Allen", population="Central", mouse=mouse, allen=allen),
        _finish_registry_rows(table.loc[table.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse=mouse, allen=allen),
    ]
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["metric", "source", "session_id", "category", "unit_id", "draw_id"]
    ).reset_index(drop=True)


def apply_cell_floor(table: pd.DataFrame, floor: int = 5) -> pd.DataFrame:
    keys = ["metric", "recording_id", "category", "draw_id"]
    count = table.groupby(keys).unit_id.transform("nunique")
    return table.loc[count.ge(floor)].copy()


def category_recording_weights(table: pd.DataFrame) -> pd.Series:
    if table.duplicated(["category", "unit_id"]).any():
        raise ValueError("Weights require at most one row per category/unit")
    k = table.category.nunique()
    animals = table.groupby("category").animal_id.transform("nunique")
    recordings = table.groupby(["category", "animal_id"]).recording_id.transform("nunique")
    cells = table.groupby(["category", "animal_id", "recording_id"]).unit_id.transform("nunique")
    weights = 1.0 / (k * animals * recordings * cells)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    return weights


@dataclass(frozen=True)
class DispersionFit:
    category_means: dict[str, float]
    dispersion: float
    root_dispersion: float
    total_variance: float
    fraction_percent: float
    rank: int
    parameters: int
    condition_number: float
    n_cells: int
    n_recordings: int
    n_animals: int


def fit_standardized_dispersion(table: pd.DataFrame) -> DispersionFit:
    if table.duplicated("unit_id").any():
        raise ValueError("A point fit must contain one outcome per unit")
    categories = sorted(table.category.unique())
    category_x = np.column_stack([table.category.eq(value).to_numpy(float) for value in categories])
    sessions = sorted(table.session_id.unique())
    nuisance = (
        np.column_stack([table.session_id.eq(value).to_numpy(float) for value in sessions[1:]])
        if len(sessions) > 1 else np.empty((len(table), 0))
    )
    x = np.column_stack([category_x, nuisance])
    weights = category_recording_weights(table).to_numpy()
    root_w = np.sqrt(weights)
    beta, _, rank, singular = np.linalg.lstsq(
        x * root_w[:, None], table.value.to_numpy(float) * root_w, rcond=1e-10
    )
    if rank != x.shape[1]:
        raise np.linalg.LinAlgError(f"Rank-deficient design ({rank}/{x.shape[1]})")
    standardized = beta[:len(categories)]
    dispersion = float(np.mean((standardized - standardized.mean()) ** 2))
    values = table.value.to_numpy(float)
    raw_mean = float(np.sum(weights * values))
    total_variance = float(np.sum(weights * (values - raw_mean) ** 2))
    return DispersionFit(
        category_means=dict(zip(categories, standardized)), dispersion=dispersion,
        root_dispersion=float(np.sqrt(dispersion)), total_variance=total_variance,
        fraction_percent=100 * dispersion / total_variance, rank=int(rank),
        parameters=x.shape[1], condition_number=float(singular[0] / singular[-1]),
        n_cells=len(table), n_recordings=table.recording_id.nunique(),
        n_animals=table.animal_id.nunique(),
    )


def point_table(registry: pd.DataFrame, metric: str, population: str, draw: int | None) -> pd.DataFrame:
    local = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)].copy()
    if metric == METRICS[2]:
        selected = int(draw) if population == "V1" else -1
        local = apply_cell_floor(local.loc[local.draw_id.eq(selected)].copy(), 5)
    if local.duplicated("unit_id").any():
        raise ValueError(f"Point population duplicates cells: {metric}/{population}/{draw}")
    return local


def _fit_row(fit: DispersionFit, metric: str, population: str, draw: int) -> dict[str, object]:
    return {
        "metric": metric, "population": population, "draw": draw,
        "dispersion": fit.dispersion, "root_dispersion": fit.root_dispersion,
        "total_variance": fit.total_variance, "fraction_percent": fit.fraction_percent,
        "n_cells": fit.n_cells, "n_recordings": fit.n_recordings,
        "n_animals": fit.n_animals, "rank": fit.rank, "parameters": fit.parameters,
        "condition_number": fit.condition_number,
        "category_means": ";".join(f"{key}={value:.12g}" for key, value in fit.category_means.items()),
    }


def primary_points(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows, effect_rows = [], []
    for metric in METRICS:
        draws = range(10) if metric == METRICS[2] else [None]
        fits: dict[str, list[DispersionFit]] = {"V1": [], "HVA": []}
        for draw in draws:
            for population in ("V1", "HVA"):
                fit = fit_standardized_dispersion(point_table(registry, metric, population, draw))
                fits[population].append(fit)
                fit_rows.append(_fit_row(fit, metric, population, -1 if draw is None else int(draw)))
        v1 = float(np.mean([fit.dispersion for fit in fits["V1"]]))
        hva = float(np.mean([fit.dispersion for fit in fits["HVA"]]))
        for population in ("V1", "HVA"):
            local = fits[population]
            effect_rows.append({
                "metric": metric, "effect": population.lower(),
                "estimate": float(np.mean([fit.dispersion for fit in local])),
                "root_response_units": float(np.mean([fit.root_dispersion for fit in local])),
                "fraction_percent": float(np.mean([fit.fraction_percent for fit in local])),
                "total_variance": float(np.mean([fit.total_variance for fit in local])),
            })
        effect_rows.append({"metric": metric, "effect": "hva_minus_v1", "estimate": hva - v1})
    return pd.DataFrame(fit_rows), pd.DataFrame(effect_rows)


def _resampling_plan(source: pd.DataFrame, *, per_session: bool):
    """Cache immutable row indices; preserve the original random stream/order."""
    source = source.reset_index(drop=True)
    animals = np.asarray(sorted(source.animal_id.unique()), dtype=object)
    blocks = {}
    draw_ids = source.draw_id.to_numpy() if "draw_id" in source else None
    for animal, animal_rows in source.groupby("animal_id", sort=False).indices.items():
        local = source.iloc[animal_rows]
        sessions = ([animal_rows[indices] for indices in local.groupby("session_id", sort=False).indices.values()]
                    if per_session else [animal_rows])
        blocks[animal] = [
            {"all": rows, **({draw: rows[draw_ids[rows] == draw] for draw in range(-1, 10)}
                            if per_session else {})}
            for rows in sessions
        ]
    return source, animals, blocks


def _resample_plan(plan, rng, *, timescale=False, population=None):
    source, animals, blocks = plan
    selected_rows, prefixes = [], []
    for occurrence, animal in enumerate(rng.choice(animals, size=len(animals), replace=True)):
        for session in blocks[animal]:
            draw = (int(rng.integers(0, 10)) if population == "V1" else -1) if timescale else "all"
            rows = session[draw]
            selected_rows.append(rows)
            prefixes.append(np.full(len(rows), f"bootstrap_{occurrence}:{animal}:", dtype=object))
    table = source.iloc[np.concatenate(selected_rows)].copy().reset_index(drop=True)
    prefix = pd.Series(np.concatenate(prefixes), index=table.index)
    for column in ("animal_id", "session_id", "recording_id", "unit_id"):
        table[column] = prefix + table[column].astype(str)
    return table


def _resample_population(
    registry: pd.DataFrame, metric: str, population: str, rng: np.random.Generator,
    *, prepared=None,
) -> pd.DataFrame:
    if prepared is None:
        source = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)]
        prepared = _resampling_plan(source, per_session=True)
    return apply_cell_floor(_resample_plan(prepared, rng, timescale=metric == METRICS[2],
                                          population=population), 5)


def primary_bootstrap(registry: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    rows = []
    for metric in METRICS:
        plans = {population: _resampling_plan(
            registry.loc[registry.metric.eq(metric) & registry.population.eq(population)],
            per_session=True) for population in ("V1", "HVA")}
        for repetition in range(repetitions):
            if repetition % 500 == 0:
                print(f"primary {metric}: {repetition}/{repetitions}", flush=True)
            fits = {
                population: fit_standardized_dispersion(
                    _resample_population(registry, metric, population, rng, prepared=plans[population])
                ) for population in ("V1", "HVA")
            }
            rows += [
                {"metric": metric, "replicate": repetition, "effect": "v1", "estimate": fits["V1"].dispersion},
                {"metric": metric, "replicate": repetition, "effect": "hva", "estimate": fits["HVA"].dispersion},
                {"metric": metric, "replicate": repetition, "effect": "hva_minus_v1", "estimate": fits["HVA"].dispersion - fits["V1"].dispersion},
                {"metric": metric, "replicate": repetition, "effect": "v1_fraction_percent", "estimate": fits["V1"].fraction_percent},
                {"metric": metric, "replicate": repetition, "effect": "hva_fraction_percent", "estimate": fits["HVA"].fraction_percent},
                {"metric": metric, "replicate": repetition, "effect": "fraction_difference_pp", "estimate": fits["HVA"].fraction_percent - fits["V1"].fraction_percent},
            ]
        print(f"bootstrapped primary dispersion: {metric}", flush=True)
    return pd.DataFrame(rows)


def add_primary_intervals(effects: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    result = effects.copy()
    result["interval_low"] = np.nan
    result["interval_high"] = np.nan
    for index, row in result.iterrows():
        values = draws.loc[
            draws.metric.eq(row.metric) & draws.effect.eq(row.effect), "estimate"
        ]
        result.loc[index, ["interval_low", "interval_high"]] = np.quantile(values, [.025, .975])
    return result


def _resample_animals(table: pd.DataFrame, rng: np.random.Generator, *, prepared=None) -> pd.DataFrame:
    return _resample_plan(_resampling_plan(table, per_session=False) if prepared is None else prepared, rng)


def _control_population(registry: pd.DataFrame, metric: str) -> pd.DataFrame:
    local = registry.loc[
        registry.metric.eq(metric) & registry.population.isin(["Central", "HVA"])
    ].copy()
    sessions = set(local.loc[local.population.eq("Central"), "session_id"]) & set(
        local.loc[local.population.eq("HVA"), "session_id"]
    )
    local = local.loc[local.session_id.isin(sessions)].copy()
    local["control_group"] = np.where(local.population.eq("Central"), "V1", "HVA")
    paired = local.groupby("session_id").control_group.nunique()
    return local.loc[local.session_id.isin(paired.loc[paired.eq(2)].index)].copy()


def _control_weights(table: pd.DataFrame) -> pd.Series:
    weights = pd.Series(0.0, index=table.index)
    for mask, is_hva in [(table.control_group.eq("V1"), False), (table.control_group.eq("HVA"), True)]:
        part = table.loc[mask]
        area_factor = (
            part.category.map(part.groupby("category").size()).astype(float) * 0
            + part.category.nunique() if is_hva else pd.Series(1.0, index=part.index)
        )
        stratum = "category" if is_hva else "control_group"
        animals = part.groupby(stratum).animal_id.transform("nunique")
        recordings = part.groupby([stratum, "animal_id"]).recording_id.transform("nunique")
        cells = part.groupby([stratum, "animal_id", "recording_id"]).unit_id.transform("nunique")
        weights.loc[part.index] = .5 / (area_factor * animals * recordings * cells)
    return weights


def _fit_control(table: pd.DataFrame) -> dict[str, float | int]:
    group_x = np.column_stack([table.control_group.eq(group).to_numpy(float) for group in ("V1", "HVA")])
    sessions = sorted(table.session_id.unique())
    nuisance = np.column_stack([table.session_id.eq(value).to_numpy(float) for value in sessions[1:]])
    x = np.column_stack([group_x, nuisance])
    root = np.sqrt(_control_weights(table).to_numpy())
    beta, _, rank, singular = np.linalg.lstsq(x * root[:, None], table.value.to_numpy(float) * root, rcond=1e-10)
    if rank != x.shape[1]:
        raise np.linalg.LinAlgError("Matched-control design is rank deficient")
    difference = float(beta[1] - beta[0])
    return {
        "v1_mean": float(beta[0]), "pooled_hva_mean": float(beta[1]),
        "hva_minus_v1": difference, "two_group_dispersion": difference**2 / 4,
        "rank": int(rank), "parameters": x.shape[1],
        "condition_number": float(singular[0] / singular[-1]),
        "n_cells": len(table), "n_sessions": table.session_id.nunique(),
    }


def matched_control(registry: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    points, draws = [], []
    for metric in METRICS:
        table = _control_population(registry, metric)
        point = _fit_control(table)
        point["metric"] = metric
        points.append(point)
        plan = _resampling_plan(table, per_session=False)
        for repetition in range(repetitions):
            if repetition % 500 == 0:
                print(f"matched control {metric}: {repetition}/{repetitions}", flush=True)
            fit = _fit_control(_resample_animals(table, rng, prepared=plan))
            draws.append({"metric": metric, "replicate": repetition, **fit})
        print(f"bootstrapped matched control: {metric}", flush=True)
    points, draws = pd.DataFrame(points), pd.DataFrame(draws)
    for column in ("hva_minus_v1", "two_group_dispersion"):
        points[column + "_low"] = points.metric.map(
            lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.025)
        )
        points[column + "_high"] = points.metric.map(
            lambda metric: draws.loc[draws.metric.eq(metric), column].quantile(.975)
        )
    return points


def moments(table: pd.DataFrame, categories: tuple[str, ...]) -> np.ndarray:
    return np.array([
        [len(values := table.loc[table.category.eq(category), "value"].to_numpy()),
         values.sum(), values @ values]
        for category in categories
    ], float)


def omega_effect(values: np.ndarray) -> tuple[float, float, float]:
    n, sums, squares = values.T
    if (n == 0).any():
        return np.nan, np.nan, np.nan
    total = n.sum()
    sst = squares.sum() - sums.sum() ** 2 / total
    ssb = (sums * sums / n).sum() - sums.sum() ** 2 / total
    mse = (sst - ssb) / (total - len(n))
    return (ssb - (len(n) - 1) * mse) / (sst + mse), ssb / sst, np.sqrt(sst / total)


def cell_omega(registry: pd.DataFrame, repetitions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260921)
    summaries, bootstrap_rows = [], []
    for metric in METRICS:
        for population, categories in (("V1", V1_CATEGORIES), ("HVA", HVA_CATEGORIES)):
            n_draws = 10 if metric == METRICS[2] and population == "V1" else 1
            tables = [point_table(registry, metric, population, draw) for draw in range(n_draws)]
            point_values = [omega_effect(moments(table, categories))[0] for table in tables]
            source = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)]
            animals = sorted(source.animal_id.unique())
            blocks = []
            for animal in animals:
                sessions = []
                for session in source.loc[source.animal_id.eq(animal), "session_id"].unique():
                    sessions.append(np.stack([
                        moments(table.loc[table.animal_id.eq(animal) & table.session_id.eq(session)], categories)
                        for table in tables
                    ]))
                blocks.append(sessions)
            values = []
            for replicate in range(repetitions):
                aggregate = np.zeros((len(categories), 3))
                for animal_index in rng.integers(0, len(animals), len(animals)):
                    for session in blocks[animal_index]:
                        aggregate += session[rng.integers(n_draws)]
                value = omega_effect(aggregate)[0]
                values.append(value)
                bootstrap_rows.append({
                    "metric": metric, "population": population,
                    "replicate": replicate, "omega_squared": value,
                })
            valid = np.asarray(values)[np.isfinite(values)]
            low, high = np.quantile(valid, [.025, .975])
            summaries.append({
                "metric": metric, "population": population,
                "omega_squared": np.mean(point_values), "low": low, "high": high,
                "n_animals": len(animals), "valid_bootstraps": len(valid),
                "failed_bootstraps": repetitions - len(valid),
            })
    return pd.DataFrame(summaries), pd.DataFrame(bootstrap_rows)


def _omega_vector(values: np.ndarray) -> np.ndarray:
    n, sums, squares = np.moveaxis(values, -1, 0)
    total = n.sum(-1)
    k = n.shape[-1]
    grand = sums.sum(-1) ** 2 / total
    sst = squares.sum(-1) - grand
    ssb = (sums * sums / n).sum(-1) - grand
    mse = (sst - ssb) / (total - k)
    return (ssb - (k - 1) * mse) / (sst + mse)


def location_variance(
    registry: pd.DataFrame, estimates: pd.DataFrame,
    bootstraps: pd.DataFrame, repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260929)
    rows = []
    for metric in METRICS:
        for population, categories in (("V1", V1_CATEGORIES), ("HVA", HVA_CATEGORIES)):
            n_draws = 10 if metric == METRICS[2] and population == "V1" else 1
            tables = [point_table(registry, metric, population, draw) for draw in range(n_draws)]
            observed = np.mean([omega_effect(moments(table, categories))[0] for table in tables])
            saved = estimates.loc[
                estimates.metric.eq(metric) & estimates.population.eq(population)
            ].iloc[0]
            null = np.zeros((repetitions, n_draws, len(categories), 3))
            sessions = fixed = 0
            all_rows = pd.concat(tables)
            for (animal, session), _ in all_rows.groupby(["animal_id", "session_id"], sort=True):
                sessions += 1
                local = [table.loc[table.animal_id.eq(animal) & table.session_id.eq(session)] for table in tables]
                block = np.stack([moments(table, categories) for table in local])
                available = np.flatnonzero(block[:, :, 0].sum(0) > 0)
                fixed += len(available) == 1
                permutation = np.argsort(rng.random((repetitions, len(available))), axis=1)
                for draw in range(n_draws):
                    null[:, draw, available, :] += block[draw, available][permutation]
            values = _omega_vector(null).mean(1)
            count = int((values >= observed).sum())
            rows.append({
                "metric": metric, "population": population, "omega_squared": observed,
                "low": saved.low, "high": saved.high,
                "shuffle_median": np.median(values), "shuffle_low": np.quantile(values, .025),
                "shuffle_high": np.quantile(values, .975),
                "permutation_p_greater": (1 + count) / (repetitions + 1),
                "exceedance_count": count, "n_shuffles": repetitions,
                "n_sessions": sessions, "fixed_single_category_sessions": fixed,
                "n_point_draws": n_draws,
            })
        paired = bootstraps.loc[bootstraps.metric.eq(metric)].pivot(
            index="replicate", columns="population", values="omega_squared"
        )
        difference = paired.HVA - paired.V1
        subset = estimates.loc[estimates.metric.eq(metric)].set_index("population")
        rows.append({
            "metric": metric, "population": "HVA_minus_V1",
            "omega_squared": subset.loc["HVA", "omega_squared"] - subset.loc["V1", "omega_squared"],
            "low": difference.quantile(.025), "high": difference.quantile(.975),
        })
    return pd.DataFrame(rows)


def display_counts(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        draw = 0 if metric == METRICS[2] else -1
        hva_order = (("LM", "PM", "RL", "AL", "AM") if metric == METRICS[2]
                     else ("LM", "PM", "RL", "AM", "AL"))
        for population, panel, order in (
            ("V1", "C", ("A", "B", "C", "E")),
            ("HVA", "D", hva_order),
        ):
            table = point_table(registry, metric, population, draw)
            for category in order:
                rows.append({
                    "figure": "Figure4_CDE_rerun", "panel": panel, "metric": metric,
                    "population": population, "category": category,
                    "display_draw": draw if population == "V1" else -1,
                    "n_cells": table.loc[table.category.eq(category), "unit_id"].nunique(),
                })
        central = registry.loc[
            registry.metric.eq(metric) & registry.population.eq("Central") & registry.draw_id.eq(-1)
        ]
        v1 = point_table(registry, metric, "V1", draw)
        for category, table in [("Central", central)] + [(value, v1) for value in ("A", "B", "C", "E")]:
            count = (table.unit_id.nunique() if category == "Central" else
                     table.loc[table.category.eq(category), "unit_id"].nunique())
            rows.append({
                "figure": "Figure4_Central_inclusive_companion", "panel": "Central",
                "metric": metric, "population": "Central-inclusive V1",
                "category": category, "display_draw": -1 if category == "Central" else draw,
                "n_cells": count,
            })
    return pd.DataFrame(rows)


def _write_csv(table: pd.DataFrame, root: Path, relative: str, *, gzip_output: bool = False) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if gzip_output else None
    table.to_csv(path, index=False, compression=compression)
    return path


def verify_outputs(output_root: Path, *, full: bool) -> bool:
    expected = manifest()["regenerated_outputs"]
    failures = []
    for relative, digest in expected.items():
        path = output_root / relative
        if not path.is_file():
            if full or not relative.endswith("response_registry_floor5.csv.gz"):
                failures.append(f"MISSING  {relative}")
            continue
        actual = sha256(path)
        if actual == digest:
            print(f"MATCH    {relative}")
            continue
        canonical = ROOT / relative
        if not canonical.is_file() or sha256(canonical) != digest:
            failures.append(f"REFERENCE DRIFT {relative}")
            continue
        try:
            expected_table = pd.read_csv(canonical)
            actual_table = pd.read_csv(path)
            pd.testing.assert_frame_equal(
                actual_table, expected_table, check_exact=False,
                rtol=1e-10, atol=1e-12, check_dtype=False,
            )
        except Exception as exc:
            failures.append(
                f"DRIFT    {relative}\n         expected {digest}\n"
                f"         actual   {actual}\n         comparison: {exc}"
            )
        else:
            print(f"EQUIV    {relative} (numerically equal; platform-level byte drift)")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return False
    return True


def regenerate(output_root: Path, *, repetitions: int, derived_only: bool) -> None:
    if output_root.resolve() == ROOT.resolve():
        raise ValueError("Refusing to overwrite the canonical repository tree")
    output_root.mkdir(parents=True, exist_ok=True)
    if derived_only:
        registry = pd.read_csv(
            ROOT / "MouseV2-Figure5/data/response_registry_floor5.csv.gz",
            dtype={"unit_id": str, "session_id": str},
        )
    else:
        if not check_sources(quiet=True):
            raise RuntimeError("Source validation failed; regeneration was not started")
        registry = apply_cell_floor(build_response_registry(), 5)
        _write_csv(
            registry, output_root,
            "MouseV2-Figure5/data/response_registry_floor5.csv.gz", gzip_output=True,
        )

    point_fits, effects = primary_points(registry)
    bootstrap = primary_bootstrap(registry, repetitions)
    effects = add_primary_intervals(effects, bootstrap)
    _write_csv(point_fits, output_root, "MouseV2-Figure5/data/draw_specific_point_fits.csv")
    _write_csv(effects, output_root, "MouseV2-Figure5/data/primary_effects.csv")
    _write_csv(bootstrap, output_root, "MouseV2-Figure5/data/bootstrap_effects.csv.gz", gzip_output=True)
    _write_csv(matched_control(registry, repetitions), output_root, "MouseV2-Figure5/data/matched_control.csv")
    _write_csv(display_counts(registry), output_root, "MouseV2-Figure5/data/display_counts.csv")
    omega_estimates, omega_bootstraps = cell_omega(registry, repetitions)
    _write_csv(
        location_variance(registry, omega_estimates, omega_bootstraps, repetitions),
        output_root, "MouseV2-Figure5/data/estimates_and_tests.csv",
    )
    if repetitions == 5000:
        if not verify_outputs(output_root, full=not derived_only):
            raise RuntimeError("Regenerated files do not match the canonical hashes")
    else:
        print("Hash verification skipped because --replicates is not 5000.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="verify every declared source hash")
    subparsers.add_parser("download-public", help="download missing public Allen references")
    regenerate_parser = subparsers.add_parser("regenerate", help="rebuild derived Figure 5 input tables")
    regenerate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    regenerate_parser.add_argument("--metrics-dir", type=Path, help="Complete, verified raw extraction output; otherwise use bundled metric inputs")
    regenerate_parser.add_argument("--replicates", type=int, default=5000)
    regenerate_parser.add_argument(
        "--derived-only", action="store_true",
        help="start from the committed locked registry; useful for verifying downstream inference",
    )
    verify_parser = subparsers.add_parser("verify", help="verify a previously regenerated artifact tree")
    verify_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify_parser.add_argument("--derived-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global METRIC_DIRECTORY
    args = parse_args()
    if getattr(args, 'metrics_dir', None):
        from extract import verify_metrics
        METRIC_DIRECTORY = args.metrics_dir.resolve()
        verify_metrics(METRIC_DIRECTORY, require_complete=True)
    if args.command == "check":
        if not check_sources():
            raise SystemExit(1)
    elif args.command == "download-public":
        download_public()
    elif args.command == "regenerate":
        regenerate(args.output_root.resolve(), repetitions=args.replicates, derived_only=args.derived_only)
    elif args.command == "verify":
        if not verify_outputs(args.output_root.resolve(), full=not args.derived_only):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
