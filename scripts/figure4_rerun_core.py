"""Core data and estimand contracts for the prespecified Figure 4 rerun.

This module deliberately has no plotting side effects.  It reconstructs metric
eligibility from the accepted source tables, creates the recording registry,
implements the declared category/animal/recording/cell weighting target, and
provides the fixed standardized-category-dispersion estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from common.figure3_mousev2 import load_allen_units, load_config, load_mousev2_units
from scripts.figure3_robust_spread_comparison import (
    ALLEN_HARMONIZED_F1_F0,
    MOUSE_CANONICAL_GRATING_METRICS,
    MOUSE_HARMONIZED_F1_F0,
    MOUSE_TIMESCALE_TRIAL_BRIDGE,
)
from scripts.figure3_full_cell_multilevel_model import MOUSE_TTFS_UNITS, ALLEN_TTFS_AUDIT
from scripts.build_figure4_variant_review import MOUSEV2_UNIT_CCF_LOCATIONS


METRICS = ("TTFS (ms)", "log10 F1/F0", "Response timescale (ms)")
V1_CATEGORIES = ("A", "E", "C", "B")
HVA_CATEGORIES = ("LM", "RL", "AL", "PM", "AM")
HIERARCHY_SCORES = {"LM": -0.093, "RL": -0.059, "AL": 0.152, "PM": 0.327, "AM": 0.441}
ANALYSIS_COLUMNS = (
    "source", "animal_id", "session_id", "physical_probe_id", "recording_id",
    "population", "category", "unit_id", "metric", "draw_id", "value",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "mouse_ttfs": Path(MOUSE_TTFS_UNITS),
        "allen_ttfs": Path(ALLEN_TTFS_AUDIT),
        "mouse_f1_f0": Path(MOUSE_HARMONIZED_F1_F0),
        "allen_f1_f0": Path(ALLEN_HARMONIZED_F1_F0),
        "mouse_timescale": Path(MOUSE_TIMESCALE_TRIAL_BRIDGE),
        "allen_units": ROOT / "data/unit_table.csv",
        "mouse_ccf_units": Path(MOUSEV2_UNIT_CCF_LOCATIONS),
        "mouse_config": ROOT / "config/figure3_mousev2.json",
    }


def source_manifest() -> list[dict[str, object]]:
    rows = []
    for label, path in source_paths().items():
        path = path.resolve()
        rows.append({
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def _metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    session_animals = {
        str(item["site_number"]): str(item["subject_id"])
        for item in config["sessions"]
    }
    locations = pd.read_csv(MOUSEV2_UNIT_CCF_LOCATIONS, dtype={"unit_id": str})
    locations = locations[["unit_id", "site_number", "subject_id", "probe", "location"]].copy()
    locations["session_id"] = locations.site_number.astype(str)
    locations["animal_id"] = locations.subject_id.astype(str)
    locations["physical_probe_id"] = locations.probe.str.removeprefix("Probe")
    if locations.unit_id.duplicated().any():
        raise ValueError("MouseV2 CCF registry contains duplicate unit IDs")
    if not locations.set_index("session_id").animal_id.eq(pd.Series(session_animals)).dropna().all():
        raise ValueError("MouseV2 animal/session mapping disagrees with the configuration")

    allen = load_allen_units()
    allen_meta = allen[[
        "ecephys_unit_id", "specimen_id", "ecephys_session_id", "ecephys_probe_id",
        "area_coarse", "amplitude_cutoff", "presence_ratio", "isi_violations",
    ]].copy()
    allen_meta = allen_meta.rename(columns={
        "ecephys_unit_id": "unit_id", "specimen_id": "animal_id",
        "ecephys_session_id": "session_id", "ecephys_probe_id": "physical_probe_id",
    })
    for column in ("unit_id", "animal_id", "session_id", "physical_probe_id"):
        allen_meta[column] = allen_meta[column].astype(str)
    allen_meta["common_qc"] = (
        pd.to_numeric(allen_meta.amplitude_cutoff, errors="coerce").lt(.1)
        & pd.to_numeric(allen_meta.presence_ratio, errors="coerce").gt(.8)
        & pd.to_numeric(allen_meta.isi_violations, errors="coerce").lt(.5)
    )
    if allen_meta.unit_id.duplicated().any():
        raise ValueError("Allen registry contains duplicate unit IDs")
    return locations, allen_meta


def _finish(
    table: pd.DataFrame,
    *,
    source: str,
    population: str,
    mouse_meta: pd.DataFrame,
    allen_meta: pd.DataFrame,
) -> pd.DataFrame:
    result = table.copy()
    result["unit_id"] = result.unit_id.astype(str)
    result["session_id"] = result.session_id.astype(str)
    result["source"] = source
    result["population"] = population
    metadata = mouse_meta if source == "MouseV2" else allen_meta
    fields = ["unit_id", "animal_id", "session_id", "physical_probe_id"]
    if source == "MouseV2":
        fields += ["location"]
    else:
        fields += ["common_qc"]
    result = result.merge(metadata[fields], on=["unit_id", "session_id"], how="left", validate="many_to_one")
    if result[["animal_id", "physical_probe_id"]].isna().any().any():
        raise ValueError(f"{source} metric rows are missing recording metadata")
    if source == "MouseV2":
        result = result.loc[result.location.str.startswith("VISp")].copy()
    else:
        result = result.loc[result.common_qc].copy()
    result["recording_id"] = (
        result.source + ":" + result.animal_id + ":" + result.session_id + ":" + result.physical_probe_id
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
    """Rebuild all response-eligible rows without inherited figure session lists."""
    mouse_meta, allen_meta = _metadata()
    frames: list[pd.DataFrame] = []

    mouse_ttfs = pd.read_csv(MOUSE_TTFS_UNITS, dtype={"session_id": str, "unit_id": str})
    mouse_ttfs = mouse_ttfs.loc[
        mouse_ttfs.cohort.eq("MouseV2") & mouse_ttfs.selected.astype(bool)
        & pd.to_numeric(mouse_ttfs.preferred_0_250_ttfs_ms, errors="coerce").lt(100)
    ].rename(columns={"location": "category", "preferred_0_250_ttfs_ms": "value"})
    mouse_ttfs["metric"] = METRICS[0]
    frames.append(_finish(mouse_ttfs, source="MouseV2", population="V1", mouse_meta=mouse_meta, allen_meta=allen_meta))

    allen_ttfs = pd.read_csv(ALLEN_TTFS_AUDIT, dtype={"session_id": str, "unit_id": str})
    allen_ttfs = allen_ttfs.loc[
        allen_ttfs.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & allen_ttfs.selected_positive_responder_area.astype(bool)
        & pd.to_numeric(allen_ttfs.preferred_0_250_ttfs_ms, errors="coerce").lt(100)
    ].rename(columns={"area_coarse": "category", "preferred_0_250_ttfs_ms": "value"})
    allen_ttfs["metric"] = METRICS[0]
    frames += [
        _finish(allen_ttfs.loc[allen_ttfs.category.eq("V1")], source="Allen", population="Central", mouse_meta=mouse_meta, allen_meta=allen_meta),
        _finish(allen_ttfs.loc[allen_ttfs.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse_meta=mouse_meta, allen_meta=allen_meta),
    ]

    mouse_f1 = pd.read_csv(MOUSE_HARMONIZED_F1_F0, dtype={"unit_id": str})
    mouse_f1 = mouse_f1.loc[
        mouse_f1.default_qc.astype(bool)
        & pd.to_numeric(mouse_f1.f1_f0_dg_common_support, errors="coerce").gt(0)
    ].rename(columns={"site_number": "session_id", "probe_letter": "category"})
    mouse_f1["value"] = np.log10(mouse_f1.f1_f0_dg_common_support.astype(float))
    mouse_f1["metric"] = METRICS[1]
    frames.append(_finish(mouse_f1, source="MouseV2", population="V1", mouse_meta=mouse_meta, allen_meta=allen_meta))

    allen_f1 = pd.read_csv(ALLEN_HARMONIZED_F1_F0, dtype={"ecephys_session_id": str, "ecephys_unit_id": str})
    allen_f1 = allen_f1.loc[
        allen_f1.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & pd.to_numeric(allen_f1.f1_f0_dg_harmonized, errors="coerce").gt(0)
    ].rename(columns={"ecephys_session_id": "session_id", "ecephys_unit_id": "unit_id", "area_coarse": "category"})
    allen_f1["value"] = np.log10(allen_f1.f1_f0_dg_harmonized.astype(float))
    allen_f1["metric"] = METRICS[1]
    frames += [
        _finish(allen_f1.loc[allen_f1.category.eq("V1")], source="Allen", population="Central", mouse_meta=mouse_meta, allen_meta=allen_meta),
        _finish(allen_f1.loc[allen_f1.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse_meta=mouse_meta, allen_meta=allen_meta),
    ]

    mouse_units = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=MOUSE_CANONICAL_GRATING_METRICS,
        population_profile="common_qc",
    )[["unit_id", "session_num", "probe_letter"]].drop_duplicates("unit_id")
    mouse_units["unit_id"] = mouse_units.unit_id.astype(str)
    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE, dtype={"unit_id": str})
    bridge = bridge.loc[
        bridge.view.eq("mouse_matched_150") & bridge.valid_timescale.astype(bool)
    ].merge(mouse_units, on="unit_id", validate="many_to_one")
    bridge = bridge.rename(columns={"session_id": "bridge_session_id", "session_num": "session_id", "probe_letter": "category", "subsample": "draw_id", "timescale_ms": "value"})
    if not bridge.bridge_session_id.astype(int).eq(bridge.session_id.astype(int)).all():
        raise ValueError("Timescale bridge session metadata disagree")
    bridge["metric"] = METRICS[2]
    frames.append(_finish(bridge, source="MouseV2", population="V1", mouse_meta=mouse_meta, allen_meta=allen_meta))

    allen_tau = load_allen_units(population_profile="common_qc")
    allen_tau = allen_tau.loc[
        allen_tau.area_coarse.isin((*HVA_CATEGORIES, "V1"))
        & pd.to_numeric(allen_tau.timescale_ac, errors="coerce").between(1, 300)
        & pd.to_numeric(allen_tau.spike_count_ac, errors="coerce").gt(50)
        & pd.to_numeric(allen_tau.err_ac, errors="coerce").lt(20)
    ].rename(columns={"ecephys_session_id": "session_id", "ecephys_unit_id": "unit_id", "area_coarse": "category", "timescale_ac": "value"})
    allen_tau["metric"] = METRICS[2]
    frames += [
        _finish(allen_tau.loc[allen_tau.category.eq("V1")], source="Allen", population="Central", mouse_meta=mouse_meta, allen_meta=allen_meta),
        _finish(allen_tau.loc[allen_tau.category.isin(HVA_CATEGORIES)], source="Allen", population="HVA", mouse_meta=mouse_meta, allen_meta=allen_meta),
    ]

    result = pd.concat(frames, ignore_index=True)
    expected = {"V1": set(V1_CATEGORIES), "HVA": set(HVA_CATEGORIES), "Central": {"V1"}}
    for population, categories in expected.items():
        found = set(result.loc[result.population.eq(population), "category"])
        if found != categories:
            raise ValueError(f"Unexpected {population} categories: {sorted(found)}")
    return result.sort_values(["metric", "source", "session_id", "category", "unit_id", "draw_id"]).reset_index(drop=True)


def apply_cell_floor(table: pd.DataFrame, floor: int = 5) -> pd.DataFrame:
    """Apply the same recording/category floor, separately for every draw."""
    keys = ["metric", "recording_id", "category", "draw_id"]
    count = table.groupby(keys).unit_id.transform("nunique")
    return table.loc[count.ge(floor)].copy()


def collapse_nondraw_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Keep draw rows for timescale and one row per cell for other metrics."""
    non_timescale = table.loc[table.metric.ne(METRICS[2])]
    timescale = table.loc[table.metric.eq(METRICS[2])]
    if non_timescale.draw_id.ne(-1).any():
        raise ValueError("Non-timescale metrics unexpectedly contain draws")
    return pd.concat([non_timescale, timescale], ignore_index=True)


def category_recording_weights(table: pd.DataFrame) -> pd.Series:
    """Equal category, animal, recording-within-animal, and cell weights."""
    required = {"category", "animal_id", "recording_id", "unit_id"}
    if missing := required.difference(table.columns):
        raise ValueError(f"Weight table is missing {sorted(missing)}")
    if table.duplicated(["category", "unit_id"]).any():
        raise ValueError("Weights require at most one row per category/unit")
    k = table.category.nunique()
    animals = table.groupby("category").animal_id.transform("nunique")
    recordings = table.groupby(["category", "animal_id"]).recording_id.transform("nunique")
    cells = table.groupby(["category", "animal_id", "recording_id"]).unit_id.transform("nunique")
    weights = 1.0 / (k * animals * recordings * cells)
    totals = weights.groupby(table.category).sum()
    np.testing.assert_allclose(totals.to_numpy(), np.full(k, 1 / k), atol=1e-12)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    return weights


def rf_features(table: pd.DataFrame) -> np.ndarray:
    az = (table.azimuth.to_numpy(float) - 50.0) / 25.0
    el = (table.elevation.to_numpy(float) - 10.0) / 25.0
    return np.column_stack([az, el, az**2, el**2, az * el])


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


def fit_standardized_dispersion(
    table: pd.DataFrame,
    *,
    adjusted: bool = False,
    reference_rf: pd.DataFrame | None = None,
    equal_cells: bool = False,
) -> DispersionFit:
    """Fit weighted category means with session nuisance and optional RF terms.

    Session effects are fixed nuisance terms.  They are omitted from standardized
    predictions (the declared population reference); category dispersion is
    invariant to the arbitrary omitted session level.  Whole-animal bootstrap
    inference is implemented by the runner, not by this point-estimate function.
    """
    local = table.copy()
    if local.duplicated("unit_id").any():
        raise ValueError("A point fit must contain one outcome per unit")
    categories = sorted(local.category.unique())
    category_x = np.column_stack([local.category.eq(value).to_numpy(float) for value in categories])
    sessions = sorted(local.session_id.unique())
    nuisance = np.column_stack([local.session_id.eq(value).to_numpy(float) for value in sessions[1:]]) if len(sessions) > 1 else np.empty((len(local), 0))
    pieces = [category_x, nuisance]
    if adjusted:
        if reference_rf is None:
            raise ValueError("Adjusted fits require a frozen RF reference")
        pieces.append(rf_features(local))
    x = np.column_stack(pieces)
    weights = np.full(len(local), 1 / len(local)) if equal_cells else category_recording_weights(local).to_numpy()
    root_w = np.sqrt(weights)
    xw, yw = x * root_w[:, None], local.value.to_numpy(float) * root_w
    beta, _, rank, singular = np.linalg.lstsq(xw, yw, rcond=1e-10)
    if rank != x.shape[1]:
        raise np.linalg.LinAlgError(f"Rank-deficient design ({rank}/{x.shape[1]})")
    standardized = beta[:len(categories)].copy()
    if adjusted:
        standardized += rf_features(reference_rf).mean(axis=0) @ beta[-5:]
    grand = standardized.mean()
    dispersion = float(np.mean((standardized - grand) ** 2))
    y = local.value.to_numpy(float)
    raw_mean = float(np.sum(weights * y))
    total_variance = float(np.sum(weights * (y - raw_mean) ** 2))
    return DispersionFit(
        category_means=dict(zip(categories, standardized)),
        dispersion=dispersion,
        root_dispersion=float(np.sqrt(dispersion)),
        total_variance=total_variance,
        fraction_percent=100 * dispersion / total_variance,
        rank=int(rank),
        parameters=x.shape[1],
        condition_number=float(singular[0] / singular[-1]),
        n_cells=len(local),
        n_recordings=local.recording_id.nunique(),
        n_animals=local.animal_id.nunique(),
    )


def registry_summary(table: pd.DataFrame) -> pd.DataFrame:
    keys = ["metric", "population", "category"]
    return table.groupby(keys, as_index=False).agg(
        animals=("animal_id", "nunique"), sessions=("session_id", "nunique"),
        physical_recordings=("recording_id", "nunique"), cells=("unit_id", "nunique"),
        draw_rows=("unit_id", "size"),
    )


def cooccurrence_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, population), part in table.groupby(["metric", "population"], sort=False):
        if population == "Central":
            continue
        sets = part.groupby(["source", "session_id"]).category.apply(set)
        categories = sorted(part.category.unique())
        for i, left in enumerate(categories):
            for right in categories[i + 1:]:
                rows.append({
                    "metric": metric, "population": population,
                    "category_a": left, "category_b": right,
                    "sessions_with_both": int(sum({left, right}.issubset(values) for values in sets)),
                })
    return pd.DataFrame(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
