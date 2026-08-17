#!/usr/bin/env python3
"""Estimate visual-field translations from V1 RF-size structure in 14 animals.

Translations are selected by cross-half V1 prediction and then transferred to
HVA cells, whose RF sizes are never used to estimate the animal's offset.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1/rf_size_map_alignment"
    / "primary_uncensored_interior_rf_size_cells.csv.gz"
)
OUTPUT = (
    ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1"
    / "visual_field_translation_v1_rf_size_v1"
)
AZIMUTH = np.arange(15.0, 86.0, 1.0)
ELEVATION = np.arange(-20.0, 41.0, 1.0)
SURFACE_BANDWIDTH_DEG = 8.0
MAX_SHIFT_DEG = 30
SHIFT_GRID = np.arange(-MAX_SHIFT_DEG, MAX_SHIFT_DEG + 1, 1, dtype=int)
BOUNDS = (5, 10, 20, 30)
PENALTIES = (0.01, 0.03, 0.10, 0.30, 1.0)
HVA_BANDWIDTHS = (0.20, 0.35, 0.55, 0.80, 1.15)


def prepare() -> pd.DataFrame:
    data = pd.read_csv(INPUT, low_memory=False)
    data["cortical_group"] = np.where(data.ecephys_structure_acronym.eq("VISp"), "V1", "HVA")
    data["group_median_log2_area"] = data.groupby(
        ["session_id", "cortical_group"], observed=True
    ).log2_rf_area_deg2.transform("median")
    data["group_centered_log2_area"] = data.log2_rf_area_deg2 - data.group_median_log2_area
    v1 = data.ecephys_structure_acronym.eq("VISp")
    v1_median = data.loc[v1].groupby("session_id", observed=True).log2_rf_area_deg2.transform("median")
    v1_iqr = data.loc[v1].groupby("session_id", observed=True).log2_rf_area_deg2.transform(
        lambda x: x.quantile(.75) - x.quantile(.25)
    )
    data.loc[v1, "v1_standardized_log2_area"] = (
        data.loc[v1, "log2_rf_area_deg2"].to_numpy() - v1_median.to_numpy()
    ) / np.maximum(v1_iqr.to_numpy(), .25)
    data["split_half"] = -1
    for session_id, indices in data.loc[v1].groupby("session_id", observed=True).groups.items():
        indices = np.asarray(list(indices))
        rng = np.random.default_rng(20260816 + int(session_id))
        shuffled = indices[rng.permutation(len(indices))]
        data.loc[shuffled, "split_half"] = np.arange(len(shuffled)) % 2
    return data


def make_surface(frame: pd.DataFrame, minimum_effective: float) -> dict[str, np.ndarray]:
    az_mesh, el_mesh = np.meshgrid(AZIMUTH, ELEVATION)
    grid = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    points = frame[["common_azimuth_deg", "common_elevation_deg"]].to_numpy(float)
    values = frame.v1_standardized_log2_area.to_numpy(float)
    distances = cdist(grid, points)
    weights = np.exp(-.5 * (distances / SURFACE_BANDWIDTH_DEG) ** 2)
    weight_sum = weights.sum(axis=1)
    effective = np.divide(
        weight_sum**2, np.square(weights).sum(axis=1),
        out=np.zeros_like(weight_sum), where=np.square(weights).sum(axis=1) > 0,
    )
    near = (distances <= 20).sum(axis=1)
    valid = (effective >= minimum_effective) & (near >= 3) & (weight_sum > 1e-12)
    estimate = np.divide(
        weights @ values, weight_sum,
        out=np.full(len(grid), np.nan), where=weight_sum > 1e-12,
    )
    estimate[~valid] = np.nan
    evidence = np.where(valid, np.sqrt(effective), 0.0)
    shape = (len(ELEVATION), len(AZIMUTH))
    return {"value": estimate.reshape(shape), "evidence": evidence.reshape(shape)}


def build_maps(data: pd.DataFrame, split_half: int | None) -> dict[int, dict[str, np.ndarray]]:
    selected = data.loc[data.ecephys_structure_acronym.eq("VISp")].copy()
    minimum_effective = 3.0
    if split_half is not None:
        selected = selected.loc[selected.split_half.eq(split_half)]
        minimum_effective = 1.5
    return {
        int(session_id): make_surface(frame, minimum_effective)
        for session_id, frame in selected.groupby("session_id", observed=True)
    }


def shift_surface(surface: dict[str, np.ndarray], azimuth_shift: int, elevation_shift: int) -> dict[str, np.ndarray]:
    def exact_integer_shift(array: np.ndarray, row_shift: int, column_shift: int, fill: float) -> np.ndarray:
        result = np.full_like(array, fill)
        source_rows = slice(max(0, -row_shift), min(array.shape[0], array.shape[0] - row_shift))
        target_rows = slice(max(0, row_shift), min(array.shape[0], array.shape[0] + row_shift))
        source_columns = slice(max(0, -column_shift), min(array.shape[1], array.shape[1] - column_shift))
        target_columns = slice(max(0, column_shift), min(array.shape[1], array.shape[1] + column_shift))
        result[target_rows, target_columns] = array[source_rows, source_columns]
        return result

    row_shift, column_shift = int(elevation_shift), int(azimuth_shift)
    return {
        "value": exact_integer_shift(surface["value"], row_shift, column_shift, np.nan),
        "evidence": exact_integer_shift(surface["evidence"], row_shift, column_shift, 0.0),
    }


def template_from_maps(maps: dict[int, dict[str, np.ndarray]], exclude: int | None = None) -> dict[str, np.ndarray]:
    selected = [value for session_id, value in maps.items() if session_id != exclude]
    values = np.stack([x["value"] for x in selected])
    evidence = np.stack([x["evidence"] for x in selected])
    valid = np.isfinite(values) & (evidence > 0)
    numerator = np.nansum(np.where(valid, values * evidence, 0.0), axis=0)
    denominator = np.sum(np.where(valid, evidence, 0.0), axis=0)
    mean = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return {"value": mean, "evidence": denominator, "supporting_animals": valid.sum(axis=0)}


def map_score(source: dict[str, np.ndarray], template: dict[str, np.ndarray]) -> dict[str, float]:
    valid = (
        np.isfinite(source["value"]) & np.isfinite(template["value"])
        & (source["evidence"] > 0) & (template["evidence"] > 0)
    )
    points = int(valid.sum())
    source_points = max(int(np.isfinite(source["value"]).sum()), 1)
    if points < 50:
        return {"correlation": np.nan, "rmse": np.nan, "coverage": points / source_points, "points": points}
    x = source["value"][valid]
    y = template["value"][valid]
    correlation = np.corrcoef(x, y)[0, 1] if np.std(x) > 1e-9 and np.std(y) > 1e-9 else np.nan
    return {
        "correlation": float(correlation),
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))),
        "coverage": float(points / source_points),
        "points": points,
    }


def loss_grid(maps: dict[int, dict[str, np.ndarray]], label: str) -> pd.DataFrame:
    rows = []
    for session_position, session_id in enumerate(sorted(maps)):
        source = maps[session_id]
        template = template_from_maps(maps, exclude=session_id)
        for elevation_shift in SHIFT_GRID:
            for azimuth_shift in SHIFT_GRID:
                shifted = shift_surface(source, azimuth_shift, elevation_shift)
                metrics = map_score(shifted, template)
                correlation = metrics["correlation"]
                base_loss = (
                    3.0 if not np.isfinite(correlation)
                    else 1.0 - correlation + .2 * (1.0 - metrics["coverage"])
                )
                rows.append({
                    "map_set": label, "session_id": int(session_id),
                    "translation_azimuth_deg": int(azimuth_shift),
                    "translation_elevation_deg": int(elevation_shift),
                    "base_loss": base_loss, **metrics,
                })
        print(f"translation loss grid {label}: {session_position + 1}/{len(maps)}", flush=True)
    return pd.DataFrame(rows)


def select_shifts(grid: pd.DataFrame, bound: int, penalty: float) -> pd.DataFrame:
    selected = grid.loc[
        grid.translation_azimuth_deg.abs().le(bound)
        & grid.translation_elevation_deg.abs().le(bound)
    ].copy()
    selected["regularized_loss"] = selected.base_loss + penalty * .5 * (
        np.square(selected.translation_azimuth_deg / 10.0)
        + np.square(selected.translation_elevation_deg / 10.0)
    )
    optimum = selected.loc[selected.groupby("session_id").regularized_loss.idxmin()].copy()
    # Fix the otherwise arbitrary common visual-field gauge.
    optimum["translation_azimuth_deg"] -= int(np.rint(optimum.translation_azimuth_deg.median()))
    optimum["translation_elevation_deg"] -= int(np.rint(optimum.translation_elevation_deg.median()))
    optimum["bound_deg"] = bound
    optimum["regularization_weight"] = penalty
    return optimum.reset_index(drop=True)


def shifted_maps(maps: dict[int, dict[str, np.ndarray]], shifts: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    lookup = shifts.set_index("session_id")
    return {
        session_id: shift_surface(
            surface,
            int(lookup.loc[session_id, "translation_azimuth_deg"]),
            int(lookup.loc[session_id, "translation_elevation_deg"]),
        )
        for session_id, surface in maps.items()
    }


def evaluate_cross_half(
    test_maps: dict[int, dict[str, np.ndarray]], shifts: pd.DataFrame, training_half: int,
    bound: int, penalty: float,
) -> pd.DataFrame:
    aligned = shifted_maps(test_maps, shifts)
    rows = []
    lookup = shifts.set_index("session_id")
    for session_id in sorted(test_maps):
        raw = map_score(test_maps[session_id], template_from_maps(test_maps, exclude=session_id))
        corrected = map_score(aligned[session_id], template_from_maps(aligned, exclude=session_id))
        rows.append({
            "session_id": session_id, "training_half": training_half, "test_half": 1 - training_half,
            "bound_deg": bound, "regularization_weight": penalty,
            "translation_azimuth_deg": lookup.loc[session_id, "translation_azimuth_deg"],
            "translation_elevation_deg": lookup.loc[session_id, "translation_elevation_deg"],
            "raw_correlation": raw["correlation"], "corrected_correlation": corrected["correlation"],
            "correlation_gain": corrected["correlation"] - raw["correlation"],
            "raw_rmse": raw["rmse"], "corrected_rmse": corrected["rmse"],
            "rmse_gain": raw["rmse"] - corrected["rmse"],
            "raw_points": raw["points"], "corrected_points": corrected["points"],
        })
    return pd.DataFrame(rows)


def tune_translation(
    half_maps: dict[int, dict[int, dict[str, np.ndarray]]],
    half_grids: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, float]]:
    all_scores = []
    summary_rows = []
    candidates = [(0, 0.0)] + [(bound, penalty) for bound in BOUNDS for penalty in PENALTIES]
    for bound, penalty in candidates:
        if bound == 0:
            shifts = {
                half: pd.DataFrame({
                    "session_id": sorted(half_maps[half]),
                    "translation_azimuth_deg": 0,
                    "translation_elevation_deg": 0,
                })
                for half in (0, 1)
            }
        else:
            shifts = {half: select_shifts(half_grids[half], bound, penalty) for half in (0, 1)}
        local = []
        for training_half, test_half in ((0, 1), (1, 0)):
            local.append(evaluate_cross_half(
                half_maps[test_half], shifts[training_half], training_half, bound, penalty
            ))
        local = pd.concat(local, ignore_index=True)
        all_scores.append(local)
        per_session = local.groupby("session_id", observed=True).agg(
            correlation_gain=("correlation_gain", "mean"), rmse_gain=("rmse_gain", "mean")
        )
        summary_rows.append({
            "bound_deg": bound, "regularization_weight": penalty,
            "median_cross_half_correlation_gain": per_session.correlation_gain.median(),
            "median_cross_half_rmse_gain": per_session.rmse_gain.median(),
            "mean_cross_half_rmse_gain": per_session.rmse_gain.mean(),
            "positive_rmse_gain_fraction": float((per_session.rmse_gain > 0).mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["median_cross_half_rmse_gain", "median_cross_half_correlation_gain", "positive_rmse_gain_fraction"],
        ascending=False,
    )
    selected = (int(summary.iloc[0].bound_deg), float(summary.iloc[0].regularization_weight))
    scores = pd.concat(all_scores, ignore_index=True)
    return scores, summary, selected


def balanced_predict(train: pd.DataFrame, test: pd.DataFrame, columns: tuple[str, str], bandwidth: float) -> np.ndarray:
    x_train = train[list(columns)].to_numpy(float)
    x_test = test[list(columns)].to_numpy(float)
    center = np.nanmedian(x_train, axis=0)
    scale = np.nanquantile(x_train, .75, axis=0) - np.nanquantile(x_train, .25, axis=0)
    scale = np.where(scale > 1e-9, scale, np.nanstd(x_train, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    x_test = (x_test - center) / scale
    predictions = []
    for _, animal in train.groupby("session_id", observed=True):
        x_animal = (animal[list(columns)].to_numpy(float) - center) / scale
        distance2 = cdist(x_test, x_animal, metric="sqeuclidean")
        weights = np.exp(-.5 * (distance2 - distance2.min(axis=1, keepdims=True)) / bandwidth**2)
        predictions.append((weights @ animal.group_centered_log2_area.to_numpy(float)) / weights.sum(axis=1))
    return np.mean(np.vstack(predictions), axis=0)


def score_cells(test: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    observed = test.group_centered_log2_area.to_numpy(float)
    return {
        "spearman_rho": float(spearmanr(observed, prediction).statistic),
        "mae_log2": float(np.median(np.abs(observed - prediction))),
        "constant_mae_log2": float(np.median(np.abs(observed))),
    }


def select_hva_bandwidth(train: pd.DataFrame, columns: tuple[str, str]) -> float:
    rows = []
    for bandwidth in HVA_BANDWIDTHS:
        rhos = []
        for session_id in sorted(train.session_id.unique()):
            test = train.loc[train.session_id.eq(session_id)]
            inner = train.loc[~train.session_id.eq(session_id)]
            rhos.append(score_cells(test, balanced_predict(inner, test, columns, bandwidth))["spearman_rho"])
        rows.append((float(np.nanmedian(rhos)), bandwidth))
    return max(rows)[1]


def evaluate_hva(data: pd.DataFrame, translations: pd.DataFrame) -> pd.DataFrame:
    lookup = translations.set_index("session_id")
    data = data.copy()
    data["corrected_azimuth_deg"] = data.common_azimuth_deg + data.session_id.map(lookup.translation_azimuth_deg)
    data["corrected_elevation_deg"] = data.common_elevation_deg + data.session_id.map(lookup.translation_elevation_deg)
    hva = data.loc[data.cortical_group.eq("HVA")].copy()
    systems = {
        "raw_rf_location": ("common_azimuth_deg", "common_elevation_deg"),
        "v1_translation_corrected_rf_location": ("corrected_azimuth_deg", "corrected_elevation_deg"),
    }
    rows = []
    for outer_session in sorted(hva.session_id.unique()):
        train = hva.loc[~hva.session_id.eq(outer_session)]
        test = hva.loc[hva.session_id.eq(outer_session)]
        for system, columns in systems.items():
            bandwidth = select_hva_bandwidth(train, columns)
            metrics = score_cells(test, balanced_predict(train, test, columns, bandwidth))
            rows.append({
                "session_id": outer_session, "system": system, "cells": len(test),
                "bandwidth_iqr_units": bandwidth, **metrics,
                "mae_gain_vs_constant_log2": metrics["constant_mae_log2"] - metrics["mae_log2"],
            })
    return pd.DataFrame(rows)


def select_cases(cross_half: pd.DataFrame) -> pd.DataFrame:
    session = cross_half.groupby("session_id", observed=True).agg(
        mean_rmse_gain=("rmse_gain", "mean"), mean_correlation_gain=("correlation_gain", "mean")
    ).reset_index()
    median = session.mean_rmse_gain.median()
    roles = [
        ("largest held-out gain", session.mean_rmse_gain.idxmax(), "maximum cross-half RMSE gain"),
        ("typical effect", (session.mean_rmse_gain - median).abs().idxmin(), "closest to median cross-half RMSE gain"),
        ("translation failure", session.mean_rmse_gain.idxmin(), "minimum cross-half RMSE gain"),
    ]
    rows = []
    for role, index, criterion in roles:
        row = session.loc[index]
        rows.append({
            "session_id": int(row.session_id), "selection_role": role, "criterion": criterion,
            "criterion_value": row.mean_rmse_gain, "mean_correlation_gain": row.mean_correlation_gain,
            "provenance": "algorithmic selection from selected cross-half translation model",
        })
    return pd.DataFrame(rows)


def render_summary(
    full_maps: dict[int, dict[str, np.ndarray]], translations: pd.DataFrame,
    cross_half: pd.DataFrame, tuning: pd.DataFrame, hva: pd.DataFrame, output: Path,
) -> None:
    raw_template = template_from_maps(full_maps)
    aligned_maps = shifted_maps(full_maps, translations)
    aligned_template = template_from_maps(aligned_maps)
    joint = np.r_[raw_template["value"][np.isfinite(raw_template["value"])],
                  aligned_template["value"][np.isfinite(aligned_template["value"])]]
    limit = max(.2, float(np.nanquantile(np.abs(joint), .98)))
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10), constrained_layout=True)
    for axis, template, title in [
        (axes[0, 0], raw_template, "Raw V1 RF-size consensus"),
        (axes[0, 1], aligned_template, "Translation-corrected V1 consensus"),
    ]:
        artist = axis.pcolormesh(AZIMUTH, ELEVATION, template["value"], shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.contour(AZIMUTH, ELEVATION, template["supporting_animals"], levels=[7, 10, 13], colors="#333333", linewidths=.7)
        axis.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title=title, aspect="equal")
    fig.colorbar(artist, ax=axes[0, :2], fraction=.025, pad=.02, label="Within-animal standardized log₂ RF area")

    axis = axes[0, 2]
    axis.quiver(np.zeros(len(translations)), np.zeros(len(translations)),
                translations.translation_azimuth_deg, translations.translation_elevation_deg,
                angles="xy", scale_units="xy", scale=1, color="#2f6b9a", alpha=.8)
    for row in translations.itertuples():
        axis.text(row.translation_azimuth_deg+.3, row.translation_elevation_deg+.3, str(row.session_id)[-3:], fontsize=7)
    extent = max(8, float(np.abs(translations[["translation_azimuth_deg", "translation_elevation_deg"]]).to_numpy().max()) + 2)
    axis.axhline(0, color="#777777", lw=.8); axis.axvline(0, color="#777777", lw=.8)
    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
             xlabel="Azimuth correction (deg)", ylabel="Elevation correction (deg)", title="V1-derived animal translations")

    axis = axes[1, 0]
    session = cross_half.groupby("session_id", observed=True).agg(raw=("raw_rmse", "mean"), corrected=("corrected_rmse", "mean")).sort_index()
    axis.scatter(session.raw, session.corrected, color="#2f6b9a", s=45)
    lo = min(session.min()); hi = max(session.max())
    axis.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
    for sid, row in session.iterrows(): axis.text(row.raw+.003, row.corrected, str(sid)[-3:], fontsize=7)
    axis.set(xlabel="Raw cross-half V1 RMSE", ylabel="Corrected cross-half V1 RMSE", title="Independent V1-cell validation", aspect="equal")

    axis = axes[1, 1]
    split = tuning.pivot(index="session_id", columns="training_half", values=["translation_azimuth_deg", "translation_elevation_deg"])
    for field, color, label in [("translation_azimuth_deg", "#2f6b9a", "Azimuth"), ("translation_elevation_deg", "#d18b2c", "Elevation")]:
        axis.scatter(split[(field, 0)], split[(field, 1)], s=42, color=color, alpha=.8, label=label)
    lo = min(split.min()); hi = max(split.max()); axis.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
    axis.set(xlabel="Shift from V1 half 0 (deg)", ylabel="Shift from V1 half 1 (deg)", title="Split-half translation reliability", aspect="equal")
    axis.legend(frameon=False)

    axis = axes[1, 2]
    wide = hva.pivot(index="session_id", columns="system", values="mae_log2")
    axis.scatter(wide.raw_rf_location, wide.v1_translation_corrected_rf_location, color="#d18b2c", s=45)
    lo = min(wide.min()); hi = max(wide.max()); axis.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
    for sid, row in wide.iterrows(): axis.text(row.raw_rf_location+.003, row.v1_translation_corrected_rf_location, str(sid)[-3:], fontsize=7)
    axis.set(xlabel="Raw HVA held-out MAE (log₂)", ylabel="V1-shift-corrected HVA MAE (log₂)",
             title="Independent transfer to HVA RF size", aspect="equal")
    for axis in axes.flat: axis.grid(alpha=.14)
    selected = translations.iloc[0]
    fig.suptitle(
        "Animal visual-field translations inferred from V1 RF-size structure\n"
        f"selected by cross-half V1 prediction: ±{int(selected.bound_deg)}° bound, penalty={selected.regularization_weight:g}",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_cases(
    full_maps: dict[int, dict[str, np.ndarray]], translations: pd.DataFrame,
    cases: pd.DataFrame, output: Path,
) -> None:
    aligned = shifted_maps(full_maps, translations)
    lookup = translations.set_index("session_id")
    fig, axes = plt.subplots(len(cases), 3, figsize=(14.5, 12), constrained_layout=True)
    all_values = np.concatenate([m["value"][np.isfinite(m["value"])] for m in full_maps.values()])
    limit = max(.2, float(np.quantile(np.abs(all_values), .98)))
    for row_index, case in cases.reset_index(drop=True).iterrows():
        sid = int(case.session_id)
        template = template_from_maps(aligned, exclude=sid)
        for column, surface, title in [
            (0, full_maps[sid], "raw session surface"),
            (1, aligned[sid], "after V1-derived translation"),
            (2, template, "other-animal aligned template"),
        ]:
            axes[row_index, column].pcolormesh(AZIMUTH, ELEVATION, surface["value"], shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row_index, column].set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", aspect="equal")
            axes[row_index, column].set_title(
                f"{case.selection_role}: {sid}\n{title}"
                + (f"\nshift=({lookup.loc[sid, 'translation_azimuth_deg']:+.0f}, {lookup.loc[sid, 'translation_elevation_deg']:+.0f})°" if column == 1 else "")
            )
            axes[row_index, column].grid(alpha=.12)
    fig.suptitle("Concrete V1 translation cases selected from cross-half prediction", fontsize=15)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = prepare()
    half_maps = {half: build_maps(data, half) for half in (0, 1)}
    full_maps = build_maps(data, None)
    half_grids = {half: loss_grid(half_maps[half], f"half_{half}") for half in (0, 1)}
    cross_all, tuning_summary, selected = tune_translation(half_maps, half_grids)
    selected_bound, selected_penalty = selected
    selected_cross = cross_all.loc[
        cross_all.bound_deg.eq(selected_bound) & cross_all.regularization_weight.eq(selected_penalty)
    ].copy()
    full_grid = loss_grid(full_maps, "full")
    if selected_bound == 0:
        translations = pd.DataFrame({
            "session_id": sorted(full_maps), "translation_azimuth_deg": 0,
            "translation_elevation_deg": 0, "bound_deg": 0, "regularization_weight": 0.0,
        })
    else:
        translations = select_shifts(full_grid, selected_bound, selected_penalty)
    translations["translation_magnitude_deg"] = np.hypot(
        translations.translation_azimuth_deg, translations.translation_elevation_deg
    )
    translations["at_bound"] = False if selected_bound == 0 else (
        translations.translation_azimuth_deg.abs().ge(selected_bound)
        | translations.translation_elevation_deg.abs().ge(selected_bound)
    )
    # Split-half shift table for identifiability audit.
    split_shifts = []
    for half in (0, 1):
        if selected_bound == 0:
            local = pd.DataFrame({"session_id": sorted(half_maps[half]), "translation_azimuth_deg": 0, "translation_elevation_deg": 0})
        else:
            local = select_shifts(half_grids[half], selected_bound, selected_penalty)
        local["training_half"] = half
        split_shifts.append(local)
    split_shifts = pd.concat(split_shifts, ignore_index=True)
    hva = evaluate_hva(data, translations)
    cases = select_cases(selected_cross)

    translations.to_csv(OUTPUT / "selected_v1_visual_field_translations.csv", index=False)
    split_shifts.to_csv(OUTPUT / "v1_translation_split_half_estimates.csv", index=False)
    selected_cross.to_csv(OUTPUT / "selected_cross_half_v1_prediction.csv", index=False)
    cross_all.to_csv(OUTPUT / "all_translation_candidate_cross_half_scores.csv.gz", index=False, compression="gzip")
    tuning_summary.to_csv(OUTPUT / "translation_hyperparameter_selection.csv", index=False)
    with gzip.open(OUTPUT / "translation_loss_grids.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        pd.concat([*half_grids.values(), full_grid], ignore_index=True).to_csv(stream, index=False)
    hva.to_csv(OUTPUT / "hva_prediction_before_after_v1_translation.csv", index=False)
    cases.to_csv(OUTPUT / "selected_translation_case_audit.csv", index=False)
    corrected = data.merge(
        translations[["session_id", "translation_azimuth_deg", "translation_elevation_deg"]],
        on="session_id", how="left", validate="many_to_one",
    )
    corrected["translation_corrected_azimuth_deg"] = corrected.common_azimuth_deg + corrected.translation_azimuth_deg
    corrected["translation_corrected_elevation_deg"] = corrected.common_elevation_deg + corrected.translation_elevation_deg
    corrected.to_csv(OUTPUT / "cells_with_v1_translation_corrected_rf_coordinates.csv.gz", index=False, compression="gzip")

    render_summary(
        full_maps, translations, selected_cross, split_shifts, hva,
        OUTPUT / "Figure_14animal_V1_visual_translation_summary.png",
    )
    render_cases(
        full_maps, translations, cases,
        OUTPUT / "Figure_V1_visual_translation_selected_cases.png",
    )
    session_cross = selected_cross.groupby("session_id", observed=True).agg(
        rmse_gain=("rmse_gain", "mean"), correlation_gain=("correlation_gain", "mean")
    )
    hva_wide = hva.pivot(index="session_id", columns="system", values="mae_log2")
    split_wide = split_shifts.pivot(index="session_id", columns="training_half", values=["translation_azimuth_deg", "translation_elevation_deg"])
    def safe_spearman(first: pd.Series, second: pd.Series) -> float | None:
        if first.nunique() < 2 or second.nunique() < 2:
            return None
        value = spearmanr(first, second).statistic
        return float(value) if np.isfinite(value) else None

    manifest = {
        "status": "exploratory translation-only diagnostic; V1 RF size estimates animal visual-field offsets",
        "selection": {
            "bound_deg": selected_bound, "regularization_weight": selected_penalty,
            "criterion": "maximum median cross-half V1 standardized-surface RMSE gain; identity included",
        },
        "support": {"animals": int(data.session_id.nunique()), "v1_cells": int((data.cortical_group == "V1").sum()), "hva_cells": int((data.cortical_group == "HVA").sum())},
        "v1_cross_half": {
            "median_rmse_gain": float(session_cross.rmse_gain.median()),
            "median_correlation_gain": float(session_cross.correlation_gain.median()),
            "animals_with_rmse_improvement": int((session_cross.rmse_gain > 0).sum()),
        },
        "translation": {
            "median_magnitude_deg": float(translations.translation_magnitude_deg.median()),
            "max_magnitude_deg": float(translations.translation_magnitude_deg.max()),
            "at_bound": int(translations.at_bound.sum()),
            "split_half_azimuth_rho": safe_spearman(split_wide[("translation_azimuth_deg", 0)], split_wide[("translation_azimuth_deg", 1)]),
            "split_half_elevation_rho": safe_spearman(split_wide[("translation_elevation_deg", 0)], split_wide[("translation_elevation_deg", 1)]),
        },
        "hva_transfer": {
            "median_raw_mae_log2": float(hva_wide.raw_rf_location.median()),
            "median_corrected_mae_log2": float(hva_wide.v1_translation_corrected_rf_location.median()),
            "median_paired_mae_gain_log2": float((hva_wide.raw_rf_location - hva_wide.v1_translation_corrected_rf_location).median()),
            "animals_improved": int((hva_wide.raw_rf_location > hva_wide.v1_translation_corrected_rf_location).sum()),
        },
        "limitations": [
            "Hyperparameters are selected on the frozen 14-animal exploratory cohort; new animals remain confirmatory.",
            "V1 cross-half validation uses independent cells but the same penetration and stimulus session.",
            "HVA transfer is independent of HVA RF size for translation estimation, but V1 and HVA share session-level measurement conditions.",
            "A common visual-field translation remains a gauge choice; median correction is fixed to zero.",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
