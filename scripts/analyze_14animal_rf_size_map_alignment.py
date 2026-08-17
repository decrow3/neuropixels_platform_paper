#!/usr/bin/env python3
"""Assess whether RF-size structure aligns across the frozen 14-animal map.

The primary estimand is shape alignment: can an animal-balanced local surface
learned in other animals predict within-animal/area deviations in log2 RF area?
The held-out unit is an animal; scores are summarized at animal-area grain.
"""

from __future__ import annotations

import hashlib
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
REGISTRATION = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1"
INPUT = REGISTRATION / "registered_cells_common_zhuang_coordinates.csv.gz"
TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = REGISTRATION / "rf_size_map_alignment"

SYSTEMS = {
    "ccf_anatomy": ("ccf_ml_mm", "ccf_ap_mm"),
    "registered_map": ("common_template_x_px", "common_template_y_px"),
    "observed_rf": ("common_azimuth_deg", "common_elevation_deg"),
}
SYSTEM_LABELS = {
    "ccf_anatomy": "Raw CCF anatomy",
    "registered_map": "Registered Zhuang map",
    "observed_rf": "Observed RF location",
}
SYSTEM_COLORS = {
    "ccf_anatomy": "#8c8c8c",
    "registered_map": "#2f6b9a",
    "observed_rf": "#d18b2c",
}
AREA_LABELS = {"VISp": "V1", "VISl": "LM", "VISal": "AL", "VISrl": "RL", "VISam": "AM"}
AREA_ORDER = ["VISp", "VISl", "VISal", "VISrl", "VISam"]
BANDWIDTHS = (0.20, 0.35, 0.55, 0.80, 1.15)
MIN_GROUP_CELLS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(INPUT, low_memory=False)
    source["axis_censored"] = source.axis_censored.astype(bool)
    source["finite_positive_area"] = np.isfinite(source.axis_area_deg2) & source.axis_area_deg2.gt(0)
    source["uncensored"] = ~source.axis_censored
    source["interior_10deg"] = source.axis_edge_distance_deg.gt(10)
    selected = source.loc[
        source.finite_positive_area & source.uncensored & source.interior_10deg
    ].copy()
    selected["log2_rf_area_deg2"] = np.log2(selected.axis_area_deg2)
    selected["animal_median_log2_area"] = selected.groupby("session_id", observed=True).log2_rf_area_deg2.transform("median")
    selected["session_centered_log2_rf_area"] = selected.log2_rf_area_deg2 - selected.animal_median_log2_area
    group = ["session_id", "ecephys_structure_acronym"]
    selected["animal_area_median_log2_area"] = selected.groupby(group, observed=True).log2_rf_area_deg2.transform("median")
    selected["centered_log2_rf_area"] = selected.log2_rf_area_deg2 - selected.animal_area_median_log2_area
    audit = (
        source.groupby(["session_id", "ecephys_structure_acronym"], observed=True)
        .agg(
            registration_cells=("ecephys_unit_id", "size"),
            uncensored_cells=("uncensored", "sum"),
            primary_cells=("interior_10deg", lambda x: int((x & source.loc[x.index, "uncensored"] & source.loc[x.index, "finite_positive_area"]).sum())),
            censored_fraction=("axis_censored", "mean"),
        )
        .reset_index()
    )
    return selected, audit


def balanced_kernel_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    coordinates: tuple[str, str],
    bandwidth: float,
    target: str,
) -> np.ndarray:
    """Equal-animal Gaussian smoother, fit separately within each visual area."""
    result = np.full(len(test), np.nan)
    for area in AREA_ORDER:
        train_area = train.loc[train.ecephys_structure_acronym.eq(area)]
        test_mask = test.ecephys_structure_acronym.eq(area).to_numpy()
        if not test_mask.any() or train_area.empty:
            continue
        x_train = train_area[list(coordinates)].to_numpy(float)
        x_test = test.loc[test_mask, list(coordinates)].to_numpy(float)
        center = np.nanmedian(x_train, axis=0)
        scale = np.nanquantile(x_train, .75, axis=0) - np.nanquantile(x_train, .25, axis=0)
        scale = np.where(scale > 1e-9, scale, np.nanstd(x_train, axis=0))
        scale = np.where(scale > 1e-9, scale, 1.0)
        x_test = (x_test - center) / scale
        animal_predictions = []
        for session_id, animal in train_area.groupby("session_id", observed=True):
            x_animal = (animal[list(coordinates)].to_numpy(float) - center) / scale
            distance2 = cdist(x_test, x_animal, metric="sqeuclidean")
            # Subtracting the row minimum prevents numerical underflow while
            # preserving the within-animal local weighting.
            weights = np.exp(-.5 * (distance2 - distance2.min(axis=1, keepdims=True)) / bandwidth**2)
            values = animal[target].to_numpy(float)
            animal_predictions.append((weights @ values) / weights.sum(axis=1))
        result[test_mask] = np.nanmean(np.vstack(animal_predictions), axis=0)
    return result


def score_groups(test: pd.DataFrame, prediction: np.ndarray, system: str, outer_session: int, bandwidth: float) -> list[dict]:
    rows = []
    local = test.copy()
    local["prediction"] = prediction
    for area, group in local.groupby("ecephys_structure_acronym", observed=True):
        if len(group) < MIN_GROUP_CELLS:
            continue
        observed = group.centered_log2_rf_area.to_numpy(float)
        predicted = group.prediction.to_numpy(float)
        rho = spearmanr(observed, predicted).statistic if np.nanstd(predicted) > 1e-12 else np.nan
        baseline_mae = float(np.median(np.abs(observed)))
        mae = float(np.median(np.abs(observed - predicted)))
        rows.append({
            "session_id": int(outer_session),
            "area": area,
            "system": system,
            "bandwidth_iqr_units": bandwidth,
            "cells": len(group),
            "spearman_rho": rho,
            "shape_mae_log2_octaves": mae,
            "constant_shape_mae_log2_octaves": baseline_mae,
            "shape_mae_improvement_log2_octaves": baseline_mae - mae,
        })
    return rows


def select_bandwidth(train: pd.DataFrame, system: str) -> tuple[float, list[dict]]:
    coordinates = SYSTEMS[system]
    rows = []
    sessions = sorted(train.session_id.unique())
    for bandwidth in BANDWIDTHS:
        scores = []
        for session_id in sessions:
            inner_test = train.loc[train.session_id.eq(session_id)]
            inner_train = train.loc[~train.session_id.eq(session_id)]
            prediction = balanced_kernel_predict(
                inner_train, inner_test, coordinates, bandwidth, "centered_log2_rf_area"
            )
            scores.extend(score_groups(inner_test, prediction, system, int(session_id), bandwidth))
        valid = pd.DataFrame(scores).spearman_rho.dropna()
        rows.append({
            "system": system,
            "bandwidth_iqr_units": bandwidth,
            "inner_animal_area_groups": len(scores),
            "inner_valid_correlations": len(valid),
            "inner_median_spearman_rho": float(valid.median()) if len(valid) else np.nan,
        })
    table = pd.DataFrame(rows).sort_values(
        ["inner_median_spearman_rho", "bandwidth_iqr_units"], ascending=[False, True]
    )
    return float(table.iloc[0].bandwidth_iqr_units), rows


def nested_animal_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows, selection_rows, prediction_rows = [], [], []
    for outer_session in sorted(data.session_id.unique()):
        train = data.loc[~data.session_id.eq(outer_session)].copy()
        test = data.loc[data.session_id.eq(outer_session)].copy()
        for system, coordinates in SYSTEMS.items():
            bandwidth, selection = select_bandwidth(train, system)
            for row in selection:
                selection_rows.append({"outer_session_id": int(outer_session), **row})
            shape_prediction = balanced_kernel_predict(
                train, test, coordinates, bandwidth, "centered_log2_rf_area"
            )
            absolute_prediction = balanced_kernel_predict(
                train, test, coordinates, bandwidth, "log2_rf_area_deg2"
            )
            score_rows.extend(score_groups(test, shape_prediction, system, int(outer_session), bandwidth))
            train_area_median = train.groupby("ecephys_structure_acronym", observed=True).log2_rf_area_deg2.median()
            for position, (_, row) in enumerate(test.iterrows()):
                area_baseline = float(train_area_median.loc[row.ecephys_structure_acronym])
                prediction_rows.append({
                    "session_id": int(outer_session),
                    "ecephys_unit_id": int(row.ecephys_unit_id),
                    "ecephys_probe_id": int(row.ecephys_probe_id),
                    "area": row.ecephys_structure_acronym,
                    "system": system,
                    "bandwidth_iqr_units": bandwidth,
                    "observed_log2_rf_area_deg2": row.log2_rf_area_deg2,
                    "observed_centered_log2_rf_area": row.centered_log2_rf_area,
                    "predicted_centered_log2_rf_area": shape_prediction[position],
                    "predicted_absolute_log2_rf_area_deg2": absolute_prediction[position],
                    "area_only_absolute_prediction_log2": area_baseline,
                    "ccf_ml_mm": row.ccf_ml_mm,
                    "ccf_ap_mm": row.ccf_ap_mm,
                    "common_template_x_px": row.common_template_x_px,
                    "common_template_y_px": row.common_template_y_px,
                    "common_azimuth_deg": row.common_azimuth_deg,
                    "common_elevation_deg": row.common_elevation_deg,
                })
        print(f"RF-size outer animal complete: {outer_session}", flush=True)
    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    absolute = predictions.groupby(["session_id", "area", "system"], observed=True).apply(
        lambda x: pd.Series({
            "absolute_mae_log2_octaves": np.median(np.abs(x.observed_log2_rf_area_deg2 - x.predicted_absolute_log2_rf_area_deg2)),
            "area_only_absolute_mae_log2_octaves": np.median(np.abs(x.observed_log2_rf_area_deg2 - x.area_only_absolute_prediction_log2)),
        }), include_groups=False
    ).reset_index()
    absolute["absolute_mae_improvement_log2_octaves"] = (
        absolute.area_only_absolute_mae_log2_octaves - absolute.absolute_mae_log2_octaves
    )
    scores = scores.merge(absolute, on=["session_id", "area", "system"], how="left", validate="one_to_one")
    return scores, pd.DataFrame(selection_rows), predictions


def balanced_whole_map_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    coordinates: tuple[str, str],
    bandwidth: float,
) -> np.ndarray:
    """Equal-animal smoother across the complete V1/HVA map."""
    x_train = train[list(coordinates)].to_numpy(float)
    x_test = test[list(coordinates)].to_numpy(float)
    center = np.nanmedian(x_train, axis=0)
    scale = np.nanquantile(x_train, .75, axis=0) - np.nanquantile(x_train, .25, axis=0)
    scale = np.where(scale > 1e-9, scale, np.nanstd(x_train, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    x_test = (x_test - center) / scale
    predictions = []
    for _, animal in train.groupby("session_id", observed=True):
        x_animal = (animal[list(coordinates)].to_numpy(float) - center) / scale
        distance2 = cdist(x_test, x_animal, metric="sqeuclidean")
        weights = np.exp(-.5 * (distance2 - distance2.min(axis=1, keepdims=True)) / bandwidth**2)
        predictions.append((weights @ animal.session_centered_log2_rf_area.to_numpy(float)) / weights.sum(axis=1))
    return np.mean(np.vstack(predictions), axis=0)


def whole_map_score(test: pd.DataFrame, prediction: np.ndarray) -> tuple[float, float, float]:
    observed = test.session_centered_log2_rf_area.to_numpy(float)
    rho = float(spearmanr(observed, prediction).statistic)
    mae = float(np.median(np.abs(observed - prediction)))
    constant_mae = float(np.median(np.abs(observed)))
    return rho, mae, constant_mae


def select_whole_map_bandwidth(train: pd.DataFrame, system: str) -> tuple[float, list[dict]]:
    rows = []
    for bandwidth in BANDWIDTHS:
        rhos = []
        for session_id in sorted(train.session_id.unique()):
            inner_test = train.loc[train.session_id.eq(session_id)]
            inner_train = train.loc[~train.session_id.eq(session_id)]
            prediction = balanced_whole_map_predict(inner_train, inner_test, SYSTEMS[system], bandwidth)
            rhos.append(whole_map_score(inner_test, prediction)[0])
        rows.append({
            "system": system, "bandwidth_iqr_units": bandwidth,
            "inner_animals": len(rhos), "inner_median_spearman_rho": float(np.median(rhos)),
        })
    table = pd.DataFrame(rows).sort_values(
        ["inner_median_spearman_rho", "bandwidth_iqr_units"], ascending=[False, True]
    )
    return float(table.iloc[0].bandwidth_iqr_units), rows


def nested_whole_map_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, selection_rows = [], []
    for outer_session in sorted(data.session_id.unique()):
        train = data.loc[~data.session_id.eq(outer_session)]
        test = data.loc[data.session_id.eq(outer_session)]
        observed = test.session_centered_log2_rf_area.to_numpy(float)
        area_means = train.groupby("ecephys_structure_acronym", observed=True).session_centered_log2_rf_area.mean()
        area_prediction = test.ecephys_structure_acronym.map(area_means).to_numpy(float)
        area_rho, area_mae, constant_mae = whole_map_score(test, area_prediction)
        rows.append({
            "session_id": int(outer_session), "system": "area_only", "bandwidth_iqr_units": np.nan,
            "cells": len(test), "spearman_rho": area_rho, "mae_log2_octaves": area_mae,
            "constant_mae_log2_octaves": constant_mae, "mae_improvement_vs_constant_log2": constant_mae - area_mae,
            "area_only_mae_log2_octaves": area_mae, "mae_improvement_vs_area_only_log2": 0.0,
        })
        for system in SYSTEMS:
            bandwidth, selection = select_whole_map_bandwidth(train, system)
            for row in selection:
                selection_rows.append({"outer_session_id": int(outer_session), **row})
            prediction = balanced_whole_map_predict(train, test, SYSTEMS[system], bandwidth)
            rho, mae, constant_mae = whole_map_score(test, prediction)
            rows.append({
                "session_id": int(outer_session), "system": system, "bandwidth_iqr_units": bandwidth,
                "cells": len(test), "spearman_rho": rho, "mae_log2_octaves": mae,
                "constant_mae_log2_octaves": constant_mae, "mae_improvement_vs_constant_log2": constant_mae - mae,
                "area_only_mae_log2_octaves": area_mae, "mae_improvement_vs_area_only_log2": area_mae - mae,
            })
    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def select_cases(scores: pd.DataFrame) -> pd.DataFrame:
    wide = scores.pivot(index=["session_id", "area"], columns="system", values="spearman_rho").reset_index()
    wide = wide.dropna(subset=["ccf_anatomy", "registered_map"]).copy()
    wide["registered_minus_ccf_rho"] = wide.registered_map - wide.ccf_anatomy
    median_delta = wide.registered_minus_ccf_rho.median()
    choices = [
        ("largest registration gain", wide.registered_minus_ccf_rho.idxmax(), "maximum registered-map minus CCF Spearman rho"),
        ("typical registration effect", (wide.registered_minus_ccf_rho - median_delta).abs().idxmin(), "closest to cohort median registered-map minus CCF rho"),
        ("registration failure", wide.registered_minus_ccf_rho.idxmin(), "minimum registered-map minus CCF Spearman rho"),
    ]
    rows = []
    for role, index, criterion in choices:
        row = wide.loc[index]
        rows.append({
            "session_id": int(row.session_id), "area": row.area, "selection_role": role,
            "criterion": criterion, "criterion_value": row.registered_minus_ccf_rho,
            "ccf_spearman_rho": row.ccf_anatomy, "registered_spearman_rho": row.registered_map,
            "observed_rf_spearman_rho": row.observed_rf,
            "provenance": "algorithmic selection from nested animal-held-out primary scores",
        })
    return pd.DataFrame(rows)


def add_boundary(axis: plt.Axes, template: np.lib.npyio.NpzFile) -> None:
    boundary = template["mean_field_sign_boundary"].astype(float)
    axis.contour(boundary, levels=[.5], colors="#222222", linewidths=.65, alpha=.8)


def render_summary(data: pd.DataFrame, scores: pd.DataFrame, audit: pd.DataFrame, output: Path) -> None:
    template = np.load(TEMPLATE)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    axis = axes[0, 0]
    image = axis.scatter(
        data.common_template_x_px, data.common_template_y_px,
        c=data.centered_log2_rf_area, s=10, alpha=.32, cmap="coolwarm", vmin=-1.5, vmax=1.5,
        linewidths=0,
    )
    add_boundary(axis, template)
    axis.set(xlim=(0, 430), ylim=(410, 15), aspect="equal", xlabel="Zhuang common x (px)", ylabel="Zhuang common y (px; high→low)",
             title="Interior RF sizes in registered map coordinates")
    bar = fig.colorbar(image, ax=axis, fraction=.045, pad=.02)
    bar.set_label("log₂ RF area minus animal-area median (octaves)")

    valid = scores.dropna(subset=["spearman_rho"])
    axis = axes[0, 1]
    rng = np.random.default_rng(20260816)
    for i, system in enumerate(SYSTEMS):
        local = valid.loc[valid.system.eq(system), "spearman_rho"]
        axis.scatter(i + rng.uniform(-.14, .14, len(local)), local, s=18, alpha=.5, color=SYSTEM_COLORS[system], linewidths=0)
        axis.plot([i-.22, i+.22], [local.median()]*2, color="#111111", lw=2.2)
    axis.axhline(0, color="#777777", lw=.8, ls="--")
    axis.set(xticks=range(len(SYSTEMS)), xticklabels=[SYSTEM_LABELS[x] for x in SYSTEMS], ylabel="Held-out within-area Spearman rho",
             title="Cross-animal RF-size surface shape alignment")
    axis.tick_params(axis="x", rotation=18)

    axis = axes[0, 2]
    for i, system in enumerate(SYSTEMS):
        local = scores.loc[scores.system.eq(system), "shape_mae_improvement_log2_octaves"]
        axis.scatter(i + rng.uniform(-.14, .14, len(local)), local, s=18, alpha=.5, color=SYSTEM_COLORS[system], linewidths=0)
        axis.plot([i-.22, i+.22], [local.median()]*2, color="#111111", lw=2.2)
    axis.axhline(0, color="#777777", lw=.8, ls="--")
    axis.set(xticks=range(len(SYSTEMS)), xticklabels=[SYSTEM_LABELS[x] for x in SYSTEMS], ylabel="MAE gain over within-area constant (log₂ octaves)",
             title="Does map position improve shape prediction?")
    axis.tick_params(axis="x", rotation=18)

    axis = axes[1, 0]
    heat = scores.pivot_table(index="area", columns="system", values="spearman_rho", aggfunc="median").reindex(AREA_ORDER).reindex(columns=list(SYSTEMS))
    artist = axis.imshow(heat, cmap="coolwarm", vmin=-.35, vmax=.35, aspect="auto")
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            value = heat.iloc[r, c]
            axis.text(c, r, f"{value:+.2f}" if np.isfinite(value) else "—", ha="center", va="center", color="#111111", fontsize=10)
    axis.set(xticks=range(len(SYSTEMS)), xticklabels=[SYSTEM_LABELS[x] for x in SYSTEMS], yticks=range(len(AREA_ORDER)),
             yticklabels=[AREA_LABELS[x] for x in AREA_ORDER], title="Median held-out shape correlation by area")
    axis.tick_params(axis="x", rotation=18)
    fig.colorbar(artist, ax=axis, fraction=.045, pad=.02, label="Spearman rho")

    axis = axes[1, 1]
    paired = scores.pivot_table(index=["session_id", "area"], columns="system", values="spearman_rho").dropna(subset=["ccf_anatomy", "registered_map"])
    animal_delta = (paired.registered_map - paired.ccf_anatomy).groupby("session_id").median().sort_values()
    colors = ["#2f6b9a" if value >= 0 else "#c6c6c6" for value in animal_delta]
    axis.bar(np.arange(len(animal_delta)), animal_delta, color=colors, edgecolor="#444444", linewidth=.4)
    axis.axhline(0, color="#333333", lw=.8)
    axis.set(xticks=np.arange(len(animal_delta)), xticklabels=[str(x)[-3:] for x in animal_delta.index], xlabel="Animal/session suffix",
             ylabel="Median Δrho: registered map − CCF", title="Registration effect is heterogeneous by animal")
    axis.tick_params(axis="x", rotation=45)

    axis = axes[1, 2]
    area_audit = audit.groupby("ecephys_structure_acronym", observed=True).agg(
        registration_cells=("registration_cells", "sum"), primary_cells=("primary_cells", "sum")
    ).reindex(AREA_ORDER)
    x = np.arange(len(area_audit))
    axis.bar(x, area_audit.registration_cells, color="#dddddd", edgecolor="#777777", label="all registered RF centers")
    axis.bar(x, area_audit.primary_cells, color="#2f6b9a", edgecolor="#244e6d", label="uncensored + >10° edge")
    axis.set(xticks=x, xticklabels=[AREA_LABELS[a] for a in AREA_ORDER], ylabel="Cells", title="RF-size support after censoring/edge control")
    axis.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.13)
    fig.suptitle(
        "RF size as a function of cross-animal retinotopic-map position\n"
        "14 frozen animals · analytic-aperture half-max area · nested held-out-animal evaluation",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_cases(data: pd.DataFrame, predictions: pd.DataFrame, cases: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(14.5, 12), constrained_layout=True)
    for r, case in cases.reset_index(drop=True).iterrows():
        local = data.loc[data.session_id.eq(case.session_id) & data.ecephys_structure_acronym.eq(case.area)]
        pred = predictions.loc[
            predictions.session_id.eq(case.session_id) & predictions.area.eq(case.area) & predictions.system.eq("registered_map")
        ]
        limits = (-2, 2)
        axes[r, 0].scatter(local.ccf_ml_mm, local.ccf_ap_mm, c=local.centered_log2_rf_area, cmap="coolwarm", vmin=limits[0], vmax=limits[1], s=32, edgecolors="#333333", linewidths=.25)
        axes[r, 0].set(xlabel="CCF ML (mm)", ylabel="CCF AP (mm)", title=f"{case.selection_role}: {case.session_id} {AREA_LABELS[case.area]}\nraw anatomy")
        axes[r, 0].invert_xaxis(); axes[r, 0].invert_yaxis(); axes[r, 0].set_aspect("equal", adjustable="datalim")
        axes[r, 1].scatter(local.common_template_x_px, local.common_template_y_px, c=local.centered_log2_rf_area, cmap="coolwarm", vmin=limits[0], vmax=limits[1], s=32, edgecolors="#333333", linewidths=.25)
        axes[r, 1].set(xlabel="Zhuang common x (px)", ylabel="Zhuang common y (px)", title=f"same cells after anatomy→map warp\nΔrho={case.criterion_value:+.2f}")
        axes[r, 1].invert_yaxis(); axes[r, 1].set_aspect("equal", adjustable="datalim")
        axes[r, 2].scatter(pred.observed_centered_log2_rf_area, pred.predicted_centered_log2_rf_area, s=30, color="#2f6b9a", alpha=.75, edgecolors="white", linewidths=.35)
        lo = min(-2, pred.observed_centered_log2_rf_area.min(), pred.predicted_centered_log2_rf_area.min())
        hi = max(2, pred.observed_centered_log2_rf_area.max(), pred.predicted_centered_log2_rf_area.max())
        axes[r, 2].plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.8)
        axes[r, 2].set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="Observed centered log₂ area", ylabel="Other-animal map prediction", title=f"registered held-out prediction\nrho={case.registered_spearman_rho:+.2f}; CCF={case.ccf_spearman_rho:+.2f}")
        for axis in axes[r]:
            axis.grid(alpha=.15)
    fig.suptitle("Concrete held-out RF-size alignment cases (algorithmically selected)", fontsize=15)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_whole_map(scores: pd.DataFrame, output: Path) -> None:
    order = ["area_only", "ccf_anatomy", "registered_map", "observed_rf"]
    labels = {"area_only": "Area only", **SYSTEM_LABELS}
    colors = {"area_only": "#6f6f6f", **SYSTEM_COLORS}
    rng = np.random.default_rng(20260816)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for i, system in enumerate(order):
        local = scores.loc[scores.system.eq(system)]
        jitter = rng.uniform(-.11, .11, len(local))
        axes[0].scatter(i+jitter, local.spearman_rho, color=colors[system], s=35, alpha=.75, edgecolors="white", linewidths=.4)
        axes[0].plot([i-.2, i+.2], [local.spearman_rho.median()]*2, color="#111111", lw=2.2)
        axes[1].scatter(i+jitter, local.mae_improvement_vs_constant_log2, color=colors[system], s=35, alpha=.75, edgecolors="white", linewidths=.4)
        axes[1].plot([i-.2, i+.2], [local.mae_improvement_vs_constant_log2.median()]*2, color="#111111", lw=2.2)
    axes[0].axhline(0, color="#777777", ls="--", lw=.8)
    axes[0].set(xticks=range(len(order)), xticklabels=[labels[x] for x in order], ylabel="Held-out animal Spearman rho",
                title="Whole-map rank alignment")
    axes[1].axhline(0, color="#777777", ls="--", lw=.8)
    axes[1].set(xticks=range(len(order)), xticklabels=[labels[x] for x in order], ylabel="MAE gain over animal constant (log₂ octaves)",
                title="Whole-map predictive amplitude")
    wide = scores.pivot(index="session_id", columns="system", values="mae_log2_octaves")
    axes[2].scatter(wide.area_only, wide.registered_map, color="#2f6b9a", s=48, alpha=.8, edgecolors="white", linewidths=.5)
    lo = min(wide.area_only.min(), wide.registered_map.min())
    hi = max(wide.area_only.max(), wide.registered_map.max())
    axes[2].plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=.9)
    for session_id, row in wide.iterrows():
        axes[2].text(row.area_only+.006, row.registered_map, str(session_id)[-3:], fontsize=7, color="#333333")
    axes[2].set(xlabel="Area-only held-out MAE", ylabel="Registered-map held-out MAE", title="Does position add beyond area identity?", aspect="equal")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(alpha=.14)
    fig.suptitle(
        "Coarse RF-size alignment across the complete V1/HVA map\n"
        "target is log₂ RF area minus each animal's global median; area differences are retained",
        fontsize=14,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, audit = prepare()
    scores, selection, predictions = nested_animal_holdout(data)
    whole_map_scores, whole_map_selection = nested_whole_map_holdout(data)
    cases = select_cases(scores)
    data.to_csv(OUTPUT / "primary_uncensored_interior_rf_size_cells.csv.gz", index=False, compression="gzip")
    audit.to_csv(OUTPUT / "rf_size_support_audit.csv", index=False)
    scores.to_csv(OUTPUT / "nested_animal_held_out_rf_size_alignment.csv", index=False)
    selection.to_csv(OUTPUT / "nested_bandwidth_selection.csv", index=False)
    predictions.to_csv(OUTPUT / "held_out_cell_predictions.csv.gz", index=False, compression="gzip")
    cases.to_csv(OUTPUT / "selected_case_audit.csv", index=False)
    whole_map_scores.to_csv(OUTPUT / "nested_animal_held_out_whole_map_rf_size_alignment.csv", index=False)
    whole_map_selection.to_csv(OUTPUT / "nested_whole_map_bandwidth_selection.csv", index=False)
    render_summary(data, scores, audit, OUTPUT / "Figure_14animal_rf_size_map_alignment_summary.png")
    render_cases(data, predictions, cases, OUTPUT / "Figure_rf_size_alignment_selected_cases.png")
    render_whole_map(whole_map_scores, OUTPUT / "Figure_14animal_rf_size_whole_map_alignment.png")

    summary = scores.groupby("system", observed=True).agg(
        animal_area_groups=("spearman_rho", "size"),
        valid_correlations=("spearman_rho", "count"),
        median_spearman_rho=("spearman_rho", "median"),
        positive_correlation_fraction=("spearman_rho", lambda x: float((x.dropna() > 0).mean())),
        median_shape_mae_improvement_log2=("shape_mae_improvement_log2_octaves", "median"),
        median_absolute_mae_improvement_log2=("absolute_mae_improvement_log2_octaves", "median"),
    ).reset_index()
    summary.to_csv(OUTPUT / "rf_size_alignment_summary.csv", index=False)
    whole_map_summary = whole_map_scores.groupby("system", observed=True).agg(
        animals=("session_id", "size"), median_spearman_rho=("spearman_rho", "median"),
        median_mae_log2=("mae_log2_octaves", "median"),
        median_mae_gain_vs_constant_log2=("mae_improvement_vs_constant_log2", "median"),
        median_mae_gain_vs_area_only_log2=("mae_improvement_vs_area_only_log2", "median"),
    ).reset_index()
    whole_map_summary.to_csv(OUTPUT / "whole_map_rf_size_alignment_summary.csv", index=False)
    chart_contract = {
        "question": "Does RF size have a reproducible spatial structure after mapping each animal's anatomy into the common Zhuang frame?",
        "takeaway_tested": "Registered map coordinates should improve held-out-animal RF-size surface prediction relative to raw CCF anatomy.",
        "grain": "animal-area score; cells are observations within each held-out animal-area and are not treated as independent animals",
        "primary_metric": "within-animal-area Spearman correlation of centered log2 half-max RF area with an other-animal prediction",
        "palette": "blue registered map; gray CCF; gold observed-RF reference; color is paired with labels/position",
        "outputs": ["Figure_14animal_rf_size_map_alignment_summary.png", "Figure_rf_size_alignment_selected_cases.png", "Figure_14animal_rf_size_whole_map_alignment.png"],
    }
    (OUTPUT / "chart_contract.json").write_text(json.dumps(chart_contract, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "exploratory frozen-14-animal RF-size alignment extension",
        "input": {"path": str(INPUT), "sha256": sha256(INPUT)},
        "selection": {
            "spatial_model": "aperture", "unit_split": "evaluation", "axis_censored": False,
            "minimum_axis_edge_distance_deg": 10, "cells": len(data), "animals": int(data.session_id.nunique()),
            "animal_area_groups_with_at_least_8_cells": int(data.groupby(["session_id", "ecephys_structure_acronym"]).size().ge(MIN_GROUP_CELLS).sum()),
        },
        "metric": "2*pi*ln(2)*sigma_x*sigma_y, log2 transformed; shape centered within animal-area",
        "validation": "nested leave-one-animal-out bandwidth selection and outer held-out-animal evaluation",
        "systems": {key: list(value) for key, value in SYSTEMS.items()},
        "bandwidth_candidates_iqr_units": list(BANDWIDTHS),
        "summary": summary.to_dict("records"),
        "whole_map_summary": whole_map_summary.to_dict("records"),
        "limitations": [
            "RF-size fits near parameter bounds and centers within 10 degrees of stimulus edges are excluded.",
            "Within-animal-area centering tests reproducible surface shape, not recovery of absolute between-animal size offsets.",
            "Observed RF location is a descriptive upper reference because RF center and size come from the same fitted response surface.",
            "The 14-animal cohort remains exploratory; newly completed animals should provide confirmation.",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWhole-map alignment")
    print(whole_map_summary.to_string(index=False))
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()
