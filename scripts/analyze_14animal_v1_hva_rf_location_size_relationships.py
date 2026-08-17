#!/usr/bin/env python3
"""Fit separate V1 and pooled-HVA RF-location to RF-size relationships."""

from __future__ import annotations

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
BASE = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1/rf_size_map_alignment"
INPUT = BASE / "primary_uncensored_interior_rf_size_cells.csv.gz"
OUTPUT = BASE / "v1_vs_hva_relationships"
BANDWIDTHS = (0.20, 0.35, 0.55, 0.80, 1.15)
SYSTEMS = {
    "ccf_anatomy": ("ccf_ml_mm", "ccf_ap_mm"),
    "registered_map": ("common_template_x_px", "common_template_y_px"),
    "observed_rf": ("common_azimuth_deg", "common_elevation_deg"),
}
LABELS = {
    "ccf_anatomy": "Raw CCF anatomy",
    "registered_map": "Registered Zhuang map",
    "observed_rf": "Observed RF location",
    "area_only": "HVA area only",
}
COLORS = {"ccf_anatomy": "#888888", "registered_map": "#2f6b9a", "observed_rf": "#d18b2c", "area_only": "#555555"}


def prepare() -> pd.DataFrame:
    data = pd.read_csv(INPUT, low_memory=False)
    data["cortical_group"] = np.where(data.ecephys_structure_acronym.eq("VISp"), "V1", "HVA")
    group = ["session_id", "cortical_group"]
    data["animal_group_median_log2_area"] = data.groupby(group, observed=True).log2_rf_area_deg2.transform("median")
    data["group_centered_log2_rf_area"] = data.log2_rf_area_deg2 - data.animal_group_median_log2_area
    return data


def balanced_predict(train: pd.DataFrame, test: pd.DataFrame, coordinates: tuple[str, str], bandwidth: float, target: str) -> np.ndarray:
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
        predictions.append((weights @ animal[target].to_numpy(float)) / weights.sum(axis=1))
    return np.mean(np.vstack(predictions), axis=0)


def score(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    rho = float(spearmanr(observed, predicted).statistic) if np.std(predicted) > 1e-12 else np.nan
    mae = float(np.median(np.abs(observed - predicted)))
    constant = float(np.median(np.abs(observed)))
    return rho, mae, constant


def select_bandwidth(train: pd.DataFrame, coordinates: tuple[str, str], target: str) -> tuple[float, list[dict]]:
    rows = []
    for bandwidth in BANDWIDTHS:
        inner_rhos = []
        for session_id in sorted(train.session_id.unique()):
            inner_test = train.loc[train.session_id.eq(session_id)]
            inner_train = train.loc[~train.session_id.eq(session_id)]
            prediction = balanced_predict(inner_train, inner_test, coordinates, bandwidth, target)
            inner_rhos.append(score(inner_test[target].to_numpy(float), prediction)[0])
        rows.append({
            "bandwidth_iqr_units": bandwidth, "inner_animals": len(inner_rhos),
            "inner_median_spearman_rho": float(np.nanmedian(inner_rhos)),
        })
    table = pd.DataFrame(rows).sort_values(["inner_median_spearman_rho", "bandwidth_iqr_units"], ascending=[False, True])
    return float(table.iloc[0].bandwidth_iqr_units), rows


def nested_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows, selection_rows, prediction_rows = [], [], []
    estimands = {
        # Primary: removes each animal-area median, so V1/HVA and HVA-area
        # offsets cannot masquerade as an RF-location relationship.
        "within_area_location": "centered_log2_rf_area",
        # Secondary: keeps LM/AL/RL/AM mean differences within the HVA map.
        "total_group_map": "group_centered_log2_rf_area",
    }
    for cortical_group in ("V1", "HVA"):
        group_data = data.loc[data.cortical_group.eq(cortical_group)].copy()
        for outer_session in sorted(group_data.session_id.unique()):
            train = group_data.loc[~group_data.session_id.eq(outer_session)]
            test = group_data.loc[group_data.session_id.eq(outer_session)]
            for estimand, target in estimands.items():
                observed = test[target].to_numpy(float)
                if estimand == "total_group_map" and cortical_group == "HVA":
                    area_means = train.groupby("ecephys_structure_acronym", observed=True)[target].mean()
                    area_prediction = test.ecephys_structure_acronym.map(area_means).to_numpy(float)
                    rho, mae, constant = score(observed, area_prediction)
                    score_rows.append({
                        "session_id": int(outer_session), "cortical_group": cortical_group,
                        "estimand": estimand, "system": "area_only", "cells": len(test),
                        "bandwidth_iqr_units": np.nan, "spearman_rho": rho,
                        "mae_log2_octaves": mae, "constant_mae_log2_octaves": constant,
                        "mae_gain_vs_constant_log2": constant - mae,
                    })
                for system, coordinates in SYSTEMS.items():
                    bandwidth, history = select_bandwidth(train, coordinates, target)
                    for row in history:
                        selection_rows.append({
                            "outer_session_id": int(outer_session), "cortical_group": cortical_group,
                            "estimand": estimand, "system": system, **row,
                        })
                    prediction = balanced_predict(train, test, coordinates, bandwidth, target)
                    rho, mae, constant = score(observed, prediction)
                    score_rows.append({
                        "session_id": int(outer_session), "cortical_group": cortical_group,
                        "estimand": estimand, "system": system, "cells": len(test),
                        "bandwidth_iqr_units": bandwidth, "spearman_rho": rho,
                        "mae_log2_octaves": mae, "constant_mae_log2_octaves": constant,
                        "mae_gain_vs_constant_log2": constant - mae,
                    })
                    for position, (_, unit) in enumerate(test.iterrows()):
                        prediction_rows.append({
                            "session_id": int(outer_session), "ecephys_unit_id": int(unit.ecephys_unit_id),
                            "area": unit.ecephys_structure_acronym, "cortical_group": cortical_group,
                            "estimand": estimand, "system": system, "observed": observed[position],
                            "prediction": prediction[position], "rf_azimuth_deg": unit.common_azimuth_deg,
                            "rf_elevation_deg": unit.common_elevation_deg,
                        })
        print(f"separate RF-location models complete: {cortical_group}", flush=True)
    return pd.DataFrame(score_rows), pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows)


def render_primary(data: pd.DataFrame, scores: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    rng = np.random.default_rng(20260816)
    for row, cortical_group in enumerate(("V1", "HVA")):
        local_data = data.loc[data.cortical_group.eq(cortical_group)]
        # V1 is one area. For the HVA map, retain the LM/AL/RL/AM mean-size
        # differences and remove only the animal's overall HVA median.
        estimand = "within_area_location" if cortical_group == "V1" else "total_group_map"
        color_field = "centered_log2_rf_area" if cortical_group == "V1" else "group_centered_log2_rf_area"
        image = axes[row, 0].scatter(
            local_data.common_azimuth_deg, local_data.common_elevation_deg,
            c=local_data[color_field], cmap="coolwarm", vmin=-1.5, vmax=1.5,
            s=11, alpha=.38, linewidths=0,
        )
        axes[row, 0].set(xlabel="Observed RF azimuth (deg)", ylabel="Observed RF elevation (deg)", aspect="equal",
                         title=(f"{cortical_group}: RF-location/size observations"
                                if cortical_group == "V1" else
                                "HVA: observations with HVA-area means retained"))
        local_scores = scores.loc[
            scores.cortical_group.eq(cortical_group) & scores.estimand.eq(estimand)
        ]
        system_order = list(SYSTEMS) if cortical_group == "V1" else ["area_only", *SYSTEMS]
        for column, metric, ylabel, title in [
            (1, "spearman_rho", "Held-out animal Spearman rho", "Cross-animal shape alignment"),
            (2, "mae_gain_vs_constant_log2", "MAE gain over constant (log₂ octaves)", "Predictive amplitude"),
        ]:
            axis = axes[row, column]
            for i, system in enumerate(system_order):
                values = local_scores.loc[local_scores.system.eq(system), metric]
                axis.scatter(i + rng.uniform(-.11, .11, len(values)), values, s=35, alpha=.75,
                             color=COLORS[system], edgecolors="white", linewidths=.4)
                axis.plot([i-.2, i+.2], [values.median()]*2, color="#111111", lw=2.2)
            axis.axhline(0, color="#777777", ls="--", lw=.8)
            axis.set(xticks=range(len(system_order)), xticklabels=[LABELS[x] for x in system_order], ylabel=ylabel,
                     title=f"{cortical_group}: {title}")
            axis.tick_params(axis="x", rotation=18)
        for axis in axes[row]:
            axis.grid(alpha=.14)
    fig.colorbar(image, ax=axes[:, 0], fraction=.035, pad=.02,
                 label="log₂ area minus animal V1/HVA-group median")
    fig.suptitle(
        "RF location → RF size is tested independently in V1 and HVAs\n"
        "HVA-area mean differences retained · only each animal's overall V1/HVA offset removed",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_secondary(scores: pd.DataFrame, output: Path) -> None:
    local = scores.loc[scores.estimand.eq("total_group_map")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    order_by_group = {"V1": list(SYSTEMS), "HVA": ["area_only", *SYSTEMS]}
    rng = np.random.default_rng(20260816)
    for axis, cortical_group in zip(axes, ("V1", "HVA")):
        group = local.loc[local.cortical_group.eq(cortical_group)]
        order = order_by_group[cortical_group]
        for i, system in enumerate(order):
            values = group.loc[group.system.eq(system), "spearman_rho"]
            axis.scatter(i + rng.uniform(-.11, .11, len(values)), values, s=38, alpha=.75,
                         color=COLORS[system], edgecolors="white", linewidths=.4)
            axis.plot([i-.2, i+.2], [values.median()]*2, color="#111111", lw=2.2)
        axis.axhline(0, color="#777777", ls="--", lw=.8)
        axis.set(xticks=range(len(order)), xticklabels=[LABELS[x] for x in order], ylabel="Held-out animal Spearman rho",
                 title=f"{cortical_group}: total within-group RF-size map")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(alpha=.14)
    fig.suptitle(
        "Secondary view: V1 and HVA models remain separate, but HVA-area mean differences are retained",
        fontsize=14,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = prepare()
    scores, selection, predictions = nested_models(data)
    scores.to_csv(OUTPUT / "nested_held_out_v1_hva_rf_location_size_scores.csv", index=False)
    selection.to_csv(OUTPUT / "nested_bandwidth_selection.csv", index=False)
    predictions.to_csv(OUTPUT / "held_out_cell_predictions.csv.gz", index=False, compression="gzip")
    summary = scores.groupby(["estimand", "cortical_group", "system"], observed=True).agg(
        animals=("session_id", "size"), median_spearman_rho=("spearman_rho", "median"),
        positive_rho_fraction=("spearman_rho", lambda x: float((x > 0).mean())),
        median_mae_gain_vs_constant_log2=("mae_gain_vs_constant_log2", "median"),
    ).reset_index()
    summary.to_csv(OUTPUT / "v1_hva_rf_location_size_summary.csv", index=False)
    render_primary(data, scores, OUTPUT / "Figure_separate_V1_HVA_RF_location_size_relationships_area_means_retained.png")
    render_secondary(scores, OUTPUT / "Figure_separate_V1_HVA_total_size_maps.png")
    manifest = {
        "status": "exploratory correction: V1 and HVA RF-location/size relationships fit independently",
        "fine_scale_sensitivity_estimand": "within-animal-area centered log2 RF area; removes individual HVA area offsets",
        "reporting_primary": {
            "V1": "within_area_location (equivalent to V1-group centering because V1 is one area)",
            "HVA": "total_group_map (LM/AL/RL/AM mean-size differences retained; only animal-wide HVA median removed)",
        },
        "secondary_estimand": "within-animal V1-or-HVA centered log2 RF area; HVA area offsets retained",
        "validation": "nested leave-one-animal-out, animal-balanced Gaussian surface",
        "cells": {group: int((data.cortical_group == group).sum()) for group in ("V1", "HVA")},
        "animals": {group: int(data.loc[data.cortical_group.eq(group), "session_id"].nunique()) for group in ("V1", "HVA")},
        "summary": summary.to_dict("records"),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
