#!/usr/bin/env python3
"""Render auditable success, median, and failure cases for staged registration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from build_14animal_retinotopy_registration import SEED, TEMPLATE_PATH, make_landmarks
from fit_14animal_staged_visual_translation import (
    AREA_WEIGHT,
    CONVENTION,
    INPUT_CELLS,
    OUTPUT as STAGED_OUTPUT,
    REFLECTION,
    fit_translation_invariant_geometry,
)
from register_allen_session_to_zhuang import (
    AREA_COLORS,
    AREA_LABELS,
    build_template,
    fit_candidate,
    sample_template,
    target_rf,
    transform_ccf,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = STAGED_OUTPUT / "selected_case_geometry_diagnostics"
FIGURE = OUTPUT / "Figure_staged_translation_success_median_failure_cases.png"
SELECTION = OUTPUT / "selected_case_geometry_diagnostics.csv"
TRAINING_RESIDUALS = OUTPUT / "selected_case_training_probe_residuals.csv"


def inverse_warp_grid(template: dict, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(template["domain"].shape)
    template_xy = np.column_stack([columns.ravel(), rows.ravel()])
    centered = template_xy - fit["template_center"]
    ccf = centered @ np.linalg.inv(fit["matrix_px_per_mm"].T) + fit["ccf_center"]
    ap = ccf[:, 0].reshape(rows.shape)
    ml = ccf[:, 1].reshape(rows.shape)
    return ml, ap


def choose_cases(comparison: pd.DataFrame) -> pd.DataFrame:
    gain = comparison.staged_gain_vs_no_offset_deg
    median_gain = float(gain.median())
    chosen = [
        ("largest improvement", gain.idxmax(), "maximum staged offset gain"),
        ("median case", (gain - median_gain).abs().idxmin(), "closest to median staged offset gain"),
        ("largest failure", gain.idxmin(), "minimum staged offset gain"),
    ]
    rows = []
    for role, index, criterion in chosen:
        row = comparison.loc[index].to_dict()
        row.update({
            "selection_role": role,
            "selection_criterion": criterion,
            "selection_reference": "staged geometry with zero offset versus same frozen geometry plus nested-selected translation",
            "selection_provenance": "algorithmic selection over all 68 held-out penetrations",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_case(case: pd.Series, cells: pd.DataFrame, template: dict) -> dict:
    session_id = int(case.session_id)
    held_probe = int(case.held_out_probe_id)
    local_cells = cells.loc[cells.session_id.eq(session_id)].copy()
    landmarks = make_landmarks(local_cells)
    held_matches = np.flatnonzero(landmarks.ecephys_probe_id.to_numpy(int) == held_probe)
    if len(held_matches) != 1:
        raise RuntimeError(f"Expected one landmark for held probe {held_probe}, found {len(held_matches)}")
    held_index = int(held_matches[0])
    held = landmarks.iloc[held_index]
    train = landmarks.drop(index=held_index).reset_index(drop=True)
    session_position = list(sorted(cells.session_id.unique())).index(session_id)
    seed = SEED + 10000 + session_position * 100 + held_index
    start = fit_candidate(template, train, AREA_WEIGHT, CONVENTION, REFLECTION, seed)
    fit = fit_translation_invariant_geometry(train, template, start)

    held_ccf = held[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)[None, :]
    held_xy = transform_ccf(held_ccf, fit["ccf_center"], fit["parameters"], REFLECTION)
    held_template, _, _ = sample_template(template, held_xy)
    held_target = target_rf(
        held[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)[None, :], CONVENTION
    )[0]
    scale = float(case.selected_offset_scale)
    retained_offset = scale * fit["raw_visual_offset"]
    held_after = held_template[0] + retained_offset
    return {
        "case": case,
        "cells": local_cells,
        "landmarks": landmarks,
        "train": train,
        "held": held,
        "held_index": held_index,
        "fit": fit,
        "held_template": held_template[0],
        "held_target": held_target,
        "held_after": held_after,
        "retained_offset": retained_offset,
        "scale": scale,
    }


def anatomy_limits(cells: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    ml = cells.ccf_ml_mm.to_numpy(float)
    ap = cells.ccf_ap_mm.to_numpy(float)
    ml_padding = max(.12, .10 * np.ptp(ml))
    ap_padding = max(.12, .10 * np.ptp(ap))
    # High-to-low axes match the previously chosen coarse Zhuang convention.
    return (ml.max() + ml_padding, ml.min() - ml_padding), (ap.max() + ap_padding, ap.min() - ap_padding)


def plot_anatomy_field(
    axis: plt.Axes,
    case_data: dict,
    template: dict,
    field_name: str,
    cell_column: str,
    cmap: str,
    norm: Normalize,
    levels: np.ndarray,
) -> None:
    cells = case_data["cells"]
    held_probe = int(case_data["held"].ecephys_probe_id)
    fit = case_data["fit"]
    ml_grid, ap_grid = inverse_warp_grid(template, fit)
    field = template["field_arrays"][field_name]
    domain = template["domain"]
    masked = np.ma.masked_where(~domain, field)
    contours = axis.contour(
        ml_grid, ap_grid, masked, levels=levels, cmap=cmap, norm=norm,
        linewidths=.9, alpha=.82,
    )
    axis.clabel(contours, inline=True, fontsize=6, fmt="%g°")
    axis.contour(ml_grid, ap_grid, domain.astype(float), levels=[.5], colors="#333333", linewidths=1.0)

    train_cells = cells.loc[~cells.ecephys_probe_id.eq(held_probe)]
    held_cells = cells.loc[cells.ecephys_probe_id.eq(held_probe)]
    axis.scatter(
        train_cells.ccf_ml_mm, train_cells.ccf_ap_mm,
        c=train_cells[cell_column], cmap=cmap, norm=norm, s=9, alpha=.32,
        linewidths=0, rasterized=True,
    )
    axis.scatter(
        held_cells.ccf_ml_mm, held_cells.ccf_ap_mm,
        c=held_cells[cell_column], cmap=cmap, norm=norm, s=16, alpha=.82,
        edgecolors="#111111", linewidths=.45, rasterized=True,
    )

    landmarks = case_data["landmarks"]
    values = landmarks.rf_azimuth_deg if field_name == "azimuth_deg" else landmarks.rf_elevation_deg
    for landmark, value in zip(landmarks.itertuples(), values):
        is_held = int(landmark.ecephys_probe_id) == held_probe
        axis.scatter(
            landmark.ccf_ml_mm, landmark.ccf_ap_mm,
            c=[value], cmap=cmap, norm=norm, s=85 if is_held else 46,
            marker="*" if is_held else "o", edgecolors="#111111",
            linewidths=1.0 if is_held else .55, zorder=5,
        )
        axis.text(
            landmark.ccf_ml_mm, landmark.ccf_ap_mm,
            f" {AREA_LABELS[landmark.ecephys_structure_acronym]}",
            fontsize=6.5, va="bottom", ha="left", color="#111111", zorder=6,
        )
    xlim, ylim = anatomy_limits(cells)
    axis.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="CCF ML (mm)", ylabel="CCF AP (mm)")
    axis.grid(alpha=.12)


def plot_translation(
    axis: plt.Axes,
    case_data: dict,
    rf_xlim: tuple[float, float],
    rf_ylim: tuple[float, float],
) -> None:
    fit = case_data["fit"]
    predicted = fit["predicted_template"]
    observed = fit["target"]
    offset = case_data["retained_offset"]
    shifted = predicted + offset
    train = case_data["train"]
    colors = [AREA_COLORS[area] for area in train.ecephys_structure_acronym]

    for index in range(len(train)):
        axis.annotate(
            "", xy=observed[index], xytext=predicted[index],
            arrowprops={"arrowstyle": "->", "color": "#999999", "lw": .9, "alpha": .72},
        )
        axis.text(
            observed[index, 0] + .7, observed[index, 1] + .5,
            AREA_LABELS[train.iloc[index].ecephys_structure_acronym],
            fontsize=7, color=colors[index], weight="bold",
        )
    axis.scatter(predicted[:, 0], predicted[:, 1], facecolors="none", edgecolors=colors, s=58, linewidths=1.2, label="Template before offset")
    axis.scatter(shifted[:, 0], shifted[:, 1], c=colors, marker="D", s=34, alpha=.82, label="Template after offset")
    axis.scatter(observed[:, 0], observed[:, 1], c=colors, marker="x", s=52, linewidths=1.4, label="Observed training RF")

    held_template = case_data["held_template"]
    held_after = case_data["held_after"]
    held_target = case_data["held_target"]
    axis.scatter(*held_template, facecolors="none", edgecolors="#111111", marker="s", s=105, linewidths=1.5, label="Held prediction before")
    axis.scatter(*held_after, color="#111111", marker="D", s=58, label="Held prediction after")
    axis.scatter(*held_target, color="#f1c40f", edgecolors="#111111", marker="*", s=155, linewidths=.8, label="Held observed RF")
    axis.annotate("", xy=held_after, xytext=held_template, arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 2.0})

    axis.set(
        xlim=rf_xlim, ylim=rf_ylim,
        aspect="equal", xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)",
    )
    case = case_data["case"]
    axis.set_title(
        f"Gradient-fit residual translation: {case_data['scale']:.2g} × "
        f"({fit['raw_visual_offset'][0]:+.1f}°, {fit['raw_visual_offset'][1]:+.1f}°)\n"
        f"held error {case.staged_no_offset_error_deg:.1f}° → {case.staged_error_deg:.1f}°",
        fontsize=9,
    )
    axis.grid(alpha=.15)


def render(cases: list[dict], template: dict) -> None:
    all_cells = pd.concat([case["cells"] for case in cases], ignore_index=True)
    az_values = np.r_[
        template["field_arrays"]["azimuth_deg"][template["domain"]],
        all_cells.visual_azimuth_deg.to_numpy(float),
    ]
    el_values = np.r_[
        template["field_arrays"]["altitude_deg"][template["domain"]],
        all_cells.visual_elevation_deg.to_numpy(float),
    ]
    az_limits = np.nanpercentile(az_values, [1, 99])
    el_abs = float(np.nanpercentile(np.abs(el_values), 99))
    az_norm = Normalize(*az_limits)
    el_norm = Normalize(-el_abs, el_abs)
    az_levels = np.linspace(np.ceil(az_limits[0] / 20) * 20, np.floor(az_limits[1] / 20) * 20, 6)
    el_levels = np.linspace(np.ceil(-el_abs / 20) * 20, np.floor(el_abs / 20) * 20, 6)
    rf_points = []
    for case_data in cases:
        fit = case_data["fit"]
        rf_points.extend([
            fit["predicted_template"], fit["target"],
            fit["predicted_template"] + case_data["retained_offset"],
            case_data["held_template"][None, :], case_data["held_after"][None, :],
            case_data["held_target"][None, :],
        ])
    rf_points = np.vstack(rf_points)
    rf_xpad = max(5.0, .08 * np.ptp(rf_points[:, 0]))
    rf_ypad = max(5.0, .08 * np.ptp(rf_points[:, 1]))
    rf_xlim = (rf_points[:, 0].min() - rf_xpad, rf_points[:, 0].max() + rf_xpad)
    rf_ylim = (rf_points[:, 1].min() - rf_ypad, rf_points[:, 1].max() + rf_ypad)

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 14.3), constrained_layout=True)
    for row, case_data in enumerate(cases):
        case = case_data["case"]
        plot_anatomy_field(axes[row, 0], case_data, template, "azimuth_deg", "visual_azimuth_deg", "viridis", az_norm, az_levels)
        plot_anatomy_field(axes[row, 1], case_data, template, "altitude_deg", "visual_elevation_deg", "coolwarm", el_norm, el_levels)
        plot_translation(axes[row, 2], case_data, rf_xlim, rf_ylim)
        prefix = (
            f"{case.selection_role}: session {int(case.session_id)}, "
            f"{AREA_LABELS[case.held_out_area]} held probe {int(case.held_out_probe_id)}"
        )
        axes[row, 0].set_title(prefix + "\nGeometry-first warped azimuth", fontsize=9)
        axes[row, 1].set_title(prefix + "\nGeometry-first warped elevation", fontsize=9)

    az_map = plt.cm.ScalarMappable(norm=az_norm, cmap="viridis")
    el_map = plt.cm.ScalarMappable(norm=el_norm, cmap="coolwarm")
    fig.colorbar(az_map, ax=axes[:, 0], location="bottom", shrink=.72, pad=.045, label="Azimuth (deg): contours and cells use the same scale")
    fig.colorbar(el_map, ax=axes[:, 1], location="bottom", shrink=.72, pad=.045, label="Elevation (deg): contours and cells use the same scale")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels, frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        "Staged registration case audit: anatomy is fixed, Zhuang contours warp, then one RF translation is applied\n"
        "Faint points are cells; circles are training probe medians; outlined stars are held-out probe medians",
        fontsize=14,
    )
    fig.savefig(FIGURE, dpi=210, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(STAGED_OUTPUT / "staged_no_offset_joint_fold_comparison.csv")
    selected = choose_cases(comparison)
    cells = pd.read_csv(INPUT_CELLS, low_memory=False)
    template = build_template(TEMPLATE_PATH)
    cases = []
    diagnostic_rows = []
    residual_rows = []
    for case in selected.itertuples(index=False):
        data = reconstruct_case(pd.Series(case._asdict()), cells, template)
        cases.append(data)
        fit = data["fit"]
        diagnostic_rows.append({
            **pd.Series(case._asdict()).to_dict(),
            "training_landmarks": len(data["train"]),
            "training_pair_rmse_deg": fit["pair_rmse_deg"],
            "mean_training_area_distance_px": fit["mean_area_distance_px"],
            "raw_offset_azimuth_deg": fit["raw_visual_offset"][0],
            "raw_offset_elevation_deg": fit["raw_visual_offset"][1],
            "retained_offset_azimuth_deg": data["retained_offset"][0],
            "retained_offset_elevation_deg": data["retained_offset"][1],
            "retained_offset_magnitude_deg": float(np.linalg.norm(data["retained_offset"])),
            "held_prediction_before_azimuth_deg": data["held_template"][0],
            "held_prediction_before_elevation_deg": data["held_template"][1],
            "held_prediction_after_azimuth_deg": data["held_after"][0],
            "held_prediction_after_elevation_deg": data["held_after"][1],
            "held_observed_azimuth_deg": data["held_target"][0],
            "held_observed_elevation_deg": data["held_target"][1],
        })
        for index, landmark in data["train"].reset_index(drop=True).iterrows():
            predicted = fit["predicted_template"][index]
            observed = fit["target"][index]
            residual_rows.append({
                "selection_role": case.selection_role,
                "session_id": int(case.session_id),
                "held_out_probe_id": int(case.held_out_probe_id),
                "training_probe_id": int(landmark.ecephys_probe_id),
                "training_area": landmark.ecephys_structure_acronym,
                "training_area_label": AREA_LABELS[landmark.ecephys_structure_acronym],
                "template_azimuth_deg": predicted[0],
                "template_elevation_deg": predicted[1],
                "observed_azimuth_deg": observed[0],
                "observed_elevation_deg": observed[1],
                "residual_azimuth_deg": observed[0] - predicted[0],
                "residual_elevation_deg": observed[1] - predicted[1],
            })
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(SELECTION, index=False)
    pd.DataFrame(residual_rows).to_csv(TRAINING_RESIDUALS, index=False)
    render(cases, template)
    manifest = {
        "status": "exploratory concrete-case audit",
        "selection": "maximum, closest-to-median, and minimum staged offset gain over 68 held-out probes",
        "figure": str(FIGURE),
        "selection_table": str(SELECTION),
        "training_residual_table": str(TRAINING_RESIDUALS),
        "cases": diagnostics[[
            "selection_role", "session_id", "held_out_probe_id", "held_out_area",
            "staged_no_offset_error_deg", "staged_error_deg",
        ]].to_dict(orient="records"),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
