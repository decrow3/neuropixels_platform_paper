#!/usr/bin/env python3
"""Corpus registration and cross-animal transfer audit for improved Allen RFs.

All eligible animals use one fixed retinal convention (100 - Allen-native
azimuth) and the same Zhuang template handedness.  Validation holds out one
whole penetration, refits the remaining penetrations in that animal, and tests
whether an area residual learned only from the other animal helps prediction.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_session_cell_gradient_registration_evidence import oriented_probe_coordinate
from register_allen_session_to_zhuang import (
    AREA_COLORS,
    AREA_LABELS,
    build_template,
    fit_candidate,
    load_session,
    sample_template,
    target_rf,
    transform_ccf,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "artifacts/allen_multisession_rf_validation_v1/07_registration_readiness/rf_size_visual_anatomy_unit_support.csv"
UNITS = ROOT / "data/unit_table.csv"
TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_v1"
TARGET_AREAS = tuple(AREA_LABELS)
COMMON_CONVENTION = "100_minus_azimuth"
COMMON_REFLECTION = 1


def inventory(support: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    merged = support.merge(
        units[["ecephys_unit_id", "ecephys_probe_id", "specimen_id"]],
        on="ecephys_unit_id", how="left", validate="many_to_one",
    )
    rows = []
    for session_id, frame in merged.groupby("session_id", sort=True):
        usable = frame.loc[
            frame.ccf_available & frame.ecephys_structure_acronym.isin(TARGET_AREAS)
            & frame.ecephys_probe_id.notna()
        ]
        probes = usable[["ecephys_probe_id", "ecephys_structure_acronym"]].drop_duplicates()
        if len(usable) == 0:
            status = "excluded: no CCF-localized improved-RF cells"
        elif len(probes) < 3:
            status = "excluded: fewer than 3 localized penetrations"
        else:
            status = "eligible"
        rows.append({
            "session_id": int(session_id),
            "specimen_id": int(frame.specimen_id.dropna().iloc[0]),
            "improved_rf_cells": len(frame),
            "ccf_localized_target_area_cells": len(usable),
            "localized_penetrations": len(probes),
            "localized_areas": probes.ecephys_structure_acronym.nunique(),
            "status": status,
        })
    return pd.DataFrame(rows)


def fit_full_session(
    session_id: int, template: dict, support_path: Path, units_path: Path, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cells, landmarks = load_session(support_path, units_path, session_id)
    fit = fit_candidate(
        template, landmarks, 2.0, COMMON_CONVENTION, COMMON_REFLECTION, seed
    )
    return cells, landmarks, fit


def common_cell_table(cells: pd.DataFrame, fit: dict, template: dict) -> pd.DataFrame:
    result = cells.copy()
    ccf = result[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    xy = transform_ccf(ccf, fit["ccf_center"], fit["parameters"], fit["reflection"])
    prediction, _, _ = sample_template(template, xy)
    observed_native = result[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
    observed_common = target_rf(observed_native, fit["convention"])
    result["common_template_x_px"] = xy[:, 0]
    result["common_template_y_px"] = xy[:, 1]
    result["common_azimuth_deg"] = observed_common[:, 0]
    result["common_elevation_deg"] = observed_common[:, 1]
    result["template_azimuth_deg"] = prediction[:, 0]
    result["template_elevation_deg"] = prediction[:, 1]
    result["template_minus_observed_azimuth_deg"] = prediction[:, 0] - observed_common[:, 0]
    result["template_minus_observed_elevation_deg"] = prediction[:, 1] - observed_common[:, 1]
    return result


def gradient_table(
    session_id: int, cells: pd.DataFrame, landmarks: pd.DataFrame, fit: dict, template: dict
) -> pd.DataFrame:
    rows = []
    for probe_id, frame in cells.groupby("ecephys_probe_id", sort=True):
        t, direction = oriented_probe_coordinate(frame)
        span = float(np.ptp(t))
        rf_native = frame[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
        coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(len(t)), t]), rf_native, rcond=None
        )[0][1]
        observed_delta = coefficient * span
        if fit["convention"] == "100_minus_azimuth":
            observed_delta[0] *= -1.0
        landmark = landmarks.loc[landmarks.ecephys_probe_id.eq(probe_id)].iloc[0]
        center = np.array([landmark.ccf_ap_mm, landmark.ccf_ml_mm], dtype=float)
        endpoints = np.vstack([
            center - direction * span / 2.0,
            center + direction * span / 2.0,
        ])
        endpoint_xy = transform_ccf(
            endpoints, fit["ccf_center"], fit["parameters"], fit["reflection"]
        )
        predicted_endpoints, _, _ = sample_template(template, endpoint_xy)
        predicted_delta = predicted_endpoints[1] - predicted_endpoints[0]
        template_vector = endpoint_xy[1] - endpoint_xy[0]
        rows.append({
            "session_id": session_id,
            "ecephys_probe_id": int(probe_id),
            "area": landmark.ecephys_structure_acronym,
            "cells": len(frame),
            "ccf_span_mm": span,
            "template_span_px": float(np.linalg.norm(template_vector)),
            "template_direction_x": template_vector[0] / np.linalg.norm(template_vector),
            "template_direction_y": template_vector[1] / np.linalg.norm(template_vector),
            "observed_delta_azimuth_deg": observed_delta[0],
            "observed_delta_elevation_deg": observed_delta[1],
            "predicted_delta_azimuth_deg": predicted_delta[0],
            "predicted_delta_elevation_deg": predicted_delta[1],
            "gradient_vector_error_deg": float(np.linalg.norm(predicted_delta - observed_delta)),
            "azimuth_sign_match": bool(np.sign(predicted_delta[0]) == np.sign(observed_delta[0])),
            "elevation_sign_match": bool(np.sign(predicted_delta[1]) == np.sign(observed_delta[1])),
        })
    return pd.DataFrame(rows)


def landmark_table(session_id: int, landmarks: pd.DataFrame, fit: dict) -> pd.DataFrame:
    target = fit["target"]
    predicted = fit["predicted"]
    result = landmarks.copy()
    result.insert(0, "session_id", session_id)
    result["template_x_px"] = fit["xy"][:, 0]
    result["template_y_px"] = fit["xy"][:, 1]
    result["common_azimuth_deg"] = target[:, 0]
    result["common_elevation_deg"] = target[:, 1]
    result["template_azimuth_deg"] = predicted[:, 0]
    result["template_elevation_deg"] = predicted[:, 1]
    result["residual_azimuth_deg"] = predicted[:, 0] - target[:, 0]
    result["residual_elevation_deg"] = predicted[:, 1] - target[:, 1]
    result["rf_vector_error_deg"] = np.linalg.norm(predicted - target, axis=1)
    result["named_area_distance_px"] = fit["area_distances"]
    return result


def cross_animal_transfer(
    sessions: dict[int, dict], template: dict, seed: int
) -> pd.DataFrame:
    rows = []
    session_ids = sorted(sessions)
    for session_id in session_ids:
        item = sessions[session_id]
        landmarks = item["landmarks"]
        other_ids = [value for value in session_ids if value != session_id]
        other_landmarks = pd.concat([sessions[value]["landmark_table"] for value in other_ids])
        for held_index in range(len(landmarks)):
            held = landmarks.iloc[held_index]
            area = held.ecephys_structure_acronym
            residual_support = other_landmarks.loc[
                other_landmarks.ecephys_structure_acronym.eq(area),
                ["residual_azimuth_deg", "residual_elevation_deg"],
            ]
            if residual_support.empty:
                continue
            train = landmarks.drop(index=held_index).reset_index(drop=True)
            fold_fit = fit_candidate(
                template, train, 2.0, COMMON_CONVENTION, COMMON_REFLECTION,
                seed + session_id % 10000 + held_index,
            )
            held_ccf = held[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)[None, :]
            held_xy = transform_ccf(
                held_ccf, fold_fit["ccf_center"], fold_fit["parameters"], fold_fit["reflection"]
            )
            baseline, _, _ = sample_template(template, held_xy)
            observed = target_rf(
                held[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)[None, :],
                COMMON_CONVENTION,
            )[0]
            learned_residual = residual_support.mean(axis=0).to_numpy(float)
            transferred = baseline[0] - learned_residual
            rows.append({
                "held_out_session_id": session_id,
                "held_out_probe_id": int(held.ecephys_probe_id),
                "held_out_area": area,
                "training_session_ids": ";".join(map(str, other_ids)),
                "other_animal_residual_support_probes": len(residual_support),
                "observed_azimuth_deg": observed[0],
                "observed_elevation_deg": observed[1],
                "baseline_predicted_azimuth_deg": baseline[0, 0],
                "baseline_predicted_elevation_deg": baseline[0, 1],
                "transferred_predicted_azimuth_deg": transferred[0],
                "transferred_predicted_elevation_deg": transferred[1],
                "baseline_vector_error_deg": float(np.linalg.norm(baseline[0] - observed)),
                "transferred_vector_error_deg": float(np.linalg.norm(transferred - observed)),
                "transfer_improvement_deg": float(
                    np.linalg.norm(baseline[0] - observed) - np.linalg.norm(transferred - observed)
                ),
            })
    return pd.DataFrame(rows)


def render(
    inventory_table: pd.DataFrame,
    landmarks: pd.DataFrame,
    gradients: pd.DataFrame,
    transfer: pd.DataFrame,
    template: dict,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)

    axis = axes[0, 0]
    y = np.arange(len(inventory_table))
    axis.barh(y, inventory_table.improved_rf_cells, color="#d0d0d0", label="improved RF fits")
    axis.barh(y, inventory_table.ccf_localized_target_area_cells, color="#326fa8", label="CCF + target area")
    axis.set_yticks(y, inventory_table.session_id.astype(str))
    axis.set(xlabel="cells", ylabel="session", title="Corpus registration eligibility")
    axis.legend(fontsize=8)
    axis.grid(axis="x", color="#dddddd", linewidth=0.45)

    axis = axes[0, 1]
    axis.contour(template["boundary"].astype(float), levels=[0.5], colors="#777777", linewidths=0.55)
    markers = {755434585: "o", 798911424: "s"}
    for (session_id, area), frame in landmarks.groupby(["session_id", "ecephys_structure_acronym"]):
        axis.scatter(
            frame.template_x_px, frame.template_y_px, s=85,
            marker=markers.get(session_id, "o"), color=AREA_COLORS[area],
            edgecolors="white", linewidths=0.9,
            label=f"{str(session_id)[-3:]} {AREA_LABELS[area]}",
        )
    axis.set(
        xlim=(-0.5, template["boundary"].shape[1] - 0.5),
        ylim=(template["boundary"].shape[0] - 0.5, -0.5),
        xlabel="Zhuang common x (px)", ylabel="Zhuang common y (px; down +)",
        title="Independent animal fits in one common frame",
    )
    axis.set_aspect("equal")

    axis = axes[0, 2]
    for session_id, frame in landmarks.groupby("session_id"):
        axis.scatter(
            frame.residual_azimuth_deg, frame.residual_elevation_deg,
            s=80, marker=markers.get(session_id, "o"),
            c=[AREA_COLORS[a] for a in frame.ecephys_structure_acronym],
            edgecolors="#222222", linewidths=0.5, label=str(session_id),
        )
        for row in frame.itertuples():
            axis.text(row.residual_azimuth_deg + 0.6, row.residual_elevation_deg + 0.6,
                      AREA_LABELS[row.ecephys_structure_acronym], fontsize=7)
    axis.axhline(0, color="#999999", linewidth=0.7)
    axis.axvline(0, color="#999999", linewidth=0.7)
    axis.set(xlabel="template − observed azimuth (°)", ylabel="template − observed elevation (°)",
             title="Are area residuals reproducible across animals?")
    axis.grid(color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    index = np.arange(len(transfer))
    axis.scatter(index - 0.1, transfer.baseline_vector_error_deg, color="#777777", s=45, label="same-animal LOPO")
    axis.scatter(index + 0.1, transfer.transferred_vector_error_deg, color="#7b3294", s=45, label="+ other-animal area residual")
    for i, row in transfer.reset_index(drop=True).iterrows():
        axis.plot([i - 0.1, i + 0.1], [row.baseline_vector_error_deg, row.transferred_vector_error_deg], color="#bbbbbb", linewidth=0.7)
    labels = [f"{str(r.held_out_session_id)[-3:]}:{AREA_LABELS[r.held_out_area]}" for r in transfer.itertuples()]
    axis.set_xticks(index, labels, rotation=45, ha="right")
    axis.set(ylabel="held-out RF-vector error (°)", title="Cross-animal residual transfer")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    x = np.arange(len(gradients))
    axis.scatter(x - 0.1, np.hypot(gradients.observed_delta_azimuth_deg, gradients.observed_delta_elevation_deg),
                 color="#111111", s=45, label="observed cells")
    axis.scatter(x + 0.1, np.hypot(gradients.predicted_delta_azimuth_deg, gradients.predicted_delta_elevation_deg),
                 color="#4c78a8", s=45, label="registered Zhuang")
    gradient_labels = [f"{str(r.session_id)[-3:]}:{str(r.ecephys_probe_id)[-3:]}" for r in gradients.itertuples()]
    axis.set_xticks(x, gradient_labels, rotation=45, ha="right")
    axis.set(ylabel="end-to-end RF change magnitude (°)", title="Within-probe gradient magnitude")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    grouped = transfer.groupby("held_out_session_id").agg(
        baseline=("baseline_vector_error_deg", "mean"),
        transferred=("transferred_vector_error_deg", "mean"),
        mean_improvement=("transfer_improvement_deg", "mean"),
    )
    bx = np.arange(len(grouped))
    axis.bar(bx - 0.18, grouped.baseline, width=0.36, color="#999999", label="baseline")
    axis.bar(bx + 0.18, grouped.transferred, width=0.36, color="#7b3294", label="transferred")
    axis.set_xticks(bx, [str(value) for value in grouped.index])
    axis.set(xlabel="held-out animal/session", ylabel="mean held-out error (°)",
             title="Transfer summary by animal")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)

    figure.suptitle(
        "Cross-animal retinotopy registration from improved RF fits and CCF anatomy\n"
        "Fixed common azimuth convention · animal-specific anatomy-to-template affine · probe-level validation",
        fontsize=14,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--units", type=Path, default=UNITS)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(args.support.resolve(), low_memory=False)
    units = pd.read_csv(args.units.resolve(), low_memory=False)
    inventory_table = inventory(support, units)
    inventory_table.to_csv(output / "session_registration_inventory.csv", index=False)
    eligible = inventory_table.loc[inventory_table.status.eq("eligible"), "session_id"].tolist()
    template = build_template(args.template.resolve())

    sessions: dict[int, dict] = {}
    cell_tables, landmark_tables, gradient_tables = [], [], []
    for position, session_id in enumerate(eligible):
        cells, landmarks, fit = fit_full_session(
            session_id, template, args.support.resolve(), args.units.resolve(),
            args.seed + 100 * position,
        )
        cells_common = common_cell_table(cells, fit, template)
        landmarks_common = landmark_table(session_id, landmarks, fit)
        gradients = gradient_table(session_id, cells, landmarks, fit, template)
        sessions[session_id] = {
            "cells": cells, "landmarks": landmarks, "fit": fit,
            "landmark_table": landmarks_common,
        }
        cell_tables.append(cells_common)
        landmark_tables.append(landmarks_common)
        gradient_tables.append(gradients)

    all_cells = pd.concat(cell_tables, ignore_index=True)
    all_landmarks = pd.concat(landmark_tables, ignore_index=True)
    all_gradients = pd.concat(gradient_tables, ignore_index=True)
    with gzip.open(output / "registered_cells_common_zhuang_coordinates.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        all_cells.to_csv(stream, index=False)
    all_landmarks.to_csv(output / "registered_probe_landmarks.csv", index=False)
    all_gradients.to_csv(output / "probe_gradient_comparison.csv", index=False)

    transfer = cross_animal_transfer(sessions, template, args.seed + 1000)
    transfer.to_csv(output / "leave_one_probe_out_cross_animal_transfer.csv", index=False)
    figure_path = output / "Figure_cross_animal_registration_summary.png"
    render(inventory_table, all_landmarks, all_gradients, transfer, template, figure_path)

    paired = transfer.transfer_improvement_deg.to_numpy(float)
    manifest = {
        "checkpoint": "corpus cross-animal registration and transfer audit",
        "status": "exploratory; only two animals are anatomically eligible",
        "common_convention": COMMON_CONVENTION,
        "common_cortical_reflection": COMMON_REFLECTION,
        "eligible_sessions": eligible,
        "excluded_sessions": inventory_table.loc[~inventory_table.status.eq("eligible")].to_dict("records"),
        "counts": {
            "registered_animals": len(eligible),
            "registered_cells": len(all_cells),
            "registered_probes": len(all_landmarks),
            "gradient_probes": len(all_gradients),
            "cross_animal_transfer_folds": len(transfer),
        },
        "transfer": {
            "mean_baseline_error_deg": float(transfer.baseline_vector_error_deg.mean()),
            "mean_transferred_error_deg": float(transfer.transferred_vector_error_deg.mean()),
            "mean_improvement_deg": float(paired.mean()),
            "median_improvement_deg": float(np.median(paired)),
            "folds_improved": int(np.sum(paired > 0)),
        },
        "validation": "leave one whole probe out; fit remaining probes in same animal; residual correction comes only from other animal probes in the same area",
        "limitations": [
            "Only two of four improved-RF sessions have enough CCF-localized penetrations.",
            "With two animals, other-animal transfer is a pairwise replication test, not a learned population prior.",
            "The per-animal map is currently a global anatomy-constrained affine.",
            "Gradient transfer is not attempted because one directional derivative per area/animal does not identify a full 2D local gradient.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
