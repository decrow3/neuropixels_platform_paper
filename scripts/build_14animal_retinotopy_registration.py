#!/usr/bin/env python3
"""Frozen 14-animal retinotopy registration from production RF fits.

The fit unit is a probe/area median. Cells refine those medians and provide
within-probe gradient diagnostics, but never count as independent folds.
Shared residual corrections are evaluated with an outer held-out animal and
an inner animal-held-out model-selection loop.
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
from scipy import ndimage
from scipy.optimize import minimize

from build_cross_animal_retinotopy_registration import (
    common_cell_table,
    gradient_table,
    landmark_table,
)
from register_allen_session_to_zhuang import (
    AREA_COLORS,
    AREA_LABELS,
    affine,
    build_template,
    fit_candidate,
    sample_template,
    target_rf,
    transform_ccf,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "artifacts/allen_full_rf_production_v1"
UNIT_TABLE = ROOT / "data/unit_table.csv"
TEMPLATE_PATH = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1"
SESSIONS = (
    715093703, 755434585, 756029989, 757216464, 768515987, 774875821,
    778240327, 781842082, 786091066, 794812542, 798911424, 829720705,
    831882777, 847657808,
)
TARGET_AREAS = tuple(AREA_LABELS)
SEED = 20260816


def production_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    units = pd.read_csv(
        UNIT_TABLE,
        usecols=[
            "ecephys_unit_id", "ecephys_probe_id", "ecephys_session_id", "specimen_id",
            "anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate",
            "ecephys_structure_acronym",
        ],
        low_memory=False,
    )
    frames = []
    audit_rows = []
    for session_id in SESSIONS:
        path = PRODUCTION / "02_session_fits" / f"session_{session_id}" / "unit_geometry_fits.csv"
        fits = pd.read_csv(path, low_memory=False)
        selected = fits.loc[
            fits.spatial_model.eq("aperture") & fits.unit_split.eq("evaluation")
        ].merge(units, on="ecephys_unit_id", how="left", suffixes=("", "_unit"), validate="one_to_one")
        selected = selected.loc[
            selected.ecephys_structure_acronym.isin(TARGET_AREAS)
            & selected.anterior_posterior_ccf_coordinate.notna()
            & selected.left_right_ccf_coordinate.notna()
            & np.isfinite(selected.axis_center_x_deg)
            & np.isfinite(selected.axis_center_y_deg)
            & np.isfinite(selected.axis_test_deviance)
        ].copy()
        selected["ccf_ap_mm"] = selected.anterior_posterior_ccf_coordinate / 1000.0
        selected["ccf_ml_mm"] = selected.left_right_ccf_coordinate / 1000.0
        selected["visual_azimuth_deg"] = selected.axis_center_x_deg + 50.0
        selected["visual_elevation_deg"] = selected.axis_center_y_deg + 10.0
        selected["session_id"] = session_id
        probes = selected[["ecephys_probe_id", "ecephys_structure_acronym"]].drop_duplicates()
        audit_rows.append({
            "session_id": session_id,
            "specimen_id": int(selected.specimen_id.iloc[0]),
            "selected_cells": len(selected),
            "probe_area_landmarks": len(probes),
            "areas": probes.ecephys_structure_acronym.nunique(),
            "censored_fraction": float(selected.axis_censored.mean()),
            "median_axis_test_deviance": float(selected.axis_test_deviance.median()),
        })
        frames.append(selected)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit_rows)


def make_landmarks(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby(["ecephys_probe_id", "ecephys_structure_acronym"], as_index=False)
        .agg(
            units=("ecephys_unit_id", "size"),
            ccf_ap_mm=("ccf_ap_mm", "median"),
            ccf_ml_mm=("ccf_ml_mm", "median"),
            rf_azimuth_deg=("visual_azimuth_deg", "median"),
            rf_azimuth_iqr_deg=("visual_azimuth_deg", lambda x: x.quantile(.75) - x.quantile(.25)),
            rf_elevation_deg=("visual_elevation_deg", "median"),
            rf_elevation_iqr_deg=("visual_elevation_deg", lambda x: x.quantile(.75) - x.quantile(.25)),
        )
        .sort_values("ecephys_probe_id")
        .reset_index(drop=True)
    )


def candidate_row(session_id: int, fit: dict) -> dict:
    return {
        "session_id": session_id,
        "azimuth_convention": fit["convention"],
        "cortical_reflection": fit["reflection"],
        "objective": fit["objective"],
        "penetration_rf_rmse_deg": fit["retinal_rmse_deg"],
        "penetration_median_vector_error_deg": fit["retinal_median_vector_error_deg"],
        "mean_named_area_distance_px": fit["mean_area_distance_px"],
        "penetrations_in_named_area": fit["landmarks_in_named_area"],
    }


def local_refit(
    landmarks: pd.DataFrame,
    template: dict,
    convention: str,
    reflection: int,
    full_fit: dict,
) -> dict:
    ccf = landmarks[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    observed = landmarks[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    target = target_rf(observed, convention)
    areas = landmarks.ecephys_structure_acronym.tolist()
    ccf_center = ccf.mean(axis=0)
    height, width = template["domain"].shape
    old_center = full_fit["ccf_center"]
    old_template_center = full_fit["template_center"]
    old_matrix = full_fit["matrix_px_per_mm"]
    start = full_fit["parameters"].copy()
    start[:2] = old_template_center + (ccf_center - old_center) @ old_matrix.T

    def objective(parameters: np.ndarray) -> float:
        xy = transform_ccf(ccf, ccf_center, parameters, reflection)
        predicted, outside, bounds_distance = sample_template(template, xy)
        retinal = np.mean(2.0 * (np.sqrt(1.0 + ((predicted - target) / 10.0) ** 2) - 1.0))
        row_col = np.clip(xy, [0, 0], [width - 1, height - 1])[:, ::-1]
        area_distances = np.asarray([
            template["area_distance"][area](row_col[i:i + 1])[0]
            for i, area in enumerate(areas)
        ])
        area_penalty = float(np.mean((area_distances / 12.0) ** 2))
        domain_penalty = float(np.mean((outside / 15.0) ** 2 + bounds_distance**2))
        scale_x, scale_y, shear = parameters[3], parameters[4], parameters[5]
        geometry_penalty = float(
            .02 * (np.log(scale_x / 180.0) ** 2 + np.log(scale_y / 180.0) ** 2)
            + .02 * shear**2
        )
        return float(retinal + 2.0 * area_penalty + domain_penalty + geometry_penalty)

    bounds = [
        (40.0, width - 30.0), (20.0, height - 20.0), (-np.pi, np.pi),
        (70.0, 320.0), (70.0, 320.0), (-.8, .8),
    ]
    result = minimize(
        objective, start, method="Powell", bounds=bounds,
        options={"maxiter": 1800, "xtol": 1e-6, "ftol": 1e-6},
    )
    xy = transform_ccf(ccf, ccf_center, result.x, reflection)
    predicted, _, _ = sample_template(template, xy)
    center, matrix = affine(result.x, reflection)
    return {
        "parameters": result.x, "ccf_center": ccf_center, "template_center": center,
        "matrix_px_per_mm": matrix, "predicted": predicted, "target": target,
        "xy": xy, "objective": float(result.fun), "reflection": reflection,
        "convention": convention,
    }


def correction_for(
    row: pd.Series,
    pool: pd.DataFrame,
    method: str,
    alpha: float,
    length_px: float | None,
) -> np.ndarray:
    local = pool.loc[pool.ecephys_structure_acronym.eq(row.held_out_area)]
    if local.empty or method == "none":
        return np.zeros(2)
    residual = local[["residual_azimuth_deg", "residual_elevation_deg"]].to_numpy(float)
    if method == "area":
        mean = residual.mean(axis=0)
    else:
        distance = np.hypot(
            local.template_x_px.to_numpy() - row.template_x_px,
            local.template_y_px.to_numpy() - row.template_y_px,
        )
        weights = np.exp(-.5 * (distance / float(length_px)) ** 2)
        mean = np.average(residual, axis=0, weights=weights) if weights.sum() > 1e-8 else residual.mean(axis=0)
    return alpha * mean


def configurations() -> list[dict]:
    result = [{"name": "none", "method": "none", "alpha": 0.0, "length_px": None}]
    for alpha in (.25, .5, 1.0):
        result.append({"name": f"area_a{alpha:g}", "method": "area", "alpha": alpha, "length_px": None})
        for length in (25.0, 50.0, 100.0):
            result.append({"name": f"kernel_l{length:g}_a{alpha:g}", "method": "kernel", "alpha": alpha, "length_px": length})
    return result


def predict_config(row: pd.Series, pool: pd.DataFrame, config: dict) -> tuple[np.ndarray, float]:
    baseline = np.array([row.baseline_azimuth_deg, row.baseline_elevation_deg])
    observed = np.array([row.observed_azimuth_deg, row.observed_elevation_deg])
    # Stored residual is template - observed, so subtracting it corrects template prediction.
    prediction = baseline - correction_for(
        row, pool, config["method"], config["alpha"], config["length_px"]
    )
    return prediction, float(np.linalg.norm(prediction - observed))


def nested_animal_transfer(
    folds: pd.DataFrame, full_landmarks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = configurations()
    outer_rows, selection_rows = [], []
    for outer_session in SESSIONS:
        train_sessions = set(SESSIONS) - {outer_session}
        scores = []
        for config in configs:
            errors = []
            for _, row in folds.loc[folds.session_id.isin(train_sessions)].iterrows():
                pool = full_landmarks.loc[
                    ~full_landmarks.session_id.isin({outer_session, int(row.session_id)})
                ]
                _, error = predict_config(row, pool, config)
                errors.append(error)
            scores.append({
                "outer_session_id": outer_session, **config,
                "inner_median_error_deg": float(np.median(errors)),
                "inner_mean_error_deg": float(np.mean(errors)),
            })
        score_table = pd.DataFrame(scores).sort_values(
            ["inner_median_error_deg", "inner_mean_error_deg", "name"]
        )
        selected = score_table.iloc[0].to_dict()
        selection_rows.extend(scores)
        pool = full_landmarks.loc[~full_landmarks.session_id.eq(outer_session)]
        config = {key: selected[key] for key in ("name", "method", "alpha", "length_px")}
        for _, row in folds.loc[folds.session_id.eq(outer_session)].iterrows():
            prediction, error = predict_config(row, pool, config)
            outer_rows.append({
                **row.to_dict(),
                "selected_config": config["name"],
                "selected_method": config["method"],
                "selected_alpha": config["alpha"],
                "selected_length_px": config["length_px"],
                "hierarchical_azimuth_deg": prediction[0],
                "hierarchical_elevation_deg": prediction[1],
                "hierarchical_vector_error_deg": error,
                "hierarchical_improvement_deg": row.baseline_vector_error_deg - error,
            })
    return pd.DataFrame(outer_rows), pd.DataFrame(selection_rows)


def render(
    candidates: pd.DataFrame,
    landmarks: pd.DataFrame,
    gradients: pd.DataFrame,
    outer: pd.DataFrame,
    audit: pd.DataFrame,
    template: dict,
    convention: str,
    reflection: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10.8), constrained_layout=True)
    axis = axes[0, 0]
    summary = candidates.groupby(["azimuth_convention", "cortical_reflection"], as_index=False).agg(
        median_objective=("objective", "median"), mean_objective=("objective", "mean")
    )
    labels = [f"{r.azimuth_convention}\nreflection {r.cortical_reflection:+d}" for r in summary.itertuples()]
    colors = ["#7b3294" if r.azimuth_convention == convention and r.cortical_reflection == reflection else "#aaaaaa" for r in summary.itertuples()]
    axis.bar(np.arange(len(summary)), summary.median_objective, color=colors)
    axis.set_xticks(np.arange(len(summary)), labels, rotation=20, ha="right")
    axis.set(ylabel="median joint objective", title="One retinal frame across animals")
    axis.grid(axis="y", color="#dddddd", linewidth=.45)

    axis = axes[0, 1]
    axis.contour(template["boundary"].astype(float), levels=[.5], colors="#888888", linewidths=.5)
    for area, local in landmarks.groupby("ecephys_structure_acronym"):
        axis.scatter(local.template_x_px, local.template_y_px, s=28, color=AREA_COLORS[area],
                     alpha=.72, edgecolors="white", linewidths=.3, label=AREA_LABELS[area])
    axis.set(xlim=(-.5, 469.5), ylim=(429.5, -.5), xlabel="Zhuang common x (px)",
             ylabel="Zhuang common y (px; down +)", title="68 registered probe landmarks")
    axis.set_aspect("equal")
    axis.legend(fontsize=7, ncol=2)

    axis = axes[0, 2]
    per_animal = outer.groupby("session_id").agg(
        baseline=("baseline_vector_error_deg", "median"),
        hierarchical=("hierarchical_vector_error_deg", "median"),
        improvement=("hierarchical_improvement_deg", "median"),
    ).sort_values("baseline")
    x = np.arange(len(per_animal))
    axis.plot(x, per_animal.baseline, "o-", color="#777777", label="animal LOPO baseline")
    axis.plot(x, per_animal.hierarchical, "D-", color="#7b3294", label="nested shared correction")
    axis.set_xticks(x, [str(v)[-3:] for v in per_animal.index], rotation=45)
    axis.set(xlabel="session suffix", ylabel="median held-out probe error (°)",
             title="Outer animal-held-out performance")
    axis.grid(color="#dddddd", linewidth=.45)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    areas = sorted(outer.held_out_area.unique())
    values = [outer.loc[outer.held_out_area.eq(area), "hierarchical_improvement_deg"].to_numpy() for area in areas]
    axis.boxplot(values, labels=[AREA_LABELS[a] for a in areas], showfliers=True)
    axis.axhline(0, color="#444444", linewidth=.8)
    axis.set(ylabel="baseline − hierarchical error (°)", title="Does pooling help consistently by area?")
    axis.grid(axis="y", color="#dddddd", linewidth=.45)

    axis = axes[1, 1]
    sign = gradients.groupby("area").agg(
        probes=("ecephys_probe_id", "size"),
        azimuth=("azimuth_sign_match", "mean"),
        elevation=("elevation_sign_match", "mean"),
    ).loc[areas]
    gx = np.arange(len(sign))
    axis.bar(gx - .18, 100 * sign.azimuth, width=.36, color="#4c78a8", label="azimuth")
    axis.bar(gx + .18, 100 * sign.elevation, width=.36, color="#d95f5f", label="elevation")
    axis.axhline(50, color="#777777", linestyle="--", linewidth=.8)
    axis.set_xticks(gx, [f"{AREA_LABELS[a]}\nn={sign.loc[a, 'probes']}" for a in sign.index])
    axis.set(ylabel="observed/template gradient sign agreement (%)", title="Cell-gradient diagnostic")
    axis.legend(fontsize=8)
    axis.grid(axis="y", color="#dddddd", linewidth=.45)

    axis = axes[1, 2]
    axis.scatter(audit.selected_cells, audit.probe_area_landmarks, c=audit.areas, cmap="viridis", s=65)
    for row in audit.itertuples():
        axis.text(row.selected_cells + 3, row.probe_area_landmarks + .03, str(row.session_id)[-3:], fontsize=7)
    axis.set(xlabel="selected cells", ylabel="probe-area landmarks", title="Animal support and balance")
    axis.grid(color="#dddddd", linewidth=.45)

    figure.suptitle(
        "Fourteen-animal anatomy + RF registration to Zhuang Figure 9\n"
        "Fixed anatomy · shared retinal frame · probe-held-out fits · nested animal-held-out pooling",
        fontsize=15,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells, audit = production_support()
    cells.to_csv(OUTPUT / "production_registration_cell_support.csv.gz", index=False, compression="gzip")
    audit.to_csv(OUTPUT / "session_support_audit.csv", index=False)
    template = build_template(TEMPLATE_PATH)

    session_data: dict[int, dict] = {}
    candidate_rows = []
    for session_position, session_id in enumerate(SESSIONS):
        local_cells = cells.loc[cells.session_id.eq(session_id)].copy()
        landmarks = make_landmarks(local_cells)
        fits = {}
        candidate_number = 0
        for convention in ("native", "100_minus_azimuth"):
            for reflection in (-1, 1):
                fit = fit_candidate(
                    template, landmarks, 2.0, convention, reflection,
                    SEED + session_position * 10 + candidate_number,
                )
                candidate_number += 1
                fits[(convention, reflection)] = fit
                candidate_rows.append(candidate_row(session_id, fit))
        session_data[session_id] = {"cells": local_cells, "landmarks": landmarks, "fits": fits}
        print(f"candidate fits complete: {session_position + 1}/{len(SESSIONS)}", flush=True)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUTPUT / "shared_frame_candidate_fits.csv", index=False)
    frame_summary = candidates.groupby(["azimuth_convention", "cortical_reflection"]).agg(
        median_objective=("objective", "median"), mean_objective=("objective", "mean")
    ).reset_index().sort_values(["median_objective", "mean_objective"])
    selected_convention = str(frame_summary.iloc[0].azimuth_convention)
    selected_reflection = int(frame_summary.iloc[0].cortical_reflection)
    frame_summary.to_csv(OUTPUT / "shared_frame_selection.csv", index=False)

    cell_tables, landmark_tables, gradient_tables = [], [], []
    for session_id, item in session_data.items():
        fit = item["fits"][(selected_convention, selected_reflection)]
        cell_tables.append(common_cell_table(item["cells"], fit, template))
        landmark_tables.append(landmark_table(session_id, item["landmarks"], fit))
        gradient_tables.append(gradient_table(session_id, item["cells"], item["landmarks"], fit, template))
        item["selected_fit"] = fit
    common_cells = pd.concat(cell_tables, ignore_index=True)
    full_landmarks = pd.concat(landmark_tables, ignore_index=True)
    gradients = pd.concat(gradient_tables, ignore_index=True)
    common_cells.to_csv(OUTPUT / "registered_cells_common_zhuang_coordinates.csv.gz", index=False, compression="gzip")
    full_landmarks.to_csv(OUTPUT / "registered_probe_landmarks.csv", index=False)
    gradients.to_csv(OUTPUT / "probe_gradient_comparison.csv", index=False)

    fold_rows = []
    for session_position, session_id in enumerate(SESSIONS):
        item = session_data[session_id]
        landmarks = item["landmarks"]
        for held_index in range(len(landmarks)):
            held = landmarks.iloc[held_index]
            train = landmarks.drop(index=held_index).reset_index(drop=True)
            fold_fit = local_refit(
                train, template, selected_convention, selected_reflection, item["selected_fit"]
            )
            held_ccf = held[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)[None, :]
            xy = transform_ccf(
                held_ccf, fold_fit["ccf_center"], fold_fit["parameters"], selected_reflection
            )
            prediction, _, _ = sample_template(template, xy)
            observed = target_rf(
                held[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)[None, :],
                selected_convention,
            )[0]
            fold_rows.append({
                "session_id": session_id,
                "held_out_probe_id": int(held.ecephys_probe_id),
                "held_out_area": held.ecephys_structure_acronym,
                "template_x_px": xy[0, 0], "template_y_px": xy[0, 1],
                "observed_azimuth_deg": observed[0], "observed_elevation_deg": observed[1],
                "baseline_azimuth_deg": prediction[0, 0], "baseline_elevation_deg": prediction[0, 1],
                "baseline_vector_error_deg": float(np.linalg.norm(prediction[0] - observed)),
            })
        print(f"probe folds complete: {session_position + 1}/{len(SESSIONS)}", flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUTPUT / "leave_one_probe_out_baseline.csv", index=False)
    outer, selection = nested_animal_transfer(folds, full_landmarks)
    outer.to_csv(OUTPUT / "nested_animal_held_out_results.csv", index=False)
    selection.to_csv(OUTPUT / "nested_model_selection.csv", index=False)

    figure_path = OUTPUT / "Figure_14animal_registration_summary.png"
    render(candidates, full_landmarks, gradients, outer, audit, template,
           selected_convention, selected_reflection, figure_path)
    manifest = {
        "checkpoint": "frozen fourteen-animal hierarchical registration",
        "status": "exploratory; production fits complete for the frozen cohort; pooling evaluated with nested animal holdout",
        "sessions": list(SESSIONS),
        "selection": {
            "spatial_model": "aperture", "unit_split": "evaluation",
            "areas": list(TARGET_AREAS), "requires_finite_rf_center_and_ccf": True,
            "selected_cells": len(cells), "probe_area_landmarks": len(full_landmarks),
        },
        "shared_frame": {
            "azimuth_convention": selected_convention,
            "cortical_reflection": selected_reflection,
        },
        "validation": {
            "inner": "select residual pooling configuration using animals other than the outer held-out animal",
            "outer": "evaluate selected configuration on every probe fold from one unseen animal",
            "independent_unit": "probe/area landmark; animal is the outer generalization unit",
            "baseline_median_error_deg": float(outer.baseline_vector_error_deg.median()),
            "hierarchical_median_error_deg": float(outer.hierarchical_vector_error_deg.median()),
            "median_improvement_deg": float(outer.hierarchical_improvement_deg.median()),
            "mean_improvement_deg": float(outer.hierarchical_improvement_deg.mean()),
            "folds_improved": int(outer.hierarchical_improvement_deg.gt(0).sum()),
            "folds": len(outer),
        },
        "gradient_diagnostic": {
            "probes": len(gradients),
            "azimuth_sign_agreement": float(gradients.azimuth_sign_match.mean()),
            "elevation_sign_agreement": float(gradients.elevation_sign_match.mean()),
        },
        "limitations": [
            "The shared-frame and residual families are selected on this exploratory fourteen-animal cohort.",
            "Animal transforms remain global anatomy-constrained affines; local animal warps are not yet fitted.",
            "Censored RF-size fits are retained when their RF centers are finite because size censoring does not itself invalidate location.",
        ],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
