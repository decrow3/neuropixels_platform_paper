#!/usr/bin/env python3
"""Estimate Allen session translations from a cross-session V1 CCF-to-RF map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.allen_bo11_tuning_driven_limited_affine import PARAMETER_NAMES


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_SUPPORT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = AUDIT / "ccf_retinotopy_alignment"
BO_COHORT = "Brain Observatory 1.1"
CCF_COLUMNS = (
    "anterior_posterior_ccf_coordinate",
    "dorsal_ventral_ccf_coordinate",
    "left_right_ccf_coordinate",
)
RF_COLUMNS = ("azimuth_rf", "elevation_rf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--translation-bound-deg", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polynomial_features(values: np.ndarray, degree: int) -> np.ndarray:
    """Return linear or quadratic terms without an intercept."""
    values = np.asarray(values, dtype=float)
    if degree == 1:
        return values
    if degree != 2:
        raise ValueError("degree must be 1 or 2")
    x, y, z = values.T
    return np.column_stack([x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


def session_balanced_weights(session_ids: np.ndarray) -> np.ndarray:
    """Give every training session equal total prior weight."""
    session_ids = np.asarray(session_ids)
    _, inverse, counts = np.unique(session_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def fit_robust_ridge(
    predictors: np.ndarray,
    outcomes: np.ndarray,
    session_ids: np.ndarray,
    *,
    degree: int,
    ridge: float,
    iterations: int = 12,
) -> dict[str, np.ndarray | float | int]:
    """Fit a session-balanced multivariate Huber ridge model."""
    predictors = np.asarray(predictors, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    center = np.median(predictors, axis=0)
    scale = np.subtract(*np.percentile(predictors, [75, 25], axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    features = polynomial_features((predictors - center) / scale, degree)
    design = np.column_stack([np.ones(len(features)), features])
    prior = session_balanced_weights(session_ids)
    robust = np.ones(len(design))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.zeros((design.shape[1], outcomes.shape[1]))
    for _ in range(iterations):
        weights = prior * robust
        normal = design.T @ (design * weights[:, None]) + penalty
        coefficients = np.linalg.solve(normal, design.T @ (outcomes * weights[:, None]))
        residual = outcomes - design @ coefficients
        distance = np.sqrt(np.sum(np.square(residual), axis=1))
        median = np.median(distance)
        mad = np.median(np.abs(distance - median))
        cutoff = median + 1.5 * max(1.4826 * mad, 1e-6)
        robust = np.minimum(1.0, cutoff / np.maximum(distance, 1e-9))
    return {
        "center": center,
        "scale": scale,
        "coefficients": coefficients,
        "degree": degree,
    }


def predict_robust_ridge(model: dict[str, np.ndarray | float | int], predictors: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(predictors, dtype=float) - model["center"]) / model["scale"]
    features = polynomial_features(standardized, int(model["degree"]))
    design = np.column_stack([np.ones(len(features)), features])
    return design @ model["coefficients"]


def leave_one_session_out_predictions(
    table: pd.DataFrame,
    *,
    degree: int,
    ridge: float,
) -> np.ndarray:
    """Predict each session only from CCF-to-RF mappings learned in other sessions."""
    predictors = table[list(CCF_COLUMNS)].to_numpy(float)
    outcomes = table[list(RF_COLUMNS)].to_numpy(float)
    sessions = table["ecephys_session_id"].to_numpy()
    predictions = np.full_like(outcomes, np.nan)
    for session_id in np.unique(sessions):
        test = sessions == session_id
        model = fit_robust_ridge(
            predictors[~test], outcomes[~test], sessions[~test], degree=degree, ridge=ridge
        )
        predictions[test] = predict_robust_ridge(model, predictors[test])
    return predictions


def session_translations(
    table: pd.DataFrame,
    predictions: np.ndarray,
    *,
    bound_deg: float,
) -> pd.DataFrame:
    """Move observed RFs toward their anatomy-predicted positions using a robust residual."""
    local = table.copy()
    local[["predicted_azimuth_deg", "predicted_elevation_deg"]] = predictions
    local["delta_azimuth_deg"] = local["predicted_azimuth_deg"] - local["azimuth_rf"]
    local["delta_elevation_deg"] = local["predicted_elevation_deg"] - local["elevation_rf"]
    rows = []
    for session_id, group in local.groupby("ecephys_session_id", observed=True):
        raw = group[["delta_azimuth_deg", "delta_elevation_deg"]].median().to_numpy(float)
        bounded = np.clip(raw, -bound_deg, bound_deg)
        before = group[list(RF_COLUMNS)].to_numpy(float) - group[["predicted_azimuth_deg", "predicted_elevation_deg"]].to_numpy(float)
        after = before + bounded
        rows.append(
            {
                "ecephys_session_id": int(session_id),
                "selected_model": "ccf_to_v1_rf_translation",
                "ccf_available": True,
                "v1_rf_units": len(group),
                "raw_translation_azimuth_deg": raw[0],
                "raw_translation_elevation_deg": raw[1],
                "translation_azimuth_deg": bounded[0],
                "translation_elevation_deg": bounded[1],
                "rotation_deg": 0.0,
                "log_scale_azimuth": 0.0,
                "log_scale_elevation": 0.0,
                "shear": 0.0,
                "translation_was_bounded": bool(np.any(raw != bounded)),
                "rf_rmse_before_deg": float(np.sqrt(np.mean(np.square(before)))),
                "rf_rmse_after_translation_deg": float(np.sqrt(np.mean(np.square(after)))),
            }
        )
    return pd.DataFrame(rows)


def render_diagnostic(predictions: pd.DataFrame, sessions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
    for ax, observed, predicted, title in [
        (axes[0, 0], "azimuth_rf", "predicted_azimuth_deg", "Azimuth"),
        (axes[0, 1], "elevation_rf", "predicted_elevation_deg", "Elevation"),
    ]:
        ax.scatter(predictions[observed], predictions[predicted], s=7, alpha=0.18, color="#315b7d")
        limits = np.nanpercentile(np.r_[predictions[observed], predictions[predicted]], [1, 99])
        ax.plot(limits, limits, color="black", linewidth=1)
        ax.set(xlabel=f"Observed RF {title.lower()} (deg)", ylabel=f"Held-out CCF prediction (deg)", title=title)
    axes[1, 0].scatter(
        sessions["rf_rmse_before_deg"], sessions["rf_rmse_after_translation_deg"],
        s=38, color="#b14c3a", alpha=0.8,
    )
    limits = np.nanpercentile(
        sessions[["rf_rmse_before_deg", "rf_rmse_after_translation_deg"]].to_numpy(), [0, 100]
    )
    axes[1, 0].plot(limits, limits, color="black", linewidth=1)
    axes[1, 0].set(xlabel="Before session residual (deg RMSE)", ylabel="After session residual (deg RMSE)", title="Session translation evidence")
    axes[1, 1].quiver(
        np.zeros(len(sessions)), np.zeros(len(sessions)),
        sessions["translation_azimuth_deg"], sessions["translation_elevation_deg"],
        angles="xy", scale_units="xy", scale=1, width=0.006, color="#6a3d9a", alpha=0.72,
    )
    axes[1, 1].axhline(0, color="#999999", linewidth=0.7)
    axes[1, 1].axvline(0, color="#999999", linewidth=0.7)
    axes[1, 1].set(xlim=(-16, 16), ylim=(-16, 16), aspect="equal", xlabel="Azimuth translation (deg)", ylabel="Elevation translation (deg)", title="Applied CCF-conditioned translations")
    fig.suptitle("Allen BO 1.1: held-out V1 CCF→RF registration evidence", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(args.support.resolve(), low_memory=False)
    unit_columns = ["ecephys_unit_id", *CCF_COLUMNS]
    units = pd.read_csv(args.unit_table.resolve(), usecols=unit_columns, low_memory=False)
    population = support.loc[support["cohort"].eq(BO_COHORT) & support["area"].eq("V1")].merge(
        units, on="ecephys_unit_id", how="left", validate="one_to_one"
    )
    all_sessions = [int(value) for value in sorted(population["ecephys_session_id"].astype(int).unique())]
    population = population.dropna(subset=[*CCF_COLUMNS, *RF_COLUMNS]).reset_index(drop=True)
    available_sessions = [int(value) for value in sorted(population["ecephys_session_id"].astype(int).unique())]
    if len(available_sessions) < 4:
        raise ValueError("At least four sessions with V1 CCF and RF coordinates are required")

    candidates = []
    prediction_sets = {}
    for degree, label in [(1, "linear"), (2, "quadratic")]:
        predicted = leave_one_session_out_predictions(population, degree=degree, ridge=args.ridge)
        prediction_sets[label] = predicted
        residual = population[list(RF_COLUMNS)].to_numpy(float) - predicted
        session_rmse = []
        for session_id in available_sessions:
            mask = population["ecephys_session_id"].eq(session_id).to_numpy()
            centered = residual[mask] - np.median(residual[mask], axis=0)
            session_rmse.append(np.sqrt(np.mean(np.square(centered))))
        candidates.append({"model": label, "degree": degree, "median_session_centered_rmse_deg": np.median(session_rmse)})
    model_cv = pd.DataFrame(candidates).sort_values("median_session_centered_rmse_deg").reset_index(drop=True)
    selected_model = str(model_cv.loc[0, "model"])
    predictions = prediction_sets[selected_model]
    transforms = session_translations(population, predictions, bound_deg=args.translation_bound_deg)
    missing_sessions = sorted(set(all_sessions) - set(available_sessions))
    if missing_sessions:
        missing = pd.DataFrame(
            {
                "ecephys_session_id": missing_sessions,
                "selected_model": "ccf_unavailable_identity",
                "ccf_available": False,
                **{name: 0.0 for name in PARAMETER_NAMES},
            }
        )
        transforms = pd.concat([transforms, missing], ignore_index=True, sort=False)
    transforms = transforms.sort_values("ecephys_session_id").reset_index(drop=True)
    predicted_units = population.copy()
    predicted_units[["predicted_azimuth_deg", "predicted_elevation_deg"]] = predictions
    predicted_units["residual_azimuth_deg"] = predicted_units["azimuth_rf"] - predicted_units["predicted_azimuth_deg"]
    predicted_units["residual_elevation_deg"] = predicted_units["elevation_rf"] - predicted_units["predicted_elevation_deg"]

    model_cv.to_csv(output_dir / "ccf_retinotopy_model_cv.csv", index=False, float_format="%.6g")
    predicted_units.to_csv(output_dir / "ccf_retinotopy_unit_predictions.csv", index=False, float_format="%.6g")
    transforms.to_csv(output_dir / "selected_ccf_retinotopy_transforms.csv", index=False, float_format="%.6g")
    diagnostic = output_dir / "Figure_allen_bo11_ccf_to_rf_prediction.png"
    render_diagnostic(predicted_units, transforms.loc[transforms["ccf_available"].eq(True)], diagnostic)
    available = transforms.loc[transforms["ccf_available"].eq(True)]
    lines = [
        "# Allen BO 1.1 V1 CCF→RF registration",
        "",
        f"V1 CCF coordinates were available in **{len(available_sessions)}/{len(all_sessions)}** simultaneous V1/HVA sessions.",
        "Each session was predicted by a robust, session-balanced CCF→RF model trained only on the other sessions.",
        f"The selected RF-only model was **{selected_model}**; SF, TF, and HVA units were not used for fitting or selection.",
        "The robust median V1 prediction residual supplies one bounded translation shared by that session's V1 and HVA maps.",
        "Sessions without reconstructed CCF coordinates are marked unavailable and assigned identity only in the transform table; the four-row comparison excludes them from every row.",
        "",
        f"Median V1 RF RMSE before session translation: **{available['rf_rmse_before_deg'].median():.2f}°**.",
        f"Median V1 RF RMSE after session translation: **{available['rf_rmse_after_translation_deg'].median():.2f}°**.",
        f"Translations reaching the ±{args.translation_bound_deg:g}° bound: **{int(available['translation_was_bounded'].sum())}/{len(available)}**.",
    ]
    (output_dir / "ALLEN_BO11_CCF_RETINOTOPY_ALIGNMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output_dir.iterdir() if p.is_file() and p.name != "run_manifest.json"}
    manifest = {
        "checkpoint": "06c_allen_bo11_ccf_retinotopy_alignment",
        "status": "held-out V1 CCF-to-RF translation model",
        "inputs": {"support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())}, "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())}},
        "parameters": {"ccf_columns": CCF_COLUMNS, "candidate_degrees": [1, 2], "selected_model": selected_model, "ridge": args.ridge, "translation_bound_deg": args.translation_bound_deg, "fit_population": "V1 only", "held_out": "ecephys_session_id", "available_sessions": available_sessions, "missing_sessions": missing_sessions},
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen CCF→RF alignment written to {output_dir}")


if __name__ == "__main__":
    main()
