#!/usr/bin/env python3
"""Initial evidence checkpoint for cell/probe contributions to registration.

This deliberately does not refit the retinotopy map.  It separates two ways
that cells can help: precision of each penetration's RF centroid, and a
directional RF derivative along the AP/ML trajectory sampled by that probe.
The latter is assessed with leave-one-position-bin-out prediction so that a
gradient must generalize along the probe rather than merely improve in-sample
fit to individual cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "artifacts/retinotopy_registration_pilot/session_798911424/"
    "cell_scatter_CCF_RF_support.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/retinotopy_registration_pilot/session_798911424/"
    "cell_gradient_evidence_checkpoint1"
)
AREA_COLORS = {
    "VISp": "#2864a8", "VISl": "#d78318", "VISal": "#b33f62",
    "VISrl": "#5f8f3e", "VISam": "#7356a8",
}


def oriented_probe_coordinate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xy = frame[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    centered = xy - xy.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0].copy()
    # Fix the arbitrary PCA sign: positive coordinate is increasing AP; if
    # nearly ML-only, positive is increasing ML.
    if abs(direction[0]) >= abs(direction[1]):
        if direction[0] < 0:
            direction *= -1
    elif direction[1] < 0:
        direction *= -1
    return centered @ direction, direction


def blocked_cv(t: np.ndarray, rf: np.ndarray, bins: int = 4) -> tuple[float, float, float]:
    order = np.argsort(t)
    fold_indices = np.array_split(order, min(bins, len(order)))
    constant_errors: list[np.ndarray] = []
    gradient_errors: list[np.ndarray] = []
    for held_out in fold_indices:
        train = np.ones(len(t), dtype=bool)
        train[held_out] = False
        if train.sum() < 3 or np.ptp(t[train]) <= 1e-9:
            continue
        constant = np.median(rf[train], axis=0)
        design = np.column_stack([np.ones(train.sum()), t[train]])
        coefficients = np.linalg.lstsq(design, rf[train], rcond=None)[0]
        prediction = np.column_stack([np.ones(len(held_out)), t[held_out]]) @ coefficients
        constant_errors.append(np.sum((rf[held_out] - constant) ** 2, axis=1))
        gradient_errors.append(np.sum((rf[held_out] - prediction) ** 2, axis=1))
    constant_squared = np.concatenate(constant_errors)
    gradient_squared = np.concatenate(gradient_errors)
    constant_rmse = float(np.sqrt(np.mean(constant_squared)))
    gradient_rmse = float(np.sqrt(np.mean(gradient_squared)))
    return constant_rmse, gradient_rmse, constant_rmse - gradient_rmse


def bootstrap_median_precision(rf: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    indices = rng.integers(0, len(rf), size=(draws, len(rf)))
    medians = np.median(rf[indices], axis=1)
    center = np.median(medians, axis=0)
    return float(np.sqrt(np.mean(np.sum((medians - center) ** 2, axis=1))))


def bootstrap_slopes(
    t: np.ndarray, rf: np.ndarray, rng: np.random.Generator, draws: int
) -> np.ndarray:
    slopes = np.empty((draws, 2), dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(t), size=len(t))
        if np.ptp(t[selected]) <= 1e-9:
            slopes[draw] = np.nan
            continue
        slopes[draw] = np.linalg.lstsq(
            np.column_stack([np.ones(len(t)), t[selected]]), rf[selected], rcond=None
        )[0][1]
    return slopes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input.resolve())
    rng = np.random.default_rng(args.seed)

    records: list[dict] = []
    traces: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for probe_id, frame in data.groupby("ecephys_probe_id", sort=True):
        frame = frame.copy()
        t, direction = oriented_probe_coordinate(frame)
        t = t - np.median(t)
        rf = frame[["visual_azimuth_deg", "visual_elevation_deg"]].to_numpy(float)
        design = np.column_stack([np.ones(len(t)), t])
        coefficients = np.linalg.lstsq(design, rf, rcond=None)[0]
        prediction = design @ coefficients
        residual = rf - prediction
        total = rf - rf.mean(axis=0)
        r2_vector = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
        constant_rmse, gradient_rmse, improvement = blocked_cv(t, rf)
        slopes = bootstrap_slopes(t, rf, rng, args.bootstrap_draws)
        slope_ci = np.nanpercentile(slopes, [2.5, 97.5], axis=0)
        median_precision = bootstrap_median_precision(rf, rng, args.bootstrap_draws)
        area = frame.ecephys_structure_acronym.iloc[0]
        records.append({
            "ecephys_probe_id": int(probe_id),
            "area": area,
            "cells": len(frame),
            "probe_ap_direction": direction[0],
            "probe_ml_direction": direction[1],
            "sampled_span_mm": float(np.ptp(t)),
            "rf_median_azimuth_deg": float(np.median(rf[:, 0])),
            "rf_median_elevation_deg": float(np.median(rf[:, 1])),
            "bootstrap_rf_median_vector_se_deg": median_precision,
            "azimuth_gradient_deg_per_mm": coefficients[1, 0],
            "azimuth_gradient_ci_low": slope_ci[0, 0],
            "azimuth_gradient_ci_high": slope_ci[1, 0],
            "elevation_gradient_deg_per_mm": coefficients[1, 1],
            "elevation_gradient_ci_low": slope_ci[0, 1],
            "elevation_gradient_ci_high": slope_ci[1, 1],
            "gradient_vector_magnitude_deg_per_mm": float(np.linalg.norm(coefficients[1])),
            "in_sample_vector_r2": r2_vector,
            "blocked_cv_constant_vector_rmse_deg": constant_rmse,
            "blocked_cv_gradient_vector_rmse_deg": gradient_rmse,
            "blocked_cv_gradient_improvement_deg": improvement,
        })
        traces[int(probe_id)] = (frame, t, coefficients)

    summary = pd.DataFrame(records)
    summary.to_csv(output / "probe_cell_gradient_evidence.csv", index=False)

    figure, axes = plt.subplots(3, len(summary), figsize=(19, 9.5), constrained_layout=True)
    for column, row in enumerate(summary.itertuples(index=False)):
        frame, t, coefficients = traces[row.ecephys_probe_id]
        color = AREA_COLORS[row.area]
        x_line = np.linspace(t.min(), t.max(), 100)
        for axis, field, component, ylabel in (
            (axes[0, column], "visual_azimuth_deg", 0, "RF azimuth (°)"),
            (axes[1, column], "visual_elevation_deg", 1, "RF elevation (°)"),
        ):
            axis.scatter(t, frame[field], s=17, color=color, alpha=0.68, linewidths=0)
            axis.plot(x_line, coefficients[0, component] + coefficients[1, component] * x_line,
                      color="#111111", linewidth=1.5)
            axis.axhline(np.median(frame[field]), color="#777777", linestyle="--", linewidth=0.8)
            axis.set_xlabel("position along AP–ML probe trajectory (mm)")
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.grid(color="#dddddd", linewidth=0.45)
        axes[0, column].set_title(
            f"{str(row.ecephys_probe_id)[-3:]} · {row.area} · n={row.cells}\n"
            f"span={row.sampled_span_mm:.2f} mm",
            fontsize=10,
        )
        axis = axes[2, column]
        values = [row.blocked_cv_constant_vector_rmse_deg, row.blocked_cv_gradient_vector_rmse_deg]
        axis.bar([0, 1], values, color=["#a6a6a6", color], width=0.7)
        axis.set_xticks([0, 1], ["constant", "gradient"], rotation=20)
        if column == 0:
            axis.set_ylabel("blocked-CV RF-vector RMSE (°)")
        axis.set_title(
            f"ΔRMSE={row.blocked_cv_gradient_improvement_deg:+.1f}°\n"
            f"median SE={row.bootstrap_rf_median_vector_se_deg:.1f}°",
            fontsize=9,
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.45)

    figure.suptitle(
        "Session 798911424: what cells add beyond one RF centroid per penetration\n"
        "Raw RFs and linear gradients along each probe's AP–ML trajectory; prediction holds out contiguous position bins",
        fontsize=14,
    )
    figure.savefig(output / "Figure_probe_cell_gradient_evidence.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "checkpoint": "initial cell and within-probe gradient evidence",
        "status": "exploratory; no map refit at this checkpoint",
        "session_id": int(data.session_id.iloc[0]),
        "input": str(args.input.resolve()),
        "cells": len(data),
        "penetrations": len(summary),
        "cell_role": "reduce uncertainty of each penetration RF centroid",
        "gradient_role": "one directional derivative per RF coordinate along each probe's sampled AP/ML trajectory",
        "gradient_test": "four-fold leave-one-contiguous-position-bin-out linear prediction versus a training-set median constant",
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "important_limitations": [
            "Cells on a penetration are not independent registration landmarks.",
            "A probe supplies only a one-dimensional directional derivative, not the full two-dimensional map gradient.",
            "Cell bootstrap intervals ignore shared fit/systematic errors and should not be interpreted as population confidence intervals.",
            "Apparent gradients may include laminar or probe-track effects and have not yet been tested against the warped Zhuang gradient direction.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
