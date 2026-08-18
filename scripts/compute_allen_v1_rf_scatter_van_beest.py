#!/usr/bin/env python3
"""RF scatter in Allen V1, following van Beest et al. 2021 (Nat Commun,
https://www.nature.com/articles/s41467-021-24311-5, "Mouse visual cortex contains a region of
enhanced spatial resolution") -- Methods, "Measurement of cortical magnification factor and RF
scatter":

1. Rotate cortical-position axes so azimuth's gradient runs principally along the new x-axis and
   elevation's along the new y-axis ("We first rotated the axes of the cortical image so that
   the representation of azimuth changed principally along the x-axis and the representation of
   elevation along the y-axis").
2. Fit v = a * x^b relating cortical position (x, mm, one axis) to RF position (v, deg, azimuth
   or elevation) via robust nonlinear least-absolute-residual (L1) regression, separately per
   axis ("fitting an exponential function using robust nonlinear least-absolute residual
   regression").
3. Residual = observed RF position - fitted prediction (deg).
4. Order cells by RF-space coordinate (azimuth, elevation, or eccentricity), split into 10
   non-overlapping equal-count bins; IQR of residuals per bin.
5. Linear regression of bin IQR vs. bin-center coordinate; bootstrap significance test (resample
   cells, one-tailed: is the slope significantly positive, i.e. does scatter increase with
   eccentricity).

The paper derives the axis rotation from ITS OWN imaging data (self-contained, no external
atlas) -- this reimplementation does the same from Allen's own CCF-vs-RF relationship, rather
than reusing this project's Zhuang-anchored -8.1 deg rotation (a different, atlas-relative
quantity, not what the paper's method asks for).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
OUTPUT = ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint/van_beest_rf_scatter"

N_BINS = 10
N_BOOTSTRAP = 2000


def fit_power_law_l1(x: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """v = a * x^b via least-absolute-residual (L1) regression, matching the paper's robust
    nonlinear fit. x must be positive (shifted cortical position)."""
    log_x = np.log(x)

    def loss(params):
        log_a, b = params
        predicted = np.exp(log_a) * x**b
        return np.sum(np.abs(v - predicted))

    # initialize from an OLS fit in log-log space (valid only where v > 0; robust to that via a
    # small floor, used only for initialization, not the final robust fit)
    v_floor = np.maximum(v - v.min() + 1.0, 1e-3)
    b0, log_a0 = np.polyfit(log_x, np.log(v_floor), 1)
    result = minimize(loss, x0=[log_a0, b0], method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    log_a, b = result.x
    return float(np.exp(log_a)), float(b)


def predict_power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x**b


def binned_iqr(order_values: np.ndarray, residuals: np.ndarray, n_bins: int) -> pd.DataFrame:
    order = np.argsort(order_values)
    sorted_values = order_values[order]
    sorted_residuals = residuals[order]
    edges = np.array_split(np.arange(len(sorted_values)), n_bins)
    rows = []
    for edge in edges:
        bin_values = sorted_values[edge]
        bin_residuals = sorted_residuals[edge]
        q25, q75 = np.percentile(bin_residuals, [25, 75])
        rows.append({"bin_center": float(bin_values.mean()), "iqr": float(q75 - q25), "n": len(edge)})
    return pd.DataFrame(rows)


def slope_with_bootstrap(order_values: np.ndarray, residuals: np.ndarray, n_bins: int, rng: np.random.Generator):
    bins = binned_iqr(order_values, residuals, n_bins)
    slope, intercept = np.polyfit(bins.bin_center, bins.iqr, 1)
    n = len(order_values)
    boot_slopes = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot_bins = binned_iqr(order_values[idx], residuals[idx], n_bins)
        boot_slopes[i] = np.polyfit(boot_bins.bin_center, boot_bins.iqr, 1)[0]
    p_one_tailed = float((boot_slopes <= 0).mean())
    return bins, float(slope), float(intercept), p_one_tailed, boot_slopes


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(NAIVE_CELLS)
    v1 = cells.loc[cells.map_area.eq("VISp")].copy()
    v1["ccf_ap_mm"] = v1.anterior_posterior_ccf_coordinate / 1000.0
    v1["ccf_ml_mm"] = v1.left_right_ccf_coordinate / 1000.0
    azimuth = v1.rf_x.to_numpy(float)
    elevation = v1.rf_y.to_numpy(float)
    ccf = v1[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    print(f"Allen V1 cells: {len(v1)}")

    # 1. rotate axes: gradient direction of azimuth in CCF space becomes the new x-axis
    design = np.column_stack([ccf, np.ones(len(ccf))])
    coef_az, *_ = np.linalg.lstsq(design, azimuth, rcond=None)
    gradient_ap, gradient_ml = coef_az[0], coef_az[1]
    theta = np.arctan2(gradient_ml, gradient_ap)
    x_rot = ccf[:, 0] * np.cos(theta) + ccf[:, 1] * np.sin(theta)
    y_rot = -ccf[:, 0] * np.sin(theta) + ccf[:, 1] * np.cos(theta)
    print(f"rotation angle (azimuth gradient -> x-axis): {np.degrees(theta):+.1f} deg")

    # shift rotated positions positive for the power-law fit (x must be > 0)
    x_pos = x_rot - x_rot.min() + 0.05
    y_pos = y_rot - y_rot.min() + 0.05

    # 2-3. fit + residuals, per axis
    a_az, b_az = fit_power_law_l1(x_pos, azimuth)
    predicted_az = predict_power_law(x_pos, a_az, b_az)
    residual_az = azimuth - predicted_az

    a_el, b_el = fit_power_law_l1(y_pos, elevation)
    predicted_el = predict_power_law(y_pos, a_el, b_el)
    residual_el = elevation - predicted_el

    print(f"azimuth power-law fit: v = {a_az:.3f} * x^{b_az:.3f}, residual std={residual_az.std():.2f} deg")
    print(f"elevation power-law fit: v = {a_el:.3f} * y^{b_el:.3f}, residual std={residual_el.std():.2f} deg")

    # 4-5. binned IQR + slope + bootstrap significance, for azimuth and elevation
    rng = np.random.default_rng(20260817)
    bins_az, slope_az, intercept_az, p_az, boot_az = slope_with_bootstrap(azimuth, residual_az, N_BINS, rng)
    bins_el, slope_el, intercept_el, p_el, boot_el = slope_with_bootstrap(elevation, residual_el, N_BINS, rng)

    # eccentricity version: distance from the population RF centroid, combined 2D residual magnitude
    centroid = np.array([np.median(azimuth), np.median(elevation)])
    eccentricity = np.hypot(azimuth - centroid[0], elevation - centroid[1])
    residual_combined = np.hypot(residual_az, residual_el)
    bins_ecc, slope_ecc, intercept_ecc, p_ecc, boot_ecc = slope_with_bootstrap(eccentricity, residual_combined, N_BINS, rng)

    print(f"\nazimuth:      slope={slope_az:+.4f} deg_IQR/deg, bootstrap one-tailed p={p_az:.4f}")
    print(f"elevation:    slope={slope_el:+.4f} deg_IQR/deg, bootstrap one-tailed p={p_el:.4f}")
    print(f"eccentricity: slope={slope_ecc:+.4f} deg_IQR/deg, bootstrap one-tailed p={p_ecc:.4f} "
          f"(distance from population RF centroid az={centroid[0]:.1f}, el={centroid[1]:.1f})")

    summary = {
        "n_cells": len(v1), "rotation_deg": float(np.degrees(theta)),
        "azimuth_fit": {"a": a_az, "b": b_az, "slope_iqr_vs_position": slope_az,
                         "intercept": intercept_az, "bootstrap_p_one_tailed": p_az},
        "elevation_fit": {"a": a_el, "b": b_el, "slope_iqr_vs_position": slope_el,
                           "intercept": intercept_el, "bootstrap_p_one_tailed": p_el},
        "eccentricity_fit": {"centroid_azimuth": float(centroid[0]), "centroid_elevation": float(centroid[1]),
                              "slope_iqr_vs_eccentricity": slope_ecc, "intercept": intercept_ecc,
                              "bootstrap_p_one_tailed": p_ecc},
        "method": "van Beest et al. 2021 Nat Commun (PMC8242089), Methods: 'Measurement of cortical "
                  "magnification factor and RF scatter' -- rotated CCF axes, robust L1 power-law fit, "
                  "10 equal-count bins, IQR of residuals, linear regression of bin IQR vs. bin center, "
                  "one-tailed bootstrap test for positive slope.",
    }
    (OUTPUT / "van_beest_scatter_summary.json").write_text(json.dumps(summary, indent=2))
    v1_out = v1[["ecephys_session_id", "ecephys_probe_id", "ccf_ap_mm", "ccf_ml_mm"]].copy()
    v1_out["rf_azimuth_deg"] = azimuth
    v1_out["rf_elevation_deg"] = elevation
    v1_out["x_rot_mm"] = x_rot
    v1_out["y_rot_mm"] = y_rot
    v1_out["residual_azimuth_deg"] = residual_az
    v1_out["residual_elevation_deg"] = residual_el
    v1_out["eccentricity_from_centroid_deg"] = eccentricity
    v1_out.to_csv(OUTPUT / "allen_v1_cell_residuals.csv", index=False)
    print(f"\nwrote {OUTPUT / 'van_beest_scatter_summary.json'}")

    # -- figure: 3-panel, mirroring the paper's scatter-vs-eccentricity style --
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (bins, slope, p_value, label, xlabel) in zip(axes, (
        (bins_az, slope_az, p_az, "Azimuth", "RF azimuth (deg)"),
        (bins_el, slope_el, p_el, "Elevation", "RF elevation (deg)"),
        (bins_ecc, slope_ecc, p_ecc, "Eccentricity (from RF centroid)", "RF eccentricity (deg)"),
    )):
        ax.scatter(bins.bin_center, bins.iqr, s=60, color="#2864a8", zorder=3)
        fit_x = np.linspace(bins.bin_center.min(), bins.bin_center.max(), 50)
        ax.plot(fit_x, slope * fit_x + (bins.iqr - slope * bins.bin_center).mean(), color="#b33f62",
                linewidth=1.5, linestyle="--", zorder=2)
        ax.set(title=f"{label}\nslope={slope:+.3f}, bootstrap one-tailed p={p_value:.3f}",
               xlabel=xlabel, ylabel="IQR of residuals (deg)")
    fig.suptitle(f"Allen V1 RF scatter vs. position (van Beest et al. 2021 method), n={len(v1)} cells", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_allen_v1_van_beest_rf_scatter.png", dpi=170)
    plt.close(fig)
    print(OUTPUT / "Figure_allen_v1_van_beest_rf_scatter.png")


if __name__ == "__main__":
    main()
