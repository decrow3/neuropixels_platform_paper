#!/usr/bin/env python3
"""Render bound sensitivity for the Allen V1 RF-size translation pilot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
BOUNDS = (10, 15, 20, 30)


def main() -> None:
    summary = pd.read_csv(AUDIT / "v1_rf_size_translation_bound_sensitivity.csv")
    transforms = {
        bound: pd.read_csv(
            AUDIT / f"v1_rf_size_translation_fixed_penalty_bound_{bound}" / "selected_v1_rf_size_translations.csv"
        ).set_index("ecephys_session_id")
        for bound in BOUNDS
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))
    axes[0, 0].plot(summary.bound, summary.med_abs_az, marker="o", linewidth=2, label="Azimuth")
    axes[0, 0].plot(summary.bound, summary.med_abs_el, marker="s", linewidth=2, label="Elevation")
    axes[0, 0].set(xlabel="Allowed translation bound (deg)", ylabel="Median |estimated offset| (deg)", title="Offset magnitude versus allowed range")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(summary.bound, summary.bound_hits, marker="o", color="#a13d2d", linewidth=2)
    axes[0, 1].set(xlabel="Allowed translation bound (deg)", ylabel="Sessions at bound", title="Boundary saturation")
    axes[1, 0].plot(summary.bound, summary.split_az, marker="o", linewidth=2, label="Azimuth")
    axes[1, 0].plot(summary.bound, summary.split_el, marker="s", linewidth=2, label="Elevation")
    axes[1, 0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="Allowed translation bound (deg)", ylabel="Split-half Spearman ρ", title="Independent-unit offset reliability")
    axes[1, 0].legend(frameon=False)
    base = transforms[30]
    axes[1, 1].scatter(base.translation_azimuth_deg, base.translation_elevation_deg, s=40, alpha=.75, color="#6a3d9a")
    axes[1, 1].axhline(0, color="#888888", linewidth=.7)
    axes[1, 1].axvline(0, color="#888888", linewidth=.7)
    axes[1, 1].set(xlabel="Azimuth offset at ±30° bound", ylabel="Elevation offset at ±30° bound", title="Unconstrained-looking endpoint estimates", aspect="equal")
    limit = 32
    axes[1, 1].set(xlim=(-limit, limit), ylim=(-limit, limit))
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.suptitle("Allen BO 1.1 V1 RF-size alignment: fixed-penalty bound sensitivity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    output = AUDIT / "Figure_allen_bo11_v1_rf_size_translation_bound_sensitivity.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    colors = {10: "#4c78a8", 15: "#f58518", 20: "#54a24b", 30: "#b279a2"}
    for column, (axis_name, label) in enumerate((("azimuth", "Azimuth"), ("elevation", "Elevation"))):
        ordered = transforms[30].sort_values(f"translation_{axis_name}_deg").index
        x = np.arange(len(ordered))
        for bound in BOUNDS:
            values = transforms[bound].loc[ordered, f"translation_{axis_name}_deg"]
            axes[0, column].plot(x, values, marker="o", markersize=3, linewidth=1.3, color=colors[bound], alpha=.82, label=f"±{bound}°")
        axes[0, column].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[0, column].set(xlabel="Sessions ordered by ±30° estimate", ylabel=f"{label} offset (deg)", title=f"Per-session {axis_name} estimates")
        axes[0, column].legend(frameon=False, ncol=2, fontsize=8)

        base = transforms[30]
        half = pd.read_csv(
            AUDIT / "v1_rf_size_translation_fixed_penalty_bound_30" / "v1_rf_size_split_half_transforms.csv"
        )
        wide = half.pivot(index="ecephys_session_id", columns="split_half", values=f"translation_{axis_name}_deg")
        axes[1, column].scatter(wide[0], wide[1], s=38, alpha=.72, color=colors[30])
        limits = np.nanpercentile(wide.to_numpy(), [0, 100])
        padding = max(2.0, .08 * np.ptp(limits))
        limits = [limits[0] - padding, limits[1] + padding]
        axes[1, column].plot(limits, limits, color="#333333", linewidth=1)
        axes[1, column].set(xlim=limits, ylim=limits, aspect="equal", xlabel="Offset from unit half 1 (deg)", ylabel="Offset from unit half 2 (deg)", title=f"{label} split-half reproducibility")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    figure.suptitle("Allen BO 1.1 RF-size offsets: session trajectories and reproducibility", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .96))
    detail_output = AUDIT / "Figure_allen_bo11_v1_rf_size_offset_session_detail.png"
    figure.savefig(detail_output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(detail_output)


if __name__ == "__main__":
    main()
