#!/usr/bin/env python3
"""Render raw stacked Allen V1 RF-size gradients across border exclusions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.render_allen_bo11_registration_comparison import DEFAULT_CCF_TRANSFORMS, load_ccf_parameters
from scripts.render_allen_bo11_v1_rf_size_interior import DEFAULT_INPUT, prepare_population, session_balanced_surface
from scripts.render_allen_bo11_rf_size_registration_breakout import build_absolute_size_maps
from scripts.allen_bo11_tuning_driven_limited_affine import load_maps, template_from_maps
from scripts.render_allen_bo11_registration_comparison import DEFAULT_SURFACE_GRID


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
OUTPUT = AUDIT / "v1_rf_size_border_exclusion_surfaces"
CUTOFFS = (20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0)
BANDWIDTH_DEG = 8.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tuning_maps, _, _ = load_maps(DEFAULT_SURFACE_GRID)
    all_sessions = sorted({key[0] for key in tuning_maps})
    _, _, sessions = load_ccf_parameters(DEFAULT_CCF_TRANSFORMS, all_sessions)
    population = prepare_population(pd.read_csv(DEFAULT_INPUT, low_memory=False))
    population = population.loc[population["ecephys_session_id"].isin(sessions)].copy()
    az_grid = np.linspace(10, 90, 65)
    el_grid = np.linspace(-30, 50, 65)
    absolute = {}
    standardized = {}
    effective = {}
    rows = []
    gradient_rows = []
    for cutoff in CUTOFFS:
        selected = population.loc[population["distance_to_nearest_grid_edge_deg"].ge(cutoff)].copy()
        maps = build_absolute_size_maps(selected, sessions, az_grid, el_grid)
        template = template_from_maps(maps, "V1", "rf_size_absolute")
        absolute[cutoff] = np.exp2(template["value"])
        standard, local_effective = session_balanced_surface(
            selected,
            az_grid,
            el_grid,
            BANDWIDTH_DEG,
            minimum_effective_sessions=5.0,
        )
        standardized[cutoff] = standard
        effective[cutoff] = local_effective
        az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
        geometric_support = (
            (az_mesh >= 10 + cutoff)
            & (az_mesh <= 90 - cutoff)
            & (el_mesh >= -30 + cutoff)
            & (el_mesh <= 50 - cutoff)
        )
        absolute[cutoff] = np.where(geometric_support, absolute[cutoff], np.nan)
        standardized[cutoff] = np.where(geometric_support, standardized[cutoff], np.nan)
        effective[cutoff] = np.where(geometric_support, effective[cutoff], np.nan)
        for session_id, group in selected.groupby("ecephys_session_id", observed=True):
            if len(group) < 10:
                continue
            x = group[["azimuth_rf", "elevation_rf"]].to_numpy(float)
            x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
            y = group["session_standardized_log2_rf_area"].to_numpy(float)
            coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
            gradient_rows.append(
                {
                    "border_exclusion_deg": cutoff,
                    "ecephys_session_id": session_id,
                    "units": len(group),
                    "standardized_azimuth_slope": coefficients[1],
                    "standardized_elevation_slope": coefficients[2],
                }
            )
        rows.append(
            {
                "border_exclusion_deg": cutoff,
                "units": len(selected),
                "sessions": selected["ecephys_session_id"].nunique(),
                "supported_grid_fraction": np.mean(np.isfinite(standard)),
                "absolute_area_p10_deg2": np.nanquantile(absolute[cutoff], .10),
                "absolute_area_p90_deg2": np.nanquantile(absolute[cutoff], .90),
                "standardized_range_p10_p90_iqr": np.nanquantile(standard, .90) - np.nanquantile(standard, .10),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT / "v1_rf_size_border_exclusion_surface_audit.csv", index=False, float_format="%.6g")
    gradients = pd.DataFrame(gradient_rows)
    gradients.to_csv(OUTPUT / "v1_rf_size_border_exclusion_session_gradients.csv", index=False, float_format="%.6g")
    gradient_summary = (
        gradients.groupby("border_exclusion_deg", observed=True)
        .agg(
            sessions=("ecephys_session_id", "nunique"),
            median_azimuth_slope=("standardized_azimuth_slope", "median"),
            q25_azimuth_slope=("standardized_azimuth_slope", lambda x: x.quantile(.25)),
            q75_azimuth_slope=("standardized_azimuth_slope", lambda x: x.quantile(.75)),
            median_elevation_slope=("standardized_elevation_slope", "median"),
            q25_elevation_slope=("standardized_elevation_slope", lambda x: x.quantile(.25)),
            q75_elevation_slope=("standardized_elevation_slope", lambda x: x.quantile(.75)),
        )
        .reset_index()
    )
    gradient_summary.to_csv(OUTPUT / "v1_rf_size_border_exclusion_gradient_summary.csv", index=False, float_format="%.6g")

    all_absolute = np.concatenate([value[np.isfinite(value)] for value in absolute.values()])
    absolute_limits = np.quantile(all_absolute, [.02, .98])
    all_standard = np.concatenate([value[np.isfinite(value)] for value in standardized.values()])
    standard_limit = max(float(np.quantile(np.abs(all_standard), .98)), .1)
    figure, axes = plt.subplots(2, len(CUTOFFS), figsize=(23.0, 8.2), sharex=True, sharey=True)
    for column, cutoff in enumerate(CUTOFFS):
        absolute_artist = axes[0, column].pcolormesh(
            az_grid,
            el_grid,
            absolute[cutoff],
            shading="gouraud",
            cmap="YlGnBu",
            norm=Normalize(*absolute_limits),
        )
        standard_artist = axes[1, column].pcolormesh(
            az_grid,
            el_grid,
            standardized[cutoff],
            shading="gouraud",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-standard_limit, vcenter=0, vmax=standard_limit),
        )
        for row in range(2):
            axes[row, column].contour(
                az_grid,
                el_grid,
                effective[cutoff],
                levels=[5, 10, 20],
                colors="#333333",
                linewidths=.55,
                alpha=.48,
            )
            axes[row, column].set(aspect="equal", xlim=(10, 90), ylim=(-30, 50))
            axes[row, column].grid(alpha=.14)
        axes[0, column].set_title(
            f"Exclude <{cutoff:g}° from border\n{len(population.loc[population['distance_to_nearest_grid_edge_deg'].ge(cutoff)]):,} units",
            fontsize=10,
        )
        axes[1, column].set_xlabel("RF azimuth (deg)")
    axes[0, 0].set_ylabel("Absolute fitted RF area\nRF elevation (deg)")
    axes[1, 0].set_ylabel("Within-session standardized RF area\nRF elevation (deg)")
    figure.colorbar(absolute_artist, ax=axes[0, :].tolist(), fraction=.012, pad=.018, label="Fitted RF area (deg²)", extend="both")
    figure.colorbar(standard_artist, ax=axes[1, :].tolist(), fraction=.012, pad=.018, label="Standardized log₂ RF area (IQR units)", extend="both")
    figure.suptitle(
        "Allen BO 1.1 raw stacked V1 RF-size gradient across stimulus-border exclusions\n"
        "shared axes, smoothing, session support, and row-specific color scales",
        fontsize=15,
    )
    figure.subplots_adjust(left=.06, right=.92, bottom=.08, top=.87, wspace=.10, hspace=.22)
    figure_path = OUTPUT / "Figure_allen_bo11_v1_rf_size_border_exclusion_surfaces.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    gradient_figure, gradient_axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, label, color in (
        (gradient_axes[0], "azimuth", "#4c78a8"),
        (gradient_axes[1], "elevation", "#e45756"),
    ):
        median = gradient_summary[f"median_{label}_slope"]
        q25 = gradient_summary[f"q25_{label}_slope"]
        q75 = gradient_summary[f"q75_{label}_slope"]
        ax.fill_between(gradient_summary["border_exclusion_deg"], q25, q75, color=color, alpha=.2)
        ax.plot(gradient_summary["border_exclusion_deg"], median, color=color, marker="o", linewidth=2)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set(xlabel="Border exclusion (deg)", title=f"Within-session {label} gradient")
        ax.grid(alpha=.18)
    gradient_axes[0].set_ylabel("Median standardized plane slope\n(IQR RF-size units per SD position)")
    gradient_figure.suptitle("Interior V1 RF-size gradient stability beyond 20° exclusion", fontsize=14)
    gradient_figure.tight_layout(rect=(0, 0, 1, .94))
    gradient_figure.savefig(OUTPUT / "Figure_allen_bo11_v1_rf_size_border_exclusion_gradient_stability.png", dpi=180, bbox_inches="tight")
    plt.close(gradient_figure)
    report = [
        "# Allen BO 1.1 raw V1 RF-size gradient across border exclusions",
        "",
        "Columns progressively remove RF centers near the released RF-stimulus boundary; all panels remain on the full 10–90° azimuth × -30–50° elevation canvas.",
        "The top row preserves absolute fitted RF area. The bottom row removes each session's median and scales by its IQR before session-balanced smoothing.",
        "Gray contours mark effective session support (5, 10, and 20 sessions). Shared row-specific color scales make gradients comparable across exclusions.",
        "Cells outside the geometrically eligible RF-center rectangle at each cutoff are masked; no kernel extrapolation beyond the retained center support is displayed.",
        "A separate stability figure fits RF size jointly against standardized azimuth and elevation within each session, then summarizes slopes across sessions.",
    ]
    (OUTPUT / "ALLEN_BO11_V1_RF_SIZE_BORDER_EXCLUSION_SURFACES.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "checkpoint": "06c_allen_bo11_v1_rf_size_border_exclusion_surfaces",
        "inputs": {"support": {"path": str(DEFAULT_INPUT), "sha256": sha256(DEFAULT_INPUT)}},
        "parameters": {"sessions": sessions, "border_exclusions_deg": CUTOFFS, "bandwidth_deg": BANDWIDTH_DEG, "grid_limits_deg": {"azimuth": [10, 90], "elevation": [-30, 50]}},
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
