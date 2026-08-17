#!/usr/bin/env python3
"""Plot corrected RF-size surfaces in visual-field and anatomical CCF frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS = (
    ROOT
    / "artifacts"
    / "allen_multisession_rf_validation_v1"
    / "03_geometry"
    / "all_session_unit_geometry_fits.csv"
)
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "allen_multisession_rf_validation_v1"
    / "07_registration_readiness"
)
SESSIONS = (746083955, 755434585, 760693773, 798911424)
GROUPS = ("V1", "HVA")
# Rounded outward from the observed finite smoothed-surface range
# (8.056–9.994) so every map and page has the same interpretable scale.
COLOR_LIMITS = (8.0, 10.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_population(fits_path: Path, unit_path: Path) -> pd.DataFrame:
    fits = pd.read_csv(fits_path, low_memory=False)
    fits = fits.loc[
        fits.spatial_model.eq("aperture")
        & ~fits.axis_censored.astype(bool)
        & fits.axis_edge_distance_deg.gt(10)
    ].copy()
    coordinates = pd.read_csv(
        unit_path,
        usecols=[
            "ecephys_unit_id",
            "anterior_posterior_ccf_coordinate",
            "left_right_ccf_coordinate",
        ],
        low_memory=False,
    )
    data = fits.merge(coordinates, on="ecephys_unit_id", how="left", validate="many_to_one")
    data["visual_azimuth_deg"] = data.axis_center_x_deg + 50
    data["visual_elevation_deg"] = data.axis_center_y_deg + 10
    data["ccf_ap_mm"] = data.anterior_posterior_ccf_coordinate / 1000
    data["ccf_ml_mm"] = data.left_right_ccf_coordinate / 1000
    data["log2_area_deg2"] = np.log2(data.axis_area_deg2)
    data["ccf_available"] = data[["ccf_ap_mm", "ccf_ml_mm"]].notna().all(axis=1)
    return data


def kernel_surface(local, x_grid, y_grid, x_name, y_name, bandwidth, radius):
    points = local[[x_name, y_name]].to_numpy(float)
    values = local.log2_area_deg2.to_numpy(float)
    surface = np.full((len(y_grid), len(x_grid)), np.nan)
    effective = np.zeros_like(surface)
    for row, y_value in enumerate(y_grid):
        for column, x_value in enumerate(x_grid):
            distance = np.sqrt(np.sum((points - [x_value, y_value]) ** 2, axis=1))
            weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
            if weights.sum() > 0:
                effective[row, column] = weights.sum() ** 2 / np.square(weights).sum()
            if effective[row, column] >= 3 and np.sum(distance <= radius) >= 3:
                surface[row, column] = np.average(values, weights=weights)
            else:
                effective[row, column] = 0
    return surface, effective


def grids(data):
    # Show the full stimulus neighborhood rather than cropping closely around
    # the trusted fitted centers.
    visual = (np.linspace(-10, 110, 49), np.linspace(-50, 70, 49))
    ccf = data.loc[data.ccf_available]
    # The anatomical kernel has a 0.55-mm local-support radius. A 0.75-mm
    # margin ensures the supported field closes before reaching the axes.
    ap_limits = np.array([ccf.ccf_ap_mm.min(), ccf.ccf_ap_mm.max()]) + np.array([-0.75, 0.75])
    ml_limits = np.array([ccf.ccf_ml_mm.min(), ccf.ccf_ml_mm.max()]) + np.array([-0.75, 0.75])
    anatomy = (np.linspace(*ap_limits, 49), np.linspace(*ml_limits, 49))
    return visual, anatomy


def build_surface(local, frame, frame_grids):
    if frame == "visual":
        x_grid, y_grid = frame_grids
        return kernel_surface(
            local,
            x_grid,
            y_grid,
            "visual_azimuth_deg",
            "visual_elevation_deg",
            bandwidth=15,
            radius=20,
        )
    x_grid, y_grid = frame_grids
    return kernel_surface(
        local,
        x_grid,
        y_grid,
        "ccf_ap_mm",
        "ccf_ml_mm",
        bandwidth=0.35,
        radius=0.55,
    )


def add_surface(axis, x_grid, y_grid, surface, effective, vmin, vmax):
    image = axis.pcolormesh(
        x_grid, y_grid, surface, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    maximum = np.nanmax(effective)
    levels = [value for value in (3, 6, 12, 24) if value <= maximum]
    if levels:
        axis.contour(x_grid, y_grid, effective, levels=levels, colors="#333333", linewidths=0.55)
    return image


def pooled_surfaces(data, visual_grids, anatomy_grids):
    maps = {}
    available_sessions = tuple(
        int(value)
        for value in sorted(data.loc[data.ccf_available, "session_id"].unique())
        if data.loc[data.session_id.eq(value) & data.ccf_available].groupby("group").size().min() >= 3
    )
    for group in GROUPS:
        local = data.loc[
            data.group.eq(group)
            & data.session_id.isin(available_sessions)
            & data.ccf_available
        ]
        maps[(group, "visual")] = (*build_surface(local, "visual", visual_grids), len(local))
        maps[(group, "anatomy")] = (*build_surface(local, "anatomy", anatomy_grids), len(local))
    return maps, available_sessions


def render_pooled(data, visual_grids, anatomy_grids, path):
    maps, sessions = pooled_surfaces(data, visual_grids, anatomy_grids)
    vmin, vmax = COLOR_LIMITS
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), constrained_layout=True)
    for row, group in enumerate(GROUPS):
        visual, visual_support, count = maps[(group, "visual")]
        image = add_surface(axes[row, 0], *visual_grids, visual, visual_support, vmin, vmax)
        axes[row, 0].set(
            title=f"{group} · visual field · n={count}",
            xlabel="RF azimuth (deg)",
            ylabel="RF elevation (deg)",
            aspect="equal",
        )
        anatomy, anatomy_support, count = maps[(group, "anatomy")]
        add_surface(axes[row, 1], *anatomy_grids, anatomy, anatomy_support, vmin, vmax)
        axes[row, 1].set(
            title=f"{group} · CCF anatomy · n={count}",
            xlabel="Anterior–posterior CCF (mm)",
            ylabel="Medial–lateral CCF (mm)",
            aspect="equal",
        )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Smoothed log₂ aperture RF area (deg²)")
    session_text = ", ".join(str(value) for value in sessions)
    fig.suptitle(
        f"Corrected RF-size surfaces in visual and anatomical coordinates\n"
        f"same trusted CCF-matched neurons; sessions {session_text}",
        fontsize=15,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return sessions


def render_by_session(data, sessions, visual_grids, anatomy_grids, path):
    maps = {}
    for session_id in sessions:
        for group in GROUPS:
            local = data.loc[
                data.session_id.eq(session_id) & data.group.eq(group) & data.ccf_available
            ]
            maps[(session_id, group, "visual")] = (*build_surface(local, "visual", visual_grids), len(local))
            maps[(session_id, group, "anatomy")] = (*build_surface(local, "anatomy", anatomy_grids), len(local))
    vmin, vmax = COLOR_LIMITS
    columns = (("V1", "visual"), ("V1", "anatomy"), ("HVA", "visual"), ("HVA", "anatomy"))
    fig, axes = plt.subplots(len(sessions), 4, figsize=(15, 4 * len(sessions)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, session_id in enumerate(sessions):
        for column, (group, frame) in enumerate(columns):
            surface, effective, count = maps[(session_id, group, frame)]
            frame_grids = visual_grids if frame == "visual" else anatomy_grids
            image = add_surface(axes[row, column], *frame_grids, surface, effective, vmin, vmax)
            axes[row, column].set_title(f"{session_id} · {group} · {frame}\nn={count}")
            axes[row, column].set_aspect("equal")
            if frame == "visual":
                axes[row, column].set_xlabel("RF azimuth (deg)")
                axes[row, column].set_ylabel("RF elevation (deg)")
            else:
                axes[row, column].set_xlabel("AP CCF (mm)")
                axes[row, column].set_ylabel("ML CCF (mm)")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Smoothed log₂ aperture RF area (deg²)")
    fig.suptitle("Per-session RF-size surfaces in visual and CCF anatomical frames", fontsize=15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_multipage_pdf(data, visual_grids, anatomy_grids, path):
    """Write one page per session, retaining explicit CCF-coverage gaps."""
    vmin, vmax = COLOR_LIMITS
    with PdfPages(path) as pdf:
        for session_id in SESSIONS:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
            image = None
            for row, group in enumerate(GROUPS):
                visual = data.loc[data.session_id.eq(session_id) & data.group.eq(group)]
                visual_surface, visual_support = build_surface(visual, "visual", visual_grids)
                image = add_surface(
                    axes[row, 0],
                    *visual_grids,
                    visual_surface,
                    visual_support,
                    vmin,
                    vmax,
                )
                axes[row, 0].set(
                    title=f"{group} · visual field · trusted n={len(visual)}",
                    xlabel="RF azimuth (deg)",
                    ylabel="RF elevation (deg)",
                    aspect="equal",
                )

                anatomy = visual.loc[visual.ccf_available]
                axes[row, 1].set(
                    xlabel="Anterior–posterior CCF (mm)",
                    ylabel="Medial–lateral CCF (mm)",
                    aspect="equal",
                    xlim=(anatomy_grids[0][0], anatomy_grids[0][-1]),
                    ylim=(anatomy_grids[1][0], anatomy_grids[1][-1]),
                )
                if len(anatomy) >= 3:
                    anatomy_surface, anatomy_support = build_surface(
                        anatomy, "anatomy", anatomy_grids
                    )
                    add_surface(
                        axes[row, 1],
                        *anatomy_grids,
                        anatomy_surface,
                        anatomy_support,
                        vmin,
                        vmax,
                    )
                    axes[row, 1].set_title(f"{group} · CCF anatomy · n={len(anatomy)}")
                else:
                    axes[row, 1].set_title(f"{group} · CCF anatomy · n={len(anatomy)}")
                    axes[row, 1].text(
                        0.5,
                        0.5,
                        "Insufficient released CCF coordinates",
                        ha="center",
                        va="center",
                        transform=axes[row, 1].transAxes,
                        fontsize=12,
                        color="#555555",
                    )
            colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
            colorbar.set_label("Smoothed log₂ aperture RF area (deg²)")
            fig.suptitle(
                f"Session {session_id}: RF-size surfaces in visual and anatomical frames\n"
                f"fixed smoothed-surface color range {vmin:.1f}–{vmax:.1f} log₂(deg²)",
                fontsize=15,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def export_grid(data, sessions, visual_grids, anatomy_grids, path):
    rows = []
    for session_id in sessions:
        for group in GROUPS:
            local = data.loc[
                data.session_id.eq(session_id) & data.group.eq(group) & data.ccf_available
            ]
            for frame, frame_grids in (("visual", visual_grids), ("anatomy_ccf", anatomy_grids)):
                surface, effective = build_surface(
                    local, "visual" if frame == "visual" else "anatomy", frame_grids
                )
                for row, y_value in enumerate(frame_grids[1]):
                    for column, x_value in enumerate(frame_grids[0]):
                        rows.append(
                            {
                                "session_id": session_id,
                                "group": group,
                                "coordinate_frame": frame,
                                "x": x_value,
                                "y": y_value,
                                "log2_aperture_area_deg2": surface[row, column],
                                "effective_units": effective[row, column],
                                "source_units": len(local),
                            }
                        )
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.9g")


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_population(args.fits.resolve(), args.units.resolve())
    visual_grids, anatomy_grids = grids(data)
    sessions = render_pooled(
        data,
        visual_grids,
        anatomy_grids,
        output / "Figure_rf_size_visual_vs_anatomy_pooled.png",
    )
    render_by_session(
        data,
        sessions,
        visual_grids,
        anatomy_grids,
        output / "Figure_rf_size_visual_vs_anatomy_by_session.png",
    )
    render_multipage_pdf(
        data,
        visual_grids,
        anatomy_grids,
        output / "RF_size_visual_and_anatomy_maps_by_session.pdf",
    )
    export_grid(
        data,
        sessions,
        visual_grids,
        anatomy_grids,
        output / "rf_size_visual_anatomy_surface_grid.csv",
    )
    data.to_csv(output / "rf_size_visual_anatomy_unit_support.csv", index=False, float_format="%.9g")
    print(
        data.groupby(["session_id", "group"], observed=True)
        .agg(trusted_units=("ecephys_unit_id", "size"), ccf_units=("ccf_available", "sum"))
        .to_string()
    )


if __name__ == "__main__":
    main()
