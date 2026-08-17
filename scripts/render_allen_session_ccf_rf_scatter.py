#!/usr/bin/env python3
"""Render cell-level CCF scatter maps colored by RF azimuth and elevation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT = (
    ROOT
    / "artifacts/allen_multisession_rf_validation_v1/07_registration_readiness"
    / "rf_size_visual_anatomy_unit_support.csv"
)
DEFAULT_UNITS = ROOT / "data/unit_table.csv"
DEFAULT_SESSION = 798911424
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/retinotopy_registration_pilot"

AREA_MARKERS = {
    "VISp": "o",
    "VISl": "s",
    "VISal": "^",
    "VISrl": "D",
    "VISam": "P",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(support_path: Path, units_path: Path, session_id: int) -> pd.DataFrame:
    support = pd.read_csv(support_path, low_memory=False)
    units = pd.read_csv(
        units_path,
        usecols=["ecephys_unit_id", "ecephys_probe_id"],
        low_memory=False,
    )
    data = support.loc[support.session_id.eq(session_id) & support.ccf_available].merge(
        units, on="ecephys_unit_id", how="left", validate="one_to_one"
    )
    data = data.loc[data.ecephys_structure_acronym.isin(AREA_MARKERS)].copy()
    if data.empty:
        raise RuntimeError(f"No CCF-matched RF cells for session {session_id}")
    return data.sort_values(["ecephys_probe_id", "ecephys_unit_id"]).reset_index(drop=True)


def render(data: pd.DataFrame, output: Path, session_id: int) -> None:
    azimuth_norm = Normalize(vmin=20, vmax=80, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=40)
    panels = (
        ("visual_azimuth_deg", "RF azimuth", "viridis", azimuth_norm, "degrees"),
        ("visual_elevation_deg", "RF elevation", "coolwarm", elevation_norm, "degrees"),
    )
    x_limits = (data.ccf_ap_mm.min() - 0.12, data.ccf_ap_mm.max() + 0.12)
    y_limits = (data.ccf_ml_mm.min() - 0.12, data.ccf_ml_mm.max() + 0.12)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), constrained_layout=True)
    for axis, (field, title, cmap, norm, colorbar_label) in zip(axes, panels):
        for area, group in data.groupby("ecephys_structure_acronym", sort=True):
            axis.scatter(
                group.ccf_ap_mm,
                group.ccf_ml_mm,
                c=group[field],
                cmap=cmap,
                norm=norm,
                marker=AREA_MARKERS[area],
                s=31,
                alpha=0.82,
                linewidths=0.25,
                edgecolors="#252525",
                rasterized=True,
            )

        probe_centers = (
            data.groupby("ecephys_probe_id", as_index=False)
            .agg(ccf_ap_mm=("ccf_ap_mm", "median"), ccf_ml_mm=("ccf_ml_mm", "median"))
        )
        axis.scatter(
            probe_centers.ccf_ap_mm,
            probe_centers.ccf_ml_mm,
            marker="o",
            s=115,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.1,
            zorder=5,
        )
        for row in probe_centers.itertuples():
            axis.text(
                row.ccf_ap_mm + 0.025,
                row.ccf_ml_mm + 0.025,
                str(int(row.ecephys_probe_id))[-3:],
                fontsize=7,
                color="#222222",
                zorder=6,
            )

        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label(colorbar_label)
        axis.set(
            title=title,
            xlabel="Anterior–posterior CCF (mm)",
            ylabel="Medial–lateral CCF (mm)",
            xlim=x_limits,
            ylim=y_limits,
            aspect="equal",
        )
        axis.grid(color="#dddddd", linewidth=0.45, alpha=0.65)
        axis.set_axisbelow(True)

    area_handles = [
        Line2D(
            [],
            [],
            marker=AREA_MARKERS[area],
            markerfacecolor="#a9a9a9",
            markeredgecolor="#252525",
            markeredgewidth=0.4,
            linestyle="",
            markersize=7,
            label=area,
        )
        for area in sorted(AREA_MARKERS)
        if area in set(data.ecephys_structure_acronym)
    ]
    axes[1].legend(
        handles=area_handles,
        title="Allen area (marker)",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    figure.suptitle(
        f"Session {session_id}: cell RF centers over CCF surface coordinates\n"
        f"trusted aperture RF fits with released CCF coordinates · n={len(data)} cells · "
        f"{data[['ccf_ap_mm', 'ccf_ml_mm']].drop_duplicates().shape[0]} unique AP/ML positions",
        fontsize=14,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    support_path = args.support.resolve()
    units_path = args.units.resolve()
    output = (args.output_root / f"session_{args.session_id}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(support_path, units_path, args.session_id)

    support_columns = [
        "session_id",
        "ecephys_unit_id",
        "ecephys_probe_id",
        "ecephys_structure_acronym",
        "ccf_ap_mm",
        "ccf_ml_mm",
        "visual_azimuth_deg",
        "visual_elevation_deg",
        "axis_area_deg2",
        "axis_edge_distance_deg",
        "axis_test_deviance",
        "unit_split",
    ]
    data[support_columns].to_csv(
        output / "cell_scatter_CCF_RF_support.csv", index=False, float_format="%.9g"
    )
    render(data, output / "Figure_cell_scatter_CCF_colored_by_RF.png", args.session_id)

    manifest = {
        "session_id": args.session_id,
        "status": "descriptive cell-level view; no spatial independence claim",
        "sources": {
            "unit_support": {"path": str(support_path), "sha256": sha256(support_path)},
            "unit_table": {"path": str(units_path), "sha256": sha256(units_path)},
        },
        "counts": {
            "cells": len(data),
            "unique_ccf_ap_ml_positions": int(
                data[["ccf_ap_mm", "ccf_ml_mm"]].drop_duplicates().shape[0]
            ),
            "probes": int(data.ecephys_probe_id.nunique()),
            "areas": int(data.ecephys_structure_acronym.nunique()),
        },
        "chart_contract": {
            "question": "How do cell RF azimuth and elevation vary over the released AP/ML CCF positions in one session?",
            "takeaway": "Show the observed cell-level spatial structure without treating overlapping cells as independent cortical landmarks.",
            "family": "paired spatial scatter",
            "grain": "one trusted aperture-RF unit",
            "renderer": "static Matplotlib",
            "shared_axes": True,
            "color": {
                "azimuth": "viridis, clipped display range 20–80 degrees",
                "elevation": "coolwarm, centered at 0 degrees, display range −20–40 degrees",
            },
            "non_color_encoding": "marker shape identifies Allen area; open circles and numeric suffixes identify penetration medians",
            "output": "Figure_cell_scatter_CCF_colored_by_RF.png",
        },
    }
    (output / "cell_scatter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
