#!/usr/bin/env python3
"""Show all cells after the selected anatomy-constrained CCF→Zhuang affine."""

from __future__ import annotations

import argparse
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
DEFAULT_SESSION = 798911424
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/retinotopy_registration_pilot"
DEFAULT_TEMPLATE = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
)

AREA_MARKERS = {"VISp": "o", "VISl": "s", "VISal": "^", "VISrl": "D", "VISam": "P"}
AREA_LABELS = {"VISp": "V1", "VISl": "LM", "VISal": "AL", "VISrl": "RL", "VISam": "AM"}
AREA_SEEDS_XY = {
    "VISp": (200, 240),
    "VISl": (100, 260),
    "VISal": (75, 190),
    "VISrl": (180, 80),
    "VISam": (240, 80),
}
ZHUANG_FIG3_SCALE_BAR_PX = 62.0
ZHUANG_FIG3_SCALE_BAR_MM = 0.5
FIG3_TO_FIG9_SIMILARITY_SCALE = 0.8432313316638625
MEDIAL_DISPLAY_CUTOFF_ML_MM = 6.75


def background_layers(
    axis,
    template: dict[str, np.ndarray],
    metric: str,
    cmap: str,
    norm,
    template_to_ccf,
) -> None:
    boundary = template["mean_field_sign_boundary"].astype(bool)
    values = template[metric]
    value_y, value_x = np.nonzero(np.isfinite(values))
    value_ccf = template_to_ccf(np.column_stack([value_x, value_y]))
    axis.scatter(
        value_ccf[:, 0],
        value_ccf[:, 1],
        c=values[value_y, value_x],
        cmap=cmap,
        norm=norm,
        marker="s",
        s=2.2,
        alpha=0.72,
        linewidths=0,
        zorder=1,
        rasterized=True,
    )
    height, width = boundary.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_ccf = template_to_ccf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    grid_ap = grid_ccf[:, 0].reshape(boundary.shape)
    grid_ml = grid_ccf[:, 1].reshape(boundary.shape)
    axis.contour(
        grid_ap,
        grid_ml,
        boundary.astype(float),
        levels=[0.5],
        colors="#343434",
        linewidths=0.55,
        zorder=2,
    )
    for acronym, (x, y) in AREA_SEEDS_XY.items():
        label_ap, label_ml = template_to_ccf(np.array([[x, y]], dtype=float))[0]
        axis.text(
            label_ap,
            label_ml,
            AREA_LABELS[acronym],
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 1.0},
            zorder=2,
        )


def render(
    cells: pd.DataFrame,
    template: dict[str, np.ndarray],
    fit: dict,
    output: Path,
    session_id: int,
) -> None:
    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=40)
    template_px_per_mm = (
        ZHUANG_FIG3_SCALE_BAR_PX
        / ZHUANG_FIG3_SCALE_BAR_MM
        * FIG3_TO_FIG9_SIMILARITY_SCALE
    )
    panels = (
        (
            "observed_azimuth_deg",
            "azimuth_deg",
            "RF azimuth and Zhuang azimuth contours",
            "viridis",
            azimuth_norm,
        ),
        (
            "observed_elevation_deg",
            "altitude_deg",
            "RF elevation and Zhuang altitude contours",
            "coolwarm",
            elevation_norm,
        ),
    )
    center = np.asarray(fit["ccf_center_ap_ml_mm"], dtype=float)
    template_center = np.asarray(fit["template_center_xy_px"], dtype=float)
    matrix = np.asarray(fit["affine_matrix_xy_px_per_ap_ml_mm"], dtype=float)
    inverse_matrix = np.linalg.inv(matrix)

    def template_to_ccf(points_xy: np.ndarray) -> np.ndarray:
        ap_ml = center + (np.asarray(points_xy, dtype=float) - template_center) @ inverse_matrix.T
        return ap_ml[:, [1, 0]]  # Plot ML horizontally and AP vertically.

    boundary = template["mean_field_sign_boundary"].astype(bool)
    boundary_y, boundary_x = np.nonzero(boundary)
    warped_boundary = template_to_ccf(np.column_stack([boundary_x, boundary_y]))
    all_ml = np.r_[warped_boundary[:, 0], cells.ccf_ml_mm.to_numpy()]
    all_ap = np.r_[warped_boundary[:, 1], cells.ccf_ap_mm.to_numpy()]
    # Bound the actual warped cortex rather than the unused rectangular image canvas.
    # Equal aspect below guarantees that one millimetre has the same rendered length
    # on both axes; independent tight limits avoid unnecessary blank anatomy.
    padding_mm = 0.12
    x_limits = (
        max(all_ml.min() - padding_mm, MEDIAL_DISPLAY_CUTOFF_ML_MM),
        all_ml.max() + padding_mm,
    )
    y_limits = (all_ap.min() - padding_mm, all_ap.max() + padding_mm)
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), constrained_layout=True)

    for axis, (field, template_metric, title, cmap, norm) in zip(axes, panels):
        background_layers(axis, template, template_metric, cmap, norm, template_to_ccf)
        for area, group in cells.groupby("ecephys_structure_acronym", sort=True):
            axis.scatter(
                group.ccf_ml_mm,
                group.ccf_ap_mm,
                c=group[field],
                cmap=cmap,
                norm=norm,
                marker=AREA_MARKERS[area],
                s=28,
                alpha=0.84,
                linewidths=0.3,
                edgecolors="#202020",
                zorder=4,
                rasterized=True,
            )

        probe_centers = (
            cells.groupby("ecephys_probe_id", as_index=False)
            .agg(ccf_ap_mm=("ccf_ap_mm", "median"), ccf_ml_mm=("ccf_ml_mm", "median"))
        )
        axis.scatter(
            probe_centers.ccf_ml_mm,
            probe_centers.ccf_ap_mm,
            marker="o",
            s=115,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.1,
            zorder=5,
        )
        for row in probe_centers.itertuples():
            axis.text(
                row.ccf_ml_mm + 0.025,
                row.ccf_ap_mm + 0.025,
                str(int(row.ecephys_probe_id))[-3:],
                fontsize=7,
                color="#111111",
                zorder=6,
            )
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees; shared by cells and atlas contours")
        axis.set(
            title=title,
            xlabel="Medial–lateral CCF (mm)",
            ylabel="Anterior–posterior CCF (mm)",
            xlim=x_limits[::-1],
            ylim=y_limits[::-1],
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        axis.set_axisbelow(True)

    area_handles = [
        Line2D(
            [],
            [],
            marker=AREA_MARKERS[area],
            markerfacecolor="#aaaaaa",
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            linestyle="",
            markersize=7,
            label=f"{area}→{AREA_LABELS[area]}",
        )
        for area in sorted(AREA_MARKERS)
        if area in set(cells.ecephys_structure_acronym)
    ]
    axes[1].legend(
        handles=area_handles,
        title="Marker = Allen CCF area",
        loc="upper right",
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )

    singular_px_per_mm = np.linalg.svd(matrix, compute_uv=False)
    equivalent_px_per_mm = float(np.sqrt(abs(np.linalg.det(matrix))))
    figure.suptitle(
        f"Session {session_id}: cells fixed in CCF anatomy; Zhuang map inverse-warped into CCF\n"
        f"n={len(cells)} cells on six penetrations · contours and cells share one color scale per panel\n"
        f"template calibration {template_px_per_mm:.1f} px/mm · fitted CCF→template principal scales "
        f"{singular_px_per_mm[0]:.1f}, {singular_px_per_mm[1]:.1f} px/mm · native azimuth convention",
        fontsize=12.5,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    output = (args.output_root / f"session_{args.session_id}").resolve()
    residual_path = output / "unit_registration_residuals.csv.gz"
    manifest_path = output / "run_manifest.json"
    template_path = args.template.resolve()
    cells = pd.read_csv(residual_path)
    cells = cells.loc[cells.model.eq("joint_anatomy_rf")].copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit = manifest["selected_models"]["joint_anatomy_rf"]
    template_npz = np.load(template_path)
    template = {name: template_npz[name] for name in template_npz.files}

    figure_path = output / "Figure_CCF_cells_with_inverse_warped_Zhuang_map.png"
    render(cells, template, fit, figure_path, args.session_id)
    matrix = np.asarray(fit["affine_matrix_xy_px_per_ap_ml_mm"], dtype=float)
    singular_px_per_mm = np.linalg.svd(matrix, compute_uv=False)
    equivalent_px_per_mm = float(np.sqrt(abs(np.linalg.det(matrix))))
    template_px_per_mm = (
        ZHUANG_FIG3_SCALE_BAR_PX
        / ZHUANG_FIG3_SCALE_BAR_MM
        * FIG3_TO_FIG9_SIMILARITY_SCALE
    )
    scale_audit = {
        "zhuang_figure3_scale_bar": {
            "length_px": ZHUANG_FIG3_SCALE_BAR_PX,
            "length_mm": ZHUANG_FIG3_SCALE_BAR_MM,
            "px_per_mm": ZHUANG_FIG3_SCALE_BAR_PX / ZHUANG_FIG3_SCALE_BAR_MM,
        },
        "figure3_to_figure9_similarity": {
            "scale_figure9_px_per_figure3_px": FIG3_TO_FIG9_SIMILARITY_SCALE,
            "method": "symmetric trimmed Chamfer fit of the published mean-area boundary network",
        },
        "zhuang_figure9_template": {
            "px_per_mm": template_px_per_mm,
            "micrometers_per_px": 1000.0 / template_px_per_mm,
        },
        "selected_affine": {
            "principal_px_per_mm": singular_px_per_mm.tolist(),
            "principal_scale_ratios_to_template": (singular_px_per_mm / template_px_per_mm).tolist(),
            "area_equivalent_px_per_mm": equivalent_px_per_mm,
            "area_equivalent_ratio_to_template": equivalent_px_per_mm / template_px_per_mm,
        },
        "interpretation": (
            "The mismatch is anisotropic: one affine principal direction is enlarged while the "
            "orthogonal direction is close to the published template scale."
        ),
    }
    (output / "zhuang_scale_agreement_audit.json").write_text(
        json.dumps(scale_audit, indent=2) + "\n", encoding="utf-8"
    )
    chart_manifest = {
        "session_id": args.session_id,
        "model": "joint_anatomy_rf",
        "cells": len(cells),
        "penetrations": int(cells.ecephys_probe_id.nunique()),
        "fit": fit,
        "sources": {
            "cell_results": str(residual_path),
            "fit_manifest": str(manifest_path),
            "zhuang_template": str(template_path),
        },
        "chart_contract": {
            "question": "How does the average Zhuang retinotopic map warp onto the session's fixed anatomical CCF cell locations?",
            "takeaway": "Keep measured anatomy fixed and express animal/session-specific registration as deformation of the average retinotopic map.",
            "family": "paired spatial scatter over atlas contours",
            "grain": "one trusted aperture-RF unit",
            "renderer": "static Matplotlib",
            "color": "one shared viridis scale for azimuth cells/contours; one shared zero-centered coolwarm scale for elevation/altitude cells/contours",
            "non_color_encoding": "marker shape identifies Allen area; open circles and numeric suffixes mark penetration medians",
            "coordinate_semantics": "cells are plotted at released CCF ML/AP (ML horizontal, AP vertical), with both displayed axes descending high-to-low; all Zhuang layers are mapped into CCF by the exact inverse of the selected CCF-to-template affine",
            "spatial_scaling": "equal data aspect: one millimetre has the same rendered length on ML and AP",
            "axis_limits": "tight bounds of the warped cortical boundary plus cells, padded by 0.12 mm; unused rectangular template-canvas corners are excluded and the medial display edge is cropped at ML=6.75 mm",
            "output": figure_path.name,
        },
    }
    (output / "anatomy_constrained_cell_mapping_manifest.json").write_text(
        json.dumps(chart_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(figure_path), "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
