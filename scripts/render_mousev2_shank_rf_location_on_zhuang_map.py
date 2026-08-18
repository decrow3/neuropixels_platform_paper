#!/usr/bin/env python3
"""Re-render the depth-constrained probe-shank registration (`register_mousev2_units_along_probe_shank.py`)
with the Zhuang span-matched azimuth/elevation FIELDS as an actual background heatmap (not just a
boundary contour), and units colored by their own OBSERVED azimuth/elevation on the same
colormap/scale as the background -- so a unit's color can be visually checked against the field
color at the position the shank-line model placed it, the same overlay convention already used in
`render_all_cells_over_zhuang_per_session_registered.py`. Reads the CSVs the fitting script already
wrote; does not refit anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
SHANK_DIR = ROOT / "artifacts/figure3/06g_mousev2_rf_units_along_probe_shank"
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}


def main() -> None:
    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    boundary = template["boundary"].astype(float)
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    height, width = smoothed["azimuth_span_matched_deg"].shape

    probe_lines = pd.read_csv(SHANK_DIR / "probe_shank_lines.csv")
    per_unit = pd.read_csv(SHANK_DIR / "unit_positions_along_shank.csv")
    session_delta = pd.read_csv(SHANK_DIR / "session_delta.csv")
    per_unit = per_unit.merge(session_delta, on="site", how="left")
    # background field shows the model's PREDICTED value, which the fit drives toward
    # observed + delta, not raw observed -- color by the same delta-corrected quantity or a
    # nonzero session delta shows up as a spurious color mismatch unrelated to fit quality.
    per_unit["azimuth_deg_corrected"] = per_unit.observed_azimuth_deg + per_unit.delta_azimuth_deg
    per_unit["elevation_deg_corrected"] = per_unit.observed_elevation_deg + per_unit.delta_elevation_deg
    print(f"probes: {len(probe_lines)}, units: {len(per_unit)}")

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    panels = (
        ("azimuth_span_matched_deg", "azimuth_deg_corrected", "Azimuth", "viridis", azimuth_norm),
        ("elevation_span_matched_deg", "elevation_deg_corrected", "Elevation", "coolwarm", elevation_norm),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), constrained_layout=True)
    for ax, (field_key, unit_col, label, cmap, norm) in zip(axes, panels):
        field = np.where(visp_mask, smoothed[field_key], np.nan)
        im = ax.imshow(field, cmap=cmap, norm=norm, origin="upper", zorder=1)
        ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55, zorder=2)

        for _, row in probe_lines.iterrows():
            color = PROBE_COLORS.get(row.probe, "black")
            ax.plot([row.p0_col, row.p1_col], [row.p0_row, row.p1_row], color=color,
                     linewidth=1.4, alpha=0.9, zorder=3)
            ax.scatter([row.p0_col], [row.p0_row], marker="o", s=22, color=color,
                        edgecolors="white", linewidths=0.5, zorder=4)
            ax.scatter([row.p1_col], [row.p1_row], marker="s", s=22, color=color,
                        edgecolors="white", linewidths=0.5, zorder=4)

        ax.scatter(per_unit.inferred_col, per_unit.inferred_row, c=per_unit[unit_col], cmap=cmap,
                    norm=norm, s=14, edgecolors="white", linewidths=0.3, alpha=0.9, zorder=5)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label=f"{label} (deg)")
        ax.set(title=f"{label}: MouseV2 units (own observed RF value) on shank lines,\nover Zhuang span-matched field",
               xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
        ax.set_xlim(0, width); ax.set_ylim(height, 0)

    for probe, color in PROBE_COLORS.items():
        axes[0].plot([], [], color=color, linewidth=1.4, label=f"probe {probe}")
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle("MouseV2 probe shanks + RF-location scatter over the Zhuang V1 retinotopic map\n"
                  "(o=deepest unit, sq=shallowest; point color = observed RF value + session delta, matching what the fit targets)",
                  fontsize=12.5)
    figure_path = SHANK_DIR / "Figure_mousev2_probe_shanks_rf_location_on_zhuang_map.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
