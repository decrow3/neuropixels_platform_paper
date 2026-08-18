#!/usr/bin/env python3
"""Combined figure comparing the two probe-shank registration methods for a SINGLE animal/session
first (easier to visually inspect than all 32 probes at once): Nelder-Mead regularized fit
(`register_mousev2_units_along_probe_shank.py`, directly optimizes RF-value agreement) vs.
gradient-anchored closed-form fit (`register_mousev2_units_along_probe_shank_gradient_anchor.py`,
anchored to the hard-constrained 6E position with direction/length derived analytically, never
optimized against RF values).

Laid out as ROWS (one method per row), not overlapping on the same axes -- top row Nelder-Mead,
bottom row gradient-anchored, same azimuth/elevation columns, same color scale, so the two are
directly comparable panel-by-panel without one method's lines obscuring the other's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
NM_DIR = ROOT / "artifacts/figure3/06g_mousev2_rf_units_along_probe_shank"
GRAD_DIR = ROOT / "artifacts/figure3/06i_mousev2_gradient_anchored_shank"
OUTPUT = ROOT / "artifacts/figure3/06j_nm_vs_gradient_anchored_comparison"
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
SITE = "site2"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    boundary = template["boundary"].astype(float)
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    height, width = smoothed["azimuth_span_matched_deg"].shape
    dist = template["area_distance_arrays"]["VISp"]
    row_axis, col_axis = np.arange(dist.shape[0]), np.arange(dist.shape[1])
    dist_interp = RegularGridInterpolator((row_axis, col_axis), dist, bounds_error=False, fill_value=100.0)

    nm_lines = pd.read_csv(NM_DIR / "probe_shank_lines.csv")
    grad_lines = pd.read_csv(GRAD_DIR / "probe_shank_lines_gradient_anchored.csv")
    for lines in (nm_lines, grad_lines):
        lines["d_p0"] = dist_interp(lines[["p0_row", "p0_col"]].to_numpy())
        lines["d_p1"] = dist_interp(lines[["p1_row", "p1_col"]].to_numpy())
        lines["max_endpoint_dist"] = lines[["d_p0", "d_p1"]].max(axis=1)

    nm_delta = pd.read_csv(NM_DIR / "session_delta.csv").set_index("site")
    nm_units = pd.read_csv(NM_DIR / "unit_positions_along_shank.csv").merge(nm_delta, on="site", how="left")
    nm_units["resid"] = np.hypot(
        nm_units.predicted_azimuth_deg - nm_units.delta_azimuth_deg - nm_units.observed_azimuth_deg,
        nm_units.predicted_elevation_deg - nm_units.delta_elevation_deg - nm_units.observed_elevation_deg)
    # the background field shows the model's PREDICTED value, which the fit drives toward
    # observed + delta (not raw observed) -- color points by the same delta-corrected quantity
    # the residual is computed against, or a nonzero session delta shows up as a spurious color
    # mismatch that has nothing to do with fit quality.
    nm_units["azimuth_deg_corrected"] = nm_units.observed_azimuth_deg + nm_units.delta_azimuth_deg
    nm_units["elevation_deg_corrected"] = nm_units.observed_elevation_deg + nm_units.delta_elevation_deg

    grad_units = pd.read_csv(GRAD_DIR / "unit_positions_along_shank_gradient_anchored.csv")
    grad_units["resid"] = np.hypot(
        grad_units.predicted_azimuth_deg - grad_units.delta_azimuth_deg - grad_units.observed_azimuth_deg,
        grad_units.predicted_elevation_deg - grad_units.delta_elevation_deg - grad_units.observed_elevation_deg)
    grad_units["azimuth_deg_corrected"] = grad_units.observed_azimuth_deg + grad_units.delta_azimuth_deg
    grad_units["elevation_deg_corrected"] = grad_units.observed_elevation_deg + grad_units.delta_elevation_deg

    nm_lines_s = nm_lines.loc[nm_lines.site == SITE]
    grad_lines_s = grad_lines.loc[grad_lines.site == SITE]
    nm_units_s = nm_units.loc[nm_units.site == SITE]
    grad_units_s = grad_units.loc[grad_units.site == SITE]
    print(f"{SITE}: NM {len(nm_lines_s)} probes / {len(nm_units_s)} units, "
          f"gradient {len(grad_lines_s)} probes / {len(grad_units_s)} units")

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    methods = (
        ("Nelder-Mead", nm_lines_s, nm_units_s),
        ("Gradient-anchored", grad_lines_s, grad_units_s),
    )
    field_specs = (
        ("azimuth_span_matched_deg", "azimuth_deg_corrected", "Azimuth", "viridis", azimuth_norm),
        ("elevation_span_matched_deg", "elevation_deg_corrected", "Elevation", "coolwarm", elevation_norm),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13))
    for row, (method_name, lines_s, units_s) in enumerate(methods):
        for col, (field_key, unit_col, label, cmap, norm) in enumerate(field_specs):
            ax = axes[row, col]
            field = np.where(visp_mask, smoothed[field_key], np.nan)
            im = ax.imshow(field, cmap=cmap, norm=norm, origin="upper", zorder=1)
            ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55, zorder=2)
            for _, prow in lines_s.iterrows():
                color = PROBE_COLORS.get(prow.probe, "black")
                ax.plot([prow.p0_col, prow.p1_col], [prow.p0_row, prow.p1_row], color=color,
                         linewidth=2.2, alpha=0.95, zorder=3)
                ax.scatter([prow.p0_col], [prow.p0_row], marker="o", s=45, color=color,
                            edgecolors="white", linewidths=0.7, zorder=4)
                ax.scatter([prow.p1_col], [prow.p1_row], marker="s", s=45, color=color,
                            edgecolors="white", linewidths=0.7, zorder=4)
            ax.scatter(units_s.inferred_col, units_s.inferred_row, c=units_s[unit_col], cmap=cmap,
                        norm=norm, s=22, edgecolors="white", linewidths=0.4, alpha=0.9, zorder=5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label=f"{label} (deg)")
            median_resid = units_s.resid.median()
            max_dist = lines_s.max_endpoint_dist.max()
            ax.set(title=f"{method_name} -- {label}\nmedian resid={median_resid:.1f} deg, "
                         f"worst endpoint {max_dist:.0f}px outside VISp",
                   xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
            ax.set_xlim(0, width); ax.set_ylim(height, 0)
            if row == 0 and col == 0:
                for probe, color in PROBE_COLORS.items():
                    if probe in lines_s.probe.values:
                        ax.plot([], [], color=color, linewidth=2, label=f"probe {probe}")
                ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"{SITE}: Nelder-Mead (top row) vs. gradient-anchored (bottom row) probe-shank registration\n"
                 "(o=shallowest unit, sq=deepest; point color = observed RF value + session delta, matching what the fit targets)",
                 fontsize=13)
    fig.tight_layout()
    figure_path = OUTPUT / f"Figure_nm_vs_gradient_anchored_{SITE}.png"
    fig.savefig(figure_path, dpi=165)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
