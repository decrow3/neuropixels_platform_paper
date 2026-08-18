#!/usr/bin/env python3
"""MouseV2 counterpart to render_naive_vs_per_session_translation_comparison.py: NAIVE (one
single pooled RF offset for every session) vs. PER-SESSION (each session's own additional delta,
already fit by register_mousev2_rf_to_zhuang_v1.py) -- same background span-matched Zhuang field,
same color scale, so the two are directly comparable.

Adapted at PROBE level, not unit level -- deliberately, to avoid circularity. The Allen figure
plots each cell at its own independently-known CCF anatomical position (from histology), so
comparing the RF value color at that FIXED position against naive vs. per-session calibration is
a fair, non-circular check. MouseV2 has no such independent per-unit anatomical position -- a
unit's position along its own probe is itself fit FROM RF values elsewhere in this project, so
plotting units that way here would be comparing RF-value-derived positions against RF values,
which would look artificially good by construction. Each PROBE's own anatomical entry point
(`probe_anatomical_position.csv`, from the hand-traced-border + apex-anchor registration --
independent of RF values entirely) is a genuinely independent position to plot against, at the
cost of far fewer points (27 probes vs thousands of cells).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
RF_REGISTRATION_DIR = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1"
ANATOMICAL_PROBE_POSITIONS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/probe_anatomical_position.csv"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
OUTPUT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang"

MIN_UNITS_PER_PROBE = 15


def main() -> None:
    reg_manifest = json.loads((RF_REGISTRATION_DIR / "registration_manifest.json").read_text())
    pooled_offset_az = reg_manifest["calibrated_azimuth_offset_deg"]
    pooled_offset_el = reg_manifest["calibrated_elevation_offset_deg"]
    session_delta = pd.read_csv(RF_REGISTRATION_DIR / "mousev2_session_delta.csv")

    rf = pd.read_csv(RF_FITS, low_memory=False)
    supported = rf.loc[rf.pilot_qc & rf.rf_model_supported]
    probe_rf_rows = []
    for (site, probe), group in supported.groupby(["site", "probe"]):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        raw_az, raw_el = huber_location(group[["supported_rf_center_x_deg", "supported_rf_center_y_deg"]].to_numpy(float))
        probe_rf_rows.append({"site": site, "probe": probe, "n_units": len(group),
                               "raw_azimuth": raw_az, "raw_elevation": raw_el})
    probe_rf = pd.DataFrame(probe_rf_rows)

    anatomical = pd.read_csv(ANATOMICAL_PROBE_POSITIONS)[["site", "probe", "zhuang_row", "zhuang_col"]]
    probes = probe_rf.merge(anatomical, on=["site", "probe"], how="inner")
    probes = probes.merge(session_delta[["site", "delta_azimuth_deg", "delta_elevation_deg"]], on="site", how="left")
    probes["delta_azimuth_deg"] = probes.delta_azimuth_deg.fillna(0.0)
    probes["delta_elevation_deg"] = probes.delta_elevation_deg.fillna(0.0)
    print(f"probes plotted: {len(probes)} (>= {MIN_UNITS_PER_PROBE} units, matched to an anatomical anchor, "
          f"{probes.delta_azimuth_deg.eq(0.0).sum()} missing a session delta -> naive-only for that session)")

    probes["naive_azimuth_deg"] = probes.raw_azimuth + pooled_offset_az
    probes["naive_elevation_deg"] = probes.raw_elevation + pooled_offset_el
    probes["per_session_azimuth_deg"] = probes.raw_azimuth + pooled_offset_az + probes.delta_azimuth_deg
    probes["per_session_elevation_deg"] = probes.raw_elevation + pooled_offset_el + probes.delta_elevation_deg

    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    boundary_rows, boundary_cols = np.nonzero(boundary)

    azimuth_norm = Normalize(vmin=0, vmax=90, clip=True)
    elevation_norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=40)
    row_defs = (
        ("naive", "naive_azimuth_deg", "naive_elevation_deg",
         f"NAIVE: one pooled offset for all sessions (az={pooled_offset_az:+.1f}, el={pooled_offset_el:+.1f} deg)"),
        ("per_session", "per_session_azimuth_deg", "per_session_elevation_deg",
         f"PER-SESSION: each session's own additional delta ({len(session_delta)} sessions)"),
    )
    col_defs = (
        (az_field, "viridis", azimuth_norm, "Azimuth"),
        (el_field, "coolwarm", elevation_norm, "Elevation"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.6), constrained_layout=True)
    for row_idx, (row_key, az_col, el_col, row_title) in enumerate(row_defs):
        for col_idx, (probe_col, (field, cmap, norm, panel_title)) in enumerate(
            zip((az_col, el_col), col_defs)
        ):
            ax = axes[row_idx, col_idx]
            rows, cols = np.nonzero(np.isfinite(field))
            ax.scatter(cols, rows, c=field[rows, cols], cmap=cmap, norm=norm,
                       marker="s", s=1.4, alpha=0.55, linewidths=0, zorder=1, rasterized=True)
            ax.scatter(boundary_cols, boundary_rows, s=0.6, color="#343434", zorder=2, rasterized=True)
            ax.scatter(probes.zhuang_col, probes.zhuang_row, c=probes[probe_col], cmap=cmap, norm=norm,
                       s=90, edgecolors="black", linewidths=0.8, zorder=3)
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.025)
            colorbar.set_label("degrees")
            ax.set(title=f"{panel_title}\n{row_title}" if col_idx == 0 else panel_title,
                   xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
            ax.set_axisbelow(True)

    fig.suptitle(
        f"MouseV2: naive (single pooled offset) vs. per-session RF offset -- same anatomical geometry, "
        f"same span-matched field, same color scale (n={len(probes)} probes, PROBE-level not unit-level "
        f"-- see module docstring)",
        fontsize=12.5,
    )
    figure_path = OUTPUT / "Figure_mousev2_naive_vs_per_session_translation_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
