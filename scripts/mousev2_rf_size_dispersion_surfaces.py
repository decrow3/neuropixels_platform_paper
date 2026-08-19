#!/usr/bin/env python3
"""RF size and RF dispersion for MouseV2, mapped onto cortical V1 position, compared against
Allen's V1 "roughly matching group" (`v1_absolute_size_dispersion_translation_checkpoint/
v1_unit_descriptors.csv.gz`).

RF size: already fit per unit (`rf_sigma_major_deg`, `rf_sigma_minor_deg`, elliptical Gaussian).
Reported as ellipse area deg^2 = pi * sigma_major * sigma_minor, log2-scaled to match Allen's
`log2_rf_area` convention -- NOT claimed to be a methodologically identical estimator (different
stimulus family/fit target), only descriptively comparable, consistent with the caveat already
on record for the SF/TF surface comparison (06d).

RF dispersion: genuinely new for MouseV2. Unlike Allen (independent CCF anatomy lets you compute
residual RF scatter around a smooth anatomy-based mean), MouseV2's only position axis IS derived
from RF value, so an anatomy-residual dispersion would be circular. The non-circular analog used
throughout this project (`verify_warp_variants_via_rf_dispersion.py`) is WITHIN-PROBE RF-center
scatter: a probe's units all sit at one fixed physical location, so the trace of the covariance
of their RF centers around the probe's own robust center is a genuine local-dispersion measure,
untouched by the position-inference step.

Per-unit cortical position (revised 2026-08-18): earlier used an independent per-unit nearest-
RF-value match against the dense V1 candidate grid, with no anatomical anchor at all -- flagged as
an artifact by `register_mousev2_units_along_probe_shank.py`'s (06g) own docstring ("exactly the
artifact visible in the per-unit RF-size figure from" this file). Switching to 06g's free
2-endpoint line fit was considered next, but `render_allen_vs_mousev2_units_on_map_comparison.py`
found (2026-08-18) that 06g's free-fit shallow/entry endpoint sits a median 95px from the TRUE
anatomical penetration point -- about half of V1's own diameter -- so it isn't anatomically
grounded either. This file now uses `direction_search_unit_positions.csv`
(`render_mousev2_direction_search_depth_spread.py`, written into the 06j directory): entry point
FIXED to the independent anatomy-registered position (06j, never consults RF value), shank length
from an independently-derived depth-span estimate, and only ONE free parameter (shank angle theta)
fit from each probe's own RF-vs-depth trend.

PUTATIVE, NOT FULLY RESOLVED: that single angle parameter is weakly identified. Without any
anatomical prior it pointed away from V1's center for 59% of probes, so the search is hard-
restricted to a +/-90deg "toward V1 center" cone (a soft penalty was tried and found far too weak
to matter -- see that script's own docstring). Rerun 2026-08-18: 26/26 probes fit within that cone,
but the median cosine to the inward direction is only +0.15 (near-orthogonal, not confidently
inward) and the median Huber fit loss is ~42deg. Treat every plotted probe DIRECTION (not the
anatomy-anchored entry point itself) as putative -- a plausible, anatomically-constrained guess,
not a resolved measurement. Units within a probe still don't form an independently scattered
cloud (that specific 06f-vs-06g artifact is fixed), but the line's own angle could be wrong.

Anatomy-based overlay: `register_mousev2_area_borders_to_zhuang.py` (06j) independently places
each probe from cranial-window photo anatomy (area-border shape matching), never consulting RF
value. It is overlaid on both cortical-space panels below as a second marker layer at per-probe
(not per-unit) resolution. Because the per-unit position above now shares its own entry point with
this same anatomy table, the overlay is no longer a fully independent cross-check at the shallow
end -- it mainly tests whether the fitted DIRECTION carries the per-probe median position far from
its own anchor, not whether the anchor itself is right. See 06j and 06p
(`mousev2_frequency_preference_cortical_surfaces.py`) for the same convention applied to SF/TF
preference.
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
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from register_allen_session_to_zhuang import build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
REGISTRATION_DIR = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1"
ANATOMY_POSITIONS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/probe_anatomical_position.csv"
DIRECTION_SEARCH_POSITIONS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/direction_search_unit_positions.csv"
ALLEN_V1 = ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint/v1_unit_descriptors.csv.gz"
OUTPUT = ROOT / "artifacts/figure3/06f_mousev2_rf_size_dispersion_surfaces"

MIN_UNITS_PER_PROBE = 15
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)

    reg_manifest = json.loads((REGISTRATION_DIR / "registration_manifest.json").read_text())
    azimuth_offset = reg_manifest["calibrated_azimuth_offset_deg"]
    elevation_offset = reg_manifest["calibrated_elevation_offset_deg"]
    print(f"using calibrated RF-value offsets az={azimuth_offset:+.2f}, el={elevation_offset:+.2f}")

    rf = pd.read_csv(RF_FITS, low_memory=False)
    units = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    units["azimuth_deg"] = units.supported_rf_center_x_deg + azimuth_offset
    units["elevation_deg"] = units.supported_rf_center_y_deg + elevation_offset
    units["rf_area_deg2"] = np.pi * units.rf_sigma_major_deg * units.rf_sigma_minor_deg
    units["log2_rf_area"] = np.log2(units.rf_area_deg2)

    # Per-unit cortical position: anatomy-anchored direction search (06j), NOT an independent
    # per-unit RF-value match -- see module docstring. Left-joined so units on probes below that
    # fit's own MIN_UNITS_PER_PROBE threshold are kept (with no position) for the RF-value-only
    # histogram/summary below, and only dropped from the cortical-space panels.
    ds_positions = pd.read_csv(DIRECTION_SEARCH_POSITIONS)[["unit_id", "inferred_row", "inferred_col"]]
    if ds_positions["unit_id"].duplicated().any():
        raise ValueError(f"{DIRECTION_SEARCH_POSITIONS} contains duplicate unit IDs")
    units = units.merge(ds_positions, on="unit_id", how="left")
    units.to_csv(OUTPUT / "mousev2_unit_inferred_position_and_size.csv", index=False)
    positioned = units.dropna(subset=["inferred_row", "inferred_col"]).copy()
    print(f"direction-search cortical positions: {len(positioned)}/{len(units)} RF-supported units matched "
          f"(rest are on probes below the direction-search fit's own {MIN_UNITS_PER_PROBE}-unit threshold)")

    # RF dispersion: within-probe RF-center scatter (trace of covariance), non-circular, computed
    # on ALL RF-supported units regardless of cortical-position availability. Plot location for
    # each probe is the median (row, col) of that probe's own POSITIONED units, or NaN (dropped
    # from the cortical panel, kept in the histogram below) if none matched.
    dispersion_rows = []
    for (site, probe), group in units.groupby(["site", "probe"]):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        rf_centers = group[["azimuth_deg", "elevation_deg"]].to_numpy(float)
        center = huber_location(rf_centers)
        centered = rf_centers - center
        trace_deg2 = float(np.mean(np.sum(centered ** 2, axis=1)))
        positioned_group = group.dropna(subset=["inferred_row", "inferred_col"])
        probe_row = float(positioned_group.inferred_row.median()) if len(positioned_group) else np.nan
        probe_col = float(positioned_group.inferred_col.median()) if len(positioned_group) else np.nan
        dispersion_rows.append({
            "site": site, "probe": probe, "n_units": len(group), "n_units_positioned": len(positioned_group),
            "dispersion_trace_deg2": trace_deg2, "log2_dispersion_trace": np.log2(trace_deg2),
            "median_log2_rf_area": float(group.log2_rf_area.median()),
            "inferred_row": probe_row, "inferred_col": probe_col,
        })
    dispersion_table = pd.DataFrame(dispersion_rows)
    dispersion_table.to_csv(OUTPUT / "mousev2_probe_rf_dispersion.csv", index=False)
    dispersion_plotted = dispersion_table.dropna(subset=["inferred_row", "inferred_col"])

    # Anatomy-based overlay: independent per-probe position from photo area-border matching (06j),
    # never consulting RF value. Note this is the SAME entry point the direction-search per-unit
    # position above is anchored to -- see module docstring for what this cross-check now means.
    anatomy = pd.read_csv(ANATOMY_POSITIONS)[["site", "probe", "zhuang_row", "zhuang_col"]]
    anatomy_overlay = dispersion_plotted.merge(anatomy, on=["site", "probe"], how="inner")
    anatomy_overlay["position_offset_px"] = np.hypot(
        anatomy_overlay.inferred_col - anatomy_overlay.zhuang_col,
        anatomy_overlay.inferred_row - anatomy_overlay.zhuang_row,
    )
    anatomy_overlay.to_csv(OUTPUT / "mousev2_probe_anatomy_vs_rf_inferred_position.csv", index=False)
    print(f"anatomy-vs-direction-search median position: {len(anatomy_overlay)} probes matched, "
          f"median offset {anatomy_overlay.position_offset_px.median():.1f}px "
          "(distance the fitted line's median-depth position drifted from its own anchor)")

    allen = pd.read_csv(ALLEN_V1)
    print("\n=== MouseV2 vs. Allen V1 -- descriptive comparison (different stimulus/estimator; not a matched test) ===")
    print(f"RF area (log2 deg^2): MouseV2 median={units.log2_rf_area.median():.2f} (n={len(units)}), "
          f"Allen V1 median={allen.log2_rf_area.median():.2f} (n={len(allen)})")
    print(f"RF dispersion trace (log2 deg^2): MouseV2 median={dispersion_table.log2_dispersion_trace.median():.2f} "
          f"(n={len(dispersion_table)} probes), Allen V1 median={allen.dispersion_log2_trace.median():.2f} "
          f"(n={allen.dispersion_log2_trace.notna().sum()} units, different unit-of-analysis: per-probe vs. per-unit-neighborhood)")

    summary = {
        "mousev2_n_units": len(units), "mousev2_n_probes_with_dispersion": len(dispersion_table),
        "mousev2_median_log2_rf_area": float(units.log2_rf_area.median()),
        "allen_v1_median_log2_rf_area": float(allen.log2_rf_area.median()),
        "mousev2_median_log2_dispersion_trace": float(dispersion_table.log2_dispersion_trace.median()),
        "allen_v1_median_log2_dispersion_trace": float(allen.dispersion_log2_trace.median()),
        "caveat": "descriptive comparison only -- different stimulus families/estimators (see 06d) and "
                  "different dispersion unit-of-analysis (MouseV2: within-probe; Allen: within-250um-CCF-neighborhood)",
        "cortical_position": {
            "method": "direction_search_unit_positions.csv (render_mousev2_direction_search_depth_spread.py): "
                      "entry point anchored to independent anatomy (06j), shank angle is the only free parameter",
            "n_units_positioned": int(len(positioned)), "n_units_total": int(len(units)),
            "n_probes_positioned": int(len(dispersion_plotted)), "n_probes_total": int(len(dispersion_table)),
            "direction_putative_caveat": "the fitted shank ANGLE is weakly identified (hard-restricted to a "
                      "+/-90deg toward-V1-center cone; as of the 2026-08-18 rerun, median cos-to-inward is "
                      "only +0.15 and median Huber fit loss ~42deg) -- read every plotted probe DIRECTION as "
                      "putative, not a resolved measurement; only the anatomy-anchored entry point is trusted",
        },
        "anatomy_vs_direction_search_position": {
            "n_probes_matched": int(len(anatomy_overlay)),
            "median_offset_px": float(anatomy_overlay.position_offset_px.median()) if len(anatomy_overlay) else None,
            "caveat": "distance between the direction-search line's median-depth position (this checkpoint) "
                      "and its own anchor, the independent anatomy-registered entry point (06j) -- measures how "
                      "far the putative fitted DIRECTION carries the probe from its anchor, not two independent "
                      "position estimates (the entry point is shared by construction)",
        },
    }
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    # -- figures --
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    boundary = template["boundary"].astype(float)

    ax = axes[0, 0]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    norm = Normalize(vmin=positioned.log2_rf_area.quantile(0.02), vmax=positioned.log2_rf_area.quantile(0.98))
    scatter = ax.scatter(positioned.inferred_col, positioned.inferred_row, c=positioned.log2_rf_area, cmap="viridis",
                          norm=norm, s=10, alpha=0.6, rasterized=True)
    fig.colorbar(scatter, ax=ax, fraction=0.046, label="log2 RF area (deg^2)")
    if len(anatomy_overlay):
        ax.scatter(anatomy_overlay.zhuang_col, anatomy_overlay.zhuang_row, c=anatomy_overlay.median_log2_rf_area,
                   cmap="viridis", norm=norm,
                   s=110, marker="o", edgecolors="white", linewidths=1.3, zorder=5)
    ax.set(title=f"MouseV2 RF size across V1 (per-unit position, n={len(positioned)}; angle is PUTATIVE)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    height, width = template["domain"].shape
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    ax = axes[0, 1]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    norm = Normalize(vmin=dispersion_plotted.log2_dispersion_trace.min(), vmax=dispersion_plotted.log2_dispersion_trace.max())
    scatter = ax.scatter(dispersion_plotted.inferred_col, dispersion_plotted.inferred_row,
                          c=dispersion_plotted.log2_dispersion_trace, cmap="magma", norm=norm,
                          s=90, edgecolors="white", linewidths=0.8)
    fig.colorbar(scatter, ax=ax, fraction=0.046, label="log2 within-probe RF dispersion trace (deg^2)")
    if len(anatomy_overlay):
        ax.scatter(anatomy_overlay.zhuang_col, anatomy_overlay.zhuang_row, c=anatomy_overlay.log2_dispersion_trace,
                   cmap="magma", norm=norm, s=140, marker="o", edgecolors="#00e5ff", linewidths=1.6, zorder=5)
    ax.set(title=f"MouseV2 RF dispersion across V1 (per-probe, n={len(dispersion_plotted)}; angle is PUTATIVE)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    ax = axes[1, 0]
    bins = np.linspace(min(units.log2_rf_area.min(), allen.log2_rf_area.min()),
                        max(units.log2_rf_area.max(), allen.log2_rf_area.max()), 40)
    ax.hist(allen.log2_rf_area, bins=bins, density=True, alpha=0.5, label=f"Allen V1 (n={len(allen)})", color="#4575b4")
    ax.hist(units.log2_rf_area, bins=bins, density=True, alpha=0.5, label=f"MouseV2 (n={len(units)})", color="#d73027")
    ax.set(title="RF size distribution: MouseV2 vs. Allen V1\n(descriptive -- different stimulus/estimator)",
           xlabel="log2 RF area (deg^2)", ylabel="density")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    allen_disp = allen.dispersion_log2_trace.dropna()
    bins = np.linspace(min(dispersion_table.log2_dispersion_trace.min(), allen_disp.min()),
                        max(dispersion_table.log2_dispersion_trace.max(), allen_disp.max()), 30)
    ax.hist(allen_disp, bins=bins, density=True, alpha=0.5, label=f"Allen V1 (per-unit-neighborhood, n={len(allen_disp)})", color="#4575b4")
    ax.hist(dispersion_table.log2_dispersion_trace, bins=bins, density=True, alpha=0.5,
            label=f"MouseV2 (per-probe, n={len(dispersion_table)})", color="#d73027")
    ax.set(title="RF dispersion distribution: MouseV2 vs. Allen V1\n(different unit-of-analysis -- not a matched test)",
           xlabel="log2 dispersion trace (deg^2)", ylabel="density")
    ax.legend(fontsize=9)

    fig.suptitle(
        "MouseV2 RF size + dispersion mapped to cortical V1 position, vs. Allen V1\n"
        "top row: small/plain markers = anatomy-anchored position (entry fixed to 06j, shank ANGLE is putative)\n"
        "white/cyan-edged large circles = independent anatomy-registered entry point (06j)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    fig.savefig(OUTPUT / "Figure_mousev2_rf_size_dispersion_vs_allen.png", dpi=170)
    plt.close(fig)
    print(f"\n{OUTPUT / 'Figure_mousev2_rf_size_dispersion_vs_allen.png'}")


if __name__ == "__main__":
    main()
