#!/usr/bin/env python3
"""Reconstruction of a lost exploratory script (2026-08-17, never committed) whose output figures
were `depth_spread_directionsearch_azimuth.png` / `..._elevation.png` in
`artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/`. Rebuilt 2026-08-18 (per user
request) using this session's substantially improved anatomical anchors (all 8 animals now
hand-traced; probe-A apex anchor; area-matched scale -- see
`register_mousev2_area_borders_to_zhuang.py`).

Unlike `register_mousev2_units_along_probe_shank.py`, which freely fits BOTH shank endpoints (4
free params per probe) from RF values alone, this fixes the SHALLOW/entry endpoint from the
anatomical registration (`probe_anatomical_position.csv`) and the shank LENGTH from the
RF-depth-span-implied per-probe angle estimate (same convention as that script's own length
regularization target, `mousev2_rf_depth_span.csv`) -- leaving only a 1D search over shank ANGLE
(theta) to place the deep end. This is a more anatomically-constrained cross-check of the same "do
individual units' RF values, placed along their own probe, track the Zhuang retinotopic gradient"
question -- with an anatomical anchor doing most of the work instead of RF values alone.

Per-session joint azimuth+elevation delta (the same delta_s role used throughout this project) is
fit jointly with each probe's angle via the same alternating scheme as the sibling script.

Update 2026-08-18 (per user direction): with only ONE free parameter (theta) per probe, a single
noisy unit could disproportionately swing the whole probe's fitted angle. `smooth_along_depth`
now smooths each probe's own azimuth/elevation-vs-depth trend (robust rank-based rolling median)
BEFORE fitting, so theta tracks the smooth per-probe gradient rather than any one unit's raw
value. Plotted dots and `observed_azimuth/elevation_deg` in the output CSV still show the RAW,
unsmoothed per-unit values -- smoothing only changes what the fit optimizes against, not what gets
reported as the data (the smoothed values are also kept, as `..._smoothed` columns, for anyone
who wants to see exactly what the fit saw).
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
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location, huber_mean_loss  # noqa: E402
from register_allen_session_to_zhuang import build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
RF_REGISTRATION_MANIFEST = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1/registration_manifest.json"
ANATOMICAL_PROBE_POSITIONS = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/probe_anatomical_position.csv"
RF_DEPTH_SPAN_TABLE = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle/mousev2_rf_depth_span.csv"
ZHUANG_SCALE_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang"

MIN_UNITS_PER_PROBE = 15
MAX_OUTER_ITER = 4
FALLBACK_LENGTH_PX = 50.0  # used only if a probe has no RF-depth-span angle estimate (rare)
COARSE_GRID_N = 181  # 2deg steps over the full circle before local refinement
DEPTH_OUTLIER_MAD_THRESHOLD = 3.0
DEPTH_SMOOTH_WINDOW_FRACTION = 1 / 4


def smooth_along_depth(t: np.ndarray, values: np.ndarray, window_fraction: float = DEPTH_SMOOTH_WINDOW_FRACTION) -> np.ndarray:
    """Robust local smoothing of RF values along a probe's own depth axis (rank-based rolling
    median) -- per user direction: fitting theta directly against raw per-unit RF values lets any
    single noisy unit disproportionately swing the whole probe's fitted direction, since there is
    only one free angle parameter to absorb it. Smoothing first, and fitting the resulting SMOOTH
    per-depth gradient instead of the raw scatter, makes one outlier's influence local (it shifts
    its own neighborhood's median a little) rather than global (it no longer single-handedly
    determines the best-fit angle). Neighbors are selected by DEPTH RANK (not t-value distance) in
    a window sized as a FRACTION of the probe's own unit count, matching `detect_ring_apex`'s
    convention elsewhere in this project for the same reason (works across very differently-sized
    probes without retuning). Median, not mean, for the same robustness reason as everywhere else
    in this project. Used only for the FIT target -- plotted dots still show the raw, unsmoothed
    per-unit values, since smoothing is a fitting-robustness device, not a claim about the data."""
    order = np.argsort(t)
    n = len(t)
    window = max(3, int(round(n * window_fraction)))
    half = window // 2
    values_ordered = values[order]
    smoothed_ordered = np.array([
        np.median(values_ordered[max(0, i - half):min(n, i + half + 1)]) for i in range(n)
    ])
    smoothed = np.empty(n)
    smoothed[order] = smoothed_ordered
    return smoothed


def trim_deep_depth_outliers(group: pd.DataFrame) -> pd.DataFrame:
    """Drops units whose cortical_depth is a robust outlier on the DEEP (small-depth) side -- per
    user direction: a probe's recorded tip occasionally includes a few cells well past the rest of
    its own population's center of mass (white-matter / adjacent-structure stragglers), which
    inflates the probe's apparent depth SPAN -- and therefore the span-derived shank-length
    estimate this script fixes each probe's length to -- even though those cells aren't
    representative of where the bulk of the probe's responsive units actually sit. Robust
    (median + MAD, not mean + SD) so the handful of stray points being screened for can't skew the
    very statistic used to detect them (matches this project's general preference for robust/Huber
    statistics over raw mean/SD). Only trims the DEEP tail (smallest cortical_depth = closest to
    the physical tip, per this project's cortical_depth convention -- see
    register_mousev2_units_along_probe_shank.py's module docstring), not the shallow tail, since
    the concern raised was specifically about deepest cells."""
    depth = group.cortical_depth.to_numpy(float)
    center = np.median(depth)
    mad = np.median(np.abs(depth - center))
    if mad < 1e-6:
        return group
    deep_z = (center - depth) / mad  # positive = deeper (smaller depth) than the group's center
    return group[deep_z <= DEPTH_OUTLIER_MAD_THRESHOLD]

# Without ANY anatomical prior, the 1D theta search below turned out to point AWAY from V1's
# center for 59% of probes (checked directly, 2026-08-18) -- probe A in particular pointed away in
# nearly every session (cos toward center -0.80 to -0.96). Unlike the sibling script
# (register_mousev2_units_along_probe_shank.py), which fits BOTH endpoints and so has some freedom
# to compensate, here the shank length is a hard FIXED value (from the depth-span-implied angle
# estimate) that is often long relative to V1's own extent (median 86px, up to 262px, vs V1's own
# ~200px diameter) -- so the deep end frequently lands outside V1 regardless of theta, where the
# field-matching objective is only weakly informative about anatomical plausibility and can prefer
# an outward direction. A SOFT penalty (matching the sibling script's weight) was tried first and
# found far too weak here: this script's fit-loss scale (~50deg) is ~20x the sibling script's
# (~22deg for a 2D residual), so a penalty capped at a few points barely moved the coarse-grid
# argmin (still 11/27 within +/-90deg, unchanged from unconstrained). Fixed instead by HARD-
# restricting the search itself to the +/-90deg "toward V1 center" cone -- directly matching the
# original assumption ("probes aim roughly inward, +/-90deg") rather than tuning a soft weight.


def load_depth_table() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("site*_processed/layer_info.csv")):
        frames.append(pd.read_csv(path, usecols=["unit_id", "cortical_depth"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates("unit_id")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)
    boundary = template["boundary"].astype(float)
    height, width = template["domain"].shape
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]
    row_axis, col_axis = np.arange(height), np.arange(width)
    az_interp = RegularGridInterpolator((row_axis, col_axis), az_field, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el_field, bounds_error=False, fill_value=np.nan)
    visp_rows, visp_cols = np.nonzero(template["area_masks"]["VISp"])
    visp_centroid_rc = np.array([visp_rows.mean(), visp_cols.mean()])

    reg_manifest = json.loads(RF_REGISTRATION_MANIFEST.read_text())
    azimuth_offset = reg_manifest["calibrated_azimuth_offset_deg"]
    elevation_offset = reg_manifest["calibrated_elevation_offset_deg"]

    anatomical = pd.read_csv(ANATOMICAL_PROBE_POSITIONS).set_index(["site", "probe"])[["zhuang_row", "zhuang_col"]]
    px_per_mm = json.loads(ZHUANG_SCALE_MANIFEST.read_text())["fixed_scale_px_per_mm"]
    angle_table = pd.read_csv(RF_DEPTH_SPAN_TABLE).set_index(["site", "probe"])["estimated_angle_from_vertical_deg"]
    print(f"loaded {len(anatomical)} anatomical entry points, {len(angle_table)} depth-span angle estimates")

    rf = pd.read_csv(RF_FITS, low_memory=False)
    units = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    units["azimuth_deg"] = units.supported_rf_center_x_deg + azimuth_offset
    units["elevation_deg"] = units.supported_rf_center_y_deg + elevation_offset

    depth_table = load_depth_table()
    units = units.merge(depth_table, on="unit_id", how="inner")
    print(f"units with depth: {len(units)}")

    probe_line_rows = []
    per_unit_rows = []
    session_delta_rows = []
    n_skipped_no_anchor = 0
    for site, session_units in units.groupby("site"):
        probes = {}
        for probe, group in session_units.groupby("probe"):
            n_before_trim = len(group)
            group = trim_deep_depth_outliers(group)
            n_trimmed = n_before_trim - len(group)
            if n_trimmed:
                print(f"{site} probe {probe}: trimmed {n_trimmed} deep-outlier unit(s) "
                      f"({n_before_trim} -> {len(group)})")
            if len(group) < MIN_UNITS_PER_PROBE:
                continue
            key = (site, probe)
            if key not in anatomical.index:
                n_skipped_no_anchor += 1
                continue
            entry_rc = anatomical.loc[key].to_numpy(float)
            depth = group.cortical_depth.to_numpy(float)
            t = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
            observed_raw = group[["azimuth_deg", "elevation_deg"]].to_numpy(float)
            observed = np.column_stack([smooth_along_depth(t, observed_raw[:, 0]),
                                         smooth_along_depth(t, observed_raw[:, 1])])
            depth_range_um = float(depth.max() - depth.min())
            angle_deg = angle_table.get(key, np.nan)
            if np.isfinite(angle_deg):
                expected_tangential_mm = (depth_range_um / 1000.0) * np.sin(np.radians(angle_deg))
                length_px = expected_tangential_mm * px_per_mm
            else:
                length_px = FALLBACK_LENGTH_PX
            probes[probe] = {"t": t, "observed": observed, "observed_raw": observed_raw, "n_units": len(group),
                              "entry_rc": entry_rc, "length_px": length_px, "unit_ids": group.unit_id.to_numpy()}
        if not probes:
            continue

        delta = np.zeros(2)
        fitted = {}
        for outer in range(MAX_OUTER_ITER):
            for probe, info in probes.items():
                entry_rc, length_px, t = info["entry_rc"], info["length_px"], info["t"]

                def positions_for_theta(theta, entry_rc=entry_rc, length_px=length_px, t=t):
                    direction = np.array([np.cos(theta), np.sin(theta)])
                    deep_rc = entry_rc + length_px * direction
                    return entry_rc[None, :] + (1 - t)[:, None] * (deep_rc - entry_rc)[None, :]

                inward_direction = visp_centroid_rc - entry_rc
                inward_direction = inward_direction / max(np.linalg.norm(inward_direction), 1e-6)
                inward_angle = float(np.arctan2(inward_direction[1], inward_direction[0]))

                def objective(theta, info=info, positions_for_theta=positions_for_theta):
                    positions = positions_for_theta(theta)
                    row_col = positions[:, ::-1]
                    predicted = np.column_stack([az_interp(row_col), el_interp(row_col)])
                    residual = predicted - delta - info["observed"]
                    return huber_mean_loss(residual)

                # HARD-restrict the search to the +/-90deg "toward V1 center" cone (a soft penalty
                # was tried first and found far too weak to matter, checked directly: this script's
                # fit-loss scale is ~50deg -- 20x this script's own sibling script -- so a penalty
                # capped at a few points barely nudges the coarse grid argmin at all, 11/27 within
                # +/-90deg with the soft version, unchanged from the unconstrained baseline).
                # Restricting the search space itself instead directly matches the original
                # assumption ("probes aim roughly inward, +/-90deg") as a hard cone, not a nudge.
                cone_low, cone_high = inward_angle - np.pi / 2, inward_angle + np.pi / 2
                thetas = np.linspace(cone_low, cone_high, COARSE_GRID_N)
                losses = np.array([objective(th) for th in thetas])
                best_theta = thetas[np.argmin(losses)]
                step = thetas[1] - thetas[0]
                # clip the local-refine window to the cone itself -- otherwise "bounded" can still
                # nudge ~1 grid step past the intended edge (checked directly: this is exactly what
                # was happening, landing right around cos=-0.017 for many probes instead of >=0).
                result = minimize_scalar(objective, bounds=(max(best_theta - step, cone_low),
                                                              min(best_theta + step, cone_high)), method="bounded")
                theta_fit = result.x
                positions = positions_for_theta(theta_fit)
                row_col = positions[:, ::-1]
                predicted = np.column_stack([az_interp(row_col), el_interp(row_col)])
                fitted[probe] = {"theta": theta_fit, "positions": positions, "predicted": predicted,
                                  "loss": float(result.fun)}
            pooled_residual = np.concatenate([
                fitted[probe]["predicted"] - probes[probe]["observed"] for probe in probes
            ], axis=0)
            delta = huber_location(pooled_residual)

        for probe, info in probes.items():
            fit = fitted[probe]
            inward_direction = visp_centroid_rc - info["entry_rc"]
            inward_direction = inward_direction / max(np.linalg.norm(inward_direction), 1e-6)
            shank_direction = np.array([np.cos(fit["theta"]), np.sin(fit["theta"])])
            probe_line_rows.append({
                "site": site, "probe": probe, "n_units": info["n_units"], "fit_loss": fit["loss"],
                "theta_deg": float(np.degrees(fit["theta"])), "length_px": info["length_px"],
                "entry_row": info["entry_rc"][0], "entry_col": info["entry_rc"][1],
                "cos_angle_toward_v1_center": float(np.dot(shank_direction, inward_direction)),
            })
            for unit_id, position, predicted, observed_smoothed, observed_raw in zip(
                info["unit_ids"], fit["positions"], fit["predicted"], info["observed"], info["observed_raw"]
            ):
                per_unit_rows.append({
                    "site": site, "probe": probe, "unit_id": unit_id,
                    "inferred_row": position[0], "inferred_col": position[1],
                    "predicted_azimuth_deg": predicted[0], "predicted_elevation_deg": predicted[1],
                    "observed_azimuth_deg": observed_raw[0], "observed_elevation_deg": observed_raw[1],
                    "observed_azimuth_deg_smoothed": observed_smoothed[0],
                    "observed_elevation_deg_smoothed": observed_smoothed[1],
                })
        session_delta_rows.append({"site": site, "delta_azimuth_deg": delta[0], "delta_elevation_deg": delta[1],
                                    "n_probes": len(probes)})

    probe_lines = pd.DataFrame(probe_line_rows)
    per_unit = pd.DataFrame(per_unit_rows)
    session_deltas = pd.DataFrame(session_delta_rows)
    probe_lines.to_csv(OUTPUT / "direction_search_probe_lines.csv", index=False)
    per_unit.to_csv(OUTPUT / "direction_search_unit_positions.csv", index=False)
    print(f"probes without an anatomical anchor (skipped): {n_skipped_no_anchor}")
    print(f"probes fit (1D angle search): {len(probe_lines)}, sessions: {len(session_deltas)}, units: {len(per_unit)}")
    print(f"median fit loss (huber, deg): {probe_lines.fit_loss.median():.3f}")
    n_within_90 = int((probe_lines.cos_angle_toward_v1_center >= 0).sum())
    print(f"direction prior: {n_within_90}/{len(probe_lines)} probes fitted within +/-90deg of "
          f"'toward V1 center' (median cos={probe_lines.cos_angle_toward_v1_center.median():+.2f})")

    sites = sorted(per_unit.site.unique())
    for field_name, field_arr in (("azimuth", az_field), ("elevation", el_field)):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        vmin, vmax = np.nanpercentile(field_arr, [2, 98])
        im = None
        for site, ax in zip(sites, axes.ravel()):
            ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.6)
            im = ax.imshow(field_arr, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.55, zorder=0)
            site_lines = probe_lines[probe_lines.site == site]
            for _, row in site_lines.iterrows():
                direction = np.array([np.cos(np.radians(row.theta_deg)), np.sin(np.radians(row.theta_deg))])
                deep_rc = np.array([row.entry_row, row.entry_col]) + row.length_px * direction
                ax.plot([row.entry_col, deep_rc[1]], [row.entry_row, deep_rc[0]], "-", color="black",
                        linewidth=1.0, alpha=0.6, zorder=2)
                ax.scatter([row.entry_col], [row.entry_row], marker="+", s=80, color="red", zorder=4)
                ax.text(row.entry_col, row.entry_row - 8, row.probe, color="black", fontsize=9, ha="center", zorder=5)
            site_units = per_unit[per_unit.site == site]
            ax.scatter(site_units.inferred_col, site_units.inferred_row,
                       c=site_units[f"observed_{field_name}_deg"], cmap="viridis", vmin=vmin, vmax=vmax,
                       s=14, edgecolors="black", linewidths=0.3, zorder=3)
            ax.set_title(f"{site} (n={len(site_units)})", fontsize=10)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(sites):]:
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01,
                     label=f"{field_name} (deg); dot fill = each UNIT's own RF, background = Zhuang field")
        fig.suptitle(f"Direction-search depth spread: RF {field_name} vs Zhuang field "
                     f"(1D theta search per probe; + = anatomical entry anchor)")
        out_path = OUTPUT / f"depth_spread_directionsearch_{field_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
