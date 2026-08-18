#!/usr/bin/env python3
"""Alternative to `register_mousev2_units_along_probe_shank.py`'s Nelder-Mead line fit: anchor each
probe at its already-validated 6E position (`register_mousev2_rf_to_zhuang_v1.py`'s per-probe
RF-value nearest-match, which is hard-constrained to VISp by construction, unlike 6G's soft domain
penalty) and derive the shank's DIRECTION and LENGTH analytically instead of optimizing them:

- Direction: locally linearize the Zhuang azimuth/elevation field around the anchor (kernel-free
  local-linear regression of (az, el) on (row, col) over nearby VISp candidates -> a 2x2 Jacobian
  J = d(az,el)/d(row,col)), then invert it against the probe's OWN observed RF gradient along depth
  (robust linear regression of (az, el) vs cortical_depth) to find the cortical direction u such
  that J @ u matches the observed per-depth RF change. This is "which way would you have to move in
  the map to see the RF change this probe's units actually show as depth increases" -- a genuine,
  data-driven direction, not a free optimizer parameter that can wander.
- Length: the independent, non-map-derived estimate from `compare_rf_depth_span_mousev2_vs_allen.py`
  (per-probe insertion angle from vertical, via RF-significant-unit depth span vs. Allen) -- the
  same physical quantity 6G's regularization target used, but now the ONLY source of magnitude
  (the gradient inversion only supplies direction; its own magnitude is not trusted, since a local
  Jacobian estimated from one small neighborhood is far noisier for magnitude than for direction).

No Nelder-Mead, no domain penalty, no length-regularization weight to tune -- p0/p1 fall out of
anchor + direction + length directly. Session delta (the shared RF-value offset) is fit AFTER
positions are fixed, as a simple pooled Huber location of residuals -- it no longer needs to be
co-optimized with position since position no longer depends on it.
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
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "artifacts/figure3/06i_mousev2_gradient_anchored_shank"
RF_DEPTH_SPAN_TABLE = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle/mousev2_rf_depth_span.csv"

MIN_UNITS_PER_PROBE = 15
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
JACOBIAN_RADII_PX = (20.0, 35.0, 50.0, 70.0)
JACOBIAN_MIN_POINTS = 25
COND_THRESHOLD = 50.0


def load_depth_table() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("site*_processed/layer_info.csv")):
        frames.append(pd.read_csv(path, usecols=["unit_id", "cortical_depth"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates("unit_id")


def local_jacobian(anchor_row: float, anchor_col: float, candidate_rows: np.ndarray, candidate_cols: np.ndarray,
                    candidates: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    dist = np.hypot(candidate_rows - anchor_row, candidate_cols - anchor_col)
    for radius in JACOBIAN_RADII_PX:
        mask = dist <= radius
        if mask.sum() >= JACOBIAN_MIN_POINTS:
            dr = (candidate_rows[mask] - anchor_row).astype(float)
            dc = (candidate_cols[mask] - anchor_col).astype(float)
            design = np.column_stack([dr, dc, np.ones(mask.sum())])
            coef_az, *_ = np.linalg.lstsq(design, candidates[mask, 0], rcond=None)
            coef_el, *_ = np.linalg.lstsq(design, candidates[mask, 1], rcond=None)
            jac = np.array([[coef_az[0], coef_az[1]], [coef_el[0], coef_el[1]]])
            cond = float(np.linalg.cond(jac))
            return jac, cond, radius
    return None, np.nan, np.nan


def rf_gradient_along_depth(depth_um: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    depth_c = depth_um - depth_um.mean()
    design = np.column_stack([depth_c, np.ones_like(depth_c)])
    coef_az, *_ = np.linalg.lstsq(design, azimuth_deg, rcond=None)
    coef_el, *_ = np.linalg.lstsq(design, elevation_deg, rcond=None)
    return np.array([coef_az[0], coef_el[0]])  # deg per um


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]
    height, width = az_field.shape
    row_axis, col_axis = np.arange(height), np.arange(width)
    az_interp = RegularGridInterpolator((row_axis, col_axis), az_field, bounds_error=False, fill_value=np.nan)
    el_interp = RegularGridInterpolator((row_axis, col_axis), el_field, bounds_error=False, fill_value=np.nan)
    domain_distance_interp = RegularGridInterpolator(
        (row_axis, col_axis), template["area_distance_arrays"]["VISp"], bounds_error=False, fill_value=100.0)
    candidate_rows, candidate_cols = np.nonzero(visp_mask & np.isfinite(az_field) & np.isfinite(el_field))
    candidates = np.column_stack([az_field[candidate_rows, candidate_cols], el_field[candidate_rows, candidate_cols]])
    candidate_rows = candidate_rows.astype(float)
    candidate_cols = candidate_cols.astype(float)

    reg_manifest = json.loads((REGISTRATION_DIR / "registration_manifest.json").read_text())
    azimuth_offset = reg_manifest["calibrated_azimuth_offset_deg"]
    elevation_offset = reg_manifest["calibrated_elevation_offset_deg"]
    anchors = pd.read_csv(REGISTRATION_DIR / "mousev2_probe_inferred_v1_position.csv").set_index(["site", "probe"])
    px_per_mm = json.loads(Path(ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"
                                 / "translation_rotation_fit_manifest.json").read_text())["fixed_scale_px_per_mm"]
    angle_table = pd.read_csv(RF_DEPTH_SPAN_TABLE).set_index(["site", "probe"])["estimated_angle_from_vertical_deg"]
    print(f"anchors: {len(anchors)}, angle estimates: {len(angle_table)}, px_per_mm={px_per_mm:.1f}")

    rf = pd.read_csv(RF_FITS, low_memory=False)
    units = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    units["azimuth_deg"] = units.supported_rf_center_x_deg + azimuth_offset
    units["elevation_deg"] = units.supported_rf_center_y_deg + elevation_offset
    units["rf_area_deg2"] = np.pi * units.rf_sigma_major_deg * units.rf_sigma_minor_deg
    units["log2_rf_area"] = np.log2(units.rf_area_deg2)
    depth_table = load_depth_table()
    units = units.merge(depth_table, on="unit_id", how="inner")
    print(f"units with depth: {len(units)}")

    probe_rows = []
    per_unit_rows = []
    residuals_by_session: dict[str, list[np.ndarray]] = {}
    for (site, probe), group in units.groupby(["site", "probe"]):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        if (site, probe) not in anchors.index:
            print(f"[{site} {probe}] no 6E anchor, skipping")
            continue
        anchor_row = float(anchors.loc[(site, probe), "inferred_row"])
        anchor_col = float(anchors.loc[(site, probe), "inferred_col"])
        depth = group.cortical_depth.to_numpy(float)
        observed = group[["azimuth_deg", "elevation_deg"]].to_numpy(float)
        depth_range_um = float(depth.max() - depth.min())

        jac, cond, radius_used = local_jacobian(anchor_row, anchor_col, candidate_rows, candidate_cols, candidates)
        g_obs = rf_gradient_along_depth(depth, observed[:, 0], observed[:, 1])  # deg per um

        direction_source = "no_local_jacobian"
        direction_unit = None
        if jac is not None and cond < COND_THRESHOLD:
            try:
                direction_raw = np.linalg.solve(jac, g_obs)  # px per um, along (row, col)
            except np.linalg.LinAlgError:
                direction_raw = None
            if direction_raw is not None and np.linalg.norm(direction_raw) > 1e-6:
                direction_unit = direction_raw / np.linalg.norm(direction_raw)
                direction_source = "gradient_inversion"
        if direction_unit is None:
            # fallback: per-unit nearest-candidate match, regressed against depth
            distances = np.sum((candidates[:, None, :] - observed[None, :, :]) ** 2, axis=2)
            nearest_idx = np.argmin(distances, axis=0)
            init_positions = np.column_stack([candidate_rows[nearest_idx], candidate_cols[nearest_idx]])
            design = np.column_stack([depth - depth.mean(), np.ones_like(depth)])
            coef_row, *_ = np.linalg.lstsq(design, init_positions[:, 0], rcond=None)
            coef_col, *_ = np.linalg.lstsq(design, init_positions[:, 1], rcond=None)
            raw = np.array([coef_row[0], coef_col[0]])
            if np.linalg.norm(raw) > 1e-9:
                direction_unit = raw / np.linalg.norm(raw)
                direction_source = "nearest_match_fallback"
            else:
                direction_unit = np.array([1.0, 0.0])
                direction_source = "degenerate_fallback"

        angle_deg = angle_table.get((site, probe), np.nan)
        if np.isfinite(angle_deg):
            expected_tangential_mm = (depth_range_um / 1000.0) * np.sin(np.radians(angle_deg))
            length_px = expected_tangential_mm * px_per_mm
            length_source = "rf_depth_span_angle"
        else:
            length_px = float(np.linalg.norm(g_obs @ np.linalg.pinv(jac).T if jac is not None else [0, 0])) * depth_range_um
            length_source = "gradient_magnitude_fallback"

        p0 = np.array([anchor_row, anchor_col]) - 0.5 * length_px * direction_unit
        p1 = np.array([anchor_row, anchor_col]) + 0.5 * length_px * direction_unit
        t = (depth - depth.min()) / max(depth_range_um, 1e-6)
        positions = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        row_col = positions[:, ::-1]
        predicted = np.column_stack([az_interp(row_col), el_interp(row_col)])
        d_p0 = float(domain_distance_interp(p0[::-1].reshape(1, -1))[0])
        d_p1 = float(domain_distance_interp(p1[::-1].reshape(1, -1))[0])

        probe_rows.append({
            "site": site, "probe": probe, "n_units": len(group), "depth_range_um": depth_range_um,
            "anchor_row": anchor_row, "anchor_col": anchor_col,
            "p0_row": p0[0], "p0_col": p0[1], "p1_row": p1[0], "p1_col": p1[1],
            "shank_length_px": length_px, "length_source": length_source,
            "direction_source": direction_source, "jacobian_cond": cond, "jacobian_radius_px": radius_used,
            "rf_span_estimated_angle_deg": angle_deg,
            "domain_dist_p0": d_p0, "domain_dist_p1": d_p1, "max_endpoint_domain_dist": max(d_p0, d_p1),
        })
        residual = predicted - observed
        residuals_by_session.setdefault(site, []).append(residual)
        for unit_id, position, predicted_i, observed_i in zip(group.unit_id.to_numpy(), positions, predicted, observed):
            per_unit_rows.append({
                "site": site, "probe": probe, "unit_id": unit_id,
                "inferred_row": position[0], "inferred_col": position[1],
                "predicted_azimuth_deg": predicted_i[0], "predicted_elevation_deg": predicted_i[1],
                "observed_azimuth_deg": observed_i[0], "observed_elevation_deg": observed_i[1],
            })

    session_deltas = {site: huber_location(np.concatenate(res, axis=0)) for site, res in residuals_by_session.items()}
    session_delta_rows = [{"site": s, "delta_azimuth_deg": d[0], "delta_elevation_deg": d[1]} for s, d in session_deltas.items()]

    probe_lines = pd.DataFrame(probe_rows)
    per_unit = pd.DataFrame(per_unit_rows)
    per_unit = per_unit.merge(units[["unit_id", "log2_rf_area"]], on="unit_id", how="left")
    per_unit["site"] = per_unit.site
    per_unit["delta_azimuth_deg"] = per_unit.site.map({s: d[0] for s, d in session_deltas.items()})
    per_unit["delta_elevation_deg"] = per_unit.site.map({s: d[1] for s, d in session_deltas.items()})
    per_unit["fit_residual_deg"] = np.hypot(
        per_unit.predicted_azimuth_deg - per_unit.delta_azimuth_deg - per_unit.observed_azimuth_deg,
        per_unit.predicted_elevation_deg - per_unit.delta_elevation_deg - per_unit.observed_elevation_deg,
    )

    probe_lines.to_csv(OUTPUT / "probe_shank_lines_gradient_anchored.csv", index=False)
    pd.DataFrame(session_delta_rows).to_csv(OUTPUT / "session_delta.csv", index=False)
    per_unit.to_csv(OUTPUT / "unit_positions_along_shank_gradient_anchored.csv", index=False)

    print(f"\nprobes: {len(probe_lines)}")
    print(f"direction source counts:\n{probe_lines.direction_source.value_counts().to_string()}")
    print(f"length source counts:\n{probe_lines.length_source.value_counts().to_string()}")
    print(f"median shank length (px): {probe_lines.shank_length_px.median():.1f}")
    print(f"median jacobian condition number: {probe_lines.jacobian_cond.median():.2f}")
    print(f"median fit residual (deg, delta-corrected): {per_unit.fit_residual_deg.median():.3f}")
    n_out = int((probe_lines.max_endpoint_domain_dist > 10).sum())
    print(f"probes with an endpoint >10px outside VISp domain: {n_out}/{len(probe_lines)}")
    print(probe_lines.sort_values("max_endpoint_domain_dist", ascending=False)
          [["site", "probe", "n_units", "direction_source", "jacobian_cond", "shank_length_px",
            "rf_span_estimated_angle_deg", "max_endpoint_domain_dist"]].head(10).to_string(index=False))

    manifest = {
        "n_probes": len(probe_lines),
        "direction_source_counts": probe_lines.direction_source.value_counts().to_dict(),
        "length_source_counts": probe_lines.length_source.value_counts().to_dict(),
        "median_shank_length_px": float(probe_lines.shank_length_px.median()),
        "median_jacobian_cond": float(probe_lines.jacobian_cond.median()),
        "median_fit_residual_deg": float(per_unit.fit_residual_deg.median()),
        "n_probes_endpoint_outside_domain_gt10px": n_out,
        "cond_threshold": COND_THRESHOLD, "jacobian_min_points": JACOBIAN_MIN_POINTS,
        "jacobian_radii_px": list(JACOBIAN_RADII_PX),
    }
    (OUTPUT / "gradient_anchor_manifest.json").write_text(json.dumps(manifest, indent=2))

    # -- figure --
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    boundary = template["boundary"].astype(float)
    ax = axes[0]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    for _, row in probe_lines.iterrows():
        color = PROBE_COLORS.get(row.probe, "black")
        ax.plot([row.p0_col, row.p1_col], [row.p0_row, row.p1_row], color=color, linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter([row.anchor_col], [row.anchor_row], marker="*", s=60, color=color, edgecolors="black", linewidths=0.4, zorder=4)
        ax.scatter([row.p0_col], [row.p0_row], marker="o", s=22, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter([row.p1_col], [row.p1_row], marker="s", s=22, color=color, edgecolors="white", linewidths=0.5, zorder=3)
    for probe, color in PROBE_COLORS.items():
        ax.plot([], [], color=color, linewidth=1.6, label=f"probe {probe}")
    ax.legend(fontsize=8)
    ax.set(title="Gradient-anchored shanks: anchor=6E position (star),\ndirection=map-gradient inversion, length=RF-span angle\n(o=shallowest, sq=deepest)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    height, width = template["domain"].shape
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    ax = axes[1]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    norm = plt.Normalize(vmin=per_unit.log2_rf_area.quantile(0.02), vmax=per_unit.log2_rf_area.quantile(0.98))
    scatter = ax.scatter(per_unit.inferred_col, per_unit.inferred_row, c=per_unit.log2_rf_area, cmap="viridis",
                          norm=norm, s=10, alpha=0.7, rasterized=True)
    fig.colorbar(scatter, ax=ax, fraction=0.046, label="log2 RF area (deg^2)")
    ax.set(title="MouseV2 RF size, units on gradient-anchored shanks",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    fig.suptitle("Gradient-anchored probe-shank registration (closed-form: no Nelder-Mead)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_mousev2_probe_shanks_gradient_anchored.png", dpi=170)
    plt.close(fig)
    print(f"\n{OUTPUT / 'Figure_mousev2_probe_shanks_gradient_anchored.png'}")


if __name__ == "__main__":
    main()
