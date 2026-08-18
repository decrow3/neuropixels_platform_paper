#!/usr/bin/env python3
"""Replace independent per-unit nearest-RF-value matching with a physically constrained model:
all units on one probe sit along a single straight shank through cortex, so their inferred V1
positions should form a coherent LINE (parameterized by known cortical depth), not an
independent scattered cloud per unit -- exactly the artifact visible in the per-unit RF-size
figure from `mousev2_rf_size_dispersion_surfaces.py`.

Model per probe: position(depth) = p0 + t(depth) * (p1 - p0), where p0, p1 are the line's two
endpoints in Zhuang V1 pixel space (continuous, not grid-snapped) and t is depth linearly
rescaled to [0, 1] (0 = shallowest unit, 1 = deepest). Fit p0, p1 (4 free parameters) by
minimizing Huber loss between each unit's own predicted RF value (bilinearly interpolated at its
depth-implied position) and its observed RF value, offset by that session's delta -- alternating
with a delta refit, same joint scheme used throughout this project's registration work.
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
from scipy.optimize import minimize

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
REGISTRATION_DIR = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1"
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "artifacts/figure3/06g_mousev2_rf_units_along_probe_shank"

MIN_UNITS_PER_PROBE = 15
MAX_OUTER_ITER = 4
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}

# MouseV2's "cortical_depth" is raw probe_vertical_position (distance along the physical shank
# from a reference point; see generate_retinotopic_csvs.py L419-424) -- NOT a laminar/pia-normal
# depth, and we don't independently know these probes' insertion angle. Per-probe angle is
# instead estimated from the along-probe DEPTH SPAN of units with a significant receptive field
# (`compare_rf_depth_span_mousev2_vs_allen.py`), calibrated against a real multi-session Allen V1
# reference (n=24 probes, median RF-significant depth span 481 um) -- a relative (span) measure,
# so it does not depend on matching absolute depth-reference conventions between datasets, only
# that "span of RF-significant units" measures along-probe distance through visually-responsive
# cortex the same way in both. MouseV2 spans were significantly larger (median 844 um vs. 481 um,
# Mann-Whitney p=2.2e-08), consistently across 27/27 probes -- median estimated angle 55.3 deg
# from vertical (IQR 46.4-66.1), plausible given this project's deliberate multi-probe dispersal
# design (angling probes reaches different retinotopic positions from a shared craniotomy).
# This regularizes fitted shank length toward the per-probe angle-implied expectation (not a
# single population-average ratio) -- unconstrained fits ranged 0.03x-6.2x a population-average
# expectation, clear evidence of noise-driven overfitting without some such regularization.
RF_DEPTH_SPAN_TABLE = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle/mousev2_rf_depth_span.csv"
REGULARIZATION_WEIGHT = 3.0


def load_depth_table() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("site*_processed/layer_info.csv")):
        frames.append(pd.read_csv(path, usecols=["unit_id", "cortical_depth"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates("unit_id")


def fit_probe_line(depths_t: np.ndarray, observed: np.ndarray, delta: np.ndarray,
                    az_interp, el_interp, domain_distance_interp, init_p0: np.ndarray, init_p1: np.ndarray,
                    expected_shank_length_px: float | None = None):
    def predicted_positions(p0, p1):
        return p0[None, :] + depths_t[:, None] * (p1 - p0)[None, :]

    def objective(params):
        p0 = params[:2]
        p1 = params[2:]
        positions = predicted_positions(p0, p1)
        row_col = positions[:, ::-1]
        predicted = np.column_stack([az_interp(row_col), el_interp(row_col)])
        residual = predicted - delta - observed
        loss = huber_mean_loss(residual)
        domain_penalty = float(np.mean(np.square(domain_distance_interp(row_col) / 10.0)))
        length_penalty = 0.0
        if expected_shank_length_px is not None and expected_shank_length_px > 0:
            fitted_length = max(float(np.linalg.norm(p1 - p0)), 1e-3)
            length_penalty = REGULARIZATION_WEIGHT * float(np.log(fitted_length / expected_shank_length_px)) ** 2
        return loss + 0.5 * domain_penalty + length_penalty

    x0 = np.concatenate([init_p0, init_p1])
    result = minimize(objective, x0=x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-6})
    p0, p1 = result.x[:2], result.x[2:]
    positions = predicted_positions(p0, p1)
    row_col = positions[:, ::-1]
    predicted = np.column_stack([az_interp(row_col), el_interp(row_col)])
    return p0, p1, positions, predicted, float(result.fun)


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

    reg_manifest = json.loads((REGISTRATION_DIR / "registration_manifest.json").read_text())
    azimuth_offset = reg_manifest["calibrated_azimuth_offset_deg"]
    elevation_offset = reg_manifest["calibrated_elevation_offset_deg"]
    px_per_mm = json.loads(Path(ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases"
                                 / "translation_rotation_fit_manifest.json").read_text())["fixed_scale_px_per_mm"]
    angle_table = pd.read_csv(RF_DEPTH_SPAN_TABLE).set_index(["site", "probe"])["estimated_angle_from_vertical_deg"]
    print(f"loaded per-probe angle estimates for {len(angle_table)} probes, px_per_mm={px_per_mm:.1f}")

    rf = pd.read_csv(RF_FITS, low_memory=False)
    units = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    units["azimuth_deg"] = units.supported_rf_center_x_deg + azimuth_offset
    units["elevation_deg"] = units.supported_rf_center_y_deg + elevation_offset
    units["rf_area_deg2"] = np.pi * units.rf_sigma_major_deg * units.rf_sigma_minor_deg
    units["log2_rf_area"] = np.log2(units.rf_area_deg2)

    depth_table = load_depth_table()
    units = units.merge(depth_table, on="unit_id", how="inner")
    print(f"units with depth: {len(units)}")

    probe_line_rows = []
    session_delta_rows = []
    per_unit_rows = []
    for site, session_units in units.groupby("site"):
        probes = {}
        for probe, group in session_units.groupby("probe"):
            if len(group) < MIN_UNITS_PER_PROBE:
                continue
            depth = group.cortical_depth.to_numpy(float)
            t = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
            observed = group[["azimuth_deg", "elevation_deg"]].to_numpy(float)
            # init from a simple per-unit nearest-match regression against depth
            targets = observed
            distances = np.sum((candidates[:, None, :] - targets[None, :, :]) ** 2, axis=2)
            nearest_idx = np.argmin(distances, axis=0)
            init_positions = np.column_stack([candidate_rows[nearest_idx], candidate_cols[nearest_idx]]).astype(float)
            design = np.column_stack([t, np.ones_like(t)])
            coef_row, *_ = np.linalg.lstsq(design, init_positions[:, 0], rcond=None)
            coef_col, *_ = np.linalg.lstsq(design, init_positions[:, 1], rcond=None)
            init_p0 = np.array([coef_row[1], coef_col[1]])
            init_p1 = np.array([coef_row[0] + coef_row[1], coef_col[0] + coef_col[1]])
            depth_range_um = float(depth.max() - depth.min())
            angle_deg = angle_table.get((site, probe), np.nan)
            if np.isfinite(angle_deg):
                expected_tangential_mm = (depth_range_um / 1000.0) * np.sin(np.radians(angle_deg))
                expected_shank_length_px = expected_tangential_mm * px_per_mm
            else:
                expected_shank_length_px = None
            probes[probe] = {"t": t, "observed": observed, "n_units": len(group),
                              "init_p0": init_p0, "init_p1": init_p1, "unit_ids": group.unit_id.to_numpy(),
                              "expected_shank_length_px": expected_shank_length_px, "angle_deg": angle_deg}
        if not probes:
            continue

        delta = np.zeros(2)
        for outer in range(MAX_OUTER_ITER):
            fitted = {}
            for probe, info in probes.items():
                p0, p1, positions, predicted, loss = fit_probe_line(
                    info["t"], info["observed"], delta, az_interp, el_interp, domain_distance_interp,
                    info["init_p0"], info["init_p1"], expected_shank_length_px=info["expected_shank_length_px"])
                fitted[probe] = {"p0": p0, "p1": p1, "positions": positions, "predicted": predicted, "loss": loss}
            pooled_residual = np.concatenate([
                fitted[probe]["predicted"] - probes[probe]["observed"] for probe in probes
            ], axis=0)
            delta = huber_location(pooled_residual)

        for probe, info in probes.items():
            fit = fitted[probe]
            probe_line_rows.append({
                "site": site, "probe": probe, "n_units": info["n_units"], "fit_loss": fit["loss"],
                "p0_row": fit["p0"][0], "p0_col": fit["p0"][1], "p1_row": fit["p1"][0], "p1_col": fit["p1"][1],
                "shank_length_px": float(np.linalg.norm(fit["p1"] - fit["p0"])),
                "rf_span_estimated_angle_deg": info["angle_deg"],
                "expected_shank_length_px": info["expected_shank_length_px"],
            })
            for unit_id, position, predicted, observed in zip(
                info["unit_ids"], fit["positions"], fit["predicted"], info["observed"]
            ):
                per_unit_rows.append({
                    "site": site, "probe": probe, "unit_id": unit_id,
                    "inferred_row": position[0], "inferred_col": position[1],
                    "predicted_azimuth_deg": predicted[0], "predicted_elevation_deg": predicted[1],
                    "observed_azimuth_deg": observed[0], "observed_elevation_deg": observed[1],
                })
        session_delta_rows.append({"site": site, "delta_azimuth_deg": delta[0], "delta_elevation_deg": delta[1],
                                    "n_probes": len(probes)})

    probe_lines = pd.DataFrame(probe_line_rows)
    session_deltas = pd.DataFrame(session_delta_rows)
    per_unit = pd.DataFrame(per_unit_rows)
    per_unit = per_unit.merge(units[["unit_id", "log2_rf_area", "rf_area_deg2"]], on="unit_id", how="left")

    probe_lines.to_csv(OUTPUT / "probe_shank_lines.csv", index=False)
    session_deltas.to_csv(OUTPUT / "session_delta.csv", index=False)
    per_unit.to_csv(OUTPUT / "unit_positions_along_shank.csv", index=False)
    print(f"probes fit as shank lines: {len(probe_lines)}, sessions: {len(session_deltas)}, units: {len(per_unit)}")
    print(f"median shank length in Zhuang px: {probe_lines.shank_length_px.median():.1f}")
    print(f"median fit loss (huber, deg): {probe_lines.fit_loss.median():.3f}")

    # -- figure: shank lines over V1, colored by probe, + per-unit RF size along shanks --
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    boundary = template["boundary"].astype(float)
    ax = axes[0]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    for _, row in probe_lines.iterrows():
        color = PROBE_COLORS.get(row.probe, "black")
        ax.plot([row.p0_col, row.p1_col], [row.p0_row, row.p1_row], color=color, linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter([row.p0_col], [row.p0_row], marker="o", s=25, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter([row.p1_col], [row.p1_row], marker="s", s=25, color=color, edgecolors="white", linewidths=0.5, zorder=3)
    for probe, color in PROBE_COLORS.items():
        ax.plot([], [], color=color, linewidth=1.6, label=f"probe {probe}")
    ax.legend(fontsize=8)
    ax.set(title="MouseV2 probe shanks as depth-constrained lines in Zhuang V1\n(o=shallowest unit, sq=deepest)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    height, width = template["domain"].shape
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    ax = axes[1]
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    norm = plt.Normalize(vmin=per_unit.log2_rf_area.quantile(0.02), vmax=per_unit.log2_rf_area.quantile(0.98))
    scatter = ax.scatter(per_unit.inferred_col, per_unit.inferred_row, c=per_unit.log2_rf_area, cmap="viridis",
                          norm=norm, s=10, alpha=0.7, rasterized=True)
    fig.colorbar(scatter, ax=ax, fraction=0.046, label="log2 RF area (deg^2)")
    ax.set(title="MouseV2 RF size, units placed along their own probe's shank",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.set_xlim(0, width); ax.set_ylim(height, 0)

    fig.suptitle("Depth-constrained probe-shank registration (replaces independent per-unit matching)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_mousev2_probe_shanks_in_v1.png", dpi=170)
    plt.close(fig)
    print(OUTPUT / "Figure_mousev2_probe_shanks_in_v1.png")


if __name__ == "__main__":
    main()
