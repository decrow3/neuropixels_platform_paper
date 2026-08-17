#!/usr/bin/env python3
"""Three-way local-gradient comparison for session 781842082: Zhuang, Garrett, and the naive
V1-centered cross-session ephys pooling, all against this session's own local-linear Jacobian.

The naive set needs no atlas affine at all -- it's built directly in CCF space (unlike Zhuang
and Garrett, which live in a figure-pixel/panel frame needing a fitted CCF<->template
transform), so its pipeline is: rasterize the pooled (CCF, normalized RF) points onto a plain
CCF grid, then reuse the exact same reconstruct()/normalized_smooth() functions already used
for the atlases, so all three "maps" get built the same way before comparison.
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
from build_14animal_retinotopy_registration import production_support  # noqa: E402
from render_zhuang_interpolated_field_sign_qa import normalized_smooth, reconstruct  # noqa: E402
from fit_multistructure_fixed_effect_translation import (  # noqa: E402
    CCF2, DOMAINS, evaluate_jacobian, jacobian_interpolators, local_linear_jacobian_field, make_grid,
)
from compare_atlas_gradient_to_ephys_pilot_781842082 import (  # noqa: E402
    recover_ccf_to_pixel_affine, zhuang_gradient_interpolators, sample_zhuang_jacobian_px,
)
from build_garrett2014_smoothed_field_and_ccf_affine import (  # noqa: E402
    GARRETT_ROOT as GARRETT_FIELD_DIR, build_fields as build_garrett_fields, GRID_N as GARRETT_GRID_N,
)

ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 781842082
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
GARRETT_MANIFEST = (
    ROOT / "artifacts/retinotopy_template/garrett2014_figure5/smoothed_field_and_ccf_affine"
    / f"session_{SESSION_ID}_fit_manifest.json"
)
OUTPUT = ROOT / "artifacts/retinotopy_template/atlas_gradient_vs_ephys_pilot_781842082"
GRID_STEP_UM = 100.0
GRID_MARGIN_UM = 300.0
NAIVE_MIN_EFFECTIVE_N = 3


def build_naive_field():
    cells = pd.read_csv(NAIVE_CELLS)
    points = cells[CCF2].to_numpy(float)
    axis0, axis1, grid = make_grid(points, GRID_STEP_UM, GRID_MARGIN_UM)
    n0, n1 = len(axis0), len(axis1)
    col = np.clip(np.searchsorted(axis1, cells[CCF2[1]].to_numpy(float)) - 1, 0, n1 - 1)
    row = np.clip(np.searchsorted(axis0, cells[CCF2[0]].to_numpy(float)) - 1, 0, n0 - 1)
    fields = {}
    for value_col, key in (("normalized_rf_x", "azimuth"), ("normalized_rf_y", "elevation")):
        sums = np.zeros((n0, n1))
        counts = np.zeros((n0, n1))
        np.add.at(sums, (row, col), cells[value_col].to_numpy(float))
        np.add.at(counts, (row, col), 1)
        sparse = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts >= NAIVE_MIN_EFFECTIVE_N)
        surface, support = reconstruct(sparse)
        smoothed = normalized_smooth(surface, support, sigma=1.5)
        fields[key] = smoothed
    return axis0, axis1, fields


def naive_jacobian_interpolators(axis0, axis1, fields):
    az_drow, az_dcol = np.gradient(np.where(np.isfinite(fields["azimuth"]), fields["azimuth"], 0.0), axis0, axis1)
    el_drow, el_dcol = np.gradient(np.where(np.isfinite(fields["elevation"]), fields["elevation"], 0.0), axis0, axis1)
    valid = np.isfinite(fields["azimuth"]) & np.isfinite(fields["elevation"])
    components = {}
    for name, array in (("da_dap", az_drow), ("da_dml", az_dcol), ("de_dap", el_drow), ("de_dml", el_dcol)):
        components[name] = RegularGridInterpolator((axis0, axis1), np.where(valid, array, np.nan), bounds_error=False, fill_value=np.nan)
    return components


def sample_naive_jacobian(components, points_um: np.ndarray) -> np.ndarray:
    jac = np.full((len(points_um), 2, 2), np.nan)
    jac[:, 0, 0] = components["da_dap"](points_um)
    jac[:, 0, 1] = components["da_dml"](points_um)
    jac[:, 1, 0] = components["de_dap"](points_um)
    jac[:, 1, 1] = components["de_dml"](points_um)
    return jac


def sample_garrett_jacobian_px(fields: dict, xy: np.ndarray) -> np.ndarray:
    x_axis, y_axis = fields["x_axis"], fields["y_axis"]
    y_increasing = y_axis[::-1]
    row_col = np.column_stack([xy[:, 1], xy[:, 0]])
    smoothed_az = fields["azimuth_deg_smoothed_for_gradient"][::-1, :]
    smoothed_el = fields["elevation_deg_smoothed_for_gradient"][::-1, :]
    az_drow, az_dcol = np.gradient(np.where(np.isfinite(smoothed_az), smoothed_az, 0.0), y_increasing, x_axis)
    el_drow, el_dcol = np.gradient(np.where(np.isfinite(smoothed_el), smoothed_el, 0.0), y_increasing, x_axis)
    valid = np.isfinite(smoothed_az) & np.isfinite(smoothed_el)
    components = {
        "da_dx": RegularGridInterpolator((y_increasing, x_axis), np.where(valid, az_dcol, np.nan), bounds_error=False, fill_value=np.nan),
        "da_dy": RegularGridInterpolator((y_increasing, x_axis), np.where(valid, az_drow, np.nan), bounds_error=False, fill_value=np.nan),
        "de_dx": RegularGridInterpolator((y_increasing, x_axis), np.where(valid, el_dcol, np.nan), bounds_error=False, fill_value=np.nan),
        "de_dy": RegularGridInterpolator((y_increasing, x_axis), np.where(valid, el_drow, np.nan), bounds_error=False, fill_value=np.nan),
    }
    jac = np.full((len(xy), 2, 2), np.nan)
    jac[:, 0, 0] = components["da_dx"](row_col)
    jac[:, 0, 1] = components["da_dy"](row_col)
    jac[:, 1, 0] = components["de_dx"](row_col)
    jac[:, 1, 1] = components["de_dy"](row_col)
    return jac


def main() -> None:
    cells, audit = production_support()
    session_cells = cells.loc[cells.session_id.eq(SESSION_ID)].copy()
    ccf_mm = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    ccf_um = ccf_mm * 1000.0

    # ephys local Jacobian (deg/mm)
    from checkpoint_joint_multistructure_dispersion_likelihood import load_all  # noqa: E402
    pop = load_all()
    pop = pop.loc[~pop.center_bound & pop[CCF2 + ["rf_x", "rf_y"]].notna().all(axis=1)].copy()
    spec = DOMAINS["cortex"]
    ephys_cells = pop.loc[pop.structure_group.isin(spec["groups"])].reset_index(drop=True)
    e_axis0, e_axis1, e_grid = make_grid(ephys_cells[CCF2].to_numpy(float), spec["grid_step"], spec["grid_margin"])
    field = local_linear_jacobian_field(ephys_cells, e_grid, spec["bandwidth"], min_effective_n=spec["min_effective_n"], min_cell_count=spec["min_cell_count"])
    components, eff_n_interp = jacobian_interpolators(field, e_axis0, e_axis1)
    ephys_jac_mm = evaluate_jacobian(components, ccf_um) * 1000.0
    ephys_eff_n = eff_n_interp(ccf_um)

    # Zhuang (deg/mm)
    linear, intercept = recover_ccf_to_pixel_affine(SESSION_ID)
    pixel_xy = ccf_mm @ linear.T + intercept
    zhuang_fields = zhuang_gradient_interpolators()
    zhuang_jac_mm = sample_zhuang_jacobian_px(zhuang_fields, pixel_xy) @ linear

    # Garrett (deg/mm)
    garrett_manifest = json.loads(GARRETT_MANIFEST.read_text())
    g_linear = np.asarray(garrett_manifest["matrix_panel_units_per_mm"], dtype=float)
    g_center = np.asarray(garrett_manifest["ccf_center_ap_ml_mm"], dtype=float)
    g_template_center = np.asarray(garrett_manifest["template_center_xy"], dtype=float)
    garrett_fields = build_garrett_fields()
    garrett_xy = (ccf_mm - g_center) @ g_linear.T + g_template_center
    garrett_jac_panel = sample_garrett_jacobian_px(garrett_fields, garrett_xy)
    garrett_jac_mm = garrett_jac_panel @ g_linear

    # Naive pooled ephys map (deg/mm directly, built in CCF space already)
    n_axis0, n_axis1, naive_fields = build_naive_field()
    naive_components = naive_jacobian_interpolators(n_axis0, n_axis1, naive_fields)
    naive_jac_um = sample_naive_jacobian(naive_components, ccf_um)
    naive_jac_mm = naive_jac_um * 1000.0

    rows = []
    for i in range(len(session_cells)):
        row = {
            "ecephys_unit_id": session_cells.iloc[i].ecephys_unit_id,
            "area": session_cells.iloc[i].ecephys_structure_acronym,
            "ephys_effective_n": ephys_eff_n[i],
        }
        for name, jac in (("zhuang", zhuang_jac_mm[i]), ("garrett", garrett_jac_mm[i]),
                          ("naive", naive_jac_mm[i]), ("ephys", ephys_jac_mm[i])):
            row[f"{name}_azimuth_grad"] = jac[0]
            row[f"{name}_elevation_grad"] = jac[1]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "three_way_per_cell_gradients.csv", index=False)

    valid = result.ephys_effective_n >= spec["min_effective_n"]
    summary_rows = []
    for source in ("zhuang", "garrett", "naive"):
        for kind in ("azimuth", "elevation"):
            atlas = np.stack(result[f"{source}_{kind}_grad"].to_numpy())
            ephys = np.stack(result[f"ephys_{kind}_grad"].to_numpy())
            ok = valid.to_numpy() & np.isfinite(atlas).all(axis=1) & np.isfinite(ephys).all(axis=1)
            if ok.sum() < 3:
                summary_rows.append({"source": source, "kind": kind, "n": int(ok.sum())})
                continue
            cos = np.sum(atlas[ok] * ephys[ok], axis=1) / (np.linalg.norm(atlas[ok], axis=1) * np.linalg.norm(ephys[ok], axis=1) + 1e-12)
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            summary_rows.append({
                "source": source, "kind": kind, "n": int(ok.sum()),
                "median_angle_deg": float(np.median(angle)),
                "fraction_within_45deg": float(np.mean(angle <= 45)),
                "atlas_median_magnitude": float(np.median(np.linalg.norm(atlas[ok], axis=1))),
                "ephys_median_magnitude": float(np.median(np.linalg.norm(ephys[ok], axis=1))),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "three_way_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
