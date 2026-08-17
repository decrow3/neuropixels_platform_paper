#!/usr/bin/env python3
"""Pilot: compare smoothed-Zhuang local RF gradient to the ephys-derived local Jacobian,
in deg/mm of cortex, for one session (781842082).

Step 1 of the atlas-anchor pilot (Garrett comparison and the full multi-session anchor come
after this is inspected). Zhuang's CCF<->pixel affine for this session is recovered exactly
(not refit) from the five saved probe-landmark correspondences in
`registered_probe_landmarks.csv` via ordinary least squares -- this is provably identical to
the original nonlinear fit for an affine model (residual ~1e-13 px), and far cheaper than
rerunning `differential_evolution`. The already-built support-masked smoothed Zhuang fields
(`interpolation_field_sign_qa/interpolated_fields_and_field_sign.npz`) are reused as-is.
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
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_14animal_retinotopy_registration import production_support  # noqa: E402
from fit_multistructure_fixed_effect_translation import (  # noqa: E402
    CCF2, DOMAINS, evaluate_jacobian, jacobian_interpolators, local_linear_jacobian_field, make_grid,
)


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 781842082
ZHUANG_SMOOTH = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign.npz"
)
LANDMARKS_14 = ROOT / "artifacts/retinotopy_cross_animal_registration_14_v1/registered_probe_landmarks.csv"
OUTPUT = ROOT / "artifacts/retinotopy_template/atlas_gradient_vs_ephys_pilot_781842082"


def recover_ccf_to_pixel_affine(session_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(linear, intercept) for pixel = ccf_mm @ linear.T + intercept.

    Recovered exactly (not refit) via OLS on saved landmarks -- both pieces are needed:
    `linear` alone (d(pixel)/d(ccf_mm)) is correct for converting gradients via the chain
    rule, but sampling the field at the right LOCATION needs the full affine including the
    intercept, or every query lands at the wrong pixel.
    """
    landmarks = pd.read_csv(LANDMARKS_14)
    session = landmarks.loc[landmarks.session_id.eq(session_id)]
    ccf = session[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    pixel = session[["template_x_px", "template_y_px"]].to_numpy(float)
    design = np.column_stack([ccf, np.ones(len(ccf))])
    coefficient, *_ = np.linalg.lstsq(design, pixel, rcond=None)
    residual = np.abs(design @ coefficient - pixel).max()
    print(f"affine recovery max residual: {residual:.2e} px (should be ~machine precision)")
    return coefficient[:2].T, coefficient[2]  # linear (2 pixel, 2 ccf), intercept (2,)


def zhuang_gradient_interpolators() -> dict:
    source = np.load(ZHUANG_SMOOTH)
    azimuth = source["azimuth_smoothed_for_gradient_deg"]
    elevation = source["elevation_smoothed_for_gradient_deg"]
    valid = np.isfinite(azimuth) & np.isfinite(elevation)
    azimuth_filled = np.where(valid, azimuth, 0.0)
    elevation_filled = np.where(valid, elevation, 0.0)
    da_drow, da_dx = np.gradient(azimuth_filled)
    de_drow, de_dx = np.gradient(elevation_filled)
    da_dy = -da_drow  # image row points down; template y points up (matches field_sign())
    de_dy = -de_drow
    rows, cols = azimuth.shape
    axis_row = np.arange(rows)
    axis_col = np.arange(cols)
    components = {}
    for name, array in (("da_dx", da_dx), ("da_dy", da_dy), ("de_dx", de_dx), ("de_dy", de_dy)):
        masked = np.where(valid, array, np.nan)
        components[name] = RegularGridInterpolator((axis_row, axis_col), masked, bounds_error=False, fill_value=np.nan)
    valid_interp = RegularGridInterpolator(
        (axis_row, axis_col), valid.astype(float), bounds_error=False, fill_value=0.0
    )
    return {"components": components, "valid": valid_interp}


def sample_zhuang_jacobian_px(fields: dict, pixel_xy: np.ndarray) -> np.ndarray:
    row_col = pixel_xy[:, ::-1]
    jac = np.full((len(pixel_xy), 2, 2), np.nan)
    jac[:, 0, 0] = fields["components"]["da_dx"](row_col)
    jac[:, 0, 1] = fields["components"]["da_dy"](row_col)
    jac[:, 1, 0] = fields["components"]["de_dx"](row_col)
    jac[:, 1, 1] = fields["components"]["de_dy"](row_col)
    return jac


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ccf_to_px, ccf_to_px_intercept = recover_ccf_to_pixel_affine(SESSION_ID)  # d(pixel)/d(ccf_mm), offset

    cells, audit = production_support()
    session_cells = cells.loc[cells.session_id.eq(SESSION_ID)].copy()
    print(f"session {SESSION_ID}: {len(session_cells)} cells across "
          f"{session_cells.ecephys_probe_id.nunique()} probes / {session_cells.ecephys_structure_acronym.nunique()} areas")

    ccf_mm = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
    pixel_xy = ccf_mm @ ccf_to_px.T + ccf_to_px_intercept  # full affine, needed for correct sampling location

    zhuang_fields = zhuang_gradient_interpolators()
    jac_px = sample_zhuang_jacobian_px(zhuang_fields, pixel_xy)
    # chain rule: d(RF)/d(ccf_mm) = d(RF)/d(pixel) @ d(pixel)/d(ccf_mm)
    zhuang_jac_mm = jac_px @ ccf_to_px

    # Ephys-derived local Jacobian field (cortex domain), sampled at this session's own cells.
    from checkpoint_joint_multistructure_dispersion_likelihood import load_all  # noqa: E402
    pop = load_all()
    pop = pop.loc[~pop.center_bound & pop[CCF2 + ["rf_x", "rf_y"]].notna().all(axis=1)].copy()
    spec = DOMAINS["cortex"]
    ephys_cortex_cells = pop.loc[pop.structure_group.isin(spec["groups"])].reset_index(drop=True)
    axis0, axis1, grid = make_grid(ephys_cortex_cells[CCF2].to_numpy(float), spec["grid_step"], spec["grid_margin"])
    field = local_linear_jacobian_field(
        ephys_cortex_cells, grid, spec["bandwidth"],
        min_effective_n=spec["min_effective_n"], min_cell_count=spec["min_cell_count"],
    )
    components, eff_n_interp = jacobian_interpolators(field, axis0, axis1)

    # ephys CCF columns are in um (anterior_posterior_ccf_coordinate etc.); session_cells' ccf_ap_mm/ccf_ml_mm are in mm.
    ephys_query_um = ccf_mm * 1000.0
    ephys_jac_um = evaluate_jacobian(components, ephys_query_um)  # deg per um
    ephys_jac_mm = ephys_jac_um * 1000.0  # deg per mm
    ephys_eff_n = eff_n_interp(ephys_query_um)

    rows = []
    for i in range(len(session_cells)):
        row = {
            "ecephys_unit_id": session_cells.iloc[i].ecephys_unit_id,
            "ecephys_probe_id": session_cells.iloc[i].ecephys_probe_id,
            "area": session_cells.iloc[i].ecephys_structure_acronym,
            "ccf_ap_mm": ccf_mm[i, 0], "ccf_ml_mm": ccf_mm[i, 1],
            "ephys_effective_n": ephys_eff_n[i],
        }
        for name, jac in (("zhuang", zhuang_jac_mm[i]), ("ephys", ephys_jac_mm[i])):
            row[f"{name}_da_dap_degmm"] = jac[0, 0]
            row[f"{name}_da_dml_degmm"] = jac[0, 1]
            row[f"{name}_de_dap_degmm"] = jac[1, 0]
            row[f"{name}_de_dml_degmm"] = jac[1, 1]
            row[f"{name}_azimuth_gradient_magnitude_degmm"] = float(np.hypot(jac[0, 0], jac[0, 1]))
            row[f"{name}_elevation_gradient_magnitude_degmm"] = float(np.hypot(jac[1, 0], jac[1, 1]))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "per_cell_atlas_vs_ephys_jacobian.csv", index=False)

    valid = (
        result[[c for c in result.columns if "degmm" in c]].notna().all(axis=1)
        & (result.ephys_effective_n >= spec["min_effective_n"])
    )
    valid_result = result.loc[valid]
    print(f"cells with both atlas and ephys Jacobian available: {valid.sum()} / {len(result)}")

    summary = {}
    for kind in ("azimuth", "elevation"):
        z = valid_result[[f"zhuang_d{kind[0]}_dap_degmm", f"zhuang_d{kind[0]}_dml_degmm"]].to_numpy(float)
        e = valid_result[[f"ephys_d{kind[0]}_dap_degmm", f"ephys_d{kind[0]}_dml_degmm"]].to_numpy(float)
        cos_angle = np.sum(z * e, axis=1) / (np.linalg.norm(z, axis=1) * np.linalg.norm(e, axis=1) + 1e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        summary[kind] = {
            "n_cells": int(len(valid_result)),
            "median_angle_between_gradients_deg": float(np.median(angle_deg)),
            "fraction_within_45deg": float(np.mean(angle_deg <= 45)),
            "zhuang_median_gradient_magnitude_degmm": float(valid_result[f"zhuang_{kind}_gradient_magnitude_degmm"].median()),
            "ephys_median_gradient_magnitude_degmm": float(valid_result[f"ephys_{kind}_gradient_magnitude_degmm"].median()),
        }
    print(json.dumps(summary, indent=2))
    (OUTPUT / "summary.json").write_text(json.dumps({
        "session_id": SESSION_ID,
        "note": "gradients in deg RF per mm cortex (CCF AP/ML); Zhuang affine recovered exactly via OLS on saved landmarks",
        **summary,
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, kind in zip(axes, ("azimuth", "elevation")):
        z = valid_result[[f"zhuang_d{kind[0]}_dap_degmm", f"zhuang_d{kind[0]}_dml_degmm"]].to_numpy(float)
        e = valid_result[[f"ephys_d{kind[0]}_dap_degmm", f"ephys_d{kind[0]}_dml_degmm"]].to_numpy(float)
        for row_z, row_e, area in zip(z, e, valid_result.area):
            ax.plot([0, row_z[0]], [0, row_z[1]], color="#4c78a8", alpha=.5, lw=1)
            ax.plot([0, row_e[0]], [0, row_e[1]], color="#d95f5f", alpha=.5, lw=1)
        ax.scatter(*z.T, s=10, color="#4c78a8", label="Zhuang (smoothed)")
        ax.scatter(*e.T, s=10, color="#d95f5f", label="ephys local Jacobian")
        ax.axhline(0, color=".8", lw=.6)
        ax.axvline(0, color=".8", lw=.6)
        ax.set(xlabel="d(RF)/d(AP) (deg/mm)", ylabel="d(RF)/d(ML) (deg/mm)",
               title=f"{kind} gradient vectors, session {SESSION_ID}", aspect="equal")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_gradient_vectors_zhuang_vs_ephys.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
