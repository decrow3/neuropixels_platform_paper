#!/usr/bin/env python3
"""Scatter-over-map overlay for the translation+rotation+isotropic-scale variant, at whatever
grid range is currently being tested in `fit_translation_rotation_isotropic_scale_cv.py` --
visual companion to the numeric CV result, not a replacement for it. Each session gets its own
best IN-SAMPLE (all own cells, no held-out probe) rotation+scale+offset fit over the given grid,
then all sessions are pooled and rendered over the same span-matched Zhuang background used
throughout, same style as `render_warp_variant_overlays.py`.

Run this alongside every rotation/scale CV variant from now on: the numbers say whether a
candidate generalizes, but the overlay is what makes it legible WHY -- e.g. a wide, unconstrained
grid can win narrowly on a held-out numeric score for reasons that look obviously wrong (sessions
scattered to incoherent positions, cells landing outside the domain, systematically implausible
per-session parameters) the moment you look at the map.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from render_warp_variant_overlays import (  # noqa: E402
    MIN_CELLS, build_interpolators, render_overlay, sample,
)

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_warp_cv"

# Match the "wider" grid currently being tested for the CV comparison.
DTHETA_GRID_DEG = np.linspace(-60, 60, 25)
DSCALE_GRID = np.linspace(0.3, 3.0, 28)


def best_isotropic_in_sample_fit(ccf, naive_rf, geometry, az_interp, el_interp, dtheta_grid, dscale_grid):
    """Same idea as render_warp_variant_overlays.best_in_sample_fit, but scale is a single
    isotropic factor (dscale_ap == dscale_ml), matching the actual CV grid dimensionality --
    NOT the independent-AP/ML search `best_in_sample_fit` performs."""
    theta_mesh, scale_mesh = np.meshgrid(dtheta_grid, dscale_grid, indexing="ij")
    dtheta = theta_mesh.ravel()
    dscale = scale_mesh.ravel()
    predicted = sample(ccf, geometry, np.radians(dtheta), dscale, dscale, az_interp, el_interp)

    losses = np.full(len(dtheta), np.inf)
    offsets = np.full((len(dtheta), 2), np.nan)
    for k in range(len(dtheta)):
        valid = np.isfinite(predicted[k]).all(axis=1)
        if valid.sum() < MIN_CELLS:
            continue
        offset = huber_location(predicted[k][valid] - naive_rf[valid])
        residual = predicted[k][valid] - offset - naive_rf[valid]
        losses[k] = float(np.median(np.linalg.norm(residual, axis=1)))
        offsets[k] = offset
    best = int(np.argmin(losses))
    return {
        "dtheta_deg": float(dtheta[best]), "dscale_ap": float(dscale[best]), "dscale_ml": float(dscale[best]),
        "offset_az": float(offsets[best, 0]), "offset_el": float(offsets[best, 1]),
        "in_sample_median_error_deg": float(losses[best]),
    }


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    smoothed, az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    rows = []
    for sid, session_cells in cells.groupby("ecephys_session_id"):
        ccf = session_cells[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        naive_rf = session_cells[["normalized_rf_x", "normalized_rf_y"]].to_numpy(float)
        fit = best_isotropic_in_sample_fit(ccf, naive_rf, geometry, az_interp, el_interp,
                                            DTHETA_GRID_DEG, DSCALE_GRID)
        fit["ecephys_session_id"] = int(sid)
        fit["n_cells"] = len(session_cells)
        rows.append(fit)
    fits = pd.DataFrame(rows)
    fits.to_csv(OUTPUT / "in_sample_fit_rotation_isotropic_scale_wider.csv", index=False)

    print(f"chosen dtheta: median={fits.dtheta_deg.median():+.1f} deg, "
          f"IQR=[{fits.dtheta_deg.quantile(.25):+.1f}, {fits.dtheta_deg.quantile(.75):+.1f}]")
    print(f"chosen dscale: median={fits.dscale_ap.median():.2f}x, "
          f"IQR=[{fits.dscale_ap.quantile(.25):.2f}, {fits.dscale_ap.quantile(.75):.2f}]")
    print(f"median in-sample error: {fits.in_sample_median_error_deg.median():.2f} deg")

    title = "Rotation + isotropic scale + offset -- WIDE grid (theta +/-60 deg, scale 0.3-3.0x), in-sample best fit per session"
    subtitle = (
        "CAVEAT: in-sample (no held-out probe) -- an optimistic upper bound. The leave-one-probe-out CV at this "
        "grid range showed median held-out error CHANGE of -13.7 deg (i.e. WORSE), only 18% of folds improved, "
        "p=1.6e-9 -- this is what unconstrained overfitting looks like spatially, not a real registration."
    )
    render_overlay(cells, fits, geometry, smoothed, title, subtitle,
                    OUTPUT / "Figure_variant_rotation_isotropic_scale_wider_over_zhuang.png")

    render_own_transform_pixel_space(cells, fits, geometry, smoothed,
                                      OUTPUT / "Figure_variant_rotation_isotropic_scale_wider_own_transform_pixel_space.png")


def render_own_transform_pixel_space(cells, fits, geometry, smoothed, output_path):
    """The more honest 'why this fails' view: each session's cells placed in ATLAS PIXEL SPACE
    using THAT SESSION'S OWN fitted (dtheta, dscale) -- not the shared default geometry. Wild
    per-session rotation/scale should show up directly here as incoherent, overlapping, or
    absurdly spread-out point clouds relative to the actual atlas domain."""
    merged = cells.merge(fits, on="ecephys_session_id", how="left")
    tx, ty = geometry["fitted_translation_px"]
    px_per_mm = geometry["fixed_scale_px_per_mm"]
    ml_sign = -1.0 if geometry["fixed_reflection_ml"] else 1.0
    v1_anchor = np.array(geometry["v1_anchor_ccf_ap_ml_mm"])
    v1_seed_col, v1_seed_row = geometry["v1_seed_xy_px"]
    pixel_center = np.array([v1_seed_col + tx, v1_seed_row + ty])

    domain = smoothed["domain"].astype(bool)
    boundary = smoothed["published_field_sign_boundary"].astype(bool)
    height, width = domain.shape

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.imshow(domain, cmap="Greys", alpha=0.15, origin="upper")
    boundary_rows, boundary_cols = np.nonzero(boundary)
    ax.scatter(boundary_cols, boundary_rows, s=0.5, color="#343434", zorder=1, rasterized=True)

    session_ids = sorted(merged.ecephys_session_id.unique())
    cmap = plt.get_cmap("turbo")
    for i, sid in enumerate(session_ids):
        group = merged.loc[merged.ecephys_session_id == sid]
        theta = np.radians(geometry["fitted_rotation_deg"]) + np.radians(group.dtheta_deg.iloc[0])
        dscale = group.dscale_ap.iloc[0]  # isotropic: dscale_ap == dscale_ml
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale_reflect = np.diag([ml_sign * px_per_mm * dscale, px_per_mm * dscale])
        matrix = rotation @ scale_reflect
        ccf = group[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        delta = ccf - v1_anchor
        delta_ml_ap = delta[:, [1, 0]]
        xy = delta_ml_ap @ matrix.T + pixel_center
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.5, color=cmap(i / max(len(session_ids) - 1, 1)),
                   linewidths=0, zorder=2, rasterized=True)

    ax.set(title="Each session's cells in ATLAS PIXEL SPACE, using THAT SESSION'S OWN fitted\n"
                 "rotation+isotropic-scale (wide grid) -- color = session (violet->red, arbitrary order)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)")
    ax.set_aspect("equal")
    margin = 0.5 * max(height, width)
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
