#!/usr/bin/env python3
"""Extend the Zhuang domain mask using the raw sparse contour points themselves, not a guess.

v1 (rectangular dilation over a guessed pinch window) and v2 (a hand-drawn line between
user-specified endpoints) both patched based on visual inspection of the domain SHAPE. The
user then pointed out something more direct: the raw digitized contour points
(`azimuth_deg`/`altitude_deg` sparse arrays in retinotopy_contour_grid.npz) already cover
much of the region missing from the boundary-line-derived `domain` -- 1852 of 2405 sparse
azimuth points and 495 of 793 sparse altitude points in the row[140:300]/col[300:400] gap
region fall OUTSIDE the current domain. That's direct evidence Zhuang's figure has real
mapped cortex there; the boundary-line extraction just didn't capture it as an enclosed
region. So domain is redefined here as the union of the boundary-derived domain with a
small dilation of "has any contour point" (radius=5, closing radius=2 to smooth the result
and fill small internal specks), which is evidence-based rather than a manual guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
RAW_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
OUTPUT = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
CONTOUR_DILATION_PX = 5
CLOSING_PX = 2
GRADIENT_SIGMA_PX = 2.0


def reconstruct(sparse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(sparse.shape)
    observed = np.isfinite(sparse)
    linear = griddata(
        np.column_stack([columns[observed], rows[observed]]), sparse[observed], (columns, rows), method="linear",
    )
    support = np.isfinite(linear)
    surface = linear.copy()
    missing = ~support
    nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    surface[missing] = surface[tuple(nearest[:, missing])]
    return surface.astype(np.float32), support


def normalized_smooth(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weights = ndimage.gaussian_filter(mask.astype(float), sigma=sigma, mode="constant")
    numerator = ndimage.gaussian_filter(np.where(mask, field, 0.0), sigma=sigma, mode="constant")
    result = numerator / np.maximum(weights, 1e-9)
    result[~mask] = np.nan
    return result


def main() -> None:
    raw = np.load(RAW_TEMPLATE)
    boundary = raw["mean_field_sign_boundary"].astype(bool)
    domain = ndimage.binary_fill_holes(boundary)
    original_px = int(domain.sum())

    sparse_altitude = raw["altitude_deg"].astype(float)
    sparse_azimuth = raw["azimuth_deg"].astype(float)
    has_data = np.isfinite(sparse_azimuth) | np.isfinite(sparse_altitude)
    print(f"raw contour-data pixels: {int(has_data.sum())}")

    dilated = ndimage.binary_dilation(
        has_data, structure=ndimage.generate_binary_structure(2, 2), iterations=CONTOUR_DILATION_PX,
    )
    closed = ndimage.binary_closing(
        dilated, structure=ndimage.generate_binary_structure(2, 1), iterations=CLOSING_PX,
    )
    patched = ndimage.binary_fill_holes(domain | closed)

    struct = ndimage.generate_binary_structure(2, 1)
    n_before = ndimage.label(ndimage.binary_erosion(domain, structure=struct, iterations=1))[1]
    n_after = ndimage.label(ndimage.binary_erosion(patched, structure=struct, iterations=1))[1]
    n_full = ndimage.label(patched)[1]
    print(f"before: domain={original_px}px, {n_before} components after 1px erosion")
    print(f"after: domain={int(patched.sum())}px (+{int(patched.sum()) - original_px}px from contour evidence), "
          f"{n_full} connected component(s) unfiltered, {n_after} after 1px erosion")

    altitude, _ = reconstruct(sparse_altitude)
    azimuth, _ = reconstruct(sparse_azimuth)
    altitude_smoothed = normalized_smooth(altitude, patched, GRADIENT_SIGMA_PX)
    azimuth_smoothed = normalized_smooth(azimuth, patched, GRADIENT_SIGMA_PX)

    out_path = OUTPUT / "interpolated_fields_and_field_sign_domain_patched.npz"
    np.savez_compressed(
        out_path,
        azimuth_deg=azimuth, elevation_deg=altitude,
        azimuth_smoothed_for_gradient_deg=azimuth_smoothed,
        elevation_smoothed_for_gradient_deg=altitude_smoothed,
        domain=patched, published_field_sign_boundary=boundary,
    )
    print(out_path)

    manifest = {
        "status": "domain patch v3: contour-point-evidence-based extension, replacing the v1/v2 manual guesses",
        "method": "domain = fill_holes(boundary_domain OR closing(dilation(has_contour_point, r=5), r=2))",
        "contour_dilation_px": CONTOUR_DILATION_PX,
        "closing_px": CLOSING_PX,
        "domain_px_before": original_px,
        "domain_px_after": int(patched.sum()),
        "components_after_1px_erosion_before": int(n_before),
        "components_after_1px_erosion_after": int(n_after),
    }
    (OUTPUT / "domain_patch_manifest.json").write_text(json.dumps(manifest, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(np.where(domain, azimuth, np.nan), origin="upper", cmap="viridis", vmin=0, vmax=90)
    axes[0].set_title("before patch")
    axes[1].imshow(np.where(patched, azimuth_smoothed, np.nan), origin="upper", cmap="viridis", vmin=0, vmax=90)
    axes[1].set_title("after patch (contour-evidence-based, smoothed)")
    fig.savefig(OUTPUT / "Figure_domain_patch_before_after.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
