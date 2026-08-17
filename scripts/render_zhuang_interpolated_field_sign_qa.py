#!/usr/bin/env python3
"""Render the continuous Zhuang Figure 9 interpolants and their field sign.

The registration code uses piecewise-linear interpolation between decoded
5-degree contour pixels, followed by nearest-contour filling where linear
interpolation has no support.  This script reproduces that surface exactly.
A small Gaussian smoothing is used only when differentiating the maps for the
field-sign diagnostic; it is not part of RF sampling in the registration fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
)
DEFAULT_OUTPUT = DEFAULT_TEMPLATE.parent / "interpolation_field_sign_qa"


def reconstruct(sparse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the linear + nearest-fill surface used by registration."""
    rows, columns = np.indices(sparse.shape)
    observed = np.isfinite(sparse)
    linear = griddata(
        np.column_stack([columns[observed], rows[observed]]),
        sparse[observed],
        (columns, rows),
        method="linear",
    )
    support = np.isfinite(linear)
    surface = linear.copy()
    missing = ~support
    nearest = ndimage.distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    surface[missing] = surface[tuple(nearest[:, missing])]
    return surface.astype(np.float32), support


def normalized_smooth(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing without bleeding across the displayed domain edge."""
    weights = ndimage.gaussian_filter(mask.astype(float), sigma=sigma, mode="constant")
    numerator = ndimage.gaussian_filter(
        np.where(mask, field, 0.0), sigma=sigma, mode="constant"
    )
    result = numerator / np.maximum(weights, 1e-9)
    result[~mask] = np.nan
    return result


def field_sign(
    azimuth: np.ndarray, elevation: np.ndarray, mask: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized gradient cross-product in an x-right/y-up frame."""
    azimuth_smooth = normalized_smooth(azimuth, mask, sigma)
    elevation_smooth = normalized_smooth(elevation, mask, sigma)

    # Fill outside-domain NaNs only for stable numerical differentiation; all
    # resulting values are masked again below.
    azi_for_gradient = np.where(mask, azimuth_smooth, 0.0)
    ele_for_gradient = np.where(mask, elevation_smooth, 0.0)
    da_drow, da_dx = np.gradient(azi_for_gradient)
    de_drow, de_dx = np.gradient(ele_for_gradient)
    da_dy = -da_drow  # image row points down, while template y points up
    de_dy = -de_drow
    azi_strength = np.hypot(da_dx, da_dy)
    ele_strength = np.hypot(de_dx, de_dy)
    strength_product = azi_strength * ele_strength
    sign = (da_dx * de_dy - da_dy * de_dx) / np.maximum(strength_product, 1e-12)
    sign[~mask] = np.nan
    return sign, strength_product, azimuth_smooth, elevation_smooth


def sign_reversal_mask(sign: np.ndarray, valid: np.ndarray) -> np.ndarray:
    positive = sign >= 0
    reversal = np.zeros(sign.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (positive[:, 1:] != positive[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (positive[1:, :] != positive[:-1, :])
    reversal[:, 1:] |= horizontal
    reversal[:, :-1] |= horizontal
    reversal[1:, :] |= vertical
    reversal[:-1, :] |= vertical
    return reversal


def add_boundary(ax: plt.Axes, boundary: np.ndarray, *, color: str = "#d000d0") -> None:
    ax.contour(
        boundary.astype(float), levels=[0.5], colors=color, linewidths=0.8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gradient-sigma-px", type=float, default=2.0)
    parser.add_argument("--weak-gradient-percentile", type=float, default=10.0)
    args = parser.parse_args()

    template_path = args.template.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = np.load(template_path)

    boundary = source["mean_field_sign_boundary"].astype(bool)
    domain = ndimage.binary_fill_holes(boundary)
    sparse_elevation = source["altitude_deg"].astype(float)
    sparse_azimuth = source["azimuth_deg"].astype(float)
    elevation, elevation_support = reconstruct(sparse_elevation)
    azimuth, azimuth_support = reconstruct(sparse_azimuth)
    joint_support = domain & elevation_support & azimuth_support

    sign, gradient_product, azimuth_smooth, elevation_smooth = field_sign(
        azimuth, elevation, domain, args.gradient_sigma_px
    )
    threshold = float(
        np.nanpercentile(gradient_product[joint_support], args.weak_gradient_percentile)
    )
    sign_valid = joint_support & (gradient_product >= threshold)
    sign_display = np.where(sign_valid, sign, np.nan)
    reversals = sign_reversal_mask(sign, sign_valid)

    boundary_distance = ndimage.distance_transform_edt(~boundary)
    reversal_distances = boundary_distance[reversals]
    metrics = {
        "template": str(template_path),
        "shape_rows_columns": list(map(int, boundary.shape)),
        "interpolation": "piecewise linear, then nearest-contour fill outside linear support",
        "field_sign_definition": "det(grad(azimuth), grad(elevation)) / (|grad(azimuth)| |grad(elevation)|), x right and y up",
        "gradient_smoothing_sigma_px": args.gradient_sigma_px,
        "weak_gradient_percentile_excluded": args.weak_gradient_percentile,
        "domain_pixels": int(domain.sum()),
        "azimuth_linear_support_fraction_of_domain": float((azimuth_support & domain).sum() / domain.sum()),
        "elevation_linear_support_fraction_of_domain": float((elevation_support & domain).sum() / domain.sum()),
        "joint_linear_support_fraction_of_domain": float(joint_support.sum() / domain.sum()),
        "field_sign_valid_fraction_of_domain": float(sign_valid.sum() / domain.sum()),
        "field_sign_positive_fraction_of_valid": float(np.mean(sign[sign_valid] >= 0)),
        "computed_reversal_pixels": int(reversals.sum()),
        "computed_reversal_median_distance_to_published_boundary_px": float(np.median(reversal_distances)),
        "computed_reversal_fraction_within_5px_of_published_boundary": float(np.mean(reversal_distances <= 5)),
        "computed_reversal_fraction_within_10px_of_published_boundary": float(np.mean(reversal_distances <= 10)),
        "caveat": "The published mask includes outer and inter-area borders, so reversal-to-border distance is a directional diagnostic, not a segmentation score.",
    }

    np.savez_compressed(
        output / "interpolated_fields_and_field_sign.npz",
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        azimuth_smoothed_for_gradient_deg=azimuth_smooth,
        elevation_smoothed_for_gradient_deg=elevation_smooth,
        field_sign=sign,
        field_sign_valid=sign_valid,
        linear_support_azimuth=azimuth_support,
        linear_support_elevation=elevation_support,
        domain=domain,
        published_field_sign_boundary=boundary,
        computed_sign_reversal=reversals,
    )
    (output / "run_manifest.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11), constrained_layout=True)
    map_specs = (
        (axes[0, 0], azimuth, sparse_azimuth, "Azimuth", 0, 90),
        (axes[0, 1], elevation, sparse_elevation, "Elevation (published as altitude)", -25, 30),
    )
    for ax, surface, sparse, title, lower, upper in map_specs:
        shown = ax.imshow(
            surface, cmap="turbo", vmin=lower, vmax=upper, origin="upper"
        )
        ax.contour(
            sparse, levels=np.arange(lower, upper + 0.1, 5), cmap="turbo",
            vmin=lower, vmax=upper, linewidths=0.65,
        )
        add_boundary(ax, boundary)
        ax.contour(
            (~joint_support & domain).astype(float), levels=[0.5], colors="#555555",
            linewidths=0.55, linestyles="dashed",
        )
        ax.set_title(f"{title}: current registration interpolant", loc="left")
        ax.set_xlabel("Zhuang common-map x (px)")
        ax.set_ylabel("Zhuang common-map y (px; high → low)")
        ax.set_aspect("equal")
        figure.colorbar(shown, ax=ax, fraction=0.046, label="degrees")

    sign_ax = axes[1, 0]
    shown = sign_ax.imshow(
        sign_display, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        origin="upper",
    )
    add_boundary(sign_ax, boundary)
    sign_ax.contour(
        np.where(sign_valid, sign, np.nan), levels=[0], colors="black",
        linewidths=0.85, linestyles="dashed",
    )
    sign_ax.set_title(
        f"Field sign from map gradients (Gaussian σ={args.gradient_sigma_px:g} px)",
        loc="left",
    )
    sign_ax.set_xlabel("Zhuang common-map x (px)")
    sign_ax.set_ylabel("Zhuang common-map y (px; high → low)")
    sign_ax.set_aspect("equal")
    figure.colorbar(shown, ax=sign_ax, fraction=0.046, label="normalized field sign")

    qa_ax = axes[1, 1]
    qa = np.zeros((*domain.shape, 3), dtype=float)
    qa[:] = (0.92, 0.92, 0.92)
    qa[domain] = (0.76, 0.86, 0.96)
    qa[joint_support] = (0.88, 0.95, 0.88)
    qa[sign_valid] = (1.0, 1.0, 1.0)
    qa_ax.imshow(qa, origin="upper")
    add_boundary(qa_ax, boundary)
    qa_ax.contour(
        reversals.astype(float), levels=[0.5], colors="black", linewidths=0.8,
    )
    qa_ax.set_title("Evidence support and sign-reversal comparison", loc="left")
    qa_ax.set_xlabel("Zhuang common-map x (px)")
    qa_ax.set_ylabel("Zhuang common-map y (px; high → low)")
    qa_ax.set_aspect("equal")
    qa_ax.text(
        0.02, 0.02,
        "white: joint linear support + stable gradients\n"
        "green: joint linear support, weak gradients\n"
        "blue: nearest-filled in ≥1 map\n"
        "magenta: published mean field-sign borders\n"
        "black: computed sign reversal",
        transform=qa_ax.transAxes, va="bottom", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.7"},
    )

    figure.suptitle(
        "Zhuang et al. (2017) Figure 9: continuous contour interpolation and field sign",
        fontsize=15,
    )
    figure.savefig(
        output / "Figure_interpolated_azimuth_elevation_field_sign.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
