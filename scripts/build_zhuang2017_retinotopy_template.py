#!/usr/bin/env python3
"""Extract Zhuang et al. 2017 Figure 9C/D as an auditable template.

Figure 9 publishes population altitude and azimuth isolines at 5-degree
spacing over repeated mean field-sign borders.  The source is a lossless TIFF,
but the scientific layers are rasterized.  This script therefore preserves
them as sparse, degree-labelled grids and points rather than manufacturing a
continuous retinotopy surface between the published contours.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage


DEFAULT_OUTPUT = Path("artifacts/retinotopy_template/zhuang2017_figure9")
ARTICLE_URL = "https://elifesciences.org/articles/18372"
FIGURE_URL = (
    "https://iiif.elifesciences.org/lax/18372%2F"
    "elife-18372-fig9-v2.tif/full/full/0/default.tif"
)

# Pixel bounds in the official 1001 x 1452 eLife Figure 9 v2 TIFF.
# Both map crops have the same size. Panel C is registered onto panel D below
# using the repeated black field-sign borders rather than assuming translation.
Y0, Y1 = 535, 965
PANEL_BOUNDS = {
    "altitude": (0, Y0, 470, Y1),
    "azimuth": (500, Y0, 970, Y1),
}
SOURCE_CROP = (0, 475, 1001, 985)
EXPECTED_LEVELS = {
    "altitude": np.arange(-25.0, 30.1, 5.0),
    "azimuth": np.arange(0.0, 90.1, 5.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_crop(rgb: np.ndarray, name: str) -> np.ndarray:
    x0, y0, x1, y1 = PANEL_BOUNDS[name]
    return rgb[y0:y1, x0:x1]


def legend_mask(name: str, shape: tuple[int, int]) -> np.ndarray:
    """Mask the printed line-key while retaining the lateral cortical patch."""
    rows, columns = np.indices(shape)
    if name == "altitude":
        return (columns >= 355) & (rows >= 245)
    return (columns >= 395) & (rows >= 210)


def jet_base_colors(crop: np.ndarray, expected_count: int) -> np.ndarray:
    """Find exact publication stroke colors using frequency and jet proximity."""
    lut = np.asarray(colormaps["jet"](np.linspace(0, 1, 256)))[:, :3]
    counts = Counter(map(tuple, crop.reshape(-1, 3)))
    candidates: list[tuple[int, tuple[int, int, int]]] = []
    for rgb, count in counts.items():
        if count <= 40 or max(rgb) - min(rgb) <= 100:
            continue
        color = np.asarray(rgb, dtype=float) / 255.0
        squared = ((lut - color) ** 2).sum(axis=1)
        index = int(np.argmin(squared))
        if np.sqrt(squared[index]) < 0.015:
            candidates.append((index, rgb))
    candidates.sort()
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} contour base colors, found {len(candidates)}: {candidates}"
        )
    return np.asarray([rgb for _, rgb in candidates], dtype=np.uint8)


def decode_contours(
    crop: np.ndarray, name: str, base_colors: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign colored and anti-aliased contour pixels to their nearest base color."""
    pixels = crop.astype(float) / 255.0
    bases = base_colors.astype(float) / 255.0
    squared = ((pixels[:, :, None, :] - bases[None, None, :, :]) ** 2).sum(axis=3)
    nearest = squared.argmin(axis=2)
    error = np.sqrt(squared.min(axis=2))
    saturation = crop.max(axis=2).astype(int) - crop.min(axis=2).astype(int)
    keep = (error <= 0.08) & (saturation >= 70) & ~legend_mask(name, crop.shape[:2])
    values = np.full(crop.shape[:2], np.nan, dtype=np.float32)
    values[keep] = levels[nearest[keep]].astype(np.float32)
    errors = np.full(crop.shape[:2], np.nan, dtype=np.float32)
    errors[keep] = error[keep].astype(np.float32)
    return values, errors, keep


def boundary_mask(crop: np.ndarray, name: str) -> np.ndarray:
    mask = crop.max(axis=2) <= 35
    mask &= ~legend_mask(name, crop.shape[:2])
    # Remove isolated antialias/noise specks without thickening the evidence.
    labels, count = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep_labels = np.flatnonzero(sizes >= 3) + 1
    return np.isin(labels, keep_labels)


def translate_array(array: np.ndarray, dx: int, dy: int, fill: float | bool) -> np.ndarray:
    output = np.full(array.shape, fill, dtype=array.dtype)
    height, width = array.shape
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    output[target_y0:target_y1, target_x0:target_x1] = array[source_y0:source_y1, source_x0:source_x1]
    return output


def estimate_panel_translation(
    altitude_boundary: np.ndarray, azimuth_boundary: np.ndarray
) -> tuple[int, int, dict[str, float]]:
    """Register C to D from their duplicated borders using a small integer search."""
    rows, columns = np.indices(altitude_boundary.shape)
    comparison = (rows >= 10) & (columns < 355)
    source = altitude_boundary & comparison
    target = azimuth_boundary & comparison
    target_distance = ndimage.distance_transform_edt(~target)
    source_y, source_x = np.where(source)
    scores: list[tuple[float, int, int]] = []
    height, width = source.shape
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            y = source_y + dy
            x = source_x + dx
            valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
            scores.append((float(target_distance[y[valid], x[valid]].mean()), dx, dy))
    _, dx, dy = min(scores)

    moved = translate_array(source, dx, dy, False)
    moved_distance = ndimage.distance_transform_edt(~moved)
    forward = target_distance[moved]
    reverse = moved_distance[target]
    distances = np.concatenate([forward, reverse])
    qa = {
        "mean_symmetric_boundary_distance_px": float(distances.mean()),
        "median_symmetric_boundary_distance_px": float(np.median(distances)),
        "p95_symmetric_boundary_distance_px": float(np.quantile(distances, 0.95)),
    }
    return dx, dy, qa


def write_points(
    path: Path,
    layers: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    translations: dict[str, tuple[int, int]],
) -> int:
    height, width = next(iter(layers.values())).shape
    fields = [
        "map",
        "value_deg",
        "x_common_px",
        "y_common_px",
        "x_common_fraction",
        "y_common_fraction_up",
        "source_panel",
        "source_x_px",
        "source_y_px",
        "nearest_base_color_error",
    ]
    count = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in layers.items():
            dx, dy = translations[name]
            source_values = translate_array(values, -dx, -dy, np.nan) if (dx or dy) else values
            source_errors = translate_array(errors[name], -dx, -dy, np.nan) if (dx or dy) else errors[name]
            for source_y, source_x in np.argwhere(np.isfinite(source_values)):
                x, y = int(source_x + dx), int(source_y + dy)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                writer.writerow(
                    {
                        "map": name,
                        "value_deg": float(source_values[source_y, source_x]),
                        "x_common_px": x,
                        "y_common_px": y,
                        "x_common_fraction": x / (width - 1),
                        "y_common_fraction_up": (height - 1 - y) / (height - 1),
                        "source_panel": "C" if name == "altitude" else "D",
                        "source_x_px": int(source_x),
                        "source_y_px": int(source_y),
                        "nearest_base_color_error": float(source_errors[source_y, source_x]),
                    }
                )
                count += 1
    return count


def render_qa(
    source_crop: Path,
    altitude: np.ndarray,
    azimuth: np.ndarray,
    altitude_boundary: np.ndarray,
    azimuth_boundary: np.ndarray,
    dx: int,
    dy: int,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.9, 1.1))
    source_ax = figure.add_subplot(grid[0, :])
    source_ax.imshow(Image.open(source_crop))
    source_ax.set_title("Published Figure 9C/D (lossless eLife source evidence)", loc="left")
    source_ax.axis("off")

    for column, name, values, limits in (
        (0, "altitude", altitude, (-25, 30)),
        (1, "azimuth", azimuth, (0, 90)),
    ):
        ax = figure.add_subplot(grid[1, column])
        shown = ax.imshow(values, cmap="jet", vmin=limits[0], vmax=limits[1], origin="upper")
        ax.contour(azimuth_boundary.astype(float), levels=[0.5], colors="black", linewidths=0.45)
        ax.set_title(f"Decoded {name} isolines")
        ax.set_xlabel("common-map x (panel-D pixels)")
        ax.set_ylabel("common-map y (panel-D pixels; down +)")
        ax.set_aspect("equal")
        figure.colorbar(shown, ax=ax, fraction=0.046, label="degrees")

    agreement_ax = figure.add_subplot(grid[1, 2])
    moved_altitude_boundary = translate_array(altitude_boundary, dx, dy, False)
    overlay = np.ones((*azimuth_boundary.shape, 3), dtype=float)
    overlay[moved_altitude_boundary] = (0.0, 0.75, 0.85)
    overlay[azimuth_boundary] = (0.9, 0.1, 0.75)
    overlay[moved_altitude_boundary & azimuth_boundary] = (0.0, 0.0, 0.0)
    agreement_ax.imshow(overlay, origin="upper")
    agreement_ax.set_title(f"Repeated-border agreement\nC → D translation: dx={dx}, dy={dy} px")
    agreement_ax.set_xlabel("common-map x (panel-D pixels)")
    agreement_ax.set_ylabel("common-map y (panel-D pixels; down +)")
    agreement_ax.set_aspect("equal")
    figure.suptitle("Zhuang et al. 2017 Figure 9 population-retinotopy extraction", fontsize=15)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_readme(output: Path, manifest: dict) -> None:
    source = manifest["source"]
    text = f"""# Zhuang et al. 2017 Figure 9 template — initial evidence checkpoint

This is the primary population retinotopy template for the cross-animal
registration project. It encodes Figure 9C (altitude) and Figure 9D (azimuth)
from Zhuang et al. (2017), *An extended retinotopic map of mouse cortex*.

## Why this reference

The paper pooled maps across mice after centering each map on the V1 centroid,
rotating it to the major azimuth-gradient axis, and correcting retinal position
at the V1/LM/RL junction. It therefore captures the shared V1-and-HVA map
geometry that we want as a population prior. The repeated distortions and map
reversals across HVAs are retained as useful registration structure; they are
not split into unrelated area-specific maps.

## Source and evidence type

- Article: {ARTICLE_URL}
- Official eLife IIIF TIFF: {FIGURE_URL}
- TIFF SHA-256: `{source['sha256']}`
- Source dimensions: {source['width_px']} × {source['height_px']} pixels

Figure 9C/D publish **5-degree isolines**, not continuous numerical map
rasters: altitude −25° to 30° and azimuth 0° to 90°. This extraction stores
the labeled contour evidence sparsely and does not interpolate unobserved
pixels. The black mean field-sign borders are stored as a separate mask.

## Coordinate frame

Panel D is the common source-figure frame. Panel C is translated onto it using
the duplicated borders (dx={manifest['registration']['panel_C_to_D_translation_px'][0]},
dy={manifest['registration']['panel_C_to_D_translation_px'][1]} pixels).
Coordinates are available as pixels and panel fractions; image y points down,
while `y_common_fraction_up` in the point table points up.

This is still a publication-figure frame—not CCF AP/ML and not the historical
Han/Bonin `retino_map.png` pixel frame. The previously retained MATLAB affine
must not be applied blindly to this v2 eLife image; its landmarks should be
re-fitted if the Han common map is recovered.

## Files

- `source_figure9_v2.tif`: exact lossless source used.
- `source_figure9_panels_CD.png`: cropped source evidence.
- `retinotopy_contour_grid.npz`: sparse altitude/azimuth degree grids,
  color-decoding errors, borders, coordinates, and base colors.
- `retinotopy_contour_points.csv.gz`: one row per decoded contour pixel.
- `Figure_template_extraction_QA.png`: source-to-extraction visual audit.
- `run_manifest.json`: provenance, parameters, counts, QA, and chart contract.

## Current limitations

1. Raster anti-aliasing is decoded to the nearest exact publication stroke
   color; the retained per-pixel color error makes that choice auditable.
2. Area names and landmark identities are not inferred from pixels.
3. No continuous retinal-coordinate surface is created yet. That interpolation
   belongs in the registration model and should be validated against cells with
   measured RFs.
4. No Han or Allen CCF transform has been fitted at this checkpoint.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=Path, required=True, help="Official eLife Figure 9 v2 TIFF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figure_path = args.figure.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_image = Image.open(figure_path).convert("RGB")
    if source_image.size != (1001, 1452):
        raise RuntimeError(f"Expected official Figure 9 size (1001, 1452), got {source_image.size}")
    rgb = np.asarray(source_image)
    crops = {name: panel_crop(rgb, name) for name in PANEL_BOUNDS}
    base_colors = {
        name: jet_base_colors(crop, len(EXPECTED_LEVELS[name])) for name, crop in crops.items()
    }
    decoded = {
        name: decode_contours(crops[name], name, base_colors[name], EXPECTED_LEVELS[name])
        for name in crops
    }
    boundaries = {name: boundary_mask(crops[name], name) for name in crops}
    dx, dy, boundary_qa = estimate_panel_translation(boundaries["altitude"], boundaries["azimuth"])

    altitude = translate_array(decoded["altitude"][0], dx, dy, np.nan)
    altitude_error = translate_array(decoded["altitude"][1], dx, dy, np.nan)
    azimuth = decoded["azimuth"][0]
    azimuth_error = decoded["azimuth"][1]
    common_boundary = boundaries["azimuth"]
    height, width = azimuth.shape
    rows, columns = np.indices((height, width))
    x_fraction = columns / (width - 1)
    y_fraction_up = (height - 1 - rows) / (height - 1)

    source_copy = output / "source_figure9_v2.tif"
    shutil.copy2(figure_path, source_copy)
    source_crop = output / "source_figure9_panels_CD.png"
    source_image.crop(SOURCE_CROP).save(source_crop)
    np.savez_compressed(
        output / "retinotopy_contour_grid.npz",
        altitude_deg=altitude,
        azimuth_deg=azimuth,
        altitude_nearest_base_color_error=altitude_error,
        azimuth_nearest_base_color_error=azimuth_error,
        mean_field_sign_boundary=common_boundary,
        panel_C_boundary_registered=translate_array(boundaries["altitude"], dx, dy, False),
        panel_D_boundary=boundaries["azimuth"],
        x_common_px=columns.astype(np.int16),
        y_common_px=rows.astype(np.int16),
        x_common_fraction=x_fraction.astype(np.float32),
        y_common_fraction_up=y_fraction_up.astype(np.float32),
        altitude_base_rgb=base_colors["altitude"],
        altitude_levels_deg=EXPECTED_LEVELS["altitude"],
        azimuth_base_rgb=base_colors["azimuth"],
        azimuth_levels_deg=EXPECTED_LEVELS["azimuth"],
        panel_C_to_D_translation_px=np.asarray([dx, dy], dtype=np.int16),
    )
    point_count = write_points(
        output / "retinotopy_contour_points.csv.gz",
        {"altitude": altitude, "azimuth": azimuth},
        {"altitude": altitude_error, "azimuth": azimuth_error},
        {"altitude": (dx, dy), "azimuth": (0, 0)},
    )
    render_qa(
        source_crop,
        altitude,
        azimuth,
        boundaries["altitude"],
        boundaries["azimuth"],
        dx,
        dy,
        output / "Figure_template_extraction_QA.png",
    )

    manifest = {
        "checkpoint": "zhuang2017_figure9_initial_evidence",
        "status": "primary source-derived template; visual approval required before downstream warps",
        "source": {
            "article": ARTICLE_URL,
            "figure_url": FIGURE_URL,
            "figure": 9,
            "panels": {"C": "altitude", "D": "azimuth"},
            "sha256": sha256(figure_path),
            "width_px": source_image.width,
            "height_px": source_image.height,
            "license": "CC BY 4.0",
        },
        "parameters": {
            "panel_bounds_source_px": PANEL_BOUNDS,
            "source_crop_px": SOURCE_CROP,
            "contour_interval_deg": 5,
            "contour_color_error_threshold": 0.08,
            "contour_min_rgb_range": 70,
            "base_color_rule": "frequent exact source colors within 0.015 RGB distance of matplotlib jet",
            "boundary_rule": "max RGB <= 35; connected components >= 3 pixels; printed legends masked",
            "common_frame": "panel D crop",
            "coordinate_interpolation": "none",
        },
        "registration": {
            "panel_C_to_D_translation_px": [dx, dy],
            "method": "integer translation minimizing mean distance between duplicated black borders",
            **boundary_qa,
        },
        "counts": {
            "altitude_contour_pixels": int(np.isfinite(altitude).sum()),
            "azimuth_contour_pixels": int(np.isfinite(azimuth).sum()),
            "contour_point_rows": point_count,
            "panel_C_boundary_pixels": int(boundaries["altitude"].sum()),
            "panel_D_boundary_pixels": int(boundaries["azimuth"].sum()),
        },
        "qa": {
            "altitude_levels_deg": EXPECTED_LEVELS["altitude"].tolist(),
            "azimuth_levels_deg": EXPECTED_LEVELS["azimuth"].tolist(),
            "median_altitude_color_error": float(np.nanmedian(altitude_error)),
            "median_azimuth_color_error": float(np.nanmedian(azimuth_error)),
        },
        "chart_contract": {
            "question": "Does the extraction preserve Figure 9C/D degree-labelled contours and their shared border geometry?",
            "takeaway": "Published evidence and decoded layers must agree before interpolation or Han/CCF registration.",
            "family": "spatial image and contour QA",
            "renderer": "static Matplotlib",
            "palette": "published jet scale; black borders; cyan/magenta border audit",
            "output": "Figure_template_extraction_QA.png",
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, manifest)
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "registration": manifest["registration"]}, indent=2))


if __name__ == "__main__":
    main()
