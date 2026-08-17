#!/usr/bin/env python3
"""Extract an auditable retinotopy template from Garrett et al. 2014 Figure 5.

The PDF contains the field-sign surface as a raster and the azimuth/altitude
contours and area boundaries as vector paths.  This script preserves those
different evidence types and expresses all panels in one V1-centred,
panel-width-normalized coordinate frame.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


DEFAULT_OUTPUT = Path("artifacts/retinotopy_template/garrett2014_figure5")
SOURCE_URL = "https://doi.org/10.1523/JNEUROSCI.1124-14.2014"
DOWNLOAD_URL = "https://cseweb.ucsd.edu/~gary/cs200/f14/Mouse%20Maps%20Published.pdf"
PDF_PAGE = 8
ARTICLE_PAGE = 12594
RASTER_SIZE = 435


@dataclass(frozen=True)
class Panel:
    name: str
    left: float
    top: float
    width: float
    height: float
    value_min: float | None = None
    value_max: float | None = None

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


# Locations are taken from the vector PDF in page points.  The three panels
# share size and vertical placement; their horizontal offsets differ slightly.
PANELS = {
    "sign": Panel("sign", 214.914, 435.084, 104.772, 104.772, -1.0, 1.0),
    "azimuth": Panel("azimuth", 324.534, 435.084, 104.772, 104.772, -60.0, 60.0),
    # The altitude content is offset 108.785 pt from azimuth. This was refined
    # from the duplicated vector area boundaries (median cross-panel mismatch
    # 0.0015 panel widths before refinement), rather than from the wider frame.
    "altitude": Panel("altitude", 433.319, 435.084, 104.772, 104.772, -40.0, 40.0),
}

PATH_RE = re.compile(r"([ML])\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)")
RGB_RE = re.compile(r"stroke:rgb\(([^)]*)\)")
WIDTH_RE = re.compile(r"stroke-width:([^;]+)")
MATRIX_RE = re.compile(r"matrix\(([^)]*)\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_pdf_assets(pdf: Path, temp_dir: Path) -> tuple[Path, Path, Path]:
    svg = temp_dir / "page8.svg"
    run(["pdftocairo", "-f", str(PDF_PAGE), "-l", str(PDF_PAGE), "-svg", str(pdf), str(svg)])

    image_prefix = temp_dir / "embedded"
    run(["pdfimages", "-f", str(PDF_PAGE), "-l", str(PDF_PAGE), "-png", str(pdf), str(image_prefix)])
    embedded = sorted(temp_dir.glob("embedded-*.png"))
    sign_candidates = [p for p in embedded if Image.open(p).size == (RASTER_SIZE, RASTER_SIZE)]
    if not sign_candidates:
        raise RuntimeError("Could not find the 435x435 field-sign raster on PDF page 8")

    page_prefix = temp_dir / "page"
    run(["pdftoppm", "-f", str(PDF_PAGE), "-l", str(PDF_PAGE), "-png", "-r", "600", str(pdf), str(page_prefix)])
    pages = sorted(temp_dir.glob("page-*.png"))
    if len(pages) != 1:
        raise RuntimeError(f"Expected one rendered page, found {len(pages)}")
    return svg, sign_candidates[0], pages[0]


def nearest_jet_values(rgb: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    lut = np.asarray(colormaps["jet"](np.linspace(0, 1, 256)))[:, :3]
    flat = rgb.reshape(-1, 3).astype(float) / 255.0
    # Chunk to avoid materializing a 189k x 256 x 3 array.
    indices = np.empty(len(flat), dtype=np.uint8)
    distances = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), 4096):
        block = flat[start : start + 4096]
        squared = ((block[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        nearest = squared.argmin(axis=1)
        indices[start : start + len(block)] = nearest
        distances[start : start + len(block)] = np.sqrt(squared[np.arange(len(block)), nearest])
    values = low + (high - low) * indices.astype(float) / 255.0
    shape = rgb.shape[:2]
    return values.reshape(shape), distances.reshape(shape)


def field_sign_and_v1(sign_image: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    rgb = np.asarray(Image.open(sign_image).convert("RGB"))
    values, distances = nearest_jet_values(rgb, -1.0, 1.0)
    candidate = values < -0.5
    labels, count = ndimage.label(candidate)
    if count == 0:
        raise RuntimeError("No negative field-sign component was detected")
    sizes = ndimage.sum(candidate, labels, index=np.arange(1, count + 1))
    v1_label = int(np.argmax(sizes)) + 1
    v1_mask = labels == v1_label
    row, col = ndimage.center_of_mass(v1_mask)
    return values, distances, v1_mask, (float(row), float(col))


def transform_point(x: float, y: float, matrix: tuple[float, ...]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def path_segments(d: str) -> list[tuple[float, float, float, float]]:
    segments: list[tuple[float, float, float, float]] = []
    current: tuple[float, float] | None = None
    for command, raw_x, raw_y in PATH_RE.findall(d):
        point = (float(raw_x), float(raw_y))
        if command == "M":
            current = point
        elif current is not None:
            segments.append((*current, *point))
            current = point
    return segments


def parse_rgb(style: str) -> tuple[float, float, float] | None:
    match = RGB_RE.search(style)
    if not match:
        return None
    return tuple(float(item.strip().rstrip("%")) / 100.0 for item in match.group(1).split(","))


def locate_panel(x: float, y: float, margin: float = 0.0) -> Panel | None:
    for panel in (PANELS["azimuth"], PANELS["altitude"]):
        if panel.left + margin <= x <= panel.right - margin and panel.top + margin <= y <= panel.bottom - margin:
            return panel
    return None


def canonical_xy(x: float, y: float, panel: Panel, v1_fraction: tuple[float, float]) -> tuple[float, float]:
    row_fraction, col_fraction = v1_fraction
    return (x - panel.left) / panel.width - col_fraction, -((y - panel.top) / panel.height - row_fraction)


def jet_position(rgb: tuple[float, float, float]) -> tuple[int, float]:
    lut = np.asarray(colormaps["jet"](np.linspace(0, 1, 256)))[:, :3]
    distances = ((lut - np.asarray(rgb)[None, :]) ** 2).sum(axis=1)
    index = int(np.argmin(distances))
    return index, float(np.sqrt(distances[index]))


def build_contour_color_maps(svg: Path) -> dict[str, dict[tuple[float, float, float], float]]:
    colors: dict[str, set[tuple[float, float, float]]] = {"azimuth": set(), "altitude": set()}
    namespace = "{http://www.w3.org/2000/svg}"
    for _, element in ET.iterparse(svg, events=("end",)):
        if element.tag != f"{namespace}path":
            continue
        style = element.attrib.get("style", "")
        transform = element.attrib.get("transform", "")
        matrix_match = MATRIX_RE.fullmatch(transform)
        width_match = WIDTH_RE.search(style)
        rgb = parse_rgb(style)
        segments = path_segments(element.attrib.get("d", ""))
        if (
            not matrix_match
            or not width_match
            or rgb is None
            or not segments
            or abs(float(width_match.group(1)) - 0.991) >= 0.02
            or max(rgb) - min(rgb) <= 0.05
        ):
            element.clear()
            continue
        matrix = tuple(float(item) for item in matrix_match.group(1).split(","))
        x0, y0, x1, y1 = segments[0]
        p0, p1 = transform_point(x0, y0, matrix), transform_point(x1, y1, matrix)
        panel = locate_panel((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        if panel is not None:
            colors[panel.name].add(rgb)
        element.clear()

    result: dict[str, dict[tuple[float, float, float], float]] = {}
    for panel_name, unique_colors in colors.items():
        panel = PANELS[panel_name]
        ordered = sorted(unique_colors, key=lambda color: jet_position(color)[0])
        raw_endpoints = [
            panel.value_min + (panel.value_max - panel.value_min) * jet_position(color)[0] / 255.0
            for color in (ordered[0], ordered[-1])
        ]
        start, end = (5.0 * round(value / 5.0) for value in raw_endpoints)
        levels = np.arange(start, end + 0.1, 5.0)
        if len(levels) != len(ordered):
            raise RuntimeError(
                f"Could not reconcile {len(ordered)} {panel_name} stroke colors with 5-degree levels {start:g}..{end:g}"
            )
        result[panel_name] = dict(zip(ordered, levels.tolist()))
    return result


def extract_vector_layers(
    svg: Path,
    v1_fraction: tuple[float, float],
    color_maps: dict[str, dict[tuple[float, float, float], float]],
) -> tuple[list[dict], list[dict]]:
    contours: list[dict] = []
    boundaries: list[dict] = []
    namespace = "{http://www.w3.org/2000/svg}"
    for _, element in ET.iterparse(svg, events=("end",)):
        if element.tag != f"{namespace}path":
            continue
        style = element.attrib.get("style", "")
        transform = element.attrib.get("transform", "")
        matrix_match = MATRIX_RE.fullmatch(transform)
        width_match = WIDTH_RE.search(style)
        rgb = parse_rgb(style)
        if not matrix_match or not width_match or rgb is None:
            element.clear()
            continue
        matrix = tuple(float(item) for item in matrix_match.group(1).split(","))
        stroke_width = float(width_match.group(1))
        for x0, y0, x1, y1 in path_segments(element.attrib.get("d", "")):
            p0 = transform_point(x0, y0, matrix)
            p1 = transform_point(x1, y1, matrix)
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-8:
                continue
            midpoint = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            panel = locate_panel(*midpoint)
            if panel is None:
                continue
            cx0, cy0 = canonical_xy(*p0, panel, v1_fraction)
            cx1, cy1 = canonical_xy(*p1, panel, v1_fraction)
            if abs(stroke_width - 0.991) < 0.02 and max(rgb) - min(rgb) > 0.05:
                value = color_maps[panel.name][rgb]
                _, color_error = jet_position(rgb)
                contours.append(
                    {
                        "map": panel.name,
                        "value_deg": value,
                        "x0": cx0,
                        "y0": cy0,
                        "x1": cx1,
                        "y1": cy1,
                        "source_r": rgb[0],
                        "source_g": rgb[1],
                        "source_b": rgb[2],
                        "jet_color_error": color_error,
                    }
                )
            elif abs(stroke_width - 0.495) < 0.02 and max(rgb) < 0.05:
                # Exclude the rectangular panel frame but retain internal area borders.
                if locate_panel(*p0, margin=0.8) == panel and locate_panel(*p1, margin=0.8) == panel:
                    boundaries.append(
                        {"source_panel": panel.name, "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1}
                    )
        element.clear()
    return deduplicate(contours), deduplicate(boundaries)


def deduplicate(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for row in rows:
        key = tuple((name, round(value, 6) if isinstance(value, float) else value) for name, value in row.items())
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def write_csv_gz(rows: list[dict], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_source_figure(page_image: Path, output: Path) -> None:
    scale = 600.0 / 72.0
    crop_points = (208.0, 410.0, 542.5, 558.0)
    box = tuple(round(value * scale) for value in crop_points)
    Image.open(page_image).crop(box).save(output)


def render_qa(
    source_crop: Path,
    sign_values: np.ndarray,
    v1_mask: np.ndarray,
    v1_fraction: tuple[float, float],
    contours: list[dict],
    boundaries: list[dict],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.9, 1.1))
    source_ax = figure.add_subplot(grid[0, :])
    source_ax.imshow(Image.open(source_crop))
    source_ax.set_title("Published Figure 5 panels (source evidence)", loc="left")
    source_ax.axis("off")

    row_fraction, col_fraction = v1_fraction
    extent = (-col_fraction, 1 - col_fraction, -(1 - row_fraction), row_fraction)
    sign_ax = figure.add_subplot(grid[1, 0])
    sign_plot = sign_ax.imshow(sign_values, cmap="jet", vmin=-1, vmax=1, extent=extent, origin="upper")
    sign_ax.contour(v1_mask.astype(float), levels=[0.5], colors="black", linewidths=1.2, extent=extent, origin="upper")
    sign_ax.scatter([0], [0], marker="+", color="white", linewidth=1.4, s=95, zorder=5)
    sign_ax.set_title("Digitized visual-field sign")
    sign_ax.set_xlabel("source-panel width from V1 centroid")
    sign_ax.set_ylabel("source-panel width from V1 centroid (up +)")
    sign_ax.set_aspect("equal")
    figure.colorbar(sign_plot, ax=sign_ax, fraction=0.046, label="field sign")

    for column, map_name, limits in ((1, "azimuth", (-60, 60)), (2, "altitude", (-40, 40))):
        ax = figure.add_subplot(grid[1, column])
        selected = [row for row in contours if row["map"] == map_name]
        contour_lines = [[(row["x0"], row["y0"]), (row["x1"], row["y1"])] for row in selected]
        contour_colors = [
            colormaps["jet"]((row["value_deg"] - limits[0]) / (limits[1] - limits[0])) for row in selected
        ]
        ax.add_collection(LineCollection(contour_lines, colors=contour_colors, linewidths=0.75))
        border_rows = [row for row in boundaries if row["source_panel"] == map_name]
        border_lines = [[(row["x0"], row["y0"]), (row["x1"], row["y1"])] for row in border_rows]
        ax.add_collection(LineCollection(border_lines, colors="black", linewidths=0.45))
        ax.autoscale_view()
        ax.scatter([0], [0], marker="+", color="black", s=55, zorder=5)
        ax.set_title(f"Extracted {map_name} contours")
        ax.set_xlabel("source-panel width from V1 centroid")
        ax.set_ylabel("source-panel width from V1 centroid (up +)")
        ax.set_aspect("equal")
        scalar = plt.cm.ScalarMappable(norm=plt.Normalize(*limits), cmap="jet")
        figure.colorbar(scalar, ax=ax, fraction=0.046, label="degrees")
    figure.suptitle("Garrett et al. 2014 Figure 5 retinotopy-template extraction", fontsize=15)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_readme(output: Path, manifest: dict) -> None:
    text = f"""# Garrett et al. 2014 Figure 5 template — initial evidence checkpoint

This directory is a source-derived, exploratory template from Figure 5 of
Garrett et al. (2014), *Topography and Areal Organization of Mouse Visual
Cortex*. It is **not** the unpublished Allen 35-experiment canonical atlas.

## Source and extraction

- Article: {SOURCE_URL}
- PDF used: {DOWNLOAD_URL}
- PDF SHA-256: `{manifest['source']['sha256']}`
- Source location: PDF page {PDF_PAGE}, article page {ARTICLE_PAGE}, Figure 5
- Published population: 14 mice; maps were aligned on V1 center of mass and
  the mean V1 horizontal-retinotopy-gradient direction.

The field-sign surface is decoded from the embedded 435×435 raster using the
published `jet` color scale. Azimuth and altitude isolines and internal black
boundaries are extracted from vector paths, not re-traced by hand. Stroke
colors are decoded against `jet` and rounded to the published 5-degree contour
spacing.

## Coordinate frame

The origin is the center of mass of the largest connected component with
decoded field sign below -0.5, provisionally identified as V1. One unit equals
one source-panel width. Positive x points right in the published figure and
positive y points up. No additional rotation is applied because the published
population map was already rotation-normalized.

This is a figure coordinate frame, not yet AP/ML CCF space. The anatomical
axis orientation and absolute millimetre scale remain to be established from
the Allen/MouseV2 observations.

## Files

- `source_figure5_panels.png`: cropped primary evidence.
- `field_sign_grid.npz`: decoded sign field, color-decoding error, V1 mask,
  and normalized grid coordinates.
- `retinotopy_contours.csv.gz`: vector-derived azimuth/altitude line segments.
- `area_boundaries.csv.gz`: black internal boundary segments from both maps.
- `Figure_template_extraction_QA.png`: source-to-extraction visual audit.
- `run_manifest.json`: source hashes, parameters, counts, and chart contract.

## Current limitations

1. The V1 mask and centroid are algorithmic provisional choices and require
   visual approval.
2. Area names/centroids have not yet been manually transcribed; this avoids
   silently treating typography as anatomical data.
3. Contour values inherit raster/vector publication resolution and should be
   used as an initialization prior, not as ground truth.
4. Redistribution should preserve the article citation and source rights.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def boundary_agreement(boundaries: list[dict]) -> dict[str, float]:
    points = {}
    for panel in ("azimuth", "altitude"):
        selected = [row for row in boundaries if row["source_panel"] == panel]
        points[panel] = np.asarray([[row["x0"], row["y0"]] for row in selected], dtype=float)
    distances = np.concatenate(
        [
            cKDTree(points["altitude"]).query(points["azimuth"], k=1)[0],
            cKDTree(points["azimuth"]).query(points["altitude"], k=1)[0],
        ]
    )
    return {
        "cross_panel_boundary_median_distance_panel_width": float(np.median(distances)),
        "cross_panel_boundary_p99_distance_panel_width": float(np.quantile(distances, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True, help="Local Garrett et al. 2014 PDF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="garrett2014_template_") as raw_temp:
        temp = Path(raw_temp)
        svg, sign_image, page_image = extract_pdf_assets(pdf, temp)
        sign_values, sign_error, v1_mask, (v1_row, v1_col) = field_sign_and_v1(sign_image)
        v1_fraction = ((v1_row + 0.5) / RASTER_SIZE, (v1_col + 0.5) / RASTER_SIZE)
        color_maps = build_contour_color_maps(svg)
        contours, boundaries = extract_vector_layers(svg, v1_fraction, color_maps)

        source_crop = output / "source_figure5_panels.png"
        crop_source_figure(page_image, source_crop)
        rows, columns = np.indices(sign_values.shape)
        x = (columns + 0.5) / RASTER_SIZE - v1_fraction[1]
        y = -((rows + 0.5) / RASTER_SIZE - v1_fraction[0])
        np.savez_compressed(
            output / "field_sign_grid.npz",
            field_sign=sign_values.astype(np.float32),
            jet_color_error=sign_error.astype(np.float32),
            v1_mask=v1_mask,
            x_panel_width=x.astype(np.float32),
            y_panel_width=y.astype(np.float32),
        )
        write_csv_gz(contours, output / "retinotopy_contours.csv.gz")
        write_csv_gz(boundaries, output / "area_boundaries.csv.gz")
        render_qa(
            source_crop,
            sign_values,
            v1_mask,
            v1_fraction,
            contours,
            boundaries,
            output / "Figure_template_extraction_QA.png",
        )

    counts = {
        "azimuth_segments": sum(row["map"] == "azimuth" for row in contours),
        "altitude_segments": sum(row["map"] == "altitude" for row in contours),
        "azimuth_boundary_segments": sum(row["source_panel"] == "azimuth" for row in boundaries),
        "altitude_boundary_segments": sum(row["source_panel"] == "altitude" for row in boundaries),
        "v1_mask_pixels": int(v1_mask.sum()),
    }
    contour_levels = {
        name: sorted({row["value_deg"] for row in contours if row["map"] == name})
        for name in ("azimuth", "altitude")
    }
    manifest = {
        "checkpoint": "garrett2014_figure5_initial_evidence",
        "status": "exploratory source-derived template; visual approval required",
        "source": {
            "article": SOURCE_URL,
            "pdf_url": DOWNLOAD_URL,
            "sha256": sha256(pdf),
            "pdf_page": PDF_PAGE,
            "article_page": ARTICLE_PAGE,
            "figure": 5,
        },
        "parameters": {
            "panels_pdf_points": {name: panel.__dict__ for name, panel in PANELS.items()},
            "field_sign_v1_threshold": -0.5,
            "v1_component_rule": "largest_connected_component",
            "v1_centroid_raster_row_col": [v1_row, v1_col],
            "v1_centroid_panel_fraction_row_col": list(v1_fraction),
            "contour_interval_deg": 5,
            "contour_color_rule": "order unique non-grey strokes by nearest jet position; anchor rounded endpoints and assign 5-degree levels",
            "coordinate_unit": "source panel width",
            "positive_y": "up in published figure",
        },
        "counts": counts,
        "qa": {
            **boundary_agreement(boundaries),
            "median_field_sign_jet_color_error": float(np.median(sign_error)),
            "contour_levels_deg": contour_levels,
        },
        "chart_contract": {
            "question": "Does the machine extraction preserve the published sign field, retinal-coordinate contours, and area boundaries?",
            "takeaway": "Source and extracted layers must agree visibly before area labels or animal warps are added.",
            "family": "spatial image and contour QA",
            "renderer": "static Matplotlib",
            "palette": "published jet scale; black boundaries and centroid marker",
            "output": "Figure_template_extraction_QA.png",
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, manifest)
    print(json.dumps({"output": str(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
