#!/usr/bin/env python3
"""QA: draw the isolated V1 border ring directly on each animal's own reference photograph.

`register_mousev2_area_borders_to_zhuang.py` isolates a "V1-only" border ring per animal (nearest
border material to the probe centroid in every direction) before registering it to the Zhuang
atlas -- but that isolation, and the photo-to-mm calibration it depends on, were never checked
directly against the source image. This script checks both, in the ONE space that is ground truth:
each animal's own original photo pixel coordinates (no Zhuang transform, no rotation fit involved
-- if this looks wrong here, the registration fit downstream cannot be trusted regardless of how
good its own QA figures look).

Per animal, overlaid on the reference photo:
- all cleaned border pixels (faint, for context)
- the isolated V1 ring (bright, primary evidence in the registration fit)
- the "other" border material (dim orange, the weak rotation anchor)
- probe positions and their centroid
- a physical 1mm scale bar AND a dashed 3mm-diameter reference circle centered on the probe
  centroid, both drawn using ONLY that animal's own px/mm calibration -- if the dashed circle
  looks obviously too big or small next to the traced ring in the photo, the calibration (not the
  ring isolation) is the thing to distrust.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_mousev2_area_borders_to_zhuang import (  # noqa: E402
    ANIMALS, BORDER_LAYER_OVERRIDE, DEFAULT_BORDER_LAYER, GEOMETRY_ZIP, SITE_SEMANTICS,
    filter_small_components, isolate_nearest_ring_border, load_layer_points, read_probe_metadata,
    remove_single_probe_loops,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/reference_photo_QA"

REFERENCE_IMAGE_OVERRIDE = {"813810": "recording_day_reference.jpg"}
DEFAULT_REFERENCE_IMAGE = "authoritative_reference.jpg"
REFERENCE_CIRCLE_DIAMETER_MM = 3.0  # rough mouse V1 diameter at these eccentricities, for scale sanity-check only


def load_reference_image(zf: zipfile.ZipFile, animal: str) -> np.ndarray:
    name = REFERENCE_IMAGE_OVERRIDE.get(animal, DEFAULT_REFERENCE_IMAGE)
    path = f"all_animals_geometry/{animal}/{animal}_{name}"
    return np.asarray(Image.open(io.BytesIO(zf.read(path))).convert("RGB"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(GEOMETRY_ZIP)

    for animal in ANIMALS:
        px_per_mm, probe_labels, probe_px = read_probe_metadata(zf, animal)
        centroid_px = probe_px.mean(axis=0)

        border_layer = BORDER_LAYER_OVERRIDE.get(animal, DEFAULT_BORDER_LAYER)
        _, border_mask = load_layer_points(zf, animal, border_layer)
        cleaned_mask = filter_small_components(border_mask)
        filtered_mask, n_removed = remove_single_probe_loops(cleaned_mask, probe_px, px_per_mm)
        removed_rows, removed_cols = np.nonzero(cleaned_mask & ~filtered_mask)
        removed_px = np.column_stack([removed_cols, removed_rows]).astype(float)
        rows, cols = np.nonzero(filtered_mask)
        cleaned_px = np.column_stack([cols, rows]).astype(float)
        ring_px, other_px, ring_coverage = isolate_nearest_ring_border(cleaned_px, centroid_px, px_per_mm)

        image = load_reference_image(zf, animal)

        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(image)
        ax.scatter(cleaned_px[:, 0], cleaned_px[:, 1], s=1, color="#4da6ff", alpha=0.25, label="all cleaned border px")
        if len(removed_px):
            ax.scatter(removed_px[:, 0], removed_px[:, 1], s=5, color="#39ff14", marker="s", alpha=0.8,
                       label="removed: single-probe-loop artifact")
        if len(other_px):
            ax.scatter(other_px[:, 0], other_px[:, 1], s=4, color="#ff9f1c", alpha=0.6, label="other border (rotation anchor)")
        ax.scatter(ring_px[:, 0], ring_px[:, 1], s=9, color="#ff1744", label="isolated V1 ring (primary)")
        ax.scatter(probe_px[:, 0], probe_px[:, 1], s=110, marker="*", color="#ffee00", edgecolors="black",
                   linewidths=0.8, zorder=5, label="probes")
        for label, (x, y) in zip(probe_labels, probe_px):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=11,
                        color="white", weight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
        ax.scatter(*centroid_px, s=80, marker="x", color="black", zorder=5, label="probe centroid")

        # Physical scale sanity check, drawn from this animal's OWN calibration only.
        circle = plt.Circle(centroid_px, REFERENCE_CIRCLE_DIAMETER_MM / 2 * px_per_mm, fill=False,
                             linestyle="--", linewidth=1.6, color="#00e5ff")
        ax.add_patch(circle)
        bar_x0 = image.shape[1] * 0.05
        bar_y0 = image.shape[0] * 0.95
        ax.plot([bar_x0, bar_x0 + px_per_mm], [bar_y0, bar_y0], color="white", linewidth=3,
                path_effects=[patheffects.withStroke(linewidth=5, foreground="black")])
        ax.text(bar_x0 + px_per_mm / 2, bar_y0 - image.shape[0] * 0.02, "1 mm", color="white", ha="center",
                fontsize=10, weight="bold",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

        ax.set_title(f"{animal} ({SITE_SEMANTICS[animal]})  --  {px_per_mm:.1f} px/mm, "
                     f"ring coverage {ring_coverage:.0%}, {n_removed} artifact loop(s) removed\n"
                     f"dashed cyan circle = {REFERENCE_CIRCLE_DIAMETER_MM}mm reference diameter centered on probe centroid",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        fig.tight_layout()
        out_path = OUTPUT / f"{animal}_v1_isolation_on_reference.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
