#!/usr/bin/env python3
"""Extract each animal's V1-ring border via the skeleton-bridging pipeline developed interactively
against 817334, and save every intermediate step's figure per animal for visual audit.

Why this replaced the angular-nearest-ring approach in register_mousev2_area_borders_to_zhuang.py:
that approach used the RAW border-pixel cloud directly, so gaps in the traced border made the
"nearest border material per direction from the probe centroid" rule grab small disconnected
fragments (window-edge annotation loops, vessel-detection noise) instead of the true V1 boundary.
This pipeline instead reconstructs a topologically coherent skeleton first -- filling small real
gaps by bridging skeleton endpoints that mutually point at each other -- so that a small isolated
fragment (which was never bridged into the main structure) can be excluded entirely before ray-
casting, rather than being the accidental nearest hit in some direction.

Pipeline (verified against 817334 2026-08-17, see step_by_step_817334/ for the interactive
derivation this was consolidated from):
  1. Load the raw `area_borders_relaxed.png` (or animal-specific override) layer; drop connected
     components smaller than `MIN_COMPONENT_PX` (speckle noise).
  2. Gaussian-blur (`BLUR_SIGMA`) the cleaned binary mask and rethreshold at `BINARY_THRESH_FRAC`
     of the blurred max -- this reconnects small pixel-level gaps and smooths jagged edges before
     skeletonizing.
  3. Skeletonize to a 1px-wide curve network.
  4. Find skeleton endpoints (pixels with exactly one skeleton neighbor) and, for each, estimate
     its local outward direction by walking `WALK_STEPS` pixels back along its own branch.
  5. Match endpoint pairs that are close (`MAX_BRIDGE_DIST`) AND mutually point at each other
     (both direction vectors' cosine similarity with the connecting line exceeds `COS_THRESH`) --
     candidate bridges across real gaps.
  6. Keep a candidate bridge only if it does not connect two small, likely-spurious fragments: keep
     when the bridge is no longer than the smaller fragment's own skeleton length, OR when either
     fragment is already substantial (`MAIN_STRUCTURE_PX`). Draw kept bridges into the skeleton.
  7. Re-label the bridged skeleton's connected components; keep only ones at least
     `MAIN_COMPONENT_PX` (drops the leftover small fragments that were correctly never bridged).
  8. Ray-cast from the probe centroid (`N_BINS` angular bins); take the nearest surviving point per
     bin and join in angular order -- the final V1-ring border.
"""

from __future__ import annotations

import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_mousev2_area_borders_to_zhuang import (  # noqa: E402
    ANIMALS, BORDER_LAYER_OVERRIDE, DEFAULT_BORDER_LAYER, GEOMETRY_ZIP, read_probe_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang"
REFERENCE_DIR = ROOT / "reference"

# Animals where a human hand-corrected `reference/<animal>_layers.svg` exists (as of 2026-08-17):
# a vector <path> traced directly over the same raster the automated pipeline struggled with,
# added to the "Area borders - relaxed" layer group. Checked against the raw extraction for both
# and it cleanly wraps all 4 probes with none of the spike/gap artifacts the automated pipeline
# produced for 810531 -- use it directly instead of the skeleton-bridging reconstruction.
HAND_TRACED_ANIMALS = {"810531", "810532"}
SVG_NS = {"svg": "http://www.w3.org/2000/svg", "inkscape": "http://www.inkscape.org/namespaces/inkscape"}

MIN_COMPONENT_PX = 40
BLUR_SIGMA = 5
BINARY_THRESH_FRAC = 0.15
WALK_STEPS = 8
MAX_BRIDGE_DIST = 60
COS_THRESH = 0.75
MAIN_STRUCTURE_PX = 200
MAIN_COMPONENT_PX = 100
N_BINS = 360
UNKINK_TURN_THRESHOLD_DEG = 40
UNKINK_MAX_ITER = 10


def unkink_ring(ring: np.ndarray) -> np.ndarray:
    """Ray-casting occasionally grabs a point from a short nearby spur/side-branch for a couple of
    adjacent angular bins instead of the main curve (found 2026-08-17 in 817335, a small detour
    near probe A that pulled the anterior-apex curvature detector off the true peak). Iteratively
    drop ring points whose turning angle relative to their immediate neighbors exceeds
    `UNKINK_TURN_THRESHOLD_DEG` and rejoin, until stable or `UNKINK_MAX_ITER` is reached. Harmless
    on an already-smooth ring (nothing exceeds the threshold, loop exits on iteration 1)."""

    def turning_angle(points: np.ndarray) -> np.ndarray:
        nxt, prv = np.roll(points, -1, axis=0), np.roll(points, 1, axis=0)
        v1, v2 = points - prv, nxt - points
        v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
        v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
        return np.degrees(np.arccos(np.clip(np.sum(v1n * v2n, axis=1), -1, 1)))

    current = ring.copy()
    for _ in range(UNKINK_MAX_ITER):
        flagged = turning_angle(current) > UNKINK_TURN_THRESHOLD_DEG
        if not flagged.any():
            break
        current = current[~flagged]
    return current


def parse_svg_path(d: str, n_samples_per_curve: int = 8) -> np.ndarray:
    """Minimal SVG path sampler covering the M/L/C (absolute and relative) + Z commands used by
    Inkscape's hand-traced paths here. Cubic Beziers are sampled at `n_samples_per_curve` points."""
    tokens = re.findall(r"[MmLlCcZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    i = 0
    cur = np.array([0.0, 0.0])
    start = None
    points = []
    cmd = None

    def nextnum():
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        if tokens[i] in "MmLlCcZz":
            cmd = tokens[i]
            i += 1
        if cmd in ("M", "m"):
            x, y = nextnum(), nextnum()
            if cmd == "m" and start is not None:
                x, y = x + cur[0], y + cur[1]
            cur = np.array([x, y])
            start = cur.copy()
            points.append(cur.copy())
            cmd = "L" if cmd == "M" else "l"  # subsequent implicit pairs are linetos
        elif cmd in ("L", "l"):
            x, y = nextnum(), nextnum()
            if cmd == "l":
                x, y = x + cur[0], y + cur[1]
            cur = np.array([x, y])
            points.append(cur.copy())
        elif cmd in ("C", "c"):
            x1, y1, x2, y2, x, y = (nextnum() for _ in range(6))
            p0 = cur
            if cmd == "c":
                p1, p2, p3 = cur + [x1, y1], cur + [x2, y2], cur + [x, y]
            else:
                p1, p2, p3 = np.array([x1, y1]), np.array([x2, y2]), np.array([x, y])
            for t in np.linspace(0, 1, n_samples_per_curve)[1:]:
                points.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
            cur = p3
        elif cmd in ("Z", "z"):
            if start is not None:
                points.append(start.copy())
            i += 1
        else:
            i += 1
    return np.array(points)


def load_hand_traced_ring(animal: str) -> np.ndarray | None:
    """Returns the hand-traced V1 border path from reference/<animal>_layers.svg's
    "Area borders - relaxed" layer, or None if no such file exists for this animal."""
    svg_path = REFERENCE_DIR / f"{animal}_layers.svg"
    if not svg_path.exists():
        return None
    root = ET.fromstring(svg_path.read_text(encoding="utf-8", errors="replace"))
    for group in root.findall("svg:g", SVG_NS):
        if group.get("{http://www.inkscape.org/namespaces/inkscape}label") != "Area borders - relaxed":
            continue
        for child in group:
            if child.tag.endswith("path"):
                return parse_svg_path(child.get("d"))
    return None


def walk_direction(skeleton_coords: set, start: np.ndarray, steps: int) -> np.ndarray | None:
    def neighbors(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if not (dx == 0 and dy == 0) and (x + dx, y + dy) in skeleton_coords]

    prev, current = None, tuple(start.astype(int))
    path = [current]
    for _ in range(steps):
        candidates = [p for p in neighbors(current) if p != prev]
        if len(candidates) != 1:
            break
        prev, current = current, candidates[0]
        path.append(current)
    far, near = np.array(path[-1], float), np.array(path[0], float)
    direction = near - far
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


def extract_v1_ring(zf: zipfile.ZipFile, animal: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    px_per_mm, probe_labels, probe_px = read_probe_metadata(zf, animal)
    centroid = probe_px.mean(axis=0)

    hand_ring = load_hand_traced_ring(animal) if animal in HAND_TRACED_ANIMALS else None
    if hand_ring is not None:
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(np.append(hand_ring[:, 0], hand_ring[0, 0]), np.append(hand_ring[:, 1], hand_ring[0, 1]),
                "-", color="green", linewidth=2)
        ax.scatter(*centroid, s=100, marker="x", color="black", zorder=5)
        ax.scatter(probe_px[:, 0], probe_px[:, 1], s=60, marker="*", color="blue", zorder=5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{animal}: hand-traced V1 ring (reference/{animal}_layers.svg)")
        fig.savefig(out_dir / "step6_final_ring.png", dpi=130)
        plt.close(fig)
        return {
            "animal": animal, "ring_px": hand_ring, "centroid_px": centroid,
            "probe_px": probe_px, "probe_labels": probe_labels, "px_per_mm": px_per_mm,
            "ring_coverage": 1.0, "source": "hand_traced",
        }

    layer_name = BORDER_LAYER_OVERRIDE.get(animal, DEFAULT_BORDER_LAYER)
    raw = zf.read(f"all_animals_geometry/{animal}/layers/{layer_name}")
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
    mask = arr[..., 3] > 0

    def scatter_fig(title):
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(*centroid, s=100, marker="x", color="black", zorder=5)
        ax.scatter(probe_px[:, 0], probe_px[:, 1], s=60, marker="*", color="blue", zorder=5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{animal}: {title}")
        return fig, ax

    # Step 1: raw + small-component removal
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    keep_ids = np.nonzero(sizes >= MIN_COMPONENT_PX)[0] + 1
    cleaned = np.isin(labeled, keep_ids)
    rows, cols = np.nonzero(cleaned)
    fig, ax = scatter_fig(f"step1: raw ({mask.sum()}px) -> small-removed ({cleaned.sum()}px, <{MIN_COMPONENT_PX}px dropped)")
    ax.scatter(cols, rows, s=1, color="#2864a8")
    fig.savefig(out_dir / "step1_cleaned.png", dpi=130)
    plt.close(fig)

    # Step 2: blur + rethreshold + skeletonize
    blurred = ndimage.gaussian_filter(cleaned.astype(float), sigma=BLUR_SIGMA)
    binary = blurred > (BINARY_THRESH_FRAC * blurred.max())
    skeleton = skeletonize(binary)
    sk_rows, sk_cols = np.nonzero(skeleton)
    fig, ax = scatter_fig(f"step2: gaussian(sigma={BLUR_SIGMA}) + threshold + skeletonize ({skeleton.sum()}px)")
    ax.scatter(sk_cols, sk_rows, s=2, color="#2864a8")
    fig.savefig(out_dir / "step2_skeleton.png", dpi=130)
    plt.close(fig)

    # Step 3: endpoints + directions
    neighbor_count = ndimage.convolve(skeleton.astype(int), np.ones((3, 3)), mode="constant") - skeleton.astype(int)
    endpoints_mask = skeleton & (neighbor_count == 1)
    ep_rows, ep_cols = np.nonzero(endpoints_mask)
    endpoints = np.column_stack([ep_cols, ep_rows]).astype(float)
    n_ep = len(endpoints)

    sk_labeled, n_comp = ndimage.label(skeleton, structure=np.ones((3, 3)))
    comp_sizes = ndimage.sum(skeleton, sk_labeled, index=np.arange(1, n_comp + 1))
    endpoint_comp = sk_labeled[ep_rows, ep_cols]

    sk_coords = set(map(tuple, np.column_stack(np.nonzero(skeleton))[:, ::-1]))
    directions = [walk_direction(sk_coords, e, WALK_STEPS) for e in endpoints]

    fig, ax = scatter_fig(f"step3: {n_ep} skeleton endpoints")
    ax.scatter(sk_cols, sk_rows, s=2, color="#2864a8")
    ax.scatter(endpoints[:, 0], endpoints[:, 1], s=25, color="red")
    fig.savefig(out_dir / "step3_endpoints.png", dpi=130)
    plt.close(fig)

    # Step 4: mutual-pointing candidate pairs, kept vs dropped by fragment-size rule
    pairs = []
    used = set()
    for i in range(n_ep):
        if directions[i] is None or i in used:
            continue
        best_j, best_score = None, -2.0
        for j in range(n_ep):
            if i == j or directions[j] is None or j in used:
                continue
            d_ij = endpoints[j] - endpoints[i]
            dist = np.linalg.norm(d_ij)
            if dist > MAX_BRIDGE_DIST or dist < 1e-6:
                continue
            d_ij_n = d_ij / dist
            cos_i = np.dot(directions[i], d_ij_n)
            cos_j = np.dot(directions[j], -d_ij_n)
            if cos_i > COS_THRESH and cos_j > COS_THRESH:
                score = cos_i + cos_j
                if score > best_score:
                    best_score, best_j = score, j
        if best_j is not None:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)

    kept, dropped = [], []
    for i, j in pairs:
        bridge_len = np.linalg.norm(endpoints[i] - endpoints[j])
        size_i, size_j = comp_sizes[endpoint_comp[i] - 1], comp_sizes[endpoint_comp[j] - 1]
        smaller, larger = min(size_i, size_j), max(size_i, size_j)
        (kept if (bridge_len <= smaller or larger >= MAIN_STRUCTURE_PX) else dropped).append((i, j))

    fig, ax = scatter_fig(f"step4: {len(kept)} bridges kept (green), {len(dropped)} dropped (magenta)")
    ax.scatter(sk_cols, sk_rows, s=2, color="#2864a8")
    ax.scatter(endpoints[:, 0], endpoints[:, 1], s=20, color="red")
    for i, j in kept:
        ax.plot([endpoints[i, 0], endpoints[j, 0]], [endpoints[i, 1], endpoints[j, 1]], "-", color="lime", linewidth=2)
    for i, j in dropped:
        ax.plot([endpoints[i, 0], endpoints[j, 0]], [endpoints[i, 1], endpoints[j, 1]], "--", color="magenta", linewidth=1.5)
    fig.savefig(out_dir / "step4_bridges.png", dpi=130)
    plt.close(fig)

    # Step 5: merge bridges, keep only main components
    bridged = skeleton.copy()
    for i, j in kept:
        x0, y0 = endpoints[i].astype(int)
        x1, y1 = endpoints[j].astype(int)
        rr, cc = draw_line(y0, x0, y1, x1)
        bridged[rr, cc] = True
    bridged_labeled, n_bridged_comp = ndimage.label(bridged, structure=np.ones((3, 3)))
    bridged_sizes = ndimage.sum(bridged, bridged_labeled, index=np.arange(1, n_bridged_comp + 1))
    main_ids = np.nonzero(bridged_sizes >= MAIN_COMPONENT_PX)[0] + 1
    main_mask = np.isin(bridged_labeled, main_ids)
    main_rows, main_cols = np.nonzero(main_mask)
    fig, ax = scatter_fig(f"step5: bridged + main components only ({main_mask.sum()}px of {bridged.sum()}px bridged)")
    ax.scatter(main_cols, main_rows, s=2, color="#2864a8")
    fig.savefig(out_dir / "step5_main_components.png", dpi=130)
    plt.close(fig)

    # Step 6: ray-cast from probe centroid
    pts = np.column_stack([main_cols, main_rows]).astype(float)
    rel = pts - centroid
    dist = np.hypot(rel[:, 0], rel[:, 1])
    angle = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
    bin_idx = (angle / (360.0 / N_BINS)).astype(int)
    ring = []
    for b in range(N_BINS):
        sel = np.nonzero(bin_idx == b)[0]
        if len(sel) == 0:
            continue
        ring.append(pts[sel[np.argmin(dist[sel])]])
    ring = np.array(ring)

    fig, ax = scatter_fig(f"step6: final V1 ring ({len(ring)}/{N_BINS} angular bins hit)")
    ax.scatter(main_cols, main_rows, s=1, color="lightgray")
    ax.plot(np.append(ring[:, 0], ring[0, 0]), np.append(ring[:, 1], ring[0, 1]), "-o", color="red", markersize=4, linewidth=1)
    fig.savefig(out_dir / "step6_final_ring.png", dpi=130)
    plt.close(fig)

    # Step 7: unkink -- drop points where the ray-cast grabbed a short nearby spur instead of the
    # main curve (see unkink_ring docstring)
    unkinked = unkink_ring(ring)
    fig, ax = scatter_fig(f"step7: unkinked ({len(ring)}->{len(unkinked)} pts)")
    ax.plot(np.append(ring[:, 0], ring[0, 0]), np.append(ring[:, 1], ring[0, 1]), "-", color="lightcoral", linewidth=1)
    ax.plot(np.append(unkinked[:, 0], unkinked[0, 0]), np.append(unkinked[:, 1], unkinked[0, 1]),
            "-o", color="darkred", markersize=3, linewidth=1.5)
    fig.savefig(out_dir / "step7_unkinked.png", dpi=130)
    plt.close(fig)

    return {
        "animal": animal, "ring_px": unkinked, "centroid_px": centroid,
        "probe_px": probe_px, "probe_labels": probe_labels, "px_per_mm": px_per_mm,
        "ring_coverage": len(ring) / N_BINS, "source": "automated",
    }


def main() -> None:
    zf = zipfile.ZipFile(GEOMETRY_ZIP)
    results = []
    for animal in ANIMALS:
        out_dir = OUTPUT_ROOT / f"step_by_step_{animal}"
        result = extract_v1_ring(zf, animal, out_dir)
        results.append(result)
        print(f"{animal}: ring={len(result['ring_px'])}pts ({result['ring_coverage']:.0%} coverage, "
              f"{result['source']}) -> {out_dir}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for result, ax in zip(results, axes.ravel()):
        ring = result["ring_px"]
        color = "green" if result["source"] == "hand_traced" else "red"
        ax.plot(np.append(ring[:, 0], ring[0, 0]), np.append(ring[:, 1], ring[0, 1]), "-", color=color, linewidth=1.5)
        ax.scatter(*result["centroid_px"], s=60, marker="x", color="black")
        ax.scatter(result["probe_px"][:, 0], result["probe_px"][:, 1], s=40, marker="*", color="blue")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        tag = "hand-traced" if result["source"] == "hand_traced" else f"{result['ring_coverage']:.0%}"
        ax.set_title(f"{result['animal']} ({tag})", fontsize=9)
    fig.tight_layout()
    overview_path = OUTPUT_ROOT / "step_by_step_all_animals_overview.png"
    fig.savefig(overview_path, dpi=140)
    plt.close(fig)
    print(f"\noverview: {overview_path}")


if __name__ == "__main__":
    main()
