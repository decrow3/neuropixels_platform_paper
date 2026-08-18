#!/usr/bin/env python3
"""Register MouseV2 probes to the Zhuang common map using ANATOMY (area border shape), not RF values.

This is the anatomical counterpart to `register_mousev2_rf_to_zhuang_v1.py`, which places each
probe by matching its own observed RF center against the Zhuang V1 retinotopy field. That method
never looks at where the probe actually sits relative to area borders. This script does the
opposite: `data/mouseV2_ephys_animals_geometry_extraction.zip` (2026-08-17) contains, per animal,
a hand-guided extraction of the area-border network visible in the cranial-window photograph used
for targeting (`reference/V2TargetingForTeam.pdf`), plus each probe's/opening's pixel position and
a photo-specific px/mm calibration. The extracted border pixels trace the same "V1 pentagon
surrounded by LM/AL/RL/AM blobs" topology visible in the Zhuang field-sign boundary map, so a
per-animal RIGID transform (rotation + translation + left-right reflection; SCALE IS FIXED, not
fit -- see below) can be found by matching the animal's own border-pixel cloud against the Zhuang
boundary raster.

Evidence weighting (per user direction): area borders are PRIMARY evidence (the shape-matching
term dominates the objective). The "retinotopic_contours" layer (colored rings drawn around each
opening, marking local proximity to a target retinotopic locus -- the same convention used
elsewhere for HVA targeting, per user) has no known degree calibration in this extraction, so it
cannot be shape-matched the way borders can. It is used only as a soft "stay near V1" containment
term (the same domain penalty applied to probe positions, since probes are experimentally known to
be in V1) and is carried through to the QA figures for visual inspection -- never as a hard
geometric constraint.

Scale and rotation (revived 2026-08-18, reconstructing an approach originally worked out
2026-08-17 whose script was never committed -- see `detect_ring_apex`): the earlier version of
this script FIXED scale from each animal's own px/mm calibration combined with the Zhuang
template's own px/mm constant. Per user direction, that calibration is an unreliable approximation
and should be disregarded entirely -- fitting is now done directly in raw animal-photo pixels, and
SCALE IS A FREE FIT PARAMETER instead. This works because V1's own shape supplies an independent
scale reference: V1's border is a "teardrop" with one sharp anteromedial corner (between areas RL
and AM) and an otherwise smooth, rounded rest of the loop, found by curvature (`detect_ring_apex`)
in BOTH the Zhuang template and each animal's own traced/extracted ring. That corner corresponds
biologically to where probe A was aimed (median apex-to-probe-A distance was under ~0.3mm for most
animals, though not all, when checked directly in animal-pixel space) -- so probe A's own position
anchors rotation (via its direction relative to each side's own V1 centroid) to a narrow search
window instead of the full +/-180deg range, AND is explicitly kept near the Zhuang apex by a term
in the fit objective itself (`PROBE_A_ANCHOR_WEIGHT`) -- the narrow window alone was checked against
the QA figure and found insufficient, letting several animals' border-shape term win at a rotation
that put probe A nowhere near the apex.

Scale (revised 2026-08-18 after user feedback on the first apex-only version: 816305 shrank
implausibly to 0.43x the naive calibration with only a loose apex-distance-based guess to bound
it): instead of a single apex-to-centroid distance, scale is guessed by matching V1's own ENCLOSED
AREA (`area_based_scale_guess`) between each animal's traced ring and the Zhuang VISp mask -- an
aggregate shape statistic, much less sensitive to noise in any one point than a single distance.
That guess sets BOTH a tight search window (+/-20-25%, see SCALE_GUESS_LOW/HIGH_FACTOR) AND an
explicit regularization anchor in the objective (SCALE_REGULARIZATION_WEIGHT), so the border-shape
term can no longer drift scale to an implausible value on its own. Recovered per-animal scale
factors cluster around ~0.7-0.9x what the old fixed px/mm-based calibration implied (i.e. that
calibration was systematically ~10-30% too large), consistent with the user's independent estimate
that the SVG-implied scale needed to come down by about 20%.

Border-pixel cleaning: extracted border layers mix thick, coherent curves (the real border
network) with small scattered false-positive specks (vessel edges misclassified under the relaxed
chromatic threshold; see each animal's package README). Connected components smaller than
max(15 px, 3% of the largest component) are dropped before fitting.

Rotation/reflection ambiguity: border shape alone can have more than one locally-good fit (e.g. a
partial arc can superficially match more than one place on the boundary network), so each animal
is fit at both reflections (mirrors the left-right handedness ambiguity already handled via
`REFLECT_ML` elsewhere in this project) with a global optimizer (`differential_evolution`, mirroring
the convention in `register_allen_session_to_zhuang.py::fit_candidate`) over rotation and the
transformed border centroid; the lower-objective reflection is kept. Per-animal agreement in the
chosen reflection sign is reported as an honest consistency check, not assumed.

Caveat carried through everywhere: `site_semantics` differs by animal (813810/816308/817335/810531
= realized probe location; 815152/810532 = intended penetration target; 816305/817334 =
approximate opening center, per the package README) -- intended/approximate sites may not match
where a probe actually recorded, so results for those four animals should be read as lower
precision, not pooled uncritically with the four realized-location animals.

Independent validation: `probe_anatomical_position.csv` is joined against
`06e_mousev2_rf_registered_to_zhuang_v1/mousev2_probe_inferred_v1_position.csv` (RF-value-based
method) per (site, probe) to see whether two methods that share no fitting information -- one uses
only anatomy, the other only RF values -- agree on where each probe sits in the Zhuang map.

Update 2026-08-18: for animals with a human hand-traced V1 border (`reference/<animal>_layers.svg`,
see `HAND_TRACED_ANIMALS`), that vector path is used as the V1-ring border source directly instead
of the automated `isolate_nearest_ring_border` reconstruction -- see `load_hand_traced_ring` and
`ring_source` in the per-animal/per-probe output.
"""

from __future__ import annotations

import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_allen_session_to_zhuang import AREA_SEEDS_XY, build_template, pseudo_huber  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ZIP = ROOT / "data/mouseV2_ephys_animals_geometry_extraction.zip"
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
SITE_ORDERING = ROOT / "data/imports/pilot_rf_peaks_v1/rf_probe_ordering.csv"
RF_REGISTRATION = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1/mousev2_probe_inferred_v1_position.csv"
OUTPUT = ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang"

# Same constants used in check_probe_area_labels_vs_zhuang_registration.py to place the Zhuang
# Figure 9 template in physical (mm) units.
ZHUANG_FIG3_SCALE_BAR_PX = 62.0
ZHUANG_FIG3_SCALE_BAR_MM = 0.5
FIG3_TO_FIG9_SIMILARITY_SCALE = 0.8432313316638625
ZHUANG_PX_PER_MM = ZHUANG_FIG3_SCALE_BAR_PX / ZHUANG_FIG3_SCALE_BAR_MM * FIG3_TO_FIG9_SIMILARITY_SCALE

ANIMALS = ["810531", "810532", "813810", "815152", "816305", "816308", "817334", "817335"]
BORDER_LAYER_OVERRIDE = {"813810": "area_borders_trace_guided.png"}
CONTOUR_LAYER_OVERRIDE = {"813810": "retinotopic_contours_conservative.png"}
DEFAULT_BORDER_LAYER = "area_borders_relaxed.png"
DEFAULT_CONTOUR_LAYER = "retinotopic_contours.png"

# Animals with a human hand-traced `reference/<animal>_layers.svg` V1 border (as of 2026-08-18: a
# vector <path> or <polygon> drawn directly over the same photo the automated
# `isolate_nearest_ring_border` extraction below has to reconstruct from noisy raster border
# pixels -- 813810 and 816308 use <polygon> inside the "Area borders" group; 815152 and 816305 also
# use a <polygon>/<path> respectively, but sitting as a direct child of the SVG root, outside every
# named layer group -- both gaps found only after `load_hand_traced_ring` was fixed to check for
# them; the other four use <path> inside "Area borders"). Verified by eye against the automated
# extraction for all eight: cleanly wraps all 4 probes with none of the spike/gap artifacts that
# method produces. Used directly as the V1-ring border source in `load_animal` instead of the
# automated isolation for these animals -- the "other_border" outer-HVA rotation anchor still comes
# from the automated raster extraction either way (the hand trace only covers V1's own loop, not
# the surrounding network).
REFERENCE_DIR = ROOT / "reference"
HAND_TRACED_ANIMALS = {"810531", "810532", "813810", "815152", "816305", "816308", "817334", "817335"}

# 813810's SVG (viewBox 1424x1196) turned out to be drawn at exactly 2x its own metadata's
# `authoritative_raster_dimensions_px` ([712, 598], the space probe pixel_xy is in) -- caught
# because the raw polygon didn't enclose any of the 4 probes at all when first checked against the
# QA figure. Confirmed directly: dividing the traced polygon by 2 puts it right around the probes.
# No other hand-traced animal's viewBox mismatches its own authoritative_raster_dimensions_px.
SVG_SCALE_OVERRIDE = {"813810": 0.5}
SVG_NS = {"svg": "http://www.w3.org/2000/svg", "inkscape": "http://www.inkscape.org/namespaces/inkscape"}


def parse_svg_path(d: str, n_samples_per_curve: int = 8) -> np.ndarray:
    """Minimal SVG path sampler covering the M/L/H/V/C/S (absolute and relative) + Z commands seen
    across Inkscape's hand-traced paths here -- extended 2026-08-18 from an M/L/C/Z-only version
    after 817334/817335's traces (produced with more aggressive path simplification) turned out to
    also use H/V (axis-aligned linetos) and S (smooth/shorthand cubic Bezier, reflecting the
    previous curve's second control point; degenerates to a straight line from `cur` when not
    immediately preceded by a C/S, per SVG spec). Cubic Beziers are sampled at
    `n_samples_per_curve` points."""
    tokens = re.findall(r"[MmLlHhVvCcSsZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    i = 0
    cur = np.array([0.0, 0.0])
    start = None
    points = []
    cmd = None
    last_c2 = None  # previous cubic's second control point, absolute coords -- for S/s reflection

    def nextnum():
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        if tokens[i] in "MmLlHhVvCcSsZz":
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
            last_c2 = None
        elif cmd in ("L", "l"):
            x, y = nextnum(), nextnum()
            if cmd == "l":
                x, y = x + cur[0], y + cur[1]
            cur = np.array([x, y])
            points.append(cur.copy())
            last_c2 = None
        elif cmd in ("H", "h"):
            x = nextnum()
            if cmd == "h":
                x = x + cur[0]
            cur = np.array([x, cur[1]])
            points.append(cur.copy())
            last_c2 = None
        elif cmd in ("V", "v"):
            y = nextnum()
            if cmd == "v":
                y = y + cur[1]
            cur = np.array([cur[0], y])
            points.append(cur.copy())
            last_c2 = None
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
            last_c2 = p2
        elif cmd in ("S", "s"):
            x2, y2, x, y = (nextnum() for _ in range(4))
            p0 = cur
            p1 = 2 * cur - last_c2 if last_c2 is not None else cur
            if cmd == "s":
                p2, p3 = cur + [x2, y2], cur + [x, y]
            else:
                p2, p3 = np.array([x2, y2]), np.array([x, y])
            for t in np.linspace(0, 1, n_samples_per_curve)[1:]:
                points.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
            cur = p3
            last_c2 = p2
        elif cmd in ("Z", "z"):
            if start is not None:
                points.append(start.copy())
            i += 1
            last_c2 = None
        else:
            i += 1
    return np.array(points)


def parse_svg_polygon(points_attr: str) -> np.ndarray:
    """Parses an SVG <polygon> `points` attribute ("x1,y1 x2,y2 ..."; comma/whitespace mixed)."""
    coords = [float(v) for v in points_attr.replace(",", " ").split()]
    return np.array(coords).reshape(-1, 2)


def _parse_shape_element(el: ET.Element) -> np.ndarray | None:
    tag = el.tag.split("}")[-1]
    if tag == "path":
        return parse_svg_path(el.get("d"))
    if tag == "polygon":
        return parse_svg_polygon(el.get("points"))
    return None


def load_hand_traced_ring(animal: str) -> np.ndarray | None:
    """Returns the hand-traced V1 border shape from reference/<animal>_layers.svg, or None if no
    such file/shape exists for this animal. Illustrator/Inkscape represents a closed hand-drawn
    shape as either a <path> (freehand pen tool) or a <polygon> (points-only shape) depending on
    how it was drawn -- checked 2026-08-18 after 813810 and 816308 turned out to have a traced
    shape that a path-only search silently missed (both use <polygon>, not <path>).

    Two places are checked: (1) inside a layer group labeled "Area borders..." (searching ALL
    descendants, not just direct children, in case of nested sub-groups) -- true for 6/8 animals;
    (2) a shape sitting as a direct child of the SVG root, outside every named layer group entirely
    -- true for 815152 and 816305 (both st2-class blue strokes, #1A00E9 and #602AFF respectively;
    user pointed out these two DO have a visible trace after the group-only search reported them as
    untraced, and direct inspection of the raw SVG confirmed the shape was there but not nested in
    any <g label=...>)."""
    svg_path = REFERENCE_DIR / f"{animal}_layers.svg"
    if not svg_path.exists():
        return None
    root = ET.fromstring(svg_path.read_text(encoding="utf-8", errors="replace"))
    for group in root.findall("svg:g", SVG_NS):
        label = group.get("{http://www.inkscape.org/namespaces/inkscape}label") or ""
        if not label.startswith("Area borders"):
            continue
        for el in group.iter():
            ring = _parse_shape_element(el)
            if ring is not None:
                return ring * SVG_SCALE_OVERRIDE.get(animal, 1.0)
    for el in root:
        ring = _parse_shape_element(el)
        if ring is not None:
            return ring * SVG_SCALE_OVERRIDE.get(animal, 1.0)
    return None

def detect_ring_apex(ring_xy: np.ndarray, window_fraction: float = 1 / 24) -> tuple[np.ndarray, int]:
    """Find a closed ring's single sharpest corner via smoothed local turning angle. V1's own
    border, in both the Zhuang template and every animal's own traced/extracted ring, is a
    "teardrop": one sharp anteromedial corner (between areas RL and AM) and an otherwise smooth,
    rounded rest of the loop -- so a global argmax of turning angle finds it directly, no
    directional restriction needed. `ring_xy` must be points ordered around the loop (true of both
    the hand-traced SVG paths and the angular ray-cast rings from `isolate_nearest_ring_border`).
    Window is sized as a FRACTION of the ring's own point count rather than a fixed count, since
    hand-traced rings (50-200pts) and automated ray-cast rings (100-350pts) are sampled very
    differently -- a fixed window would over- or under-smooth one or the other."""
    n = len(ring_xy)
    window = max(3, int(round(n * window_fraction)))
    prev_pt = np.roll(ring_xy, window, axis=0)
    next_pt = np.roll(ring_xy, -window, axis=0)
    v1 = ring_xy - prev_pt
    v2 = next_pt - ring_xy
    v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    turning_angle_deg = np.degrees(np.arccos(np.clip(np.sum(v1n * v2n, axis=1), -1, 1)))
    apex_idx = int(np.argmax(turning_angle_deg))
    return ring_xy[apex_idx], apex_idx


def ring_area(ring_xy: np.ndarray) -> float:
    """Enclosed-area estimate for a traced/extracted V1 ring, used to get a scale guess from V1's
    own AREA (an aggregate shape statistic, robust to noise in any single point) rather than a
    single apex-to-centroid distance -- see `area_based_scale_guess`. Uses the CONVEX HULL area,
    not a raw shoelace polygon area on the ring's own point order: checked directly that shoelace
    systematically UNDERESTIMATED area for the automated ray-cast rings (isolate_nearest_ring_border
    takes the nearest point per angular bin independently, so noisy/sparse bins can produce a small
    local zigzag that is not a true self-intersection but still partially cancels in the shoelace
    sum) -- this showed up as wildly inflated fitted scale (>1.9x) for exactly those animals, not
    the hand-traced ones. V1 is close enough to convex (see the QA figures) that hull area is a
    good proxy for true enclosed area and is immune to that failure mode."""
    return float(ConvexHull(ring_xy).volume)  # scipy 2D convention: .volume is enclosed area


SITE_SEMANTICS = {
    "813810": "realized_probe_location", "816308": "realized_probe_location",
    "817335": "realized_probe_location", "810531": "realized_probe_location",
    "815152": "intended_penetration_target", "810532": "intended_penetration_target",
    "816305": "approximate_opening_center", "817334": "approximate_opening_center",
}

MIN_COMPONENT_PIXELS = 15
MIN_COMPONENT_FRACTION = 0.03
MAX_BORDER_POINTS = 500
BORDER_WEIGHT = 1.0
DOMAIN_WEIGHT = 0.15
CONTOUR_WEIGHT = 0.05
OTHER_BORDER_WEIGHT = 0.2
PROBE_A_ANCHOR_WEIGHT = 1.5
BORDER_HUBER_PX = 6.0
DOMAIN_SOFT_PX = 15.0
OTHER_BORDER_HUBER_PX = 10.0
PROBE_A_ANCHOR_HUBER_PX = 12.0
SEED = 20260817


def load_layer_points(zf: zipfile.ZipFile, animal: str, layer_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixel_xy, boolean_mask) for all opaque pixels in a layer PNG."""
    path = f"all_animals_geometry/{animal}/layers/{layer_name}"
    img = Image.open(io.BytesIO(zf.read(path))).convert("RGBA")
    arr = np.asarray(img)
    mask = arr[..., 3] > 0
    rows, cols = np.nonzero(mask)
    return np.column_stack([cols, rows]).astype(float), mask


def filter_small_components(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    threshold = max(MIN_COMPONENT_PIXELS, MIN_COMPONENT_FRACTION * sizes.max())
    keep = np.nonzero(sizes >= threshold)[0] + 1
    return np.isin(labeled, keep)


def subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


SINGLE_PROBE_LOOP_MAX_RADIUS_MM = 0.85
SINGLE_PROBE_LOOP_MIN_COVERAGE = 0.7
SINGLE_PROBE_LOOP_BINS = 24


def remove_single_probe_loops(mask: np.ndarray, probe_px: np.ndarray, px_per_mm: float) -> tuple[np.ndarray, int]:
    """Remove border pixels that nearly fully encircle ANY one probe within a small radius -- the
    geometric signature of a real, confirmed artifact (found 2026-08-17 by checking the border
    extraction directly against each animal's reference photo): the 4 animals using a PLANNING
    reference photo (810532, 815152, 816305, 817334) have hand-drawn circles marking each intended
    target/opening, color-close enough to the genuine border pen strokes that the "relaxed
    chromatic threshold" extraction (see each animal's package README) picked up both. A real area
    border segment is a long, open arc; it does not form a small closed loop around exactly one
    probe. Checked: this is NOT the same as `implant_openings.png`'s tracked markers (overlap with
    that layer, dilated 5px, was a similar 2-8% for contaminated and clean animals alike) -- it is
    independent stray annotation content in the source photo, not a duplicated/leaked layer.

    A first version classified whole CONNECTED COMPONENTS this way and removed none of the
    artifacts: the circles turn out to be topologically merged into the same sprawling connected
    network as genuine border strokes (real or spurious connecting lines), so no component was
    "80% within 0.6mm of one probe" as originally required. This version instead tests each
    probe's local neighborhood directly, independent of what else that pixel's component connects
    to elsewhere -- proximity alone doesn't trigger removal (a real border segment can pass close
    to a probe), only near-FULL angular coverage around it does, which a grazing line cannot
    produce but a closed loop can."""
    rows, cols = np.nonzero(mask)
    points = np.column_stack([cols, rows]).astype(float)
    max_radius_px = SINGLE_PROBE_LOOP_MAX_RADIUS_MM * px_per_mm
    remove = np.zeros(len(points), dtype=bool)
    for probe in probe_px:
        relative = points - probe
        distance = np.hypot(relative[:, 0], relative[:, 1])
        nearby = distance <= max_radius_px
        if nearby.sum() < 10:
            continue
        angle = np.degrees(np.arctan2(relative[nearby, 1], relative[nearby, 0])) % 360.0
        bins_covered = len(np.unique((angle / (360.0 / SINGLE_PROBE_LOOP_BINS)).astype(int)))
        if bins_covered / SINGLE_PROBE_LOOP_BINS >= SINGLE_PROBE_LOOP_MIN_COVERAGE:
            remove |= nearby
    keep = np.zeros_like(mask)
    keep[rows[~remove], cols[~remove]] = True
    return keep, int(remove.sum())


N_RING_BINS = 72
MAX_POINTS_PER_BIN = 3
MAX_RING_RADIUS_MM = 3.0
OTHER_BORDER_MAX_RADIUS_MM = 6.0


def isolate_nearest_ring_border(points_px: np.ndarray, centroid_px: np.ndarray, px_per_mm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Matching against the FULL border network let an animal's V1 loop latch onto a neighboring
    HVA's border in Zhuang space instead of V1's own (visible as clearly wrong per-animal QA fits
    in the first version of this script). A flood-fill "region enclosing the probes" approach was
    tried next, but the traced border is not topologically closed even after generous (20px)
    morphological gap-closing (checked directly: the background stays one ~95%-of-image component
    regardless of closing radius) -- the extraction has genuine open gaps, not small pixel gaps.

    Instead: V1's own border is, by definition of anatomy, the CLOSEST border material to the
    probes in every direction (probes sit inside V1; any other area's border is necessarily
    farther out beyond V1's own edge in that same direction). So take the probe centroid as an
    origin, bin all cleaned border points by angle around it, and keep only the nearest point(s)
    per angular bin -- a "ray cast outward, take the first hit" ring trace. Points farther than
    `MAX_RING_RADIUS_MM` (V1 is roughly 2-3mm across at these eccentricities) are excluded even as
    a bin's nearest hit, so an extraction gap in one direction pulls in nothing rather than
    incorrectly grabbing a distant HVA-to-HVA border. Verified by eye against 3 animals before
    adoption: two produced a near-complete clean ring; the third (817334, `approximate_opening_center`
    -- the least reliable site_semantics category) produced a partial ring open at top/bottom,
    consistent with that animal's known lower extraction/targeting confidence rather than a bug.

    A pure V1-only fit turned out to under-constrain ROTATION: a single roughly-convex loop, scored
    only by isotropic distance-to-boundary, fits nearly as well at more than one orientation,
    whereas the full network's extra asymmetric structure (the neighboring HVA loops) pins rotation
    much more strongly -- but matching to any of that at full weight is exactly what caused the
    original wrong-line problem. The compromise: everything beyond the V1 ring radius (up to
    `OTHER_BORDER_MAX_RADIUS_MM`) is returned separately as `other_points_px` for use as a WEAK
    supporting rotation anchor (low weight, matched against the full boundary network) -- it should
    nudge orientation, not relocate V1 itself.

    Returns (ring_points_px, other_points_px, angular_coverage_fraction)."""
    relative = points_px - centroid_px
    distance = np.hypot(relative[:, 0], relative[:, 1])
    within_radius = distance <= MAX_RING_RADIUS_MM * px_per_mm
    within_outer_cap = distance <= OTHER_BORDER_MAX_RADIUS_MM * px_per_mm
    angle_deg = np.degrees(np.arctan2(relative[:, 1], relative[:, 0])) % 360.0
    bin_width = 360.0 / N_RING_BINS
    bin_idx = (angle_deg / bin_width).astype(int)

    keep = np.zeros(len(points_px), dtype=bool)
    n_bins_covered = 0
    for b in range(N_RING_BINS):
        candidates = np.nonzero(within_radius & (bin_idx == b))[0]
        if len(candidates) == 0:
            continue
        n_bins_covered += 1
        nearest = candidates[np.argsort(distance[candidates])[:MAX_POINTS_PER_BIN]]
        keep[nearest] = True
    # "other" is everything NOT selected into the ring, out to the outer cap -- not just points
    # beyond MAX_RING_RADIUS_MM. Points inside that radius but edged out by a closer per-bin
    # neighbor are still real border material (often the actual next-layer-out HVA border in a
    # direction where V1's own edge happens to sit close to the centroid) and should count too.
    other = within_outer_cap & ~keep
    return points_px[keep], points_px[other], n_bins_covered / N_RING_BINS


def read_probe_metadata(zf: zipfile.ZipFile, animal: str) -> tuple[float, list[str], np.ndarray]:
    """Returns (px_per_mm, probe_labels, probe_px) in original photo pixel coordinates. Schema
    differs: 813810 (revision 3) uses inner_optical_window_calibration/probe_locations; the other
    seven animals use calibration/site_locations. Same shape either way."""
    metadata = json.loads(zf.read(f"all_animals_geometry/{animal}/{animal}_metadata.json"))
    calibration = metadata.get("inner_optical_window_calibration", metadata.get("calibration"))
    px_per_mm = calibration["pixels_per_mm"]
    site_locations = metadata.get("probe_locations", metadata.get("site_locations"))
    probe_labels = [site_locations[k]["recording_label"] for k in sorted(site_locations)]
    probe_px = np.array([site_locations[k]["pixel_xy"] for k in sorted(site_locations)], dtype=float)
    return px_per_mm, probe_labels, probe_px


def load_animal(zf: zipfile.ZipFile, animal: str, rng: np.random.Generator) -> dict:
    px_per_mm, probe_labels, probe_px = read_probe_metadata(zf, animal)

    border_layer = BORDER_LAYER_OVERRIDE.get(animal, DEFAULT_BORDER_LAYER)
    _, border_mask = load_layer_points(zf, animal, border_layer)
    cleaned_mask = filter_small_components(border_mask)
    cleaned_mask, n_single_probe_loops_removed = remove_single_probe_loops(cleaned_mask, probe_px, px_per_mm)
    if n_single_probe_loops_removed:
        print(f"{animal}: removed {n_single_probe_loops_removed} single-probe-encircling loop "
              f"px (opening-annotation artifact, not a real border) before ring isolation")
    rows, cols = np.nonzero(cleaned_mask)
    cleaned_px = np.column_stack([cols, rows]).astype(float)
    probe_centroid_px = probe_px.mean(axis=0)
    border_px, other_border_px, ring_coverage = isolate_nearest_ring_border(cleaned_px, probe_centroid_px, px_per_mm)
    isolation_ok = len(border_px) >= 20
    if not isolation_ok:
        print(f"WARNING: {animal}: nearest-ring V1 isolation found too little border material "
              f"({len(border_px)} px, {ring_coverage:.0%} angular coverage) -- falling back to the full cleaned border network")
        border_px = cleaned_px
        other_border_px = np.empty((0, 2))

    ring_source = "automated"
    hand_ring_px = load_hand_traced_ring(animal) if animal in HAND_TRACED_ANIMALS else None
    if hand_ring_px is not None:
        # the hand trace replaces the V1 RING only; `other_border_px` (outer-HVA weak rotation
        # anchor) keeps coming from the automated raster extraction above regardless.
        border_px = hand_ring_px
        ring_coverage = 1.0
        isolation_ok = True
        ring_source = "hand_traced"

    # apex detection needs the ring's own point ORDER (cyclic around the loop), so run it before
    # subsampling shuffles/thins that order away.
    apex_px, _ = detect_ring_apex(border_px)
    probe_a_px = probe_px[probe_labels.index("A")] if "A" in probe_labels else None
    apex_to_probe_a_mm = (float(np.linalg.norm(apex_px - probe_a_px) / px_per_mm)
                           if probe_a_px is not None else None)
    animal_area_px = ring_area(border_px)

    # Fitting now happens directly in raw animal-photo pixels (see module docstring: the px/mm
    # calibration is disregarded as a scale source, so there is no reason to convert to mm first).
    border_px_sub = subsample(border_px, MAX_BORDER_POINTS, rng)
    other_border_px_sub = subsample(other_border_px, MAX_BORDER_POINTS, rng)

    contour_layer = CONTOUR_LAYER_OVERRIDE.get(animal, DEFAULT_CONTOUR_LAYER)
    contour_px, _ = load_layer_points(zf, animal, contour_layer)
    contour_px_sub = subsample(contour_px, MAX_BORDER_POINTS, rng)

    return {
        "animal": animal,
        "px_per_mm": px_per_mm,
        "border_px": border_px_sub,
        "n_border_px_raw": int(border_mask.sum()),
        "n_border_px_cleaned": int(cleaned_mask.sum()),
        "n_single_probe_loops_removed": n_single_probe_loops_removed,
        "n_border_px_v1_isolated": len(border_px),
        "v1_isolation_ok": isolation_ok,
        "v1_isolation_ring_coverage": ring_coverage,
        "ring_source": ring_source,
        "other_border_px": other_border_px_sub,
        "contour_px": contour_px_sub,
        "probe_labels": probe_labels,
        "probe_px": probe_px,
        "probe_centroid_px": probe_px.mean(axis=0),
        "apex_px": apex_px,
        "probe_a_px": probe_a_px,
        "apex_to_probe_a_mm": apex_to_probe_a_mm,
        "animal_area_px": animal_area_px,
        "site_semantics": SITE_SEMANTICS[animal],
    }


def build_transform(theta: float, center_x: float, center_y: float, reflection: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    matrix = rotation @ np.diag([reflection, 1.0]) * scale
    return matrix, np.array([center_x, center_y])


def apply_transform(points: np.ndarray, pivot: np.ndarray, matrix: np.ndarray, center_px: np.ndarray) -> np.ndarray:
    return (points - pivot) @ matrix.T + center_px


ROTATION_WINDOW_DEG = 40.0
SCALE_GUESS_LOW_FACTOR = 0.8
SCALE_GUESS_HIGH_FACTOR = 1.25
SCALE_REGULARIZATION_WEIGHT = 3.0


def apex_anchored_rotation_guess(data: dict, reflection: int,
                                  zhuang_apex_xy: np.ndarray, zhuang_centroid_xy: np.ndarray) -> float:
    """Use PROBE A's position (per user direction: probe A sits near V1's anteromedial apex, and
    that correspondence is the anchor, not the independently curvature-detected ring apex, which
    turned out to be unreliable for animals with a noisier extraction -- see PROBE_A_ANCHOR_WEIGHT
    docstring) to get a rotation estimate, independent of the disregarded px/mm calibration: the
    rotation that takes this animal's own (probe A - probe centroid) DIRECTION, under the given
    reflection, onto the Zhuang (apex - V1 centroid) direction. Falls back to the ring's own
    curvature-detected apex if this animal has no probe A."""
    anchor_px = data["probe_a_px"] if data["probe_a_px"] is not None else data["apex_px"]
    u_animal = anchor_px - data["probe_centroid_px"]
    u_animal_reflected = np.array([reflection * u_animal[0], u_animal[1]])
    u_zhuang = zhuang_apex_xy - zhuang_centroid_xy
    theta_guess = np.arctan2(u_zhuang[1], u_zhuang[0]) - np.arctan2(u_animal_reflected[1], u_animal_reflected[0])
    return float(theta_guess)


def area_based_scale_guess(animal_area_px: float, zhuang_area_px: float) -> float:
    """Scale guess from matching ENCLOSED AREA of V1's own ring against the Zhuang V1 mask, rather
    than a single apex-to-centroid distance -- an aggregate shape statistic, much less sensitive to
    noise in any one traced/detected point. Added 2026-08-18 after the apex-distance-based guess
    let scale drift as low as 0.43x the naive px/mm-implied value for one animal (816305) with
    nothing to stop it -- the border-shape objective alone was too tolerant of a shrunk, still
    roughly-teardrop-shaped ring. This is now used BOTH for the (now much tighter, see
    SCALE_GUESS_LOW/HIGH_FACTOR) search bounds AND as an explicit regularization anchor in the fit
    objective (see SCALE_REGULARIZATION_WEIGHT), matching this project's established pattern for
    keeping a free scale/length parameter near an independently-derived expectation (see
    REGULARIZATION_WEIGHT in register_mousev2_units_along_probe_shank.py)."""
    return float(np.sqrt(zhuang_area_px / max(animal_area_px, 1e-6)))


def fit_animal_candidates(data: dict, boundary_interp: RegularGridInterpolator, v1_interp: RegularGridInterpolator,
                           other_boundary_interp: RegularGridInterpolator,
                           zhuang_apex_xy: np.ndarray, zhuang_centroid_xy: np.ndarray, zhuang_area_px: float,
                           width: int, height: int, seed: int) -> list[dict]:
    """Fit both left-right reflections independently. Reflection is a shared imaging/annotation
    convention (all 8 animals are the same hemisphere, same photo convention), not a per-animal
    biological quantity, so the final choice is made globally across animals in `main()` rather
    than per-animal here -- a per-animal argmin would let weak, ambiguous border evidence in one
    animal flip its mirror independently of the other seven. Rotation and scale are no longer a
    fixed calibration and a free full-range search respectively (see module docstring): rotation
    is searched only in a +/-ROTATION_WINDOW_DEG window around the apex-anchored estimate, and
    scale is searched in a tight window around the area-matched guess (`area_based_scale_guess`),
    with a regularization term keeping the fit near it -- both refined by the same border-shape
    objective as before."""
    pivot = data["border_px"].mean(axis=0)
    probe_a_local_px = data["probe_a_px"]
    scale_guess = area_based_scale_guess(data["animal_area_px"], zhuang_area_px)

    def objective(params: np.ndarray, reflection: int) -> float:
        theta, center_x, center_y, scale = params
        matrix, center_px = build_transform(theta, center_x, center_y, reflection, scale)
        border_xy = apply_transform(data["border_px"], pivot, matrix, center_px)
        border_dist = boundary_interp(border_xy[:, ::-1])
        border_term = float(np.mean(pseudo_huber(border_dist / BORDER_HUBER_PX)))

        probe_xy = apply_transform(data["probe_px"], pivot, matrix, center_px)
        probe_dist = v1_interp(probe_xy[:, ::-1])
        domain_term = float(np.mean(np.square(probe_dist / DOMAIN_SOFT_PX)))

        if len(data["contour_px"]):
            contour_xy = apply_transform(data["contour_px"], pivot, matrix, center_px)
            contour_dist = v1_interp(contour_xy[:, ::-1])
            contour_term = float(np.mean(np.square(contour_dist / DOMAIN_SOFT_PX)))
        else:
            contour_term = 0.0

        if len(data["other_border_px"]):
            other_xy = apply_transform(data["other_border_px"], pivot, matrix, center_px)
            other_dist = other_boundary_interp(other_xy[:, ::-1])
            other_term = float(np.mean(pseudo_huber(other_dist / OTHER_BORDER_HUBER_PX)))
        else:
            other_term = 0.0

        # Explicit probe-A<->apex anchor (not just the initial guess + narrow rotation window
        # above): checked directly against the QA figure that the window alone was NOT sufficient
        # -- the border-shape term alone still preferred a rotation ~30-40deg off the apex-anchored
        # guess for several animals (its residual barely changed at the true rotation, since V1's
        # ring is locally symmetric-ish), landing probe A nowhere near Zhuang's apex. This term
        # makes that correspondence part of what is being fit, not just where the search starts.
        # Anchors PROBE A itself (per user direction), not the independently curvature-detected
        # ring apex -- the latter turned out unreliable for animals with a noisier border trace
        # (apex_to_probe_a_mm > 1mm for several), so anchoring to it would fit the wrong landmark.
        if probe_a_local_px is not None:
            probe_a_xy = apply_transform(probe_a_local_px[None, :], pivot, matrix, center_px)[0]
            apex_dist = np.linalg.norm(probe_a_xy - zhuang_apex_xy)
            apex_term = float(pseudo_huber(np.array([apex_dist / PROBE_A_ANCHOR_HUBER_PX]))[0])
        else:
            apex_term = 0.0

        # Keeps scale near the area-matched guess even within its own (now tight) bounds -- added
        # alongside the tighter bounds themselves, not instead of them, after 816305 shrank to
        # 0.43x the naive calibration with only the (looser, apex-distance-based) old bounds to
        # stop it; area matching is a much more global, noise-robust scale estimate than any
        # single point-pair distance.
        scale_penalty = SCALE_REGULARIZATION_WEIGHT * float(np.log(scale / scale_guess)) ** 2

        return (BORDER_WEIGHT * border_term + DOMAIN_WEIGHT * domain_term + CONTOUR_WEIGHT * contour_term
                + OTHER_BORDER_WEIGHT * other_term + PROBE_A_ANCHOR_WEIGHT * apex_term + scale_penalty)

    candidates = []
    for reflection_idx, reflection in enumerate((1, -1)):
        theta_guess = apex_anchored_rotation_guess(data, reflection, zhuang_apex_xy, zhuang_centroid_xy)
        rotation_window = np.radians(ROTATION_WINDOW_DEG)
        bounds = [
            (theta_guess - rotation_window, theta_guess + rotation_window),
            (-20.0, width + 20.0), (-20.0, height + 20.0),
            (SCALE_GUESS_LOW_FACTOR * scale_guess, SCALE_GUESS_HIGH_FACTOR * scale_guess),
        ]
        result = differential_evolution(
            objective, bounds, args=(reflection,), seed=seed + reflection_idx, maxiter=300, popsize=15,
            tol=1e-8, polish=True, workers=1, updating="immediate",
        )
        candidates.append({"animal": data["animal"], "reflection": reflection, "objective": float(result.fun),
                            "params": result.x, "pivot": pivot,
                            "theta_guess_deg": float(np.degrees(theta_guess)), "scale_guess": scale_guess})
    return candidates


def finalize_fit(data: dict, candidate: dict, other_objective: float,
                  boundary_interp: RegularGridInterpolator) -> dict:
    theta, center_x, center_y, scale = candidate["params"]
    matrix, center_px = build_transform(theta, center_x, center_y, candidate["reflection"], scale)
    pivot = candidate["pivot"]
    border_xy = apply_transform(data["border_px"], pivot, matrix, center_px)
    border_dist = boundary_interp(border_xy[:, ::-1])
    probe_xy = apply_transform(data["probe_px"], pivot, matrix, center_px)
    contour_xy = apply_transform(data["contour_px"], pivot, matrix, center_px) if len(data["contour_px"]) else np.empty((0, 2))
    other_border_xy = (apply_transform(data["other_border_px"], pivot, matrix, center_px)
                        if len(data["other_border_px"]) else np.empty((0, 2)))
    apex_xy = apply_transform(data["apex_px"][None, :], pivot, matrix, center_px)[0]
    naive_scale = ZHUANG_PX_PER_MM / data["px_per_mm"]

    return {
        "animal": data["animal"],
        "reflection": candidate["reflection"],
        "theta_deg": float(np.degrees(theta)),
        "center_x_px": float(center_x),
        "center_y_px": float(center_y),
        "scale_fit": float(scale),
        "scale_guess": candidate["scale_guess"],
        "scale_relative_to_naive_px_per_mm": float(scale / naive_scale),
        "theta_guess_deg": candidate["theta_guess_deg"],
        "objective": candidate["objective"],
        "objective_other_reflection": other_objective,
        "n_border_points_fit": len(data["border_px"]),
        "n_border_px_raw": data["n_border_px_raw"],
        "n_border_px_cleaned": data["n_border_px_cleaned"],
        "n_single_probe_loops_removed": data["n_single_probe_loops_removed"],
        "n_border_px_v1_isolated": data["n_border_px_v1_isolated"],
        "v1_isolation_ok": data["v1_isolation_ok"],
        "v1_isolation_ring_coverage": data["v1_isolation_ring_coverage"],
        "ring_source": data["ring_source"],
        "apex_to_probe_a_mm": data["apex_to_probe_a_mm"],
        "border_median_residual_px": float(np.median(border_dist)),
        "border_p90_residual_px": float(np.quantile(border_dist, 0.9)),
        "border_dist": border_dist,
        "border_xy": border_xy,
        "probe_xy": probe_xy,
        "contour_xy": contour_xy,
        "other_border_xy": other_border_xy,
        "apex_xy": apex_xy,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)
    height, width = template["domain"].shape

    # Isolate V1's OWN border segment out of the full mean_field_sign_boundary network: without
    # this, animal border points could match against a neighboring HVA's border in Zhuang space
    # instead of V1's (this happened in the first version of this script -- visibly wrong fits in
    # the per-animal QA figure). VISp's mask is already a clean filled connected component
    # (`build_template`'s seeded-component construction), so its own perimeter is just the part
    # of the boundary network immediately adjacent to it.
    visp_mask = template["area_masks"]["VISp"]
    visp_border_mask = ndimage.binary_dilation(visp_mask, iterations=2) & template["boundary"] & ~visp_mask
    boundary_distance = ndimage.distance_transform_edt(~visp_border_mask).astype(np.float32)
    boundary_interp = RegularGridInterpolator(
        (np.arange(height), np.arange(width)), boundary_distance, bounds_error=False, fill_value=100.0,
    )
    v1_interp = template["area_distance"]["VISp"]

    # Zhuang-side apex: the fixed anteromedial-corner landmark every animal's own apex (see
    # `detect_ring_apex`) is anchored against. VISp mask's own contour (not the dilated border
    # mask used above) gives a single clean closed ring in (row, col) order.
    zhuang_ring_rc = max(measure.find_contours(visp_mask.astype(float), level=0.5), key=len)
    zhuang_ring_xy = zhuang_ring_rc[:, ::-1]
    visp_rows, visp_cols = np.nonzero(visp_mask)
    zhuang_centroid_xy = np.array([visp_cols.mean(), visp_rows.mean()])
    zhuang_apex_xy, _ = detect_ring_apex(zhuang_ring_xy)
    zhuang_area_px = float(visp_mask.sum())
    print(f"Zhuang V1 apex (curvature-detected): row={zhuang_apex_xy[1]:.1f}, col={zhuang_apex_xy[0]:.1f} "
          f"(centroid row={zhuang_centroid_xy[1]:.1f}, col={zhuang_centroid_xy[0]:.1f}), "
          f"V1 area={zhuang_area_px:.0f}px^2")

    # V1-only fitting under-constrains rotation (a single roughly-convex loop scores nearly as well
    # at more than one orientation). Border material beyond the V1 ring (isolate_nearest_ring_border's
    # `other_points_px`) is matched against the FULL network at low weight as a supporting rotation
    # anchor -- see OTHER_BORDER_WEIGHT and its docstring.
    other_boundary_distance = ndimage.distance_transform_edt(~template["boundary"]).astype(np.float32)
    other_boundary_interp = RegularGridInterpolator(
        (np.arange(height), np.arange(width)), other_boundary_distance, bounds_error=False, fill_value=100.0,
    )

    rng = np.random.default_rng(SEED)
    zf = zipfile.ZipFile(GEOMETRY_ZIP)

    animal_data = {}
    animal_candidates = {}
    for animal in ANIMALS:
        data = load_animal(zf, animal, rng)
        animal_data[animal] = data
        animal_candidates[animal] = fit_animal_candidates(
            data, boundary_interp, v1_interp, other_boundary_interp, zhuang_apex_xy, zhuang_centroid_xy, zhuang_area_px,
            width, height, seed=SEED + int(animal) % 1000
        )

    # Reflection is one shared imaging/annotation convention across all 8 animals (see
    # fit_animal_candidates docstring), so it is chosen once here from the pooled evidence: sum
    # each reflection's best-fit objective across animals and keep whichever explains the whole
    # dataset better, rather than letting each animal's individually-best (possibly weak) argmin
    # decide independently.
    pooled_objective = {
        reflection: sum(next(c for c in animal_candidates[a] if c["reflection"] == reflection)["objective"]
                         for a in ANIMALS)
        for reflection in (1, -1)
    }
    global_reflection = min(pooled_objective, key=pooled_objective.get)
    print(f"pooled objective by reflection: {pooled_objective} -> global reflection = {global_reflection:+d}")

    fits = []
    probe_rows = []
    n_agree_with_own_argmin = 0
    for animal in ANIMALS:
        data = animal_data[animal]
        candidates = animal_candidates[animal]
        chosen = next(c for c in candidates if c["reflection"] == global_reflection)
        other = next(c for c in candidates if c["reflection"] != global_reflection)
        own_argmin_reflection = min(candidates, key=lambda c: c["objective"])["reflection"]
        n_agree_with_own_argmin += int(own_argmin_reflection == global_reflection)
        fit = finalize_fit(data, chosen, other["objective"], boundary_interp)
        fits.append({**fit, "data": data})
        if data["ring_source"] == "hand_traced":
            isolation_note = "hand-traced V1 ring"
        elif data["v1_isolation_ok"]:
            isolation_note = f"V1-isolated ({data['v1_isolation_ring_coverage']:.0%} angular ring coverage)"
        else:
            isolation_note = "ISOLATION FAILED, used full border network"
        apex_note = (f"apex-to-A={fit['apex_to_probe_a_mm']:.2f}mm" if fit["apex_to_probe_a_mm"] is not None
                     else "no probe A")
        print(f"{animal}: reflection={fit['reflection']:+d} (own argmin {own_argmin_reflection:+d}) "
              f"theta={fit['theta_deg']:+.1f}deg (apex-guess {fit['theta_guess_deg']:+.1f}deg) "
              f"scale={fit['scale_fit']:.1f}px/animalpx ({fit['scale_relative_to_naive_px_per_mm']:.2f}x naive px/mm calibration) "
              f"border_median_residual={fit['border_median_residual_px']:.2f}px "
              f"n_border_pts={fit['n_border_points_fit']} [{isolation_note}, {apex_note}] "
              f"(global-reflection objective {fit['objective']:.3f} vs other reflection {fit['objective_other_reflection']:.3f})")

        probe_row_col = fit["probe_xy"][:, ::-1]
        clipped = np.clip(probe_row_col, [0, 0], [height - 1, width - 1])
        predicted_azimuth = template["fields"]["azimuth_deg"](clipped)
        predicted_elevation = template["fields"]["altitude_deg"](clipped)
        v1_distance = v1_interp(clipped)
        for i, label in enumerate(data["probe_labels"]):
            probe_rows.append({
                "animal": animal,
                "probe": label,
                "site_semantics": data["site_semantics"],
                "zhuang_row": float(fit["probe_xy"][i, 1]),
                "zhuang_col": float(fit["probe_xy"][i, 0]),
                "predicted_azimuth_deg": float(predicted_azimuth[i]),
                "predicted_elevation_deg": float(predicted_elevation[i]),
                "v1_distance_px": float(v1_distance[i]),
                "reflection": fit["reflection"],
                "theta_deg": fit["theta_deg"],
                "border_median_residual_px": fit["border_median_residual_px"],
                "n_border_points_fit": fit["n_border_points_fit"],
                "v1_isolation_ok": fit["v1_isolation_ok"],
                "ring_source": fit["ring_source"],
                "scale_fit_zhuang_px_per_animal_px": fit["scale_fit"],
                "scale_relative_to_naive_px_per_mm": fit["scale_relative_to_naive_px_per_mm"],
                "apex_to_probe_a_mm": fit["apex_to_probe_a_mm"],
            })

    probe_table = pd.DataFrame(probe_rows)

    ordering = pd.read_csv(SITE_ORDERING)[["site", "subject_id"]].astype({"subject_id": str})
    probe_table = probe_table.merge(ordering, left_on="animal", right_on="subject_id", how="left")
    probe_table.to_csv(OUTPUT / "probe_anatomical_position.csv", index=False)

    print(f"\nglobal reflection = {global_reflection:+d}, applied to all {len(ANIMALS)} animals; "
          f"{n_agree_with_own_argmin}/{len(ANIMALS)} animals' own individually-best reflection "
          f"agreed with the pooled choice (the rest had weak/ambiguous own border evidence -- "
          f"see per-animal objective gap in the manifest)")

    # Independent cross-check against the RF-value-based registration (register_mousev2_rf_to_zhuang_v1.py).
    comparison = None
    if RF_REGISTRATION.exists():
        rf_table = pd.read_csv(RF_REGISTRATION)
        anatomy = probe_table.rename(columns={
            "predicted_azimuth_deg": "anatomy_predicted_azimuth_deg",
            "predicted_elevation_deg": "anatomy_predicted_elevation_deg",
            "zhuang_row": "anatomy_zhuang_row", "zhuang_col": "anatomy_zhuang_col",
        })[["site", "probe", "site_semantics", "anatomy_zhuang_row", "anatomy_zhuang_col",
            "anatomy_predicted_azimuth_deg", "anatomy_predicted_elevation_deg"]]
        rf = rf_table.rename(columns={
            "predicted_azimuth_deg": "rf_predicted_azimuth_deg",
            "predicted_elevation_deg": "rf_predicted_elevation_deg",
            "inferred_row": "rf_zhuang_row", "inferred_col": "rf_zhuang_col",
        })[["site", "probe", "rf_zhuang_row", "rf_zhuang_col", "rf_predicted_azimuth_deg", "rf_predicted_elevation_deg"]]
        comparison = anatomy.merge(rf, on=["site", "probe"], how="inner")
        comparison["zhuang_pixel_distance_px"] = np.hypot(
            comparison.anatomy_zhuang_row - comparison.rf_zhuang_row,
            comparison.anatomy_zhuang_col - comparison.rf_zhuang_col,
        )
        comparison["retinotopic_vector_error_deg"] = np.hypot(
            comparison.anatomy_predicted_azimuth_deg - comparison.rf_predicted_azimuth_deg,
            comparison.anatomy_predicted_elevation_deg - comparison.rf_predicted_elevation_deg,
        )
        comparison.to_csv(OUTPUT / "comparison_vs_rf_registration.csv", index=False)
        print(f"\nanatomy vs. RF-value registration ({len(comparison)} probes matched):")
        print(comparison.groupby("site_semantics")[["zhuang_pixel_distance_px", "retinotopic_vector_error_deg"]]
              .median().rename(columns=lambda c: "median_" + c))
    else:
        print(f"\nRF registration output not found at {RF_REGISTRATION}; skipping cross-check")

    manifest = {
        "method": "similarity (rotation + translation + SCALE, all fit, + left-right reflection) "
                   "fit of animal V1-ONLY border pixel cloud (isolated per animal as the nearest "
                   "border material to that animal's own probe centroid in each angular direction "
                   "-- see isolate_nearest_ring_border docstring) to the Zhuang V1-ONLY border "
                   "(isolated as VISp mask's own perimeter out of mean_field_sign_boundary). "
                   "Rotation and scale are anchored (not fixed) via an apex<->apex correspondence: "
                   "V1's border in both the animal photo and the Zhuang template has one sharp "
                   "anteromedial corner (between RL and AM), found by curvature "
                   "(detect_ring_apex), which anatomically corresponds to where probe A was aimed "
                   "-- see apex_to_probe_a_mm. Matching against the FULL multi-area boundary "
                   "network let border points latch onto a neighboring HVA border instead of V1's "
                   "own; isolating V1's own loop on both sides fixes that.",
        "zhuang_px_per_mm_for_reference_only": ZHUANG_PX_PER_MM,
        "rotation_window_deg": ROTATION_WINDOW_DEG,
        "scale_guess_bounds_factor": [SCALE_GUESS_LOW_FACTOR, SCALE_GUESS_HIGH_FACTOR],
        "scale_regularization_weight": SCALE_REGULARIZATION_WEIGHT,
        "zhuang_v1_area_px2": zhuang_area_px,
        "weights": {"border": BORDER_WEIGHT, "domain_probes": DOMAIN_WEIGHT, "domain_contours": CONTOUR_WEIGHT,
                    "other_border_rotation_anchor": OTHER_BORDER_WEIGHT, "probe_a_apex_anchor": PROBE_A_ANCHOR_WEIGHT,
                    "border_huber_px": BORDER_HUBER_PX, "domain_soft_px": DOMAIN_SOFT_PX,
                    "other_border_huber_px": OTHER_BORDER_HUBER_PX, "probe_a_apex_anchor_huber_px": PROBE_A_ANCHOR_HUBER_PX,
                    "max_ring_radius_mm": MAX_RING_RADIUS_MM, "other_border_max_radius_mm": OTHER_BORDER_MAX_RADIUS_MM},
        "component_filter": {"min_pixels": MIN_COMPONENT_PIXELS, "min_fraction_of_largest": MIN_COMPONENT_FRACTION},
        "global_reflection": global_reflection,
        "global_reflection_rationale": "reflection is one shared imaging/annotation convention across "
            "all 8 animals, chosen once from the pooled per-reflection objective sum rather than per "
            "animal, to avoid weak single-animal border evidence flipping the mirror independently",
        "pooled_objective_by_reflection": pooled_objective,
        "n_animals_agreeing_with_pooled_choice": n_agree_with_own_argmin,
        "n_animals_total": len(ANIMALS),
        "per_animal": [
            {k: v for k, v in f.items()
             if k not in ("border_dist", "border_xy", "probe_xy", "contour_xy", "other_border_xy", "apex_xy", "data")}
            for f in fits
        ],
        "site_semantics": SITE_SEMANTICS,
        "caveats": [
            "site_semantics differs by animal (see package README): only realized_probe_location "
            "animals (813810, 816308, 817335, 810531) reflect an actually-recorded site; "
            "intended_penetration_target (815152, 810532) and approximate_opening_center "
            "(816305, 817334) may not match the true recording location.",
            "retinotopic_contours layer has no known degree calibration in this extraction; used "
            "only as a soft V1-containment term and for visual QA, never as a hard geometric target.",
            "border extraction quality varies per animal (see each animal's package README); "
            "border_median_residual_px in probe_anatomical_position.csv is the primary per-animal "
            "fit-quality indicator.",
            "Animal-side V1-loop isolation is a per-direction nearest-border ring around the probe "
            "centroid (capped at MAX_RING_RADIUS_MM); it can only be as complete as the underlying "
            "extraction, so a direction with no border traced nearby stays empty rather than "
            "picking up a farther HVA border -- v1_isolation_ring_coverage (fraction of 72 angular "
            "bins with a nearby hit) is the per-animal completeness indicator, and "
            "Figure_per_animal_border_fit_QA.png should be visually checked, not just this number.",
            f"{len(HAND_TRACED_ANIMALS)}/{len(ANIMALS)} animals ({sorted(HAND_TRACED_ANIMALS)}) use a "
            "human hand-traced V1 ring (reference/<animal>_layers.svg) instead of the automated "
            "isolate_nearest_ring_border extraction -- see ring_source per animal/probe; the "
            "remaining animals still rely on the automated extraction and its known limitations.",
            "scale is now a FREE fit parameter (see module docstring), not derived from each "
            "animal's photo px/mm calibration -- that calibration is disregarded as unreliable per "
            "user direction. scale_relative_to_naive_px_per_mm in probe_anatomical_position.csv "
            "reports the fitted scale as a ratio to what the old (disregarded) calibration would "
            "have implied, purely for reference.",
            "rotation is searched only within +/-ROTATION_WINDOW_DEG of an apex-anchored estimate "
            "(not the full +/-180deg range) -- this assumes detect_ring_apex correctly finds V1's "
            "anteromedial corner on both sides; apex_to_probe_a_mm is the per-animal plausibility "
            "check (probe A is anatomically expected to sit near this corner).",
        ],
    }
    (OUTPUT / "area_border_registration_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # QA figure: all animals' fitted border clouds + probes over the Zhuang boundary. Full network
    # shown faint for context; the isolated V1-only border (what was actually fit against) in bold.
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.contour(template["boundary"].astype(float), levels=[0.5], colors="#cccccc", linewidths=0.5)
    ax.contour(visp_border_mask.astype(float), levels=[0.5], colors="#222222", linewidths=1.1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fits)))
    for fit, color in zip(fits, colors):
        ax.scatter(fit["border_xy"][:, 0], fit["border_xy"][:, 1], s=3, color=color, alpha=0.5, zorder=2)
        if len(fit["other_border_xy"]):
            ax.scatter(fit["other_border_xy"][:, 0], fit["other_border_xy"][:, 1], s=2, color=color, alpha=0.2, marker="+", zorder=1)
        if len(fit["contour_xy"]):
            ax.scatter(fit["contour_xy"][:, 0], fit["contour_xy"][:, 1], s=2, color=color, alpha=0.15, marker="x", zorder=1)
        ax.scatter(fit["probe_xy"][:, 0], fit["probe_xy"][:, 1], s=55, color=color, edgecolors="white",
                   linewidths=0.8, zorder=3, label=f"{fit['animal']} ({fit['data']['site_semantics'][:8]})")
        ax.scatter(*fit["apex_xy"], s=90, marker="^", color=color, edgecolors="black", linewidths=0.8, zorder=4)
    ax.scatter(*zhuang_apex_xy, s=180, marker="*", color="red", edgecolors="black", linewidths=1.0, zorder=5,
               label="Zhuang V1 apex (anchor)")
    for acronym, (x, y) in AREA_SEEDS_XY.items():
        ax.text(x, y, acronym.replace("VIS", ""), ha="center", va="center", fontsize=10, color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
    ax.set(title="MouseV2 probes: anatomy (area-border) registration to Zhuang common map\n"
                 "dots = V1-ring border (primary), + = other border material (weak rotation anchor), "
                 "x = retinotopic-contour points (support only), circles = probes, triangles = apex",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_area_borders_registered_to_zhuang.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Small multiples: one panel per animal, border fit only (visual QA of shape match).
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for fit, ax in zip(fits, axes.ravel()):
        ax.contour(template["boundary"].astype(float), levels=[0.5], colors="#dddddd", linewidths=0.4)
        ax.contour(visp_border_mask.astype(float), levels=[0.5], colors="#666666", linewidths=0.9)
        if len(fit["other_border_xy"]):
            ax.scatter(fit["other_border_xy"][:, 0], fit["other_border_xy"][:, 1], s=3, color="#f4a261", marker="+", alpha=0.6)
        ax.scatter(fit["border_xy"][:, 0], fit["border_xy"][:, 1], s=4, color="#2864a8")
        ax.scatter(fit["probe_xy"][:, 0], fit["probe_xy"][:, 1], s=45, color="#d73027", edgecolors="white", linewidths=0.6)
        ax.scatter(*fit["apex_xy"], s=70, marker="^", color="#2864a8", edgecolors="black", linewidths=0.7, zorder=4)
        ax.scatter(*zhuang_apex_xy, s=130, marker="*", color="red", edgecolors="black", linewidths=0.8, zorder=5)
        isolation_flag = "" if fit["v1_isolation_ok"] else " [ISOLATION FAILED]"
        apex_a_note = f", apex-A {fit['apex_to_probe_a_mm']:.2f}mm" if fit["apex_to_probe_a_mm"] is not None else ""
        ax.set_title(f"{fit['animal']} ({fit['data']['site_semantics']}){isolation_flag}\n"
                     f"median residual {fit['border_median_residual_px']:.1f}px, reflection {fit['reflection']:+d}, "
                     f"scale {fit['scale_relative_to_naive_px_per_mm']:.2f}x naive{apex_a_note}",
                     fontsize=8)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_per_animal_border_fit_QA.png", dpi=150)
    plt.close(fig)

    if comparison is not None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for semantics, group in comparison.groupby("site_semantics"):
            ax.scatter(group.zhuang_pixel_distance_px, group.retinotopic_vector_error_deg, label=semantics, s=45, alpha=0.85)
        ax.set(title="Anatomy-based vs. RF-value-based registration agreement\n(independent methods, per probe)",
               xlabel="Zhuang pixel distance between the two inferred positions (px)",
               ylabel="Retinotopic value difference at the two positions (deg)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT / "Figure_anatomy_vs_rf_agreement.png", dpi=170)
        plt.close(fig)

    print(f"\n{OUTPUT / 'Figure_area_borders_registered_to_zhuang.png'}")
    print(f"{OUTPUT / 'Figure_per_animal_border_fit_QA.png'}")


if __name__ == "__main__":
    main()
