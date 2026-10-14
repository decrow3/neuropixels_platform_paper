#!/usr/bin/env python3
"""Build a one-row schematic contrasting published and proposed visual hierarchies."""

from __future__ import annotations

from pathlib import Path
import csv
import gzip
import re

import matplotlib as mpl
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = ROOT / "MouseV2-Figure5/sources/reference/allen_ccf_2017_surface/top.nrrd"
ATLAS_LABELS_PATH = (
    ROOT / "MouseV2-Figure5/sources/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt"
)
PROBE_ENTRIES_PATH = ROOT / "MouseV2-Figure5/sources/mousev2_ccf_surface_entry_points.csv"
CONNECTION_DENSITY_PATH = ROOT / "MouseV2-Figure5/sources/reference/allen_harris_2019_connectivity/normalized_connection_density_ipsi_ctx.csv"
HIERARCHY_AREAS = {"V1": "VISp", "LM": "VISl", "RL": "VISrl", "AL": "VISal", "PM": "VISpm", "AM": "VISam"}
MAX_CONNECTION_WIDTH_PT = 2.4
HARRIS_SCORES_PATH = CONNECTION_DENSITY_PATH.parent / "harris_visualmedial_scores.csv"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Figure 4 colors.
AREA_COLORS = {
    "LM": "#4e73ae",
    "RL": "#65b2c9",
    "AL": "#cab778",
    "PM": "#db8457",
    "AM": "#c24f54",
}
PROBE_COLORS = {"A": "#d73027", "E": "#fc8d59", "C": "#1a9850", "B": "#4575b4"}
V1_COLOR = "#777777"
V2_COLOR = "#a8adb3"
INK = "#202124"
MUTED = "#6f7378"


def read_nrrd(path: Path) -> np.ndarray:
    """Read the gzip-encoded 2-D uint16 atlas without an extra dependency."""
    header, payload = path.read_bytes().split(b"\n\n", 1)
    sizes_match = re.search(rb"^sizes:\s+([0-9 ]+)$", header, re.MULTILINE)
    if sizes_match is None or b"type: uint16" not in header or b"encoding: gzip" not in header:
        raise ValueError(f"Unsupported NRRD header in {path}")
    sizes = tuple(int(value) for value in sizes_match.group(1).split())
    return np.frombuffer(gzip.decompress(payload), dtype="<u2").reshape(sizes, order="F")


def read_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    pattern = re.compile(
        r'^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+"([^"]+)"'
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    return labels

# Anatomical scores, not CCG-derived functional scores. Cortical subset of
# AllenInstitute/neuropixels_platform_paper, Figure4/Figure4.py: regions/hierScore.
# Preserve the original eight-area coordinate system (LGd and LP omitted here).
SIEGLE_SCORES = {
    "V1": -0.35733209934482374,
    "LM": -0.09388855125761343,
    "RL": -0.05987132463908328,
    "AL": 0.15221797920142832,
    "PM": 0.32766807486511995,
    "AM": 0.440986074378801,
}

ATLAS_AREAS = (
    "VISp", "VISal", "VISam", "VISl", "VISpl",
    "VISpm", "VISa", "VISli", "VISrl",
)
ATLAS_DISPLAY_LABELS = {
    "VISp": "V1", "VISal": "AL", "VISam": "AM", "VISl": "LM",
    "VISpl": "P", "VISpm": "PM", "VISa": "A", "VISli": "LI",
    "VISrl": "RL",
}
ATLAS_FIGURE4_COLORS = {
    "VISl": AREA_COLORS["LM"], "VISrl": AREA_COLORS["RL"],
    "VISal": AREA_COLORS["AL"], "VISpm": AREA_COLORS["PM"],
    "VISam": AREA_COLORS["AM"],
}


def clean_axis(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _atlas_index(labels, acronym):
    return next(index for index, label in labels.items() if label == acronym)


def draw_atlas_area_map(ax, atlas, labels, *, merged=False):
    """Render the genuine Allen CCFv3 dorsal visual-area masks used in Figure 4A."""
    rgba = np.ones((*atlas.shape, 4), dtype=float)
    rgba[..., 3] = 0.0
    visual_union = np.zeros(atlas.shape, dtype=bool)
    v2_union = np.zeros(atlas.shape, dtype=bool)

    for acronym in ATLAS_AREAS:
        mask = atlas == _atlas_index(labels, acronym)
        visual_union |= mask
        if acronym != "VISp":
            v2_union |= mask
        if merged:
            color = "#f8f8f6" if acronym == "VISp" else V2_COLOR
        else:
            color = (
                "#f8f8f6" if acronym == "VISp"
                else ATLAS_FIGURE4_COLORS.get(acronym, "#d8dade")
            )
        rgba[mask] = mpl.colors.to_rgba(color, 1.0)

    ax.imshow(
        rgba.transpose(1, 0, 2), origin="upper", extent=(0, 11.4, 13.2, 0),
        interpolation="nearest", zorder=0,
    )
    ml = np.arange(atlas.shape[0]) * 0.01
    ap = np.arange(atlas.shape[1]) * 0.01

    if merged:
        ax.contour(ml, ap, v2_union.T.astype(float), levels=[0.5], colors=INK,
                   linewidths=1.15, zorder=2)
        v1_mask = atlas == _atlas_index(labels, "VISp")
        ax.contour(ml, ap, v1_mask.T.astype(float), levels=[0.5], colors=INK,
                   linewidths=1.35, zorder=3)
        yy, xx = np.where(v1_mask.T)
        ax.text(np.median(xx) * 0.01, np.median(yy) * 0.01, "V1",
                ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)
        ax.text(3.72, 7.55, "V2", ha="center", va="center",
                fontsize=14, fontweight="bold", zorder=4)
    else:
        for acronym in ATLAS_AREAS:
            mask = atlas == _atlas_index(labels, acronym)
            ax.contour(ml, ap, mask.T.astype(float), levels=[0.5], colors=INK,
                       linewidths=1.15 if acronym == "VISp" else 0.72, zorder=2)
            yy, xx = np.where(mask.T)
            if len(xx):
                if acronym == "VISli":
                    ax.annotate("LI", xy=(np.median(xx) * 0.01, np.median(yy) * 0.01),
                                xytext=(-16, 0), textcoords="offset points", ha="right", va="center",
                                fontsize=12, color=INK,
                                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
                    continue
                ax.text(
                    np.median(xx) * 0.01, np.median(yy) * 0.01,
                    ATLAS_DISPLAY_LABELS[acronym], ha="center", va="center",
                    fontsize=12, fontweight="bold" if acronym == "VISp" else "normal",
                    color=INK, zorder=3,
                )

    # Frame all included atlas masks, including the anterior border of VISa.
    # Use the same limits for the separate-area and merged views.
    xx, yy = np.where(visual_union)
    margin_mm = 0.13
    ax.set_xlim(xx.min() * 0.01 - margin_mm, xx.max() * 0.01 + margin_mm)
    ax.set_ylim(yy.max() * 0.01 + margin_mm, yy.min() * 0.01 - margin_mm)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_probe_locations(ax, atlas, labels, *, subjects=None, location_names=False):
    """Plot equal-animal mean superficial VISp-contact positions in dorsal CCF."""
    with PROBE_ENTRIES_PATH.open() as stream:
        entries = list(csv.DictReader(stream))
    if subjects is not None:
        subjects = {str(int(value)) for value in subjects}
        entries = [row for row in entries if row["subject_id"] in subjects]
        if {row["subject_id"] for row in entries} != subjects:
            raise ValueError("Missing probe anatomy for an analysis animal")
    positions = []
    for probe in ("A", "E", "C", "B"):
        rows = [row for row in entries if row["probe"] == f"Probe{probe}"]
        if not rows or len({row["subject_id"] for row in rows}) != len(rows):
            raise ValueError(f"Expected one entry per animal for probe {probe}")
        ml = np.mean([float(row["ccf_ml_um"]) for row in rows])
        ap = np.mean([float(row["ccf_ap_um"]) for row in rows])
        region = labels[int(atlas[int(ml / 10), int(ap / 10)])]
        if region != "VISp":
            raise ValueError(f"Mean probe {probe} is outside the VISp surface mask")
        ax.scatter(ml / 1000, ap / 1000, s=95, color=PROBE_COLORS[probe],
                   edgecolor="white", linewidth=1.2, zorder=6)
        if location_names:
            name = {"A": "Anterior", "E": "Lateral", "C": "Posterior", "B": "Medial"}[probe]
            offset, ha, va = {"A": ((-10, 36), "center", "bottom"),
                              "E": ((-20, 0), "right", "center"),
                              "C": ((0, -18), "center", "top"),
                              "B": ((18, 0), "left", "center")}[probe]
            ax.annotate(name, xy=(ml / 1000, ap / 1000), xytext=offset,
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=11, color=INK, zorder=7,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=.6, shrinkA=1, shrinkB=4))
        else:
            offset = {"A": (0, -17), "E": (14, 0), "C": (14, 0), "B": (-14, 0)}[probe]
            ax.annotate(probe, xy=(ml / 1000, ap / 1000), xytext=offset,
                        textcoords="offset points", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=INK, zorder=7)
        positions.append(dict(probe=probe, n_animals=len(rows), ccf_ml_um=ml,
                              ccf_ap_um=ap, surface_area=region))
    return positions


def hierarchy_connections():
    """All directed non-self edges; Allen regional matrix rows are sources.

    Harris Fig. 1e supplies normalized connection density, used for Fig. 6e
    widths. Preserve the measured asymmetry and use one linear width scale.
    """
    with CONNECTION_DENSITY_PATH.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        index_column = reader.fieldnames[0]
        matrix = {row[index_column]: row for row in reader}
    edges = []
    for source, source_atlas in HIERARCHY_AREAS.items():
        for target, target_atlas in HIERARCHY_AREAS.items():
            if source == target:
                continue
            density = float(matrix[source_atlas][target_atlas])
            if not np.isfinite(density) or density < 0:
                raise ValueError(f"Invalid anatomical density: {source}->{target}")
            edges.append(dict(source=source, target=target,
                              source_atlas=source_atlas, target_atlas=target_atlas,
                              normalized_connection_density=density))
    maximum = max(edge["normalized_connection_density"] for edge in edges)
    if maximum <= 0:
        raise ValueError("No positive anatomical connections")
    for edge in edges:
        edge["relative_density"] = edge["normalized_connection_density"] / maximum
        edge["linewidth_pt"] = MAX_CONNECTION_WIDTH_PT * edge["relative_density"]
    return edges


def hierarchy_scores(version="siegle2021"):
    if version == "siegle2021":
        return SIEGLE_SCORES.copy()
    if version != "harris2019":
        raise ValueError(f"Unknown hierarchy version: {version}")
    with HARRIS_SCORES_PATH.open(newline="") as stream:
        scores = {row["area"]: float(row["hierarchy_score"]) for row in csv.DictReader(stream)}
    if set(scores) != set(HIERARCHY_AREAS) or not all(np.isfinite(list(scores.values()))):
        raise ValueError("Incomplete/nonfinite Harris hierarchy scores")
    return scores


def draw_published_hierarchy(ax, version="siegle2021", *, show_direction=True, compact=False, axis_x=-1.23):
    # Common limits permit a direct comparison without shifting or rescaling.
    clean_axis(ax, (-1.90, 1.30), (-0.68, 0.65))
    order = list(HIERARCHY_AREAS)
    scores = hierarchy_scores(version)
    axis_top = max(scores.values()) + .025 if compact else .57
    top_label_y = axis_top + .035 if compact else .60
    if compact:
        ax.set_ylim(-.68, axis_top + .07)
    # Keep all nodes on one column, including the nearly overlapping LM/RL.
    xs = {area: 0.10 for area in order}
    for edge in sorted(hierarchy_connections(), key=lambda row: row["linewidth_pt"], reverse=True):
        source, target = edge["source"], edge["target"]
        if edge["normalized_connection_density"] == 0:
            continue
        y_source, y_target = scores[source], scores[target]
        side = -1 if y_target > y_source else 1
        bend = .20 + 1.05 * abs(y_target - y_source)
        # Both control points stay on the same side of the node column:
        # ascending projections on the left, descending projections on the right.
        path = MplPath([(xs[source], y_source), (xs[source] + side * bend, y_source),
                        (xs[target] + side * bend, y_target), (xs[target], y_target)],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, fill=False,
                              edgecolor=AREA_COLORS.get(source, V1_COLOR),
                              linewidth=edge["linewidth_pt"], alpha=.70, zorder=1,
                              gid=f"anatomical-connection-{source}-{target}"))

    directions = [(-.77, -.16, .18, "Feedforward"), (1.06, .18, -.16, "Feedback")] if show_direction else []
    for x, start, end, label in directions:
        ax.annotate("", xy=(x, end), xytext=(x, start),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK))
        ax.text(x + (-.12 if x < 0 else .12), .01, label, rotation=90 if x < 0 else -90,
                ha="center", va="center", fontsize=10)
    ax.text(.10, -.64, "Width ∝ density", ha="center", fontsize=10)
    ax.annotate("", xy=(axis_x, axis_top), xytext=(axis_x, -.58), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK))
    ax.text(-1.90, 0.02, "anatomical\nhierarchy score", rotation=90, ha="center", va="center", fontsize=12)
    ax.text(axis_x, top_label_y, "Top", ha="center", fontsize=10)
    ax.text(axis_x, -.64, "Bottom", ha="center", fontsize=10)
    for score in (-.50, -.25, 0., .25, .50):
        if score > axis_top:
            continue
        ax.plot([axis_x - .05, axis_x], [score, score], color=INK, lw=0.8)
        ax.text(axis_x - .11, score, f"{score:.2f}", fontsize=10, va="center", ha="right", color=MUTED)

    for area in order:
        color = V1_COLOR if area == "V1" else AREA_COLORS[area]
        ax.scatter(xs[area], scores[area], s=130, color=color, edgecolor=INK, linewidth=0.9, zorder=3)
        offset_y = 0
        if area in {"LM", "RL"}:
            other = "RL" if area == "LM" else "LM"
            offset_y = .045 if scores[area] > scores[other] else -.045
        ax.text(xs[area] + (-0.22 if area == "LM" else 0.22), scores[area] + offset_y, area,
                fontsize=12, va="center", ha="right" if area == "LM" else "left",
                bbox=dict(facecolor="white", edgecolor="none", pad=.15), zorder=4)
