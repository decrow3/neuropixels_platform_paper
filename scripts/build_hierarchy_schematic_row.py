#!/usr/bin/env python3
"""Build a one-row schematic contrasting published and proposed visual hierarchies."""

from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_mousev2_tracks_to_ccf_surface import read_labels, read_nrrd


OUTPUT_STEM = ROOT / "Figure3" / "Hierarchy_schematic_row"
ATLAS_PATH = ROOT / "data/reference/allen_ccf_2017_surface/top.nrrd"
ATLAS_LABELS_PATH = (
    ROOT / "data/reference/allen_ccf_2017_surface/labelDescription_ITKSNAPColor.txt"
)
PROBE_ENTRIES_PATH = ROOT / "artifacts/figure3/06s_mousev2_ccf_surface_projection/mousev2_ccf_surface_entry_points.csv"

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


def closed_patch(points, **kwargs):
    vertices = list(points) + [points[0]]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), **kwargs)


def clean_axis(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def panel_label(ax, letter, title, subtitle=None):
    ax.text(0.00, 1.12, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.text(0.11, 1.12, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def draw_area_map(ax, merged=False):
    """Draw a deliberately schematic top view, shared between panels A and C."""
    clean_axis(ax, (-1.18, 1.18), (-1.12, 1.14))

    outer = [
        (-0.98, -0.30), (-1.05, 0.14), (-0.82, 0.63), (-0.35, 0.93),
        (0.17, 1.00), (0.66, 0.72), (0.93, 0.28), (0.84, -0.25),
        (0.52, -0.76), (0.02, -0.96), (-0.52, -0.78),
    ]
    v1 = [
        (-0.46, -0.29), (-0.55, 0.10), (-0.37, 0.47), (-0.04, 0.66),
        (0.32, 0.54), (0.48, 0.20), (0.40, -0.26), (0.12, -0.54),
        (-0.25, -0.51),
    ]

    if merged:
        ax.add_patch(closed_patch(outer, facecolor=V2_COLOR, edgecolor=INK, linewidth=1.4))
        ax.add_patch(closed_patch(v1, facecolor="white", edgecolor=INK, linewidth=1.4))
        ax.text(0.42, 0.62, "V2", fontsize=12, fontweight="bold", ha="center", va="center")
        ax.text(-0.06, 0.01, "V1", fontsize=12, fontweight="bold", ha="center", va="center")
        return

    # Five cortical HVAs used in Figure 4. Boundaries are schematic, not atlas contours.
    regions = {
        "LM": [(-0.98, -0.30), (-1.05, 0.14), (-0.82, 0.63), (-0.37, 0.47), (-0.55, 0.10), (-0.46, -0.29), (-0.52, -0.78)],
        "RL": [(-0.82, 0.63), (-0.35, 0.93), (0.17, 1.00), (0.08, 0.65), (-0.04, 0.66), (-0.37, 0.47)],
        "AM": [(0.17, 1.00), (0.66, 0.72), (0.93, 0.28), (0.48, 0.20), (0.32, 0.54), (0.08, 0.65)],
        "PM": [(0.93, 0.28), (0.84, -0.25), (0.52, -0.76), (0.12, -0.54), (0.40, -0.26), (0.48, 0.20)],
        "AL": [(0.52, -0.76), (0.02, -0.96), (-0.52, -0.78), (-0.46, -0.29), (-0.25, -0.51), (0.12, -0.54)],
    }
    for area, points in regions.items():
        ax.add_patch(
            closed_patch(points, facecolor=AREA_COLORS[area], edgecolor=INK, linewidth=1.15, alpha=0.88)
        )
    ax.add_patch(closed_patch(v1, facecolor="white", edgecolor=INK, linewidth=1.35))

    label_xy = {
        "LM": (-0.73, 0.02), "RL": (-0.29, 0.73), "AM": (0.53, 0.61),
        "PM": (0.64, -0.25), "AL": (-0.03, -0.73), "V1": (-0.06, 0.01),
    }
    for area, (x, y) in label_xy.items():
        ax.text(x, y, area, fontsize=9, fontweight="bold", ha="center", va="center")


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


def draw_published_hierarchy(ax):
    clean_axis(ax, (-1.05, 1.10), (-0.49, 0.53))
    order = ["V1", "LM", "RL", "AL", "PM", "AM"]
    xs = {"V1": 0.00, "LM": -0.10, "RL": 0.10, "AL": -0.09, "PM": 0.08, "AM": 0.00}

    # A light subset of the dense reciprocal connectivity in the source schematic.
    connections = [
        ("V1", "LM"), ("V1", "RL"), ("V1", "AL"), ("V1", "PM"), ("V1", "AM"),
        ("LM", "AL"), ("LM", "PM"), ("RL", "AL"), ("RL", "AM"),
        ("AL", "PM"), ("AL", "AM"), ("PM", "AM"),
    ]
    for idx, (low, high) in enumerate(connections):
        side = -1 if idx % 2 == 0 else 1
        start = (xs[low], SIEGLE_SCORES[low])
        end = (xs[high], SIEGLE_SCORES[high])
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-",
                connectionstyle=f"arc3,rad={side * (0.48 + 0.025 * idx):.3f}",
                color=AREA_COLORS.get(high, V1_COLOR),
                linewidth=1.15,
                alpha=0.50,
                zorder=1,
            )
        )

    ax.annotate("", xy=(-0.86, 0.48), xytext=(-0.86, -0.43), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK))
    ax.text(-1.06, 0.02, "anatomical\nhierarchy score", rotation=90, ha="center", va="center", fontsize=12)
    for score in (-0.4, 0.0, 0.4):
        ax.plot([-0.88, -0.82], [score, score], color=INK, lw=0.8)
        ax.text(-0.77, score, f"{score:.1f}", fontsize=12, va="center", ha="left", color=MUTED)

    for area in order:
        color = V1_COLOR if area == "V1" else AREA_COLORS[area]
        ax.scatter(xs[area], SIEGLE_SCORES[area], s=180, color=color, edgecolor=INK, linewidth=0.9, zorder=3)
        ax.text(xs[area] + (-0.19 if area == "LM" else 0.19), SIEGLE_SCORES[area], area,
                fontsize=12, va="center", ha="right" if area == "LM" else "left")



def draw_spread_bracket(ax, values, label):
    """Mark the illustrative center-to-center vertical extent, not an error bar."""
    low, high = min(values), max(values)
    x = 4.85
    ax.plot([x - 0.14, x, x, x - 0.14], [low, low, high, high],
            color=INK, lw=1.3, solid_capstyle="butt", zorder=4)
    ax.text(x + 0.12, (low + high) / 2, label, fontsize=12,
            ha="left", va="center", linespacing=1.15)


def draw_proposed_hierarchy(ax, *, spread_scale=1, probe_labels=None):
    clean_axis(ax, (-0.42, 6.25), (0.40, 3.00))
    ax.annotate("", xy=(-0.28, 2.87), xytext=(-0.28, 0.62), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK))
    ax.text(-0.40, 1.73, "relative hierarchy", rotation=90, ha="center", va="center", fontsize=12)

    # Tier guides emphasize two broad levels without assigning a calibrated distance.
    ax.plot([0.02, 4.50], [2.42, 2.42], color="#aeb3b8", lw=1.0, ls=(0, (3, 3)), zorder=0)
    ax.plot([0.02, 4.10], [1.00, 1.00], color="#aeb3b8", lw=1.0, ls=(0, (3, 3)), zorder=0)

    hva_y = {"LM": 2.37, "RL": 2.39, "AL": 2.43, "PM": 2.48, "AM": 2.53}
    hva_y = {key: 2.43 + spread_scale * (value - 2.43) for key, value in hva_y.items()}
    for x, area in enumerate(["LM", "RL", "AL", "PM", "AM"]):
        ax.scatter(x + 0.26, hva_y[area], s=170, color=AREA_COLORS[area], edgecolor=INK, linewidth=0.85, zorder=3)
        ax.text(x + 0.26, hva_y[area] + 0.23, area, fontsize=12, ha="center", va="bottom")

    probe_y = {"A": 0.96, "E": 1.00, "C": 1.10, "B": 1.04}
    probe_y = {key: 1.03 + spread_scale * (value - 1.03) for key, value in probe_y.items()}
    for x, probe in enumerate(["A", "E", "C", "B"]):
        probe_x = x * (1.2 if probe_labels else 1) + .55
        ax.scatter(probe_x, probe_y[probe], s=155, color=PROBE_COLORS[probe], edgecolor=INK, linewidth=0.85, zorder=3)
        ax.text(probe_x, .76 if probe_labels else probe_y[probe] - 0.23,
                probe_labels.get(probe, probe) if probe_labels else probe,
                fontsize=11 if probe_labels else 12, ha="right" if probe_labels else "center", va="top",
                rotation=22 if probe_labels else 0)

    draw_spread_bracket(ax, hva_y.values(), "HVA\nspread")
    draw_spread_bracket(ax, probe_y.values(), "V1\nspread")


def main():
    atlas = read_nrrd(ATLAS_PATH)
    labels = read_labels(ATLAS_LABELS_PATH)
    fig = plt.figure(figsize=(16, 3.7), facecolor="white")
    positions = [(0.055, 0.155), (0.247, 0.155), (0.465, 0.155), (0.655, 0.33)]
    axes = [fig.add_axes([left, 0.08, width, 0.74]) for left, width in positions]
    for (left, _), letter, title in zip(positions, "ABCD", [
        "Visual area borders", "Published hierarchy", "Two-area view", "Proposed hierarchy"
    ]):
        fig.text(left, 0.92, f"{letter}  {title}", fontsize=15, fontweight="bold", va="bottom")

    draw_atlas_area_map(axes[0], atlas, labels, merged=False)

    draw_published_hierarchy(axes[1])
    axes[1].set_aspect("auto")

    draw_atlas_area_map(axes[2], atlas, labels, merged=True)
    draw_probe_locations(axes[2], atlas, labels)

    draw_proposed_hierarchy(axes[3])
    axes[3].set_aspect("auto")

    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
