#!/usr/bin/env python3
"""Canonical Figure 5 renderer, flattened from the accepted v17 layout."""

from __future__ import annotations

import os
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]

import panels as base


OUTPUT = ROOT / "MouseV2-Figure5"
STEM = OUTPUT / "Figure5"
VECTOR_SOURCE = OUTPUT / "sources/Original_handmade_cortical_vectors.svg"
VARIANCE_STATS = ROOT / "MouseV2-Figure5/data/estimates_and_tests.csv"
CAPTION_SOURCE = ROOT / "MouseV2-Figure5/sources/Figure5_caption.md"
ARTBOARD_WIDTH = 852.648
ARTBOARD_HEIGHT = 709.0
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
SVG_NS = "{http://www.w3.org/2000/svg}"


class ExactPathPen(BasePen):
    def __init__(self):
        super().__init__(None)
        self.vertices, self.codes, self.start = [], [], None

    def _moveTo(self, point):
        self.start = point
        self.vertices.append(point)
        self.codes.append(MplPath.MOVETO)

    def _lineTo(self, point):
        self.vertices.append(point)
        self.codes.append(MplPath.LINETO)

    def _curveToOne(self, a, b, c):
        self.vertices.extend([a, b, c])
        self.codes.extend([MplPath.CURVE4] * 3)

    def _qCurveToOne(self, a, b):
        self.vertices.extend([a, b])
        self.codes.extend([MplPath.CURVE3] * 2)

    def _closePath(self):
        self.vertices.append(self.start)
        self.codes.append(MplPath.CLOSEPOLY)

    def _endPath(self):
        pass


def vector_groups():
    root = ET.parse(VECTOR_SOURCE).getroot()
    if len(root) != 8:
        raise ValueError("Canonical cortical-vector source has an unexpected structure")
    inter, intra = root[1], root[2]
    inter_dots = list(root[3:8])
    assert len(inter.findall(".//" + SVG_NS + "path")) == 6
    assert len(intra.findall(".//" + SVG_NS + "path")) == 6
    assert all(element.tag == SVG_NS + "circle" for element in inter_dots)
    return inter, intra, inter_dots


def draw_cortical_vector(ax, within: bool) -> None:
    inter, intra, inter_dots = vector_groups()
    group = intra if within else inter
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim((518, 665) if within else (391, 538))
    ax.set_ylim(146, 14)
    for element in group.findall(".//" + SVG_NS + "path"):
        assert not element.get("transform")
        pen = ExactPathPen()
        parse_path(element.attrib["d"], pen)
        ax.add_patch(PathPatch(MplPath(pen.vertices, pen.codes), facecolor="#231f20", edgecolor="none"))
    dots = group.findall(".//" + SVG_NS + "circle") if within else inter_dots
    categories = ["A", "E", "B", "C"] if within else ["LM", "AL", "RL", "AM", "PM"]
    palette = base.results.V1_COLORS if within else base.results.HVA_COLORS
    for element, category in zip(dots, categories):
        ax.add_patch(Circle((float(element.get("cx")), float(element.get("cy"))),
                            float(element.get("r")), facecolor=palette[category], edgecolor="none"))
    for element in group.findall(".//" + SVG_NS + "text"):
        label = "".join(element.itertext())
        x, y = map(float, re.search(r"translate\(([^)]+)\)", element.get("transform"))[1].split())
        ax.text(x, y, label, fontsize=11 if label == "V1" else 10,
                family="Liberation Sans", ha="left", va="baseline",
                color="#231f20" if label == "V1" else palette[label])
    ax.text(591.5 if within else 464.5, 145, "Intra-V1" if within else "Inter-HVA",
            fontsize=12, family="Liberation Sans", ha="center", va="top")


def visible_text_outside(fig) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        if text.axes is not None and not text.axes.axison and text not in text.axes.texts:
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (
            box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height
        ):
            outside.append(text.get_text())
    return outside


def apply_publication_style(fig) -> None:
    for text in fig.findobj(Text):
        text.set_fontfamily("Liberation Sans")
        text.set_fontweight("normal")
        if text.get_gid() == "panel-letter":
            text.set_fontsize(32)
        elif text.get_gid() == "panel-heading":
            text.set_fontsize(16)
        else:
            text.set_fontsize(text.get_fontsize() * 1.55)
    for line in fig.findobj(Line2D):
        line.set_linewidth(line.get_linewidth() * 1.4)
        line.set_markersize(line.get_markersize() * 1.4)
        line.set_markeredgewidth(line.get_markeredgewidth() * 1.4)
    for collection in fig.findobj(PathCollection):
        collection.set_sizes(collection.get_sizes() * 1.4**2)
        collection.set_linewidths(collection.get_linewidths() * 1.4)
    for collection in fig.findobj(LineCollection):
        collection.set_linewidths(collection.get_linewidths() * 1.4)
    for ax in fig.axes:
        ax.tick_params(pad=2.5, length=3, width=.7)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        ax.set_title(ax.get_title(), fontsize=14, pad=7, fontfamily="Liberation Sans")
        for spine in ax.spines.values():
            spine.set_linewidth(.7)


def install_cortical_vectors(fig) -> None:
    fig.axes[0].set_position([.055, .725, .195, .22])
    fig.axes[2].set_position([.535, .725, .185, .22])
    draw_cortical_vector(fig.axes[0], False)
    draw_cortical_vector(fig.axes[2], True)
    for text in fig.findobj(Text):
        label = text.get_text()
        label = label.replace("Within-area benchmark", "Intra-V1 benchmark")
        label = label.replace("Across HVAs", "Inter-HVA").replace("Within V1", "Intra-V1")
        label = label.replace("within V1", "intra-V1").replace("Within\nV1", "Intra-V1")
        label = label.replace("Across\nHVAs", "Inter-HVA")
        text.set_text(label)
    for ax in (fig.axes[8], fig.axes[13], fig.axes[18]):
        ax.set_xticks([0, 1.2, 2.1, 3.5],
                      ["HVAs\n− V1", "Intra-V1", "Inter-HVA", "Inter-HVA\n− intra-V1"])


def make_results_taller(fig) -> None:
    old_height, new_height = 574.601, 650.0
    delta = new_height - old_height
    fig.set_size_inches(ARTBOARD_WIDTH / 72, old_height / 72)
    positions = [ax.get_position().frozen() for ax in fig.axes]
    text_positions = [(text, text.get_position()) for text in fig.texts]
    fig.set_size_inches(ARTBOARD_WIDTH / 72, new_height / 72)
    for index, (ax, position) in enumerate(zip(fig.axes, positions)):
        if index < 4:
            y = (position.y0 * old_height + delta) / new_height
            height = position.height * old_height / new_height
        else:
            row = (index - 4) // 5
            y = (position.y0 * old_height + delta - (row + 1) * delta / 3) / new_height
            height = (position.height * old_height + delta / 3) / new_height
        ax.set_position([position.x0, y, position.width, height])
    for text, (x, y) in text_positions:
        text.set_position((x, (y * old_height + delta) / new_height))
    for ax in (fig.axes[8], fig.axes[13], fig.axes[18]):
        ax.set_xticks([0, 1.2, 2.1, 3.5],
                      ["HVAs\n− V1", "Intra-\nV1", "Inter-\nHVA", "Inter-HVA\n− intra-V1"])


def split_panel_e(fig) -> None:
    gap, bottom = 24.0, 35.0
    old_axes = list(fig.axes)
    positions = [ax.get_position().frozen() for ax in old_axes]
    fig.set_size_inches(ARTBOARD_WIDTH / 72, ARTBOARD_HEIGHT / 72)
    for index, (ax, position) in enumerate(zip(old_axes, positions)):
        shift = gap + bottom if index < 4 else bottom
        ax.set_position([position.x0, (position.y0 * 650 + shift) / ARTBOARD_HEIGHT,
                         position.width, position.height * 650 / ARTBOARD_HEIGHT])
    for text in fig.texts:
        x, y = text.get_position()
        shift = bottom if text.get_text() in ["C", "D", "E"] else gap + bottom
        text.set_position((x, (y * 650 + shift) / ARTBOARD_HEIGHT))
    estimates = pd.read_csv(STEM.with_name(STEM.name + "_estimates.csv"))
    specs = [
        (["matched_hva_minus_v1_mean"], ["HVAs\n− V1"], "Mean\ndifference", .041),
        (["v1_sd", "hva_sd"], ["Intra-V1", "Inter-HVA"], "SD of\nmeans", .061),
        (["hva_minus_v1_sd"], ["Inter-HVA\n− intra-V1"], "SD\ndifference", .047),
    ]
    for row, index in enumerate([8, 13, 18]):
        old = old_axes[index]
        position = old.get_position().frozen()
        ylim, ticks, ylabel = old.get_ylim(), old.get_yticks(), old.get_ylabel()
        left = position.x0
        fig.delaxes(old)
        for column, (effects, labels, title, width) in enumerate(specs):
            ax = fig.add_axes([left, position.y0, width, position.height])
            left += width + .012
            ax.set_ylim(ylim)
            ax.set_yticks(ticks)
            ax.set_ylim(ylim)
            ax.set_xlim(-.6, len(effects) - .4)
            for x, effect in enumerate(effects):
                record = estimates.loc[
                    estimates.metric.eq(METRICS[row]) & estimates.effect.eq(effect)
                ].iloc[0]
                color = "#a66a20" if column == 0 else "#555555" if column == 1 else "#333333"
                marker = "s" if effect == "hva_sd" else "o"
                ax.vlines(x, record.low, record.high, color=color, lw=1.54, zorder=3)
                ax.plot(x, record.estimate, marker=marker, color=color,
                        mfc="white" if column == 1 else color, ms=4.235, lw=0, zorder=4)
            ax.set_xticks(range(len(labels)), labels, rotation=90, ha="center", va="top")
            ax.tick_params(axis="both", labelsize=10.85, pad=3, length=3, width=.7, colors="#555555")
            if row < 2:
                ax.tick_params(labelbottom=False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e4e4e4", lw=.55)
            ax.axhline(0, color="#888888", lw=.693, ls="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_color("#999999")
                spine.set_linewidth(.7)
            if column:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax.set_ylabel(ylabel, fontsize=11.16, labelpad=3, family="Liberation Sans")
            if row == 0:
                ax.set_title(title, fontsize=10.85, pad=7, family="Liberation Sans")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("Liberation Sans")


def polish_results(fig) -> None:
    pooled = [ax for ax in fig.axes if any(hasattr(text, "_sd_span") for text in ax.texts)]
    assert len(pooled) == 6
    for ax in pooled:
        ax.set_title("")
        for text in list(ax.texts):
            if not hasattr(text, "_sd_span"):
                continue
            name = "Cells SD" if text.get_text() == "cells" else "Means SD"
            text.remove()
            ax.text(-.16 if name == "Cells SD" else 1.18, .5, name,
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=10.85, family="Liberation Sans", color="#333333", clip_on=False)
    for ax in fig.axes:
        if ax.get_title() in ["Inter-HVA", "Intra-V1"]:
            ax.set_title(ax.get_title(), fontsize=16, pad=10, family="Liberation Sans")
    for text in fig.texts:
        if text.get_text() == "Visual areas hierarchy":
            text.set_text("Anatomical hierarchy")
    for text in fig.axes[3].texts:
        if "SD of means" in text.get_text():
            x, y = text.get_position()
            text.set_position((x + .23, y))
    for ax in fig.axes[4:]:
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                faces = collection.get_facecolors().copy()
                if len(faces) and np.ptp(faces[0, :3]) > .08:
                    edges = faces.copy()
                    edges[:, 3] = .65
                    collection.set_alpha(None)
                    collection.set_facecolors(faces)
                    collection.set_edgecolors(edges)
                    collection.set_linewidth(.8)
    for ax in fig.axes:
        if not ax.axison:
            continue
        labels = ax.get_xticklabels()
        if any(label.get_rotation() == 90 for label in labels):
            for label in labels:
                label.set_rotation(55)
                label.set_ha("right")
                label.set_va("top")
                label.set_rotation_mode("anchor")
            ax.tick_params(axis="x", pad=5)
    count = 0
    for line in fig.findobj(Line2D):
        if line.get_marker() in ["D", "d"]:
            line.set_markersize(line.get_markersize() * 1.45)
            line.set_markeredgecolor("white")
            line.set_markeredgewidth(.85)
            line.set_zorder(max(line.get_zorder(), 9))
            count += 1
    assert count == 54


def arrange_top_row(fig) -> tuple[dict, dict, dict]:
    cartoon_geometry = {}
    for index, name in [(0, "A_cartoon"), (2, "B_cartoon")]:
        ax = fig.axes[index]
        position = ax.get_position().frozen()
        width, height = position.width * 1.10, position.height * 1.10
        x = position.x0 - (width - position.width) / 2 - 8 / ARTBOARD_WIDTH
        y = position.y0 - (height - position.height) / 2
        ax.set_position([x, y, width, height])
        cartoon_geometry[name] = [x, y, width, height]
    ab_geometry = {}
    for index in range(4):
        ax = fig.axes[index]
        position = ax.get_position().frozen()
        scale = 1.13 if index in (0, 2) else 1.18
        height = position.height * scale
        width = position.width * (scale if index in (0, 2) else 1.08)
        shift = {0: 0, 1: -4, 2: -22, 3: -27}[index] / ARTBOARD_WIDTH
        x = position.x0 - (width - position.width) / 2 + shift
        y = position.y1 - height
        ax.set_position([x, y, width, height])
        ab_geometry[str(index)] = [x, y, width, height]
    for text in fig.texts:
        if text.get_text() in ["B", "Intra-V1 benchmark"]:
            x, y = text.get_position()
            text.set_position((x - 22 / ARTBOARD_WIDTH, y))
    for index in (1, 3):
        ax = fig.axes[index]
        position = ax.get_position().frozen()
        ax.set_position([position.x0, position.y0 + 10 / ARTBOARD_HEIGHT,
                         position.width, position.height])
    for index in (0, 2):
        ax = fig.axes[index]
        position = ax.get_position().frozen()
        width, height = position.width * 1.07, position.height * 1.07
        ax.set_position([position.x0 - (width - position.width) / 2,
                         position.y1 - height, width, height])
        for text in ax.texts:
            if text.get_text() in ["Inter-HVA", "Intra-V1"]:
                x, _ = text.get_position()
                text.set_position((x, 134))
    for text in fig.axes[1].texts:
        if text.get_text() == "Anatomical score":
            x, y = text.get_position()
            text.set_position((x - .30, y))
    ax = fig.axes[1]
    for text in ax.texts:
        if text.get_text() == "Anatomical score":
            x, y = text.get_position()
            text.set_position((x + .30, y))
    position = ax.get_position().frozen()
    ax.set_position([position.x0 - 18 / ARTBOARD_WIDTH, position.y0,
                     position.width, position.height])
    hierarchy_geometry = {"A_hierarchy": list(ax.get_position().bounds)}
    return cartoon_geometry, ab_geometry, hierarchy_geometry


def add_variance_panel(fig) -> None:
    wide_width = 1010.0
    axes = list(fig.axes)
    positions = [ax.get_position().frozen() for ax in axes]
    texts = [(text, text.get_position()) for text in fig.texts]
    fig.set_size_inches(wide_width / 72, ARTBOARD_HEIGHT / 72)
    ratio = ARTBOARD_WIDTH / wide_width
    for ax, position in zip(axes, positions):
        ax.set_position([position.x0 * ratio, position.y0, position.width * ratio, position.height])
    for text, (x, y) in texts:
        text.set_position((x * ratio, y))
    data = pd.read_csv(VARIANCE_STATS)
    controls = axes[-9::3]
    assert len(controls) == 3
    for row, (metric, reference) in enumerate(zip(METRICS, controls)):
        position = reference.get_position()
        ax = fig.add_axes([865 / wide_width, position.y0, 125 / wide_width, position.height])
        ax.set_xlim(-.55, 1.55)
        ax.set_ylim(-.55, 5.15)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        for x, population, color in [(0, "V1", "#7c63a9"), (1, "HVA", "#4b535c")]:
            record = data.loc[data.metric.eq(metric) & data.population.eq(population)].iloc[0]
            ax.vlines(x, 100 * record.low, 100 * record.high, color=color, lw=1.54, zorder=3)
            ax.plot(x, 100 * record.omega_squared, "o", color=color, ms=4.5, zorder=4)
            probability = record.permutation_p_greater
            label = (f"p = {probability:.3f}" if population == "V1" and metric == METRICS[1]
                     else "***" if probability < .001 else "**" if probability < .01
                     else "*" if probability < .05 else "n.s.")
            ax.text(x, 4.65, label, ha="center", va="center", fontsize=10.2,
                    family="Liberation Sans", color=color)
        ax.set_xticks([0, 1], ["Intra-V1", "Inter-HVA"], rotation=55,
                      ha="right", rotation_mode="anchor")
        ax.tick_params(labelsize=10.85, pad=3, length=3, width=.7, colors="#555555")
        if row < 2:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel("ω² (%)", fontsize=11.16, labelpad=3, family="Liberation Sans")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e4e4", lw=.55)
        ax.axhline(0, color="#888888", lw=.693, ls="--")
        for key in ["top", "right"]:
            ax.spines[key].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(.7)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontfamily("Liberation Sans")
        if row == 0:
            ax.set_title("Location-associated\nvariance", fontsize=12, pad=7,
                         family="Liberation Sans")
    panel_e = next(text for text in fig.texts if text.get_text() == "E")
    fig.text(829 / wide_width, panel_e.get_position()[1], "F", fontsize=32,
             va="top", family="Liberation Sans")


def compact_panels_e_f(fig) -> None:
    wide_width = 1010.0
    axes = list(fig.axes)
    positions = [ax.get_position().frozen() for ax in axes]
    texts = [(text, text.get_position()) for text in fig.texts]
    fig.set_size_inches(ARTBOARD_WIDTH / 72, ARTBOARD_HEIGHT / 72)
    for ax, position in zip(axes, positions):
        ax.set_position([position.x0 * wide_width / ARTBOARD_WIDTH, position.y0,
                         position.width * wide_width / ARTBOARD_WIDTH, position.height])
    for text, (x, y) in texts:
        text.set_position((x * wide_width / ARTBOARD_WIDTH, y))
    e_axes, f_axes = axes[-12:-3], axes[-3:]
    origin = positions[-12].x0 * wide_width
    for index, ax in enumerate(e_axes):
        position = ax.get_position().frozen()
        ax.set_position([(origin + (position.x0 * ARTBOARD_WIDTH - origin) / 2) / ARTBOARD_WIDTH,
                         position.y0, position.width / 2, position.height])
        for text in ax.get_xticklabels():
            text.set_rotation(90)
            text.set_ha("center")
            text.set_va("top")
            text.set_rotation_mode("default")
        if index < 3:
            ax.set_title(["Mean\ndifference", "SD of\nmeans", "SD\ndifference"][index],
                         fontsize=10.85, rotation=90, ha="center", va="bottom", pad=7)
    for ax in f_axes:
        position = ax.get_position().frozen()
        ax.set_position([765 / ARTBOARD_WIDTH, position.y0, position.width / 2, position.height])
        for text in ax.get_xticklabels():
            text.set_rotation(90)
            text.set_ha("center")
            text.set_va("top")
            text.set_rotation_mode("default")
        for text in ax.texts:
            if text.get_text().startswith("p ="):
                text.set_text("p =\n0.051")
    f_axes[0].set_title("Location-\nassociated\nvariance", fontsize=10.85, pad=7)
    next(text for text in fig.texts if text.get_text() == "F").set_x(729 / ARTBOARD_WIDTH)


def finish_variance_panel(fig) -> None:
    data = pd.read_csv(VARIANCE_STATS)
    for row, (ax, metric) in enumerate(zip(fig.axes[-3:], data.metric.unique())):
        record = data.loc[data.metric.eq(metric) & data.population.eq("HVA_minus_V1")].iloc[0]
        ax.vlines(2, 100 * record.low, 100 * record.high, color="#315f9a", lw=1.7, zorder=3)
        ax.plot(2, 100 * record.omega_squared, "D", color="#315f9a", ms=5, zorder=4)
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(-2.5, 5.15)
        ax.set_yticks([-2, 0, 2, 4])
        ax.set_xticks([0, 1, 2], ["Intra-V1", "Inter-HVA", "Inter-HVA\n− intra-V1"],
                      rotation=90, ha="center", va="top")
        if row < 2:
            ax.tick_params(labelbottom=False)
        for text in ax.texts:
            text.set_fontsize(9)
        for text in ax.get_xticklabels():
            text.set_fontfamily("Liberation Sans")
            text.set_rotation_mode("default")
    adjusted = adjusted_variance_table()
    for ax, metric in zip(fig.axes[-3:], adjusted.metric.unique()):
        for text in ax.texts:
            x, _ = text.get_position()
            if x not in (0, 1):
                continue
            population = ["V1", "HVA"][int(x)]
            probability = adjusted.loc[
                adjusted.metric.eq(metric) & adjusted.population.eq(population),
                "permutation_p_bonferroni",
            ].item()
            text.set_text("***" if probability < .001 else "**" if probability < .01
                          else "*" if probability < .05 else "n.s.")


def adjusted_variance_table() -> pd.DataFrame:
    data = pd.read_csv(VARIANCE_STATS)
    assert data.permutation_p_greater.notna().sum() == 6
    data["permutation_p_bonferroni"] = (6 * data.permutation_p_greater).clip(upper=1)
    return data


def canonical_style(fig) -> None:
    apply_publication_style(fig)
    install_cortical_vectors(fig)
    make_results_taller(fig)
    split_panel_e(fig)
    polish_results(fig)
    arrange_top_row(fig)
    add_variance_panel(fig)
    compact_panels_e_f(fig)
    finish_variance_panel(fig)
    outside = visible_text_outside(fig)
    assert not outside, outside


def write_caption() -> None:
    caption = CAPTION_SOURCE.read_text()
    panel_f = """**F)** Location-associated variance, expressed as bias-corrected one-way omega-squared (ω², %), for intra-V1 recording location and inter-HVA area identity, followed by their signed inter-HVA minus intra-V1 difference (blue diamonds). The difference is expressed in percentage points and its 95% interval is calculated from independent-population whole-animal bootstrap differences. No shuffle-test significance symbol is assigned to this contrast. All three contrast intervals include zero; this does not establish equivalence. Points and whiskers show estimates and individual 95% percentile intervals from 5,000 whole-animal bootstrap replicates (not simultaneous intervals). Unlike E’s balanced, session-adjusted category-mean SDs, F uses raw neuronal responses with equal cell weights and no session adjustment; negative estimates are retained. Significance annotations use matching within-session location-block shuffles (5,000 permutations; one-sided tests with Bonferroni correction across the six component tests: three metrics × two populations): * p < .05, ** p < .01, *** p < .001; n.s., p ≥ .05. All neurons in a session/location move together; tests assume exchangeability of available labels within sessions. For MouseV2 timescale, the observed and shuffled ω² statistics average ten draw-specific values, using one coherent label permutation per session across draws. Annotations use adjusted p-values, calculated as min(1, 6 × raw p). Adjusted inter-HVA p-values are 0.002400 for TTFS and 0.001200 for timescale; all remaining component tests have adjusted p ≥ .05. These tests concern each location effect, not the difference between intra-V1 and inter-HVA effects; nonsignificance does not establish absence of an effect. Summary shuffle statistics and exact p-values are supplied in Figure5_F_estimates_and_tests.csv.

"""
    caption = caption.replace("Timescale estimation uses", panel_f + "Timescale estimation uses")
    STEM.with_name(STEM.name + "_legend.md").write_text(caption)
    (OUTPUT / "Figure5_F_condensed_caption.md").write_text(
        "**F)** Location-associated variance (bias-corrected ω²) within V1, across HVAs, and their difference (inter-HVA − intra-V1; blue diamonds, in percentage points). Whiskers show individual 95% whole-animal bootstrap intervals. Asterisks indicate one-sided within-session location-block shuffle tests, Bonferroni-adjusted across six component tests (** indicates adjusted p < .01; n.s., adjusted p ≥ .05). Difference intervals include zero for all metrics; this does not establish equivalence.\n"
    )


def finalize_manifest() -> None:
    path = STEM.with_name(STEM.name + "_layout_manifest.json")
    manifest = json.loads(path.read_text())
    manifest.update({
        "figure_number": 5,
        "renderer": "MouseV2-Figure5/render.py",
        "description": "Canonical Figure 5 v17: compact E/F, direct omega-squared contrasts, and Bonferroni-adjusted panel-F component tests.",
        "artboard_points": [ARTBOARD_WIDTH, ARTBOARD_HEIGHT],
        "style": {
            "font": "Liberation Sans (Arial-compatible substitute)",
            "panel_letters_pt": 32,
            "panel_headings_pt": 16,
            "plot_titles_pt": 14,
            "axes_linewidth_pt": .7,
        },
        "cartoons": {
            "source": str(VECTOR_SOURCE.relative_to(ROOT)),
            "sha256": hashlib.sha256(VECTOR_SOURCE.read_bytes()).hexdigest(),
            "geometry": "Exact SVG Bezier paths and original marker coordinates/radii.",
            "empirical_coordinates": False,
        },
        "panel_F_source": os.path.relpath(VARIANCE_STATS, ROOT),
        "multiplicity": {
            "method": "Bonferroni", "family": "3 metrics × 2 populations",
            "n_tests": 6, "formula": "min(1, 6 × raw p)",
            "bootstrap_intervals": "individual 95%; not simultaneous",
        },
    })
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    base.main(OUTPUT, "harris2019", figure_number=5,
              artboard_points=(ARTBOARD_WIDTH, 574.601), style_callback=canonical_style)
    adjusted_variance_table().to_csv(OUTPUT / "Figure5_F_estimates_and_tests.csv", index=False)
    write_caption()
    finalize_manifest()


if __name__ == "__main__":
    main()
