#!/usr/bin/env python3
"""Render the final Figure 4 rerun C/D/E and companion result figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import (
    HVA_CATEGORIES, METRICS, V1_CATEGORIES, apply_cell_floor, build_response_registry,
)
from scripts.run_figure4_central_companion import CATEGORIES as CENTRAL_CATEGORIES

PRIMARY = ROOT / "artifacts/figure4_rerun/v4_primary_dispersion"
HIERARCHY = ROOT / "artifacts/figure4_rerun/v5_hva_hierarchy"
CONTROL = ROOT / "artifacts/figure4_rerun/v6_matched_control"
CENTRAL = ROOT / "artifacts/figure4_rerun/v7_central_companion"
OUT = ROOT / "artifacts/figure4_rerun/v8_final"
REQUIRED_INPUTS = (
    PRIMARY / "primary_effects.csv",
    PRIMARY / "draw_specific_point_fits.csv",
    PRIMARY / "normalized_effects.csv",
    HIERARCHY / "hierarchy_slopes.csv",
    CONTROL / "matched_control.csv",
    CENTRAL / "central_inclusive_effects.csv",
    CENTRAL / "central_inclusive_draw_fits.csv",
)

V1_COLORS = {"A": "#d73027", "E": "#f28e2b", "C": "#43a047", "B": "#3575b5", "Central": "#555555"}
HVA_COLORS = {"LM": "#7b3294", "RL": "#c51b7d", "AL": "#e66101", "PM": "#4d9221", "AM": "#0571b0"}
POP_COLORS = {"V1": "#3b75af", "HVA": "#b64972", "Control": "#9a641e", "Difference": "#343434"}
DISPLAY = {"A": "Anterior", "E": "Lateral", "C": "Posterior", "B": "Medial"}
TIMESCALE_DISPLAY_DRAW = 0


def clean(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(labelsize=8.5, colors="#3f3f3f")
    if grid:
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
        ax.set_axisbelow(True)


def parse_means(value: str) -> dict[str, float]:
    return {key: float(number) for key, number in (piece.split("=", 1) for piece in value.split(";"))}


def averaged_means(rows: pd.Series, categories: tuple[str, ...]) -> dict[str, float]:
    """Average draw-specific category means; every draw must carry every category."""
    parsed = [parse_means(value) for value in rows]
    if not parsed:
        raise ValueError("No category means to average")
    for index, means in enumerate(parsed):
        if missing := set(categories).difference(means):
            raise ValueError(f"Draw {index} lacks category means for {sorted(missing)}")
    return {key: float(np.mean([means[key] for means in parsed])) for key in categories}


def display_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Return one coherent analysis population for descriptive display.

    MouseV2 timescale uses fixed display draw 1 of 10, chosen by index. Allen has
    one accepted estimate per neuron. The inferential effect panels continue to
    report the saved ten-draw-average estimator.
    """
    table = table.copy()
    if table.metric.eq(METRICS[2]).all():
        mouse = table.source.eq("MouseV2")
        table = table.loc[
            (mouse & table.draw_id.eq(TIMESCALE_DISPLAY_DRAW))
            | (~mouse & table.draw_id.eq(-1))
        ].copy()
        table = apply_cell_floor(table, 5)
    if table.duplicated("unit_id").any():
        raise ValueError("Display population contains duplicate units")
    table["weight"] = 1.0
    return table


def response_cells(registry: pd.DataFrame, population: str, metric: str) -> pd.DataFrame:
    return display_cells(registry.loc[registry.population.eq(population) & registry.metric.eq(metric)])


def cell_count_label(rows: pd.DataFrame) -> str:
    return f"n={len(rows):,}"


def violin(ax: plt.Axes, table: pd.DataFrame, order: tuple[str, ...], colors: dict[str, str], means: dict[str, float]) -> None:
    rng = np.random.default_rng(20260918)
    for x, category in enumerate(order):
        rows = table.loc[table.category.eq(category)]
        if rows.empty:
            raise ValueError(f"No display cells for category {category!r}")
        values = rows.value.to_numpy(float)
        weights = rows.weight.to_numpy(float)
        color = colors[category]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite display values for category {category!r}")
        if np.ptp(values) > 0:
            grid = np.linspace(values.min(), values.max(), 180)
            density = gaussian_kde(values, weights=weights)(grid)
            ax.fill_betweenx(grid, x, x + .34 * density / density.max(), color=color, alpha=.24, linewidth=.7)
        else:
            ax.hlines(values[0], x, x + .34, color=color, alpha=.6, linewidth=1.2)
        maximum = min(240, len(values))
        chosen = rng.choice(len(values), maximum, replace=False, p=weights / weights.sum())
        ax.scatter(x + rng.uniform(-.24, -.03, maximum), values[chosen], s=4, alpha=.16,
                   color=color, linewidth=0, rasterized=True)
        ax.plot(x, means[category], marker="D", ms=5.2, color=color, mec="white", mew=.55, zorder=5)
        ax.text(x, .98, cell_count_label(rows), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color="#666666")
    ax.set_xticks(range(len(order)), [DISPLAY.get(value, value) for value in order], rotation=22, ha="right")
    ax.set_xlim(-.5, len(order) - .5)
    clean(ax)


def effect(ax: plt.Axes, estimate: float, low: float, high: float, color: str, label: str, *, zero: bool = True) -> None:
    if zero:
        ax.axhline(0, color="#888888", lw=.9, ls="--")
    ax.vlines(0, low, high, color=color, lw=2)
    ax.plot(0, estimate, "o", color=color, ms=5.5)
    ax.set_xlim(-.45, .45)
    ax.set_xticks([0], [label])
    ax.text(.5, .98, f"{estimate:.3g}\n[{low:.3g}, {high:.3g}]", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=color)
    clean(ax)


def main_cde(registry: pd.DataFrame, output: Path) -> None:
    effects = pd.read_csv(PRIMARY / "primary_effects.csv")
    point = pd.read_csv(PRIMARY / "draw_specific_point_fits.csv")
    control = pd.read_csv(CONTROL / "matched_control.csv")
    fig = plt.figure(figsize=(16, 10.2))
    outer = fig.add_gridspec(3, 5, width_ratios=[1.55, .48, 1.65, .48, 1.18],
                             left=.055, right=.985, top=.90, bottom=.105, wspace=.48, hspace=.52)
    fig.suptitle("Functional differentiation within V1 and across higher visual areas", x=.055, ha="left", fontsize=17, fontweight="bold")
    fig.text(.055, .932, "Harmonized single-cell outcomes · equal category/animal/recording/cell weights · whole-animal bootstrap intervals", fontsize=10, color="#555555")
    for row, metric in enumerate(METRICS):
        metric_effects = effects.loc[effects.metric.eq(metric)].set_index("effect")
        crow = control.loc[control.metric.eq(metric)].iloc[0]
        bounds = [
            metric_effects.loc["v1", "interval_low"], metric_effects.loc["v1", "interval_high"],
            metric_effects.loc["hva", "interval_low"], metric_effects.loc["hva", "interval_high"],
            metric_effects.loc["hva_minus_v1", "interval_low"], metric_effects.loc["hva_minus_v1", "interval_high"],
            crow.two_group_dispersion_low, crow.two_group_dispersion_high,
        ]
        effect_span = max(bounds) - min(bounds)
        effect_ylim = (min(0, min(bounds)) - .08 * effect_span, max(bounds) + .18 * effect_span)
        effect_axes = []
        response_tables = {
            "V1": response_cells(registry, "V1", metric),
            "HVA": response_cells(registry, "HVA", metric),
        }
        response_limits = [response_tables[p].value.quantile([.002, .998]).to_numpy() for p in ["V1", "HVA"]]
        ymin, ymax = min(x[0] for x in response_limits), max(x[1] for x in response_limits)
        pad = .08 * (ymax - ymin)
        for col, population, order, colors, effect_name, title in [
            (0, "V1", V1_CATEGORIES, V1_COLORS, "v1", "C  Within V1"),
            (2, "HVA", HVA_CATEGORIES, HVA_COLORS, "hva", "D  Across HVAs"),
        ]:
            ax = fig.add_subplot(outer[row, col])
            mean_rows = point.loc[point.metric.eq(metric) & point.population.eq(population)]
            if metric == METRICS[2]:
                mean_rows = mean_rows.loc[mean_rows.draw.eq(TIMESCALE_DISPLAY_DRAW)]
            means = averaged_means(mean_rows.category_means, order)
            violin(ax, response_tables[population], order, colors, means)
            ax.set_ylim(ymin - pad, ymax + pad)
            if col == 0:
                ax.set_ylabel(metric, fontsize=9.5)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=16)
            erow = metric_effects.loc[effect_name]
            ea = fig.add_subplot(outer[row, col + 1])
            effect(ea, erow.estimate, erow.interval_low, erow.interval_high,
                   POP_COLORS[population], f"V({population})", zero=False)
            effect_axes.append(ea)
            if col == 0:
                ea.set_ylabel("Category dispersion\n(squared response units)", fontsize=8)
        esub = outer[row, 4].subgridspec(1, 2, wspace=.75)
        ca = fig.add_subplot(esub[0, 0])
        effect(ca, crow.two_group_dispersion, crow.two_group_dispersion_low,
               crow.two_group_dispersion_high, POP_COLORS["Control"], "Matched\ncontrol", zero=False)
        effect_axes.append(ca)
        ca.set_ylabel("Two-group dispersion\n(squared response units)", fontsize=8)
        if row == 0:
            # Right-anchored so the heading cannot overrun the figure edge; the
            # baseline matches the C/D axes titles (axes top + 16 pt pad).
            top = outer[0, 4].get_position(fig).y1 + 16 / 72 / fig.get_figheight()
            fig.text(.985, top, "E  Control and main contrast", fontsize=12, fontweight="bold", ha="right", va="baseline")
        da = fig.add_subplot(esub[0, 1])
        drow = metric_effects.loc["hva_minus_v1"]
        effect(da, drow.estimate, drow.interval_low, drow.interval_high,
               POP_COLORS["Difference"], "V(HVA)\n− V(V1)")
        effect_axes.append(da)
        da.set_ylabel("Dispersion difference\n(squared response units)", fontsize=8)
        for axis in effect_axes:
            axis.set_ylim(effect_ylim)
    fig.legend(handles=[
        Line2D([], [], marker="D", color="#555555", linestyle="none", label="Model-standardized category mean"),
        Line2D([], [], marker="o", color="#555555", label="Point estimate + 95% bootstrap interval"),
    ], loc="upper right", bbox_to_anchor=(.985, .985), frameon=False, fontsize=8.5)
    fig.text(.055, .028,
             f"C/D: half-violins show eligible cell distributions. MouseV2 response timescale uses fixed display draw {TIMESCALE_DISPLAY_DRAW + 1}/10, chosen by index, "
             "after its draw-specific cell floor; n and diamonds refer to that displayed population. Dots are a deterministic display subsample. "
             "Response axes show the central 99.6% for legibility; all eligible values enter the estimates. Timescale effect panels average ten complete draw-specific effects. "
             "E: matched Allen V1 versus equal-area pooled HVA control and the primary HVA-minus-V1 category-dispersion contrast. "
             "Intervals are pointwise, not equivalence tests; no significance stars are used.",
             fontsize=8.2, color="#555555", va="bottom", wrap=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"Figure4_CDE_rerun.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def central_figure(registry: pd.DataFrame, output: Path) -> None:
    points = pd.read_csv(CENTRAL / "central_inclusive_effects.csv")
    fits = pd.read_csv(CENTRAL / "central_inclusive_draw_fits.csv")
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 10.4), gridspec_kw={"width_ratios": [4.5, 1]})
    fig.subplots_adjust(left=.07, right=.985, top=.91, bottom=.11, hspace=.30, wspace=.16)
    for index, metric in enumerate(METRICS):
        table = display_cells(registry.loc[registry.metric.eq(metric) & registry.population.isin(["V1", "Central"])])
        table.loc[table.population.eq("Central"), "category"] = "Central"
        mean_rows = fits.loc[fits.metric.eq(metric)]
        if metric == METRICS[2]:
            mean_rows = mean_rows.loc[mean_rows.draw.eq(TIMESCALE_DISPLAY_DRAW)]
        means = averaged_means(mean_rows.category_means, CENTRAL_CATEGORIES)
        violin(axes[index, 0], table, CENTRAL_CATEGORIES, V1_COLORS, means)
        axes[index, 0].set_ylabel(metric)
        row = points.loc[points.metric.eq(metric)].iloc[0]
        effect(axes[index, 1], row.dispersion, row.dispersion_low, row.dispersion_high,
               "#555555", "Five-location\ndescriptive V", zero=False)
        low, high = axes[index, 1].get_ylim()
        axes[index, 1].set_ylim(low, high + .22 * (high - low))  # headroom for the estimate text
        axes[index, 1].set_ylabel("Descriptive dispersion\n(squared response units)", fontsize=8)
        axes[index, 1].text(.98, .02, f"rank {int(row.fixed_session_rank)}/{int(row.fixed_session_parameters)}",
                            transform=axes[index, 1].transAxes, ha="right", va="bottom", fontsize=6.8, color="#666666")
        if index < len(METRICS) - 1:
            for ax in axes[index]:
                ax.tick_params(labelbottom=False)
    fig.suptitle("Central-inclusive V1 companion", x=.07, ha="left", fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .945, "Central = harmonized Allen V1; four peripheral locations = MouseV2", fontsize=10, color="#555555", ha="center")
    fig.text(.01, .012,
             "Central is placed between lateral and posterior. Its cohort/session nesting makes a session-adjusted five-location effect rank deficient\n"
             f"(rank shown in each effect panel); intervals describe the mixed-cohort descriptive quantity only. MouseV2 timescale uses fixed display\n"
             f"draw {TIMESCALE_DISPLAY_DRAW + 1}/10, chosen by index, for the violin, n, and diamond; descriptive effects average ten complete draw-specific effects.",
             fontsize=8, color="#555555", va="bottom")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"Figure4_Central_inclusive_companion.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def hierarchy_figure(output: Path) -> None:
    table = pd.read_csv(HIERARCHY / "hierarchy_slopes.csv").set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        row = table.loc[metric]
        for x, prefix, color in [(0, "unadjusted", "#777777"), (1, "adjusted", "#7b3294")]:
            ax.vlines(x, row[prefix + "_slope_low"], row[prefix + "_slope_high"], color=color, lw=2)
            ax.plot(x, row[prefix + "_slope"], "o", color=color, ms=6)
        ax.axhline(0, color="#888888", lw=.9, ls="--")
        ax.set_xlim(-.5, 1.5)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + .18 * (high - low))  # headroom for the paired-change text
        ax.set_xticks([0, 1], ["Unadjusted", "RF-adjusted"], rotation=15, ha="right")
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Response / hierarchy-score unit")
        delta = row.adjusted_minus_unadjusted
        low = row.adjusted_minus_unadjusted_low
        high = row.adjusted_minus_unadjusted_high
        ax.text(.5, .98, f"paired change {delta:+.3g}\n95% interval [{low:+.3g}, {high:+.3g}]", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        clean(ax)
    fig.suptitle("HVA hierarchy slope before and after RF adjustment", fontsize=14, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"Figure4_hierarchy_RF_conditional.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def normalized_figure(output: Path) -> None:
    effects = pd.read_csv(PRIMARY / "normalized_effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        part = effects.loc[effects.metric.eq(metric)].set_index("effect")
        for x, name, color in [(0, "v1", POP_COLORS["V1"]), (1, "hva", POP_COLORS["HVA"]), (2, "hva_minus_v1", POP_COLORS["Difference"])]:
            row = part.loc[name]
            ax.vlines(x, row.interval_low, row.interval_high, color=color, lw=2)
            ax.plot(x, row.estimate, "o", color=color, ms=6)
        ax.axhline(0, color="#888888", lw=.9, ls="--")
        ax.set_xlim(-.5, 2.5)
        ax.set_xticks([0, 1, 2], ["V1", "HVA", "HVA − V1"])
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Normalized dispersion (%) or difference (pp)")
        clean(ax)
    fig.suptitle("Normalized category-dispersion companion", fontsize=14, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"Figure4_normalized_dispersion_companion.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def write_display_counts(registry: pd.DataFrame, output: Path) -> None:
    """Export every count printed above a distribution."""
    rows = []
    for metric in METRICS:
        for panel, population in [("C", "V1"), ("D", "HVA")]:
            table = response_cells(registry, population, metric)
            for category, part in table.groupby("category", sort=False):
                rows.append({
                    "figure": "Figure4_CDE_rerun", "panel": panel,
                    "metric": metric, "population": population,
                    "category": category, "display_draw": (
                        TIMESCALE_DISPLAY_DRAW if metric == METRICS[2] and part.source.eq("MouseV2").all() else -1
                    ),
                    "n_cells": len(part),
                })
        table = display_cells(
            registry.loc[registry.metric.eq(metric) & registry.population.isin(["V1", "Central"])]
        ).copy()
        table.loc[table.population.eq("Central"), "category"] = "Central"
        for category, part in table.groupby("category", sort=False):
            rows.append({
                "figure": "Figure4_Central_inclusive_companion", "panel": "Central",
                "metric": metric, "population": "Central-inclusive V1",
                "category": category, "display_draw": (
                    TIMESCALE_DISPLAY_DRAW if metric == METRICS[2] and part.source.eq("MouseV2").all() else -1
                ),
                "n_cells": len(part),
            })
    pd.DataFrame(rows).to_csv(output / "figure4_display_counts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_INPUTS if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required completed analysis outputs are missing: " + ", ".join(missing))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = apply_cell_floor(build_response_registry(), 5)
    write_display_counts(registry, output)
    main_cde(registry, output)
    central_figure(registry, output)
    hierarchy_figure(output)
    normalized_figure(output)
    print(output)


if __name__ == "__main__":
    main()
