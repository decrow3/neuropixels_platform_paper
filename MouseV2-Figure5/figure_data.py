"""Small data/display interface required by the canonical Figure 5 renderer."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "MouseV2-Figure5/data"
CONTROL = ROOT / "MouseV2-Figure5/data"

METRICS = ("TTFS (ms)", "log10 F1/F0", "Response timescale (ms)")
V1_CATEGORIES = ("A", "E", "C", "B")
HVA_CATEGORIES = ("LM", "RL", "AL", "PM", "AM")
V1_COLORS = {"A": "#d73027", "E": "#f28e2b", "C": "#43a047", "B": "#3575b5"}
HVA_COLORS = {"LM": "#7b3294", "RL": "#c51b7d", "AL": "#e66101", "PM": "#4d9221", "AM": "#0571b0"}
POP_COLORS = {"V1": "#3b75af", "HVA": "#b64972", "Control": "#9a641e", "Difference": "#343434"}
DISPLAY = {"A": "Anterior", "E": "Lateral", "C": "Posterior", "B": "Medial"}
TIMESCALE_DISPLAY_DRAW = 0


def clean(ax, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(labelsize=8.5, colors="#3f3f3f")
    if grid:
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
        ax.set_axisbelow(True)


def parse_means(value: str) -> dict[str, float]:
    return {key: float(number) for key, number in
            (piece.split("=", 1) for piece in value.split(";"))}


def averaged_means(rows: pd.Series, categories: tuple[str, ...]) -> dict[str, float]:
    parsed = [parse_means(value) for value in rows]
    if not parsed:
        raise ValueError("No category means to average")
    for index, means in enumerate(parsed):
        if missing := set(categories).difference(means):
            raise ValueError(f"Draw {index} lacks category means for {sorted(missing)}")
    return {key: float(np.mean([means[key] for means in parsed])) for key in categories}


def apply_cell_floor(table: pd.DataFrame, floor: int = 5) -> pd.DataFrame:
    keys = ["metric", "recording_id", "category", "draw_id"]
    counts = table.groupby(keys).unit_id.transform("nunique")
    return table.loc[counts.ge(floor)].copy()


def display_cells(table: pd.DataFrame) -> pd.DataFrame:
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
    return display_cells(registry.loc[
        registry.population.eq(population) & registry.metric.eq(metric)
    ])


def violin(ax, table: pd.DataFrame, order: tuple[str, ...],
           colors: dict[str, str], means: dict[str, float]) -> None:
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
            ax.fill_betweenx(grid, x, x + .34 * density / density.max(),
                             color=color, alpha=.24, linewidth=.7)
        else:
            ax.hlines(values[0], x, x + .34, color=color, alpha=.6, linewidth=1.2)
        maximum = min(240, len(values))
        chosen = rng.choice(len(values), maximum, replace=False, p=weights / weights.sum())
        ax.scatter(x + rng.uniform(-.24, -.03, maximum), values[chosen], s=4,
                   alpha=.16, color=color, linewidth=0, rasterized=True)
        ax.plot(x, means[category], marker="D", ms=5.2, color=color,
                mec="white", mew=.55, zorder=5)
        ax.text(x, .98, f"n={len(rows):,}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color="#666666")
    ax.set_xticks(range(len(order)), [DISPLAY.get(value, value) for value in order],
                  rotation=22, ha="right")
    ax.set_xlim(-.5, len(order) - .5)
    clean(ax)
