#!/usr/bin/env python3
"""Plot response-timescale distributions for the five cortical HVAs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/"
    "metric_extension_cortical_hvas/full_cell_metric_extension_input.csv"
)
OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "area_driver_distributions_cortical_hvas"
)
AREAS = ["LM", "RL", "AL", "PM", "AM"]
COLORS = {
    "LM": "#4e73ae",
    "RL": "#65b2c9",
    "AL": "#cab778",
    "PM": "#db8457",
    "AM": "#c24f54",
}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cumulative, values))


def load_timescales() -> pd.DataFrame:
    table = pd.read_csv(INPUT, dtype={"session_id": str, "unit_id": str})
    table = table.loc[
        table["dataset"].eq("Post-V1")
        & table["metric"].eq("Response timescale (ms)")
        & table["group"].isin(AREAS)
    ].copy()
    keys = ["session_id", "group", "unit_id"]
    if table.duplicated(keys).any():
        raise ValueError("A neuron occurs more than once in the timescale distribution")
    population_n = table.groupby(["group", "session_id"])["unit_id"].transform("size")
    session_n = table.groupby("group")["session_id"].transform("nunique")
    table["equal_session_weight"] = 1.0 / (population_n * session_n)
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for area in AREAS:
        local = table.loc[table["group"].eq(area)]
        values = local["value"].to_numpy(float)
        weights = local["equal_session_weight"].to_numpy(float)
        rows.append(
            {
                "area": area,
                "neurons": len(local),
                "sessions": local["session_id"].nunique(),
                "equal_session_weighted_mean_ms": np.average(values, weights=weights),
                "equal_session_weighted_q25_ms": weighted_quantile(values, weights, 0.25),
                "equal_session_weighted_median_ms": weighted_quantile(values, weights, 0.50),
                "equal_session_weighted_q75_ms": weighted_quantile(values, weights, 0.75),
            }
        )
    return pd.DataFrame(rows)


def render(table: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figure, (density_ax, session_ax) = plt.subplots(
        1, 2, figsize=(13.0, 6.2), gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    grid = np.linspace(0, 150, 500)
    for row, area in enumerate(AREAS[::-1]):
        local = table.loc[table["group"].eq(area)]
        values = local["value"].to_numpy(float)
        weights = local["equal_session_weight"].to_numpy(float)
        density = gaussian_kde(values, weights=weights, bw_method=0.20)(grid)
        density = 0.78 * density / density.max()
        density_ax.fill_between(
            grid, row, row + density, color=COLORS[area], alpha=0.42,
            edgecolor=COLORS[area], linewidth=1.1,
        )
        density_ax.plot(grid, row + density, color=COLORS[area], linewidth=1.4)
        record = summary.loc[summary["area"].eq(area)].iloc[0]
        median = record["equal_session_weighted_median_ms"]
        density_ax.vlines(median, row, row + 0.70, color=COLORS[area], linewidth=2.1)
        density_ax.text(
            148.0, row + 0.30,
            f"n={int(record['neurons']):,}; {int(record['sessions'])} sessions",
            ha="right", va="center", fontsize=8, color="#4f5962",
        )
    density_ax.set_yticks(np.arange(len(AREAS)), AREAS[::-1])
    density_ax.set_xlim(0, 150)
    density_ax.set_xlabel("Neuron response timescale, τ (ms)")
    density_ax.set_ylabel("Cortical HVA")
    density_ax.set_title(
        "Full neuronal distributions", loc="left", fontweight="bold", pad=34
    )
    density_ax.text(
        0.01, 1.015,
        "Density gives every session equal total weight; vertical segment is weighted median",
        transform=density_ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        color="#555f67",
    )
    density_ax.grid(axis="x", color="#dfe3e6", linewidth=0.7)
    density_ax.spines[["top", "right", "left"]].set_visible(False)
    density_ax.tick_params(axis="y", length=0)

    rng = np.random.default_rng(20260826)
    session_means = (
        table.groupby(["group", "session_id"], as_index=False)
        .agg(mean_timescale_ms=("value", "mean"), neurons=("unit_id", "size"))
    )
    for index, area in enumerate(AREAS):
        values = session_means.loc[
            session_means["group"].eq(area), "mean_timescale_ms"
        ].to_numpy(float)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        session_ax.scatter(
            index + jitter, values, s=25, color=COLORS[area], alpha=0.48,
            edgecolor="#38434c", linewidth=0.35, rasterized=True,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        session_ax.errorbar(
            index, mean, yerr=sem, fmt="o", markersize=9, color=COLORS[area],
            markeredgecolor="white", markeredgewidth=0.8, capsize=3,
            elinewidth=1.7, zorder=4,
        )
    session_ax.set_xticks(np.arange(len(AREAS)), AREAS)
    session_ax.set_ylabel("Session × area mean timescale (ms)")
    session_ax.set_title(
        "Session-level replication", loc="left", fontweight="bold", pad=34
    )
    session_ax.text(
        0.01, 1.015, "Points are sessions; large circles are equal-session mean ± SEM",
        transform=session_ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        color="#555f67",
    )
    session_ax.grid(axis="y", color="#dfe3e6", linewidth=0.7)
    session_ax.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Response-timescale distributions across cortical HVAs",
        fontsize=14, fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        "Conditioned on retained common-QC neurons: 1 ≤ τ ≤ 300 ms, fitted τ error < 20 ms, and >50 fitting-window spikes.",
        ha="center", va="top", fontsize=8.5, color="#555f67",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = load_timescales()
    summary = summarize(table)
    table.to_csv(OUTPUT / "timescale_area_neurons_equal_session_weighted.csv", index=False)
    summary.to_csv(OUTPUT / "timescale_area_distribution_summary.csv", index=False)
    render(table, summary, OUTPUT / "Figure_timescale_area_distributions.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
