#!/usr/bin/env python3
"""Plot trial-matched response-timescale distributions across V1 probes."""

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
REFERENCE = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "area_driver_distributions_cortical_hvas"
)
PROBES = ["B", "C", "A", "E"]
COLORS = {
    "B": "#4575b4",
    "C": "#1a9850",
    "A": "#d73027",
    "E": "#8073ac",
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
        table["dataset"].eq("Within-V1")
        & table["metric"].eq("Response timescale (ms)")
        & table["group"].isin(PROBES)
    ].copy()
    keys = ["session_id", "group", "unit_id", "draw_id"]
    if table.duplicated(keys).any():
        raise ValueError("A neuron has duplicate values within a matched draw")

    # A repeated fit divides, rather than multiplies, its neuron's weight. Every
    # neuron then has equal total weight within its session, and every session has
    # equal total weight within its probe/location.
    neuron_draw_n = table.groupby(
        ["group", "session_id", "unit_id"]
    )["draw_id"].transform("nunique")
    neuron_n = table.groupby(["group", "session_id"])["unit_id"].transform("nunique")
    session_n = table.groupby("group")["session_id"].transform("nunique")
    table["equal_session_neuron_weight"] = 1.0 / (
        neuron_draw_n * neuron_n * session_n
    )
    weight_sums = table.groupby("group")["equal_session_neuron_weight"].sum()
    if not np.allclose(weight_sums.reindex(PROBES), 1.0):
        raise ValueError(f"Probe weights do not sum to one:\n{weight_sums}")
    return table


def session_probe_means(table: pd.DataFrame) -> pd.DataFrame:
    draw_means = (
        table.groupby(["group", "session_id", "draw_id"], as_index=False)
        .agg(draw_mean_timescale_ms=("value", "mean"), neurons=("unit_id", "nunique"))
    )
    draw_counts = draw_means.groupby(["group", "session_id"])["draw_id"].nunique()
    if not draw_counts.eq(10).all():
        raise ValueError("A retained session × probe population lacks all ten draws")
    means = (
        draw_means.groupby(["group", "session_id"], as_index=False)
        .agg(
            mean_timescale_ms=("draw_mean_timescale_ms", "mean"),
            min_neurons_per_draw=("neurons", "min"),
            max_neurons_per_draw=("neurons", "max"),
        )
    )

    reference = pd.read_csv(REFERENCE, dtype={"session_id": str})
    reference = reference.loc[
        reference["dataset"].eq("Within-V1")
        & reference["metric"].eq("Response timescale (ms)"),
        ["group", "session_id", "mean"],
    ]
    checked = means.merge(
        reference, on=["group", "session_id"], how="outer", validate="one_to_one",
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("Session × probe populations differ from the validated table")
    max_error = np.max(np.abs(checked["mean_timescale_ms"] - checked["mean"]))
    if max_error > 1e-10:
        raise ValueError(f"Session means do not reproduce validated values: {max_error}")
    return means


def summarize(table: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe in PROBES:
        local = table.loc[table["group"].eq(probe)]
        values = local["value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        local_means = means.loc[means["group"].eq(probe), "mean_timescale_ms"]
        rows.append(
            {
                "probe": probe,
                "unique_neurons": local["unit_id"].nunique(),
                "sessions": local["session_id"].nunique(),
                "valid_neuron_draw_fits": len(local),
                "equal_session_neuron_weighted_mean_ms": np.average(values, weights=weights),
                "equal_session_neuron_weighted_q25_ms": weighted_quantile(values, weights, 0.25),
                "equal_session_neuron_weighted_median_ms": weighted_quantile(values, weights, 0.50),
                "equal_session_neuron_weighted_q75_ms": weighted_quantile(values, weights, 0.75),
                "equal_session_mean_of_draw_means_ms": local_means.mean(),
                "session_mean_sem_ms": local_means.sem(),
            }
        )
    return pd.DataFrame(rows)


def render(
    table: pd.DataFrame, means: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    figure, (density_ax, session_ax) = plt.subplots(
        1, 2, figsize=(13.0, 5.9), gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    grid = np.linspace(0, 200, 600)
    for row, probe in enumerate(PROBES[::-1]):
        local = table.loc[table["group"].eq(probe)]
        values = local["value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        density = gaussian_kde(values, weights=weights, bw_method=0.20)(grid)
        density = 0.78 * density / density.max()
        density_ax.fill_between(
            grid, row, row + density, color=COLORS[probe], alpha=0.42,
            edgecolor=COLORS[probe], linewidth=1.1,
        )
        density_ax.plot(grid, row + density, color=COLORS[probe], linewidth=1.4)
        record = summary.loc[summary["probe"].eq(probe)].iloc[0]
        median = record["equal_session_neuron_weighted_median_ms"]
        density_ax.vlines(median, row, row + 0.70, color=COLORS[probe], linewidth=2.1)
        density_ax.text(
            197.0, row + 0.30,
            f"n={int(record['unique_neurons']):,}; {int(record['sessions'])} sessions",
            ha="right", va="center", fontsize=8, color="#4f5962",
        )
    density_ax.set_yticks(np.arange(len(PROBES)), PROBES[::-1])
    density_ax.set_xlim(0, 200)
    density_ax.set_xlabel("Neuron response timescale, τ (ms)")
    density_ax.set_ylabel("V1 probe/location")
    density_ax.set_title(
        "Full neuronal distributions", loc="left", fontweight="bold", pad=34
    )
    density_ax.text(
        0.01, 1.015,
        "Equal session and neuron weight; valid matched draws share each neuron's weight",
        transform=density_ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        color="#555f67",
    )
    density_ax.grid(axis="x", color="#dfe3e6", linewidth=0.7)
    density_ax.spines[["top", "right", "left"]].set_visible(False)
    density_ax.tick_params(axis="y", length=0)

    rng = np.random.default_rng(20260826)
    for index, probe in enumerate(PROBES):
        values = means.loc[
            means["group"].eq(probe), "mean_timescale_ms"
        ].to_numpy(float)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        session_ax.scatter(
            index + jitter, values, s=25, color=COLORS[probe], alpha=0.48,
            edgecolor="#38434c", linewidth=0.35, rasterized=True,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        session_ax.errorbar(
            index, mean, yerr=sem, fmt="o", markersize=9, color=COLORS[probe],
            markeredgecolor="white", markeredgewidth=0.8, capsize=3,
            elinewidth=1.7, zorder=4,
        )
    session_ax.set_xticks(np.arange(len(PROBES)), PROBES)
    session_ax.set_ylabel("Session × probe mean timescale (ms)")
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
        "Response-timescale distributions across V1 locations",
        fontsize=14, fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        "Ten trial-matched 150-flash fits; retained common-QC neurons with ≥1 valid draw "
        "(1 ≤ τ ≤ 300 ms, fitted τ error < 20 ms, and >50 fitting-window spikes).",
        ha="center", va="top", fontsize=8.5, color="#555f67",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = load_timescales()
    means = session_probe_means(table)
    summary = summarize(table, means)
    table.to_csv(OUTPUT / "v1_timescale_neuron_draws_equal_weighted.csv", index=False)
    means.to_csv(OUTPUT / "v1_timescale_session_probe_means.csv", index=False)
    summary.to_csv(OUTPUT / "v1_timescale_location_distribution_summary.csv", index=False)
    render(table, means, summary, OUTPUT / "Figure_v1_timescale_location_distributions.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
