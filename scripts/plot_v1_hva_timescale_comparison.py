#!/usr/bin/env python3
"""Compare response-timescale distributions across MouseV2 V1, Allen VISp, and HVAs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units  # noqa: E402
INPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/"
    "metric_extension_cortical_hvas/full_cell_metric_extension_input.csv"
)
OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "area_driver_distributions_cortical_hvas"
)
V1_GROUPS = ["B", "C", "A", "E"]
HVA_GROUPS = ["LM", "RL", "AL", "PM", "AM"]
V1_COLORS = {"B": "#4575b4", "C": "#1a9850", "A": "#d73027", "E": "#8073ac"}
HVA_COLORS = {
    "LM": "#4e73ae", "RL": "#65b2c9", "AL": "#cab778",
    "PM": "#db8457", "AM": "#c24f54",
}
VISP_COLOR = "#59636d"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cumulative, values))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(INPUT, dtype={"session_id": str, "unit_id": str})
    source = source.loc[source["metric"].eq("Response timescale (ms)")].copy()

    v1 = source.loc[
        source["dataset"].eq("Within-V1") & source["group"].isin(V1_GROUPS)
    ].copy()
    keys = ["session_id", "group", "unit_id", "draw_id"]
    if v1.duplicated(keys).any():
        raise ValueError("Duplicate V1 neuron × matched-draw rows")
    # Match the ten-draw point-estimation scheme: every draw receives 1/10 of
    # its session's weight, divided equally among the neurons valid in that draw.
    # A neuron valid in one draw therefore does not receive the same aggregate
    # weight as a neuron that contributes to all ten separate point fits.
    v1_draw_population_n = v1.groupby(
        ["group", "session_id", "draw_id"]
    )["unit_id"].transform("size")
    v1_draw_n = v1.groupby(["group", "session_id"])["draw_id"].transform("nunique")
    v1_session_n = v1.groupby("group")["session_id"].transform("nunique")
    v1["plot_weight"] = 1.0 / (
        v1_draw_population_n * v1_draw_n * v1_session_n
    )

    hva = source.loc[
        source["dataset"].eq("Post-V1") & source["group"].isin(HVA_GROUPS)
    ].copy()
    keys = ["session_id", "group", "unit_id"]
    if hva.duplicated(keys).any():
        raise ValueError("Duplicate HVA neuron rows")
    hva_neuron_n = hva.groupby(["group", "session_id"])["unit_id"].transform("size")
    hva_session_n = hva.groupby("group")["session_id"].transform("nunique")
    hva["plot_weight"] = 1.0 / (hva_neuron_n * hva_session_n)

    allen = load_allen_units(population_profile="common_qc")
    valid = pd.to_numeric(allen["timescale_ac"], errors="coerce").between(1, 300)
    valid &= pd.to_numeric(allen["spike_count_ac"], errors="coerce").gt(50)
    valid &= pd.to_numeric(allen["err_ac"], errors="coerce").lt(20)
    visp = allen.loc[valid & allen["area_coarse"].eq("V1")].copy()
    retained_sessions = visp.groupby("ecephys_session_id").size().loc[lambda x: x.ge(5)].index
    visp = visp.loc[visp["ecephys_session_id"].isin(retained_sessions)].rename(
        columns={
            "ecephys_session_id": "session_id",
            "ecephys_unit_id": "unit_id",
            "timescale_ac": "value",
        }
    )
    visp["session_id"] = visp["session_id"].astype(str)
    visp["unit_id"] = visp["unit_id"].astype(str)
    visp["group"] = "VISp"
    visp_neuron_n = visp.groupby("session_id")["unit_id"].transform("size")
    visp["plot_weight"] = 1.0 / (visp_neuron_n * visp["session_id"].nunique())

    for name, table, groups in [
        ("V1", v1, V1_GROUPS), ("VISp", visp, ["VISp"]), ("HVA", hva, HVA_GROUPS)
    ]:
        sums = table.groupby("group")["plot_weight"].sum().reindex(groups)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"{name} weights do not sum to one:\n{sums}")
    return v1, visp, hva


def session_means(
    v1: pd.DataFrame, visp: pd.DataFrame, hva: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v1_draw_means = (
        v1.groupby(["group", "session_id", "draw_id"], as_index=False)
        .agg(draw_mean_ms=("value", "mean"))
    )
    draw_counts = v1_draw_means.groupby(["group", "session_id"])["draw_id"].nunique()
    if not draw_counts.eq(10).all():
        raise ValueError("A retained V1 population lacks all ten matched draws")
    v1_means = (
        v1_draw_means.groupby(["group", "session_id"], as_index=False)
        .agg(mean_timescale_ms=("draw_mean_ms", "mean"))
    )
    hva_means = (
        hva.groupby(["group", "session_id"], as_index=False)
        .agg(mean_timescale_ms=("value", "mean"))
    )
    visp_means = (
        visp.groupby(["group", "session_id"], as_index=False)
        .agg(mean_timescale_ms=("value", "mean"))
    )
    return v1_means, visp_means, hva_means


def summarize(
    v1: pd.DataFrame, visp: pd.DataFrame, hva: pd.DataFrame,
    v1_means: pd.DataFrame, visp_means: pd.DataFrame, hva_means: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, table, means, groups in [
        ("Within-V1", v1, v1_means, V1_GROUPS),
        ("Allen VISp", visp, visp_means, ["VISp"]),
        ("Cortical HVAs", hva, hva_means, HVA_GROUPS),
    ]:
        for group in groups:
            local = table.loc[table["group"].eq(group)]
            values = local["value"].to_numpy(float)
            weights = local["plot_weight"].to_numpy(float)
            local_means = means.loc[means["group"].eq(group), "mean_timescale_ms"]
            rows.append({
                "dataset": dataset,
                "group": group,
                "unique_neurons": local["unit_id"].nunique(),
                "sessions": local["session_id"].nunique(),
                "weighted_q25_ms": weighted_quantile(values, weights, 0.25),
                "weighted_median_ms": weighted_quantile(values, weights, 0.50),
                "weighted_q75_ms": weighted_quantile(values, weights, 0.75),
                "session_mean_ms": local_means.mean(),
                "session_mean_sem_ms": local_means.sem(),
                "fraction_over_100_ms": np.average(values > 100, weights=weights),
                "fraction_over_150_ms": np.average(values > 150, weights=weights),
            })
    summary = pd.DataFrame(rows)
    spans = summary.groupby("dataset")["weighted_median_ms"].agg(lambda x: x.max() - x.min())
    summary["dataset_median_span_ms"] = summary["dataset"].map(spans)
    return summary


def draw_ridges(
    ax: plt.Axes, table: pd.DataFrame, summary: pd.DataFrame,
    groups: list[str], colors: dict[str, str], ylabel: str,
) -> None:
    grid = np.linspace(0, 250, 700)
    for row, group in enumerate(groups[::-1]):
        local = table.loc[table["group"].eq(group)]
        values = local["value"].to_numpy(float)
        weights = local["plot_weight"].to_numpy(float)
        density = gaussian_kde(values, weights=weights, bw_method=0.20)(grid)
        density = 0.76 * density / density.max()
        ax.fill_between(
            grid, row, row + density, color=colors[group], alpha=0.40,
            edgecolor=colors[group], linewidth=1.0,
        )
        ax.plot(grid, row + density, color=colors[group], linewidth=1.3)
        record = summary.loc[summary["group"].eq(group)].iloc[0]
        ax.vlines(
            record["weighted_median_ms"], row, row + 0.68,
            color=colors[group], linewidth=2.0,
        )
        ax.text(
            247, row + 0.28,
            f"n={int(record['unique_neurons']):,}; {int(record['sessions'])} sessions",
            ha="right", va="center", fontsize=7.5, color="#515b63",
        )
    ax.set_yticks(np.arange(len(groups)), groups[::-1])
    ax.set_xlim(0, 250)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", color="#dfe3e6", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_session_panel(
    ax: plt.Axes, means: pd.DataFrame, groups: list[str],
    colors: dict[str, str], seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for index, group in enumerate(groups):
        values = means.loc[means["group"].eq(group), "mean_timescale_ms"].to_numpy(float)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(
            index + jitter, values, s=20, color=colors[group], alpha=0.42,
            edgecolor="#38434c", linewidth=0.3, rasterized=True,
        )
        ax.errorbar(
            index, np.mean(values), yerr=np.std(values, ddof=1) / np.sqrt(len(values)),
            fmt="o", markersize=8, color=colors[group], markeredgecolor="white",
            markeredgewidth=0.7, capsize=3, elinewidth=1.6, zorder=4,
        )
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylim(20, 80)
    ax.set_ylabel("Session × location mean τ (ms)")
    ax.grid(axis="y", color="#dfe3e6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def render(
    v1: pd.DataFrame, visp: pd.DataFrame, hva: pd.DataFrame,
    v1_means: pd.DataFrame, visp_means: pd.DataFrame, hva_means: pd.DataFrame,
    summary: pd.DataFrame, output: Path,
) -> None:
    figure, axes = plt.subplots(
        3, 2, figsize=(13.0, 12.0),
        gridspec_kw={
            "width_ratios": [1.48, 1.0], "height_ratios": [4.0, 1.65, 5.0],
            "hspace": 0.35, "wspace": 0.16,
        },
    )
    v1_density_ax, v1_session_ax = axes[0]
    visp_density_ax, visp_session_ax = axes[1]
    hva_density_ax, hva_session_ax = axes[2]

    draw_ridges(
        v1_density_ax, v1, summary.loc[summary["dataset"].eq("Within-V1")],
        V1_GROUPS, V1_COLORS, "V1 probe/location",
    )
    draw_ridges(
        visp_density_ax, visp, summary.loc[summary["dataset"].eq("Allen VISp")],
        ["VISp"], {"VISp": VISP_COLOR}, "Allen primary cortex",
    )
    draw_ridges(
        hva_density_ax, hva, summary.loc[summary["dataset"].eq("Cortical HVAs")],
        HVA_GROUPS, HVA_COLORS, "Cortical HVA",
    )
    draw_session_panel(v1_session_ax, v1_means, V1_GROUPS, V1_COLORS, 20260826)
    draw_session_panel(
        visp_session_ax, visp_means, ["VISp"], {"VISp": VISP_COLOR}, 20260828
    )
    draw_session_panel(hva_session_ax, hva_means, HVA_GROUPS, HVA_COLORS, 20260827)

    v1_density_ax.set_title("V1: full neuronal distributions", loc="left", fontweight="bold")
    v1_session_ax.set_title("V1: session-level replication", loc="left", fontweight="bold")
    visp_density_ax.set_title(
        "Allen VISp: full neuronal distribution", loc="left", fontweight="bold"
    )
    visp_session_ax.set_title(
        "Allen VISp: session-level replication", loc="left", fontweight="bold"
    )
    hva_density_ax.set_title("HVAs: full neuronal distributions", loc="left", fontweight="bold")
    hva_session_ax.set_title("HVAs: session-level replication", loc="left", fontweight="bold")
    v1_density_ax.tick_params(axis="x", labelbottom=False)
    v1_density_ax.set_xlabel("")
    visp_density_ax.tick_params(axis="x", labelbottom=False)
    visp_density_ax.set_xlabel("")
    hva_density_ax.set_xlabel("Neuron response timescale, τ (ms)")

    spans = summary.groupby("dataset")["dataset_median_span_ms"].first()
    figure.suptitle(
        "Response-timescale distributions: MouseV2 V1, Allen VISp, and cortical HVAs",
        fontsize=14, fontweight="bold", y=0.985,
    )
    figure.text(
        0.5, 0.953,
        f"Common axes. Group-median span: MouseV2 V1 {spans['Within-V1']:.1f} ms; "
        f"HVAs {spans['Cortical HVAs']:.1f} ms. Vertical segments are weighted medians.",
        ha="center", va="center", fontsize=9.5, color="#4f5962",
    )
    figure.text(
        0.5, 0.012,
        "Densities give every session equal weight. MouseV2 gives each matched draw equal weight and then weights its valid neurons equally, "
        "matching the ten separate point fits. Allen populations use common QC and identical timescale-fit validity rules.",
        ha="center", va="bottom", fontsize=8.3, color="#555f67",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    v1, visp, hva = load_data()
    v1_means, visp_means, hva_means = session_means(v1, visp, hva)
    summary = summarize(v1, visp, hva, v1_means, visp_means, hva_means)
    summary.to_csv(
        OUTPUT / "v1_visp_hva_timescale_comparison_model_aligned_summary.csv",
        index=False,
    )
    render(
        v1, visp, hva, v1_means, visp_means, hva_means, summary,
        OUTPUT / "Figure_v1_visp_hva_timescale_comparison_model_aligned.png",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
