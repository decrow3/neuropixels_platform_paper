#!/usr/bin/env python3
"""Plot HVA session distributions, pair distances, and area-removal sensitivity."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure3_ttfs_landmark_scale import unbiased_squared_mean_distance  # noqa: E402


AREAS = ["LM", "RL", "AL", "PM", "AM"]
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
METRIC_TITLES = {
    "TTFS (ms)": "TTFS",
    "log10 F1/F0": "log₁₀ F1/F0",
    "Response timescale (ms)": "Response timescale",
}
METRIC_UNITS = {
    "TTFS (ms)": "Session-centered TTFS (ms)",
    "log10 F1/F0": "Session-centered log₁₀ F1/F0",
    "Response timescale (ms)": "Session-centered timescale (ms)",
}
PAIR_UNITS = {
    "TTFS (ms)": "ms²",
    "log10 F1/F0": "log₁₀²",
    "Response timescale (ms)": "ms²",
}
HIERARCHY_SCORES = {
    "LM": -0.093,
    "RL": -0.059,
    "LP": 0.105,
    "AL": 0.152,
    "PM": 0.327,
    "AM": 0.441,
}
AREA_COLORS = {
    "LM": "#4e73ae",
    "RL": "#65b2c9",
    "LP": "#58a76a",
    "AL": "#cab778",
    "PM": "#db8457",
    "AM": "#c24f54",
}
SESSION_MEANS = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
TTFS_AUDIT = ROOT / (
    "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance/"
    "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
)
EXTENSION_INPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/metric_extension_cortical_hvas/"
    "full_cell_metric_extension_input.csv"
)
DEFAULT_OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/area_driver_distributions_cortical_hvas"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def load_session_distributions() -> pd.DataFrame:
    table = pd.read_csv(SESSION_MEANS, dtype={"session_id": str})
    table = table.loc[
        table["dataset"].eq("Post-V1")
        & table["metric"].isin(METRICS)
        & table["group"].isin(AREAS)
    ].copy()
    if table.duplicated(["metric", "session_id", "group"]).any():
        raise ValueError("Duplicate session × area mean")
    return table


def load_full_cell_hva_units() -> pd.DataFrame:
    ttfs = pd.read_csv(TTFS_AUDIT, dtype={"session_id": str})
    ttfs = ttfs.loc[
        ttfs["area_coarse"].isin(AREAS)
        & ttfs["selected_positive_responder_area"].astype(bool)
        & pd.to_numeric(ttfs["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    counts = ttfs.groupby(["session_id", "area_coarse"])["unit_id"].transform("size")
    ttfs = ttfs.loc[counts.ge(10)].rename(
        columns={"area_coarse": "group", "preferred_0_250_ttfs_ms": "value"}
    )
    ttfs["metric"] = "TTFS (ms)"

    extension = pd.read_csv(EXTENSION_INPUT, dtype={"session_id": str})
    extension = extension.loc[
        extension["dataset"].eq("Post-V1")
        & extension["metric"].isin(METRICS[1:])
        & extension["group"].isin(AREAS)
    ].copy()
    columns = ["metric", "session_id", "group", "unit_id", "value"]
    result = pd.concat([ttfs[columns], extension[columns]], ignore_index=True)
    if result.duplicated(["metric", "session_id", "group", "unit_id"]).any():
        raise ValueError("Duplicate neuron within an HVA population")
    return result


def compute_pair_distances(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (metric, session_id), session in units.groupby(["metric", "session_id"]):
        populations = {
            group: part["value"].to_numpy(float)
            for group, part in session.groupby("group")
        }
        for group_a, group_b in combinations(sorted(populations), 2):
            estimate = unbiased_squared_mean_distance(
                populations[group_a], populations[group_b]
            )
            rows.append(
                {
                    "metric": metric,
                    "session_id": session_id,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(populations[group_a]),
                    "n_b": len(populations[group_b]),
                    **estimate,
                }
            )
    per_session = pd.DataFrame(rows)
    pair_means = (
        per_session.groupby(["metric", "group_a", "group_b"], as_index=False)
        .agg(
            corrected_distance=("corrected_squared_mean_distance", "mean"),
            naive_distance=("naive_squared_mean_difference", "mean"),
            sampling_noise_correction=("sampling_noise_correction", "mean"),
            sessions=("session_id", "nunique"),
        )
    )
    return per_session, pair_means


def compute_area_diagnostics(session_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = (
        session_means.groupby(["metric", "group"], as_index=False)
        .agg(
            centered_effect=("centered_mean", "mean"),
            sessions=("session_id", "nunique"),
            raw_mean=("mean", "mean"),
        )
    )
    rows = []
    for metric, part in effects.groupby("metric"):
        values = part.set_index("group")["centered_effect"].reindex(AREAS)
        full_variance = float(values.var(ddof=0))
        for area in AREAS:
            remaining = float(values.drop(area).var(ddof=0))
            rows.append(
                {
                    "metric": metric,
                    "omitted_area": area,
                    "full_area_variance": full_variance,
                    "remaining_area_variance": remaining,
                    "fraction_remaining": remaining / full_variance,
                }
            )
    return effects, pd.DataFrame(rows)


def center_and_weight_units(units: pd.DataFrame) -> pd.DataFrame:
    """Center within session and weight cells so each session contributes equally."""
    result = units.copy()
    session_centers = (
        result.groupby(["metric", "session_id", "group"], observed=True, as_index=False)
        .agg(_group_mean=("value", "mean"))
        .groupby(["metric", "session_id"], observed=True, as_index=False)
        .agg(_session_center=("_group_mean", "mean"))
    )
    result = result.merge(
        session_centers, on=["metric", "session_id"], how="left", validate="many_to_one"
    )
    result["session_centered_value"] = result["value"] - result.pop("_session_center")
    population_n = result.groupby(
        ["metric", "session_id", "group"], observed=True
    )["unit_id"].transform("size")
    session_n = result.groupby(["metric", "group"], observed=True)[
        "session_id"
    ].transform("nunique")
    result["equal_session_weight"] = 1.0 / (population_n * session_n)
    return result


def render_full_cell_ecdfs(units: pd.DataFrame, output: Path) -> None:
    palette = dict(zip(AREAS, plt.get_cmap("tab10").colors[: len(AREAS)]))
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        local = units.loc[units["metric"].eq(metric)]
        for area in AREAS:
            area_units = local.loc[local["group"].eq(area)].sort_values(
                "session_centered_value"
            )
            values = area_units["session_centered_value"].to_numpy(float)
            weights = area_units["equal_session_weight"].to_numpy(float)
            cumulative = np.cumsum(weights) / weights.sum()
            ax.plot(values, cumulative, color=palette[area], linewidth=1.8,
                    alpha=0.9, label=area)
        ax.axvline(0, color="#606870", linestyle="--", linewidth=1)
        ax.set_xlabel(METRIC_UNITS[metric])
        ax.set_ylabel("Cumulative fraction of neurons")
        ax.set_ylim(0, 1)
        ax.set_title(METRIC_TITLES[metric], loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(title="HVA", frameon=False, ncol=2, loc="lower right")
    figure.suptitle(
        "Full neuronal distributions across HVAs\n"
        "session-centered; cells weighted so every session contributes equally",
        fontweight="bold", fontsize=13,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_session_distributions(table: pd.DataFrame, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        local = table.loc[table["metric"].eq(metric)]
        data = [local.loc[local["group"].eq(area), "centered_mean"].to_numpy(float) for area in AREAS]
        boxes = ax.boxplot(
            data, positions=np.arange(len(AREAS)), widths=0.55, patch_artist=True,
            showfliers=False, medianprops={"color": "#24303a", "linewidth": 1.5},
            whiskerprops={"color": "#607080"}, capprops={"color": "#607080"},
        )
        for box in boxes["boxes"]:
            box.set(facecolor="#dce8f2", edgecolor="#52708a", linewidth=1.1)
        for index, (area, values) in enumerate(zip(AREAS, data)):
            jitter = rng.uniform(-0.18, 0.18, size=len(values))
            ax.scatter(index + jitter, values, s=17, color="#51789a", alpha=0.48,
                       edgecolors="none", rasterized=True)
            ax.scatter(index, np.mean(values), s=52, color="#c9793e", marker="D",
                       edgecolor="white", linewidth=0.7, zorder=4)
            ax.text(index, -0.13, f"n={len(values)}", ha="center", va="top",
                    transform=ax.get_xaxis_transform(), clip_on=False,
                    fontsize=7.5, color="#56616b")
        ax.axhline(0, color="#606870", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(AREAS)), AREAS)
        ax.set_ylabel(METRIC_UNITS[metric])
        ax.set_title(METRIC_TITLES[metric], loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Session-level HVA distributions by area\npoints are session × area means; diamonds are equal-session area means",
        fontweight="bold", fontsize=13,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_hierarchy_session_zoom(table: pd.DataFrame, output: Path, seed: int) -> None:
    """Probe-zoom-style view of HVA session means along hierarchy score."""
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(3, 1, figsize=(11.2, 10.5), sharex=True)
    figure.subplots_adjust(hspace=0.28, left=0.11, right=0.97, top=0.94, bottom=0.09)

    for ax, metric in zip(axes, METRICS):
        local = table.loc[table["metric"].eq(metric)]
        area_x, area_y = [], []
        for area in AREAS:
            values = local.loc[local["group"].eq(area), "mean"].to_numpy(float)
            x = HIERARCHY_SCORES[area]
            jitter = rng.uniform(-0.007, 0.007, size=len(values))
            ax.scatter(
                x + jitter, values, s=37, color=AREA_COLORS[area], alpha=0.55,
                edgecolor="#303840", linewidth=0.35, zorder=3, rasterized=True,
            )
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            ax.errorbar(
                x, mean, yerr=sem, fmt="o", markersize=9.5,
                color=AREA_COLORS[area], markeredgecolor="white", markeredgewidth=0.8,
                capsize=3, capthick=1.5, elinewidth=1.8, zorder=5,
            )
            area_x.append(x)
            area_y.append(mean)

        slope, intercept = np.polyfit(area_x, area_y, deg=1)
        fit_x = np.linspace(min(area_x) - 0.035, max(area_x) + 0.035, 100)
        ax.plot(
            fit_x, slope * fit_x + intercept, linestyle="--", color="#606870",
            linewidth=1.4, alpha=0.55, zorder=1,
        )
        ax.set_ylabel(METRIC_UNITS[metric].replace("Session-centered ", ""))
        ax.set_title(METRIC_TITLES[metric], loc="left", fontweight="bold", fontsize=11)
        ax.grid(axis="y", color="#d8dde1", linewidth=0.7, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor="#8aa0b1", markeredgecolor="#303840", alpha=0.6,
                   label="session × area mean"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                   markerfacecolor="#657b8b", markeredgecolor="white",
                   label="equal-session area mean ± SEM"),
            Line2D([0], [0], linestyle="--", color="#606870", alpha=0.55,
                   label=f"OLS through {len(AREAS)} cortical area means"),
        ],
        frameon=True, fontsize=8.5, loc="upper left",
    )
    axes[-1].set_xticks(
        [HIERARCHY_SCORES[area] for area in AREAS], AREAS, rotation=35, ha="right"
    )
    axes[-1].set_xlabel("Published inter-area hierarchy score")
    axes[-1].set_xlim(min(HIERARCHY_SCORES.values()) - 0.055,
                      max(HIERARCHY_SCORES.values()) + 0.055)
    figure.suptitle(
        "HVA session distributions along the published hierarchy",
        fontweight="bold", fontsize=13,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_pair_heatmaps(pair_means: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.1, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("PuOr_r")
    for ax, metric in zip(axes, METRICS):
        local = pair_means.loc[pair_means["metric"].eq(metric)]
        values = np.full((len(AREAS), len(AREAS)), np.nan)
        counts = np.zeros_like(values)
        for row in local.itertuples(index=False):
            i, j = AREAS.index(row.group_a), AREAS.index(row.group_b)
            values[i, j] = values[j, i] = row.corrected_distance
            counts[i, j] = counts[j, i] = row.sessions
        finite = values[np.isfinite(values)]
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = min(float(np.nanmin(finite)), -0.04 * vmax)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        image = ax.imshow(values, cmap=cmap, norm=norm)
        for i in range(len(AREAS)):
            for j in range(len(AREAS)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#525b63")
                elif np.isfinite(values[i, j]):
                    color = "white" if abs(norm(values[i, j]) - 0.5) > 0.32 else "#222222"
                    fmt = ".4f" if metric == "log10 F1/F0" else ".1f"
                    ax.text(j, i, f"{values[i, j]:{fmt}}\nn={int(counts[i, j])}",
                            ha="center", va="center", fontsize=7.2, color=color)
        ax.set_xticks(np.arange(len(AREAS)), AREAS)
        ax.set_yticks(np.arange(len(AREAS)), AREAS)
        ax.set_title(f"{METRIC_TITLES[metric]} ({PAIR_UNITS[metric]})", loc="left", fontweight="bold")
        colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.set_ylabel("Corrected squared distance", rotation=270, labelpad=12)
    figure.suptitle(
        "Full-cell HVA pair distances\ncell-sampling correction applied; n is co-recorded sessions per named pair",
        fontweight="bold", fontsize=13,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_leave_one_out(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        local = table.loc[table["metric"].eq(metric)].set_index("omitted_area").reindex(AREAS)
        values = 100 * local["fraction_remaining"].to_numpy(float)
        bars = ax.bar(np.arange(len(AREAS)), values, color="#6d8faa", edgecolor="#40596d")
        ax.axhline(100, color="#555d64", linestyle="--", linewidth=1, label="No change")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(AREAS)), AREAS)
        ax.set_ylabel("Area-mean variance remaining (%)")
        ax.set_title(METRIC_TITLES[metric], loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Leave-one-HVA-out sensitivity\nlower bars identify areas carrying more of the stable area-mean spread",
        fontweight="bold", fontsize=13,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session_means = load_session_distributions()
    units = load_full_cell_hva_units()
    centered_units = center_and_weight_units(units)
    per_session_pairs, pair_means = compute_pair_distances(units)
    area_effects, leave_one_out = compute_area_diagnostics(session_means)

    session_means.to_csv(output / "hva_session_area_distributions.csv", index=False)
    centered_units.to_csv(output / "hva_full_cell_centered_distributions.csv", index=False)
    area_effects.to_csv(output / "hva_area_centered_effects.csv", index=False)
    per_session_pairs.to_csv(output / "hva_full_cell_pair_distances_by_session.csv", index=False)
    pair_means.to_csv(output / "hva_full_cell_named_pair_means.csv", index=False)
    leave_one_out.to_csv(output / "hva_leave_one_area_out.csv", index=False)

    render_session_distributions(
        session_means, output / "Figure_hva_session_area_distributions.png", args.seed
    )
    render_hierarchy_session_zoom(
        session_means, output / "Figure_hva_hierarchy_session_zoom.png", args.seed
    )
    render_full_cell_ecdfs(
        centered_units, output / "Figure_hva_full_cell_distributions.png"
    )
    render_pair_heatmaps(pair_means, output / "Figure_hva_full_cell_pair_heatmaps.png")
    render_leave_one_out(leave_one_out, output / "Figure_hva_leave_one_area_out.png")
    print(area_effects.to_string(index=False))
    print(leave_one_out.to_string(index=False))


if __name__ == "__main__":
    main()
