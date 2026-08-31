"""Build a multi-page review PDF of manuscript Figure 4 variants.

Every page uses the same frozen session-level inputs.  The variants change only
the visual hierarchy and amount of supporting evidence shown on the page.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import gaussian_kde, linregress


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure3_robust_spread_comparison import clustered_omega_squared


matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.2,
    "xtick.labelsize": 9.1,
    "ytick.labelsize": 9.1,
})

INK = "#222222"
MUTED = "#62666d"
GRID = "#e7e8eb"
V1_COLOR = "#7567a8"
HVA_COLOR = "#54575b"
DELTA_COLOR = "#315d9b"
PROBE_COLORS = {"B": "#4575b4", "C": "#1a9850", "A": "#d73027", "E": "#8073ac"}
AREA_COLORS = {"LM": "#4e73ae", "RL": "#65b2c9", "AL": "#cab778", "PM": "#db8457", "AM": "#c24f54"}
PROBE_ORDER = ["B", "C", "A", "E"]
AREA_ORDER = ["LM", "RL", "AL", "PM", "AM"]
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
SHORT_METRICS = ["TTFS", "log10 F1/F0", "Timescale"]
HIERARCHY_SCORES = {
    "VISp": -0.357, "LM": -0.093, "RL": -0.059,
    "AL": 0.152, "PM": 0.327, "AM": 0.441,
}
CELL_MODEL_ROOT = ROOT / "artifacts/figure3/07_big_picture_concrete_first/full_cell_model"
TTFS_CELL_INPUT = CELL_MODEL_ROOT / "ttfs_cortical_hvas/ttfs_full_cell_input.csv"
EXTENSION_CELL_INPUT = (
    CELL_MODEL_ROOT / "metric_extension_cortical_hvas/full_cell_metric_extension_input.csv"
)
MOUSEV2_ANATOMICAL_PENETRATIONS = (
    ROOT / "artifacts/figure3/06j_mousev2_area_borders_registered_to_zhuang/"
    "probe_anatomical_position.csv"
)
ZHUANG_TEMPLATE = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
)
DEFAULT_OPEN_SCOPE_SCHEMATIC = Path(
    "/home/huklaban5/.codex/attachments/5c48ada5-6e59-4d03-a56a-533e9b282506/"
    "OpenScopeImagingSchematic.png"
)
ZHUANG_AREA_SEEDS = {
    "V1": (200, 240), "LM": (100, 260), "AL": (75, 190),
    "RL": (180, 80), "AM": (240, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Figure3/Figure4_hierarchy_figure_variants.pdf",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schematic", type=Path, default=DEFAULT_OPEN_SCOPE_SCHEMATIC,
        help="OpenScope imaging schematic placed in the lower-left candidate panel",
    )
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(ROOT / "Figure3/Figure3_robust_session_group_means.csv")
    allen_v1 = pd.read_csv(ROOT / "Figure3/Figure3_Allen_V1_session_means.csv")
    hierarchy = pd.read_csv(ROOT / "Figure3/Figure3_hierarchy_fit_stats.csv")
    groups = groups.loc[groups["group"].isin(PROBE_ORDER + AREA_ORDER)].copy()
    allen_v1 = allen_v1.copy()
    allen_v1["group"] = "VISp"
    return groups, allen_v1, hierarchy


def load_mousev2_penetration_map() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load anatomy-registered MouseV2 V1 penetrations and the common-map outline."""
    penetrations = pd.read_csv(MOUSEV2_ANATOMICAL_PENETRATIONS)
    expected_probes = set(PROBE_ORDER)
    if (
        len(penetrations) != 32
        or penetrations["animal"].nunique() != 8
        or set(penetrations["probe"]) != expected_probes
        or penetrations.groupby("animal")["probe"].nunique().ne(4).any()
    ):
        raise ValueError("Unexpected MouseV2 anatomical-penetration inventory")
    if not np.isfinite(penetrations[["zhuang_col", "zhuang_row"]]).all().all():
        raise ValueError("MouseV2 common-map penetration coordinates contain non-finite values")

    source = np.load(ZHUANG_TEMPLATE)
    boundary = source["mean_field_sign_boundary"].astype(bool)
    domain = ndimage.binary_fill_holes(boundary)
    walls = ndimage.binary_dilation(boundary, iterations=1)
    components, _ = ndimage.label(domain & ~walls)
    seed_x, seed_y = ZHUANG_AREA_SEEDS["V1"]
    visp_component = int(components[seed_y, seed_x])
    if visp_component == 0:
        raise ValueError("Zhuang V1 seed falls on a common-map boundary")
    visp_mask = components == visp_component
    return penetrations, boundary, visp_mask


def load_weighted_cells(groups: pd.DataFrame) -> pd.DataFrame:
    """Load validated cells and give every session and neuron equal group weight."""
    ttfs = pd.read_csv(
        TTFS_CELL_INPUT, dtype={"session_id": str, "unit_id": str}
    )
    ttfs["draw_id"] = -1
    extension = pd.read_csv(
        EXTENSION_CELL_INPUT, dtype={"session_id": str, "unit_id": str}
    )
    columns = [
        "dataset", "metric", "session_id", "group", "unit_id", "value", "draw_id"
    ]
    cells = pd.concat([ttfs[columns], extension[columns]], ignore_index=True)
    cells = cells.loc[
        cells["group"].isin(PROBE_ORDER + AREA_ORDER)
        & cells["metric"].isin(METRICS)
    ].copy()
    duplicate_key = ["dataset", "metric", "session_id", "group", "unit_id", "draw_id"]
    if cells.duplicated(duplicate_key).any():
        raise ValueError("Duplicate neuron/draw row in full-cell inputs")

    # Use the frozen session centers behind the primary session-level figure.
    centers = groups[
        ["dataset", "metric", "session_id", "group", "session_mean"]
    ].drop_duplicates(["dataset", "metric", "session_id", "group"])
    centers["session_id"] = centers["session_id"].astype(str)
    cells = cells.merge(
        centers,
        on=["dataset", "metric", "session_id", "group"],
        how="inner",
        validate="many_to_one",
    )
    cells["centered_value"] = cells["value"] - cells["session_mean"]

    neuron_draw_n = cells.groupby(
        ["dataset", "metric", "session_id", "group", "unit_id"]
    )["draw_id"].transform("nunique")
    neuron_n = cells.groupby(
        ["dataset", "metric", "session_id", "group"]
    )["unit_id"].transform("nunique")
    session_n = cells.groupby(
        ["dataset", "metric", "group"]
    )["session_id"].transform("nunique")
    cells["equal_session_neuron_weight"] = 1.0 / (
        neuron_draw_n * neuron_n * session_n
    )
    weight_sums = cells.groupby(
        ["dataset", "metric", "group"]
    )["equal_session_neuron_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(float), 1.0):
        raise ValueError(f"Cell distribution weights do not sum to one:\n{weight_sums}")
    return cells


def identity_statistics(
    groups: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        local = groups.loc[groups["metric"].eq(metric)]
        v1, v1_draws = clustered_omega_squared(
            local.loc[local["dataset"].eq("Within-V1")],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        hva, hva_draws = clustered_omega_squared(
            local.loc[local["dataset"].eq("Post-V1")],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        delta_draws = hva_draws - v1_draws
        result[metric] = {
            "v1": v1,
            "v1_low": float(np.quantile(v1_draws, 0.025)),
            "v1_high": float(np.quantile(v1_draws, 0.975)),
            "hva": hva,
            "hva_low": float(np.quantile(hva_draws, 0.025)),
            "hva_high": float(np.quantile(hva_draws, 0.975)),
            "delta": hva - v1,
            "delta_low": float(np.quantile(delta_draws, 0.025)),
            "delta_high": float(np.quantile(delta_draws, 0.975)),
        }
    return result


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors="#333333")


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    ci = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return mean, ci


def plot_group_means(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    centered: bool = False,
    connect_sessions: bool = False,
    title: str | None = None,
) -> None:
    value_col = "centered_mean" if centered else "mean"
    positions = {group: index for index, group in enumerate(order)}
    if connect_sessions:
        for _, session in data.groupby(["dataset", "session_id"], sort=True):
            session = session.loc[session["group"].isin(order)].copy()
            session["x"] = session["group"].map(positions)
            session = session.sort_values("x")
            if len(session) > 1:
                ax.plot(session["x"], session[value_col], color="#777777", alpha=0.13, lw=0.7, zorder=1)
    for group in order:
        values = data.loc[data["group"].eq(group), value_col].dropna().to_numpy(float)
        if not len(values):
            continue
        x = positions[group]
        jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            x + jitter, values, s=22, color=colors[group], alpha=0.66,
            edgecolor="white", linewidth=0.45, zorder=3,
        )
        mean, ci = mean_ci(values)
        ax.errorbar(x, mean, yerr=ci, fmt="_", markersize=18, mew=2.8,
                    color=colors[group], capsize=3, lw=1.5, zorder=4)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.55, len(order) - 0.45)
    if centered:
        ax.axhline(0, color="#777777", ls="--", lw=0.9, zorder=1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
    style_axis(ax)


def hierarchy_data(groups: pd.DataFrame, allen_v1: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pd.concat([
        allen_v1.loc[allen_v1["metric"].eq(metric)],
        groups.loc[groups["metric"].eq(metric) & groups["dataset"].eq("Post-V1")],
    ], ignore_index=True)


def plot_hierarchy_profile(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    metric: str,
    hierarchy_stats: pd.DataFrame,
    show_post_v1_fit: bool = True,
) -> None:
    order = ["VISp", *AREA_ORDER]
    colors = {"VISp": INK, **AREA_COLORS}
    centers: dict[str, float] = {}
    for group in order:
        values = data.loc[data["group"].eq(group), "mean"].dropna().to_numpy(float)
        if not len(values):
            continue
        x = HIERARCHY_SCORES[group]
        jitter = np.linspace(-0.012, 0.012, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(x + jitter, values, s=12, color=colors[group], alpha=0.18,
                   edgecolor="none", zorder=2)
        mean, ci = mean_ci(values)
        centers[group] = mean
        ax.errorbar(x, mean, yerr=ci, fmt="o", ms=6.3, color=colors[group],
                    markerfacecolor="white", markeredgewidth=1.6, capsize=3,
                    lw=1.5, zorder=4)
    all_groups = [g for g in order if g in centers]
    xs = np.array([HIERARCHY_SCORES[g] for g in all_groups])
    ys = np.array([centers[g] for g in all_groups])
    fit = linregress(xs, ys)
    line_x = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 100)
    ax.plot(line_x, fit.intercept + fit.slope * line_x, color="#555555", ls="--", lw=1.4,
            label="VISp + HVA fit")
    if show_post_v1_fit:
        hxs = np.array([HIERARCHY_SCORES[g] for g in AREA_ORDER if g in centers])
        hys = np.array([centers[g] for g in AREA_ORDER if g in centers])
        hfit = linregress(hxs, hys)
        hx = np.linspace(hxs.min(), hxs.max(), 100)
        ax.plot(hx, hfit.intercept + hfit.slope * hx, color=DELTA_COLOR, lw=1.3,
                label="HVA-only fit")
    ax.set_xticks([HIERARCHY_SCORES[g] for g in order], order, rotation=28, ha="right")
    ax.set_xlim(-0.405, 0.485)
    style_axis(ax)
    row = hierarchy_stats.loc[
        hierarchy_stats["metric"].eq(metric) & hierarchy_stats["scope"].eq("post_V1_only")
    ].iloc[0]
    ax.text(
        0.02, 0.96,
        f"HVA slope {row.slope_per_hierarchy_score:+.3g} "
        f"[{row.bootstrap_95ci_low:+.3g}, {row.bootstrap_95ci_high:+.3g}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED,
    )


def plot_delta_metric(
    ax: plt.Axes,
    stats: dict[str, float],
    label: str,
    *,
    annotate: bool = False,
) -> None:
    ax.axvline(0, color="#777777", lw=0.9, ls="--", zorder=0)
    ax.errorbar(
        stats["delta"], 0,
        xerr=[[stats["delta"] - stats["delta_low"]], [stats["delta_high"] - stats["delta"]]],
        fmt="o", ms=6, color=DELTA_COLOR, capsize=3, lw=1.6,
    )
    if annotate:
        ax.text(
            stats["delta"], 0.25, f"{stats['delta']:+.2f}",
            ha="center", va="bottom", fontsize=8.0, color=DELTA_COLOR,
            fontweight="bold",
        )
    ax.set_yticks([0], [label])
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlim(-0.36, 0.36)
    style_axis(ax, grid=False)


def plot_identity_forest(ax: plt.Axes, identity: dict[str, dict[str, float]]) -> None:
    ys = np.arange(len(METRICS))[::-1]
    for index, (y, metric) in enumerate(zip(ys, METRICS)):
        local = identity[metric]
        ax.errorbar(
            local["v1"], y + 0.11,
            xerr=[[local["v1"] - local["v1_low"]], [local["v1_high"] - local["v1"]]],
            fmt="o", ms=6.5, mfc="white", mec=V1_COLOR, mew=1.5,
            color=V1_COLOR, capsize=3, lw=1.5,
            label="Within-V1 probe identity" if index == 0 else None,
        )
        ax.errorbar(
            local["hva"], y - 0.11,
            xerr=[[local["hva"] - local["hva_low"]], [local["hva_high"] - local["hva"]]],
            fmt="s", ms=6, mfc=HVA_COLOR, mec=HVA_COLOR,
            color=HVA_COLOR, capsize=3, lw=1.5,
            label="HVA identity" if index == 0 else None,
        )
    ax.axvline(0, color="#888888", lw=0.9, ls="--")
    ax.set_yticks(ys, SHORT_METRICS)
    ax.set_xlabel("Stable identity effect (ω²)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax, grid=False)


def plot_delta_forest(ax: plt.Axes, identity: dict[str, dict[str, float]]) -> None:
    ys = np.arange(len(METRICS))[::-1]
    for y, metric in zip(ys, METRICS):
        local = identity[metric]
        ax.errorbar(
            local["delta"], y,
            xerr=[[local["delta"] - local["delta_low"]], [local["delta_high"] - local["delta"]]],
            fmt="o", ms=6.5, color=DELTA_COLOR, capsize=3, lw=1.6,
        )
    ax.axvline(0, color="#666666", lw=1.0, ls="--")
    ax.set_yticks(ys, SHORT_METRICS)
    ax.set_xlabel("Δω² (HVA − V1)")
    ax.set_xlim(-0.36, 0.36)
    style_axis(ax, grid=False)


def add_page_header(fig: plt.Figure, variant: str, subtitle: str) -> None:
    fig.suptitle(variant, x=0.055, y=0.985, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.055, 0.952, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def add_footer(fig: plt.Figure) -> None:
    fig.text(
        0.055, 0.018,
        "Session-level estimates; five cortical HVAs (LM, RL, AL, PM, AM); LP excluded. "
        "Identity-effect intervals use whole-session bootstrap. V1 probes are categorical locations, not hierarchy scores.",
        ha="left", va="bottom", fontsize=7.6, color=MUTED,
    )


def variant_a(
    groups: pd.DataFrame, allen_v1: pd.DataFrame, hierarchy_stats: pd.DataFrame,
    identity: dict[str, dict[str, float]],
) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(3, 3, left=0.055, right=0.975, bottom=0.085, top=0.89,
                          width_ratios=[1.05, 1.35, 0.70], hspace=0.40, wspace=0.28)
    add_page_header(fig, "Variant A — hierarchy-led hybrid",
                    "Raw session means and hierarchy-score profiles lead; the direct HVA−V1 test remains visible.")
    for row, (metric, short) in enumerate(zip(METRICS, SHORT_METRICS)):
        v1_ax = fig.add_subplot(gs[row, 0])
        v1 = pd.concat([
            groups.loc[groups["metric"].eq(metric) & groups["dataset"].eq("Within-V1")],
            allen_v1.loc[allen_v1["metric"].eq(metric)],
        ], ignore_index=True)
        plot_group_means(v1_ax, v1, [*PROBE_ORDER, "VISp"],
                         {**PROBE_COLORS, "VISp": INK})
        v1_ax.axvline(3.5, color="#999999", ls="--", lw=0.8)
        v1_ax.set_ylabel(metric)
        if row == 0:
            v1_ax.set_title("A  V1 locations and external VISp reference", loc="left", fontweight="bold")
        profile_ax = fig.add_subplot(gs[row, 1])
        plot_hierarchy_profile(profile_ax, hierarchy_data(groups, allen_v1, metric),
                               metric=metric, hierarchy_stats=hierarchy_stats)
        if row == 0:
            profile_ax.set_title("B  Visual areas by published hierarchy score", loc="left", fontweight="bold")
            profile_ax.legend(frameon=False, fontsize=7.4, loc="lower right")
        delta_ax = fig.add_subplot(gs[row, 2])
        plot_delta_metric(delta_ax, identity[metric], short)
        if row == 0:
            delta_ax.set_title("C  Direct identity contrast", loc="left", fontweight="bold")
        if row < 2:
            delta_ax.tick_params(axis="x", labelbottom=False)
            delta_ax.set_xlabel("")
    add_footer(fig)
    return fig


def variant_b(groups: pd.DataFrame, identity: dict[str, dict[str, float]]) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(4, 2, left=0.065, right=0.975, bottom=0.085, top=0.88,
                          height_ratios=[1.05, 1, 1, 1], width_ratios=[1.45, 1],
                          hspace=0.58, wspace=0.30)
    add_page_header(fig, "Variant B — benchmark-led",
                    "The inferential comparison leads; session-centered data show the observations behind it.")
    omega_ax = fig.add_subplot(gs[0, 0])
    plot_identity_forest(omega_ax, identity)
    omega_ax.set_title("A  Stable group-identity effects", loc="left", fontweight="bold")
    delta_ax = fig.add_subplot(gs[0, 1])
    plot_delta_forest(delta_ax, identity)
    delta_ax.set_title("B  Direct HVA−V1 contrast", loc="left", fontweight="bold")
    for row, metric in enumerate(METRICS, start=1):
        ax = fig.add_subplot(gs[row, :])
        local = groups.loc[groups["metric"].eq(metric)]
        plot_group_means(ax, local, [*PROBE_ORDER, *AREA_ORDER],
                         {**PROBE_COLORS, **AREA_COLORS}, centered=True,
                         connect_sessions=True,
                         title=f"C{row}  {metric}: session-centered location/area means")
        ax.axvline(3.5, color="#aaaaaa", lw=0.9)
        ax.set_ylabel("Centered value")
    add_footer(fig)
    return fig


def variant_c(groups: pd.DataFrame, identity: dict[str, dict[str, float]]) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(3, 3, left=0.055, right=0.975, bottom=0.085, top=0.89,
                          width_ratios=[1, 1.15, 0.85], hspace=0.42, wspace=0.28)
    add_page_header(fig, "Variant C — matched raw-data comparison",
                    "Within-V1 and across-HVA session means share row-wise scales; identity effects provide the summary.")
    for row, (metric, short) in enumerate(zip(METRICS, SHORT_METRICS)):
        v1_ax = fig.add_subplot(gs[row, 0])
        hva_ax = fig.add_subplot(gs[row, 1], sharey=v1_ax)
        local = groups.loc[groups["metric"].eq(metric)]
        plot_group_means(v1_ax, local.loc[local["dataset"].eq("Within-V1")], PROBE_ORDER,
                         PROBE_COLORS, connect_sessions=True)
        plot_group_means(hva_ax, local.loc[local["dataset"].eq("Post-V1")], AREA_ORDER,
                         AREA_COLORS, connect_sessions=True)
        v1_ax.set_ylabel(metric)
        hva_ax.tick_params(axis="y", labelleft=False)
        if row == 0:
            v1_ax.set_title("A  Within V1", loc="left", fontweight="bold")
            hva_ax.set_title("B  Across cortical HVAs", loc="left", fontweight="bold")
        effect_ax = fig.add_subplot(gs[row, 2])
        local_stats = identity[metric]
        effect_ax.errorbar(
            local_stats["v1"], 0.16,
            xerr=[[local_stats["v1"] - local_stats["v1_low"]],
                  [local_stats["v1_high"] - local_stats["v1"]]],
            fmt="o", ms=6.2, mfc="white", mec=V1_COLOR, mew=1.5,
            color=V1_COLOR, capsize=3, lw=1.5,
        )
        effect_ax.errorbar(
            local_stats["hva"], -0.16,
            xerr=[[local_stats["hva"] - local_stats["hva_low"]],
                  [local_stats["hva_high"] - local_stats["hva"]]],
            fmt="s", ms=5.8, color=HVA_COLOR, capsize=3, lw=1.5,
        )
        effect_ax.axvline(0, color="#888888", ls="--", lw=0.9)
        effect_ax.set_yticks([0.16, -0.16], ["V1", "HVA"])
        effect_ax.set_ylim(-0.65, 0.65)
        effect_ax.set_xlabel("ω²" if row == 2 else "")
        if row < 2:
            effect_ax.tick_params(axis="x", labelbottom=False)
        effect_ax.text(0.98, 0.92, short, transform=effect_ax.transAxes,
                       ha="right", va="top", fontsize=8.2, color=MUTED)
        style_axis(effect_ax, grid=False)
        if row == 0:
            effect_ax.set_title("C  Identity effects", loc="left", fontweight="bold")
    add_footer(fig)
    return fig


def variant_d(hierarchy_stats: pd.DataFrame, identity: dict[str, dict[str, float]]) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 8.6))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.96, bottom=0.17, top=0.80, wspace=0.34)
    add_page_header(fig, "Variant D — compact manuscript summary",
                    "Two uncertainty panels for a small main-text footprint; raw session-level panels would move to Extended Data.")
    slope_grid = gs[0, 0].subgridspec(3, 1, hspace=0.58)
    offsets = {"VISp_plus_post_V1": 0.11, "post_V1_only": -0.11}
    styles = {
        "VISp_plus_post_V1": (HVA_COLOR, "o", "VISp + HVAs"),
        "post_V1_only": (DELTA_COLOR, "s", "HVAs only"),
    }
    for index, (metric, short) in enumerate(zip(METRICS, SHORT_METRICS)):
        slope_ax = fig.add_subplot(slope_grid[index, 0])
        rows = hierarchy_stats.loc[hierarchy_stats["metric"].eq(metric)]
        for scope in ["VISp_plus_post_V1", "post_V1_only"]:
            row = rows.loc[rows["scope"].eq(scope)].iloc[0]
            color, marker, label = styles[scope]
            value = row.slope_per_hierarchy_score
            slope_ax.errorbar(
                value, offsets[scope],
                xerr=[[value - row.bootstrap_95ci_low], [row.bootstrap_95ci_high - value]],
                fmt=marker, ms=6.2, color=color, capsize=3, lw=1.6,
                mfc="white" if scope == "VISp_plus_post_V1" else color,
                label=label if index == 0 else None,
            )
        low = min(0.0, float(rows["bootstrap_95ci_low"].min()))
        high = max(0.0, float(rows["bootstrap_95ci_high"].max()))
        pad = max((high - low) * 0.12, 0.03)
        slope_ax.set_xlim(low - pad, high + pad)
        slope_ax.set_ylim(-0.45, 0.45)
        slope_ax.set_yticks([])
        slope_ax.axvline(0, color="#777777", lw=0.9, ls="--")
        slope_ax.set_title(short, loc="left", fontsize=9.5, fontweight="bold")
        slope_ax.set_xlabel("Slope per hierarchy-score unit" if index == 2 else "")
        if index < 2:
            slope_ax.tick_params(axis="x", labelbottom=False)
        if index == 0:
            slope_ax.text(0, 1.40, "A  Hierarchy-score associations", transform=slope_ax.transAxes,
                          ha="left", va="bottom", fontsize=12, fontweight="bold")
            slope_ax.legend(frameon=False, fontsize=8.2, loc="lower right")
        style_axis(slope_ax, grid=False)

    delta_ax = fig.add_subplot(gs[0, 1])
    plot_delta_forest(delta_ax, identity)
    delta_ax.set_title("B  Does HVA identity exceed within-V1 identity?", loc="left",
                       fontsize=12, fontweight="bold")
    add_footer(fig)
    return fig


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    return float(np.interp(q, cumulative, sorted_values))


def plot_cell_ridges(
    ax: plt.Axes,
    cells: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    xlim: tuple[float, float],
    title: str,
) -> None:
    grid = np.linspace(*xlim, 500)
    for row, group in enumerate(order[::-1]):
        local = cells.loc[cells["group"].eq(group)]
        values = local["value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        density = gaussian_kde(values, weights=weights, bw_method=0.20)(grid)
        density = 0.78 * density / density.max()
        ax.fill_between(grid, row, row + density, color=colors[group], alpha=0.36,
                        edgecolor=colors[group], linewidth=1.0)
        ax.plot(grid, row + density, color=colors[group], lw=1.3)
        median = weighted_quantile(values, weights, 0.5)
        ax.vlines(median, row, row + 0.68, color=colors[group], lw=2.0)
        neurons = local["unit_id"].nunique()
        sessions = local["session_id"].nunique()
        ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), row + 0.28,
                f"n={neurons:,}; {sessions} sessions", ha="right", va="center",
                fontsize=7.3, color=MUTED)
    ax.set_yticks(np.arange(len(order)), order[::-1])
    ax.set_xlim(*xlim)
    ax.set_xlabel("Neuron response timescale, τ (ms)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_session_summary(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    title: str,
) -> None:
    rng = np.random.default_rng(20260829)
    for x, group in enumerate(order):
        values = data.loc[data["group"].eq(group), "mean"].to_numpy(float)
        jitter = rng.uniform(-0.15, 0.15, len(values))
        ax.scatter(x + jitter, values, s=21, color=colors[group], alpha=0.46,
                   edgecolor="#38434c", linewidth=0.3, zorder=3)
        mean, ci = mean_ci(values)
        ax.errorbar(x, mean, yerr=ci, fmt="o", ms=7.5, color=colors[group],
                    markeredgecolor="white", markeredgewidth=0.7, capsize=3,
                    lw=1.5, zorder=4)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax)


def plot_weighted_ecdfs(
    ax: plt.Axes,
    cells: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    xlim: tuple[float, float],
    title: str,
    legend: bool = False,
) -> None:
    for group in order:
        local = cells.loc[cells["group"].eq(group)].sort_values("centered_value")
        values = local["centered_value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        cumulative = np.cumsum(weights) / np.sum(weights)
        ax.plot(values, cumulative, color=colors[group], lw=1.7, alpha=0.92, label=group)
    ax.axvline(0, color="#777777", ls="--", lw=0.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax, grid=False)
    if legend:
        ax.legend(frameon=False, fontsize=7.6, ncol=min(5, len(order)), loc="lower right")


def plot_session_violins(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    title: str | None = None,
) -> None:
    values = [data.loc[data["group"].eq(group), "mean"].to_numpy(float) for group in order]
    violins = ax.violinplot(values, positions=np.arange(len(order)), widths=0.76,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, group in zip(violins["bodies"], order):
        body.set_facecolor(colors[group])
        body.set_edgecolor(colors[group])
        body.set_alpha(0.16)
    rng = np.random.default_rng(20260829)
    for x, (group, local_values) in enumerate(zip(order, values)):
        jitter = rng.uniform(-0.15, 0.15, len(local_values))
        ax.scatter(x + jitter, local_values, s=18, color=colors[group], alpha=0.56,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean, ci = mean_ci(local_values)
        ax.errorbar(x, mean, yerr=ci, fmt="_", markersize=16, mew=2.5,
                    color=colors[group], capsize=3, lw=1.4, zorder=4)
    ax.set_xticks(np.arange(len(order)), order)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax)


def plot_half_violins(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
) -> None:
    """Faint right-half violins with raw session observations offset left."""
    values = [data.loc[data["group"].eq(group), "mean"].to_numpy(float) for group in order]
    positions = np.arange(len(order), dtype=float)
    violins = ax.violinplot(values, positions=positions, widths=0.72,
                            showmeans=False, showmedians=False, showextrema=False)
    for x, body, group in zip(positions, violins["bodies"], order):
        for path in body.get_paths():
            vertices = path.vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], x)
        body.set_facecolor(colors[group])
        body.set_edgecolor(colors[group])
        body.set_alpha(0.11)
        body.set_linewidth(0.9)
    rng = np.random.default_rng(20260829)
    for x, (group, local_values) in enumerate(zip(order, values)):
        offsets = rng.uniform(0.055, 0.20, len(local_values))
        ax.scatter(x - offsets, local_values, s=20, color=colors[group], alpha=0.64,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean, ci = mean_ci(local_values)
        ax.errorbar(x, mean, yerr=ci, fmt="_", markersize=16, mew=2.5,
                    color=colors[group], capsize=3, lw=1.4, zorder=4)
    ax.set_xticks(positions, order)
    style_axis(ax)


def plot_hva_hierarchy_sessions(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    metric: str,
    hierarchy_stats: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(20260829)
    centers = []
    xs = []
    for group in AREA_ORDER:
        values = data.loc[data["group"].eq(group), "mean"].to_numpy(float)
        x = HIERARCHY_SCORES[group]
        jitter = rng.uniform(-0.012, 0.012, len(values))
        ax.scatter(x + jitter, values, s=20, color=AREA_COLORS[group], alpha=0.50,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean, ci = mean_ci(values)
        ax.errorbar(x, mean, yerr=ci, fmt="o", ms=6.5, mfc="white",
                    mec=AREA_COLORS[group], mew=1.5, color=AREA_COLORS[group],
                    capsize=3, lw=1.4, zorder=4)
        xs.append(x)
        centers.append(mean)
    fit = linregress(xs, centers)
    line_x = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
    ax.plot(line_x, fit.intercept + fit.slope * line_x, color="#4f555b", lw=1.25,
            ls="--", zorder=2)
    row = hierarchy_stats.loc[
        hierarchy_stats["metric"].eq(metric)
        & hierarchy_stats["scope"].eq("post_V1_only")
    ].iloc[0]
    ax.text(
        0.02, 0.96,
        f"HVA slope {row.slope_per_hierarchy_score:+.3g} "
        f"[{row.bootstrap_95ci_low:+.3g}, {row.bootstrap_95ci_high:+.3g}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED,
    )
    ax.set_xticks([HIERARCHY_SCORES[g] for g in AREA_ORDER], AREA_ORDER)
    tick_labels = ax.get_xticklabels()
    tick_labels[0].set_ha("right")
    tick_labels[1].set_ha("left")
    ax.set_xlim(min(xs) - 0.045, max(xs) + 0.045)
    style_axis(ax)


def plot_mousev2_penetration_map(
    ax: plt.Axes,
    penetrations: pd.DataFrame,
    boundary: np.ndarray,
    visp_mask: np.ndarray,
) -> None:
    """MouseV2 V1 penetration geometry on a cropped Zhuang common map."""
    ax.contourf(visp_mask.astype(float), levels=[0.5, 1.5], colors=["#edf3fa"], zorder=0)
    ax.contour(boundary.astype(float), levels=[0.5], colors="#b8bcc1",
               linewidths=0.65, zorder=1)
    ax.contour(visp_mask.astype(float), levels=[0.5], colors="#4c5661",
               linewidths=1.35, zorder=2)

    for probe in PROBE_ORDER:
        local = penetrations.loc[penetrations["probe"].eq(probe)]
        ax.scatter(
            local["zhuang_col"], local["zhuang_row"],
            s=68, marker="o", facecolors=PROBE_COLORS[probe],
            edgecolors="white", linewidths=0.9, alpha=0.86, zorder=4,
        )

    for label, (x, y) in ZHUANG_AREA_SEEDS.items():
        ax.text(
            x, y, label, ha="center", va="center", fontsize=9.5,
            color="#6b7076", fontweight="bold" if label == "V1" else "normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            zorder=3,
        )

    probe_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7.0,
               markerfacecolor=PROBE_COLORS[p], markeredgecolor="white", label=p)
        for p in PROBE_ORDER
    ]
    ax.legend(
        handles=probe_handles, title="Probe", ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        frameon=False, fontsize=8.2, title_fontsize=8.4,
        handletextpad=0.25, columnspacing=0.55, borderaxespad=0.2,
    )
    ax.set_xlim(0, 440)
    ax.set_ylim(430, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(
        "A1  MouseV2 V1 penetrations", loc="left", fontweight="bold",
        fontsize=12.0, pad=8,
    )
    ax.text(
        0.0, 1.005, "32 registered location estimates · 8 animals · Zhuang common map",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color=MUTED,
    )


def plot_open_scope_schematic(ax: plt.Axes, schematic: np.ndarray) -> None:
    ax.imshow(schematic)
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(
        "A2  OpenScope imaging schematic", loc="left", fontweight="bold",
        fontsize=12.0, pad=8,
    )


def recolor_open_scope_squares(schematic: np.ndarray) -> np.ndarray:
    """Reassign only the 12 flat square interiors to three dispersed groups of four."""
    output = schematic.copy()
    rgb = output[..., :3]
    is_float = np.issubdtype(rgb.dtype, np.floating)
    rgb_255 = np.rint(rgb * 255.0).astype(np.uint8) if is_float else rgb.astype(np.uint8)
    source_colors = np.array([
        [233, 113, 50],  # orange-red
        [255, 255, 0],
        [0, 176, 240],
    ], dtype=float)
    color_distance = np.min(
        np.sqrt(np.sum(
            (rgb_255[:, :, None, :].astype(float) - source_colors[None, None, :, :]) ** 2,
            axis=3,
        )),
        axis=2,
    )
    labels, component_count = ndimage.label(color_distance < 35.0)
    components = []
    for component_id in range(1, component_count + 1):
        rows, columns = np.where(labels == component_id)
        if 1_000 <= len(columns) <= 1_600:
            components.append((component_id, float(columns.mean()), float(rows.mean())))
    if len(components) != 12:
        raise ValueError(f"Expected 12 schematic squares; found {len(components)}")
    components.sort(key=lambda item: (item[2], item[1]))
    centers = np.array([[item[1], item[2]] for item in components])

    indices = set(range(12))
    best_score = None
    best_groups = None
    for first in combinations(range(12), 4):
        if 0 not in first:
            continue
        remaining = indices - set(first)
        for second in combinations(sorted(remaining), 4):
            third = tuple(sorted(remaining - set(second)))
            if min(second) > min(third):
                continue
            groups = (first, second, third)
            within_distances = []
            for group in groups:
                within_distances.extend(
                    np.linalg.norm(centers[i] - centers[j])
                    for i, j in combinations(group, 2)
                )
            score = (min(within_distances), sum(within_distances))
            if best_score is None or score > best_score:
                best_score = score
                best_groups = groups
    if best_groups is None:
        raise RuntimeError("Could not assign schematic square colors")

    recolored = rgb_255.copy()
    square_component_ids = []
    for target_color, group in zip(source_colors.astype(np.uint8), best_groups):
        for component_index in group:
            component_id = components[component_index][0]
            square_component_ids.append(component_id)
            recolored[labels == component_id] = target_color

    # Quiet the photographic/retinotopic context without muting the square fills.
    square_mask = np.isin(labels, square_component_ids)
    grayscale = np.sum(recolored.astype(float) * np.array([0.2126, 0.7152, 0.0722]), axis=2)
    softened = 0.86 * recolored.astype(float) + 0.14 * grayscale[:, :, None]
    softened = 0.96 * softened + 0.04 * 255.0
    softened[square_mask] = recolored[square_mask]
    softened = np.clip(np.rint(softened), 0, 255).astype(np.uint8)
    output[..., :3] = softened / 255.0 if is_float else softened
    return output


def metric_xlim(cells: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = cells.loc[cells["metric"].eq(metric), "centered_value"].to_numpy(float)
    low, high = np.percentile(values, [0.5, 99.5])
    pad = 0.08 * (high - low)
    return float(low - pad), float(high + pad)


def variant_e(groups: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.975, bottom=0.09, top=0.87,
                          height_ratios=[1.25, 0.85], hspace=0.34, wspace=0.22)
    add_page_header(fig, "Variant E — matched timescale distributions",
                    "Single-neuron ridgelines retain equal session/neuron weight; lower panels show session-level replication.")
    local_cells = cells.loc[cells["metric"].eq("Response timescale (ms)")]
    xlim = (0.0, 200.0)
    v1_ridge = fig.add_subplot(gs[0, 0])
    plot_cell_ridges(v1_ridge, local_cells.loc[local_cells["dataset"].eq("Within-V1")],
                     PROBE_ORDER, PROBE_COLORS, xlim=xlim,
                     title="A  Full neuronal distributions within V1")
    hva_ridge = fig.add_subplot(gs[0, 1])
    plot_cell_ridges(hva_ridge, local_cells.loc[local_cells["dataset"].eq("Post-V1")],
                     AREA_ORDER, AREA_COLORS, xlim=xlim,
                     title="B  Full neuronal distributions across HVAs")
    local_groups = groups.loc[groups["metric"].eq("Response timescale (ms)")]
    v1_sessions = fig.add_subplot(gs[1, 0])
    plot_session_summary(v1_sessions,
                         local_groups.loc[local_groups["dataset"].eq("Within-V1")],
                         PROBE_ORDER, PROBE_COLORS, title="C  V1 session × probe means")
    v1_sessions.set_ylabel("Response timescale (ms)")
    hva_sessions = fig.add_subplot(gs[1, 1], sharey=v1_sessions)
    plot_session_summary(hva_sessions,
                         local_groups.loc[local_groups["dataset"].eq("Post-V1")],
                         AREA_ORDER, AREA_COLORS, title="D  HVA session × area means")
    hva_sessions.tick_params(axis="y", labelleft=False)
    add_footer(fig)
    return fig


def variant_f(cells: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.975, bottom=0.085, top=0.88,
                          hspace=0.46, wspace=0.23)
    add_page_header(fig, "Variant F — three-metric full-cell ECDFs",
                    "Cells are centered by the frozen session mean and weighted so every session and neuron contributes equally.")
    for row, metric in enumerate(METRICS):
        xlim = metric_xlim(cells, metric)
        v1_ax = fig.add_subplot(gs[row, 0])
        hva_ax = fig.add_subplot(gs[row, 1], sharex=v1_ax, sharey=v1_ax)
        local = cells.loc[cells["metric"].eq(metric)]
        plot_weighted_ecdfs(v1_ax, local.loc[local["dataset"].eq("Within-V1")],
                            PROBE_ORDER, PROBE_COLORS, xlim=xlim,
                            title=f"{'A' if row == 0 else ''}  Within V1 — {metric}".strip(),
                            legend=row == 0)
        plot_weighted_ecdfs(hva_ax, local.loc[local["dataset"].eq("Post-V1")],
                            AREA_ORDER, AREA_COLORS, xlim=xlim,
                            title=f"{'B' if row == 0 else ''}  Cortical HVAs — {metric}".strip(),
                            legend=row == 0)
        hva_ax.tick_params(axis="y", labelleft=False)
        hva_ax.set_ylabel("")
        v1_ax.set_xlabel("Session-centered neuronal value")
        hva_ax.set_xlabel("Session-centered neuronal value")
    add_footer(fig)
    return fig


def variant_g(
    groups: pd.DataFrame, identity: dict[str, dict[str, float]]
) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.975, bottom=0.085, top=0.88,
                          width_ratios=[1, 1.12, 0.74], hspace=0.43, wspace=0.26)
    add_page_header(fig, "Variant G — session-level violin comparison",
                    "Violin envelopes are descriptive; dots are the inferential session × group observations.")
    for row, (metric, short) in enumerate(zip(METRICS, SHORT_METRICS)):
        local = groups.loc[groups["metric"].eq(metric)]
        v1_ax = fig.add_subplot(gs[row, 0])
        hva_ax = fig.add_subplot(gs[row, 1], sharey=v1_ax)
        plot_session_violins(v1_ax, local.loc[local["dataset"].eq("Within-V1")],
                             PROBE_ORDER, PROBE_COLORS,
                             title="A  Within V1" if row == 0 else None)
        plot_session_violins(hva_ax, local.loc[local["dataset"].eq("Post-V1")],
                             AREA_ORDER, AREA_COLORS,
                             title="B  Cortical HVAs" if row == 0 else None)
        v1_ax.set_ylabel(metric)
        hva_ax.tick_params(axis="y", labelleft=False)
        delta_ax = fig.add_subplot(gs[row, 2])
        plot_delta_metric(delta_ax, identity[metric], short)
        if row == 0:
            delta_ax.set_title("C  Direct identity contrast", loc="left", fontweight="bold")
        if row < 2:
            delta_ax.tick_params(axis="x", labelbottom=False)
            delta_ax.set_xlabel("")
    add_footer(fig)
    return fig


def plot_weighted_histograms(
    ax: plt.Axes,
    cells: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    bins: np.ndarray,
    title: str | None = None,
) -> None:
    for group in order:
        local = cells.loc[cells["group"].eq(group)]
        values = local["centered_value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        density, edges = np.histogram(values, bins=bins, weights=weights, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, density, color=colors[group], lw=1.6, label=group)
    ax.axvline(0, color="#777777", ls="--", lw=0.9)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Weighted density")
    style_axis(ax)


def variant_h(cells: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.975, bottom=0.085, top=0.88,
                          hspace=0.46, wspace=0.23)
    add_page_header(fig, "Variant H — single-cell histogram small multiples",
                    "Explicit bins show distribution shape; weights divide repeated draws and equalize neurons and sessions.")
    for row, metric in enumerate(METRICS):
        xlim = metric_xlim(cells, metric)
        bins = np.linspace(*xlim, 32)
        local = cells.loc[cells["metric"].eq(metric)]
        v1_ax = fig.add_subplot(gs[row, 0])
        hva_ax = fig.add_subplot(gs[row, 1], sharex=v1_ax)
        plot_weighted_histograms(v1_ax, local.loc[local["dataset"].eq("Within-V1")],
                                 PROBE_ORDER, PROBE_COLORS, bins=bins,
                                 title=f"{'A' if row == 0 else ''}  Within V1 — {metric}".strip())
        plot_weighted_histograms(hva_ax, local.loc[local["dataset"].eq("Post-V1")],
                                 AREA_ORDER, AREA_COLORS, bins=bins,
                                 title=f"{'B' if row == 0 else ''}  Cortical HVAs — {metric}".strip())
        if row == 0:
            v1_ax.legend(frameon=False, fontsize=7.5, ncol=4)
            hva_ax.legend(frameon=False, fontsize=7.5, ncol=5)
        v1_ax.set_xlabel("Session-centered neuronal value")
        hva_ax.set_xlabel("Session-centered neuronal value")
    add_footer(fig)
    return fig


def mean_ecdf_envelope(
    cells: pd.DataFrame, order: list[str], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for group in order:
        local = cells.loc[cells["group"].eq(group)].sort_values("centered_value")
        values = local["centered_value"].to_numpy(float)
        weights = local["equal_session_neuron_weight"].to_numpy(float)
        cumulative = np.cumsum(weights) / np.sum(weights)
        curves.append(np.interp(grid, values, cumulative, left=0.0, right=1.0))
    matrix = np.vstack(curves)
    return matrix.min(axis=0), matrix.mean(axis=0), matrix.max(axis=0)


def variant_i(groups: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.975, bottom=0.10, top=0.86,
                          height_ratios=[1, 1.12], hspace=0.35, wspace=0.28)
    add_page_header(fig, "Variant I — compact distribution evidence stack",
                    "Top: group-balanced neuronal ECDF envelopes. Bottom: every session × location/area mean.")
    for column, metric in enumerate(METRICS):
        xlim = metric_xlim(cells, metric)
        grid = np.linspace(*xlim, 500)
        top = fig.add_subplot(gs[0, column])
        local_cells = cells.loc[cells["metric"].eq(metric)]
        for dataset, order, color, label in [
            ("Within-V1", PROBE_ORDER, V1_COLOR, "Within V1"),
            ("Post-V1", AREA_ORDER, HVA_COLOR, "Cortical HVAs"),
        ]:
            low, mean, high = mean_ecdf_envelope(
                local_cells.loc[local_cells["dataset"].eq(dataset)], order, grid
            )
            top.fill_between(grid, low, high, color=color, alpha=0.12)
            top.plot(grid, mean, color=color, lw=2.0, label=label)
        top.axvline(0, color="#777777", ls="--", lw=0.9)
        top.set_xlim(*xlim); top.set_ylim(0, 1)
        top.set_title(metric, loc="left", fontweight="bold")
        top.set_xlabel("Session-centered neuronal value")
        if column == 0:
            top.set_ylabel("Mean cumulative fraction")
            top.legend(frameon=False, fontsize=8)
        else:
            top.tick_params(axis="y", labelleft=False)
        style_axis(top, grid=False)

        bottom = fig.add_subplot(gs[1, column])
        local_groups = groups.loc[groups["metric"].eq(metric)].copy()
        order = [*PROBE_ORDER, *AREA_ORDER]
        plot_group_means(bottom, local_groups, order, {**PROBE_COLORS, **AREA_COLORS},
                         centered=True, connect_sessions=True)
        bottom.axvline(3.5, color="#aaaaaa", lw=0.9)
        bottom.set_xticklabels(order, rotation=32, ha="right")
        bottom.set_ylabel("Session-centered mean" if column == 0 else "")
        if column > 0:
            bottom.tick_params(axis="y", labelleft=False)
    fig.text(0.06, 0.89, "A  Full-cell distribution envelopes", fontweight="bold", fontsize=11.5)
    fig.text(0.06, 0.47, "B  Session-level replication", fontweight="bold", fontsize=11.5)
    add_footer(fig)
    return fig


def variant_j(
    groups: pd.DataFrame,
    cells: pd.DataFrame,
    hierarchy_stats: pd.DataFrame,
    identity: dict[str, dict[str, float]],
    mousev2_penetrations: pd.DataFrame,
    zhuang_boundary: np.ndarray,
    zhuang_visp_mask: np.ndarray,
    open_scope_schematic: np.ndarray,
) -> plt.Figure:
    """Leading hybrid: raw sessions, hierarchy context, direct test, one cell example."""
    # Keep the reserved left column at its current physical width while
    # regenerating the analytical block at roughly two-thirds scale.
    fig = plt.figure(figsize=(14.0, 10.0))
    outer = fig.add_gridspec(
        1, 2, left=0.04, right=0.975, bottom=0.07, top=0.965,
        width_ratios=[1.65, 2.90], wspace=0.18,
    )
    gs = outer[0, 1].subgridspec(
        4, 3,
        width_ratios=[1.0, 1.18, 0.72], height_ratios=[1, 1, 1, 1.10],
        hspace=0.48, wspace=0.27,
    )
    left_box = outer[0, 0].get_position(fig)
    separator_x = left_box.x1
    fig.add_artist(Rectangle(
        (separator_x - 0.00225, 0.07), 0.0045, 0.895,
        transform=fig.transFigure, facecolor="black", edgecolor="none",
        clip_on=False, zorder=20,
    ))
    left_grid = outer[0, 0].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.12)
    penetration_ax = fig.add_subplot(left_grid[0, 0])
    plot_mousev2_penetration_map(
        penetration_ax, mousev2_penetrations, zhuang_boundary, zhuang_visp_mask
    )
    schematic_ax = fig.add_subplot(left_grid[1, 0])
    plot_open_scope_schematic(schematic_ax, open_scope_schematic)
    for row, (metric, short) in enumerate(zip(METRICS, SHORT_METRICS)):
        local = groups.loc[groups["metric"].eq(metric)]
        v1_ax = fig.add_subplot(gs[row, 0])
        hva_ax = fig.add_subplot(gs[row, 1], sharey=v1_ax)
        delta_ax = fig.add_subplot(gs[row, 2])

        plot_half_violins(
            v1_ax,
            local.loc[local["dataset"].eq("Within-V1")],
            PROBE_ORDER,
            PROBE_COLORS,
        )
        v1_ax.set_ylabel(metric)
        plot_hva_hierarchy_sessions(
            hva_ax,
            local.loc[local["dataset"].eq("Post-V1")],
            metric=metric,
            hierarchy_stats=hierarchy_stats,
        )
        hva_ax.tick_params(axis="y", labelleft=False)
        delta_label = {
            "log10 F1/F0": "F1/F0",
            "Response timescale (ms)": "τ",
        }.get(metric, short)
        plot_delta_metric(delta_ax, identity[metric], delta_label, annotate=True)

        if row == 0:
            v1_ax.set_title("B  Within-V1 sessions", loc="left", fontweight="bold")
            hva_ax.set_title("C  HVA sessions × hierarchy", loc="left", fontweight="bold")
            delta_ax.set_title("D  Direct Δω² contrast", loc="left", fontweight="bold")
        if row < 2:
            delta_ax.tick_params(axis="x", labelbottom=False)
            delta_ax.set_xlabel("")
        if row == 2:
            hva_ax.set_xlabel("Published inter-area hierarchy score")

    ridge_grid = gs[3, :].subgridspec(1, 2, wspace=0.20)
    timescale_cells = cells.loc[cells["metric"].eq("Response timescale (ms)")]
    v1_ridge = fig.add_subplot(ridge_grid[0, 0])
    hva_ridge = fig.add_subplot(ridge_grid[0, 1])
    plot_cell_ridges(
        v1_ridge,
        timescale_cells.loc[timescale_cells["dataset"].eq("Within-V1")],
        PROBE_ORDER,
        PROBE_COLORS,
        xlim=(0.0, 150.0),
        title="E1  Within-V1 neuronal timescales",
    )
    plot_cell_ridges(
        hva_ridge,
        timescale_cells.loc[timescale_cells["dataset"].eq("Post-V1")],
        AREA_ORDER,
        AREA_COLORS,
        xlim=(0.0, 150.0),
        title="E2  HVA neuronal timescales",
    )
    return fig


def save_page(pdf: PdfPages, fig: plt.Figure, preview: Path) -> None:
    pdf.savefig(fig, facecolor="white")
    fig.savefig(preview, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    groups, allen_v1, hierarchy_stats = load_inputs()
    mousev2_penetrations, zhuang_boundary, zhuang_visp_mask = load_mousev2_penetration_map()
    if not args.schematic.is_file():
        raise FileNotFoundError(f"OpenScope schematic not found: {args.schematic}")
    open_scope_schematic = plt.imread(args.schematic)
    if open_scope_schematic.ndim not in (2, 3):
        raise ValueError(f"Unexpected schematic image shape: {open_scope_schematic.shape}")
    open_scope_schematic = recolor_open_scope_squares(open_scope_schematic)
    cells = load_weighted_cells(groups)
    identity = identity_statistics(groups, n_bootstrap=args.n_bootstrap, seed=args.seed)
    preview_dir = output.parent / f"{output.stem}_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    builders = [
        lambda: variant_a(groups, allen_v1, hierarchy_stats, identity),
        lambda: variant_b(groups, identity),
        lambda: variant_c(groups, identity),
        lambda: variant_d(hierarchy_stats, identity),
        lambda: variant_e(groups, cells),
        lambda: variant_f(cells),
        lambda: variant_g(groups, identity),
        lambda: variant_h(cells),
        lambda: variant_i(groups, cells),
        lambda: variant_j(
            groups, cells, hierarchy_stats, identity,
            mousev2_penetrations, zhuang_boundary, zhuang_visp_mask,
            open_scope_schematic,
        ),
    ]
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Potential Figure 4 variants: OpenScope hierarchy analysis"
        metadata["Author"] = "Generated from frozen Figure 3 session-level artifacts"
        metadata["Subject"] = "Manuscript figure layout review"
        for index, builder in enumerate(builders, start=1):
            save_page(pdf, builder(), preview_dir / f"variant_{index}.png")

    hybrid_pdf = output.parent / "Figure4_hybrid_candidate.pdf"
    hybrid_png = output.parent / "Figure4_hybrid_candidate.png"
    hybrid = variant_j(
        groups, cells, hierarchy_stats, identity,
        mousev2_penetrations, zhuang_boundary, zhuang_visp_mask,
        open_scope_schematic,
    )
    hybrid.savefig(hybrid_pdf, facecolor="white")
    hybrid.savefig(hybrid_png, dpi=200, facecolor="white")
    plt.close(hybrid)
    print(output)
    print(preview_dir)
    print(hybrid_pdf)
    print(hybrid_png)


if __name__ == "__main__":
    main()
