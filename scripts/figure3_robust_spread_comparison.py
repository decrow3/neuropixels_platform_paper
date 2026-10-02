"""Robust session-level comparison of within-V1 and post-V1 response spread.

The inferential unit is a recording session.  The script writes an auditable
session-by-group table, a direct session-spread table, clustered bootstrap and
leave-one-session-out summaries, a Markdown report, and a four-column figure:

1. clustered omega-squared estimates for probe and area identity;
2. the direct HVA-minus-V1 omega-squared contrast;
3. session-centered values supporting the identity estimates;
4. per-session SD across recorded probes/areas as a diagnostic only.

V1 probe positions are categorical and are never treated as hierarchy scores.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import linregress


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import (  # noqa: E402
    load_allen_units,
    load_config,
    load_mousev2_units,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
AREA_COLORS = {
    "LM": "#4e73ae", "RL": "#65b2c9", "LP": "#58a76a",
    "AL": "#cab778", "PM": "#db8457", "AM": "#c24f54",
}
AREA_ORDER = ["LM", "RL", "AL", "PM", "AM"]
THALAMIC_COMPARISONS = ["LP"]
HIERARCHY_SCORES = {
    "Visual Coding VISp": -0.357,
    "LM": -0.093, "RL": -0.059,
    "AL": 0.152, "PM": 0.327, "AM": 0.441,
}
MOUSE_HARMONIZED_F1_F0 = (
    ROOT / "data/imports/mousev2_grating_common_support_v1/unit_metric_comparison.csv"
)
ALLEN_HARMONIZED_F1_F0 = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0"
    / "allen_bo_area_unit_f1_f0_harmonized.csv"
)
HARMONIZED_F1_COLUMN = "f1_f0_dg_figure_harmonized"
MOUSE_TIMESCALE_TRIAL_BRIDGE = (
    ROOT / "data/imports/mousev2_timescale_trial_bridge_v1/unit_subsample_metrics.csv"
)
MOUSE_CANONICAL_GRATING_METRICS = ROOT / "data/imports/mousev2_grating_metrics_v1"
MOUSE_PARAMETRIC_RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
RESPONSE_FILTERED_TTFS_OVERRIDE = (
    ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
    / "figure3_response_filtered_ttfs_override.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Figure3")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--grating-metrics-dir", type=Path, default=None)
    parser.add_argument("--flash-metrics-dir", type=Path, default=None)
    parser.add_argument("--flash-variant", choices=("pooled", "bright", "dark"), default="pooled")
    parser.add_argument("--grating-metric", choices=("f1_f0_dg", "mod_idx_dg"), default="f1_f0_dg")
    parser.add_argument("--f1-f0-mode", choices=("harmonized", "native"), default="harmonized")
    parser.add_argument("--mouse-harmonized-f1-f0", type=Path, default=MOUSE_HARMONIZED_F1_F0)
    parser.add_argument("--allen-harmonized-f1-f0", type=Path, default=ALLEN_HARMONIZED_F1_F0)
    parser.add_argument("--population-profile", default=None)
    parser.add_argument(
        "--timescale-view",
        choices=("allen_matched_150", "native_300"),
        default="allen_matched_150",
        help="MouseV2 trial support used only for the response-timescale row.",
    )
    parser.add_argument(
        "--mouse-timescale-trial-bridge",
        type=Path,
        default=MOUSE_TIMESCALE_TRIAL_BRIDGE,
    )
    parser.add_argument(
        "--timescale-population",
        choices=("historical_proxy_full20", "common_qc"),
        default="historical_proxy_full20",
        help="MouseV2 population used only for the response-timescale row.",
    )
    parser.add_argument("--mouse-parametric-rf-fits", type=Path, default=MOUSE_PARAMETRIC_RF_FITS)
    parser.add_argument(
        "--response-filtered-ttfs-override", type=Path,
        default=RESPONSE_FILTERED_TTFS_OVERRIDE,
        help="Audited preferred-polarity, response-filtered TTFS session means.",
    )
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--min-hva-areas", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_harmonized_f1_f0(
    mouse: pd.DataFrame,
    allen: pd.DataFrame,
    *,
    mouse_path: Path,
    allen_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the frozen matched-support F1/F0 values used by Figure 3.

    Mouse values use the shared 1-s/15-trial/SF=.04/contrast=.8 support.
    Allen values use the same support in the 28 Brain Observatory sessions with
    all 15 presentations per condition. Both source tables use common QC.
    """
    mouse_source = pd.read_csv(mouse_path)
    allen_source = pd.read_csv(allen_path)
    if mouse_source.unit_id.duplicated().any():
        raise ValueError("Duplicate MouseV2 unit IDs in harmonized F1/F0 source")
    if allen_source.ecephys_unit_id.duplicated().any():
        raise ValueError("Duplicate Allen unit IDs in harmonized F1/F0 source")
    required_mouse = {"unit_id", "f1_f0_dg_common_support", "default_qc"}
    required_allen = {"ecephys_unit_id", "f1_f0_dg_harmonized", "population_profile"}
    if not required_mouse.issubset(mouse_source):
        raise ValueError(f"Mouse harmonized source lacks {sorted(required_mouse.difference(mouse_source))}")
    if not required_allen.issubset(allen_source):
        raise ValueError(f"Allen harmonized source lacks {sorted(required_allen.difference(allen_source))}")
    if not allen_source.population_profile.eq("common_qc").all():
        raise ValueError("Allen harmonized source is not uniformly common_qc")

    mouse_map = mouse_source.set_index("unit_id")["f1_f0_dg_common_support"]
    allen_map = allen_source.set_index("ecephys_unit_id")["f1_f0_dg_harmonized"]
    result_mouse = mouse.copy()
    result_allen = allen.copy()
    result_mouse[HARMONIZED_F1_COLUMN] = result_mouse.unit_id.map(mouse_map)
    result_allen[HARMONIZED_F1_COLUMN] = result_allen.ecephys_unit_id.map(allen_map)
    missing_mouse_ids = ~result_mouse.unit_id.isin(mouse_source.unit_id)
    if missing_mouse_ids.any():
        raise ValueError(f"{int(missing_mouse_ids.sum())} plotted MouseV2 units lack harmonized source rows")
    if result_allen[HARMONIZED_F1_COLUMN].notna().sum() != allen_source.f1_f0_dg_harmonized.notna().sum():
        raise ValueError("Allen harmonized unit merge lost or duplicated source values")
    return result_mouse, result_allen


def valid_values(frame: pd.DataFrame, metric: str, metric_index: int) -> np.ndarray:
    if metric not in frame:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[metric], errors="coerce")
    keep = values.notna()
    if metric_index == 0:
        keep &= values < 0.1
        values = values * 1000.0
    elif metric_index == 1:
        keep &= values > 0
        values = np.log10(values.clip(lower=1e-6))
    else:
        keep &= values.between(1, 300)
        if "spike_count_ac" in frame:
            keep &= pd.to_numeric(frame["spike_count_ac"], errors="coerce") > 50
        if "err_ac" in frame:
            keep &= pd.to_numeric(frame["err_ac"], errors="coerce") < 20
    return values.loc[keep].to_numpy(dtype=float)


def session_group_table(
    frame: pd.DataFrame,
    *,
    dataset: str,
    session_column: str,
    group_column: str,
    groups: list[str],
    metric: str,
    metric_label: str,
    metric_index: int,
    min_units: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = frame.loc[frame[group_column].isin(groups)]
    for (session, group), part in subset.groupby([session_column, group_column], sort=True):
        values = valid_values(part, metric, metric_index)
        if len(values) < min_units:
            continue
        rows.append({
            "dataset": dataset,
            "metric": metric_label,
            "session_id": str(session),
            "group": str(group),
            "mean": float(np.mean(values)),
            "n_units": int(len(values)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["session_mean"] = result.groupby(["dataset", "metric", "session_id"])["mean"].transform("mean")
    result["centered_mean"] = result["mean"] - result["session_mean"]
    result["n_groups_in_session"] = result.groupby(["dataset", "metric", "session_id"])["group"].transform("size")
    return result


def matched_mouse_timescale_session_group_table(
    mouse_common_qc: pd.DataFrame,
    *,
    bridge_path: Path,
    groups: list[str],
    min_units: int,
    return_coverage: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Average session×probe means across deterministic Allen-matched draws."""
    bridge = pd.read_csv(bridge_path)
    bridge = bridge.loc[
        bridge["view"].eq("mouse_matched_150")
        & bridge["valid_timescale"].astype(bool)
    ].copy()
    metadata = mouse_common_qc[
        ["unit_id", "session_num", "probe_letter"]
    ].drop_duplicates("unit_id")
    joined = bridge.merge(metadata, on="unit_id", validate="many_to_one")
    if not joined["session_id"].astype(int).eq(joined["session_num"].astype(int)).all():
        raise ValueError("Timescale bridge session IDs disagree with MouseV2 metadata")
    joined = joined.loc[joined["probe_letter"].isin(groups)]
    draws = (
        joined.groupby(["session_id", "probe_letter", "subsample"], sort=True)
        .agg(mean=("timescale_ms", "mean"), n_units=("unit_id", "size"))
        .reset_index()
    )
    draws = draws.loc[draws["n_units"].ge(min_units)]
    coverage = (
        draws.groupby(["session_id", "probe_letter"], sort=True)
        .agg(n_complete_draws=("subsample", "nunique"), mean_valid_units=("n_units", "mean"))
        .reset_index()
    )
    coverage["retained"] = coverage["n_complete_draws"].eq(10)
    complete = coverage.loc[coverage["retained"], ["session_id", "probe_letter"]]
    draws = draws.merge(complete, on=["session_id", "probe_letter"], how="inner")
    if draws.empty:
        raise ValueError("No session × probe cells have all 10 matched timescale draws")
    result = (
        draws.groupby(["session_id", "probe_letter"], sort=True)
        .agg(mean=("mean", "mean"), n_units=("n_units", "mean"))
        .reset_index()
        .rename(columns={"probe_letter": "group"})
    )
    result["dataset"] = "Within-V1"
    result["metric"] = "Response timescale (ms)"
    result["session_id"] = result["session_id"].astype(str)
    result["n_units"] = result["n_units"].round().astype(int)
    result["session_mean"] = result.groupby("session_id")["mean"].transform("mean")
    result["centered_mean"] = result["mean"] - result["session_mean"]
    result["n_groups_in_session"] = result.groupby("session_id")["group"].transform("size")
    result = result[
        ["dataset", "metric", "session_id", "group", "mean", "n_units",
         "session_mean", "centered_mean", "n_groups_in_session"]
    ]
    if return_coverage:
        return result, coverage
    return result


def historical_proxy_full20_population(
    mouse_common_qc: pd.DataFrame,
    *,
    rf_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the frozen full-20-repeat parametric RF proxy with an audit trail."""
    required = {"unit_id", "snr", "firing_rate_dg"}
    if not required.issubset(mouse_common_qc):
        raise ValueError(f"MouseV2 table lacks {sorted(required.difference(mouse_common_qc))}")
    rf = pd.read_csv(rf_path)
    rf_required = {
        "unit_id", "rf_fit_success", "rf_lrt_p", "rf_sigma_major_deg", "rf_sigma_minor_deg"
    }
    if not rf_required.issubset(rf):
        raise ValueError(f"RF source lacks {sorted(rf_required.difference(rf))}")
    if rf["unit_id"].duplicated().any():
        raise ValueError("Duplicate unit IDs in parametric RF source")
    rf = rf[list(rf_required)].copy()
    joined = mouse_common_qc.merge(rf, on="unit_id", how="left", validate="one_to_one")
    joined["rf_area_proxy_deg2"] = (
        np.pi
        * pd.to_numeric(joined["rf_sigma_major_deg"], errors="coerce")
        * pd.to_numeric(joined["rf_sigma_minor_deg"], errors="coerce")
    )
    gates = [
        ("common_qc", pd.Series(True, index=joined.index), "Frozen MouseV2 common QC"),
        ("parametric_rf_available", joined["rf_lrt_p"].notna(), "Full-20 parametric RF fit is available"),
        ("rf_fit_success", joined["rf_fit_success"].eq(True), "Parametric RF optimizer succeeded"),
        ("rf_lrt_p_lt_0.01", pd.to_numeric(joined["rf_lrt_p"], errors="coerce").lt(0.01), "RF likelihood-ratio p < 0.01"),
        ("rf_area_lt_2500_deg2", joined["rf_area_proxy_deg2"].lt(2500), "pi × sigma_major × sigma_minor < 2500 deg²"),
        ("snr_gt_1", pd.to_numeric(joined["snr"], errors="coerce").gt(1), "SNR > 1"),
        ("firing_rate_dg_gt_0.1", pd.to_numeric(joined["firing_rate_dg"], errors="coerce").gt(0.1), "Drifting-grating firing rate > 0.1 Hz"),
    ]
    keep = pd.Series(True, index=joined.index)
    flow = []
    for step, gate, rule in gates:
        before = int(keep.sum())
        keep &= gate
        after = int(keep.sum())
        flow.append({"step": step, "rule": rule, "input_units": before,
                     "retained_units": after, "excluded_at_step": before - after})
    selected = joined.loc[keep].copy()
    audit_columns = [
        "unit_id", "session_num", "probe_letter", "snr", "firing_rate_dg",
        "rf_fit_success", "rf_lrt_p", "rf_sigma_major_deg", "rf_sigma_minor_deg",
        "rf_area_proxy_deg2",
    ]
    return selected, pd.DataFrame(flow), selected[audit_columns]


def session_spreads(group_table: pd.DataFrame, min_hva_areas: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, metric, session), part in group_table.groupby(
        ["dataset", "metric", "session_id"], sort=True
    ):
        required = 4 if dataset == "Within-V1" else min_hva_areas
        if len(part) < required:
            continue
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "session_id": session,
            "spread_sd": float(part["mean"].std(ddof=1)),
            "n_groups": int(len(part)),
            "min_group_mean": float(part["mean"].min()),
            "max_group_mean": float(part["mean"].max()),
            "spread_range": float(part["mean"].max() - part["mean"].min()),
            "groups_present": ",".join(part["group"].astype(str)),
        })
    return pd.DataFrame(rows)


def bootstrap_mean_difference(
    v1: np.ndarray, hva: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    observed = float(np.mean(hva) - np.mean(v1))
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        v1_draw = rng.choice(v1, size=len(v1), replace=True)
        hva_draw = rng.choice(hva, size=len(hva), replace=True)
        draws[index] = np.mean(hva_draw) - np.mean(v1_draw)
    low, high = np.percentile(draws, [2.5, 97.5])
    return observed, float(low), float(high), float(np.mean(draws <= 0))


def leave_one_out_differences(v1: np.ndarray, hva: np.ndarray) -> np.ndarray:
    values = [np.mean(hva) - np.mean(np.delete(v1, i)) for i in range(len(v1))]
    values.extend(np.mean(np.delete(hva, i)) - np.mean(v1) for i in range(len(hva)))
    return np.asarray(values, dtype=float)


def hierarchy_fit_summary(
    group_table: pd.DataFrame,
    allen_v1_table: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Summarize all-area and post-V1-only slopes with session resampling."""
    v1 = allen_v1_table.copy()
    v1["group"] = "Visual Coding VISp"
    hierarchy = pd.concat(
        [v1, group_table.loc[group_table.dataset.eq("Post-V1")]],
        ignore_index=True,
    )
    rows = []
    for metric, metric_table in hierarchy.groupby("metric", sort=False):
        for scope, groups in (
            ("VISp_plus_post_V1", ["Visual Coding VISp", *AREA_ORDER]),
            ("post_V1_only", AREA_ORDER),
        ):
            available = [
                group for group in groups
                if not metric_table.loc[metric_table.group.eq(group)].empty
            ]
            xs = np.asarray([HIERARCHY_SCORES[group] for group in available], dtype=float)
            values = {
                group: metric_table.loc[metric_table.group.eq(group), "mean"].to_numpy(float)
                for group in available
            }
            centers = np.asarray([values[group].mean() for group in available])
            fit = linregress(xs, centers)
            draws = np.empty(n_bootstrap)
            for draw in range(n_bootstrap):
                sampled = np.asarray(
                    [rng.choice(values[group], len(values[group]), replace=True).mean() for group in available]
                )
                draws[draw] = linregress(xs, sampled).slope
            rows.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "area_points": len(available),
                    "groups": ";".join(available),
                    "slope_per_hierarchy_score": fit.slope,
                    "bootstrap_95ci_low": np.quantile(draws, 0.025),
                    "bootstrap_95ci_high": np.quantile(draws, 0.975),
                    "area_mean_r": fit.rvalue,
                }
            )
    return pd.DataFrame(rows)


def omega_squared(group: np.ndarray, values: np.ndarray) -> float:
    """Bias-corrected fraction of variance explained by stable group identity."""
    retained = []
    for label in np.unique(group):
        group_values = values[group == label]
        if len(group_values) >= 2:
            retained.append(group_values)
    if len(retained) < 2:
        return np.nan
    all_values = np.concatenate(retained)
    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    ss_between = float(sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in retained))
    degrees_within = len(all_values) - len(retained)
    if degrees_within <= 0 or ss_total == 0:
        return np.nan
    ms_within = (ss_total - ss_between) / degrees_within
    return float(
        (ss_between - (len(retained) - 1) * ms_within) / (ss_total + ms_within)
    )


def clustered_omega_squared(
    group_table: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Resample complete session vectors, preserving all within-session groups."""
    blocks = [
        (part["group"].to_numpy(str), part["mean"].to_numpy(float))
        for _, part in group_table.groupby("session_id", sort=True)
    ]
    groups = np.concatenate([block[0] for block in blocks])
    values = np.concatenate([block[1] for block in blocks])
    observed = omega_squared(groups, values)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(blocks), len(blocks))
        boot_groups = np.concatenate([blocks[i][0] for i in selected])
        boot_values = np.concatenate([blocks[i][1] for i in selected])
        draws[index] = omega_squared(boot_groups, boot_values)
    return observed, draws


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def add_group_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    *,
    value_column: str,
    title: str,
) -> None:
    for position, group in enumerate(order):
        part = data.loc[data["group"] == group].sort_values("session_id")
        if part.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(part))
        values = part[value_column].to_numpy(float)
        ax.scatter(position + jitter, values, s=18, color=colors[group], alpha=0.58,
                   edgecolor="black", linewidth=0.25, zorder=3)
        mean, ci = mean_ci(values)
        ax.errorbar(position, mean, yerr=ci, fmt="_", markersize=15, mew=2.2,
                    color=colors[group], capsize=3, lw=1.5, zorder=4)
        ax.text(position, 0.98, f"n={len(part)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color="#555")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30 if len(order) > 4 else 0, ha="right" if len(order) > 4 else "center")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def render_figure(
    group_table: pd.DataFrame,
    spread_table: pd.DataFrame,
    identity_stats: dict[str, dict[str, float]],
    spread_stats: dict[str, dict[str, float]],
    output: Path,
) -> None:
    metrics = list(group_table["metric"].drop_duplicates())
    fig = plt.figure(figsize=(16, 11.5))
    grid = fig.add_gridspec(
        4, 4, height_ratios=[1.05, 1, 1, 1], width_ratios=[1, 1, 1, 0.78],
        hspace=0.62, wspace=0.42,
    )
    omega_ax = fig.add_subplot(grid[0, :2])
    delta_ax = fig.add_subplot(grid[0, 2])
    callout_ax = fig.add_subplot(grid[0, 3])
    probe_order = [group for group in ["B", "C", "A", "E"] if group in set(group_table["group"])]

    y_positions = np.arange(len(metrics))[::-1]
    offset = 0.12
    for y, metric in zip(y_positions, metrics):
        result = identity_stats[metric]
        omega_ax.errorbar(
            result["v1_omega"], y + offset,
            xerr=[[result["v1_omega"] - result["v1_ci_low"]], [result["v1_ci_high"] - result["v1_omega"]]],
            fmt="o", ms=6, color="#6f62a6", markerfacecolor="white", markeredgewidth=1.5,
            capsize=3, lw=1.5, label="V1 probe identity" if y == y_positions[0] else None,
        )
        omega_ax.errorbar(
            result["hva_omega"], y - offset,
            xerr=[[result["hva_omega"] - result["hva_ci_low"]], [result["hva_ci_high"] - result["hva_omega"]]],
            fmt="s", ms=5.5, color="#555555", markerfacecolor="#555555",
            capsize=3, lw=1.5, label="HVA identity" if y == y_positions[0] else None,
        )
        delta_ax.errorbar(
            result["delta"], y,
            xerr=[[result["delta"] - result["delta_ci_low"]], [result["delta_ci_high"] - result["delta"]]],
            fmt="o", ms=6, color="#2f5597", capsize=3, lw=1.6,
        )
    omega_ax.axvline(0, color="#aaaaaa", lw=0.8, ls="--")
    omega_ax.set_yticks(y_positions, metrics)
    omega_ax.set_xlabel("ω²: variance explained by stable group identity")
    omega_ax.set_title("A  Identity effect sizes (whole-session bootstrap)", loc="left", fontsize=10.5, fontweight="bold")
    omega_ax.legend(frameon=False, fontsize=8, loc="lower right")
    omega_ax.spines[["top", "right"]].set_visible(False)

    delta_ax.axvline(0, color="#555555", lw=1.0, ls="--")
    delta_ax.set_yticks(y_positions, [])
    delta_ax.set_xlabel("Δω² (HVA − V1)")
    delta_ax.set_title("B  Direct identity contrast", loc="left", fontsize=10.5, fontweight="bold")
    delta_ax.spines[["top", "right"]].set_visible(False)

    callout_ax.axis("off")
    callout_ax.text(0, 0.92, "Primary conclusion", fontsize=10.5, fontweight="bold", va="top")
    callout_ax.text(
        0, 0.72,
        "All three Δω² intervals\ncross zero.\n\nCurrent data do not establish\nthat HVA identity explains more\nvariation than V1 probe identity.",
        fontsize=9.2, va="top", linespacing=1.3,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f3f5f8", "edgecolor": "#c8ced8"},
    )

    for row, metric in enumerate(metrics):
        centered_ax = fig.add_subplot(grid[row + 1, :3])
        spread_ax = fig.add_subplot(grid[row + 1, 3])
        metric_groups = group_table.loc[group_table["metric"] == metric]
        centered_order = probe_order + AREA_ORDER
        centered_colors = {**PROBE_COLORS, **AREA_COLORS}
        add_group_panel(
            centered_ax, metric_groups, centered_order, centered_colors,
            value_column="centered_mean",
            title=f"C{row + 1}  {metric}: session-centered group means",
        )
        centered_limit = float(np.percentile(np.abs(metric_groups["centered_mean"]), 99)) * 1.12
        centered_ax.set_ylim(-centered_limit, centered_limit)
        centered_ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
        centered_ax.set_ylabel(f"Centered {metric}")
        centered_ax.axvline(len(probe_order) - 0.5, color="#bbbbbb", lw=0.8)

        ax = spread_ax
        spread = spread_table.loc[spread_table["metric"] == metric]
        for position, dataset in enumerate(["Within-V1", "Post-V1"]):
            part = spread.loc[spread["dataset"] == dataset].sort_values("session_id")
            values = part["spread_sd"].to_numpy(float)
            jitter = np.linspace(-0.16, 0.16, len(values))
            color = "#8073ac" if dataset == "Within-V1" else "#777777"
            ax.scatter(position + jitter, values, s=20, color=color, alpha=0.62,
                       edgecolor="black", linewidth=0.25)
            mean, ci = mean_ci(values)
            ax.errorbar(position, mean, yerr=ci, fmt="_", markersize=18, mew=2.4,
                        color=color, capsize=4, lw=1.7)
            ax.text(position, 0.98, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, color="#555")
        result = spread_stats[metric]
        ax.text(0.03, 0.94,
                f"Δ(HVA−V1)={result['delta']:+.3g}\n95% CI [{result['ci_low']:+.3g}, {result['ci_high']:+.3g}]\nP(Δ≤0)={result['p_le_zero']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
        ax.set_xticks([0, 1], ["V1 probes", "HVA areas"])
        ax.set_ylabel("Within-session SD")
        ax.set_title(f"D{row + 1}  Diagnostic: session SD", loc="left", fontsize=9.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Stable HVA identity versus within-V1 probe identity",
        fontsize=14, y=0.995, fontweight="bold",
    )
    fig.text(
        0.5, 0.972,
        "Primary test: does HVA label explain more session-level variance?  "
        "Dots below are sessions; V1 probes are categorical locations, not hierarchy scores.",
        ha="center", va="top", fontsize=9.5, color="#444444",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_identity_panel(identity_stats: dict[str, dict[str, float]], output: Path) -> None:
    """Render Panel A alone with enough context to stand independently."""
    metrics = list(identity_stats)
    y_positions = np.arange(len(metrics))[::-1]
    offset = 0.12
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for y, metric in zip(y_positions, metrics):
        result = identity_stats[metric]
        ax.errorbar(
            result["v1_omega"], y + offset,
            xerr=[[result["v1_omega"] - result["v1_ci_low"]], [result["v1_ci_high"] - result["v1_omega"]]],
            fmt="o", ms=7, color="#6f62a6", markerfacecolor="white", markeredgewidth=1.7,
            capsize=3.5, lw=1.7, label="Within-V1 probe identity" if y == y_positions[0] else None,
        )
        ax.errorbar(
            result["hva_omega"], y - offset,
            xerr=[[result["hva_omega"] - result["hva_ci_low"]], [result["hva_ci_high"] - result["hva_omega"]]],
            fmt="s", ms=6.5, color="#555555", markerfacecolor="#555555",
            capsize=3.5, lw=1.7, label="Post-V1 HVA identity" if y == y_positions[0] else None,
        )
    ax.axvline(0, color="#999999", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(y_positions, metrics)
    ax.set_xlabel("ω²: fraction of session-level variance explained by stable group identity")
    ax.set_title("Identity effect sizes", loc="left", fontsize=13, fontweight="bold", pad=28)
    ax.text(
        0, 1.05,
        "Points are observed ω²; horizontal lines are 95% whole-session bootstrap intervals.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color="#444444",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.12, 0.53)
    ax.set_ylim(-0.45, len(metrics) - 0.55)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_single_panel_identity_variation(
    identity_stats: dict[str, dict[str, float]],
    identity_draws: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    """One-panel visual argument: directionally larger HVA effects, overlapping uncertainty."""
    metrics = list(identity_stats)
    y_positions = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    v1_color = "#6f62a6"
    hva_color = "#666666"
    for y, metric in zip(y_positions, metrics):
        result = identity_stats[metric]
        local_draws = identity_draws[metric]
        combined = np.concatenate([local_draws["v1"], local_draws["hva"]])
        low, high = np.percentile(combined, [0.5, 99.5])
        grid = np.linspace(low, high, 400)
        for dataset, offset, color, filled in [
            ("v1", +0.11, v1_color, False),
            ("hva", -0.11, hva_color, True),
        ]:
            draws = local_draws[dataset]
            density = gaussian_kde(draws)(grid)
            density = density / density.max() * 0.20
            baseline = np.full_like(grid, y + offset)
            curve = baseline + density if dataset == "v1" else baseline - density
            if filled:
                ax.fill_between(grid, baseline, curve, color=color, alpha=0.24, linewidth=0)
            ax.plot(grid, curve, color=color, lw=1.35)
        v1_value = result["v1_omega"]
        hva_value = result["hva_omega"]
        ax.plot(v1_value, y + 0.11, "o", ms=6.5, mfc="white", mec=v1_color, mew=1.6, zorder=4)
        ax.plot(hva_value, y - 0.11, "s", ms=6, mfc=hva_color, mec=hva_color, zorder=4)
        ax.annotate(
            "", xy=(hva_value, y), xytext=(v1_value, y),
            arrowprops={"arrowstyle": "->", "color": "#2f5597", "lw": 1.3},
            zorder=5,
        )
    ax.axvline(0, color="#888888", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(y_positions, metrics)
    ax.set_xlabel("Stable identity variation (ω²)")
    ax.set_title("Identity-related variation within V1 and across HVAs", loc="left", fontsize=13, fontweight="bold", pad=28)
    ax.text(
        0, 1.05,
        "Whole-session bootstrap distributions; arrows show the observed HVA−V1 direction.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color="#444444",
    )
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=v1_color, mfc="white", lw=1.2,
                   label="Within-V1 probe identity"),
            Line2D([0], [0], marker="s", color=hva_color, mfc=hva_color, lw=1.2,
                   label="Post-V1 HVA identity"),
        ],
        frameon=False, fontsize=8.5, loc="lower right",
    )
    ax.text(0.01, -0.16, "little stable identity structure", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#666666")
    ax.text(0.99, -0.16, "more stable identity structure →", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#666666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.48, len(metrics) - 0.52)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_session_level_variation_comparison(
    group_table: pd.DataFrame,
    output: Path,
    allen_v1_table: pd.DataFrame | None = None,
) -> None:
    """Show the underlying session-level variation in the earlier split-panel grammar.

    Both datasets use the same observational unit: one session x probe/area mean.
    Faint lines join observations from the same session, and boxes summarize the
    distribution of session means rather than a unit-pooled distribution.
    """
    metrics = list(group_table["metric"].drop_duplicates())
    include_allen_v1 = allen_v1_table is not None and not allen_v1_table.empty
    plot_table = (
        pd.concat([group_table, allen_v1_table], ignore_index=True)
        if include_allen_v1 else group_table
    )
    datasets = ["Within-V1", "Post-V1"]
    left_order = ["B", "C", "A", "E"] + (["Visual Coding VISp"] if include_allen_v1 else [])
    orders = {"Within-V1": left_order, "Post-V1": AREA_ORDER}
    colors = {
        "Within-V1": {**PROBE_COLORS, "Visual Coding VISp": "#303030"},
        "Post-V1": AREA_COLORS,
    }
    titles = {
        "Within-V1": (
            "V1: within-area locations and Visual Coding VISp"
            if include_allen_v1 else "Within V1 (8 sessions)"
        ),
        "Post-V1": "Higher visual areas: Visual Coding Neuropixels",
    }

    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(11.4, 9.0),
        gridspec_kw={"hspace": 0.48, "wspace": 0.18},
    )
    for row, metric in enumerate(metrics):
        metric_data = plot_table.loc[plot_table["metric"] == metric]
        all_values = metric_data["mean"].to_numpy(float)
        value_min, value_max = float(np.min(all_values)), float(np.max(all_values))
        padding = max((value_max - value_min) * 0.09, 0.05)
        limits = (value_min - padding, value_max + padding)

        for column, dataset in enumerate(datasets):
            ax = axes[row, column]
            if dataset == "Within-V1":
                data = metric_data.loc[metric_data["dataset"].isin(["Within-V1", "Allen-V1"])]
            else:
                data = metric_data.loc[metric_data["dataset"] == dataset]
            order = [group for group in orders[dataset] if group in set(data["group"])]
            positions = {group: index for index, group in enumerate(order)}

            # Within-session trajectories expose the spatial contrast without
            # treating the more numerous Allen units as independent replicates.
            for _, session in data.groupby(["dataset", "session_id"], sort=True):
                session = session.loc[session["group"].isin(order)].copy()
                session["x"] = session["group"].map(positions)
                session = session.sort_values("x")
                if len(session) > 1:
                    ax.plot(
                        session["x"], session["mean"], color="#606060",
                        alpha=0.10 if dataset == "Post-V1" else 0.18,
                        lw=0.65, zorder=1,
                    )

            for group in order:
                position = positions[group]
                part = data.loc[data["group"] == group].sort_values("session_id")
                values = part["mean"].to_numpy(float)
                color = colors[dataset][group]
                if len(values) >= 2:
                    q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
                    ax.vlines(position, q10, q90, color=color, lw=1.2, alpha=0.65, zorder=2)
                    ax.add_patch(plt.Rectangle(
                        (position - 0.27, q25), 0.54, q75 - q25,
                        facecolor=color, edgecolor=color, alpha=0.16, lw=1.0, zorder=2,
                    ))
                jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(
                    position + jitter, values, s=23, facecolor=color,
                    edgecolor="white", linewidth=0.45, alpha=0.78, zorder=3,
                )
                mean, ci = mean_ci(values)
                ax.errorbar(
                    position, mean, yerr=ci, fmt="_", markersize=20, mew=3.0,
                    color=color, capsize=3, lw=1.5, zorder=4,
                )
                ax.text(
                    position, 0.985, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, color="#555555",
                )

            ax.set_xlim(-0.62, len(order) - 0.38)
            ax.set_ylim(*limits)
            display_labels = ["Visual Coding\nVISp" if label == "Visual Coding VISp" else label for label in order]
            ax.set_xticks(range(len(order)), display_labels)
            if include_allen_v1 and dataset == "Within-V1":
                ax.axvline(3.5, color="#999999", lw=0.8, ls="--", zorder=0)
            ax.grid(axis="y", color="#e6e6e6", lw=0.65, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel(metric)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title(titles[dataset], fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Response variation within V1 and across higher visual areas",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.968,
        "Dots are session × location/area means; faint lines join the same session; "
        "boxes show the session-level IQR and 10th–90th percentiles; thick bars are mean ± 95% CI.",
        ha="center", va="top", fontsize=8.8, color="#444444",
    )
    fig.text(
        0.5, 0.012,
        (
            "Y-scales are matched across columns within each row. Visual Coding VISp is a session-level regional reference, not another within-V1 location. "
            if include_allen_v1 else
            "Y-scales are matched across columns within each row. "
        ) + "F1/F0 uses common QC and matched 1-s, 15-trial, SF 0.04, contrast 0.8 support; Allen F1/F0 uses 28 complete Brain Observatory sessions.",
        ha="center", va="bottom", fontsize=8.2, color="#666666",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_hierarchy_fit_histogram_comparison(
    group_table: pd.DataFrame,
    allen_v1_table: pd.DataFrame,
    output: Path,
    *,
    seed: int = 42,
    timescale_view: str = "allen_matched_150",
    timescale_population: str = "common_qc",
) -> None:
    """Earlier four-column grammar with score fits and session-level histograms."""
    rng = np.random.default_rng(seed)
    metrics = list(group_table["metric"].drop_duplicates())
    v1_reference = allen_v1_table.copy()
    v1_reference["group"] = "Visual Coding VISp"
    hierarchy_table = pd.concat([
        v1_reference,
        group_table.loc[group_table["dataset"] == "Post-V1"],
    ], ignore_index=True)
    left_table = pd.concat([
        group_table.loc[group_table["dataset"] == "Within-V1"],
        v1_reference,
    ], ignore_index=True)
    left_order = ["B", "C", "A", "E", "Visual Coding VISp"]
    left_colors = {**PROBE_COLORS, "Visual Coding VISp": "#303030"}
    hierarchy_order = ["Visual Coding VISp", *AREA_ORDER]
    hierarchy_colors = {"Visual Coding VISp": "#303030", **AREA_COLORS}

    fig, axes = plt.subplots(
        3, 4, figsize=(15.2, 10.0),
        gridspec_kw={"width_ratios": [1.35, 0.38, 1.55, 0.38],
                     "wspace": 0.10, "hspace": 0.42},
    )
    for row, metric in enumerate(metrics):
        left_ax, left_hist, score_ax, score_hist = axes[row]
        local_left = left_table.loc[left_table["metric"] == metric]
        local_hierarchy = hierarchy_table.loc[hierarchy_table["metric"] == metric]
        all_values = pd.concat([local_left["mean"], local_hierarchy["mean"]]).to_numpy(float)
        lo, hi = float(np.min(all_values)), float(np.max(all_values))
        pad = max(0.08 * (hi - lo), 0.05)
        y_limits = (lo - pad, hi + pad)

        # Categorical within-V1 locations plus the external VISp reference.
        for x, group in enumerate(left_order):
            values = local_left.loc[local_left["group"] == group, "mean"].to_numpy(float)
            if not len(values):
                continue
            color = left_colors[group]
            q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
            left_ax.vlines(x, q10, q90, color=color, lw=1.2, alpha=0.7)
            left_ax.add_patch(plt.Rectangle(
                (x - 0.27, q25), 0.54, q75 - q25,
                facecolor=color, edgecolor=color, alpha=0.17, lw=1.0,
            ))
            jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else [0]
            left_ax.scatter(x + np.asarray(jitter), values, s=23, color=color,
                            edgecolor="white", linewidth=0.4, alpha=0.78, zorder=3)
            mean, ci = mean_ci(values)
            left_ax.errorbar(x, mean, yerr=ci, fmt="_", markersize=20, mew=3,
                             color=color, capsize=3, lw=1.5, zorder=4)
        left_ax.axvline(3.5, color="#999999", lw=0.8, ls="--")
        left_ax.set_xticks(range(5), ["B", "C", "A", "E", "Visual Coding\nVISp"])
        left_ax.set_xlim(-0.6, 4.6)
        left_ax.set_ylim(*y_limits)
        left_ax.set_ylabel(metric)
        left_ax.grid(axis="y", color="#e8e8e8", lw=0.6)
        left_ax.spines[["top", "right"]].set_visible(False)

        # Marginal histograms use session-level means on both sides.
        bins = np.linspace(*y_limits, 15)
        left_hist.hist(local_left["mean"], bins=bins, orientation="horizontal",
                       density=True, color="#8174b2", alpha=0.48, edgecolor="white")
        left_hist.set_ylim(*y_limits)
        left_hist.set_xticks([])
        left_hist.tick_params(axis="y", labelleft=False, left=False)
        left_hist.spines[["top", "right", "bottom", "left"]].set_visible(False)

        # Area summaries are placed at published hierarchy scores.
        xs, ys = [], []
        by_group: dict[str, np.ndarray] = {}
        for group in hierarchy_order:
            values = local_hierarchy.loc[local_hierarchy["group"] == group, "mean"].to_numpy(float)
            if not len(values):
                continue
            x = HIERARCHY_SCORES[group]
            mean, ci = mean_ci(values)
            score_ax.errorbar(x, mean, yerr=ci, fmt="o", ms=6.5,
                              color=hierarchy_colors[group], markerfacecolor="white",
                              markeredgewidth=1.6, capsize=3, lw=1.6, zorder=4)
            xs.append(x); ys.append(mean); by_group[group] = values
        fit = linregress(xs, ys)
        fit_x = np.linspace(min(xs) - 0.035, max(xs) + 0.035, 160)
        fit_y = fit.intercept + fit.slope * fit_x
        boot_lines = np.empty((1000, len(fit_x)))
        for draw in range(len(boot_lines)):
            draw_y = [float(np.mean(rng.choice(by_group[g], len(by_group[g]), replace=True)))
                      for g in hierarchy_order if g in by_group]
            draw_fit = linregress(xs, draw_y)
            boot_lines[draw] = draw_fit.intercept + draw_fit.slope * fit_x
        band_low, band_high = np.percentile(boot_lines, [2.5, 97.5], axis=0)
        score_ax.fill_between(fit_x, band_low, band_high, color="#777777", alpha=0.14, zorder=1)
        score_ax.plot(fit_x, fit_y, color="#555555", ls="--", lw=1.4, zorder=2)
        score_ax.set_xticks([HIERARCHY_SCORES[g] for g in hierarchy_order],
                            ["VISp", *AREA_ORDER], rotation=32, ha="right")
        score_ax.set_xlim(min(fit_x), max(fit_x))
        score_ax.set_ylim(*y_limits)
        score_ax.tick_params(axis="y", labelleft=False)
        score_ax.grid(axis="y", color="#e8e8e8", lw=0.6)
        score_ax.spines[["top", "right"]].set_visible(False)
        if row == 2:
            score_ax.set_xlabel("Published hierarchy score")
            support_note = (
                "MouseV2: raw-NWB, matched 150 flashes\nAllen: historical, 150 flashes"
                if timescale_view == "allen_matched_150"
                else "MouseV2: raw-NWB, 300 flashes\nAllen: historical, 150 flashes"
            )
            left_ax.text(
                0.01,
                0.985,
                support_note,
                transform=left_ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="#555555",
            )
            if timescale_population == "historical_proxy_full20":
                left_ax.text(
                    0.01, 0.80, "MouseV2 population: full-20 RF proxy",
                    transform=left_ax.transAxes, ha="left", va="top",
                    fontsize=7.0, color="#555555",
                )

        score_hist.hist(local_hierarchy["mean"], bins=bins, orientation="horizontal",
                        density=True, color="#777777", alpha=0.48, edgecolor="white")
        score_hist.set_ylim(*y_limits)
        score_hist.set_xticks([])
        score_hist.tick_params(axis="y", labelleft=False, left=False)
        score_hist.spines[["top", "right", "bottom", "left"]].set_visible(False)

        if row == 0:
            left_ax.set_title("V1: within-area locations + Visual Coding VISp", fontsize=11, fontweight="bold")
            left_hist.set_title("Distribution", fontsize=8.5, color="#555555")
            score_ax.set_title("Visual Coding areas by hierarchy score", fontsize=11, fontweight="bold")
            score_hist.set_title("Distribution", fontsize=8.5, color="#555555")

    fig.suptitle("Response variation and hierarchy-score trends", fontsize=14,
                 fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.971,
        "Dots are session-level means; bars are group means ± 95% CI. Dashed fits use area means; shaded bands are session-bootstrap 95% intervals.",
        ha="center", va="top", fontsize=8.8, color="#444444",
    )
    fig.text(
        0.5, 0.012,
        "Marginal histograms contain session × location/area means, not pooled neurons. TTFS uses preferred flash polarity, odd-trial positive-response selection (FDR q<0.01, Δrate≥2 Hz), and ≥10 selected units per estimate.\n"
        f"MouseV2 timescale view: {timescale_view}; population: {timescale_population}; Allen uses 150 flashes.\n"
        "The full-20 RF population is a parametric LRT/ellipse proxy, not Allen's categorical RF test.\n"
        "F1/F0 uses common QC and matched 1-s, 15-trial, SF 0.04, contrast 0.8 support; Allen F1/F0 uses 28 complete Brain Observatory sessions.\n"
        "The F1/F0 all-area fit includes the VISp-to-post-V1 step; the post-V1-only slope interval crosses zero.",
        ha="center", va="bottom", fontsize=7.8, color="#666666",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mouse = load_mousev2_units(
        apply_qc=args.population_profile is None,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir,
        flash_metrics_dir=args.flash_metrics_dir,
        flash_variant=args.flash_variant,
        population_profile=args.population_profile,
    )
    allen = load_allen_units(args.config, population_profile=args.population_profile)
    mouse_timescale_common = load_mousev2_units(
        apply_qc=False,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir or MOUSE_CANONICAL_GRATING_METRICS,
        flash_metrics_dir=args.flash_metrics_dir,
        flash_variant=args.flash_variant,
        population_profile="common_qc",
    )
    allen_timescale_common = load_allen_units(
        args.config, population_profile="common_qc"
    )
    population_flow = pd.DataFrame()
    population_units = pd.DataFrame()
    if args.timescale_population == "historical_proxy_full20":
        mouse_timescale_common, population_flow, population_units = historical_proxy_full20_population(
            mouse_timescale_common, rf_path=args.mouse_parametric_rf_fits.resolve()
        )
    else:
        population_flow = pd.DataFrame([{
            "step": "common_qc", "rule": "Frozen MouseV2 common QC",
            "input_units": len(mouse_timescale_common),
            "retained_units": len(mouse_timescale_common), "excluded_at_step": 0,
        }])
        population_units = mouse_timescale_common[
            ["unit_id", "session_num", "probe_letter"]
        ].copy()
    grating_metric = args.grating_metric
    if args.grating_metric == "f1_f0_dg" and args.f1_f0_mode == "harmonized":
        mouse, allen = apply_harmonized_f1_f0(
            mouse,
            allen,
            mouse_path=args.mouse_harmonized_f1_f0.resolve(),
            allen_path=args.allen_harmonized_f1_f0.resolve(),
        )
        grating_metric = HARMONIZED_F1_COLUMN
    grating_label = "log10 F1/F0" if args.grating_metric == "f1_f0_dg" else "log10 modulation index"
    metric_specs = [
        ("time_to_first_spike_fl", "TTFS (ms)"),
        (grating_metric, grating_label),
        ("timescale_ac", "Response timescale (ms)"),
    ]

    tables = []
    allen_v1_tables = []
    timescale_draw_coverage = pd.DataFrame()
    for metric_index, (metric, label) in enumerate(metric_specs):
        if label == "TTFS (ms)":
            override = pd.read_csv(args.response_filtered_ttfs_override.resolve())
            required = {"dataset", "metric", "session_id", "group", "mean", "n_units"}
            if not required.issubset(override.columns):
                raise ValueError(
                    f"TTFS override lacks {sorted(required.difference(override.columns))}"
                )
            override = override.loc[override.metric.eq(label)].copy()
            if set(override.dataset.unique()) != {"Within-V1", "Post-V1", "Allen-V1"}:
                raise ValueError("TTFS override does not contain all three Figure 3 datasets")
            tables.append(override.loc[override.dataset.eq("Within-V1")].copy())
            tables.append(override.loc[
                override.dataset.eq("Post-V1") & override.group.isin(AREA_ORDER)
            ].copy())
            allen_v1_tables.append(
                override.loc[override.dataset.eq("Allen-V1")].copy().assign(
                    group="Visual Coding VISp"
                )
            )
            continue
        mouse_metric_table = mouse
        allen_metric_table = allen
        if label == "Response timescale (ms)":
            mouse_metric_table = mouse_timescale_common
            allen_metric_table = allen_timescale_common
        if label == "Response timescale (ms)" and args.timescale_view == "allen_matched_150":
            matched_table, timescale_draw_coverage = matched_mouse_timescale_session_group_table(
                mouse_timescale_common,
                bridge_path=args.mouse_timescale_trial_bridge.resolve(),
                groups=list(config["display_probe_order"]),
                min_units=args.min_units,
                return_coverage=True,
            )
            tables.append(matched_table)
        else:
            tables.append(session_group_table(
                mouse_metric_table, dataset="Within-V1", session_column="session_num", group_column="probe_letter",
                groups=list(config["display_probe_order"]), metric=metric, metric_label=label,
                metric_index=metric_index, min_units=args.min_units,
            ))
        tables.append(session_group_table(
            allen_metric_table, dataset="Post-V1", session_column="ecephys_session_id", group_column="area_coarse",
            groups=AREA_ORDER, metric=metric, metric_label=label,
            metric_index=metric_index, min_units=args.min_units,
        ))
        allen_v1_tables.append(session_group_table(
            allen_metric_table, dataset="Allen-V1", session_column="ecephys_session_id", group_column="area_coarse",
            groups=["V1"], metric=metric, metric_label=label,
            metric_index=metric_index, min_units=args.min_units,
        ).assign(group="Visual Coding VISp"))
    group_table = pd.concat(tables, ignore_index=True)
    allen_v1_table = pd.concat(allen_v1_tables, ignore_index=True)
    spread_table = session_spreads(group_table, args.min_hva_areas)
    hierarchy_stats = hierarchy_fit_summary(
        group_table,
        allen_v1_table,
        n_bootstrap=args.n_bootstrap,
        rng=np.random.default_rng(args.seed + 101),
    )

    spread_stats: dict[str, dict[str, float]] = {}
    identity_stats: dict[str, dict[str, float]] = {}
    identity_draws: dict[str, dict[str, np.ndarray]] = {}
    for _, label in metric_specs:
        identity_subset = group_table.loc[group_table["metric"] == label]
        v1_identity = identity_subset.loc[identity_subset["dataset"] == "Within-V1"]
        hva_identity = identity_subset.loc[identity_subset["dataset"] == "Post-V1"]
        v1_omega, v1_draws = clustered_omega_squared(
            v1_identity, n_bootstrap=args.n_bootstrap, rng=rng
        )
        hva_omega, hva_draws = clustered_omega_squared(
            hva_identity, n_bootstrap=args.n_bootstrap, rng=rng
        )
        delta_draws = hva_draws - v1_draws
        delta_low, delta_high = np.percentile(delta_draws, [2.5, 97.5])
        v1_low, v1_high = np.percentile(v1_draws, [2.5, 97.5])
        hva_low, hva_high = np.percentile(hva_draws, [2.5, 97.5])
        identity_stats[label] = {
            "v1_omega": v1_omega, "v1_ci_low": float(v1_low), "v1_ci_high": float(v1_high),
            "hva_omega": hva_omega, "hva_ci_low": float(hva_low), "hva_ci_high": float(hva_high),
            "delta": float(hva_omega - v1_omega),
            "delta_ci_low": float(delta_low), "delta_ci_high": float(delta_high),
            "p_le_zero": float(np.mean(delta_draws <= 0)),
        }
        identity_draws[label] = {"v1": v1_draws, "hva": hva_draws, "delta": delta_draws}
        subset = spread_table.loc[spread_table["metric"] == label]
        v1 = subset.loc[subset["dataset"] == "Within-V1", "spread_sd"].to_numpy(float)
        hva = subset.loc[subset["dataset"] == "Post-V1", "spread_sd"].to_numpy(float)
        delta, low, high, p_le_zero = bootstrap_mean_difference(
            v1, hva, n_bootstrap=args.n_bootstrap, rng=rng
        )
        loo = leave_one_out_differences(v1, hva)
        spread_stats[label] = {
            "v1_mean": float(np.mean(v1)), "hva_mean": float(np.mean(hva)),
            "delta": delta, "ci_low": low, "ci_high": high, "p_le_zero": p_le_zero,
            "loo_min": float(np.min(loo)), "loo_max": float(np.max(loo)),
            "n_v1": int(len(v1)), "n_hva": int(len(hva)),
        }

    group_path = output_dir / "Figure3_robust_session_group_means.csv"
    spread_path = output_dir / "Figure3_robust_session_spreads.csv"
    hierarchy_stats_path = output_dir / "Figure3_hierarchy_fit_stats.csv"
    figure_path = output_dir / "Figure3_robust_identity_comparison.png"
    panel_a_path = output_dir / "Figure3_robust_identity_panel_A.png"
    single_panel_path = output_dir / "Figure3_single_panel_identity_variation.png"
    session_panel_path = output_dir / "Figure3_session_level_variation_comparison.png"
    session_panel_allen_v1_path = output_dir / "Figure3_session_level_variation_with_Visual_Coding_VISp.png"
    hierarchy_hist_path = output_dir / "Figure3_hierarchy_fits_with_session_histograms.png"
    report_path = output_dir / "Figure3_robust_identity_comparison.md"
    population_flow_path = output_dir / "Figure3_timescale_population_flow.csv"
    population_units_path = output_dir / "Figure3_timescale_population_units.csv"
    draw_coverage_path = output_dir / "Figure3_timescale_matched_draw_coverage.csv"
    group_table.to_csv(group_path, index=False)
    spread_table.to_csv(spread_path, index=False)
    hierarchy_stats.to_csv(hierarchy_stats_path, index=False)
    population_flow.to_csv(population_flow_path, index=False)
    population_units.to_csv(population_units_path, index=False)
    if not timescale_draw_coverage.empty:
        timescale_draw_coverage.to_csv(draw_coverage_path, index=False)
    render_figure(group_table, spread_table, identity_stats, spread_stats, figure_path)
    render_identity_panel(identity_stats, panel_a_path)
    render_single_panel_identity_variation(identity_stats, identity_draws, single_panel_path)
    render_session_level_variation_comparison(group_table, session_panel_path)
    render_session_level_variation_comparison(
        group_table, session_panel_allen_v1_path, allen_v1_table=allen_v1_table
    )
    render_hierarchy_fit_histogram_comparison(
        group_table, allen_v1_table, hierarchy_hist_path, seed=args.seed,
        timescale_view=args.timescale_view,
        timescale_population=args.timescale_population,
    )

    lines = [
        "# Robust within-V1 versus post-V1 spread comparison", "",
        f"_Generated {date.today().isoformat()}._", "",
        "## Primary estimand", "",
        "The cortical HVA set is LM, RL, AL, PM, and AM. The thalamic lateral",
        "posterior nucleus (LP) is excluded from the cortical-HVA estimand.", "",
        "The primary question is whether stable HVA identity explains more session-level",
        "variance than stable probe identity within V1. Omega-squared is computed from",
        "session-by-group means. Whole session vectors are resampled so simultaneously",
        "recorded groups remain together in every bootstrap draw.", "",
        "| Metric | V1 probe ω² | 95% CI | HVA area ω² | 95% CI | Δω² HVA−V1 | 95% CI | P(Δ≤0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, label in metric_specs:
        result = identity_stats[label]
        lines.append(
            f"| {label} | {result['v1_omega']:.4f} | [{result['v1_ci_low']:+.4f}, {result['v1_ci_high']:+.4f}] | "
            f"{result['hva_omega']:.4f} | [{result['hva_ci_low']:+.4f}, {result['hva_ci_high']:+.4f}] | "
            f"{result['delta']:+.4f} | [{result['delta_ci_low']:+.4f}, {result['delta_ci_high']:+.4f}] | "
            f"{result['p_le_zero']:.3f} |"
        )
    lines += [
        "", "All three direct identity-comparison intervals cross zero. The current data",
        "therefore do not establish that HVA identity explains more variation than position",
        "within V1.", "", "## Secondary heterogeneity diagnostic", "",
        "For each session, the diagnostic computes the sample SD across available group means.",
        "MouseV2 sessions require all four probes; Allen sessions require at least",
        f"{args.min_hva_areas} of the {len(AREA_ORDER)} cortical post-V1 areas. This measures within-session separation,",
        "not stable named-group identity.", "",
        "| Metric | V1 sessions | HVA sessions | Mean V1 SD | Mean HVA SD | Δ HVA−V1 | 95% CI | P(Δ≤0) | LOO Δ range |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, label in metric_specs:
        result = spread_stats[label]
        lines.append(
            f"| {label} | {result['n_v1']} | {result['n_hva']} | {result['v1_mean']:.4g} | "
            f"{result['hva_mean']:.4g} | {result['delta']:+.4g} | "
            f"[{result['ci_low']:+.4g}, {result['ci_high']:+.4g}] | "
            f"{result['p_le_zero']:.3f} | [{result['loo_min']:+.4g}, {result['loo_max']:+.4g}] |"
        )
    lines += [
        "", "## Hierarchy-score fits", "",
        "Slopes use area-level centers. Confidence intervals resample session means within",
        "each area. The post-V1-only fit is reported separately so a VISp-to-HVA step is not",
        "mistaken for a graded trend among higher areas.", "",
        "| Metric | Scope | Areas | Slope | 95% CI | Area-mean r |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in hierarchy_stats.itertuples():
        lines.append(
            f"| {row.metric} | {row.scope} | {row.area_points} | "
            f"{row.slope_per_hierarchy_score:+.4g} | "
            f"[{row.bootstrap_95ci_low:+.4g}, {row.bootstrap_95ci_high:+.4g}] | "
            f"{row.area_mean_r:+.3f} |"
        )
    lines += [
        "", "## Interpretation limits", "",
        "- Probe letters are categorical within-V1 locations, not numerical hierarchy scores.",
        "- F1/F0 uses common QC and matched 1-s, 15-trial, SF 0.04, contrast 0.8 support.",
        "  Allen F1/F0 is restricted to the 28 complete Brain Observatory sessions; TTFS",
        "  and timescale retain their existing source-session support.",
        "- Session-centering removes global session offsets; it does not match RF location,",
        "  layer, depth, firing rate, stimulus support, or cell type.",
        "- Allen sessions contain different HVA subsets. The primary minimum-area rule is",
        "  explicit, and `n_groups` plus `groups_present` are preserved in the spread CSV.",
        "- Unit-level KDEs are deliberately omitted: they weight sessions by retained unit count",
        "  and are not the inferential estimand.", "",
        "## Audit artifacts", "",
        f"- `{group_path.name}`: every retained session × group mean, unit count, and centered value.",
        f"- `{spread_path.name}`: every retained session spread and observed group composition.",
        f"- `{hierarchy_stats_path.name}`: all-area and post-V1-only hierarchy slopes with session-bootstrap intervals.",
        f"- `{population_flow_path.name}`: gate-by-gate MouseV2 timescale-population selection counts.",
        f"- `{population_units_path.name}`: retained MouseV2 timescale units and RF-proxy fields.",
        f"- `{draw_coverage_path.name}`: session × probe coverage across the ten matched-flash draws.",
        f"- `{figure_path.name}` and `{figure_path.with_suffix('.pdf').name}`: rendered figure.",
        f"- `{panel_a_path.name}` and `{panel_a_path.with_suffix('.pdf').name}`: standalone identity-effect panel.",
        f"- `{single_panel_path.name}` and `{single_panel_path.with_suffix('.pdf').name}`: single-panel visual summary without printed statistics.",
        f"- `{session_panel_path.name}` and `{session_panel_path.with_suffix('.pdf').name}`: matched session-level probe/area comparison in the earlier split-panel style.",
        f"- `{session_panel_allen_v1_path.name}` and `{session_panel_allen_v1_path.with_suffix('.pdf').name}`: the same comparison with Visual Coding VISp added as a V1 reference.",
        f"- `{hierarchy_hist_path.name}` and `{hierarchy_hist_path.with_suffix('.pdf').name}`: hierarchy-score fits with session-level marginal histograms.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {figure_path}")
    print(f"Saved {report_path}")
    for label, result in identity_stats.items():
        print("identity", label, result)
    for label, result in spread_stats.items():
        print("spread", label, result)


if __name__ == "__main__":
    main()
