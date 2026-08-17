"""Provisional response-property view using measured RF peak coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_config, load_mousev2_units  # noqa: E402
from common.figure3_rf import DEFAULT_RF_IMPORT_DIR, load_rf_import  # noqa: E402


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
np.random.seed(42)

PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
METRICS = (
    ("time_to_first_spike_fl", "TTFS (ms)", lambda values: values * 1000),
    ("f1_f0_dg", "log10 F1/F0", lambda values: np.log10(np.clip(values, 1e-6, None))),
    ("timescale_ac", "Response timescale (ms)", lambda values: values),
)


def configure_metrics(grating_metric: str) -> tuple:
    label = "log10 F1/F0" if grating_metric == "f1_f0_dg" else "log10 modulation index"
    return (
        ("time_to_first_spike_fl", "TTFS (ms)", lambda values: values * 1000),
        (grating_metric, label, lambda values: np.log10(np.clip(values, 1e-6, None))),
        ("timescale_ac", "Response timescale (ms)", lambda values: values),
    )


def metric_mask(table: pd.DataFrame, metric_index: int) -> pd.Series:
    if metric_index == 0:
        return table["time_to_first_spike_fl"].astype(float) < 0.1
    if metric_index == 1:
        return table[METRICS[1][0]].astype(float) > 0
    return (
        table["timescale_ac"].astype(float).between(1, 300)
        & (table["spike_count_ac"].astype(float) > 50)
        & (table["err_ac"].astype(float) < 20)
    )


def session_probe_metric_means(units: pd.DataFrame) -> pd.DataFrame:
    keys = ["site", "session_num", "subject_id", "probe_letter"]
    result = units[keys].drop_duplicates().copy()
    for metric_index, (metric, _, transform) in enumerate(METRICS):
        subset = units[metric_mask(units, metric_index)].copy()
        subset["metric_value"] = transform(subset[metric].astype(float))
        subset = subset[np.isfinite(subset["metric_value"])]
        grouped = subset.groupby(keys)["metric_value"].agg(
            metric_mean=lambda values: float(np.mean(values)) if len(values) >= 5 else np.nan,
            metric_n="size",
        )
        grouped = grouped.rename(
            columns={"metric_mean": metric, "metric_n": f"{metric}_n"}
        ).reset_index()
        result = result.merge(grouped, on=keys, how="left", validate="one_to_one")
    return result.rename(columns={"probe_letter": "probe"})


def asymmetric_errors(center: pd.Series, low: pd.Series, high: pd.Series) -> np.ndarray:
    return np.vstack(
        [
            (center - low).clip(lower=0).to_numpy(dtype=float),
            (high - center).clip(lower=0).to_numpy(dtype=float),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Figure3")
    parser.add_argument("--rf-import-dir", type=Path, default=DEFAULT_RF_IMPORT_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--grating-metrics-dir", type=Path, default=None)
    parser.add_argument("--flash-metrics-dir", type=Path, default=None)
    parser.add_argument(
        "--flash-variant", choices=("pooled", "bright", "dark"), default="pooled"
    )
    parser.add_argument(
        "--grating-metric", choices=("f1_f0_dg", "mod_idx_dg"), default="f1_f0_dg"
    )
    parser.add_argument("--population-profile", default=None)
    return parser.parse_args()


def main() -> None:
    global METRICS
    args = parse_args()
    METRICS = configure_metrics(args.grating_metric)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    _, summary, ordering, manifest = load_rf_import(args.rf_import_dir)
    units = load_mousev2_units(
        apply_qc=args.population_profile is None,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir,
        flash_metrics_dir=args.flash_metrics_dir,
        flash_variant=args.flash_variant,
        population_profile=args.population_profile,
    )
    metrics = session_probe_metric_means(units)
    summary_for_merge = summary.rename(columns={"site_number": "session_num"})
    plot_data = metrics.merge(
        summary_for_merge,
        on=["site", "session_num", "subject_id", "probe"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["session_num", "probe"])
    if len(plot_data) != 32:
        raise ValueError(f"Expected 32 session-probe RF/metric rows, found {len(plot_data)}")

    fig = plt.figure(figsize=(13, 15))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.35, 1, 1, 1], hspace=0.34, wspace=0.25)
    ax_rf = fig.add_subplot(grid[0, 0])
    ax_order = fig.add_subplot(grid[0, 1])
    display_order = list(config["display_probe_order"])

    for _, group in summary.groupby("site_number", sort=True):
        ordered = group.set_index("probe").loc[display_order].reset_index()
        ax_rf.plot(
            ordered["rf_center_x_deg"],
            ordered["rf_center_y_deg"],
            color="#999999",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )
        for probe in display_order:
            row = ordered[ordered["probe"] == probe]
            ax_rf.errorbar(
                row["rf_center_x_deg"],
                row["rf_center_y_deg"],
                xerr=asymmetric_errors(
                    row["rf_center_x_deg"],
                    row["rf_center_x_ci_low_deg"],
                    row["rf_center_x_ci_high_deg"],
                ),
                yerr=asymmetric_errors(
                    row["rf_center_y_deg"],
                    row["rf_center_y_ci_low_deg"],
                    row["rf_center_y_ci_high_deg"],
                ),
                fmt="o",
                color=PROBE_COLORS[probe],
                markeredgecolor="black",
                markeredgewidth=0.35,
                markersize=6,
                alpha=0.70,
                capsize=2,
                linewidth=0.8,
                zorder=2,
            )

    for probe in display_order:
        probe_summary = summary[summary["probe"] == probe]
        grand_x = float(probe_summary["rf_center_x_deg"].median())
        grand_y = float(probe_summary["rf_center_y_deg"].median())
        ax_rf.plot(
            grand_x,
            grand_y,
            marker="D",
            markersize=11,
            color=PROBE_COLORS[probe],
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=4,
        )
        ax_rf.annotate(
            probe,
            (grand_x, grand_y),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=PROBE_COLORS[probe],
        )

    ax_rf.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_rf.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_rf.set_xlim(-45, 45)
    ax_rf.set_ylim(-45, 45)
    ax_rf.set_aspect("equal", adjustable="box")
    ax_rf.set_xlabel("RF stimulus x / azimuth (deg)")
    ax_rf.set_ylabel("RF stimulus y / elevation (deg)")
    ax_rf.set_title(
        "Provisional session × probe RF centers\n"
        "Pilot-QC median per-unit grid argmax; error bars are unit-bootstrap 95% intervals",
        fontsize=11,
    )
    ax_rf.grid(alpha=0.15)

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROBE_COLORS[probe],
            markeredgecolor="black",
            markersize=8,
            label=f"Probe {probe}",
        )
        for probe in display_order
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markersize=8,
            label="Across-session probe median",
        )
    )
    ax_rf.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.85)

    probe_positions = np.arange(len(display_order))
    for _, group in summary.groupby("site_number", sort=True):
        ordered = group.set_index("probe").loc[display_order].reset_index()
        ax_order.plot(
            probe_positions,
            ordered["rf_center_x_deg"],
            color="#999999",
            alpha=0.45,
            linewidth=1.0,
            zorder=1,
        )
        for probe_index, probe in enumerate(display_order):
            row = ordered.iloc[probe_index]
            ax_order.errorbar(
                probe_index,
                row["rf_center_x_deg"],
                yerr=np.array(
                    [
                        [max(0.0, row["rf_center_x_deg"] - row["rf_center_x_ci_low_deg"])],
                        [max(0.0, row["rf_center_x_ci_high_deg"] - row["rf_center_x_deg"])],
                    ]
                ),
                fmt="o",
                color=PROBE_COLORS[probe],
                markeredgecolor="black",
                markeredgewidth=0.35,
                markersize=6,
                alpha=0.72,
                capsize=2,
                linewidth=0.8,
                zorder=2,
            )
    ax_order.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_order.set_xticks(probe_positions)
    ax_order.set_xticklabels(display_order)
    ax_order.set_ylim(-45, 45)
    ax_order.set_xlabel("Declared categorical probe order")
    ax_order.set_ylabel("RF stimulus x / azimuth (deg)")
    ax_order.set_title(
        "Ordering sensitivity across sessions\n"
        f"B>C>A>E: {int(ordering['declared_order_strictly_descending_x'].sum())}/8 strict; "
        f"{int(ordering['declared_order_descending_x_allowing_ties'].sum())}/8 allowing ties",
        fontsize=11,
    )
    ax_order.grid(alpha=0.15)
    [ax_order.spines[side].set_visible(False) for side in ("top", "right")]

    correlation_rows = []
    coordinate_specs = (
        ("rf_center_x_deg", "RF azimuth (deg)", "rf_center_x_ci_low_deg", "rf_center_x_ci_high_deg"),
        ("rf_center_y_deg", "RF elevation (deg)", "rf_center_y_ci_low_deg", "rf_center_y_ci_high_deg"),
    )
    for metric_index, (metric, metric_label, _) in enumerate(METRICS):
        for coordinate_index, (coordinate, coordinate_label, low, high) in enumerate(coordinate_specs):
            ax = fig.add_subplot(grid[metric_index + 1, coordinate_index])
            valid = plot_data[[coordinate, metric, low, high, "probe"]].dropna()
            for probe in display_order:
                probe_data = valid[valid["probe"] == probe]
                ax.errorbar(
                    probe_data[coordinate],
                    probe_data[metric],
                    xerr=asymmetric_errors(probe_data[coordinate], probe_data[low], probe_data[high]),
                    fmt="o",
                    color=PROBE_COLORS[probe],
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    markersize=6,
                    alpha=0.75,
                    capsize=2,
                    linewidth=0.7,
                )
            rho, p_value = spearmanr(valid[coordinate], valid[metric])
            correlation_rows.append(
                {
                    "metric": metric,
                    "coordinate": coordinate,
                    "n_session_probe": int(len(valid)),
                    "spearman_rho": float(rho),
                    "spearman_p_descriptive": float(p_value),
                }
            )
            ax.text(
                0.03,
                0.94,
                f"descriptive $r_S$={rho:.2f}; n={len(valid)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
            )
            ax.set_xlabel(coordinate_label)
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.12)
            [ax.spines[side].set_visible(False) for side in ("top", "right")]

    flash_label = (
        f"{args.flash_variant} flashes"
        if args.flash_metrics_dir is not None
        else "legacy pooled flashes"
    )
    population_label = args.population_profile or "legacy default QC"
    fig.suptitle(
        "MouseV2 response properties at measured retinotopic positions\n"
        f"{args.grating_metric}; {flash_label}; raw TTFS; {population_label}; no gaze correction",
        y=0.995,
        fontsize=13,
    )
    figure_path = output_dir / "Figure3_rf_position.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data_path = output_dir / "rf_metric_session_probe.csv"
    plot_data.to_csv(data_path, index=False)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output_dir / "rf_metric_correlations.csv", index=False)

    strict = int(ordering["declared_order_strictly_descending_x"].sum())
    ties = int(ordering["declared_order_descending_x_allowing_ties"].sum())
    x_sensitivity = (
        summary["rf_center_x_deg"] - summary["rf_center_x_default_qc_deg"]
    ).abs()
    y_sensitivity = (
        summary["rf_center_y_deg"] - summary["rf_center_y_default_qc_deg"]
    ).abs()
    report_lines = [
        "# Iteration 2 — provisional measured retinotopy",
        "",
        f"- RF unit mapping: {manifest['validation']['mapped_units']:,} / 20,374 units.",
        f"- Probe centers: {len(summary)} session × probe estimates.",
        f"- Position population: {manifest['validation']['pilot_qc_units']:,} Pilot-QC units.",
        f"- B>C>A>E is strictly descending in RF azimuth in {strict}/8 sessions and descending allowing ties in {ties}/8.",
        f"- Median absolute Pilot-QC versus default-QC center shift: {x_sensitivity.median():.1f} deg x, {y_sensitivity.median():.1f} deg y.",
        f"- Median per-probe grid-edge fraction: {summary['rf_grid_edge_fraction'].median():.2f}.",
        "- Gaze correction: none; no gaze-corrected all-session export is currently available.",
        "",
        "The measured centers do not support treating B>C>A>E as a universal one-dimensional order. The two-dimensional coordinates should be retained, while the categorical probe view remains a sensitivity analysis.",
        "",
        "These raw grid-argmax peaks do not provide final RF significance or area filters and must not be used for `p_value_rf`/`area_rf` selection.",
        "",
        f"Response-property checkpoint: `{args.grating_metric}`, `{flash_label}`, population `{population_label}`. TTFS is raw relative to NWB start_time.",
        "",
        "## Descriptive response-coordinate associations",
        "",
        "The correlations below use 32 session × probe observations and are descriptive only; they do not account for the matched probes within sessions.",
        "",
        "| Metric | Coordinate | n | Spearman rho |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in correlation_rows:
        report_lines.append(
            f"| {row['metric']} | {row['coordinate']} | {row['n_session_probe']} | {row['spearman_rho']:.3f} |"
        )
    (output_dir / "rf_position_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"Saved → {figure_path}")
    print(f"Saved → {data_path}")
    print(f"Saved → {output_dir / 'rf_position_report.md'}")


if __name__ == "__main__":
    main()
