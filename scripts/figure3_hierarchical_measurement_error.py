"""Build measurement-error inputs for the Figure 3 hierarchical identity model.

Checkpoint 1 only: estimate each session x probe/area mean and its neuron-bootstrap
standard error under the exact Figure 3 metric filters.  This does not fit the
hierarchical variance-component model.  The output is deliberately auditable so
the uncertainty inputs can be inspected before priors or partial pooling are added.
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_config, load_mousev2_units  # noqa: E402
from scripts.figure3_robust_spread_comparison import AREA_ORDER, valid_values  # noqa: E402


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Figure3" / "hierarchical_error_model")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--grating-metrics-dir", type=Path, default=None)
    parser.add_argument("--flash-metrics-dir", type=Path, default=None)
    parser.add_argument("--flash-variant", choices=("pooled", "bright", "dark"), default="pooled")
    parser.add_argument("--grating-metric", choices=("f1_f0_dg", "mod_idx_dg"), default="f1_f0_dg")
    parser.add_argument("--population-profile", default=None)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def bootstrap_mean_uncertainty(
    values: np.ndarray,
    *,
    repetitions: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Neuron bootstrap for a group mean, evaluated in bounded-memory batches."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite values are required")
    draws = np.empty(repetitions, dtype=float)
    cursor = 0
    while cursor < repetitions:
        count = min(batch_size, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws[cursor: cursor + count] = np.mean(values[indices], axis=1)
        cursor += count
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "unit_sd": float(np.std(values, ddof=1)),
        "analytic_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "bootstrap_bias": float(np.mean(draws) - np.mean(values)),
    }


def build_measurement_table(
    frame: pd.DataFrame,
    *,
    dataset: str,
    session_column: str,
    group_column: str,
    groups: list[str],
    metric_specs: list[tuple[str, str]],
    min_units: int,
    repetitions: int,
    batch_size: int,
    rng: np.random.Generator,
    subject_column: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = frame.loc[frame[group_column].isin(groups)]
    for metric_index, (metric, metric_label) in enumerate(metric_specs):
        for (session, group), part in selected.groupby([session_column, group_column], sort=True):
            values = valid_values(part, metric, metric_index)
            if len(values) < min_units:
                continue
            uncertainty = bootstrap_mean_uncertainty(
                values, repetitions=repetitions, batch_size=batch_size, rng=rng
            )
            subject = ""
            if subject_column is not None and subject_column in part:
                nonmissing = part[subject_column].dropna().astype(str).unique()
                if len(nonmissing) == 1:
                    subject = nonmissing[0]
            rows.append({
                "dataset": dataset,
                "metric": metric_label,
                "metric_column": metric,
                "session_id": str(session),
                "subject_id": subject,
                "group": str(group),
                "n_units": int(len(values)),
                **uncertainty,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["session_mean_unweighted"] = result.groupby(
        ["dataset", "metric", "session_id"]
    )["mean"].transform("mean")
    result["centered_mean"] = result["mean"] - result["session_mean_unweighted"]
    result["precision_weight"] = 1.0 / np.square(result["bootstrap_se"])
    result["bootstrap_to_analytic_se_ratio"] = result["bootstrap_se"] / result["analytic_se"]
    return result


def summarize_measurement_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, metric), part in table.groupby(["dataset", "metric"], sort=False):
        rho, p_value = spearmanr(np.log10(part["n_units"]), np.log10(part["bootstrap_se"]))
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "session_group_cells": int(len(part)),
            "sessions": int(part["session_id"].nunique()),
            "median_units": float(part["n_units"].median()),
            "min_units": int(part["n_units"].min()),
            "median_bootstrap_se": float(part["bootstrap_se"].median()),
            "p90_bootstrap_se": float(part["bootstrap_se"].quantile(0.9)),
            "median_bootstrap_to_analytic_ratio": float(part["bootstrap_to_analytic_se_ratio"].median()),
            "spearman_log_n_log_se": float(rho),
            "spearman_p": float(p_value),
        })
    return pd.DataFrame(rows)


def select_diagnostic_cases(table: pd.DataFrame) -> pd.DataFrame:
    """Auditable expected/noisy/typical cases for the checkpoint review."""
    selected = []
    for (dataset, metric), part in table.groupby(["dataset", "metric"], sort=False):
        ordered = part.sort_values(["bootstrap_se", "session_id", "group"])
        roles = {
            "lowest measurement error": ordered.iloc[0],
            "typical measurement error": ordered.iloc[len(ordered) // 2],
            "highest measurement error": ordered.iloc[-1],
        }
        for role, row in roles.items():
            selected.append({
                "dataset": dataset,
                "metric": metric,
                "selection_role": role,
                "session_id": row["session_id"],
                "group": row["group"],
                "n_units": int(row["n_units"]),
                "mean": float(row["mean"]),
                "bootstrap_se": float(row["bootstrap_se"]),
                "criterion": "rank of neuron-bootstrap SE within dataset and metric",
            })
    return pd.DataFrame(selected)


def render_diagnostics(table: pd.DataFrame, output: Path) -> None:
    metrics = list(table["metric"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 10), gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    colors = {"Within-V1": "#6f62a6", "Post-V1": "#666666"}
    markers = {"Within-V1": "o", "Post-V1": "s"}
    for row, metric in enumerate(metrics):
        local = table.loc[table["metric"] == metric]
        ax = axes[row, 0]
        for dataset in ["Within-V1", "Post-V1"]:
            part = local.loc[local["dataset"] == dataset]
            ax.scatter(part["n_units"], part["bootstrap_se"], s=22,
                       facecolors="none" if dataset == "Within-V1" else colors[dataset],
                       edgecolors=colors[dataset], marker=markers[dataset], alpha=0.65,
                       label=dataset if row == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Valid units in session × group (log scale)")
        ax.set_ylabel("Neuron-bootstrap SE (log scale)")
        ax.set_title(f"{metric}: precision versus unit count", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[row, 1]
        positions = [0, 1]
        for position, dataset in zip(positions, ["Within-V1", "Post-V1"]):
            values = local.loc[local["dataset"] == dataset, "bootstrap_to_analytic_se_ratio"].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(position + jitter, values, s=17, alpha=0.55, color=colors[dataset],
                       edgecolor="black", linewidth=0.2)
            ax.hlines(np.median(values), position - 0.22, position + 0.22,
                      color=colors[dataset], lw=2.5)
        ax.axhline(1, color="#888888", lw=0.9, ls="--")
        ax.set_xticks(positions, ["Within V1", "Post-V1 HVA"])
        ax.set_ylabel("Bootstrap SE / analytic SE")
        ax.set_title(f"{metric}: bootstrap sanity check", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hierarchical error model checkpoint 1: measurement-error inputs", fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    mouse = load_mousev2_units(
        apply_qc=args.population_profile is None,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir,
        flash_metrics_dir=args.flash_metrics_dir,
        flash_variant=args.flash_variant,
        population_profile=args.population_profile,
    )
    allen = load_allen_units(args.config, population_profile=args.population_profile)
    grating_label = "log10 F1/F0" if args.grating_metric == "f1_f0_dg" else "log10 modulation index"
    metric_specs = [
        ("time_to_first_spike_fl", "TTFS (ms)"),
        (args.grating_metric, grating_label),
        ("timescale_ac", "Response timescale (ms)"),
    ]
    mouse_table = build_measurement_table(
        mouse, dataset="Within-V1", session_column="session_num", group_column="probe_letter",
        groups=list(config["display_probe_order"]), metric_specs=metric_specs,
        min_units=args.min_units, repetitions=args.bootstrap_repetitions,
        batch_size=args.bootstrap_batch_size, rng=rng, subject_column="subject_id",
    )
    allen_subject = "specimen_id" if "specimen_id" in allen else None
    allen_table = build_measurement_table(
        allen, dataset="Post-V1", session_column="ecephys_session_id", group_column="area_coarse",
        groups=AREA_ORDER, metric_specs=metric_specs,
        min_units=args.min_units, repetitions=args.bootstrap_repetitions,
        batch_size=args.bootstrap_batch_size, rng=rng, subject_column=allen_subject,
    )
    table = pd.concat([mouse_table, allen_table], ignore_index=True)
    summary = summarize_measurement_table(table)
    cases = select_diagnostic_cases(table)

    table_path = output_dir / "session_group_measurement_error.csv"
    summary_path = output_dir / "measurement_error_summary.csv"
    cases_path = output_dir / "measurement_error_diagnostic_cases.csv"
    figure_path = output_dir / "measurement_error_diagnostics.png"
    report_path = output_dir / "MEASUREMENT_ERROR_CHECKPOINT.md"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    cases.to_csv(cases_path, index=False)
    render_diagnostics(table, figure_path)

    lines = [
        "# Hierarchical error model — measurement-error checkpoint", "",
        f"_Generated {date.today().isoformat()} with {args.bootstrap_repetitions} neuron-bootstrap repetitions per session × group cell._", "",
        "## Scope", "",
        "This checkpoint estimates the observed mean and sampling uncertainty for every retained",
        "session × probe/area cell. It does **not** yet fit hierarchical variance components.", "",
        "The resampling unit is a neuron within a session × group cell. This captures finite-unit",
        "uncertainty in the retained metric values, but not shared trial noise, spike-sorting",
        "dependence, or uncertainty from rerunning each per-unit timescale fit and validity gate.", "",
        "## Summary", "",
        "| Dataset | Metric | Cells | Sessions | Median units | Minimum units | Median bootstrap SE | P90 bootstrap SE | Median bootstrap/analytic SE | Spearman log(n) vs log(SE) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.metric} | {row.session_group_cells} | {row.sessions} | "
            f"{row.median_units:.1f} | {row.min_units} | {row.median_bootstrap_se:.4g} | "
            f"{row.p90_bootstrap_se:.4g} | {row.median_bootstrap_to_analytic_ratio:.3f} | "
            f"{row.spearman_log_n_log_se:+.3f} |"
        )
    lines += [
        "", "## Claim gate for the hierarchical fit", "",
        "Proceed only if bootstrap SEs are finite and positive, extreme cells are traceable to",
        "small or intrinsically heterogeneous unit samples, and bootstrap/analytic SE ratios",
        "remain near one for the mean estimand. A large departure would indicate a coding or",
        "transformation problem rather than useful extra uncertainty information.", "",
        "## Required human checkpoint", "",
        "Review the selected lowest, typical, and highest-error cells in",
        "`measurement_error_diagnostic_cases.csv`. The next stage will fit a model only after",
        "these measurement-error inputs are accepted.", "",
        "## Artifacts", "",
        "- `session_group_measurement_error.csv`: model-ready observed means and known-error inputs.",
        "- `measurement_error_summary.csv`: dataset × metric precision audit.",
        "- `measurement_error_diagnostic_cases.csv`: transparent case selection for review.",
        "- `measurement_error_diagnostics.png` and PDF: precision and sanity-check plots.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved {table_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
