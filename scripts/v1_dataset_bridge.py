#!/usr/bin/env python3
"""Diagnose absolute V1 metric offsets between Allen and MouseV2 datasets.

This checkpoint deliberately does not alter or calibrate either dataset.  It
shows session-level centers, separates the two Allen stimulus sets, audits the
Welch frequency grid, and records the raw-data work needed before absolute
cross-dataset values can support a biological interpretation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import welch_modulation_index  # noqa: E402
from common.figure3_mousev2 import (  # noqa: E402
    load_allen_units,
    load_mousev2_units,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "figure3" / "06b_v1_dataset_bridge"
ALLEN_RAW_BRIDGE = ROOT / "data" / "imports" / "allen_v1_raw_bridge_v2"
TIMESCALE_TRIAL_BRIDGE = (
    ROOT / "data" / "imports" / "mousev2_timescale_trial_bridge_v1"
)
PHASE_BRIDGE = ROOT / "data" / "imports" / "v1_grating_phase_bridge_v1"
START_PHASE_BRIDGE = (
    ROOT / "data" / "imports" / "mousev2_grating_start_phase_bridge_v1"
)
SHARED_PHASE_BRIDGE = (
    ROOT / "data" / "imports" / "mousev2_grating_shared_phase_behavior_v1"
)
CORRECTED_WELCH_BRIDGE = (
    ROOT / "data" / "imports" / "mousev2_grating_corrected_welch_bridge_v1"
)
COHORT_ORDER = (
    "Allen Brain Observatory 1.1",
    "Allen Functional Connectivity",
    "MouseV2 V1",
)
COHORT_COLORS = {
    "Allen Brain Observatory 1.1": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2 V1": "#D95F02",
}
COHORT_DISPLAY_LABELS = {
    "Allen Brain Observatory 1.1": "Allen BO",
    "Allen Functional Connectivity": "Allen FC",
    "MouseV2 V1": "Multi-site V1",
}
METRICS = {
    "log10_mod_idx": "log10 modulation index",
    "log10_f1_f0": "log10 F1/F0",
    "timescale_valid_ms": "response timescale (ms)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--grating-metrics-dir",
        type=Path,
        default=ROOT / "data" / "imports" / "mousev2_grating_metrics_v1",
    )
    parser.add_argument(
        "--flash-metrics-dir",
        type=Path,
        default=ROOT / "data" / "imports" / "mousev2_flash_metrics_v1",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    return parser.parse_args()


def _prepare_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    allen = load_allen_units(args.config, population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].eq("V1")].copy()
    allen["dataset"] = "Allen V1"
    allen["session_id"] = allen["ecephys_session_id"].astype(str)
    allen["cohort"] = allen["session_type"].map(
        {
            "brain_observatory_1.1": "Allen Brain Observatory 1.1",
            "functional_connectivity": "Allen Functional Connectivity",
        }
    )

    mouse = load_mousev2_units(
        apply_qc=False,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir,
        flash_metrics_dir=args.flash_metrics_dir,
        flash_variant="pooled",
        population_profile="common_qc",
    )
    mouse["dataset"] = "MouseV2 V1"
    mouse["session_id"] = mouse["session_num"].astype(str)
    mouse["session_type"] = "mousev2"
    mouse["cohort"] = "MouseV2 V1"

    for table in (allen, mouse):
        mod = pd.to_numeric(table["mod_idx_dg"], errors="coerce")
        f1 = pd.to_numeric(table["f1_f0_dg"], errors="coerce")
        tau = pd.to_numeric(table["timescale_ac"], errors="coerce")
        spike_count = pd.to_numeric(table["spike_count_ac"], errors="coerce")
        fit_error = pd.to_numeric(table["err_ac"], errors="coerce")
        table["log10_mod_idx"] = np.log10(mod.where(mod > 0))
        table["log10_f1_f0"] = np.log10(f1.where(f1 > 0))
        table["timescale_valid_ms"] = tau.where(
            tau.between(1, 300) & spike_count.gt(50) & fit_error.lt(20)
        )
    return allen, mouse


def _session_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, session_id), group in table.groupby(
        ["cohort", "session_id"], sort=True
    ):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) < 5:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "session_id": session_id,
                    "metric": metric,
                    "n_units": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                }
            )
    return pd.DataFrame(rows)


def _summary_row(label: str, table: pd.DataFrame, metric: str) -> dict[str, object]:
    values = pd.to_numeric(table[metric], errors="coerce").dropna()
    session = (
        table.groupby("session_id")[metric]
        .agg(["mean", "count"])
        .loc[lambda frame: frame["count"] >= 5]
    )
    return {
        "population": label,
        "metric": metric,
        "valid_units": len(values),
        "pooled_unit_mean": values.mean(),
        "pooled_unit_median": values.median(),
        "valid_sessions": len(session),
        "equal_session_mean": session["mean"].mean(),
        "session_sd": session["mean"].std(ddof=1),
        "session_min": session["mean"].min(),
        "session_max": session["mean"].max(),
    }


def _center_summary(allen: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    populations = {
        "Allen V1 — all": allen,
        "Allen V1 — Brain Observatory 1.1": allen.loc[
            allen["session_type"].eq("brain_observatory_1.1")
        ],
        "Allen V1 — Functional Connectivity": allen.loc[
            allen["session_type"].eq("functional_connectivity")
        ],
        "MouseV2 V1": mouse,
    }
    return pd.DataFrame(
        _summary_row(label, table, metric)
        for label, table in populations.items()
        for metric in METRICS
    )


def _bootstrap_difference(
    mouse_values: np.ndarray,
    allen_values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(np.mean(mouse_values) - np.mean(allen_values))
    rng = np.random.default_rng(seed)
    mouse_indices = rng.integers(
        0, len(mouse_values), size=(n_bootstrap, len(mouse_values))
    )
    allen_indices = rng.integers(
        0, len(allen_values), size=(n_bootstrap, len(allen_values))
    )
    differences = (
        mouse_values[mouse_indices].mean(axis=1)
        - allen_values[allen_indices].mean(axis=1)
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def _contrasts(session: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    mouse = session.loc[session["cohort"].eq("MouseV2 V1")]
    for metric_index, metric in enumerate(METRICS):
        mouse_values = mouse.loc[mouse["metric"].eq(metric), "mean"].to_numpy()
        for cohort_index, cohort in enumerate(COHORT_ORDER[:2]):
            allen_values = session.loc[
                session["cohort"].eq(cohort) & session["metric"].eq(metric),
                "mean",
            ].to_numpy()
            observed, low, high = _bootstrap_difference(
                mouse_values,
                allen_values,
                n_bootstrap=n_bootstrap,
                seed=20260805 + metric_index * 10 + cohort_index,
            )
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"MouseV2 V1 minus {cohort}",
                    "mouse_sessions": len(mouse_values),
                    "allen_sessions": len(allen_values),
                    "difference_equal_session_means": observed,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _welch_lookup() -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for dataset, samples in (("MouseV2", 1000), ("Allen", 2000)):
            frequencies, _ = signal.welch(
                np.zeros(samples), fs=1000.0, nperseg=1024
            )
            for tf in (1.0, 2.0, 4.0, 8.0, 15.0):
                search_index = int(np.searchsorted(frequencies, tf))
                nearest_index = int(np.argmin(np.abs(frequencies - tf)))
                rows.append(
                    {
                        "dataset": dataset,
                        "psth_samples": samples,
                        "welch_nperseg_effective": min(1024, samples),
                        "frequency_resolution_hz": frequencies[1] - frequencies[0],
                        "requested_tf_hz": tf,
                        "searchsorted_frequency_hz": frequencies[search_index],
                        "nearest_frequency_hz": frequencies[nearest_index],
                    }
                )
    return pd.DataFrame(rows)


def _protocol_sensitivity() -> pd.DataFrame:
    """Small Monte Carlo showing estimator dependence, not biological calibration."""
    protocols = (
        ("Mouse-like: 1 s, 15 repeats", 1, 15),
        ("Allen BO-like: 2 s, 15 repeats", 2, 15),
        ("Allen FC-like: 2 s, 75 repeats", 2, 75),
    )
    rows = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        for tf in (1, 2, 4, 8, 15):
            for modulation_depth in (0.0, 0.25, 0.5, 1.0):
                for protocol, duration_s, repeats in protocols:
                    samples = duration_s * 1000
                    time_s = (np.arange(samples) + 0.5) / 1000.0
                    rate_hz = 10.0 * (
                        1.0
                        + modulation_depth
                        * np.sin(2 * np.pi * tf * time_s)
                    )
                    counts = rng.poisson(
                        np.clip(rate_hz / 1000.0, 0, None),
                        size=(repeats, samples),
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        value = welch_modulation_index(counts.mean(axis=0), tf)
                    rows.append(
                        {
                            "seed": seed,
                            "protocol": protocol,
                            "duration_s": duration_s,
                            "repeats": repeats,
                            "temporal_frequency_hz": tf,
                            "simulated_rate_hz": 10.0,
                            "simulated_modulation_depth": modulation_depth,
                            "mod_idx_dg": value,
                        }
                    )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(
            [
                "protocol",
                "duration_s",
                "repeats",
                "temporal_frequency_hz",
                "simulated_rate_hz",
                "simulated_modulation_depth",
            ],
            sort=False,
        )["mod_idx_dg"]
        .agg(median="median", q10=lambda x: x.quantile(0.1), q90=lambda x: x.quantile(0.9))
        .reset_index()
    )


def _timescale_coverage(allen: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat(
        [
            allen[["cohort", "session_id", "timescale_valid_ms"]],
            mouse[["cohort", "session_id", "timescale_valid_ms"]],
        ],
        ignore_index=True,
    )
    return (
        table.groupby(["cohort", "session_id"], sort=True)["timescale_valid_ms"]
        .agg(common_qc_units="size", valid_timescale_units="count")
        .reset_index()
        .assign(
            valid_fraction=lambda frame: frame["valid_timescale_units"]
            / frame["common_qc_units"]
        )
    )


def _tf_session_summary(allen: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat(
        [
            allen[["cohort", "session_id", "pref_tf_dg", "log10_mod_idx"]],
            mouse[["cohort", "session_id", "pref_tf_dg", "log10_mod_idx"]],
        ],
        ignore_index=True,
    )
    table["pref_tf_dg"] = pd.to_numeric(table["pref_tf_dg"], errors="coerce")
    summary = (
        table.groupby(["cohort", "session_id", "pref_tf_dg"], sort=True)[
            "log10_mod_idx"
        ]
        .agg(mean="mean", n_units="count")
        .reset_index()
    )
    return summary.loc[summary["n_units"] >= 5].copy()


def _bootstrap_center(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _strip_panel(
    ax: plt.Axes,
    session: pd.DataFrame,
    metric: str,
    ylabel: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for index, cohort in enumerate(COHORT_ORDER):
        values = session.loc[
            session["cohort"].eq(cohort) & session["metric"].eq(metric), "mean"
        ].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, len(values))
        color = COHORT_COLORS[cohort]
        ax.scatter(
            index + jitter,
            values,
            s=20,
            alpha=0.58,
            color=color,
            edgecolor="none",
            zorder=2,
        )
        center, low, high = _bootstrap_center(values, seed + index + 100)
        ax.errorbar(
            index,
            center,
            yerr=[[center - low], [high - center]],
            fmt="_",
            markersize=18,
            markeredgewidth=2.5,
            linewidth=1.8,
            capsize=3,
            color="black",
            zorder=3,
        )
    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels(("Allen\nBO", "Allen\nFC", "Multi-site\nV1"))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)


def _make_figure(
    session: pd.DataFrame,
    coverage: pd.DataFrame,
    tf_session: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    _strip_panel(axes[0, 0], session, "log10_mod_idx", METRICS["log10_mod_idx"], 1)
    _strip_panel(axes[0, 1], session, "log10_f1_f0", METRICS["log10_f1_f0"], 2)
    _strip_panel(
        axes[0, 2], session, "timescale_valid_ms", METRICS["timescale_valid_ms"], 3
    )

    paired = session.pivot_table(
        index=["cohort", "session_id"], columns="metric", values="mean"
    ).reset_index()
    for cohort in COHORT_ORDER:
        group = paired.loc[paired["cohort"].eq(cohort)].dropna(
            subset=["log10_f1_f0", "log10_mod_idx"]
        )
        axes[1, 0].scatter(
            group["log10_f1_f0"],
            group["log10_mod_idx"],
            s=26,
            alpha=0.7,
            color=COHORT_COLORS[cohort],
            label=COHORT_DISPLAY_LABELS[cohort],
        )
    rho = spearmanr(
        paired["log10_f1_f0"], paired["log10_mod_idx"], nan_policy="omit"
    ).statistic
    axes[1, 0].set(
        xlabel="session mean log10 F1/F0",
        ylabel="session mean log10 modulation index",
        title=f"Metric coupling across sessions (Spearman r={rho:.2f})",
    )
    axes[1, 0].grid(alpha=0.18)
    axes[1, 0].legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(4)
    for index, cohort in enumerate(COHORT_ORDER):
        values = coverage.loc[coverage["cohort"].eq(cohort), "valid_fraction"].to_numpy()
        axes[1, 1].scatter(
            index + rng.uniform(-0.16, 0.16, len(values)),
            values,
            s=20,
            alpha=0.6,
            color=COHORT_COLORS[cohort],
            edgecolor="none",
        )
        axes[1, 1].hlines(values.mean(), index - 0.22, index + 0.22, color="black", lw=2)
    axes[1, 1].set_xticks(range(len(COHORT_ORDER)))
    axes[1, 1].set_xticklabels(("Allen\nBO", "Allen\nFC", "Multi-site\nV1"))
    axes[1, 1].set(
        ylabel="fraction passing timescale validity rules",
        title="Timescale selection coverage",
    )
    axes[1, 1].grid(axis="y", alpha=0.18)

    for cohort in COHORT_ORDER:
        group = tf_session.loc[tf_session["cohort"].eq(cohort)]
        by_tf = group.groupby("pref_tf_dg")["mean"].agg(["mean", "sem"])
        axes[1, 2].errorbar(
            by_tf.index,
            by_tf["mean"],
            yerr=by_tf["sem"],
            marker="o",
            linewidth=1.6,
            capsize=3,
            color=COHORT_COLORS[cohort],
            label=COHORT_DISPLAY_LABELS[cohort],
        )
    axes[1, 2].set_xscale("log", base=2)
    axes[1, 2].set_xticks((1, 2, 4, 8, 15))
    axes[1, 2].set_xticklabels(("1", "2", "4", "8", "15"))
    axes[1, 2].set(
        xlabel="preferred temporal frequency (Hz)",
        ylabel="session mean log10 modulation index",
        title="Protocol dependence by temporal frequency",
    )
    axes[1, 2].grid(alpha=0.18)

    fig.suptitle(
        "V1 cross-dataset bridge: absolute metric levels are not yet calibrated",
        fontsize=15,
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "Common waveform QC; each dot is one recording session. Black bars are equal-session means with session-bootstrap 95% CIs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(
    center: pd.DataFrame,
    contrasts: pd.DataFrame,
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    def center_value(population: str, metric: str) -> float:
        row = center.loc[
            center["population"].eq(population) & center["metric"].eq(metric)
        ].iloc[0]
        return float(row["equal_session_mean"])

    def contrast_row(metric: str, allen_label: str) -> pd.Series:
        label = f"MouseV2 V1 minus {allen_label}"
        return contrasts.loc[
            contrasts["metric"].eq(metric) & contrasts["contrast"].eq(label)
        ].iloc[0]

    mod_bo = contrast_row("log10_mod_idx", "Allen Brain Observatory 1.1")
    mod_fc = contrast_row("log10_mod_idx", "Allen Functional Connectivity")
    f1_bo = contrast_row("log10_f1_f0", "Allen Brain Observatory 1.1")
    tau_bo = contrast_row("timescale_valid_ms", "Allen Brain Observatory 1.1")
    pooled_coverage = coverage.groupby("cohort")[["common_qc_units", "valid_timescale_units"]].sum()
    raw_summary_path = ALLEN_RAW_BRIDGE / "session_summary.csv"
    timescale_center_path = TIMESCALE_TRIAL_BRIDGE / "analysis_centers.csv"
    phase_center_path = PHASE_BRIDGE / "center_summary.csv"
    start_phase_center_path = START_PHASE_BRIDGE / "analysis_centers.csv"
    shared_phase_center_path = SHARED_PHASE_BRIDGE / "shared_phase_center_summary.csv"
    behavior_center_path = SHARED_PHASE_BRIDGE / "behavior_center_summary.csv"
    corrected_welch_center_path = CORRECTED_WELCH_BRIDGE / "analysis_centers.csv"
    corrected_welch_test_path = CORRECTED_WELCH_BRIDGE / "paired_session_tests.csv"
    has_raw_bridge = raw_summary_path.is_file()
    has_timescale_bridge = timescale_center_path.is_file()
    has_phase_bridge = phase_center_path.is_file()
    has_start_phase_bridge = start_phase_center_path.is_file()
    has_shared_phase_bridge = (
        shared_phase_center_path.is_file() and behavior_center_path.is_file()
    )
    has_corrected_welch_bridge = (
        corrected_welch_center_path.is_file() and corrected_welch_test_path.is_file()
    )
    if has_raw_bridge:
        raw_summary = pd.read_csv(raw_summary_path)

        def raw_center(cohort: str, metric: str) -> float:
            values = raw_summary.loc[
                raw_summary["cohort"].eq(cohort)
                & raw_summary["selection_role"].eq("representative")
                & raw_summary["view"].eq("common_1s_15trials")
                & raw_summary["metric"].eq(metric),
                "mean_log10",
            ]
            return float(values.mean())

        mouse_common_mod = float(
            raw_summary.loc[
                raw_summary["cohort"].eq("MouseV2 V1")
                & raw_summary["metric"].eq("mod_idx_dg"),
                "mean_log10",
            ].mean()
        )
        mouse_common_f1 = float(
            raw_summary.loc[
                raw_summary["cohort"].eq("MouseV2 V1")
                & raw_summary["metric"].eq("f1_f0_dg"),
                "mean_log10",
            ].mean()
        )
    if has_timescale_bridge:
        timescale_centers = pd.read_csv(timescale_center_path)
        matched_mouse_timescale = float(
            timescale_centers.loc[
                timescale_centers["view"].eq("mouse_matched_150"),
                "equal_session_mean_timescale_ms",
            ].mean()
        )
    if has_phase_bridge:
        phase_centers = pd.read_csv(phase_center_path)

        def phase_value(cohort: str, metric: str) -> float:
            return float(
                phase_centers.loc[
                    phase_centers["cohort"].eq(cohort)
                    & phase_centers["metric"].eq(metric),
                    "primary_equal_session_mean",
                ].iloc[0]
            )
    if has_start_phase_bridge:
        start_phase_centers = pd.read_csv(start_phase_center_path)

        def start_phase_value(view: str, scope: str = "all_common_qc_units") -> float:
            return float(
                start_phase_centers.loc[
                    start_phase_centers["cohort"].eq("MouseV2 V1")
                    & start_phase_centers["view"].eq(view)
                    & start_phase_centers["scope"].eq(scope),
                    "equal_session_weighted_phase_coherence",
                ].iloc[0]
            )
    if has_shared_phase_bridge:
        shared_phase_center = pd.read_csv(shared_phase_center_path).iloc[0]
        behavior_centers = pd.read_csv(behavior_center_path)

        def shared_phase_value(column: str) -> float:
            return float(shared_phase_center[column])

        def behavior_value(metric: str, column: str) -> float:
            return float(
                behavior_centers.loc[
                    behavior_centers["behavior_metric"].eq(metric)
                    & behavior_centers["analysis"].eq("time_residualized"),
                    column,
                ].iloc[0]
            )
    if has_corrected_welch_bridge:
        corrected_welch_centers = pd.read_csv(corrected_welch_center_path)
        corrected_welch_tests = pd.read_csv(corrected_welch_test_path)

        def corrected_welch_value(
            view: str,
            scope: str = "all_common_qc_units",
            cohort: str = "MouseV2 V1",
        ) -> float:
            return float(
                corrected_welch_centers.loc[
                    corrected_welch_centers["cohort"].eq(cohort)
                    & corrected_welch_centers["view"].eq(view)
                    & corrected_welch_centers["scope"].eq(scope),
                    "equal_session_log10_mod_idx",
                ].iloc[0]
            )

        corrected_welch_primary_test = corrected_welch_tests.loc[
            corrected_welch_tests["scope"].eq("all_common_qc_units")
            & corrected_welch_tests["comparison"].eq(
                "source_corrected_minus_raw"
            )
        ].iloc[0]

    lines = [
        "# V1 cross-dataset bridge checkpoint",
        "",
        "## Outcome: claim gate closed",
        "",
        "The known grating and flash protocol differences have now been matched in raw",
        "diagnostic bridges. Unreset MouseV2 grating start phase explains a material part",
        "of the coherence loss. Carrying the source-phase correction through the unchanged",
        "Welch estimator substantially narrows, but does not close, the representative Allen",
        "gap; the residual cross-probe state does not provide an additional repair.",
        "The absolute Allen V1 point is therefore not a calibrated MouseV2 baseline.",
        "The defensible current",
        "claim is within-dataset only; no offset or mean-matching correction is applied.",
        "",
        "## What the current tables show",
        "",
        f"- Equal-session log10 modulation index is {center_value('MouseV2 V1', 'log10_mod_idx'):.3f} in MouseV2, "
        f"{center_value('Allen V1 — Brain Observatory 1.1', 'log10_mod_idx'):.3f} in Allen Brain Observatory, and "
        f"{center_value('Allen V1 — Functional Connectivity', 'log10_mod_idx'):.3f} in Allen Functional Connectivity.",
        f"- MouseV2 minus Allen Brain Observatory is {mod_bo['difference_equal_session_means']:+.3f} "
        f"(session-bootstrap 95% CI {mod_bo['bootstrap_95ci_low']:+.3f} to {mod_bo['bootstrap_95ci_high']:+.3f}); "
        f"MouseV2 minus Allen Functional Connectivity is {mod_fc['difference_equal_session_means']:+.3f}.",
        f"- For log10 F1/F0, MouseV2 minus Allen Brain Observatory is instead {f1_bo['difference_equal_session_means']:+.3f} "
        f"({f1_bo['bootstrap_95ci_low']:+.3f} to {f1_bo['bootstrap_95ci_high']:+.3f}). The large downward offset is therefore specific to `mod_idx_dg`, not a general loss of grating modulation.",
        f"- Valid pooled-flash timescale is {center_value('MouseV2 V1', 'timescale_valid_ms'):.2f} ms in MouseV2 and "
        f"{center_value('Allen V1 — Brain Observatory 1.1', 'timescale_valid_ms'):.2f} ms in Allen Brain Observatory, "
        f"a {tau_bo['difference_equal_session_means']:+.2f}-ms session-level difference "
        f"({tau_bo['bootstrap_95ci_low']:+.2f} to {tau_bo['bootstrap_95ci_high']:+.2f} ms).",
        f"- Timescale validity retains {int(pooled_coverage.loc['MouseV2 V1', 'valid_timescale_units']):,}/"
        f"{int(pooled_coverage.loc['MouseV2 V1', 'common_qc_units']):,} MouseV2 common-QC units and "
        f"{int(pooled_coverage.loc['Allen Brain Observatory 1.1', 'valid_timescale_units'] + pooled_coverage.loc['Allen Functional Connectivity', 'valid_timescale_units']):,}/"
        f"{int(pooled_coverage.loc['Allen Brain Observatory 1.1', 'common_qc_units'] + pooled_coverage.loc['Allen Functional Connectivity', 'common_qc_units']):,} Allen V1 common-QC units.",
        "- The completed all-session MouseV2 raw bridge restricted preference to Allen's SF = 0.04 cycles/degree. It changed the equal-site mean log10 modulation index by only +0.009 (site range -0.019 to +0.036), so varying SF/preference selection does not explain the dataset offset. The corresponding F1/F0 change was +0.061.",
        *(
            [
                f"- In representative checksum-verified Allen sessions, the common 1-s/15-trial estimator leaves MouseV2 modulation {mouse_common_mod - raw_center('Allen Brain Observatory 1.1', 'mod_idx_dg'):+.3f} log10 below Brain Observatory and {mouse_common_mod - raw_center('Allen Functional Connectivity', 'mod_idx_dg'):+.3f} below Functional Connectivity.",
                f"- The corresponding harmonized F1/F0 differences are only {mouse_common_f1 - raw_center('Allen Brain Observatory 1.1', 'f1_f0_dg'):+.3f} and {mouse_common_f1 - raw_center('Allen Functional Connectivity', 'f1_f0_dg'):+.3f} log10. The residual modulation gap is therefore metric-specific, not explained by the known spectral-window mismatch.",
            ]
            if has_raw_bridge
            else []
        ),
        *(
            [
                f"- The phase decomposition locates the discrepancy after trial averaging: weighted phase coherence is {phase_value('MouseV2 V1', 'weighted_phase_coherence'):.3f} in MouseV2 versus {phase_value('Allen Brain Observatory 1.1', 'weighted_phase_coherence'):.3f} in representative Allen BO and {phase_value('Allen Functional Connectivity', 'weighted_phase_coherence'):.3f} in representative Allen FC.",
                f"- MouseV2 mean single-trial F1 amplitude is not smaller ({phase_value('MouseV2 V1', 'log10_mean_trial_f1_hz'):+.3f} log10 versus {phase_value('Allen Brain Observatory 1.1', 'log10_mean_trial_f1_hz'):+.3f} and {phase_value('Allen Functional Connectivity', 'log10_mean_trial_f1_hz'):+.3f}), but it loses {phase_value('MouseV2 V1', 'log10_coherent_f1_hz') - phase_value('MouseV2 V1', 'log10_mean_trial_f1_hz'):+.3f} log10 during coherent averaging versus {phase_value('Allen Brain Observatory 1.1', 'log10_coherent_f1_hz') - phase_value('Allen Brain Observatory 1.1', 'log10_mean_trial_f1_hz'):+.3f} and {phase_value('Allen Functional Connectivity', 'log10_coherent_f1_hz') - phase_value('Allen Functional Connectivity', 'log10_mean_trial_f1_hz'):+.3f} in Allen. This supports phase/latency variability as the proximate cause of the Welch gap."
            ]
            if has_phase_bridge
            else []
        ),
        *(
            [
                f"- The frozen acquisition source identifies one cause: MouseV2 grating phase advances from the absolute block frame and is not reset at onset. Reconstructing the 135-frame schedule raises equal-session coherence from {start_phase_value('mouse_raw'):.3f} to {start_phase_value('mouse_source_phase_corrected'):.3f}; the affected 1/2/15-Hz units rise from {start_phase_value('mouse_raw', 'source_phase_varies_1_2_15_hz'):.3f} to {start_phase_value('mouse_source_phase_corrected', 'source_phase_varies_1_2_15_hz'):.3f}, with all eight sessions moving in the predicted direction.",
                f"- This source-defined adjustment is partial: residual coherence gaps are {start_phase_value('mouse_source_phase_corrected') - phase_value('Allen Brain Observatory 1.1', 'weighted_phase_coherence'):+.3f} versus representative Allen BO and {start_phase_value('mouse_source_phase_corrected') - phase_value('Allen Functional Connectivity', 'weighted_phase_coherence'):+.3f} versus representative Allen FC. The predicted phase-stable 4/8-Hz controls are unchanged."
            ]
            if has_start_phase_bridge and has_phase_bridge
            else []
        ),
        *(
            [
                f"- Source-corrected residual phase has weak matched-trial structure across separate probes: equal-session alignment is {shared_phase_value('equal_session_cross_probe_alignment'):.3f} versus {shared_phase_value('permutation_cross_probe_alignment_mean'):.3f} after independent within-condition trial shuffling (aggregate p={shared_phase_value('cross_probe_alignment_p'):.4f}); {int(shared_phase_value('sessions_alignment_p_le_0_05'))}/8 sessions individually exceed the one-sided 0.05 threshold.",
                f"- That signal does not repair the gap. Correcting each unit only with the other probes changes coherence from {shared_phase_value('equal_session_source_corrected_coherence'):.3f} to {shared_phase_value('equal_session_cross_probe_adjusted_coherence'):.3f}, versus {shared_phase_value('permutation_adjusted_coherence_mean'):.3f} for shuffled correspondence. A simple probe-global residual timing shift is therefore unlikely to be the main remaining cause.",
                f"- After orientation/TF stratification, a 50% valid-eye-coverage requirement, and linear/quadratic block-time control, residual population phase covaries descriptively with running ({behavior_value('running_abs_median_stim', 'equal_session_association'):.3f} versus {behavior_value('running_abs_median_stim', 'permutation_mean'):.3f}, aggregate p={behavior_value('running_abs_median_stim', 'permutation_p'):.3f}) and pupil x/y ({behavior_value('pupil_x_median_stim', 'equal_session_association'):.3f}/{behavior_value('pupil_y_median_stim', 'equal_session_association'):.3f} versus {behavior_value('pupil_x_median_stim', 'permutation_mean'):.3f}/{behavior_value('pupil_y_median_stim', 'permutation_mean'):.3f}, p={behavior_value('pupil_x_median_stim', 'permutation_p'):.3f}/{behavior_value('pupil_y_median_stim', 'permutation_p'):.3f}); pupil area is not supported (p={behavior_value('log_pupil_area_median_stim', 'permutation_p'):.3f}). These associations do not establish a behavioral cause of the dataset offset.",
            ]
            if has_shared_phase_bridge
            else []
        ),
        *(
            [
                f"- Passing the source-defined carrier correction through the unchanged released Welch estimator raises the MouseV2 equal-session log10 modulation index from {corrected_welch_value('raw'):+.3f} to {corrected_welch_value('source_corrected'):+.3f}. All {int(corrected_welch_primary_test['positive_sessions'])}/{int(corrected_welch_primary_test['sessions'])} sessions move upward (exact two-sided sign-test p={corrected_welch_primary_test['exact_sign_test_two_sided_p']:.4f}), versus {corrected_welch_value('phase_permutation'):+.3f} under phase permutation and {corrected_welch_value('opposite_sign'):+.3f} for the opposite-sign rotation.",
                f"- The effect is protocol-specific: affected 1/2/15-Hz units move from {corrected_welch_value('raw', 'source_phase_varies_1_2_15_hz'):+.3f} to {corrected_welch_value('source_corrected', 'source_phase_varies_1_2_15_hz'):+.3f}, while the predicted 4/8-Hz negative control remains {corrected_welch_value('source_corrected', 'source_phase_stable_4_8_hz'):+.3f}. Corrected MouseV2 remains {corrected_welch_value('source_corrected') - corrected_welch_value('allen_representative_common_1s_15trials', cohort='Allen Brain Observatory 1.1'):+.3f} below representative Allen BO and {corrected_welch_value('source_corrected') - corrected_welch_value('allen_representative_common_1s_15trials', cohort='Allen Functional Connectivity'):+.3f} below representative Allen FC, so this is a mechanism diagnostic rather than a replacement released field.",
            ]
            if has_corrected_welch_bridge
            else []
        ),
        *(
            [
                f"- Matching MouseV2 from 300 to Allen's balanced 150 flashes lowers its timescale center to {matched_mouse_timescale:.2f} ms, explaining {center_value('MouseV2 V1', 'timescale_valid_ms') - matched_mouse_timescale:.2f} ms of the original offset and leaving {matched_mouse_timescale - center_value('Allen V1 — Brain Observatory 1.1', 'timescale_valid_ms'):+.2f} ms versus Brain Observatory."
            ]
            if has_timescale_bridge
            else []
        ),
        "",
        "## Identified non-equivalences",
        "",
        "1. Allen drifting-grating PSTHs use a 2-s analysis window; MouseV2 gratings last 1 s.",
        "2. With `nperseg=1024`, Welch uses 1,024-sample segments for Allen but is reduced to 1,000 samples for MouseV2.",
        "3. The released `np.searchsorted` lookup lands exactly on 1/2/4/8/15 Hz for the 1-s Mouse grid but on the next higher Welch bins for Allen (for example, 2 Hz maps to 2.930 Hz rather than the nearest 1.953-Hz bin). Thus identical source code does not define an identical spectral measurement.",
        "4. Allen V1 combines Brain Observatory and Functional Connectivity stimulus sets. The latter is visibly shifted in `mod_idx_dg`, while its F1/F0 center is nearly unchanged.",
        "5. Allen uses fixed grating spatial frequency; MouseV2 varies spatial frequency and selects a preferred orientation x TF x SF condition.",
        "6. The available MouseV2 `firing_rate_dg` is preferred-condition rate, whereas the released Allen field is an overall block firing rate. It cannot yet be used as a matched covariate or cross-dataset filter.",
        "7. Flash polarity and trial count are now matched (75 bright + 75 dark). Trial matching changes both the MouseV2 center and the fraction passing the spike-count/error gate; layer/RF population support remains unmatched.",
        "8. MouseV2 grating phase is advanced as `TF * current_frame / fps` without a presentation-onset reset. The 1-s stimulus + 1.25-s blank schedule therefore mixes starting phases at 1, 2, and 15 Hz, whereas 4 and 8 Hz remain phase stable.",
        "9. `mod_idx_dg` is phase-coherence sensitive because it analyzes the trial-averaged PSTH. Source-derived start phase explains part, but not all, of MouseV2's lower coherence. A target-component source-phase correction materially narrows the Welch-index gap with TF-specific and sign/permutation controls, but does not close it; unmatched population support and other dataset differences remain.",
        "",
        "The included Monte Carlo table demonstrates estimator sensitivity only; it is not a biological correction and cannot identify how much of the observed offset is protocol versus population.",
        "",
        "## Acceptance analysis before the main result is used",
        "",
        "1. Completed: verified raw Allen NWBs reproduce common-QC released grating metrics, and all eight MouseV2 sessions are recomputed at SF = 0.04.",
        "2. Completed as a representative-session diagnostic: first 1 s, 15 trials, shared support, fixed frequency grid, and exact target frequency. Expand across Allen sessions before estimating a population dataset coefficient.",
        "3. Completed: retain released `mod_idx_dg` as a historical sensitivity and harmonized F1/F0 as a co-primary diagnostic; the two metrics lead to different cross-dataset conclusions.",
        "4. Completed: MouseV2 flash polarity/trial support is matched to Allen with repeated trial draws and explicit selection-flow reporting.",
        "5. Completed as a representative-session mechanism diagnostic: decompose per-trial F1 amplitude, coherent amplitude, phase coherence, and target/off-target PSD. Phase inconsistency explains why F1/F0 and Welch modulation lead to different cross-dataset conclusions.",
        "6. Completed: reconstruct MouseV2 start phase directly from the frozen acquisition source and chronological presentation id, with TF-specific and phase-permutation controls. It materially raises carrier coherence but leaves a residual Allen gap.",
        "7. Completed: pass the source-phase carrier correction through the unchanged released Welch estimator, preserving the mean and every non-carrier PSTH component. It raises all eight session centers and materially narrows the representative Allen gap, with TF-specific, opposite-sign, and phase-permutation controls.",
        "8. Completed: test residual presentation-level phase across simultaneously recorded probes and against running and eye state, using leave-one-trial-out phase estimates, other-probe prediction, within-condition permutations, and block-time sensitivity. Shared/behavioral structure exists, but the other-probe correction does not restore coherence.",
        "9. Remaining: expand the raw Allen diagnostic across sessions and match homologous RF/layer/population support. Do not mean-match response metrics.",
        "10. Current decision: the pass criterion still fails for absolute modulation index. Restrict the claim to within-dataset results and treat the Allen V1 point as context unless a multi-session/common-population bridge changes this conclusion.",
        "",
        "## Protocol provenance",
        "",
        "- MouseV2 protocol snapshot: `config/mousev2_stimulus_manifest.json` (1-s gratings, 15 repeats, five SFs; 300 flashes).",
        "- Released implementation: AllenSDK `brain_observatory/ecephys/stimulus_analysis/drifting_gratings.py` (`trial_duration=2.0`, `nperseg=1024`, and `np.searchsorted`).",
        "- Allen Visual Coding documentation: https://allenswdb.github.io/physiology/ephys/visual-coding/vcnp-stimulus.html",
        "- AllenSDK dataset documentation: https://allensdk.readthedocs.io/en/stable/visual_coding_neuropixels.html",
        "",
        "## Outputs",
        "",
        "- `Figure_v1_dataset_bridge.png`: session-level diagnostic figure.",
        "- `center_summary.csv`: pooled-unit and equal-session centers.",
        "- `session_metric_summary.csv`: one row per session and metric.",
        "- `dataset_contrasts.csv`: session-bootstrap dataset offsets.",
        "- `welch_frequency_lookup.csv`: duration-dependent spectral-bin audit.",
        "- `welch_protocol_sensitivity.csv`: controlled estimator simulation.",
        "- `timescale_coverage.csv` and `tf_session_summary.csv`: selection and TF diagnostics.",
        "- `data/imports/mousev2_grating_common_support_v1/`: all-session raw MouseV2 SF = 0.04 bridge and diagnostic figure.",
        "- `data/imports/allen_v1_raw_bridge_v2/`: checksum-verified raw Allen reproduction and common-window diagnostic.",
        "- `data/imports/mousev2_timescale_trial_bridge_v1/`: balanced 150-flash trial sensitivity and selection flow.",
        "- `data/imports/v1_grating_phase_bridge_v1/`: single-trial amplitude, phase-coherence, TF-stratified, and adjusted diagnostics.",
        "- `data/imports/mousev2_grating_start_phase_bridge_v1/`: acquisition-source phase reconstruction, permutation controls, and residual-gap diagnostic.",
        "- `data/imports/mousev2_grating_corrected_welch_bridge_v1/`: target-component source-phase correction through the released Welch estimator, with TF/sign/permutation controls.",
        "- `data/imports/mousev2_grating_shared_phase_behavior_v1/`: cross-probe residual-phase, behavior, time-control, and permutation diagnostics.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    allen, mouse = _prepare_tables(args)
    combined = pd.concat([allen, mouse], ignore_index=True, sort=False)
    session = _session_metrics(combined)
    center = _center_summary(allen, mouse)
    contrasts = _contrasts(session, args.n_bootstrap)
    lookup = _welch_lookup()
    simulation = _protocol_sensitivity()
    coverage = _timescale_coverage(allen, mouse)
    tf_session = _tf_session_summary(allen, mouse)

    outputs = {
        "session_metric_summary.csv": session,
        "center_summary.csv": center,
        "dataset_contrasts.csv": contrasts,
        "welch_frequency_lookup.csv": lookup,
        "welch_protocol_sensitivity.csv": simulation,
        "timescale_coverage.csv": coverage,
        "tf_session_summary.csv": tf_session,
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False)

    _make_figure(
        session,
        coverage,
        tf_session,
        output_dir / "Figure_v1_dataset_bridge.png",
    )
    _report(center, contrasts, coverage, output_dir / "V1_DATASET_BRIDGE.md")
    print(f"V1 dataset bridge written to {output_dir}")


if __name__ == "__main__":
    main()
