#!/usr/bin/env python3
"""Verify early V1 flash responders against baseline and plot polarity PSTHs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from common.flashes import prepare_flash_presentations
from generate_retinotopic_csvs import read_nwb_tables
from plot_early_v1_flash_psths import CASES, EDGES_S, CENTERS_MS, trial_counts

OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/response_verified_early_psths"
MAX_ONSET_MS = 60.0
RELEASED_CANDIDATE_MS = 60.0
MIN_EFFECT_HZ = 2.0
FDR_ALPHA = 0.01
MAX_DISPLAY_UNITS = 10_000


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.minimum(q_ranked, 1.0)
    return q


def candidate_units(case: dict, allen: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    if case["cohort"].startswith("Allen"):
        units = allen.loc[allen["ecephys_session_id"].eq(case["session_id"])].copy()
        units["source_unit_id"] = units["ecephys_unit_id"].astype(int)
        units["analysis_unit_id"] = units["ecephys_unit_id"].astype(int)
    else:
        units = mouse.loc[mouse["session_num"].eq(case["session_id"])].copy()
        units["source_unit_id"] = units["unit_id"].astype(int) - int(case["session_id"]) * 1_000_000
        units["analysis_unit_id"] = units["unit_id"].astype(int)
    units["released_ttfs_ms"] = 1000 * pd.to_numeric(units["time_to_first_spike_fl"], errors="coerce")
    return units.loc[units["released_ttfs_ms"].lt(RELEASED_CANDIDATE_MS)].sort_values(
        ["released_ttfs_ms", "analysis_unit_id"], kind="stable"
    )


def verify_case(case: dict, units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted = read_nwb_tables(str(case["path"]))
    flashes = prepare_flash_presentations(extracted.intervals_tables[case["flash_table"]])
    nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
    row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
    baseline_bins = (CENTERS_MS >= -100) & (CENTERS_MS < 0)
    response_bins = (CENTERS_MS >= 30) & (CENTERS_MS < 80)
    onset_bins = (CENTERS_MS >= 20) & (CENTERS_MS < MAX_ONSET_MS)
    audit_rows, stored = [], {}
    for _, unit in units.iterrows():
        uid = int(unit["analysis_unit_id"])
        source = int(unit["source_unit_id"])
        spikes = extracted.spikes_by_unit[row_by_id[source]]
        for polarity in ("bright", "dark"):
            mask = flashes["flash_polarity"].eq(polarity).to_numpy()
            counts = trial_counts(spikes, flashes.loc[mask, "start_time"].to_numpy(float))
            rates = counts / np.diff(EDGES_S)
            baseline_trial = rates[:, baseline_bins].mean(axis=1)
            response_trial = rates[:, response_bins].mean(axis=1)
            effect = float(np.mean(response_trial - baseline_trial))
            try:
                p = float(wilcoxon(response_trial, baseline_trial, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            mean_rate = rates.mean(axis=0)
            baseline_mean = float(mean_rate[baseline_bins].mean())
            delta = mean_rate - baseline_mean
            smooth = gaussian_filter1d(delta, 1.0)
            baseline_sd = float(np.std(smooth[baseline_bins], ddof=1))
            threshold = max(2.0, 3.0 * baseline_sd)
            above = smooth[onset_bins] > threshold
            onset_positions = np.flatnonzero(above[:-1] & above[1:])
            onset = float(CENTERS_MS[onset_bins][onset_positions[0]]) if len(onset_positions) else np.nan
            audit_rows.append({
                "cohort": case["cohort"], "session_id": case["session_id"],
                "unit_id": uid, "source_unit_id": source,
                "released_ttfs_ms": float(unit["released_ttfs_ms"]), "polarity": polarity,
                "trials": int(mask.sum()), "response_minus_baseline_hz": effect,
                "wilcoxon_p": p, "psth_threshold_hz": threshold, "verified_onset_ms": onset,
            })
            stored[(uid, polarity)] = (mean_rate, delta)
    audit = pd.DataFrame(audit_rows)
    audit["fdr_q"] = np.nan
    for polarity, index in audit.groupby("polarity").groups.items():
        audit.loc[index, "fdr_q"] = bh_adjust(audit.loc[index, "wilcoxon_p"].to_numpy())
    audit["response_significant"] = (
        audit["response_minus_baseline_hz"].ge(MIN_EFFECT_HZ) & audit["fdr_q"].lt(FDR_ALPHA)
    )
    audit["onset_early"] = audit["verified_onset_ms"].lt(MAX_ONSET_MS)
    audit["verified_early_responder"] = audit["response_significant"] & audit["onset_early"]

    verified_ids = (
        audit.loc[audit["verified_early_responder"]]
        .groupby("unit_id")["verified_onset_ms"].min().sort_values().head(MAX_DISPLAY_UNITS)
    )
    trace_rows = []
    for rank, (uid, earliest_onset) in enumerate(verified_ids.items(), start=1):
        for polarity in ("bright", "dark"):
            mean_rate, delta = stored[(uid, polarity)]
            for time, raw, change in zip(CENTERS_MS, mean_rate, delta):
                trace_rows.append({
                    "cohort": case["cohort"], "session_id": case["session_id"],
                    "unit_id": uid, "selection_rank": rank, "earliest_verified_onset_ms": earliest_onset,
                    "polarity": polarity, "time_ms": time, "rate_hz": raw,
                    "baseline_subtracted_rate_hz": change,
                })
    return audit, pd.DataFrame(trace_rows)


def bootstrap_mean(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(matrix), size=(2000, len(matrix)))
    boot = matrix[ix].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def render(traces: pd.DataFrame, audit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex=True, sharey="row", constrained_layout=True)
    styles = {"bright": ("-", 1.0), "dark": ("--", .85)}
    for col, case in enumerate(CASES):
        cohort = case["cohort"]
        case_trace = traces.loc[traces["cohort"].eq(cohort)]
        selected = case_trace["unit_id"].nunique()
        candidates = audit.loc[audit["cohort"].eq(cohort), "unit_id"].nunique()
        for polarity in ("bright", "dark"):
            subset = case_trace.loc[case_trace["polarity"].eq(polarity)]
            if subset.empty:
                continue
            matrix = subset.pivot(index="selection_rank", columns="time_ms", values="baseline_subtracted_rate_hz").to_numpy()
            mean, low, high = bootstrap_mean(matrix, 20260825 + col * 10 + (polarity == "dark"))
            ls, alpha = styles[polarity]
            axes[0, col].fill_between(CENTERS_MS, low, high, color=case["color"], alpha=.13 if polarity == "dark" else .2, lw=0)
            axes[0, col].plot(CENTERS_MS, mean, color=case["color"], ls=ls, alpha=alpha, lw=2.4, label=polarity.capitalize())
            for _, unit_trace in subset.groupby("unit_id"):
                axes[1, col].plot(CENTERS_MS, unit_trace["baseline_subtracted_rate_hz"], color=case["color"], ls=ls, alpha=.18, lw=.75)
        for row in range(2):
            axes[row, col].axhline(0, color="0.45", lw=.8)
            axes[row, col].axvline(0, color="k", lw=1, ls=":")
            axes[row, col].axvline(60, color="0.25", lw=1, ls="--")
            axes[row, col].axvspan(0, 200, color="0.85", alpha=.22, zorder=-2)
            axes[row, col].spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(f"{cohort}\nsession {case['session_id']}: {selected}/{candidates} verified")
        axes[0, col].legend(frameon=False, fontsize=9)
        axes[1, col].set_xlabel("Time from NWB flash onset (ms)")
    axes[0, 0].set_ylabel("Verified-unit mean\nbaseline-subtracted rate (Hz)")
    axes[1, 0].set_ylabel("Individual verified-unit\nPSTHs (Hz)")
    fig.suptitle("Response-verified early V1 neurons: positive flash response and onset < 60 ms")
    fig.savefig(OUT / "response_verified_early_v1_psths.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "response_verified_early_v1_psths.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].eq("V1")].copy()
    mouse = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    audits, traces = [], []
    for case in CASES:
        audit, trace = verify_case(case, candidate_units(case, allen, mouse))
        audits.append(audit); traces.append(trace)
    audit = pd.concat(audits, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    audit.to_csv(OUT / "response_verification_audit.csv", index=False)
    trace.to_csv(OUT / "verified_unit_psth_traces.csv", index=False)
    summary = audit.groupby("cohort").agg(
        candidate_units=("unit_id", "nunique"),
        verified_unit_polarities=("verified_early_responder", "sum"),
    ).reset_index()
    verified_counts = audit.loc[audit["verified_early_responder"]].groupby("cohort")["unit_id"].nunique()
    summary["verified_units"] = summary["cohort"].map(verified_counts).fillna(0).astype(int)
    summary.to_csv(OUT / "response_verification_summary.csv", index=False)
    render(trace, audit)


if __name__ == "__main__":
    main()
