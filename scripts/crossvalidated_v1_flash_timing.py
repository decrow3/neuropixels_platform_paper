#!/usr/bin/env python3
"""Test V1 flash timing without selecting units on latency or evaluation trials."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from aggregate_response_verified_early_v1_psths import (
    ALIGN_WINDOW, CENTERS_MS, COHORTS, COLORS, EDGES_S, FDR_ALPHA,
    MIN_EFFECT_HZ, OUT as LEGACY_OUT, SHIFT_GRID_MS, SOURCE_PREDICTED_SHIFT_MS,
    alignment_summary, bh_adjust, normalize, read_targeted_nwb, shifted,
    session_inputs,
)

OUT = LEGACY_OUT.parent / "crossvalidated_latency_independent_psths"
SELECTION_WINDOW = (CENTERS_MS >= 30) & (CENTERS_MS < 180)
BASELINE_WINDOW = (CENTERS_MS >= -100) & (CENTERS_MS < 0)


def all_unit_inputs(allen: pd.DataFrame, mouse: pd.DataFrame):
    # Reuse the fully audited path/session routing, but temporarily make every
    # common-QC unit eligible; no TTFS value enters this selection.
    allen_copy = allen.copy(); mouse_copy = mouse.copy()
    allen_copy["_unfiltered_released_ttfs"] = allen_copy["time_to_first_spike_fl"]
    mouse_copy["_unfiltered_released_ttfs"] = mouse_copy["time_to_first_spike_fl"]
    allen_copy["time_to_first_spike_fl"] = 0.0
    mouse_copy["time_to_first_spike_fl"] = 0.0
    for cohort, sid, units, path, flash_table in session_inputs(allen_copy, mouse_copy):
        units = units.copy()
        units["time_to_first_spike_fl"] = units["_unfiltered_released_ttfs"]
        yield cohort, sid, units, path, flash_table


def counts(spikes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    absolute_edges = starts[:, None] + EDGES_S[None, :]
    indices = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(indices, axis=1).astype(float)


def crossvalidated_session(cohort, sid, units, path, flash_table):
    spikes, starts, polarity_values = read_targeted_nwb(
        path, flash_table, units["source_unit_id"].astype(int).to_numpy()
    )
    audits, heldout = [], {}
    for _, unit in units.iterrows():
        uid, source = int(unit["analysis_unit_id"]), int(unit["source_unit_id"])
        for polarity_name, polarity_value in (("bright", 1.0), ("dark", -1.0)):
            polarity_indices = np.flatnonzero(polarity_values == polarity_value)
            selection_indices = polarity_indices[::2]
            evaluation_indices = polarity_indices[1::2]
            selection_rates = counts(spikes[source], starts[selection_indices]) / np.diff(EDGES_S)
            baseline_trial = selection_rates[:, BASELINE_WINDOW].mean(axis=1)
            response_trial = selection_rates[:, SELECTION_WINDOW].mean(axis=1)
            effect = float(np.mean(response_trial - baseline_trial))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = float(wilcoxon(response_trial, baseline_trial, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            evaluation_rates = counts(spikes[source], starts[evaluation_indices]) / np.diff(EDGES_S)
            evaluation_mean = evaluation_rates.mean(axis=0)
            evaluation_delta = evaluation_mean - float(evaluation_mean[BASELINE_WINDOW].mean())
            audits.append({
                "cohort": cohort, "session_id": sid, "unit_id": uid,
                "source_unit_id": source, "polarity": polarity_name,
                "selection_trials": len(selection_indices), "evaluation_trials": len(evaluation_indices),
                "selection_response_minus_baseline_hz": effect, "selection_wilcoxon_p": p,
            })
            heldout[(uid, polarity_name)] = evaluation_delta
    audit = pd.DataFrame(audits)
    audit["selection_fdr_q"] = np.nan
    for polarity_name, idx in audit.groupby("polarity").groups.items():
        audit.loc[idx, "selection_fdr_q"] = bh_adjust(audit.loc[idx, "selection_wilcoxon_p"].to_numpy())
    audit["selected_positive_responder"] = (
        audit["selection_response_minus_baseline_hz"].ge(MIN_EFFECT_HZ)
        & audit["selection_fdr_q"].lt(FDR_ALPHA)
    )
    trace_rows = []
    for polarity_name in ("bright", "dark"):
        selected = audit.loc[
            audit.polarity.eq(polarity_name) & audit.selected_positive_responder, "unit_id"
        ].astype(int).to_numpy()
        if len(selected) == 0:
            continue
        session_mean = np.stack([heldout[(uid, polarity_name)] for uid in selected]).mean(axis=0)
        for time, value in zip(CENTERS_MS, session_mean):
            trace_rows.append({
                "cohort": cohort, "session_id": sid, "polarity": polarity_name,
                "selected_units": len(selected), "time_ms": time,
                "heldout_session_mean_delta_rate_hz": value,
            })
    return audit, pd.DataFrame(trace_rows)


def cohort_curves(traces: pd.DataFrame) -> pd.DataFrame:
    return traces.groupby(["cohort", "polarity", "time_ms"], as_index=False).agg(
        mean_delta_rate_hz=("heldout_session_mean_delta_rate_hz", "mean"),
        session_sem_hz=("heldout_session_mean_delta_rate_hz", "sem"),
        sessions=("session_id", "nunique"),
    )


def render(curves, grid, alignment, status):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, polarity in enumerate(("bright", "dark")):
        for cohort in COHORTS:
            sub = curves.loc[(curves.cohort == cohort) & (curves.polarity == polarity)]
            y = sub.mean_delta_rate_hz.to_numpy(); sem = sub.session_sem_hz.to_numpy()
            n = int(sub.sessions.iloc[0])
            axes[0, col].plot(CENTERS_MS, y, color=COLORS[cohort], lw=2.2, label=f"{cohort} ({n} sessions)")
            axes[0, col].fill_between(CENTERS_MS, y-sem, y+sem, color=COLORS[cohort], alpha=.12, lw=0)
        axes[0, col].set_title(f"{polarity.capitalize()}: held-out trial PSTHs")
        axes[0, col].legend(frameon=False, fontsize=8)
        for cohort in COHORTS[:2]:
            y = curves.loc[(curves.cohort == cohort) & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
            axes[1, col].plot(CENTERS_MS, normalize(y), color=COLORS[cohort], lw=2.2, label=cohort)
        mouse = curves.loc[(curves.cohort == "MouseV2") & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
        axes[1, col].plot(CENTERS_MS, normalize(mouse), color=COLORS["MouseV2"], lw=1.8, alpha=.55, label="MouseV2 raw")
        axes[1, col].plot(CENTERS_MS, shifted(normalize(mouse), SOURCE_PREDICTED_SHIFT_MS), color=COLORS["MouseV2"], lw=2.5, ls="--", label="MouseV2 −8.34 ms")
        axes[1, col].set_title("Latency-independent selection; normalized timing")
        axes[1, col].set_xlabel("Time from recorded NWB flash onset (ms)")
        axes[1, col].legend(frameon=False, fontsize=8)
        for row in range(2):
            axes[row, col].axvline(0, color="k", ls=":", lw=1)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Held-out baseline-subtracted rate (Hz)")
    axes[1, 0].set_ylabel("Peak-normalized held-out response")
    for polarity, ls in (("bright", "-"), ("dark", "--")):
        for cohort in COHORTS[:2]:
            sub = grid.loc[(grid.allen_cohort == cohort) & (grid.polarity == polarity)]
            axes[0, 2].plot(sub.mouse_shift_ms, sub.normalized_rmse, color=COLORS[cohort], ls=ls, lw=2, label=f"{cohort}, {polarity}")
    axes[0, 2].axvline(SOURCE_PREDICTED_SHIFT_MS, color=COLORS["MouseV2"], ls="--", lw=2, label="source prediction")
    axes[0, 2].set(xlabel="Shift applied to MouseV2 (ms)", ylabel="Normalized waveform RMSE", title="Held-out waveform shift scan")
    axes[0, 2].legend(frameon=False, fontsize=7); axes[0, 2].spines[["top", "right"]].set_visible(False)
    axes[1, 2].axis("off")
    lines = ["No latency/TTFS filter", "Odd trials: select amplitude", "Even trials: estimate timing", ""]
    for _, row in alignment.iterrows():
        lines.append(f"{row.allen_cohort.replace('Allen ', '')}, {row.polarity}, {row['view']}:")
        lines.append(f"  shift {row.mouse_shift_ms:+.2f} ms; RMSE {row.normalized_rmse:.3f}; r {row.correlation:.3f}")
    axes[1, 2].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.2)
    fig.suptitle("Cross-validated V1 flash timing: unit selection cannot guarantee matched latency")
    fig.savefig(OUT / "crossvalidated_latency_independent_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "crossvalidated_latency_independent_alignment.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen.area_coarse.eq("V1")].copy()
    mouse = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    audits, traces, status = [], [], []
    for cohort, sid, units, path, flash_table in all_unit_inputs(allen, mouse):
        print(f"[{cohort} {sid}] {len(units)} common-QC V1 units", flush=True)
        try:
            audit, trace = crossvalidated_session(cohort, sid, units, path, flash_table)
            audits.append(audit)
            if not trace.empty: traces.append(trace)
            status.append({"cohort": cohort, "session_id": sid, "status": "ok", "eligible_units": len(units), "selected_units": audit.loc[audit.selected_positive_responder, "unit_id"].nunique(), "error": ""})
        except Exception as exc:
            status.append({"cohort": cohort, "session_id": sid, "status": "error", "eligible_units": len(units), "selected_units": 0, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
    audit = pd.concat(audits, ignore_index=True); trace = pd.concat(traces, ignore_index=True)
    curves = cohort_curves(trace)
    grid, alignment = alignment_summary(curves)
    status = pd.DataFrame(status)
    audit.to_csv(OUT / "crossvalidated_unit_selection_audit.csv", index=False)
    trace.to_csv(OUT / "heldout_session_mean_psth_traces.csv", index=False)
    curves.to_csv(OUT / "heldout_session_balanced_cohort_curves.csv", index=False)
    grid.to_csv(OUT / "heldout_waveform_shift_scan.csv", index=False)
    alignment.to_csv(OUT / "heldout_alignment_summary.csv", index=False)
    status.to_csv(OUT / "session_processing_status.csv", index=False)
    render(curves, grid, alignment, status)


if __name__ == "__main__":
    main()
