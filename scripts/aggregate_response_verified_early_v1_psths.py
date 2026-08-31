#!/usr/bin/env python3
"""Aggregate response-verified early V1 PSTHs across local MouseV2 and Allen sessions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_config, load_mousev2_units
from generate_retinotopic_csvs import _read_numeric_dset
from plot_early_v1_flash_psths import EDGES_S, CENTERS_MS
from plot_response_verified_early_v1_psths import (
    FDR_ALPHA, MAX_ONSET_MS, MIN_EFFECT_HZ, RELEASED_CANDIDATE_MS, bh_adjust,
)

OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/response_verified_early_psths_all_sessions"
ALLEN_ROOT = Path("/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions")
ALLEN_INVENTORY = ALLEN_ROOT / "session_inventory.json"
MOUSE_ROOT = Path("/media/huklaban5/Data/MouseV2/001568")
SOURCE_PREDICTED_SHIFT_MS = -8.34
SHIFT_GRID_MS = np.arange(-20.0, 5.01, 0.25)
ALIGN_WINDOW = (CENTERS_MS >= 20) & (CENTERS_MS <= 120)

COHORTS = (
    "Allen Brain Observatory", "Allen Functional Connectivity", "MouseV2",
)
COLORS = {
    "Allen Brain Observatory": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2": "#D95F02",
}


def trial_counts(spikes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    absolute_edges = starts[:, None] + EDGES_S[None, :]
    indices = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(indices, axis=1).astype(float)


def read_targeted_nwb(path: Path, flash_table: str, source_ids: np.ndarray):
    fid = h5py.h5f.open(str(path).encode(), flags=h5py.h5f.ACC_RDONLY)
    with h5py.File(path, "r") as nwb:
        ids = _read_numeric_dset(fid, "/units/id").astype(np.int64)
        index = _read_numeric_dset(fid, "/units/spike_times_index").astype(np.int64)
        id_to_row = {unit_id: row for row, unit_id in enumerate(ids)}
        spikes = {}
        # Allen and MouseV2 spike times may be stored as extended precision;
        # use the validated low-level float64 conversion rather than h5py slices.
        spike_flat = _read_numeric_dset(fid, "/units/spike_times")
        for source_id in source_ids:
            row = id_to_row[int(source_id)]
            start = 0 if row == 0 else int(index[row - 1])
            stop = int(index[row])
            spikes[int(source_id)] = spike_flat[start:stop]
        base = f"/intervals/{flash_table}"
        starts = _read_numeric_dset(fid, base + "/start_time")
        polarity = None
        for name in ("contrast", "color"):
            key = f"intervals/{flash_table}/{name}"
            if key in nwb:
                raw = nwb[key][:]
                values = pd.to_numeric(
                    pd.Series([
                        value.decode() if isinstance(value, bytes) else value
                        for value in raw
                    ]), errors="coerce"
                ).to_numpy(dtype=float)
                if set(np.unique(values)) == {-1.0, 1.0}:
                    polarity = values
                    break
        if polarity is None:
            raise ValueError(f"No -1/+1 flash polarity in {path}")
    fid.close()
    return spikes, starts, polarity


def verification_rows(
    cohort: str, session_id: int, units: pd.DataFrame, path: Path, flash_table: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spikes, starts, polarity_values = read_targeted_nwb(
        path, flash_table, units["source_unit_id"].astype(int).to_numpy()
    )
    baseline_bins = (CENTERS_MS >= -100) & (CENTERS_MS < 0)
    response_bins = (CENTERS_MS >= 30) & (CENTERS_MS < 80)
    onset_bins = (CENTERS_MS >= 20) & (CENTERS_MS < MAX_ONSET_MS)
    audit_rows, trace_store = [], {}
    for _, unit in units.iterrows():
        uid, source = int(unit["analysis_unit_id"]), int(unit["source_unit_id"])
        for polarity_name, polarity_value in (("bright", 1.0), ("dark", -1.0)):
            mask = polarity_values == polarity_value
            counts = trial_counts(spikes[source], starts[mask])
            rates = counts / np.diff(EDGES_S)
            baseline_trial = rates[:, baseline_bins].mean(axis=1)
            response_trial = rates[:, response_bins].mean(axis=1)
            effect = float(np.mean(response_trial - baseline_trial))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = float(wilcoxon(response_trial, baseline_trial, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            mean_rate = rates.mean(axis=0)
            delta = mean_rate - float(mean_rate[baseline_bins].mean())
            smooth = gaussian_filter1d(delta, 1.0)
            threshold = max(2.0, 3.0 * float(np.std(smooth[baseline_bins], ddof=1)))
            above = smooth[onset_bins] > threshold
            positions = np.flatnonzero(above[:-1] & above[1:])
            onset = float(CENTERS_MS[onset_bins][positions[0]]) if len(positions) else np.nan
            audit_rows.append({
                "cohort": cohort, "session_id": session_id, "unit_id": uid,
                "source_unit_id": source, "released_ttfs_ms": float(unit["released_ttfs_ms"]),
                "polarity": polarity_name, "trials": int(mask.sum()),
                "response_minus_baseline_hz": effect, "wilcoxon_p": p,
                "psth_threshold_hz": threshold, "verified_onset_ms": onset,
            })
            trace_store[(uid, polarity_name)] = delta
    audit = pd.DataFrame(audit_rows)
    audit["fdr_q"] = np.nan
    for polarity_name, idx in audit.groupby("polarity").groups.items():
        audit.loc[idx, "fdr_q"] = bh_adjust(audit.loc[idx, "wilcoxon_p"].to_numpy())
    audit["response_significant"] = audit["response_minus_baseline_hz"].ge(MIN_EFFECT_HZ) & audit["fdr_q"].lt(FDR_ALPHA)
    audit["onset_early"] = audit["verified_onset_ms"].lt(MAX_ONSET_MS)
    audit["verified_early_responder"] = audit["response_significant"] & audit["onset_early"]
    trace_rows = []
    for polarity_name in ("bright", "dark"):
        verified = audit.loc[
            audit["polarity"].eq(polarity_name) & audit["verified_early_responder"], "unit_id"
        ].astype(int).to_numpy()
        if len(verified) == 0:
            continue
        matrix = np.stack([trace_store[(uid, polarity_name)] for uid in verified])
        session_mean = matrix.mean(axis=0)
        for time, value in zip(CENTERS_MS, session_mean):
            trace_rows.append({
                "cohort": cohort, "session_id": session_id, "polarity": polarity_name,
                "verified_units": len(verified), "time_ms": time,
                "session_mean_delta_rate_hz": value,
            })
    return audit, pd.DataFrame(trace_rows)


def session_inputs(allen: pd.DataFrame, mouse: pd.DataFrame):
    inventory = json.loads(ALLEN_INVENTORY.read_text())
    for record in inventory:
        path = Path(record["nwb_path"])
        if not path.is_file():
            continue
        sid = int(record["ecephys_session_id"])
        units = allen.loc[allen["ecephys_session_id"].eq(sid)].copy()
        if units.empty:
            continue
        units["analysis_unit_id"] = units["ecephys_unit_id"].astype(int)
        units["source_unit_id"] = units["ecephys_unit_id"].astype(int)
        units["released_ttfs_ms"] = 1000 * pd.to_numeric(units["time_to_first_spike_fl"], errors="coerce")
        units = units.loc[units["released_ttfs_ms"].lt(RELEASED_CANDIDATE_MS)]
        if units.empty:
            continue
        cohort = "Allen Brain Observatory" if record["session_type"] == "brain_observatory_1.1" else "Allen Functional Connectivity"
        yield cohort, sid, units, path, "flashes_presentations"
    config = load_config()
    for session in config["sessions"]:
        sid = int(session["site_number"])
        units = mouse.loc[mouse["session_num"].eq(sid)].copy()
        units["analysis_unit_id"] = units["unit_id"].astype(int)
        units["source_unit_id"] = units["unit_id"].astype(int) - int(session["id_offset"])
        units["released_ttfs_ms"] = 1000 * pd.to_numeric(units["time_to_first_spike_fl"], errors="coerce")
        units = units.loc[units["released_ttfs_ms"].lt(RELEASED_CANDIDATE_MS)]
        path = MOUSE_ROOT / session["nwb_relative_path"]
        yield "MouseV2", sid, units, path, "flash_field_block_presentations"


def cohort_curves(session_traces: pd.DataFrame) -> pd.DataFrame:
    return session_traces.groupby(["cohort", "polarity", "time_ms"], as_index=False).agg(
        mean_delta_rate_hz=("session_mean_delta_rate_hz", "mean"),
        session_sem_hz=("session_mean_delta_rate_hz", "sem"),
        sessions=("session_id", "nunique"),
    )


def normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(y[ALIGN_WINDOW]))
    return y / peak if peak > 0 else y * np.nan


def shifted(y: np.ndarray, shift_ms: float) -> np.ndarray:
    # A negative shift moves the MouseV2 trace earlier on the displayed time axis.
    return np.interp(CENTERS_MS - shift_ms, CENTERS_MS, y, left=np.nan, right=np.nan)


def alignment_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polarity in ("bright", "dark"):
        mouse = curves.loc[(curves.cohort == "MouseV2") & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
        mouse_n = normalize(mouse)
        for allen_cohort in COHORTS[:2]:
            allen = curves.loc[(curves.cohort == allen_cohort) & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
            allen_n = normalize(allen)
            for shift_ms in SHIFT_GRID_MS:
                moved = shifted(mouse_n, shift_ms)
                mask = ALIGN_WINDOW & np.isfinite(moved) & np.isfinite(allen_n)
                rmse = float(np.sqrt(np.mean((moved[mask] - allen_n[mask]) ** 2)))
                corr = float(np.corrcoef(moved[mask], allen_n[mask])[0, 1])
                rows.append({"allen_cohort": allen_cohort, "polarity": polarity, "mouse_shift_ms": shift_ms, "normalized_rmse": rmse, "correlation": corr})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid.groupby(["allen_cohort", "polarity"])["normalized_rmse"].idxmin()].copy()
    best["view"] = "best_fit"
    predicted = grid.loc[np.isclose(grid["mouse_shift_ms"], SOURCE_PREDICTED_SHIFT_MS, atol=.13)].copy()
    predicted["view"] = "source_predicted"
    raw = grid.loc[np.isclose(grid["mouse_shift_ms"], 0)].copy(); raw["view"] = "raw"
    return grid, pd.concat([raw, predicted, best], ignore_index=True)


def render(curves: pd.DataFrame, grid: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, polarity in enumerate(("bright", "dark")):
        for cohort in COHORTS:
            sub = curves.loc[(curves.cohort == cohort) & (curves.polarity == polarity)]
            y = sub["mean_delta_rate_hz"].to_numpy(); sem = sub["session_sem_hz"].to_numpy()
            n = int(sub["sessions"].iloc[0])
            axes[0, col].plot(CENTERS_MS, y, color=COLORS[cohort], lw=2.2, label=f"{cohort} ({n} sessions)")
            axes[0, col].fill_between(CENTERS_MS, y-sem, y+sem, color=COLORS[cohort], alpha=.12, lw=0)
        axes[0, col].set_title(f"{polarity.capitalize()} flashes: session-balanced raw response")
        axes[0, col].legend(frameon=False, fontsize=8)
        for cohort in COHORTS[:2]:
            y = curves.loc[(curves.cohort == cohort) & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
            axes[1, col].plot(CENTERS_MS, normalize(y), color=COLORS[cohort], lw=2.2, label=cohort)
        mouse = curves.loc[(curves.cohort == "MouseV2") & (curves.polarity == polarity), "mean_delta_rate_hz"].to_numpy()
        axes[1, col].plot(CENTERS_MS, normalize(mouse), color=COLORS["MouseV2"], lw=1.8, alpha=.55, label="MouseV2 raw")
        axes[1, col].plot(CENTERS_MS, shifted(normalize(mouse), SOURCE_PREDICTED_SHIFT_MS), color=COLORS["MouseV2"], lw=2.5, ls="--", label="MouseV2 −8.34 ms")
        axes[1, col].set_title("Peak-normalized timing comparison")
        axes[1, col].set_xlabel("Time from recorded NWB flash onset (ms)")
        axes[1, col].legend(frameon=False, fontsize=8)
        for row in range(2):
            axes[row, col].axvline(0, color="k", ls=":", lw=1)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Baseline-subtracted firing rate (Hz)")
    axes[1, 0].set_ylabel("Peak-normalized response")

    for polarity, ls in (("bright", "-"), ("dark", "--")):
        for cohort in COHORTS[:2]:
            sub = grid.loc[(grid.allen_cohort == cohort) & (grid.polarity == polarity)]
            axes[0, 2].plot(sub.mouse_shift_ms, sub.normalized_rmse, color=COLORS[cohort], ls=ls, lw=2, label=f"{cohort}, {polarity}")
    axes[0, 2].axvline(SOURCE_PREDICTED_SHIFT_MS, color=COLORS["MouseV2"], ls="--", lw=2, label="source prediction −8.34 ms")
    axes[0, 2].set(xlabel="Shift applied to MouseV2 (ms)", ylabel="Normalized waveform RMSE", title="Waveform-alignment shift scan")
    axes[0, 2].legend(frameon=False, fontsize=7)
    axes[0, 2].spines[["top", "right"]].set_visible(False)

    axes[1, 2].axis("off")
    lines = ["Alignment over 20–120 ms", ""]
    for _, row in summary.iterrows():
        lines.append(f"{row['allen_cohort'].replace('Allen ', '')}, {row['polarity']}, {row['view']}:")
        lines.append(f"  shift {row['mouse_shift_ms']:+.2f} ms; RMSE {row['normalized_rmse']:.3f}; r {row['correlation']:.3f}")
    axes[1, 2].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("All-session response-verified early V1 PSTHs and MouseV2 timing correction")
    fig.savefig(OUT / "all_session_early_v1_psth_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "all_session_early_v1_psth_alignment.pdf", bbox_inches="tight")
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
    audits, traces, status = [], [], []
    for cohort, sid, units, path, flash_table in session_inputs(allen, mouse):
        print(f"[{cohort} {sid}] {len(units)} candidates", flush=True)
        try:
            audit, trace = verification_rows(cohort, sid, units, path, flash_table)
            audits.append(audit)
            if not trace.empty: traces.append(trace)
            status.append({"cohort": cohort, "session_id": sid, "status": "ok", "candidate_units": len(units), "verified_units": audit.loc[audit.verified_early_responder, "unit_id"].nunique(), "error": ""})
        except Exception as exc:
            status.append({"cohort": cohort, "session_id": sid, "status": "error", "candidate_units": len(units), "verified_units": 0, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
    audit = pd.concat(audits, ignore_index=True); session_trace = pd.concat(traces, ignore_index=True)
    curves = cohort_curves(session_trace)
    grid, alignment = alignment_summary(curves)
    audit.to_csv(OUT / "all_session_response_verification_audit.csv", index=False)
    session_trace.to_csv(OUT / "session_mean_verified_psth_traces.csv", index=False)
    curves.to_csv(OUT / "cohort_session_balanced_psth_curves.csv", index=False)
    grid.to_csv(OUT / "waveform_shift_scan.csv", index=False)
    alignment.to_csv(OUT / "waveform_alignment_summary.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "session_processing_status.csv", index=False)
    render(curves, grid, alignment)


if __name__ == "__main__":
    main()
