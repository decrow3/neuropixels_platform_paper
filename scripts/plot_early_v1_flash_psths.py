#!/usr/bin/env python3
"""Plot raw flash PSTHs for early common-QC V1 units in three datasets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from common.flashes import prepare_flash_presentations
from generate_retinotopic_csvs import read_nwb_tables

OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/early_responder_psths"
EDGES_S = np.arange(-0.100, 0.205, 0.005)
CENTERS_MS = (EDGES_S[:-1] + np.diff(EDGES_S) / 2) * 1000
EARLY_MS = 50.0
MAX_UNITS = 30


CASES = (
    {
        "cohort": "Allen Brain Observatory",
        "session_id": 737581020,
        "path": Path("/media/huklaban5/Data/MouseV2/allen_v1_bridge/000021/sub-718643564/sub-718643564_ses-737581020.nwb"),
        "flash_table": "flashes_presentations",
        "color": "#6F63A6",
    },
    {
        "cohort": "Allen Functional Connectivity",
        "session_id": 835479236,
        "path": Path("/media/huklaban5/Data/MouseV2/allen_v1_bridge/000022/sub-813701555/sub-813701555_ses-835479236.nwb"),
        "flash_table": "flashes_presentations",
        "color": "#B07AA1",
    },
    {
        "cohort": "MouseV2",
        "session_id": 7,
        "path": Path("/media/huklaban5/Data/MouseV2/001568/sub-816308/sub-816308_ses-ecephys-816308-2025-08-20-15-24-06_ecephys.nwb"),
        "flash_table": "flash_field_block_presentations",
        "color": "#D95F02",
    },
)


def trial_counts(spikes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    absolute_edges = starts[:, None] + EDGES_S[None, :]
    indices = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(indices, axis=1).astype(float)


def session_units(case: dict, allen: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    if case["cohort"].startswith("Allen"):
        units = allen.loc[allen["ecephys_session_id"].eq(case["session_id"])].copy()
        units["source_unit_id"] = units["ecephys_unit_id"].astype(int)
        units["analysis_unit_id"] = units["ecephys_unit_id"].astype(int)
    else:
        units = mouse.loc[mouse["session_num"].eq(case["session_id"])].copy()
        units["analysis_unit_id"] = units["unit_id"].astype(int)
        units["source_unit_id"] = (
            units["unit_id"].astype(int) - int(case["session_id"]) * 1_000_000
        )
    units["ttfs_ms"] = 1000 * pd.to_numeric(
        units["time_to_first_spike_fl"], errors="coerce"
    )
    units = units.loc[units["ttfs_ms"].lt(EARLY_MS)].sort_values(
        ["ttfs_ms", "analysis_unit_id"], kind="stable"
    )
    # Predeclare the auditable cap: earliest valid units, stable ID tie-break.
    return units.head(MAX_UNITS).copy()


def extract_case(case: dict, units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted = read_nwb_tables(str(case["path"]))
    flashes = prepare_flash_presentations(extracted.intervals_tables[case["flash_table"]])
    starts = flashes["start_time"].to_numpy(float)
    nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
    row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
    rows, selection = [], []
    for rank, (_, unit) in enumerate(units.iterrows(), start=1):
        source_id = int(unit["source_unit_id"])
        if source_id not in row_by_id:
            raise KeyError(f"{case['cohort']}: unit {source_id} missing from NWB")
        spikes = extracted.spikes_by_unit[row_by_id[source_id]]
        counts = trial_counts(spikes, starts)
        rate = counts.mean(axis=0) / np.diff(EDGES_S)
        baseline = float(rate[CENTERS_MS < 0].mean())
        delta = rate - baseline
        scale = float(np.nanmax(np.abs(delta)))
        zlike = delta / scale if scale > 0 else np.zeros_like(delta)
        selection.append(
            {
                "cohort": case["cohort"], "session_id": case["session_id"],
                "unit_id": int(unit["analysis_unit_id"]), "source_unit_id": source_id,
                "selection_rank": rank, "released_ttfs_ms": float(unit["ttfs_ms"]),
                "flash_trials": len(starts), "baseline_rate_hz": baseline,
                "peak_delta_rate_hz": float(delta.max()),
            }
        )
        for time, raw, change, normalized in zip(CENTERS_MS, rate, delta, zlike):
            rows.append(
                {
                    "cohort": case["cohort"], "session_id": case["session_id"],
                    "unit_id": int(unit["analysis_unit_id"]), "selection_rank": rank,
                    "released_ttfs_ms": float(unit["ttfs_ms"]), "time_ms": time,
                    "rate_hz": raw, "baseline_subtracted_rate_hz": change,
                    "within_unit_scaled_response": normalized,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selection)


def bootstrap_mean_ci(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matrix), size=(2000, len(matrix)))
    boot = matrix[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def render(traces: pd.DataFrame, selection: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1]}, constrained_layout=True,
    )
    for col, case in enumerate(CASES):
        cohort = case["cohort"]
        subset = traces.loc[traces["cohort"].eq(cohort)]
        matrix = subset.pivot(index="selection_rank", columns="time_ms", values="within_unit_scaled_response").to_numpy()
        im = axes[0, col].imshow(
            matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r",
            vmin=-1, vmax=1, extent=[CENTERS_MS[0], CENTERS_MS[-1], len(matrix) + .5, .5],
        )
        axes[0, col].axvline(0, color="k", lw=1, ls=":")
        n_trials = int(selection.loc[selection["cohort"].eq(cohort), "flash_trials"].iloc[0])
        axes[0, col].set_title(f"{cohort}\nsession {case['session_id']}; {len(matrix)} units; {n_trials} flashes")
        axes[0, col].set_ylabel("Unit rank by released TTFS" if col == 0 else "")

        raw_matrix = subset.pivot(index="selection_rank", columns="time_ms", values="baseline_subtracted_rate_hz").to_numpy()
        mean, low, high = bootstrap_mean_ci(raw_matrix, 20260825 + col)
        axes[1, col].fill_between(CENTERS_MS, low, high, color=case["color"], alpha=.22, lw=0)
        axes[1, col].plot(CENTERS_MS, mean, color=case["color"], lw=2.4)
        axes[1, col].axhline(0, color="0.45", lw=.8)
        axes[1, col].axvline(0, color="k", lw=1, ls=":")
        axes[1, col].axvspan(0, 200, color="0.85", alpha=.25, zorder=-2)
        axes[1, col].set_xlabel("Time from NWB flash onset (ms)")
        axes[1, col].set_ylabel("Mean baseline-subtracted\nfiring rate (Hz)" if col == 0 else "")
        axes[1, col].spines[["top", "right"]].set_visible(False)
    fig.colorbar(im, ax=axes[0, :], label="Within-unit scaled baseline-subtracted response", shrink=.85, pad=.01)
    fig.suptitle("Raw flash PSTHs of early common-QC V1 neurons (released TTFS < 50 ms)")
    fig.savefig(OUT / "representative_sessions_early_v1_flash_psths.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "representative_sessions_early_v1_flash_psths.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].eq("V1")].copy()
    mouse = load_mousev2_units(
        apply_qc=False,
        grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    trace_frames, selection_frames = [], []
    for case in CASES:
        units = session_units(case, allen, mouse)
        if units.empty:
            raise ValueError(f"No early units for {case['cohort']}")
        traces, selection = extract_case(case, units)
        trace_frames.append(traces)
        selection_frames.append(selection)
    traces = pd.concat(trace_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    traces.to_csv(OUT / "representative_sessions_early_v1_psth_traces.csv", index=False)
    selection.to_csv(OUT / "representative_sessions_early_v1_selection.csv", index=False)
    (OUT / "analysis_definition.json").write_text(json.dumps({
        "selection": "common-QC V1, released pooled-flash TTFS < 50 ms; earliest 30 per session; stable unit-ID tie-break",
        "sessions": [{k: str(v) if isinstance(v, Path) else v for k, v in c.items()} for c in CASES],
        "psth": "5-ms spike-count bins, pooled bright+dark trials, -100 to +200 ms, aligned to NWB interval start_time",
        "population_trace": "unit-mean baseline-subtracted firing rate; 95% unit-bootstrap interval",
        "heatmap": "each unit baseline-subtracted then divided by its maximum absolute response",
    }, indent=2) + "\n")
    render(traces, selection)


if __name__ == "__main__":
    main()
