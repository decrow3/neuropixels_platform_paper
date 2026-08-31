#!/usr/bin/env python3
"""Trace one outcome-blind MouseV2 15-Hz unit across realized start phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_f1_f0_steady_state_history import (  # noqa: E402
    finite_mean,
    window_trial_components,
)
from scripts.extract_allen_v1_f1_f0_full_cohort import sha256  # noqa: E402
from scripts.extract_allen_v1_f1_f0_full_cohort import trial_f1_f0_components  # noqa: E402
from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    phase_aware_conditions,
    phase_schedule,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


MOUSE_CONFIG = ROOT / "config/figure3_mousev2.json"
STIMULUS_MANIFEST = ROOT / "config/mousev2_stimulus_manifest.json"
UPSTREAM = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/05_steady_state_and_history"
    / "steady_state_unit_draws.csv"
)
OUTPUT = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/06_15hz_frame_phase"
    / "01_initial_case"
)
TF_HZ = 15.0
WINDOWS = {
    "full_15_cycles": (0.0, 1.0, 15),
    "drop_4_cycles": (4.0 / 15.0, 1.0, 11),
    "drop_8_cycles": (8.0 / 15.0, 1.0, 7),
    "last_4_cycles": (11.0 / 15.0, 1.0, 4),
}


def select_case(upstream: pd.DataFrame, site_number: int) -> pd.Series:
    """Select a typical well-supported case without using phase outcomes."""
    pool = upstream.loc[
        upstream.dataset.eq("MouseV2")
        & upstream.cohort.eq("BO_support")
        & upstream.session_id.eq(site_number)
        & upstream.preferred_tf_hz.eq(TF_HZ)
        & upstream.full_f1_f0.gt(0)
    ].copy()
    if pool.empty:
        raise ValueError(f"No 15-Hz MouseV2 cases for site {site_number}")
    target = float(np.log10(pool.full_f1_f0).median())
    eligible = pool.loc[
        pool.full_valid_trials.eq(15)
        & pool.steady_valid_trials.eq(15)
        & pool.steady_mean_spikes.ge(3)
    ].copy()
    if eligible.empty:
        raise ValueError("No fully supported case satisfies the outcome-blind gate")
    eligible["selection_distance"] = np.abs(np.log10(eligible.full_f1_f0) - target)
    selected = eligible.sort_values(["selection_distance", "unit_id"]).iloc[0].copy()
    selected["session_15hz_median_log10_full_f1_f0"] = target
    selected["selection_rule"] = (
        "closest full-window log10 F1/F0 to the session 15-Hz median among units "
        "with 15 valid full trials, 15 valid seven-cycle trials, and >=3 mean "
        "seven-cycle spikes; phase outcomes not used"
    )
    return selected


def carrier_frame_table(start_phases: np.ndarray, frames: int = 8) -> pd.DataFrame:
    phases = np.sort(np.unique(np.round(np.asarray(start_phases, dtype=float), 9)))
    rows = []
    for start_phase in phases:
        for frame in range(frames):
            phase = np.mod(start_phase + TF_HZ * frame / 60.0, 1.0)
            rows.append(
                {
                    "start_phase_cycles": start_phase,
                    "frame": frame,
                    "carrier_phase_cycles": phase,
                    "carrier_cosine": np.cos(2.0 * np.pi * phase),
                }
            )
    return pd.DataFrame(rows)


def phase_summary(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in trials if column.endswith("_f1_f0")]
    aggregations: dict[str, tuple[str, object]] = {
        "trials": ("trial", "size"),
        "mean_full_spikes": ("full_spikes", "mean"),
    }
    for metric in metrics:
        aggregations[f"mean_{metric}"] = (metric, "mean")
        aggregations[f"valid_{metric}"] = (metric, lambda x: int(x.notna().sum()))
    return (
        trials.groupby("start_phase_cycles", sort=True)
        .agg(**aggregations)
        .reset_index()
    )


def render(
    trials: pd.DataFrame,
    spikes_by_trial: list[np.ndarray],
    carrier: pd.DataFrame,
    output: Path,
) -> None:
    colors = {0.0: "#4C78A8", 0.25: "#F58518", 0.5: "#54A24B", 0.75: "#B279A2"}
    order = trials.sort_values(["start_phase_cycles", "trial"]).index.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))

    for phase, group in carrier.groupby("start_phase_cycles", sort=True):
        axes[0, 0].plot(
            group.frame,
            group.carrier_cosine,
            "o-",
            color=colors[float(phase)],
            label=f"start {phase:g} cycles",
        )
    axes[0, 0].axhline(0, color="0.75", lw=0.8)
    axes[0, 0].set(
        xlabel="display frame after onset",
        ylabel="nominal carrier cosine",
        title="Source-defined four-frame carrier sequences",
    )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    for row, trial_index in enumerate(order):
        phase = float(trials.loc[trial_index, "start_phase_cycles"])
        relative = spikes_by_trial[trial_index]
        axes[0, 1].vlines(relative, row - 0.38, row + 0.38, color=colors[phase], lw=0.75)
    axes[0, 1].axvline(8.0 / 15.0, color="black", ls="--", lw=1, label="seven-cycle window")
    axes[0, 1].set(
        xlim=(0, 1),
        ylim=(-1, len(order)),
        xlabel="time from recorded presentation start (s)",
        ylabel="trials sorted by source start phase",
        title="Raw spike raster",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    positions = np.arange(4)
    phases = np.array([0.0, 0.25, 0.5, 0.75])
    comparisons = (
        ("canonical_1ms_f1_f0", "legacy 66-bin fold", "o"),
        ("full_15_cycles_f1_f0", "exact nominal 15 Hz", "^"),
        ("drop_8_cycles_f1_f0", "exact last 7 cycles", "s"),
    )
    for offset, (metric, label, marker) in enumerate(comparisons):
        means = []
        for position, phase in zip(positions, phases):
            values = trials.loc[np.isclose(trials.start_phase_cycles, phase), metric].dropna()
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.zeros(len(values))
            axes[1, 0].scatter(
                position + jitter + (offset - 1.0) * 0.075,
                values,
                color=colors[phase],
                marker=marker,
                alpha=0.75,
                s=34,
            )
            means.append(values.mean())
        axes[1, 0].plot(positions + (offset - 1.0) * 0.075, means, marker=marker, color="black", lw=1, label=label)
    axes[1, 0].set_xticks(positions, ["0", "1/4", "1/2", "3/4"])
    axes[1, 0].set(
        xlabel="source start phase (cycles)",
        ylabel="trial F1/F0",
        title="Phase-stratified modulation estimates",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    for position, phase in zip(positions, phases):
        values = trials.loc[np.isclose(trials.start_phase_cycles, phase), "full_spikes"]
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.zeros(len(values))
        axes[1, 1].scatter(position + jitter, values, color=colors[phase], alpha=0.75, s=34)
        axes[1, 1].plot(position, values.mean(), "_", color="black", ms=18, mew=2)
    axes[1, 1].set_xticks(positions, ["0", "1/4", "1/2", "3/4"])
    axes[1, 1].set(
        xlabel="source start phase (cycles)",
        ylabel="spikes in first 1 s",
        title="Phase-stratified response counts",
    )

    fig.suptitle("Initial 15-Hz frame-phase case: typical well-supported MouseV2 unit")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-number", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    upstream = pd.read_csv(UPSTREAM)
    case = select_case(upstream, args.site_number)
    config = json.loads(MOUSE_CONFIG.read_text())
    session = next(record for record in config["sessions"] if int(record["site_number"]) == args.site_number)
    external_id = int(case.unit_id)
    internal_id = external_id - int(session["id_offset"])
    nwb_path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
    extracted = read_targeted_nwb(
        nwb_path,
        unit_ids=[internal_id],
        interval_name="drifting_gratings_field_block_presentations",
    )
    schedule = phase_schedule(STIMULUS_MANIFEST)
    conditions = phase_aware_conditions(extracted.interval_table, schedule)
    matches = [
        condition
        for condition in conditions
        if np.allclose(
            condition["parameters"],
            (float(case.preferred_orientation_deg), TF_HZ, 0.04, 0.8),
        )
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one frozen preferred condition, found {len(matches)}")
    condition = matches[0]
    starts = np.asarray(condition["starts"], dtype=float)
    phases = np.asarray(condition["start_phase_cycles"], dtype=float)
    spikes = extracted.spikes_by_id[internal_id]
    canonical_counts, canonical_ratios = trial_f1_f0_components(spikes, starts, TF_HZ)

    rows = []
    spikes_by_trial: list[np.ndarray] = []
    window_values = {}
    for name, window in WINDOWS.items():
        window_values[name] = window_trial_components(spikes, starts, TF_HZ, window)
    for trial, (start, phase) in enumerate(zip(starts, phases)):
        first = np.searchsorted(spikes, start, side="left")
        last = np.searchsorted(spikes, start + 1.0, side="left")
        relative = spikes[first:last] - start
        spikes_by_trial.append(relative)
        row = {
            "site": session["site"],
            "session_id": args.site_number,
            "unit_id": external_id,
            "trial": trial,
            "presentation_ordinal": int(condition["ordinals"][trial]),
            "start_time": start,
            "start_phase_cycles": phase,
            "full_spikes": len(relative),
            "canonical_1ms_spikes": int(canonical_counts[trial]),
            "canonical_1ms_f1_f0": canonical_ratios[trial],
        }
        for name, (counts, ratios, f1_hz) in window_values.items():
            row[f"{name}_spikes"] = int(counts[trial])
            row[f"{name}_f1_f0"] = ratios[trial]
            row[f"{name}_f1_hz"] = f1_hz[trial]
        rows.append(row)
    trials = pd.DataFrame(rows)
    summary = phase_summary(trials)
    carrier = carrier_frame_table(phases)

    canonical_center = finite_mean(trials.canonical_1ms_f1_f0.to_numpy())
    exact_full_center = finite_mean(trials.full_15_cycles_f1_f0.to_numpy())
    if not np.isclose(canonical_center, float(case.full_f1_f0), atol=1e-12):
        raise AssertionError(
            f"Frozen full-window value failed: raw {canonical_center}, upstream {case.full_f1_f0}"
        )
    if set(np.round(phases, 9)) != {0.0, 0.25, 0.5, 0.75}:
        raise AssertionError("Expected all four source-defined 15-Hz start phases")

    balanced_canonical = float(summary.mean_canonical_1ms_f1_f0.mean())
    balanced_exact_full = float(summary.mean_full_15_cycles_f1_f0.mean())
    natural_last_7 = finite_mean(trials.drop_8_cycles_f1_f0.to_numpy())
    balanced_last_7 = float(summary.mean_drop_8_cycles_f1_f0.mean())

    case_table = pd.DataFrame(
        [
            {
                **case.to_dict(),
                "site": session["site"],
                "nwb_path": str(nwb_path),
                "internal_unit_id": internal_id,
                "raw_canonical_1ms_full_f1_f0": canonical_center,
                "raw_exact_nominal_15hz_full_f1_f0": exact_full_center,
                "exact_minus_canonical_full_f1_f0": exact_full_center - canonical_center,
                "log10_exact_minus_canonical_full_f1_f0": (
                    np.log10(exact_full_center) - np.log10(canonical_center)
                ),
                "legacy_basis_frequency_hz": 1000.0 / 66.0,
                "legacy_used_duration_s": 0.990,
                "phase_balanced_canonical_1ms_full_f1_f0": balanced_canonical,
                "log10_phase_balanced_minus_natural_canonical": (
                    np.log10(balanced_canonical) - np.log10(canonical_center)
                ),
                "phase_balanced_exact_nominal_15hz_full_f1_f0": balanced_exact_full,
                "log10_phase_balanced_minus_natural_exact_full": (
                    np.log10(balanced_exact_full) - np.log10(exact_full_center)
                ),
                "raw_exact_last_7_cycle_f1_f0": natural_last_7,
                "phase_balanced_exact_last_7_cycle_f1_f0": balanced_last_7,
                "log10_phase_balanced_minus_natural_last_7": (
                    np.log10(balanced_last_7) - np.log10(natural_last_7)
                ),
            }
        ]
    )
    case_table.to_csv(output / "case_selection.csv", index=False)
    trials.to_csv(output / "trial_phase_trace.csv", index=False)
    summary.to_csv(output / "phase_summary.csv", index=False)
    carrier.to_csv(output / "carrier_frame_sequence.csv", index=False)
    render(trials, spikes_by_trial, carrier, output / "initial_15hz_phase_case.png")

    manifest = {
        "status": "initial_concrete_15hz_frame_phase_case",
        "selection": str(case.selection_rule),
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), UPSTREAM, MOUSE_CONFIG, STIMULUS_MANIFEST, nwb_path)
        ],
        "outputs": [],
    }
    for name in (
        "case_selection.csv",
        "trial_phase_trace.csv",
        "phase_summary.csv",
        "carrier_frame_sequence.csv",
        "initial_15hz_phase_case.png",
    ):
        path = output / name
        manifest["outputs"].append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote initial 15-Hz frame-phase case to {output}", flush=True)


if __name__ == "__main__":
    main()
