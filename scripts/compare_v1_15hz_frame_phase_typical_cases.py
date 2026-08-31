#!/usr/bin/env python3
"""Compare outcome-blind typical MouseV2 15-Hz cases across eight sessions."""

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
from scripts.extract_allen_v1_f1_f0_full_cohort import (  # noqa: E402
    sha256,
    trial_f1_f0_components,
)
from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    phase_aware_conditions,
    phase_schedule,
)
from scripts.trace_v1_15hz_frame_phase_case import (  # noqa: E402
    MOUSE_CONFIG,
    STIMULUS_MANIFEST,
    TF_HZ,
    UPSTREAM,
    WINDOWS,
    phase_summary,
    select_case,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


OUTPUT = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/06_15hz_frame_phase"
    / "02_typical_cases"
)
PHASES = np.array([0.0, 0.25, 0.5, 0.75])


def extract_case(
    case: pd.Series,
    session: dict[str, object],
    nwb_root: Path,
    schedule: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    external_id = int(case.unit_id)
    internal_id = external_id - int(session["id_offset"])
    nwb_path = nwb_root / str(session["nwb_relative_path"])
    extracted = read_targeted_nwb(
        nwb_path,
        unit_ids=[internal_id],
        interval_name="drifting_gratings_field_block_presentations",
    )
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
        raise AssertionError(f"{session['site']}: expected one frozen preferred condition")
    condition = matches[0]
    starts = np.asarray(condition["starts"], dtype=float)
    phases = np.asarray(condition["start_phase_cycles"], dtype=float)
    spikes = extracted.spikes_by_id[internal_id]
    canonical_counts, canonical_ratios = trial_f1_f0_components(spikes, starts, TF_HZ)
    window_values = {
        name: window_trial_components(spikes, starts, TF_HZ, window)
        for name, window in WINDOWS.items()
    }

    rows = []
    for trial, (start, phase) in enumerate(zip(starts, phases)):
        row = {
            "site": session["site"],
            "session_id": int(session["site_number"]),
            "unit_id": external_id,
            "trial": trial,
            "presentation_ordinal": int(condition["ordinals"][trial]),
            "start_time": start,
            "start_phase_cycles": phase,
            "full_spikes": int(canonical_counts[trial]),
            "canonical_1ms_spikes": int(canonical_counts[trial]),
            "canonical_1ms_f1_f0": canonical_ratios[trial],
        }
        for name, (counts, ratios, f1_hz) in window_values.items():
            row[f"{name}_spikes"] = int(counts[trial])
            row[f"{name}_f1_f0"] = ratios[trial]
            row[f"{name}_f1_hz"] = f1_hz[trial]
        rows.append(row)
    trials = pd.DataFrame(rows)
    phases_summary = phase_summary(trials)
    observed_phases = set(np.round(phases, 9))
    complete_phase_support = observed_phases == set(PHASES)
    missing_phases = sorted(set(PHASES).difference(observed_phases))

    canonical = finite_mean(trials.canonical_1ms_f1_f0.to_numpy())
    exact_full = finite_mean(trials.full_15_cycles_f1_f0.to_numpy())
    exact_late = finite_mean(trials.drop_8_cycles_f1_f0.to_numpy())
    if not np.isclose(canonical, float(case.full_f1_f0), atol=1e-12):
        raise AssertionError(f"{session['site']}: upstream reproduction failed")
    if not np.isclose(exact_late, float(case.steady_f1_f0), atol=1e-12):
        raise AssertionError(f"{session['site']}: steady-state reproduction failed")

    canonical_by_phase = phases_summary.mean_canonical_1ms_f1_f0.to_numpy()
    exact_by_phase = phases_summary.mean_full_15_cycles_f1_f0.to_numpy()
    late_by_phase = phases_summary.mean_drop_8_cycles_f1_f0.to_numpy()
    balanced_canonical = float(np.mean(canonical_by_phase)) if complete_phase_support else np.nan
    balanced_exact = float(np.mean(exact_by_phase)) if complete_phase_support else np.nan
    balanced_late = float(np.mean(late_by_phase)) if complete_phase_support else np.nan
    summary = {
        **case.to_dict(),
        "site": session["site"],
        "internal_unit_id": internal_id,
        "nwb_path": str(nwb_path),
        "observed_phase_count": len(observed_phases),
        "complete_four_phase_support": complete_phase_support,
        "missing_phases": ";".join(f"{phase:g}" for phase in missing_phases),
        "phase_trial_counts": ";".join(
            f"{row.start_phase_cycles:g}:{int(row.trials)}"
            for row in phases_summary.itertuples()
        ),
        "natural_canonical_f1_f0": canonical,
        "natural_exact_full_f1_f0": exact_full,
        "natural_exact_last7_f1_f0": exact_late,
        "log10_exact_minus_canonical": np.log10(exact_full) - np.log10(canonical),
        "phase_range_canonical": float(np.ptp(canonical_by_phase)),
        "phase_range_exact_full": float(np.ptp(exact_by_phase)),
        "phase_range_exact_last7": float(np.ptp(late_by_phase)),
        "phase_range_last7_minus_full": float(np.ptp(late_by_phase) - np.ptp(exact_by_phase)),
        "phase_range_fraction_remaining": float(np.ptp(late_by_phase) / np.ptp(exact_by_phase)) if np.ptp(exact_by_phase) > 0 else np.nan,
        "phase_balanced_canonical_f1_f0": balanced_canonical,
        "phase_balanced_exact_full_f1_f0": balanced_exact,
        "phase_balanced_exact_last7_f1_f0": balanced_late,
        "log10_balanced_minus_natural_canonical": np.log10(balanced_canonical) - np.log10(canonical) if complete_phase_support else np.nan,
        "log10_balanced_minus_natural_exact_full": np.log10(balanced_exact) - np.log10(exact_full) if complete_phase_support else np.nan,
        "log10_balanced_minus_natural_exact_last7": np.log10(balanced_late) - np.log10(exact_late) if complete_phase_support else np.nan,
    }
    return summary, trials, phases_summary


def render_profiles(phases: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), sharex=True, sharey=True)
    for axis, (site, group) in zip(axes.flat, phases.groupby("site", sort=True)):
        axis.plot(group.start_phase_cycles, group.mean_canonical_1ms_f1_f0, "o-", label="legacy 66-bin")
        axis.plot(group.start_phase_cycles, group.mean_full_15_cycles_f1_f0, "^-", label="exact full")
        axis.plot(group.start_phase_cycles, group.mean_drop_8_cycles_f1_f0, "s-", label="exact last 7")
        for _, row in group.iterrows():
            axis.text(row.start_phase_cycles, 0.03, f"n={int(row.trials)}", transform=axis.get_xaxis_transform(), ha="center", fontsize=7)
        unit_id = int(group.unit_id.iloc[0])
        axis.set_title(f"{site} · unit {unit_id}")
        axis.axhline(0, color="0.85", lw=0.7)
    for axis in axes[-1]:
        axis.set_xticks(PHASES, ["0", "1/4", "1/2", "3/4"])
        axis.set_xlabel("start phase (cycles)")
    for axis in axes[:, 0]:
        axis.set_ylabel("phase-stratum mean F1/F0")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Outcome-blind typical 15-Hz cases: direct phase profiles")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_paired(cases: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    x = np.arange(len(cases))
    labels = cases.site.to_numpy()
    for position, row in cases.reset_index(drop=True).iterrows():
        axes[0].plot([0, 1], [row.phase_range_exact_full, row.phase_range_exact_last7], "o-", alpha=0.75)
    axes[0].set_xticks([0, 1], ["exact full", "exact last 7"])
    axes[0].set(ylabel="max-min phase-stratum mean F1/F0", title="Does onset removal reduce phase spread?")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].bar(x, cases.log10_exact_minus_canonical, color="#4C78A8")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set(ylabel="log10(exact nominal / legacy)", title="15-Hz basis discretization")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].bar(x - 0.18, cases.log10_balanced_minus_natural_exact_full, width=0.36, label="exact full")
    axes[2].bar(x + 0.18, cases.log10_balanced_minus_natural_exact_last7, width=0.36, label="exact last 7")
    axes[2].set_xticks(x, labels, rotation=45)
    axes[2].set(ylabel="log10(phase-balanced / natural)", title="Effect of equal phase weighting")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired diagnostics across eight typical 15-Hz cases")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    upstream = pd.read_csv(UPSTREAM)
    config = json.loads(MOUSE_CONFIG.read_text())
    nwb_root = Path(config["nwb_input"]["default_root"])
    schedule = phase_schedule(STIMULUS_MANIFEST)
    case_rows = []
    trial_frames = []
    phase_frames = []
    nwb_paths = []
    for number, session in enumerate(config["sessions"], start=1):
        site_number = int(session["site_number"])
        case = select_case(upstream, site_number)
        print(f"[{number}/8] {session['site']} unit {int(case.unit_id)}", flush=True)
        result, trials, phases = extract_case(case, session, nwb_root, schedule)
        case_rows.append(result)
        trial_frames.append(trials)
        phases["site"] = session["site"]
        phases["session_id"] = site_number
        phases["unit_id"] = int(case.unit_id)
        phase_frames.append(phases)
        nwb_paths.append(Path(result["nwb_path"]))

    cases = pd.DataFrame(case_rows).sort_values("session_id").reset_index(drop=True)
    trials = pd.concat(trial_frames, ignore_index=True)
    phases = pd.concat(phase_frames, ignore_index=True)
    cases.to_csv(output / "typical_case_selection_and_metrics.csv", index=False)
    trials.to_csv(output / "typical_case_trial_phase_traces.csv", index=False)
    phases.to_csv(output / "typical_case_phase_summaries.csv", index=False)
    render_profiles(phases, output / "typical_case_phase_profiles.png")
    render_paired(cases, output / "typical_case_paired_diagnostics.png")

    manifest = {
        "status": "exploratory_multicase_15hz_frame_phase_checkpoint",
        "case_count": len(cases),
        "selection": str(cases.selection_rule.iloc[0]),
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), UPSTREAM, MOUSE_CONFIG, STIMULUS_MANIFEST)
        ],
        "large_nwb_sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": "not_recomputed_for_large_source"}
            for path in nwb_paths
        ],
        "outputs": [],
    }
    for name in (
        "typical_case_selection_and_metrics.csv",
        "typical_case_trial_phase_traces.csv",
        "typical_case_phase_summaries.csv",
        "typical_case_phase_profiles.png",
        "typical_case_paired_diagnostics.png",
    ):
        path = output / name
        manifest["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote eight-case 15-Hz checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
