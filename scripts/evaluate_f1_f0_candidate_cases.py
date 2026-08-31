#!/usr/bin/env python3
"""Evaluate conventional and cross-validated F1/F0 on frozen MouseV2 cases."""

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

from common.drifting_gratings import _bin_trial_spike_counts, f1_f0_from_trial_counts  # noqa: E402
from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    phase_aware_conditions,
    phase_schedule,
)
from scripts.trace_v1_systemic_gap_cases import (  # noqa: E402
    MOUSE_CONFIG,
    STIMULUS_MANIFEST,
    read_targeted_nwb,
    sha256,
)


CASE_SELECTION = ROOT / "artifacts/v1_systemic_gap_audit_v1/02_grating_multicase/case_selection.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/01_case_f1_f0"
REFERENCE_ATOL = 3e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def trial_components(counts: np.ndarray, tf_hz: float) -> pd.DataFrame:
    values = np.asarray(counts, dtype=float)
    cycles = int(float(tf_hz))
    bins_per_cycle = int(values.shape[1] / cycles)
    used = cycles * bins_per_cycle
    folded = values[:, :used].reshape(len(values), cycles, bins_per_cycle)
    average_cycle = folded.mean(axis=1)
    amplitude = 2.0 * np.abs(np.fft.fft(average_cycle, axis=1)) / bins_per_cycle
    # Allen's F0 is half the doubled DC amplitude; convert spikes/ms to spikes/s.
    f0_hz = 0.5 * amplitude[:, 0] * 1000.0
    f1_hz = amplitude[:, 1] * 1000.0
    ratio = np.divide(
        f1_hz,
        f0_hz,
        out=np.full(len(values), np.nan),
        where=f0_hz > 0,
    )
    return pd.DataFrame({"f0_hz": f0_hz, "f1_hz": f1_hz, "f1_f0": ratio})


def all_condition_counts(
    spikes_s: np.ndarray, conditions: list[dict[str, object]]
) -> list[tuple[tuple[float, ...], np.ndarray]]:
    return [
        (
            tuple(condition["parameters"]),
            _bin_trial_spike_counts(
                spikes_s, np.asarray(condition["starts"], dtype=float), duration_ms=1000
            ),
        )
        for condition in conditions
    ]


def select_condition(
    conditions: list[tuple[tuple[float, ...], np.ndarray]], trial_indices: np.ndarray
) -> int:
    means = [counts[trial_indices].sum(axis=1).mean() for _, counts in conditions]
    return int(np.argmax(means))


def evaluate_unit(
    spikes_s: np.ndarray, conditions: list[dict[str, object]]
) -> tuple[dict[str, object], pd.DataFrame]:
    condition_counts = all_condition_counts(spikes_s, conditions)
    n_trials = min(len(counts) for _, counts in condition_counts)
    if n_trials != 15:
        raise ValueError(f"Expected 15 trials per condition, found {n_trials}")
    all_indices = np.arange(n_trials)
    fold_a = all_indices[::2]
    fold_b = all_indices[1::2]
    conventional_index = select_condition(condition_counts, all_indices)
    train_a_index = select_condition(condition_counts, fold_a)
    train_b_index = select_condition(condition_counts, fold_b)

    parameters, preferred_counts = condition_counts[conventional_index]
    tf_hz = float(parameters[1])
    conventional_trials = trial_components(preferred_counts, tf_hz)
    conventional = float(conventional_trials.f1_f0.mean())
    reference = float(f1_f0_from_trial_counts(preferred_counts, tf_hz, 1.0))

    heldout_parts = []
    for selected_index, heldout in ((train_a_index, fold_b), (train_b_index, fold_a)):
        selected_parameters, selected_counts = condition_counts[selected_index]
        selected_tf = float(selected_parameters[1])
        part = trial_components(selected_counts[heldout], selected_tf)
        part["heldout_fold"] = "B" if np.array_equal(heldout, fold_b) else "A"
        part["selected_condition_index"] = selected_index
        part["selected_tf_hz"] = selected_tf
        heldout_parts.append(part)
    heldout = pd.concat(heldout_parts, ignore_index=True)
    crossvalidated = float(heldout.f1_f0.mean())
    crossvalidated_absolute_f1 = float(heldout.f1_hz.mean())
    preferred_mean_f0 = float(conventional_trials.f0_hz.mean())
    if preferred_mean_f0 < 1:
        f0_stratum = "<1"
    elif preferred_mean_f0 < 2:
        f0_stratum = "1-<2"
    elif preferred_mean_f0 < 5:
        f0_stratum = "2-<5"
    else:
        f0_stratum = ">=5"
    summary = {
        "preferred_orientation_deg": float(parameters[0]),
        "preferred_tf_hz": tf_hz,
        "preferred_sf_cpd": float(parameters[2]),
        "conventional_condition_index": conventional_index,
        "fold_a_condition_index": train_a_index,
        "fold_b_condition_index": train_b_index,
        "fold_preference_agreement": train_a_index == train_b_index,
        "fold_a_matches_full": train_a_index == conventional_index,
        "fold_b_matches_full": train_b_index == conventional_index,
        "conventional_f1_f0": conventional,
        "reference_f1_f0": reference,
        "reference_abs_error": abs(conventional - reference),
        "crossvalidated_f1_f0": crossvalidated,
        "crossvalidated_minus_conventional": crossvalidated - conventional,
        "log10_crossvalidated_minus_conventional": (
            np.log10(crossvalidated) - np.log10(conventional)
            if conventional > 0 and crossvalidated > 0 else np.nan
        ),
        "mean_absolute_f1_hz": float(conventional_trials.f1_hz.mean()),
        "crossvalidated_absolute_f1_hz": crossvalidated_absolute_f1,
        "log10_crossvalidated_minus_conventional_absolute_f1": (
            np.log10(crossvalidated_absolute_f1)
            - np.log10(float(conventional_trials.f1_hz.mean()))
            if crossvalidated_absolute_f1 > 0 and conventional_trials.f1_hz.mean() > 0
            else np.nan
        ),
        "preferred_mean_f0_hz": preferred_mean_f0,
        "preferred_zero_f0_trials": int(conventional_trials.f0_hz.eq(0).sum()),
        "preferred_valid_ratio_trials": int(conventional_trials.f1_f0.notna().sum()),
        "preferred_f1_f0_max": float(conventional_trials.f1_f0.max()),
        "f0_stratum": f0_stratum,
    }
    trace = conventional_trials.copy()
    trace["trial"] = np.arange(n_trials)
    trace["condition_role"] = "full_data_preferred"
    return summary, trace


def render(cases: pd.DataFrame, output: Path) -> None:
    audit = cases.loc[cases.selection_roles.str.contains("outcome_blind")].copy()
    colors = {"<1": "#E45756", "1-<2": "#F2CF5B", "2-<5": "#4C78A8", ">=5": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for stratum, group in audit.groupby("f0_stratum"):
        axes[0].scatter(
            group.conventional_f1_f0,
            group.crossvalidated_f1_f0,
            color=colors[stratum], label=f"F0 {stratum} Hz", s=32, alpha=0.75,
        )
    limit = max(audit.conventional_f1_f0.max(), audit.crossvalidated_f1_f0.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], "k--", lw=1)
    axes[0].set(xlabel="conventional F1/F0", ylabel="cross-validated F1/F0", title="Preference-selection sensitivity")
    axes[0].legend(frameon=False, fontsize=8)
    order = ["<1", "1-<2", "2-<5", ">=5"]
    for position, stratum in enumerate(order):
        values = audit.loc[audit.f0_stratum.eq(stratum), "log10_crossvalidated_minus_conventional"]
        axes[1].scatter(np.full(len(values), position), values, color=colors[stratum], alpha=0.75)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(order)), order)
    axes[1].set(xlabel="preferred-condition mean F0 (Hz)", ylabel="log10(CV / conventional)", title="Low-F0 behavior")
    axes[2].scatter(audit.preferred_mean_f0_hz, audit.preferred_f1_f0_max, color="#6F63A6", alpha=0.7)
    axes[2].set(xscale="log", xlabel="preferred-condition mean F0 (Hz)", ylabel="maximum trial F1/F0", title="Ratio extremes versus firing rate")
    fig.suptitle("F1/F0 replacement candidate: frozen MouseV2 cases")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(CASE_SELECTION)
    config = json.loads(MOUSE_CONFIG.read_text())
    by_site = {str(record["site"]): record for record in config["sessions"]}
    schedule = phase_schedule(STIMULUS_MANIFEST)
    summaries = []
    trial_frames = []
    for site, group in selected.groupby("site", sort=True):
        session = by_site[site]
        offset = int(session["id_offset"])
        local_ids = [int(unit_id) - offset for unit_id in group.unit_id]
        path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        extracted = read_targeted_nwb(
            path,
            unit_ids=local_ids,
            interval_name="drifting_gratings_field_block_presentations",
        )
        conditions = phase_aware_conditions(extracted.interval_table, schedule)
        for _, case in group.iterrows():
            local_id = int(case.unit_id) - offset
            summary, trials = evaluate_unit(extracted.spikes_by_id[local_id], conditions)
            common = {
                "site": site,
                "session_id": int(case.session_id),
                "unit_id": int(case.unit_id),
                "selection_roles": case.selection_roles,
            }
            summaries.append({**common, **summary})
            for key, value in common.items():
                trials[key] = value
            trial_frames.append(trials)
    cases = pd.DataFrame(summaries)
    trials = pd.concat(trial_frames, ignore_index=True)
    cases.to_csv(output / "case_metric_comparison.csv", index=False)
    trials.to_csv(output / "preferred_trial_components.csv", index=False)
    render(cases, output / "f1_f0_case_diagnostics.png")
    files = [
        Path(__file__).resolve(), CASE_SELECTION, MOUSE_CONFIG, STIMULUS_MANIFEST,
        ROOT / "common/drifting_gratings.py",
    ]
    manifest = {
        "schema_version": 1,
        "status": "exploratory_frozen_case_checkpoint",
        "selection_source": str(CASE_SELECTION),
        "crossvalidation": "two-fold alternating trials; select on one fold and evaluate on the other, then swap",
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "outputs": [],
    }
    for name in ("case_metric_comparison.csv", "preferred_trial_components.csv", "f1_f0_case_diagnostics.png"):
        path = output / name
        manifest["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = output / "F1_F0_CASE_CHECKPOINT.md"
    if report.is_file():
        manifest["outputs"].append(
            {"path": report.name, "bytes": report.stat().st_size, "sha256": sha256(report)}
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if cases.reference_abs_error.max() > REFERENCE_ATOL:
        raise AssertionError("Per-trial decomposition does not match canonical F1/F0")
    print(f"Wrote F1/F0 frozen-case checkpoint to {output}")


if __name__ == "__main__":
    main()
