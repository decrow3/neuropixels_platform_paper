#!/usr/bin/env python3
"""Test steady-state F1/F0 and preceding-contrast history mechanisms."""

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

from common.figure3_mousev2 import load_mousev2_units  # noqa: E402
from scripts.extract_allen_v1_bridge import COHORT_LABELS, common_qc, condition_starts  # noqa: E402
from scripts.extract_allen_v1_f1_f0_full_cohort import (  # noqa: E402
    draw_indices,
    interval_name,
    sha256,
    trial_f1_f0_components,
)
from scripts.extract_mousev2_f1_f0_tf2_support import mouse_conditions  # noqa: E402
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


ALLEN_INVENTORY = Path("/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/session_inventory.json")
ALLEN_CONFIG = ROOT / "config/allen_v1_bridge.json"
MOUSE_CONFIG = ROOT / "config/figure3_mousev2.json"
RELEASE = ROOT / "data/unit_table.csv"
FULL_BRIDGE = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/04_allen_full_cohort_harmonized"
OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/05_steady_state_and_history"


def steady_window(tf_hz: float) -> tuple[float, float, int] | None:
    """Last complete cycles after at least 500 ms in a one-second trial."""
    cycles = int(np.floor(0.5 * float(tf_hz)))
    if cycles < 1:
        return None
    end = 1.0
    start = end - cycles / float(tf_hz)
    return start, end, cycles


def window_trial_components(
    spikes_s: np.ndarray,
    starts_s: np.ndarray,
    tf_hz: float,
    window: tuple[float, float, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.asarray(starts_s, dtype=float)
    counts = np.zeros(len(starts), dtype=np.int32)
    ratios = np.full(len(starts), np.nan)
    f1_hz = np.full(len(starts), np.nan)
    if window is None:
        return counts, ratios, f1_hz
    window_start, window_end, _ = window
    duration = window_end - window_start
    for index, start in enumerate(starts):
        first = np.searchsorted(spikes_s, start + window_start, side="left")
        last = np.searchsorted(spikes_s, start + window_end, side="left")
        trial = spikes_s[first:last]
        counts[index] = len(trial)
        if len(trial) == 0:
            continue
        vector = np.exp(-2j * np.pi * tf_hz * (trial - start)).sum()
        ratios[index] = 2.0 * np.abs(vector) / len(trial)
        f1_hz[index] = 2.0 * np.abs(vector) / duration
    return counts, ratios, f1_hz


def prepare_unit(
    spikes: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
) -> list[dict[str, object]]:
    rows = []
    for parameters, starts in conditions:
        full_counts, full_ratios = trial_f1_f0_components(spikes, starts, parameters[1])
        steady_counts, steady_ratios, steady_f1 = window_trial_components(
            spikes, starts, parameters[1], steady_window(parameters[1])
        )
        rows.append(
            {
                "parameters": parameters,
                "starts": starts,
                "full_counts": full_counts,
                "full_ratios": full_ratios,
                "steady_counts": steady_counts,
                "steady_ratios": steady_ratios,
                "steady_f1_hz": steady_f1,
            }
        )
    return rows


def finite_mean(values: np.ndarray) -> float:
    selected = values[np.isfinite(values)]
    return float(selected.mean()) if len(selected) else np.nan


def summarize_draw(
    prepared: list[dict[str, object]], indices_by_condition: list[np.ndarray]
) -> dict[str, object]:
    means = [
        float(np.mean(condition["full_counts"][indices]))
        for condition, indices in zip(prepared, indices_by_condition)
    ]
    preferred = int(np.argmax(means))
    selected = prepared[preferred]
    indices = indices_by_condition[preferred]
    full = selected["full_ratios"][indices]
    steady = selected["steady_ratios"][indices]
    return {
        "preferred_condition_index": preferred,
        "preferred_orientation_deg": float(selected["parameters"][0]),
        "preferred_tf_hz": float(selected["parameters"][1]),
        "full_f1_f0": finite_mean(full),
        "steady_f1_f0": finite_mean(steady),
        "full_valid_trials": int(np.isfinite(full).sum()),
        "steady_valid_trials": int(np.isfinite(steady).sum()),
        "steady_mean_spikes": float(np.mean(selected["steady_counts"][indices])),
        "steady_mean_f1_hz": finite_mean(selected["steady_f1_hz"][indices]),
    }


def previous_contrast_labels(table: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    ordered = table.sort_values("start_time").reset_index(drop=True)
    all_starts = pd.to_numeric(ordered.start_time, errors="coerce").to_numpy(float)
    contrasts = pd.to_numeric(ordered.contrast, errors="coerce").to_numpy(float)
    labels = []
    for start in starts:
        current = int(np.searchsorted(all_starts, start, side="left"))
        if current >= len(all_starts) or not np.isclose(all_starts[current], start, atol=1e-6):
            raise AssertionError("Could not map FC trial to chronological presentation")
        labels.append(contrasts[current - 1] if current > 0 else np.nan)
    return np.asarray(labels)


def bootstrap_difference(
    mouse: np.ndarray,
    allen_draws: pd.DataFrame,
    value_column: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sessions = allen_draws.session_id.unique()
    values_by_session = {
        sid: allen_draws.loc[allen_draws.session_id.eq(sid), value_column].dropna().to_numpy()
        for sid in sessions
    }
    allen_center = np.mean([np.median(values_by_session[sid]) for sid in sessions])
    observed = float(mouse.mean() - allen_center)
    result = np.empty(draws)
    for draw in range(draws):
        mouse_sample = rng.choice(mouse, len(mouse), replace=True)
        selected_sessions = rng.choice(sessions, len(sessions), replace=True)
        allen_sample = [rng.choice(values_by_session[sid]) for sid in selected_sessions]
        result[draw] = mouse_sample.mean() - np.mean(allen_sample)
    return observed, float(np.quantile(result, .025)), float(np.quantile(result, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--trial-draws", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(ALLEN_INVENTORY.read_text())
    inventory_by_id = {int(row["ecephys_session_id"]): row for row in inventory}
    support = json.loads(ALLEN_CONFIG.read_text())["common_support"]
    release = pd.read_csv(RELEASE, low_memory=False)
    relevant = release.loc[
        release.ecephys_structure_acronym.eq("VISp")
        & release.session_type.isin(COHORT_LABELS)
    ].copy()
    relevant = relevant.loc[common_qc(relevant)]

    unit_rows = []
    history_rows = []
    for number, (session_id, selected) in enumerate(relevant.groupby("ecephys_session_id"), start=1):
        session_id = int(session_id)
        session_type = str(selected.session_type.iloc[0])
        path = Path(inventory_by_id[session_id]["nwb_path"])
        unit_ids = selected.ecephys_unit_id.astype(int).tolist()
        print(f"[Allen {number:02d}/56] {session_id}: {len(unit_ids)} units", flush=True)
        extracted = read_targeted_nwb(path, unit_ids=unit_ids, interval_name=interval_name(session_type))
        conditions = condition_starts(extracted.interval_table, common_support=support)
        draws = draw_indices(conditions, session_id=session_id, draws=args.trial_draws)
        full_indices = [np.arange(len(starts)) for _, starts in conditions]
        for unit_id in unit_ids:
            prepared = prepare_unit(extracted.spikes_by_id[unit_id], conditions)
            full_support = summarize_draw(prepared, full_indices)
            for draw, indices in enumerate(draws):
                result = summarize_draw(prepared, indices)
                unit_rows.append(
                    {
                        "dataset": "Allen",
                        "cohort": COHORT_LABELS[session_type],
                        "session_id": session_id,
                        "unit_id": unit_id,
                        "draw": draw,
                        **result,
                    }
                )
            if session_type == "functional_connectivity":
                condition = prepared[int(full_support["preferred_condition_index"])]
                labels = previous_contrast_labels(extracted.interval_table, condition["starts"])
                for trial, (start, previous) in enumerate(zip(condition["starts"], labels)):
                    history_rows.append(
                        {
                            "session_id": session_id,
                            "unit_id": unit_id,
                            "preferred_orientation_deg": full_support["preferred_orientation_deg"],
                            "trial": trial,
                            "start_time": start,
                            "previous_contrast": previous,
                            "full_f1_f0": condition["full_ratios"][trial],
                            "steady_f1_f0": condition["steady_ratios"][trial],
                            "full_spikes": condition["full_counts"][trial],
                            "steady_spikes": condition["steady_counts"][trial],
                        }
                    )

    mouse_config = json.loads(MOUSE_CONFIG.read_text())
    mouse_root = Path(mouse_config["nwb_input"]["default_root"])
    mouse_qc = load_mousev2_units(
        apply_qc=False,
        population_profile="common_qc",
        grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
    )
    for number, session in enumerate(mouse_config["sessions"], start=1):
        site = str(session["site"])
        site_number = int(session["site_number"])
        offset = int(session["id_offset"])
        path = mouse_root / session["nwb_relative_path"]
        selected = mouse_qc.loc[mouse_qc.site.eq(site)]
        external_ids = selected.unit_id.astype(int).to_numpy()
        internal_ids = (external_ids - offset).tolist()
        print(f"[Mouse {number}/8] {site}: {len(internal_ids)} units", flush=True)
        extracted = read_targeted_nwb(
            path, unit_ids=internal_ids, interval_name="drifting_gratings_field_block_presentations"
        )
        for view, conditions in (
            ("BO_support", mouse_conditions(extracted.interval_table)),
            ("FC_TF2_support", mouse_conditions(extracted.interval_table, temporal_frequency_hz=2.0)),
        ):
            indices = [np.arange(len(starts)) for _, starts in conditions]
            for external_id, internal_id in zip(external_ids, internal_ids):
                result = summarize_draw(prepare_unit(extracted.spikes_by_id[internal_id], conditions), indices)
                unit_rows.append(
                    {
                        "dataset": "MouseV2",
                        "cohort": view,
                        "session_id": site_number,
                        "unit_id": int(external_id),
                        "draw": 0,
                        **result,
                    }
                )

    units = pd.DataFrame(unit_rows)
    history_trials = pd.DataFrame(history_rows)
    units.to_csv(output / "steady_state_unit_draws.csv", index=False)
    history_trials.to_csv(output / "fc_preceding_contrast_trials.csv", index=False)

    # Gate the full-window values against the previously accepted bridge.
    accepted = pd.read_csv(FULL_BRIDGE / "harmonized_session_draws.csv")
    allen_units = units.loc[units.dataset.eq("Allen")].copy()
    allen_units["log_full"] = np.log10(allen_units.full_f1_f0.where(allen_units.full_f1_f0 > 0))
    reproduced = (
        allen_units.groupby(["session_id", "draw"]).log_full.mean().rename("reproduced").reset_index()
    )
    gate = accepted.merge(reproduced, on=["session_id", "draw"], validate="one_to_one")
    gate["absolute_error"] = (gate.mean_log10_f1_f0 - gate.reproduced).abs()
    gate.to_csv(output / "full_window_reproduction_gate.csv", index=False)
    if gate.absolute_error.max() > 5e-9:
        raise AssertionError("Full-window bridge reproduction failed")

    session_rows = []
    for (dataset, cohort, session_id, draw, tf), group in units.groupby(
        ["dataset", "cohort", "session_id", "draw", "preferred_tf_hz"]
    ):
        full = np.log10(group.full_f1_f0.where(group.full_f1_f0 > 0))
        steady = np.log10(group.steady_f1_f0.where(group.steady_f1_f0 > 0))
        session_rows.append(
            {
                "dataset": dataset,
                "cohort": cohort,
                "session_id": session_id,
                "draw": draw,
                "preferred_tf_hz": tf,
                "units": len(group),
                "full_valid_units": full.notna().sum(),
                "steady_valid_units": steady.notna().sum(),
                "mean_log10_full_f1_f0": full.mean(),
                "mean_log10_steady_f1_f0": steady.mean(),
            }
        )
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(output / "steady_state_session_draws_by_tf.csv", index=False)

    contrast_rows = []
    for tf in (2.0, 4.0, 8.0, 15.0):
        mouse = sessions.loc[
            sessions.dataset.eq("MouseV2") & sessions.cohort.eq("BO_support")
            & sessions.preferred_tf_hz.eq(tf) & sessions.steady_valid_units.ge(5),
            "mean_log10_steady_f1_f0",
        ].dropna().to_numpy()
        allen = sessions.loc[
            sessions.dataset.eq("Allen") & sessions.cohort.eq("Allen Brain Observatory 1.1")
            & sessions.preferred_tf_hz.eq(tf) & sessions.steady_valid_units.ge(5)
        ]
        observed, low, high = bootstrap_difference(
            mouse, allen, "mean_log10_steady_f1_f0", draws=args.bootstrap_draws, seed=20260825 + int(tf)
        )
        contrast_rows.append(
            {"contrast": "MouseV2 minus Allen BO", "preferred_tf_hz": tf, "mouse_sessions": len(mouse), "allen_sessions": allen.session_id.nunique(), "difference_log10_steady_f1_f0": observed, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high}
        )
    mouse = sessions.loc[
        sessions.dataset.eq("MouseV2") & sessions.cohort.eq("FC_TF2_support")
        & sessions.steady_valid_units.ge(5), "mean_log10_steady_f1_f0"
    ].dropna().to_numpy()
    allen = sessions.loc[
        sessions.dataset.eq("Allen") & sessions.cohort.eq("Allen Functional Connectivity")
        & sessions.preferred_tf_hz.eq(2.0) & sessions.steady_valid_units.ge(5)
    ]
    observed, low, high = bootstrap_difference(
        mouse, allen, "mean_log10_steady_f1_f0", draws=args.bootstrap_draws, seed=20260830
    )
    contrast_rows.append(
        {"contrast": "MouseV2 minus Allen FC", "preferred_tf_hz": 2.0, "mouse_sessions": len(mouse), "allen_sessions": allen.session_id.nunique(), "difference_log10_steady_f1_f0": observed, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high}
    )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "steady_state_contrasts.csv", index=False)

    history_unit_rows = []
    for (session_id, unit_id), group in history_trials.groupby(["session_id", "unit_id"]):
        row = {"session_id": session_id, "unit_id": unit_id}
        for metric in ("full_f1_f0", "steady_f1_f0", "full_spikes", "steady_spikes"):
            low = group.loc[np.isclose(group.previous_contrast, .1), metric]
            high = group.loc[np.isclose(group.previous_contrast, .8), metric]
            row[f"{metric}_after_0p1"] = low.mean()
            row[f"{metric}_after_0p8"] = high.mean()
            if "f1_f0" in metric and low.mean() > 0 and high.mean() > 0:
                row[f"log10_{metric}_after_0p1_minus_0p8"] = np.log10(low.mean()) - np.log10(high.mean())
            else:
                row[f"{metric}_after_0p1_minus_0p8"] = low.mean() - high.mean()
            row[f"{metric}_trials_after_0p1"] = low.notna().sum()
            row[f"{metric}_trials_after_0p8"] = high.notna().sum()
        history_unit_rows.append(row)
    history_units = pd.DataFrame(history_unit_rows)
    history_units.to_csv(output / "fc_preceding_contrast_unit_effects.csv", index=False)
    history_sessions = (
        history_units.groupby("session_id")
        .agg(
            units=("unit_id", "size"),
            full_effect=("log10_full_f1_f0_after_0p1_minus_0p8", "mean"),
            steady_effect=("log10_steady_f1_f0_after_0p1_minus_0p8", "mean"),
            full_spike_effect=("full_spikes_after_0p1_minus_0p8", "mean"),
            steady_spike_effect=("steady_spikes_after_0p1_minus_0p8", "mean"),
        )
        .reset_index()
    )
    history_sessions.to_csv(output / "fc_preceding_contrast_session_effects.csv", index=False)

    # Auditable outcome-based roles for follow-up trial traces.
    effect = "log10_steady_f1_f0_after_0p1_minus_0p8"
    finite = history_units.dropna(subset=[effect]).copy()
    median = finite[effect].median()
    selected_indices = {
        "largest_negative_history_effect": finite[effect].idxmin(),
        "typical_history_effect": (finite[effect] - median).abs().idxmin(),
        "largest_positive_history_effect": finite[effect].idxmax(),
    }
    cases = []
    for role, index in selected_indices.items():
        row = finite.loc[index].to_dict()
        row["selection_role"] = role
        row["selection_basis"] = "post-hoc outcome-based minimum, nearest median, or maximum steady-state history effect"
        cases.append(row)
    cases = pd.DataFrame(cases)
    cases.to_csv(output / "history_case_selection.csv", index=False)
    selected_keys = set(zip(cases.session_id.astype(int), cases.unit_id.astype(int)))
    history_trials.loc[
        [(int(row.session_id), int(row.unit_id)) in selected_keys for _, row in history_trials.iterrows()]
    ].to_csv(output / "history_selected_case_trials.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for label, group in contrasts.groupby("contrast"):
        axes[0].plot(group.preferred_tf_hz, group.difference_log10_steady_f1_f0, "o-", label=label)
        axes[0].fill_between(group.preferred_tf_hz, group.bootstrap_95ci_low, group.bootstrap_95ci_high, alpha=.15)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(xlabel="preferred TF (Hz)", ylabel="MouseV2 minus Allen steady-state log10 F1/F0", title="Post-500-ms complete-cycle contrast")
    axes[0].legend(frameon=False)
    axes[1].scatter(history_sessions.full_effect, history_sessions.steady_effect, alpha=.7)
    axes[1].axhline(0, color="black", lw=.8); axes[1].axvline(0, color="black", lw=.8)
    axes[1].set(xlabel="preceding 0.1 minus 0.8: full-window F1/F0", ylabel="preceding 0.1 minus 0.8: steady-state F1/F0", title="Allen FC within-session history")
    fig.tight_layout(); fig.savefig(output / "steady_state_history_diagnostics.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    manifest = {
        "status": "exploratory_steady_state_and_history_checkpoint",
        "steady_window_rule": "last floor(0.5*TF) complete cycles ending at 1.0 s; unsupported at 1 Hz",
        "preference_rule": "freeze preferred condition selected by full 1-s mean spike count",
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), ALLEN_INVENTORY, ALLEN_CONFIG, MOUSE_CONFIG, RELEASE, FULL_BRIDGE / "harmonized_session_draws.csv")
        ],
        "outputs": [],
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote steady-state/history checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
