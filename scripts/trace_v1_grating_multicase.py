#!/usr/bin/env python3
"""Trace an auditable multi-case sample for the systemic V1 grating gap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    phase_aware_conditions,
    phase_schedule,
)
from scripts.trace_v1_systemic_gap_cases import (  # noqa: E402
    MOUSE_CONFIG,
    MOUSE_MANIFEST,
    MOUSE_UNITS,
    STIMULUS_MANIFEST,
    read_targeted_nwb,
    sha256,
    trace_mouse_case,
)


DEFAULT_OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/02_grating_multicase"
TF_ORDER = (1.0, 2.0, 4.0, 8.0, 15.0)
VIEWS = (
    "raw",
    "carrier_only_source_phase_alignment",
    "full_trial_source_phase_alignment",
    "alternating_plus_minus_one_frame",
)
AUDIT_PER_TF = 10
HASH_SALT = "v1-systemic-gap-multicase-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-per-tf", type=int, default=AUDIT_PER_TF)
    return parser.parse_args()


def deterministic_rank(unit_id: int) -> str:
    return hashlib.sha256(f"{HASH_SALT}|{int(unit_id)}".encode()).hexdigest()


def select_cases(units: pd.DataFrame, audit_per_tf: int) -> pd.DataFrame:
    """Outcome-blind audit sample plus explicitly outcome-selected stress cases."""
    eligible = units.loc[
        units["preferred_tf_hz"].isin(TF_ORDER)
        & units["log10_raw_mod_idx"].notna()
        & units["log10_source_corrected_mod_idx"].notna()
    ].copy()
    records: dict[int, dict[str, object]] = {}

    def add(row: pd.Series, role: str, basis: str, value: object, timing: str) -> None:
        unit_id = int(row.unit_id)
        if unit_id not in records:
            records[unit_id] = {
                "site": str(row.site),
                "session_id": int(row.session_id),
                "unit_id": unit_id,
                "preferred_tf_hz": float(row.preferred_tf_hz),
                "selection_roles": [],
                "selection_bases": [],
                "criterion_values": [],
                "selection_timings": [],
            }
        records[unit_id]["selection_roles"].append(role)
        records[unit_id]["selection_bases"].append(basis)
        records[unit_id]["criterion_values"].append(str(value))
        records[unit_id]["selection_timings"].append(timing)

    for tf_hz, group in eligible.groupby("preferred_tf_hz", sort=True):
        ranked = group.assign(
            deterministic_hash=group.unit_id.astype(int).map(deterministic_rank)
        ).sort_values(["deterministic_hash", "unit_id"])
        for _, row in ranked.head(audit_per_tf).iterrows():
            add(
                row,
                "outcome_blind_audit_sample",
                f"lowest salted SHA-256 ranks within {tf_hz:g} Hz",
                deterministic_rank(int(row.unit_id)),
                "predeclared_outcome_blind",
            )

        if tf_hz not in (1.0, 2.0, 15.0):
            continue
        role_specs = [
            (
                "largest_carrier_only_gain",
                "source_log10_mod_idx_gain",
                "max",
            ),
            (
                "most_negative_carrier_only_gain",
                "source_log10_mod_idx_gain",
                "min",
            ),
            (
                "largest_abs_denominator_change",
                "source_log10_psd_sd_gain",
                "absmax",
            ),
        ]
        for role, field, operation in role_specs:
            values = pd.to_numeric(group[field], errors="coerce")
            if operation == "max":
                index = values.idxmax()
            elif operation == "min":
                index = values.idxmin()
            else:
                index = values.abs().idxmax()
            row = group.loc[index]
            add(
                row,
                role,
                f"{operation} {field} within {tf_hz:g} Hz",
                float(row[field]),
                "post_hoc_stress_case",
            )

    rows = []
    for record in records.values():
        rows.append(
            {
                **{key: value for key, value in record.items() if not isinstance(value, list)},
                "selection_roles": ";".join(record["selection_roles"]),
                "selection_bases": ";".join(record["selection_bases"]),
                "criterion_values": ";".join(record["criterion_values"]),
                "selection_timings": ";".join(record["selection_timings"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["preferred_tf_hz", "site", "unit_id"]
    ).reset_index(drop=True)


def build_case_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["unit_id", "view"])
    rows = []
    for unit_id in metrics.unit_id.drop_duplicates():
        raw = indexed.loc[(unit_id, "raw")]
        carrier = indexed.loc[(unit_id, "carrier_only_source_phase_alignment")]
        full = indexed.loc[(unit_id, "full_trial_source_phase_alignment")]
        frame = indexed.loc[(unit_id, "alternating_plus_minus_one_frame")]
        rows.append(
            {
                "unit_id": int(unit_id),
                "site": raw.site,
                "session_id": int(raw.session_id),
                "preferred_tf_hz": float(raw.preferred_tf_hz),
                "selection_role": raw.selection_role,
                "raw_log10_mod_idx": np.log10(raw.mod_idx),
                "carrier_only_log10_gain": np.log10(carrier.mod_idx) - np.log10(raw.mod_idx),
                "full_trial_log10_gain": np.log10(full.mod_idx) - np.log10(raw.mod_idx),
                "frame_pattern_log10_gain": np.log10(frame.mod_idx) - np.log10(raw.mod_idx),
                "full_minus_carrier_log10_gain": np.log10(full.mod_idx) - np.log10(carrier.mod_idx),
                "carrier_exact_amplitude": carrier.coherent_carrier_amplitude,
                "full_exact_amplitude": full.coherent_carrier_amplitude,
                "exact_amplitude_abs_difference": abs(
                    carrier.coherent_carrier_amplitude - full.coherent_carrier_amplitude
                ),
                "carrier_target_psd": carrier.target_psd,
                "full_target_psd": full.target_psd,
                "carrier_psd_sd": carrier.psd_sd,
                "full_psd_sd": full.psd_sd,
                "raw_f1_f0": raw.f1_f0,
                "full_f1_f0": full.f1_f0,
            }
        )
    return pd.DataFrame(rows)


def summarize(comparison: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    audit_ids = set(
        cases.loc[
            cases.selection_roles.str.contains("outcome_blind_audit_sample"), "unit_id"
        ].astype(int)
    )
    audit = comparison.loc[comparison.unit_id.isin(audit_ids)].copy()
    rows = []
    for tf_hz, group in audit.groupby("preferred_tf_hz", sort=True):
        difference = group.full_minus_carrier_log10_gain
        rows.append(
            {
                "preferred_tf_hz": tf_hz,
                "audit_units": len(group),
                "median_carrier_only_gain": group.carrier_only_log10_gain.median(),
                "median_full_trial_gain": group.full_trial_log10_gain.median(),
                "median_full_minus_carrier": difference.median(),
                "fraction_same_gain_sign": np.mean(
                    np.sign(group.carrier_only_log10_gain)
                    == np.sign(group.full_trial_log10_gain)
                ),
                "fraction_abs_disagreement_gt_0_1": np.mean(difference.abs() > 0.1),
                "median_frame_pattern_gain": group.frame_pattern_log10_gain.median(),
                "max_exact_amplitude_abs_difference": group.exact_amplitude_abs_difference.max(),
            }
        )
    return pd.DataFrame(rows)


def render(comparison: pd.DataFrame, cases: pd.DataFrame, output: Path) -> None:
    audit_ids = set(
        cases.loc[
            cases.selection_roles.str.contains("outcome_blind_audit_sample"), "unit_id"
        ].astype(int)
    )
    comparison = comparison.copy()
    comparison["audit"] = comparison.unit_id.isin(audit_ids)
    colors = {1.0: "#4C78A8", 2.0: "#F58518", 4.0: "#54A24B", 8.0: "#B279A2", 15.0: "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for tf_hz, group in comparison.groupby("preferred_tf_hz", sort=True):
        axes[0].scatter(
            group.carrier_only_log10_gain,
            group.full_trial_log10_gain,
            s=np.where(group.audit, 30, 75),
            facecolors=np.where(group.audit, colors[tf_hz], "none"),
            edgecolors=colors[tf_hz],
            alpha=0.8,
            label=f"{tf_hz:g} Hz",
        )
    limits = [-4.2, 4.2]
    axes[0].plot(limits, limits, color="black", lw=1, ls="--")
    axes[0].axhline(0, color="#999999", lw=0.7)
    axes[0].axvline(0, color="#999999", lw=0.7)
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="carrier-only log10 gain",
        ylabel="whole-trial log10 gain",
        title="Same carrier coefficient, different Welch response",
    )
    axes[0].legend(frameon=False, fontsize=8)

    audit = comparison.loc[comparison.audit]
    positions = np.arange(len(TF_ORDER))
    for position, tf_hz in zip(positions, TF_ORDER):
        group = audit.loc[audit.preferred_tf_hz.eq(tf_hz)]
        axes[1].scatter(
            np.full(len(group), position) - 0.13,
            group.carrier_only_log10_gain,
            color=colors[tf_hz], s=22, alpha=0.65,
        )
        axes[1].scatter(
            np.full(len(group), position) + 0.13,
            group.full_trial_log10_gain,
            facecolors="none", edgecolors=colors[tf_hz], s=25, alpha=0.8,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(positions, [f"{tf:g}" for tf in TF_ORDER])
    axes[1].set(xlabel="preferred temporal frequency (Hz)", ylabel="log10 gain from raw", title="Outcome-blind audit cases")

    for position, tf_hz in zip(positions, TF_ORDER):
        values = audit.loc[
            audit.preferred_tf_hz.eq(tf_hz), "frame_pattern_log10_gain"
        ]
        axes[2].scatter(
            np.full(len(values), position), values, color=colors[tf_hz], s=24, alpha=0.7
        )
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(positions, [f"{tf:g}" for tf in TF_ORDER])
    axes[2].set(xlabel="preferred temporal frequency (Hz)", ylabel="log10 gain from raw", title="Alternating ±one-frame sensitivity")
    fig.suptitle("Systemic V1 grating audit: multi-case alignment comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(output: Path, cases: pd.DataFrame, audit_per_tf: int) -> None:
    upstream = json.loads(MOUSE_MANIFEST.read_text())
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/trace_v1_systemic_gap_cases.py",
        ROOT / "scripts/mousev2_grating_start_phase_bridge.py",
        ROOT / "scripts/mousev2_grating_corrected_welch_bridge.py",
        ROOT / "common/drifting_gratings.py",
        MOUSE_CONFIG,
        STIMULUS_MANIFEST,
        MOUSE_UNITS,
    ]
    manifest = {
        "schema_version": 1,
        "status": "exploratory_multicase_checkpoint",
        "audit_sample": {
            "method": "lowest salted SHA-256 unit ranks within temporal frequency",
            "salt": HASH_SALT,
            "units_per_temporal_frequency": audit_per_tf,
            "outcome_blind": True,
        },
        "stress_cases": [
            "largest carrier-only gain within affected TF",
            "most negative carrier-only gain within affected TF",
            "largest absolute Welch-denominator change within affected TF",
        ],
        "case_count": len(cases),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "inputs": upstream["inputs"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def refresh_outputs(output: Path) -> None:
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    names = [
        "case_selection.csv",
        "case_metric_trace.csv",
        "case_comparison.csv",
        "outcome_blind_tf_summary.csv",
        "case_psth_trace.csv",
        "mouse_trial_trace.csv",
        "multicase_alignment_comparison.png",
        "MULTICASE_CHECKPOINT.md",
    ]
    manifest["outputs"] = [
        {"path": name, "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
        for name in names if (output / name).is_file()
    ]
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.audit_per_tf < 1:
        raise ValueError("--audit-per-tf must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(MOUSE_UNITS)
    cases = select_cases(units, args.audit_per_tf)
    cases.to_csv(output / "case_selection.csv", index=False)
    write_manifest(output, cases, args.audit_per_tf)

    config = json.loads(MOUSE_CONFIG.read_text())
    by_site = {str(record["site"]): record for record in config["sessions"]}
    schedule = phase_schedule(STIMULUS_MANIFEST)
    metric_rows = []
    trial_frames = []
    psth_frames = []
    for site, group in cases.groupby("site", sort=True):
        session = by_site[site]
        offset = int(session["id_offset"])
        local_ids = [int(unit_id) - offset for unit_id in group.unit_id]
        nwb_path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        extracted = read_targeted_nwb(
            nwb_path,
            unit_ids=local_ids,
            interval_name="drifting_gratings_field_block_presentations",
        )
        conditions = phase_aware_conditions(extracted.interval_table, schedule)
        for _, selected in group.iterrows():
            case = pd.Series(
                {
                    "dataset": "MouseV2",
                    "session_id": selected.session_id,
                    "site": site,
                    "unit_id": selected.unit_id,
                    "selection_role": selected.selection_roles,
                }
            )
            local_id = int(selected.unit_id) - offset
            rows, trials, psths = trace_mouse_case(
                case, extracted.spikes_by_id[local_id], conditions
            )
            metric_rows.extend(rows)
            trial_frames.append(trials)
            psth_frames.append(psths)

    metrics = pd.DataFrame(metric_rows)
    comparison = build_case_comparison(metrics)
    summary = summarize(comparison, cases)
    trials = pd.concat(trial_frames, ignore_index=True)
    psths = pd.concat(psth_frames, ignore_index=True)
    metrics.to_csv(output / "case_metric_trace.csv", index=False)
    comparison.to_csv(output / "case_comparison.csv", index=False)
    summary.to_csv(output / "outcome_blind_tf_summary.csv", index=False)
    trials.to_csv(output / "mouse_trial_trace.csv", index=False)
    psths.to_csv(output / "case_psth_trace.csv", index=False)
    render(comparison, cases, output / "multicase_alignment_comparison.png")
    refresh_outputs(output)
    print(f"Wrote grating multi-case checkpoint to {output}")


if __name__ == "__main__":
    main()
