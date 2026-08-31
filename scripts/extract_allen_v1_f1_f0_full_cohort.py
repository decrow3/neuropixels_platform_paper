#!/usr/bin/env python3
"""Full-cohort Allen V1 F1/F0 on MouseV2's shared 1-s/15-trial support."""

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

from scripts.extract_allen_v1_bridge import (  # noqa: E402
    COHORT_LABELS,
    common_qc,
    condition_presentations,
    condition_starts,
    released_metrics,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


INVENTORY = Path("/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/session_inventory.json")
CONFIG = ROOT / "config/allen_v1_bridge.json"
RELEASE = ROOT / "data/unit_table.csv"
MOUSE_COMMON = ROOT / "data/imports/mousev2_grating_common_support_v1/unit_metric_comparison.csv"
OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/04_allen_full_cohort_harmonized"
VALIDATION_SESSIONS = {"brain_observatory_1.1": 750749662, "functional_connectivity": 774875821}
F1_ATOL = 3e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_f1_f0_components(
    spikes_s: np.ndarray,
    starts_s: np.ndarray,
    temporal_frequency_hz: float,
    *,
    duration_ms: int = 1000,
    trial_duration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trial spike counts and exact cycle-fold F1/F0 ratios.

    The ratio is evaluated directly from occupied 1-ms bins. It is algebraically
    identical to folding the binned trial into cycles and taking the first FFT
    harmonic, while avoiding a dense trials-by-time matrix for every unit.
    """
    starts = np.asarray(starts_s, dtype=float)
    spikes = np.asarray(spikes_s, dtype=float)
    cycles = int(float(temporal_frequency_hz) * float(trial_duration_s))
    bins_per_cycle = int(duration_ms / cycles)
    used_ms = cycles * bins_per_cycle
    counts = np.empty(len(starts), dtype=np.int32)
    ratios = np.full(len(starts), np.nan, dtype=float)
    phase_lookup = np.exp(-2j * np.pi * np.arange(bins_per_cycle) / bins_per_cycle)
    for index, start in enumerate(starts):
        first = np.searchsorted(spikes, start, side="left")
        last = np.searchsorted(spikes, start + duration_ms / 1000.0, side="left")
        counts[index] = last - first
        used_last = np.searchsorted(spikes, start + used_ms / 1000.0, side="left")
        trial_spikes = spikes[first:used_last]
        if len(trial_spikes) == 0:
            continue
        bins = np.floor((trial_spikes - start) * 1000.0 + 1e-7).astype(int)
        bins = bins[(bins >= 0) & (bins < used_ms)]
        if len(bins) == 0:
            continue
        ratios[index] = 2.0 * np.abs(phase_lookup[bins % bins_per_cycle].sum()) / len(bins)
    return counts, ratios


def prepare_unit(
    spikes: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
) -> list[dict[str, object]]:
    prepared = []
    for parameters, starts in conditions:
        counts, ratios = trial_f1_f0_components(spikes, starts, parameters[1])
        prepared.append({"parameters": parameters, "counts": counts, "ratios": ratios})
    return prepared


def summarize_draw(
    prepared: list[dict[str, object]], indices_by_condition: list[np.ndarray]
) -> dict[str, object]:
    means = [
        float(np.mean(condition["counts"][indices]))
        for condition, indices in zip(prepared, indices_by_condition)
    ]
    preferred = int(np.argmax(means))
    condition = prepared[preferred]
    indices = indices_by_condition[preferred]
    ratios = condition["ratios"][indices]
    valid = ratios[np.isfinite(ratios)]
    return {
        "preferred_condition_index": preferred,
        "preferred_orientation_deg": float(condition["parameters"][0]),
        "preferred_tf_hz": float(condition["parameters"][1]),
        "f1_f0": float(valid.mean()) if len(valid) else np.nan,
        "preferred_mean_spikes": float(np.mean(condition["counts"][indices])),
        "selected_trials": int(len(indices)),
        "valid_trials": int(len(valid)),
        "zero_f0_trials": int(len(indices) - len(valid)),
    }


def draw_indices(
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
    *,
    session_id: int,
    draws: int,
) -> list[list[np.ndarray]]:
    repeats = min(len(starts) for _, starts in conditions)
    if repeats < 14:
        raise ValueError(f"Session {session_id} has fewer than 14 shared trials")
    if repeats <= 15:
        return [[np.arange(len(starts)) for _, starts in conditions]]
    result = []
    for draw in range(draws):
        rng = np.random.default_rng(session_id + draw)
        result.append(
            [np.sort(rng.choice(len(starts), 15, replace=False)) for _, starts in conditions]
        )
    return result


def interval_name(session_type: str) -> str:
    return (
        "drifting_gratings_presentations"
        if session_type == "brain_observatory_1.1"
        else "drifting_gratings_75_repeats_presentations"
    )


def bootstrap_contrasts(
    session_draws: pd.DataFrame,
    mouse_sessions: pd.Series,
    *,
    draws: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260825)
    mouse_values = mouse_sessions.to_numpy()
    for session_type, cohort in COHORT_LABELS.items():
        cohort_draws = session_draws.loc[session_draws.session_type.eq(session_type)]
        scopes = [("full_cohort", cohort_draws)]
        if session_type == "brain_observatory_1.1":
            scopes.append(
                ("complete_15_trial_sessions", cohort_draws.loc[cohort_draws.shared_min_repeats.eq(15)])
            )
        for scope, subset in scopes:
            session_ids = subset.session_id.unique()
            centers = subset.groupby("session_id").mean_log10_f1_f0.median()
            observed = float(mouse_values.mean() - centers.mean())
            boot = np.empty(draws)
            for draw in range(draws):
                mouse_sample = rng.choice(mouse_values, len(mouse_values), replace=True)
                selected_sessions = rng.choice(session_ids, len(session_ids), replace=True)
                allen_sample = []
                for session_id in selected_sessions:
                    values = subset.loc[subset.session_id.eq(session_id), "mean_log10_f1_f0"].dropna().to_numpy()
                    allen_sample.append(rng.choice(values))
                boot[draw] = mouse_sample.mean() - np.mean(allen_sample)
            rows.append(
                {
                    "scope": scope,
                    "contrast": f"MouseV2 minus {cohort}",
                    "mouse_sessions": len(mouse_values),
                    "allen_sessions": len(session_ids),
                    "mouse_center_mean_log10_f1_f0": mouse_values.mean(),
                    "allen_center_mean_log10_f1_f0": centers.mean(),
                    "difference_log10_f1_f0": observed,
                    "bootstrap_95ci_low": np.quantile(boot, .025),
                    "bootstrap_95ci_high": np.quantile(boot, .975),
                    "geometric_mean_ratio": 10**observed,
                }
            )
    return pd.DataFrame(rows)


def render(session_summary: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    colors = {"brain_observatory_1.1": "#6F63A6", "functional_connectivity": "#B07AA1"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    rng = np.random.default_rng(20260825)
    for position, session_type in enumerate(("brain_observatory_1.1", "functional_connectivity")):
        group = session_summary.loc[session_summary.session_type.eq(session_type)]
        axes[0].scatter(
            position + rng.uniform(-.1, .1, len(group)), group.harmonized_median_log10_f1_f0,
            color=colors[session_type], alpha=.7, s=28,
        )
        axes[0].plot([position-.18, position+.18], [group.harmonized_median_log10_f1_f0.mean()]*2, color="black", lw=2)
        for _, row in group.iterrows():
            axes[1].scatter(row.native_mean_log10_f1_f0, row.harmonized_median_log10_f1_f0, color=colors[session_type], alpha=.7, s=28)
    mouse_center = float(contrasts.loc[contrasts.scope.eq("full_cohort"), "mouse_center_mean_log10_f1_f0"].iloc[0])
    axes[0].axhline(mouse_center, color="#D95F02", linestyle="--", label="MouseV2 center")
    axes[0].set_xticks((0, 1), ("Allen BO", "Allen FC"))
    axes[0].set(ylabel="harmonized session mean log10 F1/F0", title="Full Allen V1 cohort")
    axes[0].legend(frameon=False)
    axes[1].plot([-.4, .05], [-.4, .05], "k--", lw=1)
    axes[1].set(xlabel="native 2-s session mean", ylabel="harmonized 1-s session mean", title="Window/condition effect by session")
    fig.suptitle("Allen V1 conventional F1/F0 on shared 1-s / 15-trial support")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--trial-draws", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(INVENTORY.read_text())
    inventory_by_id = {int(row["ecephys_session_id"]): row for row in inventory}
    release = pd.read_csv(RELEASE, low_memory=False)
    relevant = release.loc[
        release.ecephys_structure_acronym.eq("VISp")
        & release.session_type.isin(COHORT_LABELS)
    ].copy()
    relevant = relevant.loc[common_qc(relevant)]
    session_ids = sorted(relevant.ecephys_session_id.astype(int).unique())

    audit_rows = []
    for row in inventory:
        session_id = int(row["ecephys_session_id"])
        path = Path(row["nwb_path"])
        selected = relevant.loc[relevant.ecephys_session_id.eq(session_id)]
        audit_rows.append(
            {
                "session_id": session_id,
                "session_type": row["session_type"],
                "included": session_id in session_ids,
                "exclusion_reason": "" if session_id in session_ids else "no released common-QC VISp population",
                "common_qc_visp_units": len(selected),
                "nwb_path": str(path),
                "file_exists": path.is_file(),
                "expected_bytes": row["expected_bytes"],
                "observed_bytes": path.stat().st_size if path.is_file() else np.nan,
                "inventory_status": row["status"],
            }
        )
    audit = pd.DataFrame(audit_rows)
    if len(session_ids) != 56 or not audit.loc[audit.included, "file_exists"].all():
        raise AssertionError("Expected all 56 released common-QC VISp sessions locally")
    audit.to_csv(output / "session_inventory_audit.csv", index=False)

    config = json.loads(CONFIG.read_text())
    support = config["common_support"]
    validation_rows = []
    unit_rows = []
    session_rows = []
    for number, session_id in enumerate(session_ids, start=1):
        selected = relevant.loc[relevant.ecephys_session_id.eq(session_id)].copy()
        session_type = str(selected.session_type.iloc[0])
        path = Path(inventory_by_id[session_id]["nwb_path"])
        unit_ids = selected.ecephys_unit_id.astype(int).tolist()
        print(f"[{number:02d}/{len(session_ids)}] {session_id} {session_type}: {len(unit_ids)} units", flush=True)
        extracted = read_targeted_nwb(path, unit_ids=unit_ids, interval_name=interval_name(session_type))
        conditions = condition_starts(extracted.interval_table, common_support=support)
        expected_conditions = 20 if session_type == "brain_observatory_1.1" else 4
        if len(conditions) != expected_conditions:
            raise AssertionError(f"{session_id}: expected {expected_conditions} shared conditions, got {len(conditions)}")
        indices = draw_indices(conditions, session_id=session_id, draws=args.trial_draws)
        shared_repeats = [len(starts) for _, starts in conditions]

        if session_id == VALIDATION_SESSIONS[session_type]:
            released_conditions = condition_presentations(extracted.interval_table)
            by_id = selected.set_index("ecephys_unit_id")
            for unit_id in unit_ids:
                raw = released_metrics(extracted.spikes_by_id[unit_id], released_conditions)
                released_value = float(by_id.loc[unit_id, "f1_f0_dg"])
                error = abs(raw["raw_released_f1_f0_dg"] - released_value)
                validation_rows.append(
                    {"session_id": session_id, "session_type": session_type, "ecephys_unit_id": unit_id, "released_f1_f0": released_value, "raw_f1_f0": raw["raw_released_f1_f0_dg"], "absolute_error": error}
                )
                if not error <= F1_ATOL:
                    raise AssertionError(f"Native validation failed for {unit_id}: {error}")

        session_unit_rows = []
        native_values = np.log10(pd.to_numeric(selected.f1_f0_dg, errors="coerce").where(selected.f1_f0_dg.gt(0)))
        for unit_id in unit_ids:
            prepared = prepare_unit(extracted.spikes_by_id[unit_id], conditions)
            full_indices = [np.arange(len(starts)) for _, starts in conditions]
            full = summarize_draw(prepared, full_indices)
            for draw, chosen in enumerate(indices):
                result = summarize_draw(prepared, chosen)
                row = {
                    "session_id": session_id,
                    "session_type": session_type,
                    "ecephys_unit_id": unit_id,
                    "draw": draw,
                    **result,
                    "full_support_preferred_condition_index": full["preferred_condition_index"],
                    "preference_switched": result["preferred_condition_index"] != full["preferred_condition_index"],
                }
                unit_rows.append(row)
                session_unit_rows.append(row)
        frame = pd.DataFrame(session_unit_rows)
        for draw, group in frame.groupby("draw"):
            values = np.log10(pd.to_numeric(group.f1_f0, errors="coerce").where(group.f1_f0.gt(0)))
            session_rows.append(
                {
                    "session_id": session_id,
                    "session_type": session_type,
                    "draw": draw,
                    "valid_units": values.notna().sum(),
                    "mean_log10_f1_f0": values.mean(),
                    "preference_switch_fraction": group.preference_switched.mean(),
                    "native_mean_log10_f1_f0": native_values.mean(),
                    "shared_min_repeats": min(shared_repeats),
                    "shared_max_repeats": max(shared_repeats),
                }
            )

    validation = pd.DataFrame(validation_rows)
    unit_draws = pd.DataFrame(unit_rows)
    session_draws = pd.DataFrame(session_rows)
    validation.to_csv(output / "native_validation_units.csv", index=False)
    unit_draws.to_csv(output / "harmonized_unit_draws.csv", index=False)
    session_draws.to_csv(output / "harmonized_session_draws.csv", index=False)
    session_summary = (
        session_draws.groupby(["session_id", "session_type"])
        .agg(
            draws=("draw", "size"),
            valid_units_median=("valid_units", "median"),
            native_mean_log10_f1_f0=("native_mean_log10_f1_f0", "first"),
            harmonized_median_log10_f1_f0=("mean_log10_f1_f0", "median"),
            harmonized_p025=("mean_log10_f1_f0", lambda x: x.quantile(.025)),
            harmonized_p975=("mean_log10_f1_f0", lambda x: x.quantile(.975)),
            preference_switch_fraction_median=("preference_switch_fraction", "median"),
            shared_min_repeats=("shared_min_repeats", "first"),
            shared_max_repeats=("shared_max_repeats", "first"),
        )
        .reset_index()
    )
    session_summary["harmonized_minus_native"] = session_summary.harmonized_median_log10_f1_f0 - session_summary.native_mean_log10_f1_f0
    session_summary.to_csv(output / "harmonized_session_summary.csv", index=False)

    mouse = pd.read_csv(MOUSE_COMMON)
    mouse = mouse.loc[mouse.default_qc & mouse.f1_f0_dg_common_support.gt(0)].copy()
    mouse["log10_f1_f0"] = np.log10(mouse.f1_f0_dg_common_support)
    mouse_sessions = mouse.groupby("site_number").log10_f1_f0.mean()
    contrasts = bootstrap_contrasts(session_draws, mouse_sessions, draws=args.bootstrap_draws)
    contrasts.to_csv(output / "full_cohort_contrasts.csv", index=False)
    render(session_summary, contrasts, output / "allen_full_cohort_harmonized.png")

    manifest = {
        "status": "full_cohort_harmonized_f1_f0",
        "shared_support": support,
        "included_sessions": len(session_ids),
        "validation_sessions": VALIDATION_SESSIONS,
        "trial_draws": args.trial_draws,
        "bootstrap_draws": args.bootstrap_draws,
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), INVENTORY, CONFIG, RELEASE, MOUSE_COMMON)
        ],
        "outputs": [],
    }
    for name in (
        "session_inventory_audit.csv", "native_validation_units.csv", "harmonized_unit_draws.csv",
        "harmonized_session_draws.csv", "harmonized_session_summary.csv", "full_cohort_contrasts.csv",
        "allen_full_cohort_harmonized.png", "FULL_COHORT_CHECKPOINT.md",
    ):
        path = output / name
        if path.is_file():
            manifest["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote full-cohort harmonized bridge to {output}", flush=True)


if __name__ == "__main__":
    main()
