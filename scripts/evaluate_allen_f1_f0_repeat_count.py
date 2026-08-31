#!/usr/bin/env python3
"""Compare original Allen FC F1/F0 with 75 versus 15 repeats."""

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

from common.drifting_gratings import _bin_trial_spike_counts  # noqa: E402
from scripts.extract_allen_v1_bridge import (  # noqa: E402
    common_qc,
    condition_presentations,
    sha256,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


CONFIG = ROOT / "config/allen_v1_bridge.json"
DEFAULT_OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/02_allen_repeat_count"
DRAWS = 100
TRIALS = 15
F1_ATOL = 3e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=DRAWS)
    return parser.parse_args()


def allen_trial_components(counts: np.ndarray, tf_hz: float) -> pd.DataFrame:
    values = np.asarray(counts)
    trial_duration = 1.9985
    cycles = int(tf_hz * trial_duration)
    bins_per_cycle = int(values.shape[1] / cycles)
    used = cycles * bins_per_cycle
    folded = values[:, :used].reshape(len(values), cycles, bins_per_cycle)
    average_cycle = np.mean(folded, axis=1)
    amplitude = 2.0 * np.abs(np.fft.fft(average_cycle, axis=1)) / bins_per_cycle
    f0_hz = 0.5 * amplitude[:, 0] * 1000.0
    f1_hz = amplitude[:, 1] * 1000.0
    ratio = np.divide(f1_hz, f0_hz, out=np.full(len(values), np.nan), where=f0_hz > 0)
    return pd.DataFrame({"f0_hz": f0_hz, "f1_hz": f1_hz, "f1_f0": ratio})


def prepare_unit(
    spikes: np.ndarray,
    conditions: list[tuple[tuple[float, ...], np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    prepared = []
    for parameters, starts, stops in conditions:
        selection_counts = (
            np.searchsorted(spikes, stops, side="left")
            - np.searchsorted(spikes, starts, side="left")
        )
        metric_counts = _bin_trial_spike_counts(spikes, starts, duration_ms=1999)
        components = allen_trial_components(metric_counts, float(parameters[1]))
        prepared.append(
            {
                "parameters": parameters,
                "selection_counts": selection_counts,
                "components": components,
            }
        )
    return prepared


def summarize_selection(
    prepared: list[dict[str, object]], indices_by_condition: list[np.ndarray]
) -> dict[str, object]:
    means = [
        float(np.mean(condition["selection_counts"][indices]))
        for condition, indices in zip(prepared, indices_by_condition)
    ]
    selected_index = int(np.argmax(means))
    return summarize_condition(prepared, indices_by_condition, selected_index)


def summarize_condition(
    prepared: list[dict[str, object]],
    indices_by_condition: list[np.ndarray],
    selected_index: int,
) -> dict[str, object]:
    selected = prepared[selected_index]
    indices = indices_by_condition[selected_index]
    components = selected["components"].iloc[indices]
    f1_f0 = float(components.f1_f0.mean())
    absolute_f1 = float(components.f1_hz.mean())
    return {
        "condition_index": selected_index,
        "orientation_deg": float(selected["parameters"][0]),
        "tf_hz": float(selected["parameters"][1]),
        "f1_f0": f1_f0,
        "absolute_f1_hz": absolute_f1,
        "mean_f0_hz": float(components.f0_hz.mean()),
        "zero_f0_trials": int(components.f0_hz.eq(0).sum()),
        "valid_ratio_trials": int(components.f1_f0.notna().sum()),
    }


def subsample_indices(
    conditions: list[tuple[tuple[float, ...], np.ndarray, np.ndarray]],
    *, draws: int,
    seed: int,
) -> list[list[np.ndarray]]:
    result = []
    for draw in range(draws):
        rng = np.random.default_rng(seed + draw)
        result.append(
            [
                np.sort(rng.choice(len(starts), TRIALS, replace=False))
                for _, starts, _ in conditions
            ]
        )
    return result


def render(units: pd.DataFrame, draws: pd.DataFrame, output: Path) -> None:
    qc = units.loc[units.common_qc].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for session_id, group in qc.groupby("session_id"):
        axes[0].scatter(group.full_f1_f0, group.median_15_f1_f0, s=25, alpha=0.65, label=str(session_id))
    limit = max(qc.full_f1_f0.max(), qc.median_15_f1_f0.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], "k--", lw=1)
    axes[0].set(xlabel="75-repeat F1/F0", ylabel="median 15-repeat F1/F0", title="Unit-level repeat sensitivity")
    axes[0].legend(frameon=False, fontsize=8)
    qc_draws = draws.loc[draws.population.eq("common_qc")]
    session_ids = sorted(qc_draws.session_id.unique())
    for position, session_id in enumerate(session_ids):
        group = qc_draws.loc[qc_draws.session_id.eq(session_id)]
        axes[1].scatter(
            np.full(len(group), position - 0.08), group.session_mean_log10_delta,
            alpha=0.45, s=18, color="#D55E00", label="reselect preference" if position == 0 else None,
        )
        axes[1].scatter(
            np.full(len(group), position + 0.08), group.fixed_session_mean_log10_delta,
            alpha=0.45, s=18, color="#0072B2", label="hold 75-repeat preference" if position == 0 else None,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(session_ids)), [str(value) for value in session_ids])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set(xlabel="session", ylabel="15 minus 75 mean log10 F1/F0", title="Session-center sensitivity")
    axes[2].scatter(qc.full_mean_f0_hz, qc.preference_switch_fraction, color="#6F63A6", alpha=0.7)
    axes[2].set(xscale="log", xlabel="75-repeat selected-condition F0 (Hz)", ylabel="fraction of 15-repeat draws switching condition", title="Selection stability versus response support")
    fig.suptitle("Allen Functional Connectivity: 75 versus 15 repeats")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.draws < 1:
        raise ValueError("--draws must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    release = pd.read_csv(ROOT / "data/unit_table.csv", low_memory=False)
    unit_draw_rows = []
    unit_rows = []
    input_records = []
    for asset in config["assets"]:
        if asset["session_type"] != "functional_connectivity":
            continue
        session_id = int(asset["session_id"])
        path = Path(config["download_root"]) / asset["relative_path"]
        selected = release.loc[
            release.ecephys_session_id.eq(session_id)
            & release.ecephys_structure_acronym.eq("VISp")
        ].copy()
        unit_ids = selected.ecephys_unit_id.astype(int).tolist()
        extracted = read_targeted_nwb(path, unit_ids=unit_ids, interval_name=asset["grating_table"])
        conditions = condition_presentations(extracted.interval_table)
        if len(conditions) != 8 or min(len(starts) for _, starts, _ in conditions) != 75:
            raise ValueError(f"Unexpected FC condition support in {session_id}")
        draws = subsample_indices(conditions, draws=args.draws, seed=session_id)
        full_indices = [np.arange(len(starts)) for _, starts, _ in conditions]
        released_by_id = selected.set_index("ecephys_unit_id")
        for unit_id in unit_ids:
            prepared = prepare_unit(extracted.spikes_by_id[unit_id], conditions)
            full = summarize_selection(prepared, full_indices)
            released_value = float(released_by_id.loc[unit_id, "f1_f0_dg"])
            error = abs(full["f1_f0"] - released_value) if np.isfinite(released_value) else np.nan
            if not (error <= F1_ATOL or (np.isnan(full["f1_f0"]) and np.isnan(released_value))):
                raise AssertionError(f"Full F1/F0 reproduction failed for {unit_id}: {error}")
            draw_rows = []
            for draw, indices in enumerate(draws):
                sampled = summarize_selection(prepared, indices)
                fixed = summarize_condition(prepared, indices, int(full["condition_index"]))
                row = {
                    "session_id": session_id,
                    "ecephys_unit_id": unit_id,
                    "common_qc": bool(common_qc(released_by_id.loc[[unit_id]]).iloc[0]),
                    "draw": draw,
                    **{f"sampled_{key}": value for key, value in sampled.items()},
                    **{f"fixed_{key}": value for key, value in fixed.items()},
                    "full_condition_index": full["condition_index"],
                    "full_f1_f0": full["f1_f0"],
                    "full_absolute_f1_hz": full["absolute_f1_hz"],
                    "full_mean_f0_hz": full["mean_f0_hz"],
                    "preference_switched": sampled["condition_index"] != full["condition_index"],
                    "log10_f1_f0_delta": (
                        np.log10(sampled["f1_f0"]) - np.log10(full["f1_f0"])
                        if sampled["f1_f0"] > 0 and full["f1_f0"] > 0 else np.nan
                    ),
                    "log10_absolute_f1_delta": (
                        np.log10(sampled["absolute_f1_hz"]) - np.log10(full["absolute_f1_hz"])
                        if sampled["absolute_f1_hz"] > 0 and full["absolute_f1_hz"] > 0 else np.nan
                    ),
                    "fixed_log10_f1_f0_delta": (
                        np.log10(fixed["f1_f0"]) - np.log10(full["f1_f0"])
                        if fixed["f1_f0"] > 0 and full["f1_f0"] > 0 else np.nan
                    ),
                    "fixed_log10_absolute_f1_delta": (
                        np.log10(fixed["absolute_f1_hz"]) - np.log10(full["absolute_f1_hz"])
                        if fixed["absolute_f1_hz"] > 0 and full["absolute_f1_hz"] > 0 else np.nan
                    ),
                }
                draw_rows.append(row)
                unit_draw_rows.append(row)
            draw_frame = pd.DataFrame(draw_rows)
            unit_rows.append(
                {
                    "session_id": session_id,
                    "ecephys_unit_id": unit_id,
                    "common_qc": draw_frame.common_qc.iloc[0],
                    "full_f1_f0": full["f1_f0"],
                    "full_absolute_f1_hz": full["absolute_f1_hz"],
                    "full_mean_f0_hz": full["mean_f0_hz"],
                    "full_zero_f0_trials": full["zero_f0_trials"],
                    "median_15_f1_f0": draw_frame.sampled_f1_f0.median(),
                    "median_log10_f1_f0_delta": draw_frame.log10_f1_f0_delta.median(),
                    "p025_log10_f1_f0_delta": draw_frame.log10_f1_f0_delta.quantile(.025),
                    "p975_log10_f1_f0_delta": draw_frame.log10_f1_f0_delta.quantile(.975),
                    "median_log10_absolute_f1_delta": draw_frame.log10_absolute_f1_delta.median(),
                    "median_fixed_log10_f1_f0_delta": draw_frame.fixed_log10_f1_f0_delta.median(),
                    "median_fixed_log10_absolute_f1_delta": draw_frame.fixed_log10_absolute_f1_delta.median(),
                    "preference_switch_fraction": draw_frame.preference_switched.mean(),
                    "mean_sampled_zero_f0_trials": draw_frame.sampled_zero_f0_trials.mean(),
                    "full_reproduction_abs_error": error,
                }
            )
        input_records.append({"session_id": session_id, "path": str(path), "bytes": path.stat().st_size, "sha256": asset["sha256"]})

    unit_draws = pd.DataFrame(unit_draw_rows)
    units = pd.DataFrame(unit_rows)
    session_draw_rows = []
    for population, mask in (("all", unit_draws.common_qc.notna()), ("common_qc", unit_draws.common_qc)):
        selected_draws = unit_draws.loc[mask]
        for (session_id, draw), group in selected_draws.groupby(["session_id", "draw"]):
            valid = group.loc[group.sampled_f1_f0.gt(0) & group.full_f1_f0.gt(0)]
            fixed_valid = group.loc[group.fixed_f1_f0.gt(0) & group.full_f1_f0.gt(0)]
            session_draw_rows.append(
                {
                    "population": population,
                    "session_id": int(session_id),
                    "draw": int(draw),
                    "units": len(group),
                    "valid_units": len(valid),
                    "full_mean_log10_f1_f0": np.log10(valid.full_f1_f0).mean(),
                    "sampled_mean_log10_f1_f0": np.log10(valid.sampled_f1_f0).mean(),
                    "session_mean_log10_delta": np.log10(valid.sampled_f1_f0).mean() - np.log10(valid.full_f1_f0).mean(),
                    "fixed_valid_units": len(fixed_valid),
                    "fixed_session_mean_log10_delta": np.log10(fixed_valid.fixed_f1_f0).mean() - np.log10(fixed_valid.full_f1_f0).mean(),
                    "preference_switch_fraction": group.preference_switched.mean(),
                }
            )
    session_draws = pd.DataFrame(session_draw_rows)
    unit_draws.to_csv(output / "unit_draw_comparison.csv", index=False)
    units.to_csv(output / "unit_repeat_summary.csv", index=False)
    session_draws.to_csv(output / "session_draw_summary.csv", index=False)
    render(units, session_draws, output / "allen_repeat_count_diagnostics.png")
    manifest = {
        "schema_version": 1,
        "status": "exploratory_repeat_count_checkpoint",
        "draws": args.draws,
        "sampled_trials_per_condition": TRIALS,
        "draw_seed": "session_id + draw; one shared condition-wise draw across units",
        "inputs": input_records,
        "sources": [],
        "outputs": [],
    }
    for path in (Path(__file__).resolve(), CONFIG, ROOT / "common/drifting_gratings.py", ROOT / "scripts/extract_allen_v1_bridge.py"):
        manifest["sources"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for name in ("unit_draw_comparison.csv", "unit_repeat_summary.csv", "session_draw_summary.csv", "allen_repeat_count_diagnostics.png", "REPEAT_COUNT_CHECKPOINT.md"):
        path = output / name
        if path.is_file():
            manifest["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote Allen repeat-count checkpoint to {output}")


if __name__ == "__main__":
    main()
