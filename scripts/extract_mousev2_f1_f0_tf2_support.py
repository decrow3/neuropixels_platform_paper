#!/usr/bin/env python3
"""Compute MouseV2 F1/F0 on the exact TF=2 support of Allen FC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_mousev2_units  # noqa: E402
from common.drifting_gratings import _prepare_conditions  # noqa: E402
from scripts.extract_mousev2_grating_common_support import common_presentations  # noqa: E402
from scripts.extract_allen_v1_f1_f0_full_cohort import (  # noqa: E402
    prepare_unit,
    sha256,
    summarize_draw,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


CONFIG = ROOT / "config/figure3_mousev2.json"
ALLEN_CONFIG = ROOT / "config/allen_v1_bridge.json"
MOUSE_COMMON = ROOT / "data/imports/mousev2_grating_common_support_v1/unit_metric_comparison.csv"
ALLEN_OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/04_allen_full_cohort_harmonized"
OUTPUT = ALLEN_OUTPUT
F1_ATOL = 5e-9


def mouse_conditions(
    table: pd.DataFrame, *, temporal_frequency_hz: float | None = None
) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
    """Use MouseV2's validated lexicographic condition/tie convention."""
    selected = common_presentations(table)
    if temporal_frequency_hz is not None:
        selected = selected.loc[
            np.isclose(selected.temporal_frequency, temporal_frequency_hz)
        ].copy()
    prepared, duration_s, duration_ms = _prepare_conditions(selected, min_trials=15)
    if duration_s != 1.0 or duration_ms != 1000:
        raise AssertionError("MouseV2 shared-support duration drift")
    return [
        (
            (
                float(condition["parameters"]["orientation"]),
                float(condition["parameters"]["temporal_frequency"]),
                float(condition["parameters"]["spatial_frequency"]),
                float(condition["parameters"].get("contrast", 0.8)),
            ),
            condition["starts"],
        )
        for condition in prepared
    ]


def session_bootstrap(
    mouse: np.ndarray, allen_draws: pd.DataFrame, *, draws: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    session_ids = allen_draws.session_id.unique()
    allen_centers = allen_draws.groupby("session_id").mean_log10_f1_f0.median()
    observed = float(mouse.mean() - allen_centers.mean())
    result = np.empty(draws)
    values_by_session = {
        session_id: allen_draws.loc[
            allen_draws.session_id.eq(session_id), "mean_log10_f1_f0"
        ].dropna().to_numpy()
        for session_id in session_ids
    }
    for draw in range(draws):
        mouse_sample = rng.choice(mouse, len(mouse), replace=True)
        selected = rng.choice(session_ids, len(session_ids), replace=True)
        allen_sample = [rng.choice(values_by_session[session_id]) for session_id in selected]
        result[draw] = mouse_sample.mean() - np.mean(allen_sample)
    return observed, float(np.quantile(result, .025)), float(np.quantile(result, .975))


def render_final_comparison(
    allen_summary: pd.DataFrame,
    mouse_bo: np.ndarray,
    mouse_fc: np.ndarray,
    contrasts: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    colors = {"brain_observatory_1.1": "#6F63A6", "functional_connectivity": "#B07AA1"}
    mouse_centers = {"brain_observatory_1.1": mouse_bo.mean(), "functional_connectivity": mouse_fc.mean()}
    rng = np.random.default_rng(20260825)
    for position, session_type in enumerate(("brain_observatory_1.1", "functional_connectivity")):
        group = allen_summary.loc[allen_summary.session_type.eq(session_type)]
        values = group.harmonized_median_log10_f1_f0.to_numpy()
        axes[0].scatter(
            position + rng.uniform(-.1, .1, len(values)), values,
            color=colors[session_type], alpha=.7, s=28,
        )
        axes[0].plot([position-.18, position+.18], [values.mean()]*2, color="black", lw=2)
        axes[0].hlines(
            mouse_centers[session_type], position-.25, position+.25,
            color="#D95F02", linestyle="--", lw=2,
        )
    axes[0].set_xticks((0, 1), ("BO: all shared TFs", "FC: TF=2 only"))
    axes[0].set(ylabel="session mean log10 F1/F0", title="Matched 1-s / 15-trial support")
    axes[0].text(.02, .98, "points: Allen sessions\norange dash: MouseV2 center", transform=axes[0].transAxes, va="top", fontsize=9)

    y = np.arange(len(contrasts))
    axes[1].errorbar(
        contrasts.difference_log10_f1_f0,
        y,
        xerr=np.vstack(
            [
                contrasts.difference_log10_f1_f0 - contrasts.bootstrap_95ci_low,
                contrasts.bootstrap_95ci_high - contrasts.difference_log10_f1_f0,
            ]
        ),
        fmt="o",
        color="#D95F02",
        ecolor="0.25",
        capsize=4,
    )
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set_yticks(y, ("MouseV2 − Allen BO", "MouseV2 − Allen FC"))
    axes[1].set(xlabel="difference in session mean log10 F1/F0", title="Session-bootstrap contrasts")
    axes[1].invert_yaxis()
    fig.suptitle("Full-cohort V1 conventional F1/F0 comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    support = json.loads(ALLEN_CONFIG.read_text())["common_support"]
    tf2_support = {**support, "temporal_frequency_hz": [2.0]}
    nwb_root = Path(config["nwb_input"]["default_root"])
    common_qc = load_mousev2_units(
        apply_qc=False,
        population_profile="common_qc",
        grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
    )
    saved_common = pd.read_csv(MOUSE_COMMON).set_index("unit_id")
    unit_rows = []
    validation_rows = []
    input_rows = []
    for number, session in enumerate(config["sessions"], start=1):
        site = str(session["site"])
        site_number = int(session["site_number"])
        offset = int(session["id_offset"])
        path = nwb_root / session["nwb_relative_path"]
        selected = common_qc.loc[common_qc.site.eq(site)].copy()
        external_ids = selected.unit_id.astype(int).to_numpy()
        internal_ids = (external_ids - offset).tolist()
        print(f"[{number}/8] {site}: {len(internal_ids)} common-QC units", flush=True)
        extracted = read_targeted_nwb(
            path,
            unit_ids=internal_ids,
            interval_name="drifting_gratings_field_block_presentations",
        )
        tf2_conditions = mouse_conditions(extracted.interval_table, temporal_frequency_hz=2.0)
        if len(tf2_conditions) != 4 or set(len(starts) for _, starts in tf2_conditions) != {15}:
            raise AssertionError(f"{site}: unexpected TF=2 shared support")
        tf2_indices = [np.arange(15) for _ in tf2_conditions]

        all_conditions = None
        all_indices = None
        if site == "site2":
            all_conditions = mouse_conditions(extracted.interval_table)
            if len(all_conditions) != 20:
                raise AssertionError("site2 shared-support validation has wrong condition count")
            all_indices = [np.arange(len(starts)) for _, starts in all_conditions]

        for external_id, internal_id in zip(external_ids, internal_ids):
            spikes = extracted.spikes_by_id[internal_id]
            result = summarize_draw(prepare_unit(spikes, tf2_conditions), tf2_indices)
            unit_rows.append(
                {
                    "site": site,
                    "site_number": site_number,
                    "unit_id": int(external_id),
                    **result,
                }
            )
            if all_conditions is not None and all_indices is not None:
                reproduced = summarize_draw(prepare_unit(spikes, all_conditions), all_indices)
                expected = float(saved_common.loc[external_id, "f1_f0_dg_common_support"])
                error = abs(reproduced["f1_f0"] - expected)
                validation_rows.append(
                    {"site": site, "unit_id": int(external_id), "saved_f1_f0": expected, "raw_f1_f0": reproduced["f1_f0"], "absolute_error": error}
                )
                if not (error <= F1_ATOL or (np.isnan(expected) and np.isnan(reproduced["f1_f0"]))):
                    raise AssertionError(f"MouseV2 shared-support reproduction failed for {external_id}: {error}")
        input_rows.append({"site": site, "path": str(path), "bytes": path.stat().st_size})

    units = pd.DataFrame(unit_rows)
    validation = pd.DataFrame(validation_rows)
    units.to_csv(output / "mouse_tf2_unit_metrics.csv", index=False)
    validation.to_csv(output / "mouse_shared_support_validation.csv", index=False)
    units["log10_f1_f0"] = np.log10(units.f1_f0.where(units.f1_f0 > 0))
    sessions = (
        units.groupby(["site", "site_number"])
        .agg(
            valid_units=("log10_f1_f0", "count"),
            mean_log10_f1_f0=("log10_f1_f0", "mean"),
            median_f1_f0=("f1_f0", "median"),
            zero_f0_unit_fraction=("f1_f0", lambda x: x.isna().mean()),
        )
        .reset_index()
    )
    sessions.to_csv(output / "mouse_tf2_session_summary.csv", index=False)

    allen = pd.read_csv(output / "harmonized_session_draws.csv")
    mouse_all = pd.read_csv(MOUSE_COMMON)
    mouse_all = mouse_all.loc[mouse_all.default_qc & mouse_all.f1_f0_dg_common_support.gt(0)].copy()
    mouse_all["log10_f1_f0"] = np.log10(mouse_all.f1_f0_dg_common_support)
    mouse_bo = mouse_all.groupby("site_number").log10_f1_f0.mean().to_numpy()
    mouse_fc = sessions.mean_log10_f1_f0.to_numpy()
    contrast_rows = []
    for session_type, label, mouse_values, seed in (
        ("brain_observatory_1.1", "Allen Brain Observatory 1.1", mouse_bo, 20260825),
        ("functional_connectivity", "Allen Functional Connectivity", mouse_fc, 20260826),
    ):
        selected = allen.loc[allen.session_type.eq(session_type)]
        observed, low, high = session_bootstrap(
            mouse_values, selected, draws=args.bootstrap_draws, seed=seed
        )
        contrast_rows.append(
            {
                "contrast": f"MouseV2 minus {label}",
                "mouse_support": "TF 1,2,4,8,15" if session_type == "brain_observatory_1.1" else "TF 2 only",
                "allen_support": "TF 1,2,4,8,15" if session_type == "brain_observatory_1.1" else "TF 2 only",
                "mouse_sessions": len(mouse_values),
                "allen_sessions": selected.session_id.nunique(),
                "mouse_center_mean_log10_f1_f0": mouse_values.mean(),
                "allen_center_mean_log10_f1_f0": selected.groupby("session_id").mean_log10_f1_f0.median().mean(),
                "difference_log10_f1_f0": observed,
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "geometric_mean_ratio": 10**observed,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "tf_matched_full_cohort_contrasts.csv", index=False)
    allen_summary = pd.read_csv(output / "harmonized_session_summary.csv")
    render_final_comparison(
        allen_summary,
        mouse_bo,
        mouse_fc,
        contrasts,
        output / "tf_matched_full_cohort_comparison.png",
    )

    manifest = {
        "status": "mouse_tf2_support_and_tf_matched_full_cohort_contrast",
        "tf2_support": tf2_support,
        "site2_validation_tolerance": F1_ATOL,
        "inputs": input_rows,
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), CONFIG, ALLEN_CONFIG, MOUSE_COMMON, output / "harmonized_session_draws.csv")
        ],
        "outputs": [],
    }
    for name in (
        "mouse_tf2_unit_metrics.csv", "mouse_tf2_session_summary.csv",
        "mouse_shared_support_validation.csv", "tf_matched_full_cohort_contrasts.csv",
        "tf_matched_full_cohort_comparison.png",
        "FULL_COHORT_CHECKPOINT.md",
    ):
        path = output / name
        if path.is_file():
            manifest["outputs"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "mouse_tf2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote MouseV2 TF=2 support and matched contrasts to {output}", flush=True)


if __name__ == "__main__":
    main()
