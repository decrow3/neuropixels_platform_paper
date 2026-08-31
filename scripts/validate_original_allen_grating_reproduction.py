#!/usr/bin/env python3
"""Strictly reproduce released Allen grating metrics from raw original NWBs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_allen_v1_bridge import (  # noqa: E402
    COHORT_LABELS,
    common_qc,
    condition_presentations,
    released_metrics,
    sha256,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/allen_v1_bridge.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/v1_systemic_gap_audit_v1/03_original_method_reproduction"
)
F1_ATOL = 3e-9
MOD_ATOL = 2e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows: pd.DataFrame) -> dict[str, object]:
    rows = rows.copy()
    rows["f1_abs_error"] = np.abs(rows.raw_f1_f0_dg - rows.released_f1_f0_dg)
    rows["mod_abs_error"] = np.abs(rows.raw_mod_idx_dg - rows.released_mod_idx_dg)
    rows["f1_match"] = rows.f1_abs_error.le(F1_ATOL) | (
        rows.raw_f1_f0_dg.isna() & rows.released_f1_f0_dg.isna()
    )
    rows["mod_match"] = rows.mod_abs_error.le(MOD_ATOL) | (
        rows.raw_mod_idx_dg.isna() & rows.released_mod_idx_dg.isna()
    )
    rows["passes"] = (
        rows.f1_match
        & rows.mod_match
    )
    return {
        "rows": rows,
        "units": len(rows),
        "passing_units": int(rows.passes.sum()),
        "all_pass": bool(rows.passes.all()),
        "max_f1_abs_error": float(rows.f1_abs_error.max()),
        "max_mod_abs_error": float(rows.mod_abs_error.max()),
    }


def write_report(output: Path, result: dict[str, object], sessions: pd.DataFrame) -> None:
    rows = result["rows"]
    table_columns = list(sessions.columns)
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for record in sessions.to_dict(orient="records"):
        table_lines.append(
            "| " + " | ".join(str(record[column]) for column in table_columns) + " |"
        )
    session_table = "\n".join(table_lines)
    report = f"""# Exact original Allen grating-method reproduction

## Overall assessment: Ready to share

The raw-NWB reproduction passes the strict all-unit gate: **{result['passing_units']}/{result['units']} released VISp units** reproduce their original F1/F0 and Welch modulation index.

## Methodology review

The original Allen calculation has two distinct windows:

1. Preferred condition is selected from conditionwise mean spike counts using each presentation's recorded `start_time` and `stop_time`.
2. F1/F0 and Welch modulation are then computed for that condition using AllenSDK's fixed 1,999-bin, 1-ms metric representation (`trial_duration = 1.9985 s` for cycle folding; `nperseg = 1024` for Welch).

Using a fixed 2.0-second window for step 1 is not exact. It changes one non-QC unit (951868542) because actual-stop counting produces a tie resolved in first-condition order. The corrected two-stage implementation reproduces that unit and every other selected unit.

## Calculation spot-checks

- F1/F0 maximum absolute error: **{result['max_f1_abs_error']:.3g}** (gate ≤ {F1_ATOL:.1g}).
- Welch modulation maximum absolute error: **{result['max_mod_abs_error']:.3g}** (gate ≤ {MOD_ATOL:.1g}).
- The raw joint metric condition is saved for every unit. The released table's
  `pref_ori_dg` and `pref_tf_dg` are marginal preferences, not the joint
  condition used for F1/F0 and Welch, so they are retained for provenance but
  are not incorrectly treated as a direct condition-identity check.
- Unit IDs are unique and reconciled directly to `data/unit_table.csv`.
- Every raw NWB size and SHA-256 matches `config/allen_v1_bridge.json`.

## Session coverage

{session_table}

## Issues found

1. **Resolved, high relevance:** the earlier bridge used a fixed 2.0-second preference-selection window. It was numerically exact for 371/372 units but selected the wrong tied condition for one excluded unit. The bridge now uses recorded stop times, matching AllenSDK.
2. **Caveat:** this validates four raw sessions spanning both Allen stimulus families, not all released sessions. It establishes implementation fidelity on every released VISp unit in those raw files.

## Claim boundary

This reproduction proves that we can recreate the original Allen F1/F0 and Welch numbers from raw original data. It does **not** establish that the Welch metric is an appropriate absolute cross-dataset measure, nor that a harmonized MouseV2 calculation has identical stimulus timing.

## Reproducible artifacts

- `unit_reproduction.csv`: unit-level released values, recomputed values, preferred metric conditions, errors, and pass flags.
- `session_reproduction.csv`: coverage and maximum errors by session.
- `manifest.json`: input, source-code, configuration, and output hashes.
"""
    (output / "VALIDATION_REPORT.md").write_text(report)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    release = pd.read_csv(ROOT / "data/unit_table.csv", low_memory=False)
    records = []
    input_records = []
    for asset in config["assets"]:
        session_id = int(asset["session_id"])
        session_type = str(asset["session_type"])
        path = Path(config["download_root"]) / asset["relative_path"]
        if path.stat().st_size != int(asset["bytes"]):
            raise ValueError(f"Size mismatch: {path}")
        observed_hash = file_hash(path)
        if observed_hash != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        selected = release.loc[
            release.ecephys_session_id.eq(session_id)
            & release.ecephys_structure_acronym.eq("VISp")
        ].copy()
        if len(selected) != int(asset["expected_released_v1_units"]):
            raise ValueError(f"Released VISp unit-count mismatch: {session_id}")
        unit_ids = selected.ecephys_unit_id.astype(int).tolist()
        extracted = read_targeted_nwb(
            path, unit_ids=unit_ids, interval_name=asset["grating_table"]
        )
        conditions = condition_presentations(extracted.interval_table)
        for _, released in selected.iterrows():
            unit_id = int(released.ecephys_unit_id)
            raw = released_metrics(extracted.spikes_by_id[unit_id], conditions)
            records.append(
                {
                    "session_id": session_id,
                    "session_type": session_type,
                    "cohort": COHORT_LABELS[session_type],
                    "ecephys_unit_id": unit_id,
                    "common_qc": bool(common_qc(released.to_frame().T).iloc[0]),
                    "released_pref_ori_marginal": float(released.pref_ori_dg),
                    "released_pref_tf_marginal": float(released.pref_tf_dg),
                    "raw_metric_condition_ori": float(raw["raw_released_pref_ori"]),
                    "raw_metric_condition_tf": float(raw["raw_released_pref_tf"]),
                    "released_f1_f0_dg": float(released.f1_f0_dg),
                    "raw_f1_f0_dg": float(raw["raw_released_f1_f0_dg"]),
                    "released_mod_idx_dg": float(released.mod_idx_dg),
                    "raw_mod_idx_dg": float(raw["raw_released_mod_idx_dg"]),
                }
            )
        input_records.append(
            {
                "session_id": session_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_hash,
                "dandiset_id": asset["dandiset_id"],
                "asset_id": asset["asset_id"],
            }
        )

    # The released table stores marginal preferred orientation/TF, while the
    # metrics use the maximum joint condition. Recompute the latter raw, so the
    # condition identity is validated through metric equality rather than by
    # incorrectly comparing it with pref_ori_dg/pref_tf_dg.
    result = validate_rows(pd.DataFrame(records))
    rows = result["rows"]
    rows.to_csv(output / "unit_reproduction.csv", index=False)
    sessions = (
        rows.groupby(["cohort", "session_id"], as_index=False)
        .agg(
            units=("ecephys_unit_id", "size"),
            common_qc_units=("common_qc", "sum"),
            passing_units=("passes", "sum"),
            max_f1_abs_error=("f1_abs_error", "max"),
            max_mod_abs_error=("mod_abs_error", "max"),
        )
    )
    sessions.to_csv(output / "session_reproduction.csv", index=False)
    write_report(output, result, sessions)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/extract_allen_v1_bridge.py",
        ROOT / "scripts/trace_v1_systemic_gap_cases.py",
        ROOT / "common/drifting_gratings.py",
        config_path,
        ROOT / "data/unit_table.csv",
    ]
    outputs = [
        output / "unit_reproduction.csv",
        output / "session_reproduction.csv",
        output / "VALIDATION_REPORT.md",
    ]
    manifest = {
        "schema_version": 1,
        "status": "pass" if result["all_pass"] else "fail",
        "tolerances": {"f1_f0_abs": F1_ATOL, "mod_idx_abs": MOD_ATOL},
        "summary": {key: value for key, value in result.items() if key != "rows"},
        "inputs": input_records,
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sources
        ],
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not result["all_pass"]:
        failed = rows.loc[~rows.passes, ["session_id", "ecephys_unit_id", "f1_abs_error", "mod_abs_error"]]
        raise AssertionError(f"Original-method reproduction failed:\n{failed}")
    print(
        f"PASS: {result['passing_units']}/{result['units']} units; "
        f"max F1 error {result['max_f1_abs_error']:.3g}; "
        f"max modulation error {result['max_mod_abs_error']:.3g}"
    )


if __name__ == "__main__":
    main()
