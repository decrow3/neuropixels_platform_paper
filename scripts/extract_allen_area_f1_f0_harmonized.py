#!/usr/bin/env python3
"""Extract matched 1-s/15-trial F1/F0 for Allen BO V1, HVAs, and LP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import FINE_TO_COARSE  # noqa: E402
from scripts.extract_allen_v1_bridge import common_qc, condition_starts  # noqa: E402
from scripts.extract_allen_v1_f1_f0_full_cohort import (  # noqa: E402
    prepare_unit,
    sha256,
    summarize_draw,
)
from scripts.trace_v1_systemic_gap_cases import read_targeted_nwb  # noqa: E402


INVENTORY = Path("/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/session_inventory.json")
CONFIG = ROOT / "config/allen_v1_bridge.json"
RELEASE = ROOT / "data/unit_table.csv"
V1_REFERENCE = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/04_allen_full_cohort_harmonized"
    / "harmonized_unit_draws.csv"
)
OUTPUT = (
    ROOT
    / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0"
)
SESSION_TYPE = "brain_observatory_1.1"
FINE_AREAS = ("VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "LP")
COARSE_ORDER = ("V1", "LM", "RL", "LP", "AL", "PM", "AM")


def eligible_units(release: pd.DataFrame) -> pd.DataFrame:
    selected = release.loc[
        release.session_type.eq(SESSION_TYPE)
        & release.ecephys_structure_acronym.isin(FINE_AREAS)
    ].copy()
    selected = selected.loc[common_qc(selected)].copy()
    selected["area_coarse"] = selected.ecephys_structure_acronym.map(
        lambda value: FINE_TO_COARSE.get(value, value)
    )
    unexpected = sorted(set(selected.area_coarse).difference(COARSE_ORDER))
    if unexpected:
        raise AssertionError(f"Unexpected coarse areas: {unexpected}")
    return selected


def complete_support(
    table: pd.DataFrame, support: dict[str, object]
) -> tuple[list[tuple[tuple[float, float, float, float], np.ndarray]], bool, int, int]:
    conditions = condition_starts(table, common_support=support)
    lengths = [len(starts) for _, starts in conditions]
    complete = len(conditions) == 20 and set(lengths) == {15}
    return conditions, complete, min(lengths) if lengths else 0, max(lengths) if lengths else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(INVENTORY.read_text())
    inventory_by_id = {int(row["ecephys_session_id"]): row for row in inventory}
    support = json.loads(CONFIG.read_text())["common_support"]
    release = pd.read_csv(RELEASE, low_memory=False)
    selected = eligible_units(release)
    session_ids = sorted(selected.ecephys_session_id.astype(int).unique())

    unit_rows = []
    audit_rows = []
    retained_paths = []
    for number, session_id in enumerate(session_ids, start=1):
        group = selected.loc[selected.ecephys_session_id.eq(session_id)].copy()
        path = Path(inventory_by_id[session_id]["nwb_path"])
        unit_ids = group.ecephys_unit_id.astype(int).tolist()
        print(f"[{number:02d}/{len(session_ids)}] {session_id}: {len(unit_ids)} units", flush=True)
        extracted = read_targeted_nwb(
            path,
            unit_ids=unit_ids,
            interval_name="drifting_gratings_presentations",
        )
        conditions, complete, minimum, maximum = complete_support(
            extracted.interval_table, support
        )
        audit_rows.append(
            {
                "session_id": session_id,
                "eligible_units": len(group),
                "conditions": len(conditions),
                "minimum_repeats": minimum,
                "maximum_repeats": maximum,
                "complete_15_trial_support": complete,
                "nwb_path": str(path),
            }
        )
        if not complete:
            print(f"  excluded: incomplete support ({minimum}-{maximum} repeats)", flush=True)
            continue
        retained_paths.append(path)
        indices = [np.arange(15) for _ in conditions]
        by_id = group.set_index("ecephys_unit_id")
        for unit_id in unit_ids:
            result = summarize_draw(
                prepare_unit(extracted.spikes_by_id[unit_id], conditions), indices
            )
            source = by_id.loc[unit_id]
            unit_rows.append(
                {
                    "ecephys_session_id": session_id,
                    "ecephys_unit_id": unit_id,
                    "ecephys_structure_acronym": source.ecephys_structure_acronym,
                    "area_coarse": source.area_coarse,
                    "f1_f0_dg_harmonized": result["f1_f0"],
                    "preferred_orientation_deg_harmonized": result["preferred_orientation_deg"],
                    "preferred_tf_hz_harmonized": result["preferred_tf_hz"],
                    "preferred_mean_spikes_harmonized": result["preferred_mean_spikes"],
                    "valid_trials_harmonized": result["valid_trials"],
                    "selected_trials_harmonized": result["selected_trials"],
                    "population_profile": "common_qc",
                }
            )

    audit = pd.DataFrame(audit_rows)
    units = pd.DataFrame(unit_rows)
    if audit.complete_15_trial_support.sum() != 28:
        raise AssertionError(
            f"Expected 28 complete BO sessions, found {audit.complete_15_trial_support.sum()}"
        )
    if set(units.area_coarse) != set(COARSE_ORDER):
        raise AssertionError("Not all seven Figure 3 groups survived extraction")

    # Exact V1 gate against the accepted full-cohort bridge.
    reference = pd.read_csv(V1_REFERENCE)
    reference = reference.loc[
        reference.session_type.eq(SESSION_TYPE)
        & reference.draw.eq(0)
        & reference.session_id.isin(audit.loc[audit.complete_15_trial_support, "session_id"])
    ]
    gate = units.loc[units.area_coarse.eq("V1"), ["ecephys_session_id", "ecephys_unit_id", "f1_f0_dg_harmonized"]].merge(
        reference[["session_id", "ecephys_unit_id", "f1_f0"]],
        left_on=["ecephys_session_id", "ecephys_unit_id"],
        right_on=["session_id", "ecephys_unit_id"],
        validate="one_to_one",
    )
    gate["absolute_error"] = np.abs(gate.f1_f0_dg_harmonized - gate.f1_f0)
    if len(gate) != int((units.area_coarse == "V1").sum()) or gate.absolute_error.max() > 1e-12:
        raise AssertionError("V1 reproduction gate failed")

    valid = units.loc[units.f1_f0_dg_harmonized.gt(0)].copy()
    valid["log10_f1_f0_harmonized"] = np.log10(valid.f1_f0_dg_harmonized)
    session_areas = (
        valid.groupby(["ecephys_session_id", "area_coarse"], sort=True)
        .agg(
            mean_log10_f1_f0_harmonized=("log10_f1_f0_harmonized", "mean"),
            n_units=("ecephys_unit_id", "size"),
            median_preferred_tf_hz=("preferred_tf_hz_harmonized", "median"),
        )
        .reset_index()
    )
    audit.to_csv(output / "allen_bo_session_support_audit.csv", index=False)
    units.to_csv(output / "allen_bo_area_unit_f1_f0_harmonized.csv", index=False)
    session_areas.to_csv(output / "allen_bo_session_area_f1_f0_harmonized.csv", index=False)
    gate.to_csv(output / "allen_bo_v1_reproduction_gate.csv", index=False)

    manifest = {
        "status": "figure3_allen_bo_area_harmonized_f1_f0",
        "support": support,
        "population_profile": "common_qc",
        "session_type": SESSION_TYPE,
        "complete_sessions": int(audit.complete_15_trial_support.sum()),
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), INVENTORY, CONFIG, RELEASE, V1_REFERENCE)
        ],
        "large_nwb_sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": "not_recomputed_for_large_source"}
            for path in retained_paths
        ],
        "outputs": [],
    }
    for name in (
        "allen_bo_session_support_audit.csv",
        "allen_bo_area_unit_f1_f0_harmonized.csv",
        "allen_bo_session_area_f1_f0_harmonized.csv",
        "allen_bo_v1_reproduction_gate.csv",
    ):
        path = output / name
        manifest["outputs"].append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote harmonized Allen area F1/F0 to {output}", flush=True)


if __name__ == "__main__":
    main()
