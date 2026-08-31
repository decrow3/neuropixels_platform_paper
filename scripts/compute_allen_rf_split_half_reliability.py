#!/usr/bin/env python3
"""Compute Allen unit RF split-half reliability from compact Gabor caches.

Repeats are alternated within each position x orientation condition, then each
half is aggregated to the 81 x/y stimulus positions.  Pearson correlation
across positions is Spearman-Brown corrected, matching the reliability
convention used by ``common.parametric_models.fit_parametric_rf_models``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.parametric_models import _split_half_reliability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/allen_full_rf_production_v1/01_compact_cache"
DEFAULT_FITS = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/all_session_unit_geometry_fits.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/allen_rf_split_half_reliability.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def position_codes(trials: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    cells = (
        trials[["x_position", "y_position"]]
        .drop_duplicates()
        .sort_values(["x_position", "y_position"])
        .reset_index(drop=True)
    )
    lookup = {tuple(row): index for index, row in cells.iterrows()}
    codes = np.asarray(
        [lookup[tuple(row)] for row in trials[["x_position", "y_position"]].itertuples(index=False, name=None)],
        dtype=int,
    )
    return codes, cells


def balanced_split(trials: pd.DataFrame) -> np.ndarray:
    if "repeat_index_within_condition" in trials:
        repeat = pd.to_numeric(trials.repeat_index_within_condition, errors="raise").to_numpy(int)
        return repeat % 2
    split = np.zeros(len(trials), dtype=int)
    groups = trials.groupby(["x_position", "y_position", "orientation_index"], sort=True).groups
    for indices in groups.values():
        ordered = np.asarray(list(indices), dtype=int)
        split[ordered] = np.arange(len(ordered)) % 2
    return split


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    fits_path = args.fits.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fits = pd.read_csv(fits_path, usecols=["session_id", "ecephys_unit_id"], low_memory=False)
    expected = fits.drop_duplicates(["session_id", "ecephys_unit_id"])
    rows = []
    session_audit = []
    for session_id in sorted(expected.session_id.astype(int).unique()):
        cache = cache_root / f"session_{session_id}"
        trials = pd.read_csv(cache / "gabor_trial_gaze_table.csv", low_memory=False)
        spike_cache = np.load(cache / "gabor_spike_counts.npz")
        unit_ids = spike_cache["unit_ids"].astype(int)
        counts = spike_cache["counts"].astype(float)
        if counts.shape != (len(unit_ids), len(trials)):
            raise ValueError(
                f"Session {session_id}: counts shape {counts.shape} does not match "
                f"{len(unit_ids)} units x {len(trials)} trials"
            )
        codes, cells = position_codes(trials)
        split = balanced_split(trials)
        correlation, corrected, p_value = _split_half_reliability(
            counts, codes, split, len(cells)
        )
        session_rows = pd.DataFrame(
            {
                "ecephys_session_id": session_id,
                "ecephys_unit_id": unit_ids,
                "rf_split_half_r": correlation,
                "rf_split_half_spearman_brown": corrected,
                "rf_reliability_p": p_value,
                "rf_presentations": len(trials),
                "rf_position_cells": len(cells),
            }
        )
        rows.append(session_rows)
        expected_ids = set(
            expected.loc[expected.session_id.astype(int).eq(session_id), "ecephys_unit_id"].astype(int)
        )
        available_ids = set(unit_ids)
        session_audit.append(
            {
                "ecephys_session_id": session_id,
                "cache_units": len(unit_ids),
                "fit_units": len(expected_ids),
                "fit_units_in_cache": len(expected_ids & available_ids),
                "presentations": len(trials),
                "position_cells": len(cells),
                "half0_presentations": int(np.sum(split == 0)),
                "half1_presentations": int(np.sum(split == 1)),
            }
        )
        print(f"{session_id}: {len(unit_ids)} units, {len(trials)} trials, {len(cells)} positions", flush=True)

    result = pd.concat(rows, ignore_index=True)
    if result.duplicated("ecephys_unit_id").any():
        raise ValueError("Duplicate ecephys_unit_id values across split-half reliability output")
    result.to_csv(output, index=False, compression="gzip", float_format="%.9g")
    audit_path = output.with_name("allen_rf_split_half_reliability_session_audit.csv")
    pd.DataFrame(session_audit).to_csv(audit_path, index=False)
    manifest = {
        "method": "Pearson correlation across 81 position-wise mean spike counts, with Spearman-Brown correction",
        "split": "alternating repeat_index_within_condition within x/y/orientation condition",
        "orientation_handling": "balanced within halves, then pooled when responses are averaged by x/y position",
        "trials": "all cached Gabor trials, matching the full-production --all-gabor-trials fit",
        "sessions": int(result.ecephys_session_id.nunique()),
        "units": int(len(result)),
        "finite_corrected_reliability": int(np.isfinite(result.rf_split_half_spearman_brown).sum()),
        "fraction_gt_0p6": float((result.rf_split_half_spearman_brown > 0.6).mean()),
        "output": str(output),
        "session_audit": str(audit_path),
    }
    output.with_name("allen_rf_split_half_reliability_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
