#!/usr/bin/env python3
"""Snapshot and validate PilotAnalysis per-unit RF peak exports.

The imported peaks are a provisional retinotopic-position bridge. They are
unsmoothed per-unit argmax locations, not final Allen-compatible RF
significance or area measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_config, load_mousev2_units  # noqa: E402


REQUIRED_COLUMNS = {
    "unit_index",
    "unit_id",
    "probe",
    "peak_y_idx",
    "peak_x_idx",
    "peak_y",
    "peak_x",
    "peak_value",
    "is_qc",
}
RF_GRID = set(float(value) for value in range(-40, 41, 10))
METHOD = "per_unit_unsmoothed_argmax_spike_counts"
GAZE_CORRECTION = "none"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_record(path: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.rstrip()

    status = run("status", "--short").splitlines()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_clean": not status,
        "working_tree_status": status,
    }


def bootstrap_median_ci(
    values: pd.Series,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return np.nan, np.nan
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    estimates = np.median(array[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def summarize_group(
    group: pd.DataFrame,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, object]:
    pilot_qc = group["pilot_qc"].astype(bool)
    default_qc = group["default_qc"].astype(bool)
    used = group[pilot_qc].copy()
    if used.empty:
        raise ValueError(
            f"No Pilot-QC RF units for {group['site'].iloc[0]} {group['probe'].iloc[0]}"
        )

    x_low, x_high = bootstrap_median_ci(
        used["rf_center_x_deg"], rng=rng, iterations=iterations
    )
    y_low, y_high = bootstrap_median_ci(
        used["rf_center_y_deg"], rng=rng, iterations=iterations
    )
    edge = used["rf_center_x_deg"].abs().eq(40) | used["rf_center_y_deg"].abs().eq(40)

    def median(mask: pd.Series, column: str) -> float:
        values = group.loc[mask, column]
        return float(values.median()) if len(values) else np.nan

    return {
        "site": group["site"].iloc[0],
        "site_number": int(group["site_number"].iloc[0]),
        "subject_id": int(group["subject_id"].iloc[0]),
        "probe": group["probe"].iloc[0],
        "n_units_total": int(len(group)),
        "n_units_pilot_qc": int(pilot_qc.sum()),
        "n_units_default_qc": int(default_qc.sum()),
        "n_units_used": int(len(used)),
        "rf_center_x_deg": float(used["rf_center_x_deg"].median()),
        "rf_center_x_ci_low_deg": x_low,
        "rf_center_x_ci_high_deg": x_high,
        "rf_center_y_deg": float(used["rf_center_y_deg"].median()),
        "rf_center_y_ci_low_deg": y_low,
        "rf_center_y_ci_high_deg": y_high,
        "rf_peak_value_median": float(used["rf_peak_value"].median()),
        "rf_grid_edge_fraction": float(edge.mean()),
        "rf_center_x_all_units_deg": median(
            pd.Series(True, index=group.index), "rf_center_x_deg"
        ),
        "rf_center_y_all_units_deg": median(
            pd.Series(True, index=group.index), "rf_center_y_deg"
        ),
        "rf_center_x_default_qc_deg": median(default_qc, "rf_center_x_deg"),
        "rf_center_y_default_qc_deg": median(default_qc, "rf_center_y_deg"),
        "qc_rule": "PilotAnalysis is_qc: snr>5, d_prime>2, rp_contamination<0.1, default_qc",
        "rf_method": METHOD,
        "gaze_correction": GAZE_CORRECTION,
        "bootstrap_iterations": iterations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT.parent.parent / "PilotAnalysis" / "PilotAnalysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "imports" / "pilot_rf_peaks_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "figure3_mousev2.json",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot_root = args.pilot_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    current = load_mousev2_units(apply_qc=False, config_path=args.config)[
        ["unit_id", "site", "session_num", "subject_id", "probe_letter"]
    ].copy()
    quality_frames = []
    data_dir = ROOT / config["data_directory"]
    for session in config["sessions"]:
        quality = pd.read_csv(
            data_dir / f"{session['site']}_processed" / "unit_quality.csv",
            usecols=["unit_id", "default_qc"],
        )
        quality_frames.append(quality)
    quality = pd.concat(quality_frames, ignore_index=True)
    if not quality["unit_id"].is_unique:
        raise ValueError("unit_quality unit_id values collide across sessions")
    current = current.merge(quality, on="unit_id", validate="one_to_one")

    imported_frames = []
    source_records = []
    for session in config["sessions"]:
        site = session["site"]
        subject_id = int(session["subject_id"])
        offset = int(session["id_offset"])
        for probe in config["probe_labels"]:
            matches = sorted(
                (pilot_root / "results").glob(
                    f"sub-{subject_id}_*/Probe{probe}/*_rf_peaks.csv"
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one Pilot RF export for subject {subject_id} probe {probe}; "
                    f"found {len(matches)}"
                )
            source = matches[0]
            table = pd.read_csv(source)
            missing = REQUIRED_COLUMNS.difference(table.columns)
            if missing:
                raise ValueError(f"{source} missing columns: {sorted(missing)}")
            if table["unit_id"].duplicated().any():
                raise ValueError(f"Duplicate local unit IDs in {source}")
            if not table["unit_index"].equals(table["unit_id"]):
                raise ValueError(f"unit_index and unit_id disagree in {source}")
            if set(table["probe"].astype(str)) != {f"Probe{probe}"}:
                raise ValueError(f"Probe column disagrees with path in {source}")
            for column in ("peak_x", "peak_y"):
                values = set(table[column].dropna().astype(float).unique())
                if not values.issubset(RF_GRID):
                    raise ValueError(f"Unexpected {column} grid values in {source}: {values}")

            imported = pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "site": site,
                    "site_number": int(session["site_number"]),
                    "local_unit_id": table["unit_id"].astype(int),
                    "unit_id": table["unit_id"].astype(int) + offset,
                    "probe": probe,
                    "rf_center_x_deg": pd.to_numeric(table["peak_x"], errors="coerce"),
                    "rf_center_y_deg": pd.to_numeric(table["peak_y"], errors="coerce"),
                    "rf_peak_value": pd.to_numeric(table["peak_value"], errors="coerce"),
                    "pilot_qc": table["is_qc"].fillna(False).astype(bool),
                    "rf_method": METHOD,
                    "gaze_correction": GAZE_CORRECTION,
                    "source_file_sha256": sha256(source),
                }
            )
            imported_frames.append(imported)
            source_records.append(
                {
                    "subject_id": subject_id,
                    "site": site,
                    "probe": probe,
                    "path_relative_to_pilot_root": str(source.relative_to(pilot_root)),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                }
            )

    peaks = pd.concat(imported_frames, ignore_index=True)
    if peaks["unit_id"].duplicated().any():
        raise ValueError("Imported current unit_id values are not unique")

    mapped = peaks.merge(
        current,
        on="unit_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_current"),
    )
    if not mapped["_merge"].eq("both").all():
        raise ValueError(f"RF/current mapping failure: {mapped['_merge'].value_counts().to_dict()}")
    if not mapped["site"].eq(mapped["site_current"]).all():
        raise ValueError("Imported RF site labels disagree with current tables")
    if not mapped["subject_id"].eq(mapped["subject_id_current"]).all():
        raise ValueError("Imported RF subject IDs disagree with current tables")
    if not mapped["probe"].eq(mapped["probe_letter"]).all():
        raise ValueError("Imported RF probe labels disagree with current tables")
    mapped["default_qc"] = mapped["default_qc"].fillna(False).astype(bool)
    if not mapped.loc[mapped["pilot_qc"], "default_qc"].all():
        raise ValueError("Pilot RF QC is not a subset of current default_qc")

    keep_columns = [
        "subject_id",
        "site",
        "site_number",
        "local_unit_id",
        "unit_id",
        "probe",
        "rf_center_x_deg",
        "rf_center_y_deg",
        "rf_peak_value",
        "pilot_qc",
        "default_qc",
        "rf_method",
        "gaze_correction",
        "source_file_sha256",
    ]
    mapped = mapped[keep_columns].sort_values(["site_number", "probe", "local_unit_id"])

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for _, group in mapped.groupby(["site_number", "probe"], sort=True):
        summary_rows.append(
            summarize_group(
                group,
                rng=rng,
                iterations=args.bootstrap_iterations,
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values(["site_number", "probe"])

    declared_order = list(config["display_probe_order"])
    ordering_rows = []
    for site_number, group in summary.groupby("site_number", sort=True):
        by_probe = group.set_index("probe").loc[declared_order]
        x_values = by_probe["rf_center_x_deg"].to_numpy(dtype=float)
        strict = bool(np.all(np.diff(x_values) < 0))
        allowing_ties = bool(np.all(np.diff(x_values) <= 0))
        rho = float(spearmanr(np.arange(len(declared_order)), x_values).statistic)
        descending_order = ">".join(
            by_probe.sort_values(
                ["rf_center_x_deg", "probe"], ascending=[False, True]
            ).index.tolist()
        )
        ordering_rows.append(
            {
                "site": by_probe["site"].iloc[0],
                "site_number": int(site_number),
                "subject_id": int(by_probe["subject_id"].iloc[0]),
                "declared_order": ">".join(declared_order),
                "observed_descending_x_order": descending_order,
                "declared_order_strictly_descending_x": strict,
                "declared_order_descending_x_allowing_ties": allowing_ties,
                "spearman_declared_rank_vs_x": rho,
            }
        )
    ordering = pd.DataFrame(ordering_rows)

    peaks_path = output_dir / "rf_unit_peaks.csv"
    summary_path = output_dir / "rf_probe_summary.csv"
    ordering_path = output_dir / "rf_probe_ordering.csv"
    mapped.to_csv(peaks_path, index=False)
    summary.to_csv(summary_path, index=False)
    ordering.to_csv(ordering_path, index=False)

    pilot_code = pilot_root / "interactive_analysis.py"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "Provisional measured RF positions for Figure 3 Iteration 2.",
        "source_repository": git_record(pilot_root),
        "source_implementation": {
            "path_relative_to_pilot_root": str(pilot_code.relative_to(pilot_root)),
            "sha256": sha256(pilot_code),
        },
        "source_files": source_records,
        "method": {
            "rf_method": METHOD,
            "gaze_correction": GAZE_CORRECTION,
            "probe_center": "median per-unit peak among PilotAnalysis is_qc units",
            "uncertainty": "unit bootstrap 95% percentile interval of median",
            "bootstrap_iterations": args.bootstrap_iterations,
            "random_seed": args.seed,
            "pilot_qc_rule": "snr>5, d_prime>2, rp_contamination<0.1, default_qc",
        },
        "validation": {
            "source_file_count": len(source_records),
            "imported_units": int(len(mapped)),
            "mapped_units": int(len(mapped)),
            "pilot_qc_units": int(mapped["pilot_qc"].sum()),
            "default_qc_units": int(mapped["default_qc"].sum()),
            "probe_summary_rows": int(len(summary)),
            "pilot_qc_subset_of_default_qc": True,
            "strict_declared_order_sessions": int(
                ordering["declared_order_strictly_descending_x"].sum()
            ),
            "declared_order_sessions_allowing_ties": int(
                ordering["declared_order_descending_x_allowing_ties"].sum()
            ),
        },
        "outputs": {
            "rf_unit_peaks.csv": sha256(peaks_path),
            "rf_probe_summary.csv": sha256(summary_path),
            "rf_probe_ordering.csv": sha256(ordering_path),
        },
        "limitations": [
            "Per-unit positions are raw grid argmax locations without RF significance testing.",
            "Pilot QC is unit-quality QC, not RF-quality QC.",
            "No gaze correction is applied.",
            "The 9x9 grid discretizes positions to 10-degree steps and edge peaks are common.",
            "These positions must not be used as the final p_value_rf or area_rf filter.",
        ],
    }
    manifest_path = output_dir / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Imported {len(mapped)} units from {len(source_records)} files")
    print(
        f"Pilot QC: {mapped['pilot_qc'].sum()} units; "
        f"default_qc: {mapped['default_qc'].sum()} units"
    )
    print(
        "Declared B>C>A>E azimuth order: "
        f"{manifest['validation']['strict_declared_order_sessions']}/8 strict, "
        f"{manifest['validation']['declared_order_sessions_allowing_ties']}/8 allowing ties"
    )
    print(f"Wrote RF import snapshot to {output_dir}")


if __name__ == "__main__":
    main()
