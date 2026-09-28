#!/usr/bin/env python3
"""Run the staged Figure 4 rerun.

The preflight stage is intentionally cheap and response-blind with respect to
new effects.  It writes the registry, design/weight audits, RF refit inventory,
and a content-hashed configuration.  Primary inference refuses to start until
the completed RF source named by the frozen configuration exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure4_rerun_core import (
    apply_cell_floor, build_response_registry,
    category_recording_weights, cooccurrence_summary, registry_summary,
    sha256, source_manifest, write_json,
)

DEFAULT_CONFIG = ROOT / "config/figure4_rerun_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/figure4_rerun/v1"
LEGACY_RF = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
RF_PEAKS = ROOT / "data/imports/pilot_rf_peaks_v1/rf_unit_peaks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "primary"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or "unavailable"


def preflight(config_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    registry = build_response_registry()
    gzip = {"method": "gzip", "mtime": 0}
    registry.to_csv(output / "response_registry_draw_level.csv.gz", index=False, compression=gzip)
    floor = 5
    retained = apply_cell_floor(registry, floor=floor)
    retained.to_csv(output / "response_registry_floor5.csv.gz", index=False, compression=gzip)
    registry_summary(retained).to_csv(output / "recording_registry_summary.csv", index=False)
    cooccurrence_summary(retained).to_csv(output / "category_cooccurrence.csv", index=False)

    weight_rows = []
    for (metric, population, draw_id), part in retained.groupby(["metric", "population", "draw_id"], sort=False):
        if population == "Central":
            continue
        # Each draw is a distinct point-estimation population.  Non-timescale
        # metrics have the sentinel draw -1.
        weights = category_recording_weights(part)
        for category, total in weights.groupby(part.category).sum().items():
            weight_rows.append({
                "metric": metric, "population": population, "draw_id": draw_id,
                "category": category, "weight": total,
            })
    pd.DataFrame(weight_rows).to_csv(output / "weight_audit.csv", index=False)

    response_units = retained.loc[retained.source.eq("MouseV2"), ["unit_id"]].drop_duplicates()
    peaks = pd.read_csv(RF_PEAKS, dtype={"unit_id": str})[[
        "unit_id", "site", "site_number", "subject_id", "probe", "pilot_qc", "default_qc",
    ]]
    response_units = response_units.merge(peaks, on="unit_id", validate="one_to_one")
    legacy = pd.read_csv(LEGACY_RF, dtype={"unit_id": str})[["unit_id", "rf_model_supported"]]
    response_units = response_units.merge(legacy, on="unit_id", how="left", validate="one_to_one", indicator=True)
    response_units["legacy_fit_available"] = response_units._merge.eq("both")
    response_units["legacy_supported_rf"] = response_units.rf_model_supported.eq(True)
    response_units["analysis_selected"] = True
    response_units.drop(columns=["_merge"]).to_csv(output / "mousev2_rf_refit_inventory.csv", index=False)

    cooccurrence = cooccurrence_summary(retained)
    rf_total = len(response_units)
    rf_fitted = int(response_units.legacy_fit_available.sum())
    rf_supported = int(response_units.legacy_supported_rf.sum())
    report = [
        "# Figure 4 rerun preflight checkpoint", "",
        f"Frozen preflight configuration SHA-256: `{config_hash}`.", "",
        "This checkpoint reconstructs response eligibility from the accepted metric sources without using legacy Figure 4 session lists. It does not inspect or report the new category-dispersion effects.", "",
        "## Recording design", "",
        f"The response registry contains {retained.unit_id.nunique():,} distinct cells after the common five-cell recording/category floor. MouseV2 contributes {retained.loc[retained.source.eq('MouseV2'),'animal_id'].nunique()} animals/sessions. Every retained V1 category has at least 7 independent physical recordings. Allen animal and session IDs remain one-to-one in this population; Allen categories have 17–50 animals depending on metric/category.", "",
        f"Across all metric/population/category-pair rows, the minimum retained same-session co-occurrence is {int(cooccurrence.sessions_with_both.min())}. Exact counts are in `category_cooccurrence.csv`; cell/animal/session/probe counts are in `recording_registry_summary.csv`.", "",
        "The weight audit confirms 1/K total weight for every category in every estimable metric/population/draw population. An animal is split across repeated recordings before weight is split across cells.", "",
        "## Concrete RF coverage gate", "",
        f"The union of current MouseV2 response populations contains {rf_total:,} cells. The legacy parametric table contains fits for {rf_fitted:,} ({rf_fitted/rf_total:.1%}); only {rf_supported:,} ({rf_supported/rf_total:.1%}) currently pass its support flag. Absence from that table is classified as unattempted under the old Pilot-QC fit population, not as RF failure.", "",
        "The smallest useful next step is to fit only the missing current-population cells while reusing existing raw fits, then recompute both FDR gates over the complete selected population. Run:", "",
        "```bash",
        "MPLCONFIGDIR=/tmp/mpl-figure4-rerun python scripts/extract_mousev2_parametric_rf.py \\",
        "  --analysis-units artifacts/figure4_rerun/v1/mousev2_rf_refit_inventory.csv \\",
        "  --selection-column analysis_selected \\",
        "  --reuse-fits data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv \\",
        "  --output-dir data/imports/mousev2_parametric_rf_figure4_v1",
        "```", "",
        "This reads the eight local NWB files and is intentionally separated from preflight because it is the first expensive stage. After it completes, the next checkpoint must inspect accepted, rejected, borderline, lateral, and medial fits and freeze the two-dimensional support/reference rule before any response-effect bootstrap.", "",
        "## Current decision", "",
        "Recording replication and category co-occurrence pass the coarse preflight gate. Primary inference remains blocked by incomplete RF coverage and the consequent unfrozen RF support/reference distribution. No population-level Figure E estimate has been generated.",
    ]
    (output / "PREFLIGHT_CHECKPOINT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "preflight_complete_primary_blocked_pending_rf_refit",
        "config_path": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": config_hash,
        "git_revision": git_revision(),
        "sources": source_manifest(),
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "preflight_manifest.json":
            manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(output / "preflight_manifest.json", manifest)
    (output / "FROZEN_CONFIG_SHA256.txt").write_text(config_hash + "\n", encoding="utf-8")
    print(registry_summary(retained).to_string(index=False))
    print(f"\nFrozen config SHA-256: {config_hash}")
    print(f"Preflight outputs: {output}")


def primary(config_path: Path, output: Path) -> None:
    config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
    rf_path = ROOT / config["rf"]["completed_source"]
    if not rf_path.is_file():
        raise FileNotFoundError(
            "Primary inference is gated: completed current-population RF fits are missing at "
            f"{rf_path}. Run the RF command recorded in the preflight report first."
        )
    raise NotImplementedError(
        "The production bootstrap is deliberately unavailable until the RF/support checkpoint is reviewed."
    )


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        preflight(args.config.resolve(), args.output_dir.resolve())
    else:
        primary(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
