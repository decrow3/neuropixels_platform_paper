"""Load and validate the provisional PilotAnalysis RF import contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RF_IMPORT_DIR = ROOT / "data" / "imports" / "pilot_rf_peaks_v1"

REQUIRED_PEAK_COLUMNS = {
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
}

REQUIRED_SUMMARY_COLUMNS = {
    "site",
    "site_number",
    "subject_id",
    "probe",
    "n_units_used",
    "rf_center_x_deg",
    "rf_center_x_ci_low_deg",
    "rf_center_x_ci_high_deg",
    "rf_center_y_deg",
    "rf_center_y_ci_low_deg",
    "rf_center_y_ci_high_deg",
    "rf_grid_edge_fraction",
    "qc_rule",
    "rf_method",
    "gaze_correction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rf_import(
    import_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return per-unit peaks, probe summary, ordering, and provenance manifest."""
    directory = Path(import_dir) if import_dir is not None else DEFAULT_RF_IMPORT_DIR
    directory = directory.resolve()
    paths = {
        "rf_unit_peaks.csv": directory / "rf_unit_peaks.csv",
        "rf_probe_summary.csv": directory / "rf_probe_summary.csv",
        "rf_probe_ordering.csv": directory / "rf_probe_ordering.csv",
        "import_manifest.json": directory / "import_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete RF import snapshot: {missing}")

    manifest = json.loads(paths["import_manifest.json"].read_text(encoding="utf-8"))
    for name in ("rf_unit_peaks.csv", "rf_probe_summary.csv", "rf_probe_ordering.csv"):
        expected = manifest["outputs"].get(name)
        observed = sha256(paths[name])
        if expected != observed:
            raise ValueError(
                f"RF import checksum mismatch for {name}: expected {expected}, observed {observed}"
            )

    peaks = pd.read_csv(paths["rf_unit_peaks.csv"])
    summary = pd.read_csv(paths["rf_probe_summary.csv"])
    ordering = pd.read_csv(paths["rf_probe_ordering.csv"])

    missing_peak_columns = REQUIRED_PEAK_COLUMNS.difference(peaks.columns)
    missing_summary_columns = REQUIRED_SUMMARY_COLUMNS.difference(summary.columns)
    if missing_peak_columns:
        raise ValueError(f"RF unit import missing columns: {sorted(missing_peak_columns)}")
    if missing_summary_columns:
        raise ValueError(
            f"RF probe summary missing columns: {sorted(missing_summary_columns)}"
        )
    if peaks["unit_id"].duplicated().any():
        raise ValueError("RF unit import has duplicate current unit_id values")
    if summary.duplicated(["site_number", "probe"]).any():
        raise ValueError("RF probe summary has duplicate session-probe rows")
    return peaks, summary, ordering, manifest
