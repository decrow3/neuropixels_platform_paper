#!/usr/bin/env python3
"""Fit trial-derived elliptical Gaussian RF models for independently QC-selected MouseV2 units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import benjamini_hochberg  # noqa: E402
from common.parametric_models import fit_parametric_rf_models  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_UNITS = ROOT / "data" / "imports" / "pilot_rf_peaks_v1" / "rf_unit_peaks.csv"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_parametric_rf_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--analysis-units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-reliability", type=float, default=0.3)
    parser.add_argument("--minimum-pseudo-r2", type=float, default=0.1)
    parser.add_argument(
        "--selection-column", default="pilot_qc",
        help="Boolean analysis-units column defining the fit population",
    )
    parser.add_argument(
        "--reuse-fits", type=Path, default=None,
        help="Optional prior fit table; matching selected units are reused and support/FDR are recomputed jointly",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_rf_support(
    table: pd.DataFrame,
    *,
    fdr_alpha: float,
    minimum_reliability: float,
    minimum_pseudo_r2: float,
) -> pd.DataFrame:
    result = table.copy()
    result["rf_lrt_q"] = benjamini_hochberg(result["rf_lrt_p"].to_numpy())
    result["rf_reliability_q"] = benjamini_hochberg(
        result["rf_reliability_p"].to_numpy()
    )
    result["rf_model_supported"] = (
        result["rf_fit_success"].fillna(False).astype(bool)
        & result["rf_center_on_screen"].fillna(False).astype(bool)
        & result["rf_lrt_q"].le(fdr_alpha)
        & result["rf_reliability_q"].le(fdr_alpha)
        & result["rf_split_half_spearman_brown"].ge(minimum_reliability)
        & result["rf_pseudo_r2"].ge(minimum_pseudo_r2)
        & result["rf_sigma_major_deg"].between(3.05, 79.5)
        & result["rf_sigma_minor_deg"].between(3.05, 79.5)
    )
    for column in ("rf_center_x_deg", "rf_center_y_deg"):
        result[f"supported_{column}"] = result[column].where(result["rf_model_supported"])
    return result


def main() -> None:
    from generate_retinotopic_csvs import read_nwb_tables

    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_metadata = pd.read_csv(args.analysis_units.resolve(), low_memory=False)
    if args.selection_column not in unit_metadata:
        raise ValueError(f"Analysis units lack selection column {args.selection_column!r}")
    selected_metadata = unit_metadata.loc[
        unit_metadata[args.selection_column].fillna(False).astype(bool)
    ].copy()
    reused = None
    if args.reuse_fits is not None:
        reused = pd.read_csv(args.reuse_fits.resolve())
        reused["unit_id"] = reused.unit_id.astype(str)
        if reused.unit_id.duplicated().any():
            raise ValueError("Reuse fit table contains duplicate unit IDs")

    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or session["site"] in requested
    ]
    frames = []
    inputs = []
    for session in sessions:
        site = str(session["site"])
        offset = int(session["id_offset"])
        site_metadata = selected_metadata.loc[selected_metadata["site"].eq(site)].copy()
        reuse_site = (
            reused.loc[reused.unit_id.isin(site_metadata.unit_id.astype(str))].copy()
            if reused is not None else pd.DataFrame()
        )
        missing_metadata = site_metadata.loc[
            ~site_metadata.unit_id.astype(str).isin(set(reuse_site.get("unit_id", [])))
        ].copy()
        local_ids = missing_metadata["unit_id"].astype(int).to_numpy() - offset
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        print(f"[{site}] reusing {len(reuse_site):,}; fitting {len(local_ids):,} elliptical Gaussian RFs", flush=True)
        if len(reuse_site):
            derived = [
                "rf_lrt_q", "rf_reliability_q", "rf_model_supported",
                "supported_rf_center_x_deg", "supported_rf_center_y_deg",
            ]
            frames.append(reuse_site.drop(columns=derived, errors="ignore"))
        if len(local_ids):
            extracted = read_nwb_tables(str(nwb_path))
            rf_table = extracted.intervals_tables["receptive_field_block_presentations"]
            fits = fit_parametric_rf_models(local_ids, extracted.spikes_by_unit, rf_table)
            fits["unit_id"] = fits["unit_id"].astype(int) + offset
            fits = missing_metadata[
                ["unit_id", "site", "site_number", "subject_id", "probe", "pilot_qc", "default_qc"]
            ].merge(fits, on="unit_id", validate="one_to_one")
            frames.append(fits)
            inputs.append({"site": site, "path": str(nwb_path), "bytes": nwb_path.stat().st_size,
                           "new_fits": len(local_ids), "reused_fits": len(reuse_site)})

    supported = apply_rf_support(
        pd.concat(frames, ignore_index=True),
        fdr_alpha=args.fdr_alpha,
        minimum_reliability=args.minimum_reliability,
        minimum_pseudo_r2=args.minimum_pseudo_r2,
    )
    output_table = output_dir / "rf_unit_fits.csv"
    supported.to_csv(output_table, index=False, float_format="%.7g")
    session_summary = (
        supported.groupby("site", observed=True)
        .agg(units=("unit_id", "size"), supported_rf=("rf_model_supported", "sum"))
        .reset_index()
    )
    session_summary["supported_fraction"] = session_summary["supported_rf"] / session_summary["units"]
    session_summary.to_csv(output_dir / "rf_session_summary.csv", index=False, float_format="%.7g")
    summary_lines = [
        "| Session | Units | Supported RF | Fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in session_summary.itertuples(index=False):
        summary_lines.append(
            f"| {row.site} | {int(row.units):,} | {int(row.supported_rf):,} | "
            f"{row.supported_fraction:.1%} |"
        )
    lines = [
        "# MouseV2 parametric receptive fields",
        "",
        "Every independently Pilot-QC-selected unit was fit over all 4,860 RF",
        "presentations using a Poisson baseline-plus-rotated-elliptical-Gaussian model.",
        "Support requires dataset-wide BH-FDR significance for the model and positive",
        f"split-half reliability, corrected reliability >= {args.minimum_reliability:g},",
        f"pseudo-R2 >= {args.minimum_pseudo_r2:g}, interior width parameters, and an",
        "on-screen fitted center.",
        "Unsupported centers remain missing in the mapping contract.",
        "",
        *summary_lines,
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "status": "trial-derived supported elliptical Gaussian RF models",
        "inputs": inputs,
        "parameters": {
            "fit_population": f"Rows selected by analysis-units column {args.selection_column!r}",
            "selection_column": args.selection_column,
            "reuse_fits": str(args.reuse_fits.resolve()) if args.reuse_fits else None,
            "model": "Poisson baseline plus rotated elliptical 2D Gaussian",
            "fdr_alpha": args.fdr_alpha,
            "minimum_split_half_spearman_brown": args.minimum_reliability,
            "minimum_pseudo_r2": args.minimum_pseudo_r2,
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "model_path": str((ROOT / "common" / "parametric_models.py").resolve()),
            "model_sha256": sha256(ROOT / "common" / "parametric_models.py"),
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Supported RF models: {supported['rf_model_supported'].sum():,}/{len(supported):,}; "
        f"written to {output_dir}"
    )


if __name__ == "__main__":
    main()
