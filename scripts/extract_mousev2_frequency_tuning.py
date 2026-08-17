#!/usr/bin/env python3
"""Extract trial-derived MouseV2 marginal SF x TF surfaces and support masks."""

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

from common.drifting_gratings import (  # noqa: E402
    benjamini_hochberg,
    compute_frequency_tuning_surfaces,
)
from common.parametric_models import fit_parametric_grating_models  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_frequency_tuning_v1"
DEFAULT_ANALYSIS_UNITS = ROOT / "data" / "imports" / "pilot_rf_peaks_v1" / "rf_unit_peaks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-units", type=Path, default=DEFAULT_ANALYSIS_UNITS)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-reliability", type=float, default=0.3)
    parser.add_argument("--minimum-parametric-r2", type=float, default=0.1)
    parser.add_argument("--minimum-cell-trials", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_support_contract(
    metrics: pd.DataFrame,
    *,
    fdr_alpha: float,
    minimum_reliability: float,
    minimum_parametric_r2: float = 0.1,
) -> pd.DataFrame:
    """Apply dataset-wide multiplicity correction and blank unsupported preferences."""
    result = metrics.copy()
    for stem in ("sf_tf_joint", "sf_main", "tf_main", "surface_reliability"):
        result[f"{stem}_q"] = benjamini_hochberg(result[f"{stem}_p"].to_numpy())
    result["surface_reliable"] = (
        result["surface_reliability_q"].le(fdr_alpha)
        & result["surface_split_half_spearman_brown"].ge(minimum_reliability)
    )
    has_parametric = "parametric_lrt_p" in result.columns
    if has_parametric:
        result["parametric_lrt_q"] = benjamini_hochberg(
            result["parametric_lrt_p"].to_numpy()
        )
        result["parametric_model_supported"] = (
            result["parametric_fit_success"].fillna(False).astype(bool)
            & result["parametric_lrt_q"].le(fdr_alpha)
            & result["parametric_pseudo_r2"].ge(minimum_parametric_r2)
        )
        parametric = result["parametric_model_supported"]
        sf_identified = ~result["parametric_sf_at_extrapolation_bound"].fillna(True).astype(bool)
        tf_identified = ~result["parametric_tf_at_extrapolation_bound"].fillna(True).astype(bool)
        sf_identified &= result["sf_sigma_octaves"].between(0.151, 3.99)
        tf_identified &= result["tf_sigma_octaves"].between(0.151, 4.99)
        sf_preference = result["parametric_pref_sf_cpd"]
        tf_preference = result["parametric_pref_tf_hz"]
    else:
        result["parametric_model_supported"] = True
        parametric = pd.Series(True, index=result.index)
        sf_identified = pd.Series(True, index=result.index)
        tf_identified = pd.Series(True, index=result.index)
        sf_preference = result["surface_peak_sf_cpd"]
        tf_preference = result["surface_peak_tf_hz"]
    joint = result["sf_tf_joint_q"].le(fdr_alpha)
    result["sf_preference_supported"] = (
        joint
        & result["sf_main_q"].le(fdr_alpha)
        & result["surface_reliable"]
        & parametric
        & sf_identified
    )
    result["tf_preference_supported"] = (
        joint
        & result["tf_main_q"].le(fdr_alpha)
        & result["surface_reliable"]
        & parametric
        & tf_identified
    )
    result["pref_sf_supported_dg"] = sf_preference.where(
        result["sf_preference_supported"]
    )
    result["pref_tf_supported_dg"] = tf_preference.where(
        result["tf_preference_supported"]
    )
    if has_parametric:
        result["sf_preference_extrapolated"] = (
            result["sf_preference_supported"]
            & ~result["parametric_sf_in_tested_range"].fillna(False).astype(bool)
        )
        result["tf_preference_extrapolated"] = (
            result["tf_preference_supported"]
            & ~result["parametric_tf_in_tested_range"].fillna(False).astype(bool)
        )
    return result


def write_readme(
    support: pd.DataFrame,
    *,
    fdr_alpha: float,
    minimum_reliability: float,
    minimum_parametric_r2: float,
    output_path: Path,
) -> None:
    session_rows = []
    for site, group in support.groupby("site", sort=True):
        session_rows.append(
            f"| {site} | {len(group):,} | {group['sf_preference_supported'].sum():,} "
            f"({group['sf_preference_supported'].mean():.1%}) | "
            f"{group['tf_preference_supported'].sum():,} "
            f"({group['tf_preference_supported'].mean():.1%}) |"
        )
    lines = [
        "# MouseV2 trial-derived SF x TF tuning",
        "",
        "Each unit is fit with a joint Poisson count model using log-Gaussian SF and",
        "TF terms plus an orientation-periodic von Mises term over all 100 conditions.",
        "The empirical 5 x 5 surface remains an orientation-marginal diagnostic.",
        "Tuning is tested from presentation-level",
        "spike counts with a joint 25-cell omnibus F test and separate marginal SF",
        "and TF F tests. Reliability is the correlation between balanced alternating",
        "repeat halves of the 25-cell surface, reported with Spearman-Brown correction.",
        "",
        f"Support requires dataset-wide BH-FDR q <= {fdr_alpha:g} for the joint and",
        "axis-specific tuning tests, BH-FDR significance for positive reliability,",
        f"corrected split-half reliability >= {minimum_reliability:g}, parametric",
        f"pseudo-R2 >= {minimum_parametric_r2:g}, identified widths, and a peak not pinned",
        "to the one-octave extrapolation bound. Peaks outside the tested range are",
        "retained and explicitly flagged.",
        "Preferences that fail this contract are stored as missing and must not be mapped.",
        "",
        "| Session | Units | Supported SF | Supported TF |",
        "| --- | ---: | ---: | ---: |",
        *session_rows,
        "",
        "Files:",
        "",
        "- `frequency_tuning_support.csv`: one row per unit, tests, q-values, and gated preferences.",
        "- `site*/frequency_tuning_surface.csv.gz`: one row per unit and SF x TF cell.",
        "- `run_manifest.json`: inputs, thresholds, code, and output hashes.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Keep the pure support-contract helpers importable in lightweight test
    # environments that do not include the NWB/HDF5 dependency.
    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    args = parse_args()
    if not 0 < args.fdr_alpha < 1:
        raise ValueError("--fdr-alpha must be between zero and one")
    if not -1 < args.minimum_reliability <= 1:
        raise ValueError("--minimum-reliability must be in (-1, 1]")
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

    requested = set(args.sites) if args.sites else None
    analysis_units = pd.read_csv(
        args.analysis_units.resolve(), usecols=["unit_id", "pilot_qc"]
    )
    analysis_unit_ids = set(
        analysis_units.loc[analysis_units["pilot_qc"].fillna(False), "unit_id"].astype(int)
    )
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or session["site"] in requested
    ]
    found = {str(session["site"]) for session in sessions}
    if requested is not None and found != requested:
        raise ValueError(f"Unknown sites requested: {sorted(requested - found)}")

    metric_tables = []
    inputs = []
    for session in sessions:
        site = str(session["site"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file():
            raise FileNotFoundError(nwb_path)
        if nwb_path.stat().st_size != int(session["expected_nwb_bytes"]):
            raise ValueError(f"{site}: NWB byte size differs from the frozen config")
        print(f"[{site}] reading trials and spikes", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        _, presentations = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        metrics, surfaces = compute_frequency_tuning_surfaces(
            extracted.units_df.index,
            extracted.spikes_by_unit,
            presentations,
            min_trials_per_sf_tf=args.minimum_cell_trials,
        )
        offset = int(session["id_offset"])
        local_parametric_ids = [
            int(unit_id - offset)
            for unit_id in analysis_unit_ids
            if offset <= unit_id < offset + int(session["expected_units"])
        ]
        parametric = fit_parametric_grating_models(
            local_parametric_ids,
            extracted.spikes_by_unit,
            presentations,
        )
        metrics = metrics.merge(parametric, on="unit_id", how="left", validate="one_to_one")
        metrics["unit_id"] = metrics["unit_id"].astype(int) + offset
        surfaces["unit_id"] = surfaces["unit_id"].astype(int) + offset
        metrics["site"] = site
        metrics["subject_id"] = int(session["subject_id"])
        metric_tables.append(metrics)
        site_dir = output_dir / site
        site_dir.mkdir(exist_ok=True)
        surfaces.to_csv(
            site_dir / "frequency_tuning_surface.csv.gz",
            index=False,
            float_format="%.7g",
            compression="gzip",
        )
        inputs.append(
            {
                "site": site,
                "path": str(nwb_path),
                "bytes": nwb_path.stat().st_size,
            }
        )
        print(
            f"[{site}] extracted {len(metrics):,} empirical surfaces; "
            f"fit {len(parametric):,} independently QC-selected units",
            flush=True,
        )

    support = apply_support_contract(
        pd.concat(metric_tables, ignore_index=True),
        fdr_alpha=args.fdr_alpha,
        minimum_reliability=args.minimum_reliability,
        minimum_parametric_r2=args.minimum_parametric_r2,
    )
    support.to_csv(
        output_dir / "frequency_tuning_support.csv", index=False, float_format="%.7g"
    )
    write_readme(
        support,
        fdr_alpha=args.fdr_alpha,
        minimum_reliability=args.minimum_reliability,
        minimum_parametric_r2=args.minimum_parametric_r2,
        output_path=output_dir / "README.md",
    )
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "status": "trial-derived marginal SF x TF surfaces with gated preferences",
        "inputs": inputs,
        "parameters": {
            "fdr_method": "Benjamini-Hochberg across all extracted units",
            "fdr_alpha": args.fdr_alpha,
            "minimum_split_half_spearman_brown": args.minimum_reliability,
            "minimum_parametric_pseudo_r2": args.minimum_parametric_r2,
            "minimum_trials_per_sf_tf_cell": args.minimum_cell_trials,
            "response": "spike count in nominal 1.0-s presentation",
            "parametric_fit_population": "independently defined Pilot-QC units",
            "surface": "joint Poisson log-Gaussian(SF) x log-Gaussian(TF) x von-Mises(orientation)",
            "split": "alternating repeats balanced across nuisance conditions",
        },
        "code": {
            "extractor": str(Path(__file__).resolve()),
            "extractor_sha256": sha256(Path(__file__).resolve()),
            "metrics": str((ROOT / "common" / "drifting_gratings.py").resolve()),
            "metrics_sha256": sha256(ROOT / "common" / "drifting_gratings.py"),
            "parametric_model": str((ROOT / "common" / "parametric_models.py").resolve()),
            "parametric_model_sha256": sha256(ROOT / "common" / "parametric_models.py"),
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Frequency-tuning support written to {output_dir}")


if __name__ == "__main__":
    main()
