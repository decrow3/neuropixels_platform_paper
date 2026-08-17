#!/usr/bin/env python3
"""Extract and validate full-condition grating metrics from MouseV2 NWBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_grating_metrics_v1"
DEFAULT_PILOT_RESULTS = ROOT.parent.parent / "PilotAnalysis" / "PilotAnalysis" / "results"
EXTRACTOR = ROOT / "generate_retinotopic_csvs.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Subset such as --sites site3; default is every configured site.",
    )
    parser.add_argument("--pilot-results", type=Path, default=DEFAULT_PILOT_RESULTS)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse an existing per-site grating_metrics.csv.",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Record NWB paths and sizes without reading every byte for SHA-256.",
    )
    return parser.parse_args()


def validate_metrics(path: Path, session: dict[str, object]) -> pd.DataFrame:
    table = pd.read_csv(path)
    if (
        "firing_rate_dg" not in table.columns
        and {"preferred_mean_spikes_dg", "analysis_duration_s_dg"}.issubset(
            table.columns
        )
    ):
        # Backward-compatible read of the frozen Iteration 3 import. New raw
        # extractions write this column explicitly.
        table["firing_rate_dg"] = (
            pd.to_numeric(table["preferred_mean_spikes_dg"], errors="coerce")
            / pd.to_numeric(table["analysis_duration_s_dg"], errors="coerce")
        )
    required = {
        "unit_id",
        "f1_f0_dg",
        "mod_idx_dg",
        "pref_ori_dg",
        "pref_tf_dg",
        "pref_sf_dg",
        "preferred_mean_spikes_dg",
        "firing_rate_dg",
        "preferred_condition_ties_dg",
        "preferred_trials_dg",
        "analysis_duration_s_dg",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    expected_units = int(session["expected_units"])
    if len(table) != expected_units or not table["unit_id"].is_unique:
        raise ValueError(
            f"{path}: expected {expected_units} unique units, found {len(table)}"
        )
    expected_ids = np.arange(expected_units, dtype=int) + int(session["id_offset"])
    if not np.array_equal(np.sort(table["unit_id"].to_numpy(dtype=int)), expected_ids):
        raise ValueError(f"{path}: unit IDs do not match offset manifest")
    if not table["preferred_trials_dg"].dropna().eq(15).all():
        raise ValueError(f"{path}: preferred conditions do not all have 15 trials")
    if not table["analysis_duration_s_dg"].dropna().eq(1.0).all():
        raise ValueError(f"{path}: analysis duration drifted from nominal 1.0 s")
    expected_rate = (
        pd.to_numeric(table["preferred_mean_spikes_dg"], errors="coerce")
        / pd.to_numeric(table["analysis_duration_s_dg"], errors="coerce")
    )
    if not np.allclose(
        table["firing_rate_dg"], expected_rate, equal_nan=True, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{path}: firing_rate_dg disagrees with preferred-condition rate")
    return table


def pilot_preferences(
    session: dict[str, object], pilot_results: Path
) -> pd.DataFrame | None:
    subject_id = int(session["subject_id"])
    session_dirs = sorted(pilot_results.glob(f"sub-{subject_id}_ses-ecephys-*"))
    if len(session_dirs) != 1:
        return None
    files = sorted(session_dirs[0].glob("Probe*/*_tuning_best.csv"))
    if len(files) != 4:
        return None
    table = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {
        "unit_id",
        "pref_angle",
        "temporal_frequency",
        "spatial_frequency",
        "probe",
    }
    if not required.issubset(table.columns) or table["unit_id"].duplicated().any():
        return None
    table["unit_id"] = table["unit_id"].astype(int) + int(session["id_offset"])
    return table


def make_diagnostic_figure(comparison: pd.DataFrame, output: Path) -> None:
    selected = comparison[
        comparison["default_qc"].eq(True)
        & comparison["f1_f0_pooled_sf_legacy"].gt(0)
        & comparison["f1_f0_dg"].gt(0)
    ].copy()
    selected["legacy_log"] = np.log10(selected["f1_f0_pooled_sf_legacy"])
    selected["full_log"] = np.log10(selected["f1_f0_dg"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(
        selected["legacy_log"], selected["full_log"], s=3, alpha=0.15, color="#4C78A8"
    )
    limits = [
        min(selected["legacy_log"].quantile(0.005), selected["full_log"].quantile(0.005)),
        max(selected["legacy_log"].quantile(0.995), selected["full_log"].quantile(0.995)),
    ]
    axes[0, 0].plot(limits, limits, color="black", linewidth=1)
    axes[0, 0].set(xlabel="pooled-SF legacy log10 F1/F0", ylabel="full-condition log10 F1/F0")

    bins = np.linspace(-2, 1, 70)
    axes[0, 1].hist(
        selected["legacy_log"], bins=bins, density=True, histtype="step", linewidth=2,
        label="pooled-SF legacy",
    )
    axes[0, 1].hist(
        selected["full_log"], bins=bins, density=True, histtype="step", linewidth=2,
        label="full condition",
    )
    axes[0, 1].set(xlabel="log10 F1/F0", ylabel="density")
    axes[0, 1].legend(frameon=False)

    summary = (
        comparison[comparison["default_qc"].eq(True)]
        .groupby(["site_number", "probe_letter"])[["f1_f0_dg", "mod_idx_dg"]]
        .mean()
        .reset_index()
    )
    colors = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
    for probe, group in summary.groupby("probe_letter"):
        axes[1, 0].plot(
            group["site_number"], np.log10(group["f1_f0_dg"]), "o-",
            color=colors[probe], label=probe,
        )
        axes[1, 1].plot(
            group["site_number"], np.log10(group["mod_idx_dg"]), "o-",
            color=colors[probe], label=probe,
        )
    axes[1, 0].set(xlabel="site", ylabel="mean log10 full-condition F1/F0")
    axes[1, 1].set(xlabel="site", ylabel="mean log10 modulation index")
    axes[1, 0].legend(title="probe", frameon=False, ncol=4)
    fig.suptitle("MouseV2 drifting-grating metric validation (default QC)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or session["site"] in requested
    ]
    if requested is not None and {session["site"] for session in sessions} != requested:
        raise ValueError(f"Unknown sites requested: {sorted(requested - {s['site'] for s in sessions})}")

    input_records = []
    metric_tables = []
    preference_rows = []
    command_records = []
    started = datetime.now().astimezone()
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mousev2-gratings-matplotlib")
    environment.setdefault("XDG_CACHE_HOME", "/tmp/mousev2-gratings-cache")

    for session in sessions:
        site = str(session["site"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file():
            raise FileNotFoundError(nwb_path)
        observed_bytes = nwb_path.stat().st_size
        if observed_bytes != int(session["expected_nwb_bytes"]):
            raise ValueError(
                f"{site} NWB size drift: expected {session['expected_nwb_bytes']}, found {observed_bytes}"
            )
        input_record = {
            "site": site,
            "subject_id": int(session["subject_id"]),
            "dandiset_relative_path": str(session["nwb_relative_path"]),
            "local_path": str(nwb_path),
            "bytes": observed_bytes,
        }
        if not args.skip_input_hashes:
            print(f"[{site}] hashing {nwb_path.name}", flush=True)
            input_record["sha256"] = sha256(nwb_path)
        input_records.append(input_record)

        site_dir = output_dir / site
        site_dir.mkdir(exist_ok=True)
        metrics_path = site_dir / "grating_metrics.csv"
        if not (args.reuse_existing and metrics_path.is_file()):
            command = [
                sys.executable,
                str(EXTRACTOR),
                "--nwb",
                str(nwb_path),
                "--out_dir",
                str(site_dir),
                "--site_name",
                f"V1_{site}",
                "--id_offset",
                str(session["id_offset"]),
                "--stim_table",
                "drifting_gratings_field_block_presentations",
                "--only_gratings",
            ]
            print(f"[{site}] extracting {session['expected_units']} units", flush=True)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            log_path = output_dir / "logs" / f"{site}.log"
            log_path.write_text(completed.stdout + "\n[stderr]\n" + completed.stderr, encoding="utf-8")
            command_records.append(
                {"site": site, "command": command, "exit_code": completed.returncode, "log": str(log_path)}
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{site} extraction failed; see {log_path}")
        table = validate_metrics(metrics_path, session)
        table["site"] = site
        table["site_number"] = int(session["site_number"])
        table["subject_id"] = int(session["subject_id"])

        legacy = pd.read_csv(
            ROOT / config["data_directory"] / f"{site}_processed" / "change_modulation_data.csv"
        ).rename(columns={"modulation_index": "f1_f0_pooled_sf_legacy"})
        quality = pd.read_csv(
            ROOT / config["data_directory"] / f"{site}_processed" / "unit_quality.csv"
        )[["unit_id", "default_qc"]]
        layer = pd.read_csv(
            ROOT / config["data_directory"] / f"{site}_processed" / "layer_info.csv"
        )[["unit_id", "ecephys_structure_acronym"]]
        table = (
            table.merge(legacy, on="unit_id", validate="one_to_one")
            .merge(quality, on="unit_id", validate="one_to_one")
            .merge(layer, on="unit_id", validate="one_to_one")
        )
        table["probe_letter"] = table["ecephys_structure_acronym"].str.extract(r"_([ABCE])$")
        metric_tables.append(table)

        pilot = pilot_preferences(session, args.pilot_results.resolve())
        if pilot is not None:
            joined = table.merge(
                pilot[["unit_id", "pref_angle", "temporal_frequency", "spatial_frequency", "probe"]],
                on="unit_id",
                validate="one_to_one",
            )
            ori_match = np.isclose(joined["pref_ori_dg"], joined["pref_angle"])
            tf_match = np.isclose(joined["pref_tf_dg"], joined["temporal_frequency"])
            sf_match = np.isclose(joined["pref_sf_dg"], joined["spatial_frequency"])
            preference_rows.append(
                {
                    "site": site,
                    "subject_id": int(session["subject_id"]),
                    "pilot_units": len(joined),
                    "orientation_agreement": float(np.mean(ori_match)),
                    "temporal_frequency_agreement": float(np.mean(tf_match)),
                    "spatial_frequency_agreement": float(np.mean(sf_match)),
                    "full_triplet_agreement": float(np.mean(ori_match & tf_match & sf_match)),
                }
            )
        print(f"[{site}] validated {len(table)} units", flush=True)

    comparison = pd.concat(metric_tables, ignore_index=True, sort=False)
    comparison.to_csv(output_dir / "unit_metric_comparison.csv", index=False)
    preference_validation = pd.DataFrame(preference_rows)
    preference_validation.to_csv(output_dir / "pilot_preference_validation.csv", index=False)

    summary_rows = []
    for (site, site_number), group in comparison.groupby(["site", "site_number"], sort=True):
        for population, subset in (
            ("all", group), ("default_qc", group[group["default_qc"].eq(True)])
        ):
            finite = (
                subset["f1_f0_dg"].gt(0)
                & subset["f1_f0_pooled_sf_legacy"].gt(0)
                & subset["f1_f0_dg"].notna()
                & subset["f1_f0_pooled_sf_legacy"].notna()
            )
            rho = spearmanr(
                subset.loc[finite, "f1_f0_pooled_sf_legacy"],
                subset.loc[finite, "f1_f0_dg"],
            ).statistic
            summary_rows.append(
                {
                    "site": site,
                    "site_number": site_number,
                    "population": population,
                    "units": len(subset),
                    "legacy_f1_f0_finite": int(np.isfinite(subset["f1_f0_pooled_sf_legacy"]).sum()),
                    "full_f1_f0_finite": int(np.isfinite(subset["f1_f0_dg"]).sum()),
                    "mod_idx_finite": int(np.isfinite(subset["mod_idx_dg"]).sum()),
                    "legacy_f1_f0_median": float(np.nanmedian(subset["f1_f0_pooled_sf_legacy"])),
                    "full_f1_f0_median": float(np.nanmedian(subset["f1_f0_dg"])),
                    "mod_idx_median": float(np.nanmedian(subset["mod_idx_dg"])),
                    "legacy_vs_full_spearman": float(rho),
                    "preference_ties_gt1": int((subset["preferred_condition_ties_dg"] > 1).sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "site_metric_summary.csv", index=False)
    make_diagnostic_figure(comparison, output_dir / "grating_metric_diagnostics.png")

    output_names = [
        "unit_metric_comparison.csv",
        "pilot_preference_validation.csv",
        "site_metric_summary.csv",
        "grating_metric_diagnostics.png",
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "started_at": started.isoformat(),
        "dandiset": config["nwb_input"],
        "sites": [session["site"] for session in sessions],
        "inputs": input_records,
        "commands": command_records,
        "outputs": [
            {"path": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
            for name in output_names
        ],
        "code": {
            "extractor": {"path": str(EXTRACTOR), "sha256": sha256(EXTRACTOR)},
            "metrics": {
                "path": str(ROOT / "common" / "drifting_gratings.py"),
                "sha256": sha256(ROOT / "common" / "drifting_gratings.py"),
            },
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "environment": {"python": sys.version, "executable": sys.executable},
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    all_summary = summary[summary["population"] == "all"]
    qc_summary = summary[summary["population"] == "default_qc"]
    report = [
        "# MouseV2 drifting-grating extraction",
        "",
        f"Processed {len(sessions)} sessions and {len(comparison):,} units.",
        "Preferred conditions include orientation, temporal frequency, and spatial frequency.",
        "F1/F0 and the Allen Welch-spectrum modulation index are stored separately.",
        "",
        "## Aggregate checks",
        "",
        f"- Full-condition F1/F0 finite: {int(np.isfinite(comparison['f1_f0_dg']).sum()):,}/{len(comparison):,}.",
        f"- Modulation index finite: {int(np.isfinite(comparison['mod_idx_dg']).sum()):,}/{len(comparison):,}.",
        f"- Median site-level legacy/full F1/F0 Spearman rho (all units): {all_summary['legacy_vs_full_spearman'].median():.3f}.",
        f"- Median full-condition F1/F0 across default-QC site medians: {qc_summary['full_f1_f0_median'].median():.3f}.",
    ]
    if not preference_validation.empty:
        report.extend(
            [
                f"- Pilot full-triplet preference agreement: {preference_validation['full_triplet_agreement'].mean():.1%} mean across sessions.",
                "",
                "Pilot tuning is an independent diagnostic, not the source of the imported metric.",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Completed grating import: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
