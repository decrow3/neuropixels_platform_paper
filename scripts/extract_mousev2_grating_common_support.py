#!/usr/bin/env python3
"""Recompute MouseV2 grating metrics on the Allen-compatible condition subset.

This is the first raw-data leg of the V1 bridge.  It restricts MouseV2 to the
Allen spatial frequency (0.04 cycles/degree) while retaining the shared
orientation, temporal-frequency, contrast, 1-s window, and 15-trial support.
It does not solve the remaining Allen 2-s/window/repeat mismatch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import compute_drifting_grating_metrics  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_grating_common_support_v1"
SOURCE_METRICS = ROOT / "data" / "imports" / "mousev2_grating_metrics_v1"
ORIENTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
TEMPORAL_FREQUENCIES_HZ = (1.0, 2.0, 4.0, 8.0, 15.0)
SPATIAL_FREQUENCY_CPD = 0.04
CONTRAST = 0.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-metrics-dir", type=Path, default=SOURCE_METRICS)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Extract and summarize without rendering (useful in the pinned NWB environment).",
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Render the figure from an existing unit_metric_comparison.csv and exit.",
    )
    return parser.parse_args()


def common_presentations(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "orientation",
        "temporal_frequency",
        "spatial_frequency",
        "contrast",
        "start_time",
        "stop_time",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Grating table lacks common-support columns {missing}")
    numeric = table.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    selected = numeric.loc[
        numeric["orientation"].isin(ORIENTATIONS_DEG)
        & numeric["temporal_frequency"].isin(TEMPORAL_FREQUENCIES_HZ)
        & np.isclose(numeric["spatial_frequency"], SPATIAL_FREQUENCY_CPD)
        & np.isclose(numeric["contrast"], CONTRAST)
    ].copy()
    dimensions = ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]
    counts = selected.groupby(dimensions).size()
    if len(counts) != 20 or not counts.eq(15).all() or len(selected) != 300:
        raise ValueError(
            "Expected 20 shared ori x TF conditions with 15 trials each; "
            f"found {len(counts)} conditions and {len(selected)} presentations"
        )
    return selected


def validate_metrics(table: pd.DataFrame, session: dict[str, object]) -> None:
    if len(table) != int(session["expected_units"]) or not table["unit_id"].is_unique:
        raise ValueError(f"{session['site']}: unit count/identity validation failed")
    if not table["pref_sf_dg"].dropna().eq(SPATIAL_FREQUENCY_CPD).all():
        raise ValueError(f"{session['site']}: preferred SF escaped common support")
    if not table["preferred_trials_dg"].dropna().eq(15).all():
        raise ValueError(f"{session['site']}: preferred trial count is not 15")
    if not table["analysis_duration_s_dg"].dropna().eq(1.0).all():
        raise ValueError(f"{session['site']}: analysis duration is not 1 s")


def diagnostic_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = comparison.loc[comparison["default_qc"].eq(True)].copy()
    metrics = (
        ("mod_idx_dg", "log10 modulation index"),
        ("f1_f0_dg", "log10 F1/F0"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
    for ax, (metric, label) in zip(axes, metrics):
        current = f"{metric}_full_space"
        common = f"{metric}_common_support"
        selected[f"log_{current}"] = np.log10(selected[current].where(selected[current] > 0))
        selected[f"log_{common}"] = np.log10(selected[common].where(selected[common] > 0))
        session = (
            selected.groupby(["site_number", "probe_letter"])[
                [f"log_{current}", f"log_{common}"]
            ]
            .mean()
            .reset_index()
        )
        for probe, group in session.groupby("probe_letter"):
            for _, row in group.iterrows():
                x = row["site_number"]
                ax.plot(
                    [x - 0.12, x + 0.12],
                    [row[f"log_{current}"], row[f"log_{common}"]],
                    color=colors[probe],
                    alpha=0.35,
                    linewidth=1,
                )
            ax.scatter(
                group["site_number"] - 0.12,
                group[f"log_{current}"],
                color=colors[probe],
                marker="o",
                s=28,
                label=f"probe {probe}" if metric == "mod_idx_dg" else None,
            )
            ax.scatter(
                group["site_number"] + 0.12,
                group[f"log_{common}"],
                facecolor="white",
                edgecolor=colors[probe],
                marker="o",
                s=34,
                linewidth=1.5,
            )
        ax.set(
            xlabel="MouseV2 session (left=full SF space; right=SF 0.04)",
            ylabel=label,
        )
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("MouseV2 raw bridge: effect of restricting preference to Allen SF")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    if args.render_existing:
        comparison_path = output_dir / "unit_metric_comparison.csv"
        if not comparison_path.is_file():
            raise FileNotFoundError(comparison_path)
        figure_path = output_dir / "common_support_diagnostic.png"
        diagnostic_figure(pd.read_csv(comparison_path), figure_path)
        manifest_path = output_dir / "import_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refreshed_outputs = []
            for record in manifest.get("outputs", []):
                if record.get("path") == figure_path.name:
                    continue
                path = output_dir / str(record["path"])
                if path.is_file():
                    refreshed_outputs.append(
                        {
                            "path": record["path"],
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
            manifest["outputs"] = refreshed_outputs
            manifest["outputs"].append(
                {
                    "path": figure_path.name,
                    "bytes": figure_path.stat().st_size,
                    "sha256": sha256(figure_path),
                }
            )
            manifest.setdefault("code", {})["script"] = sha256(
                Path(__file__).resolve()
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        print(f"Rendered common-support diagnostic: {figure_path}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import the pinned h5py-based reader only for raw extraction. This keeps
    # --render-existing usable from the modern plotting environment.
    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or session["site"] in requested
    ]
    observed_sites = {session["site"] for session in sessions}
    if requested is not None and observed_sites != requested:
        raise ValueError(f"Unknown sites: {sorted(requested - observed_sites)}")

    comparison_frames = []
    output_records = []
    for session in sessions:
        site = str(session["site"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file() or nwb_path.stat().st_size != int(
            session["expected_nwb_bytes"]
        ):
            raise FileNotFoundError(f"Missing or changed NWB: {nwb_path}")
        print(f"[{site}] loading raw NWB", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        _, gratings = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        shared = common_presentations(gratings)
        print(f"[{site}] computing {len(shared)} common-support presentations", flush=True)
        metrics = compute_drifting_grating_metrics(
            extracted.units_df.index.values,
            extracted.spikes_by_unit,
            shared,
        )
        metrics["unit_id"] = metrics["unit_id"].astype(int) + int(session["id_offset"])
        validate_metrics(metrics, session)
        site_dir = output_dir / site
        site_dir.mkdir()
        metrics_path = site_dir / "grating_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        output_records.append(
            {"path": str(metrics_path.relative_to(output_dir)), "sha256": sha256(metrics_path)}
        )

        source = pd.read_csv(
            args.source_metrics_dir.resolve() / site / "grating_metrics.csv"
        )[["unit_id", "f1_f0_dg", "mod_idx_dg"]].rename(
            columns={
                "f1_f0_dg": "f1_f0_dg_full_space",
                "mod_idx_dg": "mod_idx_dg_full_space",
            }
        )
        common = metrics[["unit_id", "f1_f0_dg", "mod_idx_dg", "pref_tf_dg"]].rename(
            columns={
                "f1_f0_dg": "f1_f0_dg_common_support",
                "mod_idx_dg": "mod_idx_dg_common_support",
                "pref_tf_dg": "pref_tf_dg_common_support",
            }
        )
        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")[
            ["unit_id", "default_qc"]
        ]
        layer = pd.read_csv(ROOT / "data" / f"{site}_processed" / "layer_info.csv")[
            ["unit_id", "ecephys_structure_acronym"]
        ]
        joined = (
            source.merge(common, on="unit_id", validate="one_to_one")
            .merge(quality, on="unit_id", validate="one_to_one")
            .merge(layer, on="unit_id", validate="one_to_one")
        )
        joined["site"] = site
        joined["site_number"] = int(session["site_number"])
        joined["probe_letter"] = joined["ecephys_structure_acronym"].str.extract(
            r"_([ABCE])$", expand=False
        )
        comparison_frames.append(joined)
        print(f"[{site}] wrote {len(metrics)} units", flush=True)
        del extracted, metrics, source, common, quality, layer, joined
        gc.collect()

    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison_path = output_dir / "unit_metric_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    if not args.skip_figure:
        diagnostic_figure(comparison, output_dir / "common_support_diagnostic.png")

    rows = []
    qc = comparison.loc[comparison["default_qc"].eq(True)].copy()
    for site, group in qc.groupby("site", sort=True):
        row: dict[str, object] = {"site": site, "units": len(group)}
        for metric in ("mod_idx_dg", "f1_f0_dg"):
            for space in ("full_space", "common_support"):
                values = pd.to_numeric(group[f"{metric}_{space}"], errors="coerce")
                log_values = np.log10(values.where(values > 0))
                row[f"mean_log10_{metric}_{space}"] = log_values.mean()
                row[f"median_{metric}_{space}"] = values.median()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "site_summary.csv", index=False)

    mod_delta = (
        summary["mean_log10_mod_idx_dg_common_support"]
        - summary["mean_log10_mod_idx_dg_full_space"]
    )
    f1_delta = (
        summary["mean_log10_f1_f0_dg_common_support"]
        - summary["mean_log10_f1_f0_dg_full_space"]
    )
    report = [
        "# MouseV2 Allen-condition-support grating bridge",
        "",
        f"Processed {len(sessions)} sessions and {len(comparison):,} units from raw NWBs.",
        "Each unit's preference was recomputed using only SF = 0.04 cycles/degree,",
        "the four shared MouseV2 orientations, all five shared TFs, contrast 0.8,",
        "15 trials per condition, and the unchanged 1-s MouseV2 response window.",
        "",
        "## Result",
        "",
        f"- Equal-site common-support mean log10 modulation index: {summary['mean_log10_mod_idx_dg_common_support'].mean():+.3f}.",
        f"- Equal-site change in mean log10 modulation index: {mod_delta.mean():+.3f} "
        f"(site range {mod_delta.min():+.3f} to {mod_delta.max():+.3f}).",
        f"- Equal-site common-support mean log10 F1/F0: {summary['mean_log10_f1_f0_dg_common_support'].mean():+.3f}.",
        f"- Equal-site change in mean log10 F1/F0: {f1_delta.mean():+.3f} "
        f"(site range {f1_delta.min():+.3f} to {f1_delta.max():+.3f}).",
        "",
        "This isolates the MouseV2 preferred-SF condition-space effect. It does not",
        "yet test Allen's 2-s window, Welch grid, Functional Connectivity repeat",
        "count, flash protocol, or population support; those require original Allen NWBs.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    named_outputs = [
        comparison_path,
        output_dir / "site_summary.csv",
        output_dir / "README.md",
    ]
    figure_path = output_dir / "common_support_diagnostic.png"
    if figure_path.is_file():
        named_outputs.append(figure_path)
    manifest = {
        "schema_version": 1,
        "condition_support": {
            "orientation_deg": ORIENTATIONS_DEG,
            "temporal_frequency_hz": TEMPORAL_FREQUENCIES_HZ,
            "spatial_frequency_cpd": SPATIAL_FREQUENCY_CPD,
            "contrast": CONTRAST,
            "trials_per_condition": 15,
            "analysis_duration_s": 1.0,
        },
        "sessions": [session["site"] for session in sessions],
        "nwb_inputs": [
            {
                "site": session["site"],
                "path": session["nwb_relative_path"],
                "bytes": session["expected_nwb_bytes"],
            }
            for session in sessions
        ],
        "per_site_outputs": output_records,
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in named_outputs
        ],
        "code": {
            "script": sha256(Path(__file__).resolve()),
            "metric": sha256(ROOT / "common" / "drifting_gratings.py"),
            "nwb_reader": sha256(ROOT / "generate_retinotopic_csvs.py"),
            "config": sha256(config_path),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Common-support grating bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
