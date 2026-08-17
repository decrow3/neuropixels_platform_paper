#!/usr/bin/env python3
"""Extract versioned pooled and polarity-specific MouseV2 flash metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.flashes import (  # noqa: E402
    FLASH_VARIANTS,
    TIMESCALE_BIN_EDGES_S,
    compute_flash_metrics,
    prepare_flash_presentations,
    timescale_bin_mask,
)
from generate_retinotopic_csvs import (  # noqa: E402
    _read_numeric_dset,
    read_nwb_tables,
)


DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_flash_metrics_v1"
GRATING_MANIFEST = (
    ROOT / "data" / "imports" / "mousev2_grating_metrics_v1" / "import_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse existing per-site flash_metrics.csv files.",
    )
    return parser.parse_args()


def validate_metrics(path: Path, session: dict[str, object]) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"unit_id"}
    for variant in FLASH_VARIANTS:
        required.update(
            {
                f"time_to_first_spike_{variant}",
                f"ttfs_valid_trials_{variant}",
                f"autocorr_tau_{variant}",
                f"err_ac_{variant}",
                f"spike_count_ac_{variant}",
                f"timescale_fit_ok_{variant}",
                f"flash_trials_{variant}",
            }
        )
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    expected = int(session["expected_units"])
    expected_ids = np.arange(expected, dtype=int) + int(session["id_offset"])
    if len(table) != expected or not table["unit_id"].is_unique:
        raise ValueError(f"{path}: expected {expected} unique units, found {len(table)}")
    if not np.array_equal(np.sort(table["unit_id"].to_numpy(dtype=int)), expected_ids):
        raise ValueError(f"{path}: unit IDs do not match the session offset")
    for variant, trials in (("pooled", 300), ("bright", 150), ("dark", 150)):
        if not table[f"flash_trials_{variant}"].eq(trials).all():
            raise ValueError(f"{path}: {variant} trial count drift")
    return table


def read_timing_audit(nwb_path: Path, flash_table: pd.DataFrame) -> dict[str, object]:
    """Record what the NWB proves—and does not prove—about flash timing."""
    fid = h5py.h5f.open(str(nwb_path).encode(), flags=h5py.h5f.ACC_RDONLY)
    with h5py.File(nwb_path, "r") as nwb:
        timestamp_path = "/processing/stimulus/timestamps/timestamps"
        processed = (
            _read_numeric_dset(fid, timestamp_path)
            if timestamp_path.strip("/") in nwb
            else np.array([], dtype=float)
        )
        names: list[str] = []
        nwb.visit(names.append)
    fid.close()

    starts = flash_table["start_time"].to_numpy(dtype=float)
    stops = flash_table["stop_time"].to_numpy(dtype=float)
    durations = stops - starts
    in_processed = np.isin(starts, processed)
    photodiode_paths = [name for name in names if "photodiode" in name.lower()]
    return {
        "presentations": len(flash_table),
        "bright_presentations": int(flash_table["flash_polarity"].eq("bright").sum()),
        "dark_presentations": int(flash_table["flash_polarity"].eq("dark").sum()),
        "duration_median_s": float(np.median(durations)),
        "duration_min_s": float(np.min(durations)),
        "duration_max_s": float(np.max(durations)),
        "inter_start_median_s": float(np.median(np.diff(starts))),
        "frame_period_from_15_flash_frames_s": float(np.median(durations) / 15.0),
        "processed_stimulus_timestamps": int(len(processed)),
        "flash_starts_in_processed_timestamps": int(in_processed.sum()),
        "flash_starts_exactly_matched": bool(in_processed.all()),
        "photodiode_dataset_paths": ";".join(photodiode_paths),
        "timing_interpretation": (
            "NWB interval starts equal processed stimulus timestamps; physical light-onset "
            "provenance and display latency are not encoded"
        ),
    }


def site_comparison(
    metrics: pd.DataFrame,
    session: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    site = str(session["site"])
    old_ttfs = pd.read_csv(
        ROOT / "data" / f"{site}_processed" / "time_to_first_spike.csv"
    )
    old_tau = pd.read_csv(
        ROOT / "data" / f"{site}_processed" / "timescale_metrics.csv"
    )
    quality = pd.read_csv(
        ROOT / "data" / f"{site}_processed" / "unit_quality.csv"
    )[["unit_id", "default_qc"]]
    joined = (
        metrics.merge(old_ttfs, on="unit_id", validate="one_to_one")
        .merge(old_tau, on="unit_id", validate="one_to_one")
        .merge(quality, on="unit_id", validate="one_to_one")
    )
    joined["site"] = site
    joined["subject_id"] = int(session["subject_id"])
    joined["ttfs_delta_pooled_minus_legacy_ms"] = 1000 * (
        joined["time_to_first_spike_pooled"] - joined["time_to_first_spike"]
    )
    joined["timescale_delta_pooled_minus_legacy_ms"] = (
        joined["autocorr_tau_pooled"] - joined["autocorr_tau"]
    )

    ttfs_shared = joined[["time_to_first_spike_pooled", "time_to_first_spike"]].notna().all(axis=1)
    tau_shared = joined[["autocorr_tau_pooled", "autocorr_tau"]].notna().all(axis=1)
    tau_rho = (
        float(spearmanr(
            joined.loc[tau_shared, "autocorr_tau_pooled"],
            joined.loc[tau_shared, "autocorr_tau"],
        ).statistic)
        if tau_shared.sum() >= 3
        else np.nan
    )
    summary: dict[str, object] = {
        "site": site,
        "subject_id": int(session["subject_id"]),
        "units": len(joined),
        "ttfs_shared_units": int(ttfs_shared.sum()),
        "ttfs_exact_match_fraction": float(
            np.isclose(
                joined.loc[ttfs_shared, "time_to_first_spike_pooled"],
                joined.loc[ttfs_shared, "time_to_first_spike"],
                rtol=0.0,
                atol=1e-12,
            ).mean()
        ),
        "timescale_shared_units": int(tau_shared.sum()),
        "timescale_spearman_legacy_vs_center_corrected": tau_rho,
        "timescale_median_delta_ms": float(
            joined.loc[tau_shared, "timescale_delta_pooled_minus_legacy_ms"].median()
        ),
    }
    for variant in FLASH_VARIANTS:
        summary[f"ttfs_finite_{variant}"] = int(
            joined[f"time_to_first_spike_{variant}"].notna().sum()
        )
        summary[f"timescale_fit_ok_{variant}"] = int(
            joined[f"timescale_fit_ok_{variant}"].astype(bool).sum()
        )
        selected = joined["default_qc"].eq(True)
        summary[f"default_qc_ttfs_median_{variant}_ms"] = float(
            1000 * joined.loc[selected, f"time_to_first_spike_{variant}"].median()
        )
        summary[f"default_qc_timescale_median_{variant}_ms"] = float(
            joined.loc[selected, f"autocorr_tau_{variant}"].median()
        )
    return joined, summary


def make_diagnostic_figure(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    qc = comparison[comparison["default_qc"].eq(True)].copy()

    shared = qc[["time_to_first_spike", "time_to_first_spike_pooled"]].notna().all(axis=1)
    axes[0, 0].scatter(
        1000 * qc.loc[shared, "time_to_first_spike"],
        1000 * qc.loc[shared, "time_to_first_spike_pooled"],
        s=3,
        alpha=0.18,
    )
    axes[0, 0].plot([30, 200], [30, 200], color="black", linewidth=1)
    axes[0, 0].set(xlabel="legacy pooled TTFS (ms)", ylabel="versioned pooled TTFS (ms)")

    shared = qc[["autocorr_tau", "autocorr_tau_pooled"]].notna().all(axis=1)
    axes[0, 1].scatter(
        qc.loc[shared, "autocorr_tau"],
        qc.loc[shared, "autocorr_tau_pooled"],
        s=3,
        alpha=0.18,
    )
    axes[0, 1].plot([1, 1000], [1, 1000], color="black", linewidth=1)
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="legacy timescale: left-edge mask (ms)",
        ylabel="center-corrected timescale (ms)",
    )

    axes[1, 0].scatter(
        1000 * qc["time_to_first_spike_bright"],
        1000 * qc["time_to_first_spike_dark"],
        s=3,
        alpha=0.18,
    )
    axes[1, 0].plot([30, 200], [30, 200], color="black", linewidth=1)
    axes[1, 0].set(xlabel="bright TTFS (ms)", ylabel="dark TTFS (ms)")

    axes[1, 1].scatter(
        qc["autocorr_tau_bright"],
        qc["autocorr_tau_dark"],
        s=3,
        alpha=0.18,
    )
    axes[1, 1].plot([1, 1000], [1, 1000], color="black", linewidth=1)
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        xlabel="bright timescale (ms)",
        ylabel="dark timescale (ms)",
    )
    fig.suptitle("MouseV2 flash metric validation (default QC)")
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

    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or session["site"] in requested
    ]
    observed = {str(session["site"]) for session in sessions}
    if requested is not None and observed != requested:
        raise ValueError(f"Unknown sites requested: {sorted(requested - observed)}")

    prior_manifest = json.loads(GRATING_MANIFEST.read_text(encoding="utf-8"))
    prior_inputs = {record["site"]: record for record in prior_manifest["inputs"]}
    comparisons: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    started = datetime.now().astimezone()

    for session in sessions:
        site = str(session["site"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file():
            raise FileNotFoundError(nwb_path)
        if nwb_path.stat().st_size != int(session["expected_nwb_bytes"]):
            raise ValueError(f"{site} NWB size drift")
        prior = prior_inputs[site]
        if prior["bytes"] != nwb_path.stat().st_size:
            raise ValueError(f"{site} no longer matches the previously hashed input")
        inputs.append(
            {
                "site": site,
                "subject_id": int(session["subject_id"]),
                "dandiset_relative_path": str(session["nwb_relative_path"]),
                "local_path": str(nwb_path),
                "bytes": nwb_path.stat().st_size,
                "sha256": prior["sha256"],
                "sha256_source": str(GRATING_MANIFEST.relative_to(ROOT)),
            }
        )

        site_dir = output_dir / site
        site_dir.mkdir(exist_ok=True)
        metrics_path = site_dir / "flash_metrics.csv"
        if not (args.reuse_existing and metrics_path.is_file()):
            print(f"[{site}] reading NWB and computing flash metrics", flush=True)
            extracted = read_nwb_tables(str(nwb_path))
            flash_name = next(
                (name for name in extracted.intervals_tables if "flash" in name.lower()),
                None,
            )
            if flash_name is None:
                raise ValueError(f"{site} has no flash interval table")
            flashes = prepare_flash_presentations(extracted.intervals_tables[flash_name])
            if len(flashes) != 300:
                raise ValueError(f"{site}: expected 300 flash presentations")
            timing = read_timing_audit(nwb_path, flashes)
            timing.update({"site": site, "subject_id": int(session["subject_id"])})
            timing_rows.append(timing)

            metrics = compute_flash_metrics(extracted.spikes_by_unit, flashes)
            metrics["unit_id"] = metrics["unit_id"].astype(int) + int(session["id_offset"])
            metrics.to_csv(metrics_path, index=False)
        else:
            # Timing audit remains raw-input provenance even on a metric reuse.
            extracted = read_nwb_tables(str(nwb_path))
            flash_name = next(
                name for name in extracted.intervals_tables if "flash" in name.lower()
            )
            flashes = prepare_flash_presentations(extracted.intervals_tables[flash_name])
            timing = read_timing_audit(nwb_path, flashes)
            timing.update({"site": site, "subject_id": int(session["subject_id"])})
            timing_rows.append(timing)

        metrics = validate_metrics(metrics_path, session)
        comparison, summary = site_comparison(metrics, session)
        comparisons.append(comparison)
        summaries.append(summary)
        print(
            f"[{site}] TTFS exact={summary['ttfs_exact_match_fraction']:.3f}; "
            f"timescale rho={summary['timescale_spearman_legacy_vs_center_corrected']:.3f}",
            flush=True,
        )

    comparison_table = pd.concat(comparisons, ignore_index=True)
    comparison_table.to_csv(output_dir / "unit_metric_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "site_metric_summary.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output_dir / "timing_audit.csv", index=False)
    make_diagnostic_figure(comparison_table, output_dir / "flash_metric_diagnostics.png")

    readme = [
        "# MouseV2 full-field flash metrics v1",
        "",
        "Per-unit pooled, bright, and dark TTFS and response-decay timescales",
        "computed from the eight versioned NWBs.",
        "",
        "- TTFS is the median first occupied 1-ms bin in the released 30–200 ms window.",
        "- Timescale uses 10-ms AllenSDK bin centers in the released 40–290 ms selection",
        "  (25 bins centered at 45–285 ms), then the released bounded exponential fit.",
        "- Bright is contrast +1 (white); dark is contrast −1 (black).",
        "- Latencies are raw relative to NWB interval `start_time`; no cross-dataset",
        "  mean matching or display-latency correction is applied.",
        "- NWB starts exactly match the processed stimulus timestamp series, but no",
        "  photodiode trace or physical light-onset provenance is encoded.",
        "",
        "See `timing_audit.csv`, `site_metric_summary.csv`, and `import_manifest.json`.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    output_names = (
        "unit_metric_comparison.csv",
        "site_metric_summary.csv",
        "timing_audit.csv",
        "flash_metric_diagnostics.png",
        "README.md",
    )
    outputs = [file_record(output_dir / name) for name in output_names]
    for session in sessions:
        outputs.append(file_record(output_dir / str(session["site"]) / "flash_metrics.csv"))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "started_at": started.isoformat(),
        "dandiset": {
            "dandiset_id": config["nwb_input"]["dandiset_id"],
            "dandiset_version": config["nwb_input"]["dandiset_version"],
        },
        "sites": [str(session["site"]) for session in sessions],
        "flash_definition": {
            "ttfs_window_ms": [30, 200],
            "ttfs_bin_ms": 1,
            "timescale_bin_ms": 10,
            "timescale_selection_s": [0.04, 0.29],
            "timescale_selected_bin_centers_s": (
                (
                    TIMESCALE_BIN_EDGES_S[:-1]
                    + np.diff(TIMESCALE_BIN_EDGES_S) / 2
                )[timescale_bin_mask()]
            ).tolist(),
            "polarity": {"bright": 1.0, "dark": -1.0},
            "timing_reference": "NWB interval start_time; uncalibrated physical light onset",
        },
        "inputs": inputs,
        "outputs": outputs,
        "code": {
            "metrics": file_record(ROOT / "common" / "flashes.py"),
            "extractor": file_record(Path(__file__).resolve()),
            "nwb_reader": file_record(ROOT / "generate_retinotopic_csvs.py"),
            "test": file_record(ROOT / "tests" / "test_flashes.py"),
        },
        "environment": {"python": platform.python_version()},
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Flash metric import written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
