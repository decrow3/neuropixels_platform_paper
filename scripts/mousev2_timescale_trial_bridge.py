#!/usr/bin/env python3
"""Match MouseV2 response-timescale trials to Allen's 75 bright + 75 dark flashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.flashes import (  # noqa: E402
    bin_trial_spike_counts,
    fit_response_timescale,
    prepare_flash_presentations,
)


DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "mousev2_timescale_trial_bridge_v1"
FLASH_IMPORT = ROOT / "data" / "imports" / "mousev2_flash_metrics_v1"
FLASH_MANIFEST = FLASH_IMPORT / "import_manifest.json"
ALLEN_SESSION_SUMMARY = (
    ROOT / "artifacts" / "figure3" / "06b_v1_dataset_bridge" / "session_metric_summary.csv"
)
WINDOW_EDGES_S = np.arange(4, 30, dtype=float) / 100.0
TRIALS_PER_POLARITY = 75


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
    parser.add_argument("--trial-subsamples", type=int, default=10)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def is_valid_timescale(table: pd.DataFrame) -> pd.Series:
    tau = pd.to_numeric(table["timescale_ms"], errors="coerce")
    error = pd.to_numeric(table["fit_error_ms"], errors="coerce")
    spikes = pd.to_numeric(table["spike_count"], errors="coerce")
    return tau.between(1, 300) & spikes.gt(50) & error.lt(20)


def balanced_subsample_masks(
    flashes: pd.DataFrame,
    *,
    n_subsamples: int,
    seed: int,
) -> list[np.ndarray]:
    bright = np.flatnonzero(flashes["flash_polarity"].eq("bright").to_numpy())
    dark = np.flatnonzero(flashes["flash_polarity"].eq("dark").to_numpy())
    if len(bright) != 150 or len(dark) != 150:
        raise ValueError(
            f"Expected 150 MouseV2 flashes per polarity; got {len(bright)} and {len(dark)}"
        )
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_subsamples):
        selected = np.concatenate(
            [
                rng.choice(bright, TRIALS_PER_POLARITY, replace=False),
                rng.choice(dark, TRIALS_PER_POLARITY, replace=False),
            ]
        )
        mask = np.zeros(len(flashes), dtype=bool)
        mask[selected] = True
        masks.append(mask)
    return masks


def summarize_units(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view, session_id, subsample), group in units.groupby(
        ["view", "session_id", "subsample"], sort=True
    ):
        valid = group["valid_timescale"].astype(bool)
        values = pd.to_numeric(group.loc[valid, "timescale_ms"], errors="coerce")
        rows.append(
            {
                "view": view,
                "session_id": int(session_id),
                "subsample": int(subsample),
                "input_units": len(group),
                "fit_attempted_units": int(group["fit_attempted"].sum()),
                "fit_ok_units": int(group["fit_ok"].sum()),
                "valid_units": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "mean_timescale_ms": values.mean(),
                "median_timescale_ms": values.median(),
            }
        )
    return pd.DataFrame(rows)


def analysis_centers(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = summary.loc[summary["view"].eq("mouse_full_300")]
    rows.append(
        {
            "population": "MouseV2 V1",
            "view": "mouse_full_300",
            "subsample": -1,
            "sessions": full["session_id"].nunique(),
            "equal_session_mean_timescale_ms": full["mean_timescale_ms"].mean(),
            "equal_session_mean_valid_fraction": full["valid_fraction"].mean(),
        }
    )
    matched = summary.loc[summary["view"].eq("mouse_matched_150")]
    for subsample, group in matched.groupby("subsample"):
        rows.append(
            {
                "population": "MouseV2 V1",
                "view": "mouse_matched_150",
                "subsample": int(subsample),
                "sessions": group["session_id"].nunique(),
                "equal_session_mean_timescale_ms": group[
                    "mean_timescale_ms"
                ].mean(),
                "equal_session_mean_valid_fraction": group["valid_fraction"].mean(),
            }
        )
    allen = pd.read_csv(ALLEN_SESSION_SUMMARY)
    allen = allen.loc[
        allen["metric"].eq("timescale_valid_ms")
        & allen["cohort"].str.startswith("Allen")
    ]
    for cohort, group in allen.groupby("cohort"):
        rows.append(
            {
                "population": cohort,
                "view": "allen_released_150",
                "subsample": -1,
                "sessions": group["session_id"].nunique(),
                "equal_session_mean_timescale_ms": group["mean"].mean(),
                "equal_session_mean_valid_fraction": np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    summary: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    full = summary.loc[summary["view"].eq("mouse_full_300")].set_index("session_id")
    matched = summary.loc[summary["view"].eq("mouse_matched_150")]
    matched_center = matched.groupby("session_id")["mean_timescale_ms"].mean()
    for session_id in full.index.intersection(matched_center.index):
        axes[0].plot(
            [0, 1],
            [full.loc[session_id, "mean_timescale_ms"], matched_center.loc[session_id]],
            color="#888888",
            alpha=0.7,
        )
    axes[0].scatter(np.zeros(len(full)), full["mean_timescale_ms"], color="#D95F02")
    axes[0].scatter(np.ones(len(matched_center)), matched_center, color="#D95F02")
    axes[0].set(xticks=[0, 1], xticklabels=["300 flashes", "150 matched"], ylabel="session mean valid timescale (ms)")

    mouse_full = centers.loc[centers["view"].eq("mouse_full_300"), "equal_session_mean_timescale_ms"].iloc[0]
    mouse_matched = centers.loc[centers["view"].eq("mouse_matched_150"), "equal_session_mean_timescale_ms"]
    axes[1].scatter(np.zeros(len(mouse_matched)), mouse_matched, color="#D95F02", alpha=0.75)
    axes[1].scatter([0], [mouse_full], marker="D", color="black", label="Mouse 300")
    for index, (cohort, color) in enumerate(
        [("Allen Brain Observatory 1.1", "#6F63A6"), ("Allen Functional Connectivity", "#B07AA1")],
        start=1,
    ):
        value = centers.loc[centers["population"].eq(cohort), "equal_session_mean_timescale_ms"].iloc[0]
        axes[1].scatter([index], [value], color=color, s=50)
    axes[1].set(xticks=[0, 1, 2], xticklabels=["Mouse\n150 draws", "Allen BO\n150", "Allen FC\n150"], ylabel="equal-session mean timescale (ms)")
    axes[1].legend(frameon=False, fontsize=8)

    full_fraction = full["valid_fraction"]
    matched_fraction = matched.groupby("session_id")["valid_fraction"].mean()
    for session_id in full_fraction.index.intersection(matched_fraction.index):
        axes[2].plot([0, 1], [full_fraction.loc[session_id], matched_fraction.loc[session_id]], color="#888888", alpha=0.7)
    axes[2].scatter(np.zeros(len(full_fraction)), full_fraction, color="#D95F02")
    axes[2].scatter(np.ones(len(matched_fraction)), matched_fraction, color="#D95F02")
    axes[2].set(xticks=[0, 1], xticklabels=["300 flashes", "150 matched"], ylabel="fraction passing timescale validity")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("MouseV2 timescale sensitivity to Allen-matched flash counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, summary: pd.DataFrame, centers: pd.DataFrame) -> None:
    full = centers.loc[centers["view"].eq("mouse_full_300")].iloc[0]
    matched = centers.loc[centers["view"].eq("mouse_matched_150")]
    matched_values = matched["equal_session_mean_timescale_ms"]
    delta = matched_values - float(full["equal_session_mean_timescale_ms"])
    allen = centers.loc[centers["view"].eq("allen_released_150")]
    result = [
        "# MouseV2–Allen timescale trial bridge",
        "",
        "MouseV2 was repeatedly downsampled from 150 bright + 150 dark flashes",
        "to Allen's 75 bright + 75 dark flashes. The binning, exponential fit,",
        "and validity rules were otherwise unchanged.",
        "",
        f"- MouseV2, 300 flashes: {float(full['equal_session_mean_timescale_ms']):.2f} ms equal-session mean.",
        f"- MouseV2, matched 150 flashes: {matched_values.mean():.2f} ms "
        f"(trial-draw range {matched_values.min():.2f}–{matched_values.max():.2f}).",
        f"- Trial-count change: {delta.mean():+.2f} ms "
        f"(draw range {delta.min():+.2f} to {delta.max():+.2f}).",
    ]
    for row in allen.itertuples():
        difference = matched_values.mean() - row.equal_session_mean_timescale_ms
        result.append(
            f"- Matched MouseV2 minus {row.population}: {difference:+.2f} ms "
            f"({int(row.sessions)} Allen sessions)."
        )
    full_fraction = float(full["equal_session_mean_valid_fraction"])
    matched_fraction = matched["equal_session_mean_valid_fraction"]
    result.extend(
        [
            f"- Mean session validity fraction changes from {full_fraction:.3f} to "
            f"{matched_fraction.mean():.3f} after trial matching.",
            "",
            "This is a trial-count sensitivity, not a display-latency calibration.",
            "The matched analysis preserves the balanced polarities and reports the",
            "selection-flow change caused by halving MouseV2's flash trials.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(result) + "\n", encoding="utf-8")


def refresh_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "import_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_inputs = {
        record["site"]: record
        for record in json.loads(FLASH_MANIFEST.read_text(encoding="utf-8"))["inputs"]
    }
    for record in manifest.get("inputs", []):
        prior = prior_inputs[record["site"]]
        record["sha256"] = prior["sha256"]
        record["sha256_source"] = str(FLASH_MANIFEST.relative_to(ROOT))
    manifest["outputs"] = []
    for name in (
        "README.md",
        "analysis_centers.csv",
        "session_summary.csv",
        "timescale_trial_bridge.png",
        "unit_subsample_metrics.csv",
    ):
        path = output_dir / name
        if path.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest["code"]["script_sha256"] = sha256(Path(__file__).resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    summary = pd.read_csv(output_dir / "session_summary.csv")
    centers = pd.read_csv(output_dir / "analysis_centers.csv")
    make_figure(summary, centers, output_dir / "timescale_trial_bridge.png")
    write_report(output_dir, summary, centers)
    refresh_manifest(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        render_existing(output_dir)
        print(f"Rendered MouseV2 timescale bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or str(session["site"]) in requested
    ]
    if requested is not None and {str(s["site"]) for s in sessions} != requested:
        raise ValueError("One or more requested sites are not in the config")

    from generate_retinotopic_csvs import read_nwb_tables

    rows = []
    for session in sessions:
        site = str(session["site"])
        site_number = int(session["site_number"])
        offset = int(session["id_offset"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file() or nwb_path.stat().st_size != int(
            session["expected_nwb_bytes"]
        ):
            raise FileNotFoundError(f"Missing or size-mismatched MouseV2 NWB: {nwb_path}")
        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")
        quality = quality.loc[quality["default_qc"].eq(True), ["unit_id"]]
        print(f"[{site}] reading raw NWB for {len(quality)} common-QC units", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        flash_name = next(
            name for name in extracted.intervals_tables if "flash" in name.lower()
        )
        flashes = prepare_flash_presentations(extracted.intervals_tables[flash_name])
        if len(flashes) != 300:
            raise ValueError(f"{site}: expected 300 flash presentations")
        masks = balanced_subsample_masks(
            flashes,
            n_subsamples=args.trial_subsamples,
            seed=20260805 + site_number,
        )
        starts = flashes["start_time"].to_numpy(dtype=float)
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        local_ids = quality["unit_id"].astype(int).to_numpy() - offset
        if not set(local_ids).issubset(row_by_id):
            raise ValueError(f"{site}: common-QC unit IDs missing from NWB")
        for unit_index, (output_id, local_id) in enumerate(
            zip(quality["unit_id"].astype(int), local_ids), start=1
        ):
            spikes = extracted.spikes_by_unit[row_by_id[int(local_id)]]
            counts = bin_trial_spike_counts(spikes, starts, WINDOW_EDGES_S)
            for subsample, mask in enumerate(masks):
                selected = counts[mask]
                spike_count = float(selected.sum())
                attempted = spike_count > 50
                if attempted:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tau, error, _, fit_ok = fit_response_timescale(
                            selected,
                            bin_edges_s=WINDOW_EDGES_S,
                        )
                else:
                    tau, error, fit_ok = np.nan, np.nan, False
                rows.append(
                    {
                        "view": "mouse_matched_150",
                        "session_id": site_number,
                        "site": site,
                        "unit_id": int(output_id),
                        "subsample": subsample,
                        "flash_trials": int(mask.sum()),
                        "bright_trials": TRIALS_PER_POLARITY,
                        "dark_trials": TRIALS_PER_POLARITY,
                        "timescale_ms": tau,
                        "fit_error_ms": error,
                        "spike_count": spike_count,
                        "fit_attempted": attempted,
                        "fit_ok": fit_ok,
                    }
                )
            if unit_index % 250 == 0:
                print(f"[{site}] fitted {unit_index}/{len(quality)} units", flush=True)

    matched_units = pd.DataFrame(rows)
    matched_units["valid_timescale"] = is_valid_timescale(matched_units)

    reference = pd.read_csv(FLASH_IMPORT / "unit_metric_comparison.csv", low_memory=False)
    selected_sites = {str(session["site"]) for session in sessions}
    reference = reference.loc[
        reference["default_qc"].eq(True) & reference["site"].isin(selected_sites)
    ].copy()
    full_units = pd.DataFrame(
        {
            "view": "mouse_full_300",
            "session_id": reference["site"].str.extract(r"(\d+)")[0].astype(int),
            "site": reference["site"],
            "unit_id": reference["unit_id"].astype(int),
            "subsample": -1,
            "flash_trials": 300,
            "bright_trials": 150,
            "dark_trials": 150,
            "timescale_ms": reference["autocorr_tau_pooled"],
            "fit_error_ms": reference["err_ac_pooled"],
            "spike_count": reference["spike_count_ac_pooled"],
            "fit_attempted": pd.to_numeric(
                reference["spike_count_ac_pooled"], errors="coerce"
            ).gt(50),
            "fit_ok": reference["timescale_fit_ok_pooled"].astype(bool),
        }
    )
    full_units["valid_timescale"] = is_valid_timescale(full_units)
    all_units = pd.concat([full_units, matched_units], ignore_index=True)
    all_units.to_csv(output_dir / "unit_subsample_metrics.csv", index=False)
    summary = summarize_units(all_units)
    summary.to_csv(output_dir / "session_summary.csv", index=False)
    centers = analysis_centers(summary)
    centers.to_csv(output_dir / "analysis_centers.csv", index=False)
    write_report(output_dir, summary, centers)
    if not args.skip_figure:
        make_figure(summary, centers, output_dir / "timescale_trial_bridge.png")

    prior_inputs = {
        record["site"]: record
        for record in json.loads(FLASH_MANIFEST.read_text(encoding="utf-8"))["inputs"]
    }
    manifest = {
        "schema_version": 1,
        "trial_subsamples": args.trial_subsamples,
        "trial_support": {"pooled": 150, "bright": 75, "dark": 75},
        "timescale_window_edges_s": WINDOW_EDGES_S.tolist(),
        "inputs": [
            {
                "site": str(session["site"]),
                "path": str(nwb_root / str(session["nwb_relative_path"])),
                "bytes": int(session["expected_nwb_bytes"]),
                "sha256": prior_inputs[str(session["site"])]["sha256"],
                "sha256_source": str(FLASH_MANIFEST.relative_to(ROOT)),
            }
            for session in sessions
        ],
        "code": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "metric_sha256": sha256(ROOT / "common" / "flashes.py"),
            "config_sha256": sha256(config_path),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir)
    print(f"MouseV2 timescale bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
