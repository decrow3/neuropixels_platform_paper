#!/usr/bin/env python3
"""Render inventory and compact-extraction QA for the multisession RF pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {
    746083955: Path("/media/huklaban5/Data/MouseV2/allen_v1_bridge/000021/sub-726170927/sub-726170927_ses-746083955.nwb"),
    755434585: Path("/media/huklaban5/Data/MouseV2/allen_bo11_rf_pilot_cache/session_755434585/session_755434585.nwb"),
    760693773: Path("/media/huklaban5/Data/MouseV2/allen_bo11_rf_pilot_cache/session_760693773/session_760693773.nwb"),
    798911424: Path("/media/huklaban5/Data/MouseV2/allen_bo11_rf_pilot_cache/session_798911424/session_798911424.nwb"),
}
DEFAULT_CACHE = ROOT / "artifacts" / "allen_population_gaze_rf"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_multisession_rf_validation_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def collect(cache_root):
    rows = []
    for sid, nwb in SESSIONS.items():
        cache = cache_root / f"session_{sid}"
        population = pd.read_csv(cache / "visual_unit_population.csv", low_memory=False)
        trials = pd.read_csv(cache / "gabor_trial_gaze_table.csv", low_memory=False)
        cache_bytes = sum(p.stat().st_size for p in cache.glob("*") if p.is_file())
        rows.append({
            "session_id": sid,
            "nwb_path": str(nwb),
            "nwb_gib": nwb.stat().st_size / 2**30,
            "cache_mib": cache_bytes / 2**20,
            "visual_units": len(population),
            "qc_units": int(population["published_like_qc"].sum()),
            "qc_fraction": float(population["published_like_qc"].mean()),
            "gabor_presentations": len(trials),
            "valid_gaze_fraction": float(trials["valid_gaze"].mean()),
            "gaze_dx_sd_deg": float(trials.loc[trials["valid_gaze"], "gaze_dx_deg"].std()),
            "gaze_dy_sd_deg": float(trials.loc[trials["valid_gaze"], "gaze_dy_deg"].std()),
        })
    return pd.DataFrame(rows)


def render_inventory(table, path):
    labels = table["session_id"].astype(str)
    x = np.arange(len(table))
    usage = shutil.disk_usage("/media/huklaban5/Data")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    axes[0].bar(x, table["nwb_gib"], color="#3366aa")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="NWB size (GiB)", title="Four local native sessions")
    axes[1].bar([0], [usage.used / 2**40], color="#d97736", label="Used")
    axes[1].bar([0], [usage.free / 2**40], bottom=[usage.used / 2**40], color="#d7e3f4", label="Free")
    axes[1].set(xticks=[0], xticklabels=["Allen data disk"], ylabel="Capacity (TiB)",
                title=f"{usage.free / 2**40:.1f} TiB remains")
    axes[1].legend(frameon=False)
    axes[2].bar(x, table["cache_mib"], color="#7a8f3a")
    axes[2].set(xticks=x, xticklabels=labels, ylabel="Compact cache (MiB)",
                title="Reusable extraction footprint")
    for axis in axes: axis.tick_params(axis="x", rotation=35); axis.grid(axis="y", alpha=.16)
    fig.suptitle("Multisession Allen RF pilot: local-data inventory", fontsize=15)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def render_ingestion(table, path):
    labels = table["session_id"].astype(str); x = np.arange(len(table)); width=.36
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5), constrained_layout=True)
    axes[0,0].bar(x-width/2, table["visual_units"], width, color="#8aaed6", label="Visual units")
    axes[0,0].bar(x+width/2, table["qc_units"], width, color="#3366aa", label="RF/QC units")
    axes[0,0].set(ylabel="Units", title="Population retained for modeling"); axes[0,0].legend(frameon=False)
    axes[0,1].bar(x, 100*table["valid_gaze_fraction"], color="#7a8f3a")
    axes[0,1].set(ylim=(80,100), ylabel="Gabor trials with gaze (%)", title="Eye-track coverage")
    axes[1,0].bar(x-width/2, table["gaze_dx_sd_deg"], width, color="#3366aa", label="horizontal")
    axes[1,0].bar(x+width/2, table["gaze_dy_sd_deg"], width, color="#d97736", label="vertical")
    axes[1,0].set(ylabel="Trial-median gaze SD (deg)", title="Centered gaze excursion"); axes[1,0].legend(frameon=False)
    axes[1,1].bar(x, table["gabor_presentations"], color="#777777")
    axes[1,1].set(ylabel="Presentations", title="Balanced 9×9×3×15 design")
    for axis in axes.ravel():
        axis.set_xticks(x, labels, rotation=35); axis.grid(axis="y", alpha=.16)
    fig.suptitle("Compact extraction QA: trials, gaze, and unit cohorts", fontsize=15)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    args=parse_args(); output=args.output_dir.resolve();
    inventory=output/"00_inventory"; ingestion=output/"01_ingestion"
    inventory.mkdir(parents=True,exist_ok=True); ingestion.mkdir(parents=True,exist_ok=True)
    table=collect(args.cache_root.resolve())
    table.to_csv(inventory/"session_inventory.csv",index=False,float_format="%.9g")
    table.to_csv(ingestion/"session_extraction_summary.csv",index=False,float_format="%.9g")
    render_inventory(table,inventory/"Figure_session_inventory.png")
    render_ingestion(table,ingestion/"Figure_extraction_quality.png")
    (ingestion/"summary.json").write_text(json.dumps({
        "sessions":len(table), "total_visual_units":int(table.visual_units.sum()),
        "total_qc_units":int(table.qc_units.sum()),
        "gabor_presentations_per_session":sorted(table.gabor_presentations.unique().tolist()),
        "valid_gaze_fraction_range":[float(table.valid_gaze_fraction.min()),float(table.valid_gaze_fraction.max())],
    },indent=2)+"\n")
    print(table.to_string(index=False))


if __name__=="__main__": main()
