#!/usr/bin/env python3
"""Generate flash-evoked, z-scored CSD figures for all 32 MouseV2 probes (8 sessions x 4
probes), for VISUAL identification of the L4 source/sink reversal boundary.

This deliberately does NOT attempt automated landmark detection. Several detector variants
(earliest single-channel threshold crossing, per-channel adaptive threshold, magnitude-ranked
reversal, z-scored earliest-sustained reversal) were tried against a real Allen ground-truth
probe (session 756029989, probeD, known L4 at 2640-2700 um along probe_vertical_position) and
each found a different -- sometimes clearly wrong -- answer, while the exact same detector that
worked on MouseV2 probe A found nothing on the Allen probe (different noise/gain scale breaks
fixed thresholds). The extraction + smoothing + z-scoring pipeline itself is solid (confirmed
by the excellent visual match to Allen ground truth); only the automated peak-picking is
fragile. This produces one clean, comparable figure per probe for a manual visual pass instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_mousev2_csd_insertion_angle import (  # noqa: E402
    FLASH_WINDOW_AFTER_S, FLASH_WINDOW_BEFORE_S, PROBE_LETTERS, read_2d_row_slice, read_numeric_dset,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "figure3_mousev2.json"
OUTPUT = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"
SMOOTHING_WINDOW_SAMPLES = 8


def extract_zscored_csd(nwb_path: str, probe_letter: str) -> dict | None:
    fid = h5py.h5f.open(str(nwb_path).encode(), flags=h5py.h5f.ACC_RDONLY)
    with h5py.File(nwb_path, "r") as f:
        lfp_path = f"ElectricalSeriesProbe{probe_letter}-LFP"
        if lfp_path not in f["processing"]["ecephys"]["LFP"]:
            return None
        grp = f["processing"]["ecephys"]["LFP"][lfp_path]
        elec_idx = read_numeric_dset(fid, f"/processing/ecephys/LFP/{lfp_path}/electrodes").astype(int)
        rel_y_all = read_numeric_dset(fid, "/general/extracellular_ephys/electrodes/rel_y")
        channel_depth = rel_y_all[elec_idx]
        order = np.argsort(channel_depth)
        channel_depth = channel_depth[order]

        timestamps = read_numeric_dset(fid, f"/processing/ecephys/LFP/{lfp_path}/timestamps")
        data_path = f"/processing/ecephys/LFP/{lfp_path}/data"
        start_times = read_numeric_dset(fid, "/intervals/flash_field_block_presentations/start_time")

        fs = 1.0 / np.median(np.diff(timestamps[:2000]))
        n_before = int(FLASH_WINDOW_BEFORE_S * fs)
        n_after = int(FLASH_WINDOW_AFTER_S * fs)
        rel_t = (np.arange(-n_before, n_after)) / fs
        onset_idx = np.searchsorted(timestamps, start_times)
        valid = (onset_idx - n_before >= 0) & (onset_idx + n_after < len(timestamps))
        onset_idx = onset_idx[valid]
        n_channels = len(channel_depth)
        accum = np.zeros((n_before + n_after, n_channels), dtype=np.float64)
        for idx in onset_idx:
            segment = read_2d_row_slice(fid, data_path, idx - n_before, idx + n_after)
            baseline = segment[:n_before].mean(axis=0, keepdims=True)
            accum += segment - baseline
        lfp_avg = (accum / len(onset_idx))[:, order]

    csd = np.zeros_like(lfp_avg)
    csd[:, 1:-1] = lfp_avg[:, :-2] - 2 * lfp_avg[:, 1:-1] + lfp_avg[:, 2:]
    kernel = np.ones(SMOOTHING_WINDOW_SAMPLES) / SMOOTHING_WINDOW_SAMPLES
    csd_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=csd)
    baseline_mask = rel_t < 0
    z = (csd_smooth - csd_smooth[baseline_mask].mean(axis=0, keepdims=True)) / (
        csd_smooth[baseline_mask].std(axis=0, keepdims=True) + 1e-9)
    return {"z": z, "rel_t": rel_t, "channel_depth": channel_depth, "n_trials": len(onset_idx)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    grid_fig, grid_axes = plt.subplots(8, 4, figsize=(20, 34))
    for row, session in enumerate(config["sessions"]):
        site = session["site"]
        nwb_path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        if not nwb_path.exists():
            print(f"[{site}] NWB not found, skipping")
            continue
        print(f"[{site}] {nwb_path}")
        for col, probe_letter in enumerate(PROBE_LETTERS):
            ax = grid_axes[row, col]
            try:
                result = extract_zscored_csd(str(nwb_path), probe_letter)
            except Exception as exc:
                print(f"    probe {probe_letter} FAILED: {exc.__class__.__name__}: {exc}")
                ax.text(0.5, 0.5, "FAILED", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{site} {probe_letter}")
                continue
            if result is None:
                ax.axis("off")
                continue
            z, rel_t, depth = result["z"], result["rel_t"], result["channel_depth"]
            im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-6, vmax=6,
                            extent=[rel_t[0] * 1000, rel_t[-1] * 1000, depth[0], depth[-1]])
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set(title=f"{site} probe{probe_letter} (n={result['n_trials']})", xlim=(-10, 150))
            if col == 0:
                ax.set_ylabel("depth along probe (um)")
            if row == 7:
                ax.set_xlabel("time (ms)")

            fig, single_ax = plt.subplots(figsize=(6.5, 6))
            im2 = single_ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-6, vmax=6,
                                    extent=[rel_t[0] * 1000, rel_t[-1] * 1000, depth[0], depth[-1]])
            single_ax.axvline(0, color="black", linewidth=0.8)
            single_ax.set(title=f"{site} probe{probe_letter}, z-scored CSD (n={result['n_trials']} flashes)",
                           xlabel="time from flash onset (ms)", ylabel="depth along probe (um)", xlim=(-10, 150))
            fig.colorbar(im2, ax=single_ax, label="z-score")
            fig.tight_layout()
            fig.savefig(OUTPUT / f"Figure_csd_zscore_{site}_probe{probe_letter}.png", dpi=150)
            plt.close(fig)
            print(f"    probe {probe_letter}: {result['n_trials']} trials, saved")

    grid_fig.suptitle("MouseV2 flash-evoked z-scored CSD, all sessions x probes -- for visual reversal-boundary read", fontsize=14)
    grid_fig.tight_layout()
    grid_fig.savefig(OUTPUT / "Figure_csd_zscore_grid_all_probes.png", dpi=120)
    plt.close(grid_fig)
    print(f"\n{OUTPUT / 'Figure_csd_zscore_grid_all_probes.png'}")


if __name__ == "__main__":
    main()
