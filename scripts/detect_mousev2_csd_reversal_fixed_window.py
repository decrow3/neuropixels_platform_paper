#!/usr/bin/env python3
"""Detect the CSD source/sink reversal boundary (L4 landmark) using a FIXED time window,
informed by the visual pass over all 32 MouseV2 probes: onset is remarkably consistent at
~35-60 ms post-flash across nearly every probe. Averaging z-scored CSD over that fixed window
removes the timing-search step that broke every earlier automated attempt (each failure mode
was really a timing failure, picking noise at the wrong latency) -- what's left is a much
simpler, more robust spatial-only search: find the reversal in a stable, trial- and time-
averaged depth profile.

Validated first against the real Allen ground-truth probe (session 756029989, probeD, known L4
at 2640-2700 um) before trusting it on MouseV2.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_mousev2_csd_insertion_angle import (  # noqa: E402
    FLASH_WINDOW_AFTER_S, FLASH_WINDOW_BEFORE_S, PROBE_LETTERS, read_2d_row_slice, read_numeric_dset,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "figure3_mousev2.json"
OUTPUT = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"
SMOOTHING_WINDOW_SAMPLES = 8
FIXED_WINDOW_S = (0.035, 0.060)
EDGE_CHANNELS_EXCLUDED = 4


def extract_zscored_csd_generic(read_start_times, read_lfp) -> dict:
    """read_start_times() -> np.ndarray, read_lfp() -> (timestamps, channel_depth, data_reader)
    where data_reader(row_start, row_end) -> np.ndarray[rows, channels] in depth-sorted order."""
    start_times = read_start_times()
    timestamps, channel_depth, data_reader = read_lfp()
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
        segment = data_reader(idx - n_before, idx + n_after)
        baseline = segment[:n_before].mean(axis=0, keepdims=True)
        accum += segment - baseline
    lfp_avg = accum / len(onset_idx)

    csd = np.zeros_like(lfp_avg)
    csd[:, 1:-1] = lfp_avg[:, :-2] - 2 * lfp_avg[:, 1:-1] + lfp_avg[:, 2:]
    kernel = np.ones(SMOOTHING_WINDOW_SAMPLES) / SMOOTHING_WINDOW_SAMPLES
    csd_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=csd)
    baseline_mask = rel_t < 0
    z = (csd_smooth - csd_smooth[baseline_mask].mean(axis=0, keepdims=True)) / (
        csd_smooth[baseline_mask].std(axis=0, keepdims=True) + 1e-9)
    return {"z": z, "rel_t": rel_t, "channel_depth": channel_depth, "n_trials": len(onset_idx)}


def find_reversal(z: np.ndarray, rel_t: np.ndarray, channel_depth: np.ndarray,
                   window_s: tuple[float, float] = FIXED_WINDOW_S) -> dict | None:
    window_mask = (rel_t >= window_s[0]) & (rel_t <= window_s[1])
    profile = z[window_mask].mean(axis=0)  # time-averaged, stable depth profile
    interior = slice(EDGE_CHANNELS_EXCLUDED, len(profile) - EDGE_CHANNELS_EXCLUDED)
    depth_i = channel_depth[interior]
    profile_i = profile[interior]

    best = None
    for ch in range(len(profile_i) - 1):
        if profile_i[ch] > 0 and profile_i[ch + 1] < 0:
            combined = profile_i[ch] - profile_i[ch + 1]
            if best is None or combined > best[0]:
                v0, v1 = profile_i[ch], profile_i[ch + 1]
                frac = v0 / (v0 - v1)
                crossing_depth = depth_i[ch] + frac * (depth_i[ch + 1] - depth_i[ch])
                best = (combined, crossing_depth, profile_i[ch], profile_i[ch + 1])
    if best is None:
        return None
    return {"combined_magnitude": best[0], "reversal_depth_um": best[1],
            "source_z": best[2], "sink_z": best[3]}


def mousev2_lfp_readers(nwb_path: str, probe_letter: str):
    fid = h5py.h5f.open(str(nwb_path).encode(), flags=h5py.h5f.ACC_RDONLY)
    with h5py.File(nwb_path, "r") as f:
        if f"ElectricalSeriesProbe{probe_letter}-LFP" not in f["processing"]["ecephys"]["LFP"]:
            return None
    lfp_path = f"ElectricalSeriesProbe{probe_letter}-LFP"

    def read_start_times():
        with h5py.File(nwb_path, "r") as f:
            pass
        return read_numeric_dset(fid, "/intervals/flash_field_block_presentations/start_time")

    def read_lfp():
        elec_idx = read_numeric_dset(fid, f"/processing/ecephys/LFP/{lfp_path}/electrodes").astype(int)
        rel_y_all = read_numeric_dset(fid, "/general/extracellular_ephys/electrodes/rel_y")
        channel_depth = rel_y_all[elec_idx]
        order = np.argsort(channel_depth)
        channel_depth = channel_depth[order]
        timestamps = read_numeric_dset(fid, f"/processing/ecephys/LFP/{lfp_path}/timestamps")
        data_path = f"/processing/ecephys/LFP/{lfp_path}/data"

        def data_reader(row_start, row_end):
            return read_2d_row_slice(fid, data_path, row_start, row_end)[:, order]

        return timestamps, channel_depth, data_reader

    return read_start_times, read_lfp


def render_verification(result: dict, reversal: dict | None, title: str, output_path: Path) -> None:
    z, rel_t, depth = result["z"], result["rel_t"], result["channel_depth"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-6, vmax=6,
                    extent=[rel_t[0] * 1000, rel_t[-1] * 1000, depth[0], depth[-1]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvspan(FIXED_WINDOW_S[0] * 1000, FIXED_WINDOW_S[1] * 1000, color="black", alpha=0.08)
    if reversal is not None:
        ax.axhline(reversal["reversal_depth_um"], color="lime", linewidth=1.4, linestyle="--")
    ax.set(title=title, xlabel="time from flash onset (ms)", ylabel="depth along probe (um)", xlim=(-10, 150))
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. re-validate against Allen ground truth first
    print("=== Allen ground-truth validation (session 756029989, probeD, known L4 2640-2700 um) ===")
    allen_main = "/mnt/nvme0/ecephys_cache_dir_2/session_756029989/session_756029989.nwb"
    allen_lfp = "/mnt/nvme0/ecephys_cache_dir_2/session_756029989/probe_760640094_lfp.nwb"
    fid_main = h5py.h5f.open(allen_main.encode(), flags=h5py.h5f.ACC_RDONLY)
    fid_lfp = h5py.h5f.open(allen_lfp.encode(), flags=h5py.h5f.ACC_RDONLY)

    def allen_start_times():
        return read_numeric_dset(fid_main, "/intervals/flashes_presentations/start_time")

    def allen_lfp_reader():
        grp_path = "/acquisition/probe_760640094_lfp/probe_760640094_lfp_data"
        elec_idx = read_numeric_dset(fid_lfp, grp_path + "/electrodes").astype(int)
        channel_depth = read_numeric_dset(fid_lfp, "/general/extracellular_ephys/electrodes/probe_vertical_position")[elec_idx]
        order = np.argsort(channel_depth)
        channel_depth = channel_depth[order]
        timestamps = read_numeric_dset(fid_lfp, grp_path + "/timestamps")
        data_path = grp_path + "/data"

        def data_reader(row_start, row_end):
            return read_2d_row_slice(fid_lfp, data_path, row_start, row_end)[:, order]

        return timestamps, channel_depth, data_reader

    allen_result = extract_zscored_csd_generic(allen_start_times, allen_lfp_reader)
    allen_reversal = find_reversal(allen_result["z"], allen_result["rel_t"], allen_result["channel_depth"])
    print(f"detected reversal depth: {allen_reversal['reversal_depth_um']:.1f} um "
          f"(known truth: 2640-2700 um) -- {'PASS' if 2500 <= allen_reversal['reversal_depth_um'] <= 2850 else 'CHECK'}")
    render_verification(allen_result, allen_reversal,
                         f"Allen ground truth: detected={allen_reversal['reversal_depth_um']:.0f} um (true L4: 2640-2700)",
                         OUTPUT / "Figure_allen_ground_truth_fixed_window_validation.png")

    # 2. apply to all 32 MouseV2 probes
    print("\n=== MouseV2, all 32 probes ===")
    config = json.loads(CONFIG.read_text())
    rows = []
    for session in config["sessions"]:
        site = session["site"]
        nwb_path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        if not nwb_path.exists():
            continue
        for probe_letter in PROBE_LETTERS:
            readers = mousev2_lfp_readers(str(nwb_path), probe_letter)
            if readers is None:
                continue
            read_start_times, read_lfp = readers
            result = extract_zscored_csd_generic(read_start_times, read_lfp)
            reversal = find_reversal(result["z"], result["rel_t"], result["channel_depth"])
            if reversal is None:
                print(f"[{site} {probe_letter}] no reversal found")
                rows.append({"site": site, "probe": probe_letter, "reversal_depth_um": np.nan})
                continue
            print(f"[{site} {probe_letter}] reversal at {reversal['reversal_depth_um']:.0f} um "
                  f"(combined |z|={reversal['combined_magnitude']:.2f})")
            rows.append({"site": site, "probe": probe_letter,
                         "reversal_depth_um": reversal["reversal_depth_um"],
                         "combined_magnitude": reversal["combined_magnitude"],
                         "n_trials": result["n_trials"]})
            render_verification(result, reversal, f"{site} probe{probe_letter}: reversal={reversal['reversal_depth_um']:.0f} um",
                                 OUTPUT / f"Figure_csd_fixed_window_{site}_probe{probe_letter}.png")

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "mousev2_csd_reversal_fixed_window.csv", index=False)
    print(f"\nwrote {len(table)} probe estimates")
    print(table.to_string(index=False))
    print(f"\nmedian reversal depth: {table.reversal_depth_um.median():.1f} um "
          f"(IQR {table.reversal_depth_um.quantile(.25):.1f}-{table.reversal_depth_um.quantile(.75):.1f})")


if __name__ == "__main__":
    main()
