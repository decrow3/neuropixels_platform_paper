#!/usr/bin/env python3
"""Estimate MouseV2 probe insertion angle (relative to vertical) from flash-evoked CSD.

Motivation: the depth-constrained shank-line fit (`register_mousev2_units_along_probe_shank.py`)
needs to know how much of a probe's along-shank distance ("cortical_depth" = raw
probe_vertical_position, see generate_retinotopic_csvs.py L419-424) projects onto TANGENTIAL
(2D cortical surface / retinotopic-relevant) distance versus PERPENDICULAR depth. Without a
known insertion angle this was regularized toward Allen's own empirical population-average
ratio (0.287) -- workable, but not probe-specific. Since these NWBs have full-field flash
presentations AND per-channel LFP, we can do better: current source density (CSD) analysis
reveals the classic short-latency layer-4 current sink, a physiologically identifiable landmark.
Comparing the ALONG-PROBE distance to that sink against the literature value for L4's true
PERPENDICULAR depth from pia in mouse V1 gives a per-probe angle estimate via
    along_probe_distance_to_L4 = true_perpendicular_L4_depth / cos(angle_from_vertical)
    angle_from_vertical = arccos(true_perpendicular_L4_depth / along_probe_distance_to_L4)

This is a first pass: CSD heatmaps are produced for VISUAL verification alongside the automated
sink-latency estimate, since sink identification can fail silently (multiple sinks, noisy
channels, etc.) and should not be trusted without inspection.

L4 reference depth: mouse V1 L4 lower boundary is commonly reported around ~400-450 um from
pia (varies by source/method); this uses 420 um as a representative literature value. This is
a genuine source of uncertainty in the resulting angle estimate and is recorded explicitly in
the output, not hidden.
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "figure3_mousev2.json"
OUTPUT = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"

L4_REFERENCE_DEPTH_UM = 420.0
FLASH_WINDOW_BEFORE_S = 0.05
FLASH_WINDOW_AFTER_S = 0.20
PROBE_LETTERS = ("A", "B", "C", "E")
# response-onset search window: excludes both pre-stimulus noise and implausibly early
# "responses" (minimum retino-thalamo-cortical conduction delay is >~15 ms), and excludes late
# CSD structure that is more likely polysynaptic/feedback than the initial thalamocortical sink.
RESPONSE_WINDOW_S = (0.015, 0.12)
SMOOTHING_WINDOW_SAMPLES = 8  # ~6-7 ms at 1250 Hz, suppresses single-sample noise before onset detection
MIN_SUSTAINED_SAMPLES = 10  # ~8 ms sustained below threshold to count as a real sink, not a blip
EDGE_CHANNELS_EXCLUDED = 4  # channels at each end of the probe are prone to reference/edge artifacts
MIN_ADJACENT_CHANNELS = 4  # a genuine sink is spatially coherent across several adjacent channels;
# a single isolated channel can cross a 3 SD threshold by chance alone across thousands of
# (channel, time) samples -- this is what a single-channel-only criterion kept finding first.


def read_numeric_dset(fid, path: str) -> np.ndarray:
    dset = h5py.h5d.open(fid, path.encode())
    n = dset.get_space().get_simple_extent_npoints()
    arr = np.empty(n, dtype="<f8")
    mem_type = h5py.h5t.py_create(np.dtype("<f8"), logical=True)
    dset.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, mem_type)
    return arr


def read_2d_row_slice(fid, path: str, row_start: int, row_end: int) -> np.ndarray:
    """Low-level hyperslab read of dataset[row_start:row_end, :] as float64. The high-level
    h5py API chokes on this file's on-disk integer type ('Unsupported integer size (0)'),
    same underlying quirk as the 80-bit float timestamps handled by read_numeric_dset."""
    dset = h5py.h5d.open(fid, path.encode())
    space = dset.get_space()
    n_rows_total, n_cols = space.get_simple_extent_dims()
    row_start = max(0, row_start)
    row_end = min(n_rows_total, row_end)
    n_rows = row_end - row_start
    space.select_hyperslab((row_start, 0), (n_rows, n_cols))
    mem_space = h5py.h5s.create_simple((n_rows, n_cols))
    out = np.empty((n_rows, n_cols), dtype="<f8")
    mem_type = h5py.h5t.py_create(np.dtype("<f8"), logical=True)
    dset.read(mem_space, space, out, mem_type)
    return out


def compute_csd_for_probe(nwb_path: str, probe_letter: str) -> dict | None:
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

        # index each flash onset into the LFP timebase
        onset_idx = np.searchsorted(timestamps, start_times)
        valid = (onset_idx - n_before >= 0) & (onset_idx + n_after < len(timestamps))
        onset_idx = onset_idx[valid]
        print(f"    probe {probe_letter}: {len(onset_idx)} flash trials, {len(channel_depth)} channels, fs={fs:.1f} Hz")

        n_channels = len(channel_depth)
        accum = np.zeros((n_before + n_after, n_channels), dtype=np.float64)
        for idx in onset_idx:
            segment = read_2d_row_slice(fid, data_path, idx - n_before, idx + n_after)
            baseline = segment[:n_before].mean(axis=0, keepdims=True)
            accum += segment - baseline
        lfp_avg = accum / len(onset_idx)
        lfp_avg = lfp_avg[:, order]  # depth-ascending order

    # second spatial derivative (CSD), channels assumed evenly spaced (40 um)
    csd = np.zeros_like(lfp_avg)
    csd[:, 1:-1] = lfp_avg[:, :-2] - 2 * lfp_avg[:, 1:-1] + lfp_avg[:, 2:]

    # smooth in time to suppress single-sample noise before onset detection (boxcar, causal-ish
    # centered filter is fine here since this is an offline trial-averaged trace)
    kernel = np.ones(SMOOTHING_WINDOW_SAMPLES) / SMOOTHING_WINDOW_SAMPLES
    csd_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=csd)

    # GLOBAL threshold (pooled across all interior channels' baseline), not per-channel -- a
    # per-channel adaptive threshold lets a low-variance (near-silent) channel trivially "cross"
    # its own tiny noise floor and falsely win the earliest-onset comparison, which is exactly
    # what happened on the first two attempts (both landed on flat, structure-free channels).
    interior = slice(EDGE_CHANNELS_EXCLUDED, n_channels - EDGE_CHANNELS_EXCLUDED)
    interior_depth = channel_depth[interior]
    interior_csd = csd_smooth[:, interior]
    baseline_mask = rel_t < 0
    global_baseline_std = interior_csd[baseline_mask].std()
    threshold = -3.0 * global_baseline_std
    # additionally require the sink to be among the largest deflections actually observed in the
    # response window (top 15% most negative), so a threshold-crossing that is technically "3 SD"
    # but tiny in absolute terms next to the real sink cannot win
    window_mask = (rel_t >= RESPONSE_WINDOW_S[0]) & (rel_t <= RESPONSE_WINDOW_S[1])
    window_idx = np.nonzero(window_mask)[0]
    magnitude_floor = np.percentile(interior_csd[window_idx], 15)
    effective_threshold = min(threshold, magnitude_floor)

    # below[time, channel]: True where that channel is beyond the effective threshold
    below = interior_csd[window_idx, :] < effective_threshold
    n_ch = below.shape[1]
    # for each (time, channel), how many channels in a MIN_ADJACENT_CHANNELS-wide depth band
    # centered there are ALSO below threshold at that same time -- spatial coherence at a single
    # instant, not yet requiring temporal persistence
    half = MIN_ADJACENT_CHANNELS // 2
    coherent = np.zeros_like(below)
    for ch in range(n_ch):
        lo, hi = max(0, ch - half), min(n_ch, ch + half + 1)
        coherent[:, ch] = below[:, lo:hi].sum(axis=1) >= MIN_ADJACENT_CHANNELS

    found = None
    for k in range(coherent.shape[0]):
        if not coherent[k].any():
            continue
        # require the coherent band to persist for MIN_SUSTAINED_SAMPLES from here
        end = min(k + MIN_SUSTAINED_SAMPLES, coherent.shape[0])
        persists = coherent[k:end].all(axis=0)
        if persists.any():
            found = (k, np.nonzero(persists)[0])
            break

    if found is None:
        return None
    k, candidate_channels = found
    # within the coherent band at onset, report the single most negative channel as the landmark
    sub_scores = interior_csd[window_idx[k], candidate_channels]
    strongest = candidate_channels[np.argmin(sub_scores)]
    sink_depth_um = float(interior_depth[strongest])
    sink_time_s = float(rel_t[window_idx[k]])
    sink_magnitude = float(interior_csd[window_idx[k], strongest])

    return {
        "rel_t": rel_t, "channel_depth": channel_depth, "csd": csd, "lfp_avg": lfp_avg,
        "n_trials": len(onset_idx), "sink_depth_um": sink_depth_um, "sink_time_s": sink_time_s,
        "sink_magnitude": sink_magnitude,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    rows = []
    for session in config["sessions"]:
        site = session["site"]
        nwb_path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        if not nwb_path.exists():
            print(f"[{site}] NWB not found at {nwb_path}, skipping")
            continue
        print(f"[{site}] {nwb_path}")
        for probe_letter in PROBE_LETTERS:
            try:
                result = compute_csd_for_probe(str(nwb_path), probe_letter)
            except Exception as exc:
                print(f"    probe {probe_letter} FAILED: {exc.__class__.__name__}: {exc}")
                continue
            if result is None:
                continue
            along_probe_to_sink_um = result["sink_depth_um"]
            if along_probe_to_sink_um > 0:
                ratio = L4_REFERENCE_DEPTH_UM / along_probe_to_sink_um
                angle_deg = float(np.degrees(np.arccos(np.clip(ratio, -1.0, 1.0)))) if ratio <= 1.0 else 0.0
            else:
                angle_deg = np.nan
            rows.append({
                "site": site, "probe": probe_letter, "n_trials": result["n_trials"],
                "sink_depth_along_probe_um": along_probe_to_sink_um, "sink_time_s": result["sink_time_s"],
                "sink_magnitude": result["sink_magnitude"], "estimated_angle_from_vertical_deg": angle_deg,
            })

            # figure for visual verification
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
            for ax, field, title, cmap in ((axes[0], result["lfp_avg"], "Flash-evoked LFP", "RdBu_r"),
                                            (axes[1], result["csd"], "CSD (2nd spatial deriv.)", "RdBu_r")):
                vmax = np.percentile(np.abs(field), 99)
                im = ax.imshow(field.T, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                                extent=[result["rel_t"][0]*1000, result["rel_t"][-1]*1000,
                                        result["channel_depth"][0], result["channel_depth"][-1]])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set(title=title, xlabel="time from flash onset (ms)", ylabel="depth along probe (um)")
                fig.colorbar(im, ax=ax, fraction=0.046)
            axes[1].axhline(along_probe_to_sink_um, color="lime", linewidth=1.2, linestyle="--")
            axes[1].scatter([result["sink_time_s"]*1000], [along_probe_to_sink_um], color="lime", s=40, zorder=5)
            fig.suptitle(f"{site} probe {probe_letter}: earliest strong sink at {along_probe_to_sink_um:.0f} um along "
                         f"probe, t={result['sink_time_s']*1000:.0f} ms (n={result['n_trials']} flashes)", fontsize=11)
            fig.tight_layout()
            fig.savefig(OUTPUT / f"Figure_csd_{site}_probe{probe_letter}.png", dpi=150)
            plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "mousev2_csd_insertion_angle.csv", index=False)
    print(f"\nwrote {len(table)} probe CSD estimates")
    print(table.to_string(index=False))
    print(f"\nmedian estimated angle from vertical: {table.estimated_angle_from_vertical_deg.median():.1f} deg")
    (OUTPUT / "csd_manifest.json").write_text(json.dumps({
        "l4_reference_depth_um": L4_REFERENCE_DEPTH_UM,
        "flash_window_before_s": FLASH_WINDOW_BEFORE_S, "flash_window_after_s": FLASH_WINDOW_AFTER_S,
        "n_probes": len(table),
        "caveat": "sink identification is automated (earliest strong negative deflection among the most "
                  "negative quartile of interior channels) -- verify against the per-probe CSD heatmaps "
                  "before trusting any individual angle estimate.",
    }, indent=2))


if __name__ == "__main__":
    main()
