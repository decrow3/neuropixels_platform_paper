#!/usr/bin/env python3
"""Estimate MouseV2 probe insertion angle from RELATIVE along-probe thickness of the
flash-responsive cortical band, compared to an Allen reference probe -- avoids the absolute
reference-point-convention problem that broke the earlier landmark-depth approach (a thickness
is a difference between two points measured the same way on the SAME probe, so any fixed
per-probe zero-point offset cancels out; the earlier approach needed an absolute depth to match
across datasets and that assumption was not verifiable).

Method: response power = mean(z_scored_CSD^2) over the 0-150ms post-flash window, spatially
smoothed (~360 um), thresholded at its own 60th percentile, restricted to the 1500-3500 um
along-probe window (where cortex plausibly sits for all these probes based on the earlier CSD
visual pass and the Allen reference below). Thickness = span of the single LARGEST contiguous
responsive run in that window.

Validated against a real Allen ground-truth probe (session 756029989, probeD) where histology
gives an EXACT known cortical (VISl) extent of 2240-2980 um (740 um true thickness): this method
detects 2160-3080 um (920 um), a ~24% over-estimate -- present in both datasets since the same
method is applied to both, so it should mostly cancel in the MouseV2/Allen thickness RATIO used
for the angle estimate:

    angle_from_vertical = arccos(allen_reference_thickness / mousev2_thickness)

Only one Allen LFP-equipped probe was available locally (no internet fetch was performed for
more); this is flagged explicitly as an n=1 reference, not "a few" sessions.
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
from detect_mousev2_csd_reversal_fixed_window import extract_zscored_csd_generic, mousev2_lfp_readers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "figure3_mousev2.json"
OUTPUT = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"
SEARCH_WINDOW_UM = (1500.0, 3500.0)
SPATIAL_SMOOTH_CHANNELS = 9
POWER_PERCENTILE = 60.0
POWER_WINDOW_S = (0.0, 0.15)


def responsive_band_thickness(z: np.ndarray, rel_t: np.ndarray, channel_depth: np.ndarray) -> dict | None:
    window_mask = (rel_t >= POWER_WINDOW_S[0]) & (rel_t <= POWER_WINDOW_S[1])
    power = (z[window_mask] ** 2).mean(axis=0)
    kernel = np.ones(SPATIAL_SMOOTH_CHANNELS) / SPATIAL_SMOOTH_CHANNELS
    power_smooth = np.convolve(power, kernel, mode="same")

    search_mask = (channel_depth >= SEARCH_WINDOW_UM[0]) & (channel_depth <= SEARCH_WINDOW_UM[1])
    idx = np.nonzero(search_mask)[0]
    if len(idx) < SPATIAL_SMOOTH_CHANNELS:
        return None
    depth_s = channel_depth[idx]
    power_s = power_smooth[idx]
    threshold = np.percentile(power_s, POWER_PERCENTILE)
    responsive = power_s > threshold

    runs = []
    start = None
    for i, r in enumerate(responsive):
        if r and start is None:
            start = i
        if not r and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(responsive) - 1))
    runs = [(s, e) for s, e in runs if e > s]
    if not runs:
        return None
    runs.sort(key=lambda r: depth_s[r[1]] - depth_s[r[0]], reverse=True)
    s, e = runs[0]
    return {"thickness_um": float(depth_s[e] - depth_s[s]), "band_start_um": float(depth_s[s]),
            "band_end_um": float(depth_s[e]), "threshold": float(threshold)}


def render(result: dict, band: dict | None, title: str, output_path: Path) -> None:
    z, rel_t, depth = result["z"], result["rel_t"], result["channel_depth"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-6, vmax=6,
                    extent=[rel_t[0] * 1000, rel_t[-1] * 1000, depth[0], depth[-1]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhspan(SEARCH_WINDOW_UM[0], SEARCH_WINDOW_UM[1], color="grey", alpha=0.06)
    if band is not None:
        ax.axhspan(band["band_start_um"], band["band_end_um"], color="lime", alpha=0.25)
    ax.set(title=title, xlabel="time from flash onset (ms)", ylabel="depth along probe (um)", xlim=(-10, 150))
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # -- Allen reference --
    print("=== Allen reference (session 756029989, probeD; true VISl extent 2240-2980 um, 740 um) ===")
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
    allen_band = responsive_band_thickness(allen_result["z"], allen_result["rel_t"], allen_result["channel_depth"])
    print(f"Allen detected band: {allen_band['band_start_um']:.0f}-{allen_band['band_end_um']:.0f} um "
          f"(thickness {allen_band['thickness_um']:.0f} um; true = 740 um)")
    render(allen_result, allen_band,
           f"Allen reference: detected {allen_band['thickness_um']:.0f} um (true VISl = 740 um)",
           OUTPUT / "Figure_allen_responsive_band_validation.png")
    allen_reference_thickness = allen_band["thickness_um"]

    # -- MouseV2, all 32 probes --
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
            band = responsive_band_thickness(result["z"], result["rel_t"], result["channel_depth"])
            if band is None:
                print(f"[{site} {probe_letter}] no responsive band found")
                rows.append({"site": site, "probe": probe_letter, "thickness_um": np.nan})
                continue
            ratio = band["thickness_um"] / allen_reference_thickness
            angle_deg = float(np.degrees(np.arccos(np.clip(1.0 / ratio, -1.0, 1.0)))) if ratio >= 1.0 else 0.0
            print(f"[{site} {probe_letter}] band {band['band_start_um']:.0f}-{band['band_end_um']:.0f} um "
                  f"(thickness {band['thickness_um']:.0f} um, ratio={ratio:.2f}, angle={angle_deg:.1f} deg)")
            rows.append({"site": site, "probe": probe_letter, "thickness_um": band["thickness_um"],
                         "band_start_um": band["band_start_um"], "band_end_um": band["band_end_um"],
                         "ratio_to_allen": ratio, "estimated_angle_from_vertical_deg": angle_deg,
                         "n_trials": result["n_trials"]})
            render(result, band, f"{site} probe{probe_letter}: thickness={band['thickness_um']:.0f} um, "
                                  f"angle~{angle_deg:.0f} deg",
                   OUTPUT / f"Figure_thickness_{site}_probe{probe_letter}.png")

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "mousev2_cortical_thickness_vs_allen.csv", index=False)
    print(f"\nwrote {len(table)} probe estimates")
    print(table.to_string(index=False))
    valid = table.dropna(subset=["thickness_um"])
    print(f"\nMouseV2 median thickness: {valid.thickness_um.median():.0f} um "
          f"(IQR {valid.thickness_um.quantile(.25):.0f}-{valid.thickness_um.quantile(.75):.0f})")
    print(f"Allen reference thickness: {allen_reference_thickness:.0f} um (n=1 probe -- no additional Allen "
          f"sessions were downloaded)")
    print(f"median estimated angle from vertical: {valid.estimated_angle_from_vertical_deg.median():.1f} deg")

    (OUTPUT / "thickness_method_manifest.json").write_text(json.dumps({
        "allen_reference_thickness_um": allen_reference_thickness,
        "allen_reference_n_probes": 1,
        "allen_true_thickness_um": 740.0,
        "allen_detection_overestimate_fraction": (allen_reference_thickness - 740.0) / 740.0,
        "search_window_um": SEARCH_WINDOW_UM,
        "median_mousev2_thickness_um": float(valid.thickness_um.median()),
        "median_estimated_angle_deg": float(valid.estimated_angle_from_vertical_deg.median()),
        "caveat": "Only one Allen LFP-equipped probe was available locally; this is an n=1 reference, "
                  "not a robust multi-session baseline. The detector has a known ~24% over-estimation "
                  "bias (validated against known VISl histology on the same probe) which should mostly "
                  "cancel in the ratio but is not guaranteed to cancel exactly.",
    }, indent=2))


if __name__ == "__main__":
    main()
