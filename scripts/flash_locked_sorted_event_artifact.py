"""Use high-precision sorted spike events as a surrogate AP-artifact screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/sorted_events"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="site2")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    session = next(s for s in config["sessions"] if s["site"] == args.site)
    path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
    OUT.mkdir(parents=True, exist_ok=True)
    edges = np.arange(-10, 40.0001, .1) / 1000
    centers_ms = (edges[:-1] + edges[1:]) * 500
    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)

    with h5py.File(path, "r") as handle:
        flashes = handle["intervals/flash_field_block_presentations"]
        starts = flashes["start_time"][:]
        polarity = flashes["contrast"][:].astype(float)
        ends = handle["units/spike_times_index"][:].astype(int)
        starts_index = np.r_[0, ends[:-1]]
        spike_data = handle["units/spike_times"]
        devices = np.char.decode(handle["units/device_name"][:].astype("S"), "utf8")
        qc = handle["units/default_qc"][:].astype(bool)

        for row, probe in enumerate(config["probe_labels"]):
            unit_indices = np.flatnonzero((devices == f"Probe{probe}") & qc)
            for column, (label, select) in enumerate((("bright", polarity > 0), ("dark", polarity < 0))):
                event_times = starts[select]
                histogram = np.zeros(len(edges) - 1, dtype=float)
                unit_histograms = []
                for unit in unit_indices:
                    spikes = spike_data[starts_index[unit]:ends[unit]]
                    local = np.zeros_like(histogram)
                    for event in event_times:
                        lo, hi = np.searchsorted(spikes, [event + edges[0], event + edges[-1]])
                        local += np.histogram(spikes[lo:hi] - event, bins=edges)[0]
                    histogram += local
                    unit_histograms.append(local)
                unit_histograms = np.asarray(unit_histograms)
                rate = histogram / (len(unit_indices) * len(event_times) * np.diff(edges)[0])
                # Count units whose event-aggregated histogram exceeds their own
                # pre-onset 99th percentile in each 0.1-ms bin.
                baseline = (centers_ms >= -9) & (centers_ms <= -1)
                thresholds = np.percentile(unit_histograms[:, baseline], 99, axis=1)
                active = (unit_histograms > thresholds[:, None]).mean(axis=0)
                axes[row, 0].plot(centers_ms, rate, label=label)
                axes[row, 1].plot(centers_ms, active, label=label)
                for time, value, fraction in zip(centers_ms, rate, active):
                    rows.append({"site": args.site, "probe": probe, "polarity": label,
                                 "time_ms": time, "population_rate_hz": value,
                                 "fraction_units_above_pre99": fraction,
                                 "n_units": len(unit_indices)})
            axes[row, 0].set_ylabel(f"Probe {probe}\nHz/unit")
            axes[row, 1].set_ylabel(f"Probe {probe}\nfraction")
    for ax in axes.flat:
        ax.axvline(0, color="0.5", lw=.8)
        ax.axvspan(0, 16.68, color="0.8", alpha=.2)
        ax.set_xlim(-10, 40)
    axes[0, 0].set_title("Sorted-event population rate (0.1-ms bins)")
    axes[0, 1].set_title("Fraction of units above own pre-onset 99th percentile")
    axes[0, 0].legend(frameon=False); axes[0, 1].legend(frameon=False)
    axes[-1, 0].set_xlabel("Time from NWB flash start (ms)")
    axes[-1, 1].set_xlabel("Time from NWB flash start (ms)")
    fig.suptitle(f"AP-surrogate photoartifact screen: {args.site} / {session['subject_id']}")
    fig.tight_layout()
    fig.savefig(OUT / f"{args.site}_sorted_event_artifact_screen.png", dpi=180)
    pd.DataFrame(rows).to_csv(OUT / f"{args.site}_sorted_event_artifact_screen.csv", index=False)
    print(args.site, "complete")


if __name__ == "__main__":
    main()
