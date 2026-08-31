"""Render channel-resolved flash onset/offset contrasts for artifact inspection."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from validate_flash_photoartifact_with_offsets import evoked_contrast


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/channel_geometry"


def main():
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for session in config["sessions"]:
        path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
        derivative_fig, derivative_axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
        with h5py.File(path, "r") as handle:
            flashes = handle["intervals/flash_field_block_presentations"]
            starts, stops = flashes["start_time"][:], flashes["stop_time"][:]
            polarity = flashes["contrast"][:].astype(float)
            for row, probe in enumerate(config["probe_labels"]):
                series = handle[f"processing/ecephys/LFP/ElectricalSeriesProbe{probe}-LFP"]
                timestamps, data = series["timestamps"][:], series["data"]
                dt = np.median(np.diff(timestamps[:10000]))
                offsets = np.arange(round(-.010 / dt), round(.035 / dt) + 1)
                time_ms = offsets * dt * 1000
                conversion = float(data.attrs["conversion"]) * 1e6
                onset = evoked_contrast(data, timestamps, starts, polarity, offsets, conversion)
                offset = evoked_contrast(data, timestamps, stops, polarity, offsets, conversion)
                limit = np.percentile(np.abs(np.r_[onset.ravel(), offset.ravel()]), 99)
                for column, (values, title) in enumerate(((onset, "Onset: (bright−dark)/2"),
                                                          (-offset, "Reversed offset"))):
                    axes[row, column].imshow(values.T, origin="lower", aspect="auto",
                        extent=[time_ms[0], time_ms[-1], 0, values.shape[1]],
                        cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                    axes[row, column].axvline(0, color="black", lw=.7)
                    axes[row, column].set_ylabel(f"Probe {probe} channel")
                    if row == 0:
                        axes[row, column].set_title(title)
                    derivative = np.diff(values, axis=0)
                    derivative_time = (time_ms[:-1] + time_ms[1:]) / 2
                    derivative_limit = np.percentile(np.abs(derivative), 99)
                    derivative_axes[row, column].imshow(
                        derivative.T, origin="lower", aspect="auto",
                        extent=[derivative_time[0], derivative_time[-1], 0, derivative.shape[1]],
                        cmap="RdBu_r", vmin=-derivative_limit, vmax=derivative_limit,
                        interpolation="nearest",
                    )
                    derivative_axes[row, column].axvline(0, color="black", lw=.7)
                    derivative_axes[row, column].set_ylabel(f"Probe {probe} channel")
                    if row == 0:
                        derivative_axes[row, column].set_title("Δ " + title)
        axes[-1, 0].set_xlabel("Time from transition (ms)")
        axes[-1, 1].set_xlabel("Time from transition (ms)")
        fig.suptitle(f"Flash transition channel geometry: {session['site']} / {session['subject_id']}")
        fig.tight_layout()
        fig.savefig(OUT / f"{session['site']}_onset_offset_channel_geometry.png", dpi=180)
        plt.close(fig)
        derivative_axes[-1, 0].set_xlabel("Time from transition (ms)")
        derivative_axes[-1, 1].set_xlabel("Time from transition (ms)")
        derivative_fig.suptitle(
            f"Flash transition derivative geometry: {session['site']} / {session['subject_id']}"
        )
        derivative_fig.tight_layout()
        derivative_fig.savefig(OUT / f"{session['site']}_onset_offset_derivative_geometry.png", dpi=180)
        plt.close(derivative_fig)
        print(session["site"])


if __name__ == "__main__":
    main()
