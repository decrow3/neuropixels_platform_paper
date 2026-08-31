"""Validate candidate photoartifacts by matching flash-on and flash-off LFP geometry."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"


def evoked_contrast(data, timestamps, events, polarity, offsets, conversion_uv):
    indices = np.searchsorted(timestamps, events)
    trials = np.stack([
        data[index + offsets, :].astype(float) * conversion_uv
        for index in indices
        if index + offsets[0] >= 0 and index + offsets[-1] < data.shape[0]
    ])
    bright = trials[polarity > 0].mean(axis=0)
    dark = trials[polarity < 0].mean(axis=0)
    return (bright - dark) / 2


def main():
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    probes = config["probe_labels"]
    summary = []
    fig, axes = plt.subplots(len(config["sessions"]), 2, figsize=(12, 16), sharex=True)

    for row, session in enumerate(config["sessions"]):
        site = session["site"]
        path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        probe_scores = []
        score_time = None
        with h5py.File(path, "r") as handle:
            flashes = handle["intervals/flash_field_block_presentations"]
            starts = flashes["start_time"][:]
            stops = flashes["stop_time"][:]
            polarity = flashes["contrast"][:].astype(float)
            for probe in probes:
                series = handle[f"processing/ecephys/LFP/ElectricalSeriesProbe{probe}-LFP"]
                timestamps = series["timestamps"][:]
                data = series["data"]
                dt = np.median(np.diff(timestamps[:10000]))
                offsets = np.arange(round(-0.015 / dt), round(0.040 / dt) + 1)
                time_ms = offsets * dt * 1000
                onset = evoked_contrast(data, timestamps, starts, polarity, offsets,
                                         float(data.attrs["conversion"]) * 1e6)
                offset = evoked_contrast(data, timestamps, stops, polarity, offsets,
                                          float(data.attrs["conversion"]) * 1e6)
                # Match transition vectors after temporal differentiation. Returning
                # from bright/dark to gray predicts the negative onset geometry.
                d_on = np.diff(onset, axis=0)
                d_off = np.diff(offset, axis=0)
                score_time = (time_ms[:-1] + time_ms[1:]) / 2
                numerator = np.sum(d_on * -d_off, axis=1)
                denominator = np.linalg.norm(d_on, axis=1) * np.linalg.norm(d_off, axis=1)
                similarity = numerator / np.maximum(denominator, np.finfo(float).eps)
                amplitude = np.sqrt(np.median(d_on ** 2, axis=1) * np.median(d_off ** 2, axis=1))
                base = (score_time >= -12) & (score_time <= -3)
                amp_z = (amplitude - amplitude[base].mean()) / max(amplitude[base].std(ddof=1), 1e-12)
                score = similarity * np.maximum(amp_z, 0)
                probe_scores.append(score)
                axes[row, 0].plot(score_time, similarity, label=probe, lw=1)

        matrix = np.vstack(probe_scores)
        consensus = np.median(matrix, axis=0)
        search = (score_time >= -1) & (score_time <= 25)
        index = np.flatnonzero(search)[np.argmax(consensus[search])]
        axes[row, 1].plot(score_time, consensus, color="black")
        axes[row, 1].axvline(score_time[index], color="tab:red", ls="--")
        axes[row, 0].set_ylabel(site)
        axes[row, 1].set_ylabel(site)
        summary.append({
            "site": site,
            "candidate_latency_ms": score_time[index],
            "onset_offset_consensus_score": consensus[index],
            "n_probes_positive": int((matrix[:, index] > 0).sum()),
        })

    for ax in axes.flat:
        ax.axvline(0, color="0.6", lw=.8)
        ax.axhline(0, color="0.8", lw=.7)
        ax.set_xlim(-10, 30)
    axes[0, 0].set_title("Onset versus reversed-offset channel-vector similarity")
    axes[0, 0].legend(frameon=False, ncol=4)
    axes[0, 1].set_title("Cross-probe median: similarity × transition amplitude z")
    axes[-1, 0].set_xlabel("Time from recorded transition (ms)")
    axes[-1, 1].set_xlabel("Time from recorded transition (ms)")
    fig.suptitle("Does the candidate flash artifact repeat at stimulus offset?")
    fig.tight_layout()
    fig.savefig(OUTPUT / "all_sessions_onset_offset_validation.png", dpi=180)
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUT / "all_sessions_onset_offset_validation.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
