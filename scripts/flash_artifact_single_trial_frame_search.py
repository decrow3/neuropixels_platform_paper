"""Single-trial search for a probe-wide artifact within one display frame."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/single_trial"


def main():
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_trial_rows = []
    for session in config["sessions"]:
        path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        with h5py.File(path, "r") as handle:
            flashes = handle["intervals/flash_field_block_presentations"]
            transitions = {"onset": flashes["start_time"][:], "offset": flashes["stop_time"][:]}
            polarity = flashes["contrast"][:].astype(float)
            probe_trial_scores = {kind: [] for kind in transitions}
            common_time = np.arange(-20, 25.01, .2)
            for probe in config["probe_labels"]:
                series = handle[f"processing/ecephys/LFP/ElectricalSeriesProbe{probe}-LFP"]
                timestamps, data = series["timestamps"][:], series["data"]
                dt = np.median(np.diff(timestamps[:10000]))
                offsets = np.arange(round(-.022 / dt), round(.027 / dt) + 1)
                local_time = (offsets[:-1] + .5) * dt * 1000
                for kind, events in transitions.items():
                    scores = []
                    for event in events:
                        index = np.searchsorted(timestamps, event)
                        trial = data[index + offsets, :].astype(float)
                        derivative_rms = np.sqrt(np.median(np.diff(trial, axis=0) ** 2, axis=1))
                        scores.append(np.interp(common_time, local_time, derivative_rms))
                    probe_trial_scores[kind].append(np.stack(scores))

            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for row, kind in enumerate(("onset", "offset")):
                matrix = np.stack(probe_trial_scores[kind])  # probe × trial × time
                base = (common_time >= -18) & (common_time <= -2)
                center = np.median(matrix[:, :, base], axis=2, keepdims=True)
                scale = np.median(np.abs(matrix[:, :, base] - center), axis=2, keepdims=True) * 1.4826
                z = (matrix - center) / np.maximum(scale, .05)
                consensus = np.median(z, axis=0)
                post = (common_time >= 0) & (common_time <= 16.68)
                control = (common_time >= -18) & (common_time <= -1.32)
                post_indices = np.flatnonzero(post)
                control_indices = np.flatnonzero(control)
                post_arg = post_indices[np.argmax(consensus[:, post], axis=1)]
                control_arg = control_indices[np.argmax(consensus[:, control], axis=1)]
                post_peak = consensus[np.arange(len(consensus)), post_arg]
                control_peak = consensus[np.arange(len(consensus)), control_arg]
                post_latency = common_time[post_arg]
                probe_post_arg = post_indices[np.argmax(z[:, :, post], axis=2)]
                probe_control_arg = control_indices[np.argmax(z[:, :, control], axis=2)]
                probe_post_latency = common_time[probe_post_arg]
                probe_control_latency = common_time[probe_control_arg]
                period = 16.68
                post_resultant = np.abs(np.mean(np.exp(2j * np.pi * probe_post_latency / period), axis=0))
                control_resultant = np.abs(np.mean(np.exp(2j * np.pi * probe_control_latency / period), axis=0))

                bins = np.arange(-.4, 17.21, .8)
                axes[row, 0].hist(post_latency[polarity > 0], bins=bins, alpha=.6, label="bright")
                axes[row, 0].hist(post_latency[polarity < 0], bins=bins, alpha=.6, label="dark")
                axes[row, 0].axhline(len(post_latency) / len(bins), color="0.5", ls="--", lw=.8)
                axes[row, 0].set(title=f"{kind}: candidate latency", xlabel="ms", ylabel="trials")
                axes[row, 1].scatter(control_peak, post_peak, c=np.where(polarity > 0, "tab:orange", "tab:blue"), s=10, alpha=.55)
                low = min(control_peak.min(), post_peak.min()); high = max(control_peak.max(), post_peak.max())
                axes[row, 1].plot([low, high], [low, high], color="0.5", lw=.8)
                axes[row, 1].set(title=f"{kind}: one-frame peak vs pre-event control",
                                 xlabel="pre-event maximum z", ylabel="post-event maximum z")
                if row == 0:
                    axes[row, 0].legend(frameon=False)

                summary_rows.append({
                    "site": session["site"], "transition": kind,
                    "median_post_peak_z": np.median(post_peak),
                    "median_control_peak_z": np.median(control_peak),
                    "median_excess_z": np.median(post_peak - control_peak),
                    "median_candidate_latency_ms": np.median(post_latency),
                })
                all_trial_rows.extend({
                    "site": session["site"], "transition": kind, "trial": i,
                    "polarity": "bright" if polarity[i] > 0 else "dark",
                    "candidate_latency_ms": post_latency[i], "post_peak_z": post_peak[i],
                    "control_peak_z": control_peak[i],
                    "post_probe_phase_resultant": post_resultant[i],
                    "control_probe_phase_resultant": control_resultant[i],
                } for i in range(len(post_latency)))
            fig.suptitle(f"Single-trial one-frame artifact search: {session['site']} / {session['subject_id']}")
            fig.tight_layout()
            fig.savefig(OUT / f"{session['site']}_single_trial_frame_search.png", dpi=180)
            plt.close(fig)
            print(session["site"])
    pd.DataFrame(summary_rows).to_csv(OUT / "single_trial_frame_summary.csv", index=False)
    pd.DataFrame(all_trial_rows).to_csv(OUT / "single_trial_candidates.csv", index=False)


if __name__ == "__main__":
    main()
