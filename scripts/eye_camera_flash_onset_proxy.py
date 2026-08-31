"""Test eye-tracking features as an optical proxy for physical flash onset."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/eye_camera_proxy"
SIGNALS = {
    "corneal reflection area": "corneal_reflection/area_raw",
    "pupil area": "pupil/area_raw",
    "eye ellipse area": "ellipse/area_raw",
}


def main():
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(config["sessions"]), len(SIGNALS), figsize=(14, 16), sharex=True)
    rows = []
    offsets = np.arange(-4, 10)
    for row, session in enumerate(config["sessions"]):
        path = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        with h5py.File(path, "r") as handle:
            flashes = handle["intervals/flash_field_block_presentations"]
            starts = flashes["start_time"][:]
            polarity = flashes["contrast"][:].astype(float)
            for column, (label, dataset_path) in enumerate(SIGNALS.items()):
                group = handle[f"processing/eye_tracking/{dataset_path.rsplit('/', 1)[0]}"]
                values = handle[f"processing/eye_tracking/{dataset_path}"][:]
                timestamps = group["timestamps"][:]
                nearest = np.searchsorted(timestamps, starts)
                trial_values = np.stack([values[index + offsets] for index in nearest])
                trial_times = np.stack([timestamps[index + offsets] - event for index, event in zip(nearest, starts)]) * 1000
                baseline = np.nanmedian(trial_values[:, offsets < 0], axis=1, keepdims=True)
                scale = np.nanmedian(np.abs(trial_values[:, offsets < 0] - baseline), axis=1, keepdims=True) * 1.4826
                normalized = (trial_values - baseline) / np.maximum(scale, np.nanmedian(scale[scale > 0]) * .1)
                ax = axes[row, column]
                for name, selector, color in (("bright", polarity > 0, "tab:orange"),
                                               ("dark", polarity < 0, "tab:blue")):
                    mean_time = np.nanmedian(trial_times[selector], axis=0)
                    center = np.nanmedian(normalized[selector], axis=0)
                    lo, hi = np.nanpercentile(normalized[selector], [25, 75], axis=0)
                    ax.plot(mean_time, center, color=color, marker="o", ms=2.5, label=name)
                    ax.fill_between(mean_time, lo, hi, color=color, alpha=.12)
                    for t, value in zip(mean_time, center):
                        rows.append({"site": session["site"], "signal": label,
                                     "polarity": name, "time_ms": t,
                                     "median_baseline_mad_units": value,
                                     "camera_dt_ms": np.median(np.diff(timestamps)) * 1000})
                ax.axvline(0, color="0.5", lw=.8)
                ax.axvline(8.34, color="tab:green", ls="--", lw=.8)
                if row == 0: ax.set_title(label)
                if column == 0: ax.set_ylabel(session["site"])
        print(session["site"])
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]: ax.set_xlabel("Time from NWB flash start (ms)")
    fig.suptitle("Can eye-camera measurements reveal physical flash onset?\n"
                 "median and IQR; green dashed line = half display frame")
    fig.tight_layout()
    fig.savefig(OUT / "all_sessions_eye_camera_flash_proxy.png", dpi=180)
    pd.DataFrame(rows).to_csv(OUT / "all_sessions_eye_camera_flash_proxy.csv", index=False)


if __name__ == "__main__":
    main()
