"""Concrete pilot for a flash-locked Neuropixels photoelectric artifact.

Reads one MouseV2 NWB, averages LFP around bright and dark flash onsets, and
writes probe-level traces plus an onset-latency summary.  This is deliberately
an artifact diagnostic, not a TTFS correction estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def robust_z(x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    center = np.median(baseline)
    scale = 1.4826 * np.median(np.abs(baseline - center))
    if scale <= np.finfo(float).eps:
        scale = np.std(baseline, ddof=1)
    return (x - center) / max(scale, np.finfo(float).eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="site2")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot",
    )
    args = parser.parse_args()

    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    session = next(row for row in config["sessions"] if row["site"] == args.site)
    nwb = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
    args.output.mkdir(parents=True, exist_ok=True)

    window_ms = (-25.0, 100.0)
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)

    with h5py.File(nwb, "r") as handle:
        flashes = handle["intervals/flash_field_block_presentations"]
        starts = flashes["start_time"][:]
        contrasts = flashes["contrast"][:].astype(float)

        for row_index, probe in enumerate(config["probe_labels"]):
            series = handle[f"processing/ecephys/LFP/ElectricalSeriesProbe{probe}-LFP"]
            timestamps = series["timestamps"][:]
            data = series["data"]
            conversion_uv = float(data.attrs["conversion"]) * 1e6
            dt = float(np.median(np.diff(timestamps[:10000])))
            offsets = np.arange(round(window_ms[0] / (dt * 1000)),
                                round(window_ms[1] / (dt * 1000)) + 1)
            time_ms = offsets * dt * 1000
            onset_indices = np.searchsorted(timestamps, starts)

            evoked = {}
            for polarity, selector in (("bright", contrasts > 0), ("dark", contrasts < 0)):
                trials = np.stack([
                    data[index + offsets, :].astype(float) * conversion_uv
                    for index in onset_indices[selector]
                    if index + offsets[0] >= 0 and index + offsets[-1] < data.shape[0]
                ])
                evoked[polarity] = np.mean(trials, axis=0)

            # A polarity contrast suppresses common visually evoked activity while
            # retaining an onset artifact that reverses with screen luminance.
            contrast_evoked = (evoked["bright"] - evoked["dark"]) / 2
            channel_rms = np.sqrt(np.median(contrast_evoked ** 2, axis=1))
            derivative_rms = np.sqrt(np.median(np.diff(contrast_evoked, axis=0) ** 2, axis=1))
            baseline = (time_ms >= -20) & (time_ms <= -3)
            derivative_time = (time_ms[:-1] + time_ms[1:]) / 2
            derivative_z = robust_z(derivative_rms, derivative_rms[baseline[:-1]])
            curve_rows.extend(
                {
                    "site": args.site,
                    "probe": probe,
                    "time_ms": float(t),
                    "derivative_rms_uv": float(value),
                    "derivative_z": float(z),
                }
                for t, value, z in zip(derivative_time, derivative_rms, derivative_z)
            )
            search = (derivative_time >= -2) & (derivative_time <= 30)
            peak_index = np.flatnonzero(search)[np.argmax(derivative_z[search])]

            rows.append({
                "site": args.site,
                "subject_id": session["subject_id"],
                "probe": probe,
                "n_bright": int((contrasts > 0).sum()),
                "n_dark": int((contrasts < 0).sum()),
                "lfp_dt_ms": dt * 1000,
                "candidate_latency_ms": derivative_time[peak_index],
                "candidate_derivative_z": derivative_z[peak_index],
                "candidate_contrast_rms_uv": channel_rms[min(peak_index + 1, len(channel_rms)-1)],
            })

            ax = axes[row_index, 0]
            ax.plot(time_ms, np.median(evoked["bright"], axis=1), label="bright")
            ax.plot(time_ms, np.median(evoked["dark"], axis=1), label="dark")
            ax.set_ylabel(f"Probe {probe}\nmedian LFP (µV)")
            if row_index == 0:
                ax.legend(frameon=False)

            axes[row_index, 1].imshow(
                contrast_evoked.T, aspect="auto", origin="lower",
                extent=[time_ms[0], time_ms[-1], 0, contrast_evoked.shape[1]],
                cmap="RdBu_r", vmin=-np.percentile(np.abs(contrast_evoked), 99),
                vmax=np.percentile(np.abs(contrast_evoked), 99),
            )
            axes[row_index, 1].set_ylabel(f"Probe {probe}\nchannel")
            axes[row_index, 2].plot(derivative_time, derivative_z, color="black")
            axes[row_index, 2].axvline(derivative_time[peak_index], color="tab:red", ls="--")
            axes[row_index, 2].set_ylabel(f"Probe {probe}\nderivative z")

    for ax in axes.flat:
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlim(window_ms)
    axes[-1, 0].set_xlabel("Time from NWB flash start (ms)")
    axes[-1, 1].set_xlabel("Time from NWB flash start (ms)")
    axes[-1, 2].set_xlabel("Time from NWB flash start (ms)")
    axes[0, 1].set_title("(bright − dark)/2 evoked LFP")
    axes[0, 2].set_title("Across-channel derivative RMS")
    figure.suptitle(f"MouseV2 flash photoartifact pilot: {args.site} / subject {session['subject_id']}")
    figure.tight_layout()
    figure.savefig(args.output / f"{args.site}_flash_photoartifact_pilot.png", dpi=180)
    pd.DataFrame(rows).to_csv(args.output / f"{args.site}_candidate_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.output / f"{args.site}_early_score_curves.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
