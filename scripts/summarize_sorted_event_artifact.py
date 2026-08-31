"""Aggregate sorted-event AP-surrogate flash-artifact screens."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/sorted_events"


def main():
    data = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("site*_sorted_event_artifact_screen.csv"))])
    sites = sorted(data.site.unique(), key=lambda x: int(x.removeprefix("site")))
    probes = ["A", "B", "C", "E"]
    fig, axes = plt.subplots(len(sites), len(probes), figsize=(14, 15), sharex=True, sharey=True)
    rows = []
    for r, site in enumerate(sites):
        for c, probe in enumerate(probes):
            ax = axes[r, c]
            for polarity, color in (("bright", "tab:blue"), ("dark", "tab:orange")):
                part = data[(data.site == site) & (data.probe == probe) & (data.polarity == polarity)].sort_values("time_ms")
                base = part[(part.time_ms >= -9) & (part.time_ms <= -1)].population_rate_hz
                z = (part.population_rate_hz - base.mean()) / base.std(ddof=1)
                ax.plot(part.time_ms, z, color=color, lw=.8, label=polarity)
                early = (part.time_ms >= 0) & (part.time_ms <= 20)
                index = np.flatnonzero(early)[np.argmax(z[early])]
                rows.append({"site": site, "probe": probe, "polarity": polarity,
                             "early_peak_latency_ms": part.time_ms.iloc[index],
                             "early_peak_z": z.iloc[index]})
            ax.axvline(0, color="0.5", lw=.6); ax.axhline(3, color="0.8", lw=.6)
            ax.set_xlim(-10, 25); ax.set_ylim(-4, 6)
            if r == 0: ax.set_title(f"Probe {probe}")
            if c == 0: ax.set_ylabel(site)
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]: ax.set_xlabel("ms")
    fig.suptitle("Flash-locked sorted-event rate at 0.1-ms resolution (baseline z score)")
    fig.tight_layout()
    fig.savefig(OUT / "all_sessions_sorted_event_artifact_summary.png", dpi=180)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "all_sessions_sorted_event_artifact_summary.csv", index=False)
    print(summary.sort_values("early_peak_z", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
