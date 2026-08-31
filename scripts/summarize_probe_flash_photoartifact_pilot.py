"""Summarize cross-probe reproducibility of candidate flash artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"


def main() -> None:
    tables = [pd.read_csv(path) for path in sorted(SOURCE.glob("site*_early_score_curves.csv"))]
    data = pd.concat(tables, ignore_index=True)
    sites = sorted(data.site.unique(), key=lambda value: int(value.removeprefix("site")))
    probes = ["A", "B", "C", "E"]
    grid = np.arange(-10, 35.01, 0.2)
    consensus_rows = []

    fig, axes = plt.subplots(len(sites), 2, figsize=(12, 2.05 * len(sites)), sharex=True)
    for row, site in enumerate(sites):
        subset = data[data.site.eq(site)]
        interpolated = []
        for probe in probes:
            local = subset[subset.probe.eq(probe)].sort_values("time_ms")
            values = np.interp(grid, local.time_ms, local.derivative_rms_uv)
            baseline = values[(grid >= -10) & (grid <= -3)]
            z = (values - baseline.mean()) / max(baseline.std(ddof=1), np.finfo(float).eps)
            interpolated.append(z)
            axes[row, 0].plot(grid, z, label=probe, lw=1)
        matrix = np.vstack(interpolated)
        # Median rewards a component present on at least three of four probes.
        consensus = np.median(matrix, axis=0)
        axes[row, 1].plot(grid, consensus, color="black")
        search = (grid >= -1) & (grid <= 30)
        candidate = np.flatnonzero(search)[np.argmax(consensus[search])]
        axes[row, 1].axvline(grid[candidate], color="tab:red", ls="--")
        consensus_rows.append({
            "site": site,
            "candidate_latency_ms": grid[candidate],
            "consensus_z": consensus[candidate],
            "probe_z_A": matrix[0, candidate],
            "probe_z_B": matrix[1, candidate],
            "probe_z_C": matrix[2, candidate],
            "probe_z_E": matrix[3, candidate],
            "n_probes_z_gt_3": int((matrix[:, candidate] > 3).sum()),
        })
        axes[row, 0].set_ylabel(site)
        axes[row, 1].set_ylabel(site)
        for ax in axes[row]:
            ax.axvline(0, color="0.6", lw=0.8)
            ax.axhline(3, color="0.8", lw=0.7)
            ax.set_ylim(-3, min(30, max(8, np.nanpercentile(matrix, 99.5))))

    axes[0, 0].set_title("Probe-specific early derivative score")
    axes[0, 0].legend(frameon=False, ncol=4)
    axes[0, 1].set_title("Cross-probe median (≥3/4-probe consensus)")
    axes[-1, 0].set_xlabel("Time from NWB flash start (ms)")
    axes[-1, 1].set_xlabel("Time from NWB flash start (ms)")
    fig.suptitle("Flash-locked LFP transient search across all MouseV2 sessions")
    fig.tight_layout()
    fig.savefig(SOURCE / "all_sessions_cross_probe_consensus.png", dpi=180)
    summary = pd.DataFrame(consensus_rows)
    summary.to_csv(SOURCE / "all_sessions_cross_probe_consensus.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
