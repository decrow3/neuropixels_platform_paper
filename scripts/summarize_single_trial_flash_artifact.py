"""Aggregate the single-trial one-frame artifact search with matched controls."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/single_trial"


def main():
    data = pd.read_csv(OUT / "single_trial_candidates.csv")
    sites = sorted(data.site.unique(), key=lambda x: int(x.removeprefix("site")))
    bins = np.linspace(0, 16.8, 15)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    for transition, color in (("onset", "tab:purple"), ("offset", "tab:green")):
        part = data[data.transition.eq(transition)].copy()
        excess = part.post_peak_z - part.control_peak_z
        stat, p = wilcoxon(excess, alternative="greater")
        counts, _ = np.histogram(part.candidate_latency_ms, bins=bins)
        chi, uniform_p = chisquare(counts)
        axes[0, 0].hist(part.candidate_latency_ms, bins=bins, histtype="step", lw=2,
                        label=transition, color=color)
        axes[0, 1].hist(excess, bins=45, histtype="step", lw=2, label=transition, color=color)
        rows.append({"scope": "all_sessions", "transition": transition, "n": len(part),
                     "median_excess_z": excess.median(), "wilcoxon_greater_p": p,
                     "latency_uniform_chisquare_p": uniform_p})
        for site in sites:
            local = part[part.site.eq(site)]
            local_excess = local.post_peak_z - local.control_peak_z
            _, local_p = wilcoxon(local_excess, alternative="greater")
            rows.append({"scope": site, "transition": transition, "n": len(local),
                         "median_excess_z": local_excess.median(),
                         "wilcoxon_greater_p": local_p,
                         "latency_uniform_chisquare_p": np.nan})

    axes[0, 0].axvline(8.3405, color="0.4", ls="--", label="half frame")
    axes[0, 0].set(xlabel="Selected latency within first frame (ms)", ylabel="trials",
                   title="Candidate latencies pooled across sessions")
    axes[0, 1].axvline(0, color="0.4", lw=.8)
    axes[0, 1].set(xlabel="Post-transition peak − matched pre-transition peak (z)", ylabel="trials",
                   title="Matched single-trial excess")
    axes[0, 0].legend(frameon=False); axes[0, 1].legend(frameon=False)

    for column, transition in enumerate(("onset", "offset")):
        part = data[data.transition.eq(transition)].copy()
        values = [
            (part.loc[part.site.eq(site), "post_peak_z"] -
             part.loc[part.site.eq(site), "control_peak_z"]).to_numpy()
            for site in sites
        ]
        axes[1, column].boxplot(values, labels=sites, showfliers=False)
        axes[1, column].axhline(0, color="0.4", lw=.8)
        axes[1, column].tick_params(axis="x", rotation=45)
        axes[1, column].set(ylabel="Post − pre peak (z)", title=f"{transition}: session distributions")

    fig.suptitle("Single-trial search for a flash-locked probe transient within one 16.68-ms frame")
    fig.tight_layout()
    fig.savefig(OUT / "all_sessions_single_trial_frame_summary.png", dpi=180)

    sync_fig, sync_axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, transition in zip(sync_axes, ("onset", "offset")):
        part = data[data.transition.eq(transition)]
        difference = part.post_probe_phase_resultant - part.control_probe_phase_resultant
        _, p = wilcoxon(difference, alternative="greater")
        values = [difference[part.site.eq(site)].to_numpy() for site in sites]
        ax.boxplot(values, tick_labels=sites, showfliers=False)
        ax.axhline(0, color="0.4", lw=.8)
        ax.tick_params(axis="x", rotation=45)
        ax.set(title=f"{transition}: median ΔR={np.median(difference):.3f}, p={p:.2g}",
               ylabel="Post − pre cross-probe phase resultant")
    sync_fig.suptitle("Are candidate one-frame transients simultaneous across four probes?")
    sync_fig.tight_layout()
    sync_fig.savefig(OUT / "all_sessions_cross_probe_single_trial_synchrony.png", dpi=180)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "single_trial_statistical_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
