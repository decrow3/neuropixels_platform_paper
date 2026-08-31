#!/usr/bin/env python3
"""Show how spike occupancy and the 100-ms cutoff shape MouseV2 TTFS histograms."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"
OUT = BASE / "ttfs_source_provenance"


def main() -> None:
    components = pd.read_csv(BASE / "ttfs_metric_decomposition/unit_level_ttfs_components.csv")
    preferred = pd.read_csv(OUT / "mousev2_preferred_polarity_ttfs_unit_level.csv")
    mouse = components.loc[components.cohort.eq("MouseV2")].merge(
        preferred[["unit_id", "preferred_0_250_ttfs_ms"]], on="unit_id", validate="one_to_one"
    )
    mouse = mouse.dropna(subset=["preferred_0_250_ttfs_ms", "occupancy_30_200"]).copy()
    mouse["occupancy_quartile"] = pd.qcut(mouse.occupancy_30_200, 4, labels=["Q1", "Q2", "Q3", "Q4"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    axes[0].hist(mouse.preferred_0_250_ttfs_ms, bins=np.arange(29.5, 201.5, 2),
                 density=True, color="#D95F02", alpha=.8)
    axes[0].axvline(100, color="k", ls="--", lw=1)
    axes[0].set(title="Before the Figure 3 cutoff", xlabel="Preferred TTFS (ms)", ylabel="Density")

    selected = mouse.loc[mouse.preferred_0_250_ttfs_ms.lt(100)]
    axes[1].hist(selected.preferred_0_250_ttfs_ms, bins=np.arange(29.5, 100.5, 2),
                 density=True, color="#D95F02", alpha=.8)
    axes[1].set(title="After TTFS <100 ms", xlabel="Preferred TTFS (ms)", ylabel="Density")

    colors = ["#C7E9C0", "#74C476", "#31A354", "#006D2C"]
    for color, (quartile, group) in zip(colors, selected.groupby("occupancy_quartile", observed=True)):
        label = f"{quartile}: occupancy {group.occupancy_30_200.mean():.2f}, n={len(group):,}"
        axes[2].hist(group.preferred_0_250_ttfs_ms, bins=np.arange(29.5, 100.5, 2),
                     density=True, histtype="step", lw=2, color=color, label=label)
    axes[2].set(title="The shape depends on flash-spike occupancy", xlabel="Preferred TTFS (ms)", ylabel="Density")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Why the MouseV2 per-unit TTFS histogram rises toward the cutoff")
    fig.savefig(OUT / "mousev2_ttfs_histogram_shape_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "mousev2_ttfs_histogram_shape_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = selected.groupby("occupancy_quartile", observed=True).agg(
        units=("unit_id", "size"), mean_ttfs_ms=("preferred_0_250_ttfs_ms", "mean"),
        median_ttfs_ms=("preferred_0_250_ttfs_ms", "median"),
        mean_occupancy=("occupancy_30_200", "mean"),
        mean_positive_evoked_delta_hz=("positive_evoked_delta_hz", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "mousev2_ttfs_histogram_shape_by_occupancy.csv", index=False)


if __name__ == "__main__":
    main()
