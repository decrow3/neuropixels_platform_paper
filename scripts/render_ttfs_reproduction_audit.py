#!/usr/bin/env python3
"""Render released-versus-raw TTFS reproduction audit by dataset."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_metric_decomposition"
COLORS = {
    "Allen Brain Observatory": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2": "#D95F02",
}


def main():
    data = pd.read_csv(BASE / "unit_level_ttfs_components.csv")
    finite = data[["released_ttfs_ms", "recomputed_ttfs_ms"]].notna().all(axis=1)
    data = data.loc[finite].copy()
    data["signed_difference_ms"] = data["recomputed_ttfs_ms"] - data["released_ttfs_ms"]
    session = data.groupby(["cohort", "session_id"], as_index=False).agg(
        n_units=("unit_id", "size"),
        median_abs_error_ms=("ttfs_abs_error_ms", "median"),
        mean_signed_difference_ms=("signed_difference_ms", "mean"),
        exact_fraction=("ttfs_abs_error_ms", lambda x: float(np.mean(x < .001))),
    )
    cohort = data.groupby("cohort", as_index=False).agg(
        units=("unit_id", "size"),
        median_abs_error_ms=("ttfs_abs_error_ms", "median"),
        p95_abs_error_ms=("ttfs_abs_error_ms", lambda x: x.quantile(.95)),
        mean_signed_difference_ms=("signed_difference_ms", "mean"),
        exact_fraction=("ttfs_abs_error_ms", lambda x: float(np.mean(x < .001))),
    )
    cohort.to_csv(BASE / "ttfs_reproduction_by_cohort.csv", index=False)
    session.to_csv(BASE / "ttfs_reproduction_by_session.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for name, group in data.groupby("cohort"):
        sample = group.sample(min(len(group), 1800), random_state=20260825)
        axes[0].scatter(sample.released_ttfs_ms, sample.recomputed_ttfs_ms, s=7, alpha=.16, color=COLORS[name], label=name)
    axes[0].plot([30, 200], [30, 200], color="k", ls="--", lw=1)
    axes[0].set(xlabel="Released TTFS (ms)", ylabel="Recomputed from local raw spikes (ms)", title="Only MouseV2 lies on the identity line")
    axes[0].legend(frameon=False, fontsize=7)

    bins = np.arange(-80, 82, 2)
    for name, group in data.groupby("cohort"):
        axes[1].hist(group.signed_difference_ms, bins=bins, density=True, histtype="step", lw=2, color=COLORS[name], label=name)
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Recomputed minus released TTFS (ms)", ylabel="Density", title="Allen discrepancies are unit-specific, not one offset")

    order=list(COLORS)
    for i,name in enumerate(order):
        g=session.loc[session.cohort.eq(name)]
        axes[2].scatter(np.full(len(g),i)+np.linspace(-.12,.12,len(g)),g.median_abs_error_ms,s=20,alpha=.7,color=COLORS[name])
        axes[2].plot([i-.22,i+.22],[g.median_abs_error_ms.mean()]*2,color="k",lw=2)
    axes[2].set(xticks=range(3),xticklabels=["Allen BO","Allen FC","MouseV2"],ylabel="Session median absolute error (ms)",title="Mismatch recurs across Allen sessions")
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("TTFS provenance audit: released fields are not computationally equivalent")
    fig.savefig(BASE / "ttfs_reproduction_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(BASE / "ttfs_reproduction_audit.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
