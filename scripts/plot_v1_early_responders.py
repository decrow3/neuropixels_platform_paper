#!/usr/bin/env python3
"""Compare the lower tail of V1 flash TTFS across Allen and MouseV2."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units


OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/early_responders"


def main() -> None:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].eq("V1")].copy()
    allen["cohort"] = allen["session_type"].map(
        {
            "brain_observatory_1.1": "Allen Brain Observatory",
            "functional_connectivity": "Allen Functional Connectivity",
        }
    )
    mouse = load_mousev2_units(
        apply_qc=False,
        grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled",
        population_profile="common_qc",
    )
    mouse["cohort"] = "MouseV2"
    data = pd.concat([allen, mouse], ignore_index=True)
    data["ttfs_ms"] = 1000 * pd.to_numeric(
        data["time_to_first_spike_fl"], errors="coerce"
    )
    data = data.loc[data["ttfs_ms"].between(30, 100, inclusive="left")].copy()

    colors = {
        "Allen Brain Observatory": "#6F63A6",
        "Allen Functional Connectivity": "#B07AA1",
        "MouseV2": "#D95F02",
    }
    thresholds = [35, 40, 45, 50, 55, 60]
    rows = []
    for cohort, group in data.groupby("cohort", sort=False):
        values = group["ttfs_ms"].to_numpy()
        for threshold in thresholds:
            rows.append(
                {
                    "cohort": cohort,
                    "threshold_ms": threshold,
                    "n_early": int(np.sum(values < threshold)),
                    "n_valid_ttfs": len(values),
                    "fraction_early": float(np.mean(values < threshold)),
                }
            )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for cohort, group in data.groupby("cohort", sort=False):
        x = np.sort(group["ttfs_ms"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        axes[0].plot(x, y, lw=2.3, color=colors[cohort], label=f"{cohort} (n={len(x):,})")
    mouse_x = np.sort(data.loc[data["cohort"].eq("MouseV2"), "ttfs_ms"].to_numpy() - 8.34)
    axes[0].plot(
        mouse_x,
        np.arange(1, len(mouse_x) + 1) / len(mouse_x),
        color="#D95F02",
        lw=1.8,
        ls="--",
        label="MouseV2 − 8.34 ms prediction",
    )
    axes[0].axvline(30, color="0.3", ls=":", lw=1.5)
    axes[0].set(xlim=(25, 80), ylim=(0, 0.72), xlabel="Flash TTFS (ms)", ylabel="Cumulative fraction of valid V1 units")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Lower-tail latency distributions")

    for cohort, group in summary.groupby("cohort", sort=False):
        axes[1].plot(
            group["threshold_ms"], 100 * group["fraction_early"],
            marker="o", lw=2.3, color=colors[cohort], label=cohort,
        )
    axes[1].set(xlabel="Definition of early responder: TTFS < threshold", ylabel="V1 units classified early (%)", xticks=thresholds)
    axes[1].set_title("Early responders exist in every dataset")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V1 early responders: same 30-ms estimator floor, shifted lower-tail prevalence", y=1.02)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "early_responder_threshold_summary.csv", index=False)
    data[["cohort", "unit_id", "ttfs_ms"]].sort_values(["cohort", "ttfs_ms"]).to_csv(
        OUT / "v1_unit_ttfs_lower_tail.csv", index=False
    )
    fig.savefig(OUT / "v1_early_responder_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "v1_early_responder_comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
