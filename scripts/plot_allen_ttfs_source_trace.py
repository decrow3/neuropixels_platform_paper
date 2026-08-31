#!/usr/bin/env python3
"""Plot numerical fingerprints of the standalone Allen TTFS table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(ROOT / "data/time_to_first_spike.csv")
    values = pd.to_numeric(table["time_to_first_spike_fl"], errors="coerce").dropna() * 1000

    counts = values.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(values, bins=np.arange(29.5, 201.5, 1), color="#4C78A8")
    axes[0].axvline(30, color="#D62728", ls="--", lw=1.5, label="hard minimum: 30 ms")
    axes[0].axvline(200, color="#D62728", ls="--", lw=1.5, label="hard maximum: 200 ms")
    axes[0].set(xlabel="Released TTFS (ms)", ylabel="Units", title="Standalone 2019 TTFS table")
    axes[0].legend(frameon=False, fontsize=8)

    residual = values - np.round(values)
    axes[1].hist(residual, bins=np.linspace(-0.5, 0.5, 51), color="#59A14F")
    axes[1].set(
        xlabel="TTFS minus nearest millisecond (ms)",
        ylabel="Units",
        title="Every finite value lies on the 1-ms grid",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The released values fingerprint a 30–200 ms, 1-ms-binned implementation")
    fig.savefig(OUT / "released_ttfs_numerical_fingerprint.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "released_ttfs_numerical_fingerprint.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [{
            "rows": len(table),
            "finite": len(values),
            "nan": int(table["time_to_first_spike_fl"].isna().sum()),
            "minimum_ms": values.min(),
            "maximum_ms": values.max(),
            "median_ms": values.median(),
            "below_30_ms": int((values < 30).sum()),
            "at_or_above_200_ms": int((values >= 200).sum()),
            "off_1ms_grid": int((np.abs(values - np.round(values)) > 1e-9).sum()),
            "unique_finite_values": counts.size,
        }]
    )
    summary.to_csv(OUT / "released_ttfs_numerical_fingerprint.csv", index=False)


if __name__ == "__main__":
    main()
