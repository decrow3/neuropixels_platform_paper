"""Plot fixed response-timescale distributions for MouseV2 V1 groups and Allen VISp."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units
from scripts.figure3_robust_spread_comparison import session_group_table


def main() -> None:
    output_dir = ROOT / "artifacts/figure3/06v_v1_timescale_group_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_means = pd.read_csv(
        ROOT / "artifacts/figure3/06u_full20_rf_proxy_matched_timescale"
        / "Figure3_robust_session_group_means.csv"
    )
    mouse = group_means.loc[
        group_means["dataset"].eq("Within-V1")
        & group_means["metric"].eq("Response timescale (ms)")
    ].copy()
    mouse["display_group"] = "MouseV2 " + mouse["group"].astype(str)

    allen_units = load_allen_units(population_profile="common_qc")
    allen = session_group_table(
        allen_units, dataset="Allen-V1", session_column="ecephys_session_id",
        group_column="area_coarse", groups=["V1"], metric="timescale_ac",
        metric_label="Response timescale (ms)", metric_index=2, min_units=5,
    )
    allen["display_group"] = "Allen VISp"
    plot_data = pd.concat([
        mouse[["display_group", "session_id", "mean", "n_units"]],
        allen[["display_group", "session_id", "mean", "n_units"]],
    ], ignore_index=True)
    plot_data.to_csv(output_dir / "V1_timescale_group_session_means.csv", index=False)

    order = ["MouseV2 B", "MouseV2 C", "MouseV2 A", "MouseV2 E", "Allen VISp"]
    colors = ["#4575b4", "#1a9850", "#d73027", "#8073ac", "#303030"]
    values = [plot_data.loc[plot_data["display_group"].eq(group), "mean"].to_numpy() for group in order]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    violins = ax.violinplot(values, positions=np.arange(len(order)), widths=0.78,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.18)
    rng = np.random.default_rng(42)
    for x, (group, color, group_values) in enumerate(zip(order, colors, values)):
        jitter = rng.uniform(-0.18, 0.18, len(group_values))
        ax.scatter(x + jitter, group_values, s=24, color=color, alpha=0.68,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean = float(np.mean(group_values))
        sem = float(np.std(group_values, ddof=1) / np.sqrt(len(group_values)))
        ax.errorbar(x, mean, yerr=1.96 * sem, fmt="_", markersize=22, mew=3,
                    color=color, capsize=4, lw=1.6, zorder=4)
        ax.text(x, 0.98, f"n={len(group_values)}\nmean={mean:.1f}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=color)
    ax.set_xticks(np.arange(len(order)), ["B", "C", "A", "E", "Allen\nVISp"])
    ax.axvline(3.5, color="#999999", ls="--", lw=0.9)
    ax.set_ylabel("Session × group response timescale (ms)")
    ax.set_title("V1 response-timescale distributions", fontweight="bold")
    ax.grid(axis="y", color="#e7e7e7", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01,
             "MouseV2: full-20 parametric RF proxy, Allen-matched 150-flash draws; Allen: historical common-QC VISp, 150 flashes. Bars show mean ± 95% CI.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output = output_dir / "V1_timescale_group_distributions.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(plot_data.groupby("display_group")["mean"].agg(["count", "mean", "std"]).loc[order])


if __name__ == "__main__":
    main()
