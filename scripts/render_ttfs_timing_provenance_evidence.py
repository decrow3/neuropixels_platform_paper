"""Render the acquisition-timing evidence relevant to the MouseV2 TTFS offset."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(-2, 22); ax.set_ylim(-.2, 3.5); ax.axis("off")
    ax.annotate("", xy=(18, 2.65), xytext=(0, 2.65), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(19, 2.65, "time", va="center")
    ax.vlines(0, 2.25, 3.05, color="tab:blue", lw=2)
    ax.text(0, 3.15, "frame pulse rises", ha="center", color="tab:blue")
    ax.hlines(1.6, 0, 16.68, color="tab:orange", lw=8)
    ax.text(8.34, 1.9, "blocking window.flip() / wait for refresh", ha="center", color="tab:orange")
    ax.vlines(16.68, 1.2, 2.0, color="tab:green", lw=2)
    ax.text(16.68, .95, "frame pulse falls\nafter flip returns", ha="center", color="tab:green")
    ax.annotate("possible timestamp-to-display delay: 0–1 frame\nmean prediction = 8.34 ms",
                xy=(8.34, 2.65), xytext=(8.34, .15), ha="center",
                arrowprops={"arrowstyle": "-[,widthB=5.2", "lw": 1.2})
    ax.set_title("MouseV2 acquisition code establishes a before/after-flip timing bracket", loc="left", fontweight="bold")

    ax = axes[1]
    observed = 7.803
    half_frame = 8.3405
    ax.barh([1, 0], [observed, half_frame], color=["tab:blue", "tab:green"], height=.55)
    ax.set_yticks([1, 0], ["Observed MouseV2 − Allen VISp TTFS", "Half of measured 16.68-ms frame"])
    ax.set_xlim(0, 10); ax.set_xlabel("Milliseconds")
    ax.axvline(observed, color="tab:blue", lw=.8, alpha=.5)
    ax.axvline(half_frame, color="tab:green", lw=.8, alpha=.5)
    ax.text(observed-.15, 1, f"{observed:.2f}", ha="right", va="center", color="white", fontweight="bold")
    ax.text(half_frame-.15, 0, f"{half_frame:.2f}", ha="right", va="center", color="white", fontweight="bold")
    ax.text(5, -1.0,
            "Allen pipeline evidence: falling vsync edges + photodiode-derived frame starts.\n"
            "Unknown: whether MouseV2 NWB timestamps used pulse rise, pulse fall, or another derived edge.",
            ha="center", va="top", fontsize=9)
    ax.set_title("The independent timing prediction matches the observed offset", loc="left", fontweight="bold")
    fig.suptitle("TTFS timing-provenance evidence", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ttfs_timing_provenance_evidence.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
