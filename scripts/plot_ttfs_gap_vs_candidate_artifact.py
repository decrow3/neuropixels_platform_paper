"""Compare the TTFS alignment required by biology with LFP transient candidates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"


def main():
    groups = pd.read_csv(ROOT / "Figure3/Figure3_robust_session_group_means.csv")
    mouse = groups[(groups.dataset == "Within-V1") & (groups.metric == "TTFS (ms)")]
    mouse = mouse.groupby("session_id", as_index=False).session_mean.first()
    mouse["site"] = "site" + mouse.session_id.astype(int).astype(str)
    allen = pd.read_csv(ROOT / "Figure3/Figure3_Allen_V1_session_means.csv")
    allen = allen[allen.metric == "TTFS (ms)"]
    allen_center = allen["mean"].mean()
    candidates = pd.read_csv(OUT / "all_sessions_onset_offset_validation.csv")
    table = mouse.merge(candidates, on="site", validate="one_to_one")
    table["shift_required_ms"] = table.session_mean - allen_center
    frame_period_ms = 16.681
    half_frame_ms = frame_period_ms / 2

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhspan(
        table.shift_required_ms.mean() - table.shift_required_ms.std(ddof=1),
        table.shift_required_ms.mean() + table.shift_required_ms.std(ddof=1),
        color="tab:blue", alpha=.12, label="MouseV2→Allen required shift: mean ± session SD",
    )
    ax.scatter(x - .12, table.shift_required_ms, color="tab:blue", s=55,
               label="Required shift to Allen VISp mean")
    sizes = 35 + 25 * np.clip(table.onset_offset_consensus_score, 0, 4)
    ax.scatter(x + .12, table.candidate_latency_ms, color="tab:red", marker="^", s=sizes,
               label="Onset/offset LFP candidate (size = evidence score)")
    for i, row in table.iterrows():
        ax.plot([i-.12, i+.12], [row.shift_required_ms, row.candidate_latency_ms],
                color="0.75", lw=1)
    ax.axhline(0, color="0.4", lw=.8)
    ax.axhline(half_frame_ms, color="tab:green", ls="--", lw=1.8,
               label=f"Half of 16.68-ms display frame = {half_frame_ms:.2f} ms")
    ax.set_xticks(x, table.site)
    ax.set_ylabel("Latency relative to NWB flash timestamp (ms)")
    ax.set_title("Do flash-locked probe transients explain the MouseV2–Allen TTFS offset?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ttfs_gap_vs_candidate_artifact.png", dpi=180)
    table.to_csv(OUT / "ttfs_gap_vs_candidate_artifact.csv", index=False)
    print(f"Allen VISp equal-session mean: {allen_center:.3f} ms")
    print(f"Mean required shift: {table.shift_required_ms.mean():.3f} ms; half frame: {half_frame_ms:.3f} ms")
    print(table[["site", "session_mean", "shift_required_ms", "candidate_latency_ms",
                 "onset_offset_consensus_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
