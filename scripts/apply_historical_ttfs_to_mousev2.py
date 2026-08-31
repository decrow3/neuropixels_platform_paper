#!/usr/bin/env python3
"""Apply the inferred historical Allen preferred-flash TTFS rule to MouseV2 V1."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from crossvalidated_v1_flash_timing import all_unit_inputs
from reconstruct_historical_allen_ttfs import OUT, process_session

ALLEN_UNIT_PATH = OUT / "historical_ttfs_reconstruction_unit_level.csv"
COLORS = {
    "Allen Brain Observatory": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2": "#D95F02",
}


def session_summary(data: pd.DataFrame) -> pd.DataFrame:
    finite = data.dropna(subset=["released_ttfs_ms", "preferred_0_250_ttfs_ms"]).copy()
    return finite.groupby(["cohort", "session_id"], as_index=False).agg(
        units=("unit_id", "size"),
        pooled_mean_ms=("released_ttfs_ms", "mean"),
        pooled_median_ms=("released_ttfs_ms", "median"),
        preferred_mean_ms=("preferred_0_250_ttfs_ms", "mean"),
        preferred_median_ms=("preferred_0_250_ttfs_ms", "median"),
        mean_change_ms=("preferred_0_250_signed_error_ms", "mean"),
        median_change_ms=("preferred_0_250_signed_error_ms", "median"),
        bright_preferred_fraction=("preferred_polarity", lambda x: np.mean(x == "bright")),
    )


def cohort_summary(session: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in session.groupby("cohort"):
        for metric in ("pooled_mean_ms", "preferred_mean_ms", "mean_change_ms", "bright_preferred_fraction"):
            rows.append({
                "cohort": cohort, "metric": metric, "session_mean": group[metric].mean(),
                "session_sem": group[metric].sem(), "sessions": group.session_id.nunique(),
            })
    return pd.DataFrame(rows)


def render(mouse: pd.DataFrame, combined: pd.DataFrame, session: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    finite = mouse.dropna(subset=["released_ttfs_ms", "preferred_0_250_ttfs_ms"])
    axes[0].hexbin(finite.released_ttfs_ms, finite.preferred_0_250_ttfs_ms,
                   gridsize=45, mincnt=1, cmap="Oranges")
    axes[0].plot([30, 200], [30, 200], "k:", lw=1)
    axes[0].set(xlabel="MouseV2 pooled TTFS (ms)", ylabel="MouseV2 preferred-polarity TTFS (ms)",
                title=f"Unit-level change (n={len(finite):,})")

    bins = np.arange(29.5, 201.5, 2)
    for cohort, group in combined.groupby("cohort"):
        values = group.preferred_0_250_ttfs_ms.dropna()
        axes[1].hist(values, bins=bins, density=True, histtype="step", lw=2,
                     color=COLORS[cohort], label=f"{cohort} (n={len(values):,})")
    axes[1].set(xlabel="Preferred-polarity TTFS (ms)", ylabel="Density",
                title="Same estimator in all three datasets")
    axes[1].legend(frameon=False, fontsize=8)

    order = ["Allen Brain Observatory", "Allen Functional Connectivity", "MouseV2"]
    positions = np.arange(len(order))
    for i, cohort in enumerate(order):
        values = session.loc[session.cohort == cohort, "preferred_mean_ms"]
        axes[2].scatter(np.full(len(values), i), values, color=COLORS[cohort], alpha=.65, s=28)
        axes[2].errorbar(i, values.mean(), yerr=values.sem(), fmt="_", ms=20, lw=2.5,
                         color="k", capsize=4)
    axes[2].set(xticks=positions, xticklabels=["Allen BO", "Allen FC", "MouseV2"],
                ylabel="Session mean preferred TTFS (ms)", title="Session-balanced comparison")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Historical Allen preferred-flash TTFS rule applied to MouseV2 V1\nNo latency-based unit selection")
    fig.savefig(OUT / "mousev2_preferred_polarity_ttfs.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "mousev2_preferred_polarity_ttfs.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen.area_coarse.eq("V1")].copy()
    mouse_units = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc",
    )
    frames, status = [], []
    for cohort, sid, units, path, flash_table in all_unit_inputs(allen, mouse_units):
        if cohort != "MouseV2":
            continue
        print(f"[MouseV2 {sid}] {len(units)} common-QC V1 units", flush=True)
        try:
            frame = process_session(cohort, sid, units, path, flash_table)
            frames.append(frame)
            status.append({"session_id": sid, "status": "ok", "units": len(frame), "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            status.append({"session_id": sid, "status": "error", "units": 0,
                           "error": f"{type(exc).__name__}: {exc}"})
    mouse = pd.concat(frames, ignore_index=True)
    allen_reconstructed = pd.read_csv(ALLEN_UNIT_PATH)
    combined = pd.concat([allen_reconstructed, mouse], ignore_index=True, sort=False)
    session = session_summary(combined)
    cohort = cohort_summary(session)
    mouse.to_csv(OUT / "mousev2_preferred_polarity_ttfs_unit_level.csv", index=False)
    session.to_csv(OUT / "preferred_polarity_ttfs_session_summary.csv", index=False)
    cohort.to_csv(OUT / "preferred_polarity_ttfs_cohort_summary.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "mousev2_preferred_polarity_ttfs_status.csv", index=False)
    render(mouse, combined, session)
    print(session.to_string(index=False))
    print("\nSession-balanced cohort summary:\n", cohort.to_string(index=False))


if __name__ == "__main__":
    main()
