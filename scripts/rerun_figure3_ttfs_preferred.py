#!/usr/bin/env python3
"""Rerun the Figure 3 TTFS<100 ms cohort using one preferred-flash estimator."""

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

COLORS = {
    "Allen Brain Observatory": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2": "#D95F02",
}


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, session_rows = [], []
    definitions = {
        "original_mixed": "released_ttfs_ms",
        "harmonized_preferred": "preferred_0_250_ttfs_ms",
    }
    for definition, column in definitions.items():
        selected = data.loc[data[column].notna() & data[column].lt(100)].copy()
        for (cohort, sid), group in selected.groupby(["cohort", "session_id"]):
            session_rows.append({"definition": definition, "cohort": cohort, "session_id": sid,
                                 "units": len(group), "mean_ttfs_ms": group[column].mean(),
                                 "median_ttfs_ms": group[column].median()})
        for cohort, group in selected.groupby("cohort"):
            rows.append({"definition": definition, "cohort": cohort, "units": len(group),
                         "unit_weighted_mean_ms": group[column].mean(),
                         "unit_weighted_median_ms": group[column].median()})
    session = pd.DataFrame(session_rows)
    cohort = pd.DataFrame(rows)
    balanced = session.groupby(["definition", "cohort"], as_index=False).agg(
        sessions=("session_id", "nunique"), selected_units=("units", "sum"),
        session_balanced_mean_ms=("mean_ttfs_ms", "mean"),
        session_sem_ms=("mean_ttfs_ms", "sem"),
        mean_units_per_session=("units", "mean"),
    )
    cohort = cohort.merge(balanced, on=["definition", "cohort"], validate="one_to_one")
    return cohort, session


def gaps(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition, group in summary.groupby("definition"):
        values = group.set_index("cohort")
        mouse = values.loc["MouseV2"]
        for allen in ("Allen Brain Observatory", "Allen Functional Connectivity"):
            row = values.loc[allen]
            rows.append({
                "definition": definition, "allen_cohort": allen,
                "unit_weighted_mouse_minus_allen_ms": mouse.unit_weighted_mean_ms - row.unit_weighted_mean_ms,
                "session_balanced_mouse_minus_allen_ms": mouse.session_balanced_mean_ms - row.session_balanced_mean_ms,
                "mouse_units": int(mouse.units), "allen_units": int(row.units),
            })
    return pd.DataFrame(rows)


def render(data: pd.DataFrame, summary: pd.DataFrame, gap: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    bins = np.arange(29.5, 100.5, 2)
    for col, (definition, metric) in enumerate((
        ("original_mixed", "released_ttfs_ms"),
        ("harmonized_preferred", "preferred_0_250_ttfs_ms"),
    )):
        ax = axes[col]
        for cohort, group in data.groupby("cohort"):
            values = group.loc[group[metric].lt(100), metric].dropna()
            ax.hist(values, bins=bins, density=True, histtype="step", lw=2,
                    color=COLORS[cohort], label=f"{cohort} (n={len(values):,})")
        ax.set(xlabel="TTFS among selected units (ms)", ylabel="Density",
               title="Original mixed definitions" if col == 0 else "Harmonized preferred polarity")
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    order = ["Allen Brain Observatory", "Allen Functional Connectivity", "MouseV2"]
    x = np.arange(3); width = .34
    for offset, definition in ((-.5, "original_mixed"), (.5, "harmonized_preferred")):
        sub = summary.set_index(["definition", "cohort"])
        y = [sub.loc[(definition, cohort), "session_balanced_mean_ms"] for cohort in order]
        sem = [sub.loc[(definition, cohort), "session_sem_ms"] for cohort in order]
        ax.bar(x + offset*width, y, width, yerr=sem, capsize=3,
               color=[COLORS[c] for c in order], alpha=.45 if definition == "original_mixed" else .95,
               edgecolor="k" if definition == "harmonized_preferred" else "none",
               label="original" if definition == "original_mixed" else "harmonized")
    ax.set(xticks=x, xticklabels=["Allen BO", "Allen FC", "MouseV2"], ylabel="Session-balanced mean TTFS (ms)",
           title="Exact TTFS <100 ms re-selection")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figure 3 V1 TTFS after applying the same preferred-flash definition\nPipeline-baseline population; selection recomputed from each definition")
    fig.savefig(OUT / "figure3_exact_filter_preferred_ttfs.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "figure3_exact_filter_preferred_ttfs.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Figure3_with_V1sites.py defaults to the unfiltered pipeline-baseline population.
    allen = load_allen_units(population_profile=None)
    allen = allen.loc[allen.area_coarse.eq("V1")].copy()
    mouse = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile=None,
    )
    frames, status = [], []
    for cohort, sid, units, path, flash_table in all_unit_inputs(allen, mouse):
        print(f"[{cohort} {sid}] {len(units)} pipeline-baseline V1 units", flush=True)
        try:
            frames.append(process_session(cohort, sid, units, path, flash_table))
            status.append({"cohort": cohort, "session_id": sid, "status": "ok", "units": len(units), "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            status.append({"cohort": cohort, "session_id": sid, "status": "error", "units": 0,
                           "error": f"{type(exc).__name__}: {exc}"})
    data = pd.concat(frames, ignore_index=True)
    summary, session = summarize(data)
    gap = gaps(summary)
    data.to_csv(OUT / "figure3_exact_filter_preferred_ttfs_unit_level.csv", index=False)
    summary.to_csv(OUT / "figure3_exact_filter_preferred_ttfs_summary.csv", index=False)
    session.to_csv(OUT / "figure3_exact_filter_preferred_ttfs_session.csv", index=False)
    gap.to_csv(OUT / "figure3_exact_filter_preferred_ttfs_gaps.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "figure3_exact_filter_preferred_ttfs_status.csv", index=False)
    render(data, summary, gap)
    print(summary.to_string(index=False))
    print("\nGaps (positive = MouseV2 later):\n", gap.to_string(index=False))


if __name__ == "__main__":
    main()
