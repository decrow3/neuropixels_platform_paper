#!/usr/bin/env python3
"""Figure-3-style V1 TTFS panel using latency-independent response selection."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from common.figure3_mousev2 import load_mousev2_units

BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot"
OUT = BASE / "ttfs_source_provenance"
AUDIT = BASE / "crossvalidated_latency_independent_psths/crossvalidated_unit_selection_audit.csv"
COLORS = {"Allen Brain Observatory":"#6F63A6", "Allen Functional Connectivity":"#B07AA1", "MouseV2":"#D95F02"}
MIN_UNITS_PER_ESTIMATE = 10


def main():
    audit = pd.read_csv(AUDIT)
    allen = pd.read_csv(OUT / "historical_ttfs_reconstruction_unit_level.csv")
    mouse = pd.read_csv(OUT / "mousev2_preferred_polarity_ttfs_unit_level.csv")
    preferred = pd.concat([allen, mouse], ignore_index=True, sort=False)
    d = audit.merge(preferred[["cohort","session_id","unit_id","preferred_polarity","preferred_0_250_ttfs_ms"]],
                    on=["cohort","session_id","unit_id"], validate="many_to_one")
    d = d.loc[d.polarity.eq(d.preferred_polarity)].copy()
    d["selected"] = d.selected_positive_responder & d.preferred_0_250_ttfs_ms.lt(100)

    mouse_units = load_mousev2_units(apply_qc=False,
        grating_metrics_dir=ROOT/"data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT/"data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc")
    d = d.merge(mouse_units[["unit_id","probe_letter"]], on="unit_id", how="left", validate="many_to_one")
    selected = d.loc[d.selected].copy()
    selected["location"] = np.where(selected.cohort.eq("MouseV2"), selected.probe_letter,
                                    selected.cohort.str.replace("Allen ", "", regex=False))
    session = selected.groupby(["cohort","session_id","location"], as_index=False).agg(
        units=("unit_id","size"), mean_ttfs_ms=("preferred_0_250_ttfs_ms","mean"),
        median_ttfs_ms=("preferred_0_250_ttfs_ms","median"))
    session["included_min_units"] = session.units.ge(MIN_UNITS_PER_ESTIMATE)
    displayed = session.loc[session.included_min_units].copy()
    eligible_keys = displayed[["cohort", "session_id", "location"]]
    selected_for_hist = selected.merge(
        eligible_keys, on=["cohort", "session_id", "location"], how="inner",
        validate="many_to_one"
    )
    summary = displayed.groupby(["cohort","location"], as_index=False).agg(
        sessions=("session_id","nunique"), units=("units","sum"), mean_ms=("mean_ttfs_ms","mean"),
        sem_ms=("mean_ttfs_ms","sem"))

    order = ["B","C","A","E","Brain Observatory","Functional Connectivity"]
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True,gridspec_kw={"width_ratios":[1.7,1]})
    for i, loc in enumerate(order):
        g=displayed.loc[displayed.location.eq(loc)]
        if g.empty: continue
        cohort=g.cohort.iloc[0]; color=COLORS[cohort]
        jitter=np.linspace(-.09,.09,len(g)) if len(g)>1 else np.array([0.])
        sizes = 18 + 1.2 * g.units.to_numpy()
        axes[0].scatter(i+jitter,g.mean_ttfs_ms,s=sizes,color=color,alpha=.65,
                        edgecolor="white", linewidth=.35)
        axes[0].errorbar(i,g.mean_ttfs_ms.mean(),yerr=g.mean_ttfs_ms.sem(),fmt="_",ms=18,lw=2.2,color="k",capsize=4)
    axes[0].set(xticks=np.arange(len(order)),xticklabels=["B","C","A","E","Allen BO","Allen FC"],
                ylabel="Session mean preferred TTFS (ms)",title="Response-selected V1 sessions / probe locations")
    for cohort,g in selected_for_hist.groupby("cohort"):
        axes[1].hist(g.preferred_0_250_ttfs_ms,bins=np.arange(29.5,100.5,2),density=True,histtype="step",lw=2,
                     color=COLORS[cohort],label=f"{cohort} (n={len(g):,})")
    axes[1].set(xlabel="Preferred-polarity TTFS (ms)",ylabel="Density",title="Selected-unit distributions")
    axes[1].legend(frameon=False,fontsize=7)
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    excluded = session.loc[~session.included_min_units].groupby("cohort").size().to_dict()
    fig.suptitle(
        "Figure 3 V1 TTFS with latency-independent positive-response filtering\n"
        f"Odd trials select amplitude (FDR q<0.01, Δrate≥2 Hz); ≥{MIN_UNITS_PER_ESTIMATE} selected units per estimate; point area reflects n"
    )
    fig.savefig(OUT/"figure3_response_filtered_preferred_ttfs.png",dpi=220,bbox_inches="tight")
    fig.savefig(OUT/"figure3_response_filtered_preferred_ttfs.pdf",bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(OUT/"figure3_response_filtered_preferred_ttfs_units.csv",index=False)
    session.to_csv(OUT/"figure3_response_filtered_preferred_ttfs_sessions.csv",index=False)
    session.loc[~session.included_min_units].to_csv(
        OUT/"figure3_response_filtered_preferred_ttfs_excluded_low_n.csv", index=False
    )
    summary.to_csv(OUT/"figure3_response_filtered_preferred_ttfs_summary.csv",index=False)
    print(summary.to_string(index=False))
    print(f"\nExcluded estimates with n<{MIN_UNITS_PER_ESTIMATE}: {excluded}")

if __name__ == "__main__": main()
