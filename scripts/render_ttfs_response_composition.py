#!/usr/bin/env python3
"""Show how independently verified flash responsiveness changes the TTFS gap."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/figure3/06r_probe_flash_photoartifact_pilot"
OUT=BASE/"ttfs_metric_decomposition"
COLORS={"Allen Brain Observatory":"#6F63A6","Allen Functional Connectivity":"#B07AA1","MouseV2":"#D95F02"}


def centers(data, metric, population):
    x=data.loc[data[metric].lt(100)]
    s=x.groupby(["cohort","session_id"])[metric].mean().reset_index()
    rows=[]
    for cohort,g in s.groupby("cohort"):
        rows.append({"cohort":cohort,"metric":metric,"population":population,"sessions":g.session_id.nunique(),"equal_session_mean_ms":g[metric].mean(),"session_sd_ms":g[metric].std()})
    return rows


def main():
    components=pd.read_csv(OUT/"unit_level_ttfs_components.csv")
    selection=pd.read_csv(BASE/"crossvalidated_latency_independent_psths/crossvalidated_unit_selection_audit.csv")
    ids=selection.loc[selection.selected_positive_responder].groupby(["cohort","session_id","unit_id"]).size().reset_index()[["cohort","session_id","unit_id"]]
    responsive=components.merge(ids,on=["cohort","session_id","unit_id"],validate="one_to_one")
    rows=[]
    for metric in ("released_ttfs_ms","recomputed_ttfs_ms"):
        rows+=centers(components,metric,"all common-QC V1")
        rows+=centers(responsive,metric,"independently selected positive responders")
    summary=pd.DataFrame(rows)
    mouse=summary.loc[summary.cohort.eq("MouseV2"),["metric","population","equal_session_mean_ms"]].rename(columns={"equal_session_mean_ms":"mouse_ms"})
    gaps=summary.loc[~summary.cohort.eq("MouseV2")].merge(mouse,on=["metric","population"])
    gaps["mouse_minus_allen_ms"]=gaps.mouse_ms-gaps.equal_session_mean_ms
    raw=gaps.loc[gaps.population.eq("all common-QC V1"),["cohort","metric","mouse_minus_allen_ms"]].rename(columns={"mouse_minus_allen_ms":"raw_gap_ms"})
    gaps=gaps.merge(raw,on=["cohort","metric"])
    gaps["gap_reduction_fraction"]=1-gaps.mouse_minus_allen_ms/gaps.raw_gap_ms
    summary.to_csv(OUT/"ttfs_response_composition_centers.csv",index=False)
    gaps.to_csv(OUT/"ttfs_response_composition_gaps.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    order=["Allen Brain Observatory","Allen Functional Connectivity","MouseV2"]
    labels=["Allen BO","Allen FC","MouseV2"]
    x=np.arange(3);width=.34
    for j,population in enumerate(("all common-QC V1","independently selected positive responders")):
        sub=summary.loc[(summary.metric=="released_ttfs_ms")&(summary.population==population)].set_index("cohort").loc[order]
        axes[0].bar(x+(j-.5)*width,sub.equal_session_mean_ms,width,color=[COLORS[c] for c in order],alpha=1 if j else .42,hatch="" if j else "//",label="response-selected" if j else "all common-QC")
    axes[0].set(xticks=x,xticklabels=labels,ylabel="Released TTFS equal-session mean (ms)",title="Independent response selection removes most of the gap")
    axes[0].legend(frameon=False)
    for i,cohort in enumerate(order[:2]):
        sub=gaps.loc[(gaps.cohort==cohort)&(gaps.metric=="released_ttfs_ms")]
        vals=sub.set_index("population").loc[["all common-QC V1","independently selected positive responders"],"mouse_minus_allen_ms"]
        axes[1].plot([0,1],vals,marker="o",lw=2.5,color=COLORS[cohort],label=labels[i])
    axes[1].set(xticks=[0,1],xticklabels=["All common-QC","Response-selected\non independent trials"],ylabel="MouseV2 minus Allen TTFS (ms)",title="Residual gap after removing nonresponders")
    axes[1].axhline(0,color="0.4",lw=.8);axes[1].legend(frameon=False)
    for ax in axes:ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("TTFS population composition, not flash-response onset, drives much of the discrepancy")
    fig.savefig(OUT/"ttfs_response_composition.png",dpi=220,bbox_inches="tight")
    fig.savefig(OUT/"ttfs_response_composition.pdf",bbox_inches="tight")


if __name__=="__main__":main()
