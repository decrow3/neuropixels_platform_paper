#!/usr/bin/env python3
"""Summarize original Allen-code reproduction and bounded-baseline refits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
SESSIONS=(746083955,755434585,760693773,798911424)
DEFAULT_INPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"02_allen_baseline"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT)
    return p.parse_args()


def collect(root):
    tables=[]; summaries=[]
    for sid in SESSIONS:
        x=pd.read_csv(root/f"session_{sid}"/"session_rf_fit_population.csv",low_memory=False)
        x["session_id"]=sid; tables.append(x)
        q=x.loc[x["published_like_qc"].astype(bool)].copy()
        finite=lambda a,b: q[[a,b]].dropna()
        exact=lambda a,b,tol=1e-5: float(np.isclose(finite(a,b)[a],finite(a,b)[b],rtol=tol,atol=tol).mean())
        primary=q.loc[q["allen_finite"].astype(bool)&q["corrected_success"].astype(bool)&
                      ~q["corrected_censored"].astype(bool)].copy()
        summaries.append({
            "session_id":sid,"visual_units":len(x),"qc_units":len(q),"primary_units":len(primary),
            "azimuth_exact_fraction":exact("released_azimuth_rf_deg","threshold_azimuth_deg"),
            "elevation_exact_fraction":exact("released_elevation_rf_deg","threshold_elevation_deg"),
            "threshold_area_exact_fraction":exact("released_area_rf_deg2","threshold_area_deg2"),
            "width_exact_fraction_after_axis_mapping":exact("released_width_rf_deg","allen_sigma_y_deg",1e-3),
            "height_exact_fraction_after_axis_mapping":exact("released_height_rf_deg","allen_sigma_x_deg",1e-3),
            "median_log2_corrected_over_allen_area":float(np.log2(primary.corrected_halfmax_area_deg2/
                                                                   primary.allen_halfmax_area_deg2).median()),
            "corrected_censored_qc_fraction":float(q.corrected_censored.mean()),
        })
    return pd.concat(tables,ignore_index=True),pd.DataFrame(summaries)


def render(all_units,summary,path):
    q=all_units.loc[all_units["published_like_qc"].astype(bool)].copy()
    primary=q.loc[q["allen_finite"].astype(bool)&q["corrected_success"].astype(bool)&
                  ~q["corrected_censored"].astype(bool)].copy()
    labels=summary.session_id.astype(str); x=np.arange(len(summary)); width=.25
    colors={"V1":"#3366aa","HVA":"#d97736"}
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.2),constrained_layout=True)
    for offset,col,label in [(-width,"azimuth_exact_fraction","Azimuth"),(0,"elevation_exact_fraction","Elevation"),
                             (width,"threshold_area_exact_fraction","Area")]:
        axes[0,0].bar(x+offset,100*summary[col],width,label=label)
    axes[0,0].set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel="Exact matches (%)",
                  title="Released threshold metrics reproduced")
    axes[0,0].tick_params(axis="x", rotation=30)
    axes[0,0].legend(frameon=False)
    for group in ("V1","HVA"):
        local=q.loc[q.group.eq(group)]
        axes[0,1].scatter(local.released_area_rf_deg2,local.threshold_area_deg2,s=8,alpha=.18,
                          color=colors[group],label=f"{group} (n={len(local)})")
        axes[1,0].scatter(local.released_width_rf_deg,local.allen_sigma_y_deg,s=8,alpha=.18,
                          color=colors[group],label=group)
        axes[1,0].scatter(local.released_height_rf_deg,local.allen_sigma_x_deg,s=8,alpha=.18,
                          color=colors[group],marker="x")
    axes[0,1].plot([100,2500],[100,2500],"--",color="#555",lw=1)
    axes[0,1].set(xscale="log",yscale="log",xlim=(80,3000),ylim=(80,3000),
                  xlabel="Released threshold area (deg²)",ylabel="Direct recomputation (deg²)",
                  title="Peak-component area")
    axes[0,1].legend(frameon=False)
    lim=np.nanquantile(np.r_[q.released_width_rf_deg,q.released_height_rf_deg,
                             q.allen_sigma_x_deg,q.allen_sigma_y_deg],.99)
    axes[1,0].plot([0,lim],[0,lim],"--",color="#555",lw=1)
    axes[1,0].set(xlim=(0,lim),ylim=(0,lim),xlabel="Released Gaussian dimension (deg)",
                  ylabel="Direct recomputation (deg)",title="Width→row σ; height→column σ")
    for group in ("V1","HVA"):
        local=primary.loc[primary.group.eq(group)]
        edge=np.minimum.reduce([local.threshold_azimuth_deg-10,90-local.threshold_azimuth_deg,
                                local.threshold_elevation_deg+30,50-local.threshold_elevation_deg])
        axes[1,1].scatter(edge,np.log2(local.corrected_halfmax_area_deg2/local.allen_halfmax_area_deg2),
                          s=9,alpha=.2,color=colors[group],label=f"{group} (n={len(local)})")
    axes[1,1].axhline(0,color="#555",ls="--",lw=1); axes[1,1].axvline(10,color="#777",ls=":",lw=1)
    axes[1,1].set(xlabel="Threshold center distance from screen edge (deg)",
                  ylabel="log₂(bounded-baseline / Allen area)",title="Size correction versus edge proximity")
    axes[1,1].legend(frameon=False)
    for axis in axes.ravel(): axis.grid(alpha=.14)
    fig.suptitle("Four-session check of Allen RF code and bounded-baseline correction",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();root=args.input_dir.resolve();all_units,summary=collect(root)
    all_units.to_csv(root/"all_session_baseline_fits.csv",index=False,float_format="%.9g")
    summary.to_csv(root/"all_session_baseline_summary.csv",index=False,float_format="%.9g")
    render(all_units,summary,root/"Figure_all_session_baseline_summary.png")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
