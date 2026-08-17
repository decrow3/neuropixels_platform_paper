#!/usr/bin/env python3
"""Render paired RF maps and cross-session summary for aperture gaze validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_GAZE=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"04_gaze"
DEFAULT_CACHE=ROOT/"artifacts"/"allen_population_gaze_rf"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gaze-dir",type=Path,default=DEFAULT_GAZE)
    p.add_argument("--cache-root",type=Path,default=DEFAULT_CACHE)
    return p.parse_args()


def kernel_map(counts,x,y,grid,bandwidth=4.0):
    xx,yy=np.meshgrid(grid,grid);out=np.full(xx.shape,np.nan)
    for r in range(len(grid)):
        for c in range(len(grid)):
            w=np.exp(-.5*(((x-xx[r,c])/bandwidth)**2+((y-yy[r,c])/bandwidth)**2))
            effective=w.sum()**2/np.square(w).sum() if w.sum()>0 else 0
            if effective>=3:out[r,c]=np.average(counts,weights=w)
    return out


def render_cases(sid,row,gaze_dir,cache_root,path):
    cache=cache_root/f"session_{sid}";population=pd.read_csv(cache/"visual_unit_population.csv",low_memory=False)
    trials=pd.read_csv(cache/"gabor_trial_gaze_table.csv",low_memory=False);counts=np.load(cache/"gabor_spike_counts.npz")["counts"]
    evaluation=pd.read_csv(gaze_dir/f"session_{sid}"/"evaluation_unit_results.csv",low_memory=False)
    chosen=evaluation.loc[evaluation.candidate.eq(row.chosen_candidate)].copy()
    ordered=[chosen.loc[chosen.test_deviance_improvement.idxmax()],
             chosen.loc[(chosen.test_deviance_improvement-chosen.test_deviance_improvement.median()).abs().idxmin()],
             chosen.loc[chosen.test_deviance_improvement.idxmin()]]
    labels=["largest held-out gain","typical held-out change","largest held-out loss"]
    valid=trials.valid_gaze.to_numpy(bool)&trials.trial_split.eq("test").to_numpy(bool)
    x0=trials.x_position.to_numpy(float);y0=trials.y_position.to_numpy(float)
    x1=x0-row.chosen_gain_x*trials.gaze_dx_deg.to_numpy(float)
    y1=y0-row.chosen_gain_y*trials.gaze_dy_deg.to_numpy(float);grid=np.linspace(-50,50,51)
    fig,axes=plt.subplots(3,3,figsize=(11.5,10.2),constrained_layout=True)
    for r,(unit,label) in enumerate(zip(ordered,labels)):
        index=population.index[population.ecephys_unit_id.eq(unit.ecephys_unit_id)][0]
        c=counts[index].astype(float)[valid]
        nominal=kernel_map(c,x0[valid],y0[valid],grid);corrected=kernel_map(c,x1[valid],y1[valid],grid)
        finite=np.r_[nominal[np.isfinite(nominal)],corrected[np.isfinite(corrected)]]
        vmin,vmax=np.quantile(finite,[.02,.98])
        for col,(matrix,title) in enumerate(((nominal,"Nominal"),(corrected,"Chosen gaze transform"))):
            im=axes[r,col].pcolormesh(grid,grid,matrix,shading="auto",cmap="viridis",vmin=vmin,vmax=vmax)
            fig.colorbar(im,ax=axes[r,col],label="spikes / 249 ms")
            axes[r,col].set_title(title)
        diff=corrected-nominal;limit=max(float(np.nanquantile(np.abs(diff),.98)),1e-8)
        im=axes[r,2].pcolormesh(grid,grid,diff,shading="auto",cmap="coolwarm",vmin=-limit,vmax=limit)
        fig.colorbar(im,ax=axes[r,2],label="corrected − nominal")
        axes[r,2].set_title("Paired difference")
        axes[r,0].set_ylabel(f"{label}\nunit {int(unit.ecephys_unit_id)} · {unit.group}\nΔdev={unit.test_deviance_improvement:+.4g}")
        for axis in axes[r]:axis.set(xlim=(-50,50),ylim=(-50,50),xlabel="Azimuth (deg)",aspect="equal")
    fig.suptitle(f"Session {sid}: gaze-corrected RF map audit · {row.chosen_candidate}",fontsize=15)
    fig.savefig(path,dpi=170,bbox_inches="tight");plt.close(fig)


def render_summary(summary,path):
    labels=summary.session_id.astype(str);x=np.arange(len(summary))
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.2),constrained_layout=True)
    axes[0].scatter(summary.chosen_gain_x,summary.chosen_gain_y,s=100,c=x,cmap="Blues",edgecolor="#222")
    zero=summary.loc[summary.chosen_gain_x.eq(0)&summary.chosen_gain_y.eq(0)]
    if len(zero):
        axes[0].text(.06,.04,"zero gain:\n"+"\n".join(zero.session_id.astype(str)),fontsize=8,va="bottom")
    for _,r in summary.loc[~(summary.chosen_gain_x.eq(0)&summary.chosen_gain_y.eq(0))].iterrows():
        axes[0].text(r.chosen_gain_x+.03,r.chosen_gain_y+.03,str(int(r.session_id)),fontsize=8)
    axes[0].set(xlim=(-.1,1.6),ylim=(-.1,1.6),xlabel="Chosen horizontal gain",ylabel="Chosen vertical gain",title="Shared transform per session")
    axes[1].bar(x,summary.median_evaluation_test_gain,color="#3366aa")
    axes[1].axhline(0,color="#555",ls="--");axes[1].set(xticks=x,xticklabels=labels,
        ylabel="Median held-out deviance gain",title="Unseen-neuron prediction")
    width=.36
    axes[2].bar(x-width/2,summary.median_evaluation_log2_area_ratio,width,color="#d97736",label="RF area")
    axes[2].bar(x+width/2,summary.median_evaluation_log2_amplitude_ratio,width,color="#7a8f3a",label="Amplitude")
    axes[2].axhline(0,color="#555",ls="--");axes[2].set(xticks=x,xticklabels=labels,
        ylabel="Median log₂ corrected / nominal",title="Sharpness and magnitude")
    axes[1].tick_params(axis="x",rotation=30);axes[2].tick_params(axis="x",rotation=30)
    axes[2].legend(frameon=False)
    for axis in axes:axis.grid(alpha=.14)
    fig.suptitle("Cross-session aperture RF gaze validation",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();gaze=args.gaze_dir.resolve();cache=args.cache_root.resolve()
    summary=pd.read_csv(gaze/"all_session_gaze_summary.csv")
    for row in summary.itertuples(index=False):
        render_cases(int(row.session_id),row,gaze,cache,gaze/f"session_{int(row.session_id)}"/"Figure_concrete_gaze_maps.png")
    render_summary(summary,gaze/"Figure_all_session_gaze_summary.png")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
