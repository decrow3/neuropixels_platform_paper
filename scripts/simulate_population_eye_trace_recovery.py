#!/usr/bin/env python3
"""Synthetic upper-bound test for inferring a shared eye trace from population spikes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
SESSION_ID=746083955
DEFAULT_CACHE=ROOT/"artifacts"/"allen_population_gaze_rf"/f"session_{SESSION_ID}"
DEFAULT_OUTPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"05_synthetic"
SEED=20260815
POPULATION_SIZES=(16,32,64,128,256,512,1024)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--trials",type=int,default=800)
    return p.parse_args()


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan


def infer_one(counts,x,y,orientation,parameters,grid):
    cx,cy,sigma,baseline,amplitudes=parameters
    dx,dy=np.meshgrid(grid,grid);candidate=np.column_stack([dx.ravel(),dy.ravel()])
    px=x-candidate[:,0,None];py=y-candidate[:,1,None]
    spatial=np.exp(-.5*(((px-cx[None,:])/sigma[None,:])**2+((py-cy[None,:])/sigma[None,:])**2))
    rate=baseline[None,:]+amplitudes[None,:,orientation]*spatial
    contribution=counts[None,:]*np.log(np.maximum(rate,1e-9))-rate
    cumulative=np.cumsum(contribution,axis=1)
    result={}
    for n in POPULATION_SIZES:
        best=int(np.argmax(cumulative[:,n-1]));result[n]=candidate[best]
    return result


def simulate(trials,trial_limit):
    rng=np.random.default_rng(SEED)
    valid=trials.loc[trials.valid_gaze.astype(bool)].sort_values("start_time")
    indices=np.linspace(0,len(valid)-1,min(trial_limit,len(valid)),dtype=int)
    selected=valid.iloc[indices].reset_index(drop=True)
    n=max(POPULATION_SIZES)
    cx=rng.uniform(-35,35,n);cy=rng.uniform(-35,35,n)
    sigma=np.exp(rng.normal(np.log(13),.32,n)).clip(5,32)
    baseline=rng.uniform(.015,.12,n)
    amplitudes=np.exp(rng.normal(np.log(.55),.55,(n,3))).clip(.08,3.0)
    parameters=(cx,cy,sigma,baseline,amplitudes)
    true_dx=selected.gaze_dx_deg.to_numpy(float);true_dy=selected.gaze_dy_deg.to_numpy(float)
    x=selected.x_position.to_numpy(float)-true_dx;y=selected.y_position.to_numpy(float)-true_dy
    orientation=selected.orientation_index.to_numpy(int)
    spatial=np.exp(-.5*(((x[:,None]-cx[None,:])/sigma[None,:])**2+
                        ((y[:,None]-cy[None,:])/sigma[None,:])**2))
    rate=baseline[None,:]+np.take_along_axis(amplitudes[None,:,:],orientation[:,None,None],axis=2)[:,:,0]*spatial
    counts=rng.poisson(rate)
    grid=np.arange(-5,5.001,.5)
    inferred={n:np.zeros((len(selected),2)) for n in POPULATION_SIZES}
    for i in range(len(selected)):
        local=infer_one(counts[i],selected.x_position.iloc[i],selected.y_position.iloc[i],
                        orientation[i],parameters,grid)
        for n,value in local.items():inferred[n][i]=value
        if (i+1)%100==0:print(f"Synthetic inference: {i+1}/{len(selected)} trials",flush=True)
    rows=[];trial_rows=[]
    for n in POPULATION_SIZES:
        pred=inferred[n]
        rows.append({"population_units":n,"trials":len(selected),
            "x_correlation":correlation(true_dx,pred[:,0]),"y_correlation":correlation(true_dy,pred[:,1]),
            "x_rmse_deg":float(np.sqrt(np.mean((true_dx-pred[:,0])**2))),
            "y_rmse_deg":float(np.sqrt(np.mean((true_dy-pred[:,1])**2))),
            "vector_rmse_deg":float(np.sqrt(np.mean((true_dx-pred[:,0])**2+(true_dy-pred[:,1])**2)))})
        for i in range(len(selected)):
            trial_rows.append({"population_units":n,"trial_index":i,"start_time":selected.start_time.iloc[i],
                "true_dx_deg":true_dx[i],"true_dy_deg":true_dy[i],
                "inferred_dx_deg":pred[i,0],"inferred_dy_deg":pred[i,1]})
    # Negative control: preserve counts but permute the trace labels used for scoring.
    perm=rng.permutation(len(selected));pred=inferred[max(POPULATION_SIZES)]
    control={"population_units":max(POPULATION_SIZES),"x_correlation":correlation(true_dx[perm],pred[:,0]),
             "y_correlation":correlation(true_dy[perm],pred[:,1])}
    return pd.DataFrame(rows),pd.DataFrame(trial_rows),control


def render(summary,trials,control,path):
    fig,axes=plt.subplots(2,2,figsize=(12,8.7),constrained_layout=True)
    axes[0,0].plot(summary.population_units,summary.x_correlation,"o-",color="#3366aa",label="horizontal")
    axes[0,0].plot(summary.population_units,summary.y_correlation,"s-",color="#d97736",label="vertical")
    axes[0,0].axhline(0,color="#555",ls="--");axes[0,0].set(xscale="log",xlabel="Simultaneously recorded units",
        ylabel="True–inferred correlation",title="Trace recovery versus population size")
    axes[0,0].legend(frameon=False)
    axes[0,1].plot(summary.population_units,summary.vector_rmse_deg,"o-",color="#7a8f3a")
    axes[0,1].set(xscale="log",xlabel="Simultaneously recorded units",ylabel="Vector RMSE (deg)",title="Trial-wise displacement error")
    largest=trials.loc[trials.population_units.eq(summary.population_units.max())].copy()
    axes[1,0].scatter(largest.true_dx_deg,largest.inferred_dx_deg,s=10,alpha=.35,color="#3366aa",label="horizontal")
    axes[1,0].scatter(largest.true_dy_deg,largest.inferred_dy_deg,s=10,alpha=.35,color="#d97736",label="vertical")
    axes[1,0].plot([-5,5],[-5,5],"--",color="#555",lw=1);axes[1,0].set(xlim=(-5,5),ylim=(-5,5),
        xlabel="True displacement (deg)",ylabel="Inferred displacement (deg)",title="1,024-unit trial estimates")
    axes[1,0].legend(frameon=False)
    segment=largest.iloc[:120]
    axes[1,1].plot(segment.trial_index,segment.true_dy_deg,color="#d97736",lw=2,label="true vertical")
    axes[1,1].plot(segment.trial_index,segment.inferred_dy_deg,color="#222",lw=1,alpha=.8,label="inferred vertical")
    axes[1,1].set(xlabel="Ordered Gabor trials",ylabel="Displacement (deg)",title="Example recovered trace segment")
    axes[1,1].legend(frameon=False)
    for axis in axes.ravel():axis.grid(alpha=.14)
    fig.suptitle(f"Synthetic population eye-trace identifiability · session {SESSION_ID} stimulus/gaze sequence",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    trials=pd.read_csv(args.cache_dir.resolve()/"gabor_trial_gaze_table.csv",low_memory=False)
    summary,trial_results,control=simulate(trials,args.trials)
    summary.to_csv(output/"synthetic_recovery_by_population_size.csv",index=False,float_format="%.9g")
    trial_results.to_csv(output/"synthetic_trial_inference.csv",index=False,float_format="%.9g")
    (output/"control_summary.json").write_text(json.dumps(control,indent=2)+"\n")
    render(summary,trial_results,control,output/"Figure_synthetic_eye_trace_recovery.png")
    print(summary.to_string(index=False));print(control)


if __name__=="__main__":main()
