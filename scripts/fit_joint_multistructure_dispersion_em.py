#!/usr/bin/env python3
"""Alternating zero-mean registration of V1/HVA/LGd dispersion fields."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.test_v1_rf_size_corroboration import smooth_values, interpolator, robust_scale, best_shift


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"
INPUT=BASE/"joint_multistructure_dispersion_checkpoint"/"all_structure_dispersion_descriptors.csv.gz"
ELIGIBLE=BASE/"joint_multistructure_dispersion_checkpoint"/"three_structure_eligible_sessions.csv"
V1_OFFSET=BASE/"gaze_censor_anchor_checkpoint"/"all_session_anatomy_offsets.csv"
OUT=BASE/"joint_multistructure_dispersion_em"
GROUPS=("V1","HVA","LGd");MINIMUM={"V1":30,"HVA":50,"LGd":10}


def grids():
    axis=np.arange(-75.,75.1,2.);xx,yy=np.meshgrid(axis,axis);grid=np.c_[xx.ravel(),yy.ravel()]
    sa=np.arange(-30.,30.1,2.);sx,sy=np.meshgrid(sa,sa);shifts=np.c_[sx.ravel(),sy.ravel()]
    return axis,grid,sa,shifts


def make_base_surfaces(data,session_ids,grid,axis,bandwidth=12.):
    """Smooth each session once in its observed RF coordinates.

    A translation moves the resulting scalar field rigidly, so subsequent EM
    iterations only need interpolation rather than another O(cells x pixels)
    kernel-smoothing pass.
    """
    surfaces={g:{} for g in GROUPS}
    for g in GROUPS:
        selected=data.loc[data.structure_group.eq(g)&data.ecephys_session_id.isin(session_ids)].dropna(subset=["log2_residual_trace"])
        for sid,local in selected.groupby("ecephys_session_id",observed=True):
            points=local[["rf_x","rf_y"]].to_numpy(float)
            values=smooth_values(points,local.log2_residual_trace.to_numpy(float),grid,bandwidth)
            surfaces[g][int(sid)]=interpolator(values,axis)
    return surfaces


def translated_surfaces(base_surfaces,current,grid):
    surfaces={g:{} for g in GROUPS}
    for g in GROUPS:
        for sid,base_interp in base_surfaces[g].items():
            # If g(r)=f(r-d), evaluate the unshifted field at r-d.
            shifted_grid=grid-current[sid]
            surfaces[g][sid]=base_interp(shifted_grid[:,[1,0]])
    return surfaces


def loo_template(surfaces,target,min_sessions=5):
    values=[v for sid,v in surfaces.items() if sid!=target]
    stack=np.stack(values);support=np.isfinite(stack).sum(0);total=np.nansum(stack,axis=0)
    out=np.divide(total,support,out=np.full(support.shape,np.nan,float),where=support>0);out[support<min_sessions]=np.nan
    return out


def loss_grid_vectorized(points,observed,template_interpolator,shifts,scale):
    """Equivalent to loss_grid, with one interpolator call for all candidates."""
    query=(points[None,:,:]+shifts[:,None,:])[:,:,[1,0]]
    predicted=template_interpolator(query.reshape(-1,2)).reshape(len(shifts),len(points))
    valid=np.isfinite(predicted)&np.isfinite(observed)[None,:]
    residual=(observed[None,:]-predicted)/scale
    absolute=np.abs(residual)
    huber=np.where(absolute<=1,.5*residual**2,absolute-.5)
    huber[~valid]=0
    counts=valid.sum(1)
    losses=np.divide(huber.sum(1),counts,out=np.full(len(shifts),np.nan),where=counts>=10)
    losses+=.75*(1-valid.mean(1))
    losses[counts<10]=np.nan
    return losses


def update_one(sid,data,surfaces,axis,candidates,scales,groups=GROUPS):
    component={};joint=np.zeros(len(candidates));used=0
    for g in groups:
        local=data.loc[data.ecephys_session_id.eq(sid)&data.structure_group.eq(g)].dropna(subset=["log2_residual_trace"])
        if len(local)<MINIMUM[g] or sid not in surfaces[g]:continue
        interp=interpolator(loo_template(surfaces[g],sid),axis)
        loss=loss_grid_vectorized(local[["rf_x","rf_y"]].to_numpy(float),local.log2_residual_trace.to_numpy(float),interp,candidates,scales[g])
        component[g]=loss;joint+=np.nan_to_num(loss,nan=5.);used+=1
    joint/=max(used,1);shift,minimum=best_shift(joint,candidates)
    return shift,minimum,component,joint


def recenter(shifts,ids,bound=30.):
    matrix=np.stack([shifts[s] for s in ids]);matrix-=matrix.mean(0);matrix=np.clip(matrix,-bound,bound)
    matrix-=matrix.mean(0)
    return {s:matrix[i] for i,s in enumerate(ids)}


def run_em(name,initial,data,ids,axis,grid,candidates,scales,base_surfaces,max_iter=60,damping=.25):
    current=recenter({s:np.asarray(initial[s],float).copy() for s in ids},ids);trace=[]
    for iteration in range(max_iter):
        surfaces=translated_surfaces(base_surfaces,current,grid);proposed={};objective=0
        for sid in ids:
            shift,minimum,_,_=update_one(sid,data,surfaces,axis,candidates,scales);proposed[sid]=shift;objective+=minimum
        proposed=recenter(proposed,ids)
        updated=recenter({s:(1-damping)*current[s]+damping*proposed[s] for s in ids},ids)
        movement=np.median([np.linalg.norm(updated[s]-current[s]) for s in ids]);maximum=max(np.linalg.norm(updated[s]-current[s]) for s in ids)
        trace.append({"initialization":name,"iteration":iteration,"objective":objective,"median_movement_deg":movement,"maximum_movement_deg":maximum})
        current=updated
        if maximum<=.5:break
    surfaces=translated_surfaces(base_surfaces,current,grid);details={};objective=0
    for sid in ids:
        shift,minimum,component,joint=update_one(sid,data,surfaces,axis,candidates,scales);details[sid]={"joint_shift":shift,"minimum":minimum,"component":component,"joint":joint};objective+=minimum
    return current,pd.DataFrame(trace),details,objective


def component_optima(details,candidates):
    rows=[]
    for sid,item in details.items():
        joint_shift,_=best_shift(item["joint"],candidates)
        rows.append({"ecephys_session_id":sid,"component":"joint","shift_az_deg":joint_shift[0],"shift_el_deg":joint_shift[1]})
        for g,loss in item["component"].items():
            shift,_=best_shift(loss,candidates);rows.append({"ecephys_session_id":sid,"component":g,"shift_az_deg":shift[0],"shift_el_deg":shift[1],"distance_from_joint_deg":np.linalg.norm(shift-joint_shift)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);data=pd.read_csv(INPUT,low_memory=False);ids=pd.read_csv(ELIGIBLE).ecephys_session_id.astype(int).tolist();data=data.loc[data.ecephys_session_id.isin(ids)]
    axis,grid,sa,candidates=grids();scales={g:robust_scale(data.loc[data.structure_group.eq(g),"log2_residual_trace"].dropna().to_numpy(float),.05) for g in GROUPS}
    base_surfaces=make_base_surfaces(data,ids,grid,axis)
    zero={s:np.zeros(2) for s in ids};v1=pd.read_csv(V1_OFFSET).set_index("ecephys_session_id");anatomy={s:np.clip(v1.loc[s,["offset_az_relative_deg","offset_el_relative_deg"]].to_numpy(float),-30,30) for s in ids}
    rng=np.random.default_rng(20260817);initials={"zero":zero,"v1_anatomy":anatomy}
    for k in range(3):initials[f"random_{k}"]={s:rng.uniform(-10,10,2) for s in ids}
    summaries=[];all_traces=[];solutions={};detail_by_name={}
    for name,initial in initials.items():
        solution,trace,detail,objective=run_em(name,initial,data,ids,axis,grid,candidates,scales,base_surfaces);solutions[name]=solution;detail_by_name[name]=detail;all_traces.append(trace);summaries.append({"initialization":name,"iterations":len(trace),"final_objective":objective,"final_median_movement_deg":trace.iloc[-1].median_movement_deg,"final_maximum_movement_deg":trace.iloc[-1].maximum_movement_deg})
    summary=pd.DataFrame(summaries).sort_values("final_objective");summary.to_csv(OUT/"initialization_summary.csv",index=False);pd.concat(all_traces).to_csv(OUT/"iteration_trace.csv",index=False)
    best=summary.iloc[0].initialization;best_solution=solutions[best];rows=[]
    for name,solution in solutions.items():
        for sid in ids:rows.append({"initialization":name,"ecephys_session_id":sid,"shift_az_deg":solution[sid][0],"shift_el_deg":solution[sid][1],"distance_from_best_deg":np.linalg.norm(solution[sid]-best_solution[sid])})
    sol=pd.DataFrame(rows);sol.to_csv(OUT/"all_initialization_session_shifts.csv",index=False)
    comp=component_optima(detail_by_name[best],candidates);comp.to_csv(OUT/"best_solution_component_optima.csv",index=False)
    agreement=comp.loc[comp.component.ne("joint")].groupby("ecephys_session_id").distance_from_joint_deg.agg(["median","max"]).reset_index();agreement.to_csv(OUT/"component_joint_agreement.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for name,local in sol.groupby("initialization"):
        axes[0].scatter(local.shift_az_deg,local.shift_el_deg,label=name,alpha=.7)
    axes[0].axhline(0,color=".8");axes[0].axvline(0,color=".8");axes[0].set(xlabel="az shift",ylabel="el shift",title="Solutions by initialization",aspect="equal");axes[0].legend(fontsize=7,frameon=False)
    pivot=sol.pivot(index="ecephys_session_id",columns="initialization",values="distance_from_best_deg")
    axes[1].boxplot([pivot[c].dropna() for c in pivot],labels=list(pivot.columns),vert=True);axes[1].tick_params(axis="x",rotation=35);axes[1].set(ylabel="distance from best solution (deg)",title="Initialization sensitivity")
    for g,local in comp.loc[comp.component.ne("joint")].groupby("component"):
        axes[2].scatter(local.ecephys_session_id.astype(str).str[-3:],local.distance_from_joint_deg,label=g)
    axes[2].set(ylabel="component optimum distance from joint (deg)",xlabel="session suffix",title="Do structures agree after EM?");axes[2].tick_params(axis="x",rotation=90);axes[2].legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/"Figure_em_convergence_and_component_agreement.png",dpi=180);plt.close(fig)
    manifest={"eligible_sessions":ids,"groups":GROUPS,"zero_mean_translation_constraint":True,"leave_one_session_out_updates":True,"surface_bandwidth_deg":12,"shift_grid_step_deg":2,"shift_bound_deg":30,"update_damping":.25,"convergence_maximum_update_deg":.5,"component_weighting":"equal mean Huber losses after group-specific descriptor robust scaling","best_initialization":best,"scales":scales}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2));print(summary.to_string(index=False));print('\ninitialization distance median\n',sol.groupby('initialization').distance_from_best_deg.median());print('\ncomponent agreement\n',agreement.describe().to_string())


if __name__=="__main__":main()
