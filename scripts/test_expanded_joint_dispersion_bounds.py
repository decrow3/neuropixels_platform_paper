#!/usr/bin/env python3
"""Expanded-bound, support-limited test of multi-structure dispersion localization."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.fit_joint_multistructure_dispersion_em import GROUPS, loss_grid_vectorized
from scripts.test_v1_rf_size_corroboration import smooth_values, interpolator, robust_scale, best_shift

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"
DESC=BASE/"joint_multistructure_dispersion_checkpoint"/"all_structure_dispersion_descriptors.csv.gz"
ELIG=BASE/"joint_multistructure_dispersion_checkpoint"/"three_structure_eligible_sessions.csv"
EM=BASE/"joint_multistructure_dispersion_em"
OUT=BASE/"expanded_bound_support_limited_test"
CASES=(715093703,754829445,760345702)
BANDWIDTH=12.; MAX_THIRD_NEIGHBOR=24.; MIN_SESSIONS=5


def support_limited_surface(local,shift,grid):
    points=local[["rf_x","rf_y"]].to_numpy(float)+shift
    values=local.log2_residual_trace.to_numpy(float)
    surface=smooth_values(points,values,grid,BANDWIDTH)
    distance=cKDTree(points).query(grid,k=min(3,len(points)))[0]
    if distance.ndim==2:distance=distance[:,-1]
    surface[distance>MAX_THIRD_NEIGHBOR]=np.nan
    return surface


def loo_template(surfaces,target):
    stack=np.stack([v for sid,v in surfaces.items() if sid!=target])
    support=np.isfinite(stack).sum(0)
    out=np.divide(np.nansum(stack,axis=0),support,out=np.full(len(support),np.nan),where=support>0)
    out[support<MIN_SESSIONS]=np.nan
    return out,support


def relative(loss):
    finite=loss[np.isfinite(loss)];spread=max(np.quantile(finite,.75)-np.quantile(finite,.25),.02)
    return (loss-np.nanmin(loss))/spread


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DESC,low_memory=False);ids=pd.read_csv(ELIG).ecephys_session_id.astype(int).tolist();data=data[data.ecephys_session_id.isin(ids)]
    init=pd.read_csv(EM/"initialization_summary.csv").sort_values("final_objective").iloc[0].initialization
    shift_table=pd.read_csv(EM/"all_initialization_session_shifts.csv");shift_table=shift_table[shift_table.initialization.eq(init)].set_index("ecephys_session_id")
    current={sid:shift_table.loc[sid,["shift_az_deg","shift_el_deg"]].to_numpy(float) for sid in ids}
    axis=np.arange(-120.,120.1,2.);xx,yy=np.meshgrid(axis,axis);grid=np.c_[xx.ravel(),yy.ravel()]
    sa=np.arange(-60.,60.1,2.);sx,sy=np.meshgrid(sa,sa);candidates=np.c_[sx.ravel(),sy.ravel()]
    scales={g:robust_scale(data.loc[data.structure_group.eq(g),"log2_residual_trace"].dropna().to_numpy(float),.05) for g in GROUPS}
    surfaces={g:{} for g in GROUPS}
    for g in GROUPS:
        for sid,local in data[data.structure_group.eq(g)].groupby("ecephys_session_id"):
            surfaces[g][int(sid)]=support_limited_surface(local,current[int(sid)],grid)
    rows=[];payload={}
    for sid in CASES:
        payload[sid]={};rels={}
        for g in GROUPS:
            local=data[(data.ecephys_session_id.eq(sid))&data.structure_group.eq(g)].dropna(subset=["log2_residual_trace"])
            template,support=loo_template(surfaces[g],sid);interp=interpolator(template,axis)
            loss=loss_grid_vectorized(local[["rf_x","rf_y"]].to_numpy(float),local.log2_residual_trace.to_numpy(float),interp,candidates,scales[g])
            shift,minimum=best_shift(loss,candidates);rels[g]=relative(loss);payload[sid][g]=(loss,shift,template,support)
            rows.append({"ecephys_session_id":sid,"component":g,"shift_az_deg":shift[0],"shift_el_deg":shift[1],"minimum_loss":minimum,"at_30deg_boundary":bool(np.any(np.abs(shift)>=30)),"at_60deg_boundary":bool(np.any(np.abs(shift)>=60)),"valid_cells":len(local)})
        for label,groups in (("V1+LGd",("V1","LGd")),("all",GROUPS)):
            loss=np.mean([rels[g] for g in groups],axis=0);shift,minimum=best_shift(loss,candidates);payload[sid][label]=(loss,shift,None,None)
            rows.append({"ecephys_session_id":sid,"component":label,"shift_az_deg":shift[0],"shift_el_deg":shift[1],"minimum_loss":minimum,"at_30deg_boundary":bool(np.any(np.abs(shift)>=30)),"at_60deg_boundary":bool(np.any(np.abs(shift)>=60)),"valid_cells":np.nan})
    result=pd.DataFrame(rows);result.to_csv(OUT/"expanded_bound_component_optima.csv",index=False)
    panels=("V1","HVA","LGd","V1+LGd","all");fig,axes=plt.subplots(len(CASES),len(panels),figsize=(19,11),squeeze=False)
    for r,sid in enumerate(CASES):
        for c,label in enumerate(panels):
            loss,shift,_,_=payload[sid][label];z=(loss-np.nanmin(loss)).reshape(len(sa),len(sa))
            vmax=np.nanquantile(z,.8);im=axes[r,c].imshow(z,origin="lower",extent=[-60,60,-60,60],cmap="viridis_r",vmin=0,vmax=vmax,aspect="equal")
            axes[r,c].axvline(-30,color="w",lw=.5,ls=":");axes[r,c].axvline(30,color="w",lw=.5,ls=":");axes[r,c].axhline(-30,color="w",lw=.5,ls=":");axes[r,c].axhline(30,color="w",lw=.5,ls=":")
            axes[r,c].scatter(*shift,color="red",marker="x",s=60);axes[r,c].set_title(f"{sid} {label}\n({shift[0]:+.0f}, {shift[1]:+.0f})°")
            axes[r,c].set_xlabel("azimuth shift (deg)");axes[r,c].set_ylabel("elevation shift (deg)")
    fig.suptitle("Expanded ±60° translation test with distance-limited template support",y=.995);fig.tight_layout();fig.savefig(OUT/"Figure_expanded_bound_landscapes.png",dpi=180);plt.close(fig)
    (OUT/"run_manifest.json").write_text(json.dumps({"cases":CASES,"translation_bound_deg":60,"translation_step_deg":2,"surface_axis_deg":[-120,120],"surface_bandwidth_deg":BANDWIDTH,"support_rule":f"third nearest cell <= {MAX_THIRD_NEIGHBOR} deg","minimum_training_sessions_per_template_pixel":MIN_SESSIONS,"training_shifts":f"best damped EM initialization {init}","component_combination":"equal mean of component relative losses scaled by each landscape IQR"},indent=2))
    print(result.to_string(index=False))

if __name__=="__main__":main()
