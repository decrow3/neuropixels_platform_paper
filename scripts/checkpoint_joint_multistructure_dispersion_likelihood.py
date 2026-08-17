#!/usr/bin/env python3
"""Concrete V1/HVA/LGd dispersion likelihoods and their joint translation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.checkpoint_multistructure_dispersion_fields import anatomical_residuals, dispersion, CCF
from scripts.test_v1_rf_size_corroboration import smooth_values, interpolator, loss_grid, robust_scale, best_shift


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"/"joint_multistructure_dispersion_checkpoint"
FITS=ROOT/"artifacts"/"allen_full_rf_production_v1"/"03_aggregate"/"all_session_unit_geometry_fits.csv"
UNITS=ROOT/"data"/"unit_table.csv";LGD=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"/"lgd_gabor_boundary_pilot"
CASES={715093703:"maximum held-out-positive LGd support",760345702:"prior V1 covariance localization success",754829445:"multi-probe LGd transverse-geometry case"}
GROUPS=("V1","HVA","LGd")


def load_all():
    units=pd.read_csv(UNITS,low_memory=False);keep=["ecephys_unit_id","ecephys_session_id","ecephys_structure_acronym","ecephys_probe_id","specimen_id",*CCF]
    small=units[keep];fits=pd.read_csv(FITS,low_memory=False);fits=fits.loc[fits.spatial_model.eq("aperture")].copy()
    cortex=fits.merge(small,on="ecephys_unit_id",how="left",suffixes=("","_unit"));cortex["ecephys_session_id"]=cortex.session_id.astype(int)
    cortex["rf_x"]=cortex.axis_center_x_deg;cortex["rf_y"]=cortex.axis_center_y_deg;cortex["structure_group"]=cortex.group;cortex["map_area"]=cortex.ecephys_structure_acronym
    cortex["center_bound"]=cortex.rf_x.abs().ge(59.9)|cortex.rf_y.abs().ge(59.9);cortex["source_gain"]=np.nan
    cols=[*keep,"rf_x","rf_y","structure_group","map_area","center_bound","source_gain"]
    frames=[cortex[cols]]
    for path in sorted(LGD.glob("session_*/lgd_aperture_fits.csv")):
        f=pd.read_csv(path);sid=int(f.ecephys_session_id.iloc[0]);f=f.loc[f.heldout_spatial_gain>0].merge(small.drop(columns="ecephys_session_id"),on="ecephys_unit_id",how="left")
        f["ecephys_session_id"]=sid;f["rf_x"]=f.center_x_deg;f["rf_y"]=f.center_y_deg;f["structure_group"]="LGd";f["map_area"]="LGd"
        f["center_bound"]=f.rf_x.abs().ge(59.9)|f.rf_y.abs().ge(59.9);f["source_gain"]=f.heldout_spatial_gain
        frames.append(f[cols])
    return pd.concat(frames,ignore_index=True)


def descriptors(pop):
    rows=[];audit=[]
    usable=pop.loc[~pop.center_bound&pop[CCF+["rf_x","rf_y"]].notna().all(axis=1)]
    for (sid,group),local in usable.groupby(["ecephys_session_id","structure_group"],observed=True):
        found=dispersion(anatomical_residuals(local));rows.append(found)
        audit.append({"ecephys_session_id":sid,"structure_group":group,"coordinate_units":len(local),"valid_dispersion":found.log2_residual_trace.notna().sum(),"center_bound_excluded":int(pop.loc[pop.ecephys_session_id.eq(sid)&pop.structure_group.eq(group),"center_bound"].sum())})
    return pd.concat(rows,ignore_index=True),pd.DataFrame(audit)


def session_surfaces(data,grid,bandwidth=12.):
    return {int(sid):smooth_values(local[["rf_x","rf_y"]].to_numpy(float),local.log2_residual_trace.to_numpy(float),grid,bandwidth) for sid,local in data.groupby("ecephys_session_id",observed=True)}


def template(surfaces,target,min_sessions=5):
    stack=np.stack([v for sid,v in surfaces.items() if sid!=target]);support=np.isfinite(stack).sum(0);out=np.nanmean(stack,axis=0);out[support<min_sessions]=np.nan;return out


def landscape(target,group,data,surfaces,axis,grid_shifts,rng,repeats=50):
    local=data.loc[data.ecephys_session_id.eq(target)].dropna(subset=["log2_residual_trace"]);train=data.loc[~data.ecephys_session_id.eq(target)]
    scale=robust_scale(train.log2_residual_trace.to_numpy(float),.05);interp=interpolator(template(surfaces,target),axis)
    points=local[["rf_x","rf_y"]].to_numpy(float);values=local.log2_residual_trace.to_numpy(float)
    losses=loss_grid(points,values,interp,grid_shifts,scale);shift,minimum=best_shift(losses,grid_shifts)
    null=[]
    for _ in range(repeats):null.append(best_shift(loss_grid(points,rng.permutation(values),interp,grid_shifts,scale),grid_shifts)[1])
    null=np.asarray(null);p=(1+np.sum(null<=minimum))/(len(null)+1);z=(np.median(null)-minimum)/(1.4826*np.median(np.abs(null-np.median(null)))+1e-6)
    return {"losses":losses,"shift":shift,"minimum":minimum,"shuffle_p":p,"shuffle_z":z,"units":len(local)}


def main():
    OUT.mkdir(parents=True,exist_ok=True);pop=load_all();data,audit=descriptors(pop);audit.to_csv(OUT/"session_structure_eligibility.csv",index=False);data.to_csv(OUT/"all_structure_dispersion_descriptors.csv.gz",index=False,compression="gzip")
    minimum={"V1":30,"HVA":50,"LGd":10};wide=audit.pivot(index="ecephys_session_id",columns="structure_group",values="valid_dispersion").fillna(0)
    eligible=wide.index[np.logical_and.reduce([wide.get(g,0)>=minimum[g] for g in GROUPS])].astype(int).tolist();pd.DataFrame({"ecephys_session_id":eligible}).to_csv(OUT/"three_structure_eligible_sessions.csv",index=False)
    axis=np.arange(-75.,75.1,2.);xx,yy=np.meshgrid(axis,axis);grid=np.c_[xx.ravel(),yy.ravel()];sa=np.arange(-30.,30.1,2.);sx,sy=np.meshgrid(sa,sa);shifts=np.c_[sx.ravel(),sy.ravel()]
    surfaces={g:session_surfaces(data.loc[data.structure_group.eq(g)],grid) for g in GROUPS};rng=np.random.default_rng(20260816);rows=[];payload={}
    cases=[sid for sid in CASES if sid in eligible]
    for sid in cases:
        payload[sid]={};joint=np.zeros(len(shifts));weight_total=0
        for g in GROUPS:
            item=landscape(sid,g,data.loc[data.structure_group.eq(g)],surfaces[g],axis,shifts,rng);payload[sid][g]=item
            finite=item["losses"][np.isfinite(item["losses"])]
            spread=max(np.nanquantile(finite,.75)-np.nanquantile(finite,.25),.02);relative=(item["losses"]-item["minimum"])/spread
            weight=float(np.clip(item["shuffle_z"],0,3)/3);joint+=weight*np.nan_to_num(relative,nan=10);weight_total+=weight
            rows.append({"ecephys_session_id":sid,"component":g,"units":item["units"],"shift_az_deg":item["shift"][0],"shift_el_deg":item["shift"][1],"shuffle_p":item["shuffle_p"],"shuffle_z":item["shuffle_z"],"joint_weight":weight})
        joint/=max(weight_total,1e-9);jshift,jmin=best_shift(joint,shifts);payload[sid]["joint"]={"losses":joint,"shift":jshift,"minimum":jmin}
        rows.append({"ecephys_session_id":sid,"component":"joint","units":sum(payload[sid][g]["units"] for g in GROUPS),"shift_az_deg":jshift[0],"shift_el_deg":jshift[1],"shuffle_p":np.nan,"shuffle_z":np.nan,"joint_weight":weight_total})
    result=pd.DataFrame(rows);result.to_csv(OUT/"concrete_component_and_joint_results.csv",index=False)
    fig,axes=plt.subplots(len(cases),4,figsize=(16,4*len(cases)),squeeze=False)
    for r,sid in enumerate(cases):
        for c,g in enumerate((*GROUPS,"joint")):
            item=payload[sid][g];loss=item["losses"].reshape(len(sa),len(sa));rel=loss-np.nanmin(loss)
            im=axes[r,c].imshow(rel,origin="lower",extent=[-30,30,-30,30],aspect="equal",cmap="viridis_r")
            axes[r,c].scatter(item["shift"][0],item["shift"][1],marker="x",s=80,color="red")
            title=f"{sid} {g}\nopt=({item['shift'][0]:+.0f},{item['shift'][1]:+.0f})"
            if g!="joint":title+=f"; p={item['shuffle_p']:.2f}, z={item['shuffle_z']:.1f}"
            axes[r,c].set_title(title);axes[r,c].set_xlabel("az shift (deg)");axes[r,c].set_ylabel("el shift (deg)");fig.colorbar(im,ax=axes[r,c],shrink=.7,label="relative loss")
    fig.suptitle("Structure-specific anatomy-residual dispersion likelihoods and reliability-weighted joint",y=.995);fig.tight_layout();fig.savefig(OUT/"Figure_concrete_joint_dispersion_likelihoods.png",dpi=180);plt.close(fig)
    pd.DataFrame([{"ecephys_session_id":k,"selection_role":v} for k,v in CASES.items()]).to_csv(OUT/"concrete_case_selection.csv",index=False)
    manifest={"anatomical_bandwidth_um":250,"rf_bandwidth_deg":15,"surface_bandwidth_deg":12,"translation_bound_deg":30,"shuffle_repeats":50,"minimum_valid_dispersion":minimum,"joint_weight":"clip(exact-support shuffle z,0,3)/3; component relative loss divided by its grid IQR","exploratory":True}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2));print('eligible',len(eligible),eligible);print(result.to_string(index=False))


if __name__=="__main__":main()
