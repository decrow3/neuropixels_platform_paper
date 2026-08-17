#!/usr/bin/env python3
"""Area-specific corrected RF-size surface reliability relative to V1."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"03_geometry"/"all_session_unit_geometry_fits.csv"
OUTPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"07_registration_readiness"
AREAS=("VISp","VISl","VISrl","VISal","VISpm","VISam")


def surface(local,az_grid,el_grid,bandwidth=15):
    points=local[["azimuth_deg","elevation_deg"]].to_numpy(float)
    values=local.centered_log2_area.to_numpy(float);result=np.full((len(el_grid),len(az_grid)),np.nan)
    effective=np.zeros_like(result)
    for r,elevation in enumerate(el_grid):
        for c,azimuth in enumerate(az_grid):
            distance=np.sqrt(np.sum((points-[azimuth,elevation])**2,axis=1));w=np.exp(-.5*(distance/bandwidth)**2)
            eff=w.sum()**2/np.square(w).sum() if w.sum()>0 else 0
            if eff>=3 and np.sum(distance<=20)>=3:
                result[r,c]=np.average(values,weights=w);effective[r,c]=eff
    return result,effective


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fits=pd.read_csv(INPUT,low_memory=False)
    selected=fits.loc[fits.spatial_model.eq("aperture")&~fits.axis_censored.astype(bool)&
                      fits.axis_edge_distance_deg.gt(10)].copy()
    selected["azimuth_deg"]=selected.axis_center_x_deg+50;selected["elevation_deg"]=selected.axis_center_y_deg+10
    selected["log2_area"]=np.log2(selected.axis_area_deg2)
    selected["centered_log2_area"]=selected.groupby(["session_id","ecephys_structure_acronym"],observed=True).log2_area.transform(lambda x:x-x.median())
    az=np.linspace(10,90,33);el=np.linspace(-30,50,33);maps={};grid_rows=[];session_rows=[]
    for (sid,area),local in selected.groupby(["session_id","ecephys_structure_acronym"],observed=True):
        if area not in AREAS or len(local)<8:continue
        value,effective=surface(local,az,el);maps[(sid,area)]=(value,effective)
        finite=np.isfinite(value)
        session_rows.append({"session_id":sid,"area":area,"source_units":len(local),"supported_cells":int(finite.sum()),
                             "surface_spatial_sd_log2_area":float(np.nanstd(value))})
        for r,elevation in enumerate(el):
            for c,azimuth in enumerate(az):
                grid_rows.append({"session_id":sid,"area":area,"azimuth_deg":azimuth,"elevation_deg":elevation,
                                  "centered_log2_aperture_area":value[r,c],"effective_units":effective[r,c]})
    agreements=[]
    for area in AREAS:
        sessions=sorted(sid for sid,a in maps if a==area)
        for a,b in combinations(sessions,2):
            ma,ea=maps[(a,area)];mb,eb=maps[(b,area)];valid=np.isfinite(ma)&np.isfinite(mb)&(ea>=3)&(eb>=3)
            corr=np.corrcoef(ma[valid],mb[valid])[0,1] if valid.sum()>=10 and np.std(ma[valid])>0 and np.std(mb[valid])>0 else np.nan
            agreements.append({"area":area,"session_a":a,"session_b":b,"overlap_cells":int(valid.sum()),"surface_correlation":corr})
    sessions=pd.DataFrame(session_rows);agreement=pd.DataFrame(agreements);grid=pd.DataFrame(grid_rows)
    sessions.to_csv(OUTPUT/"area_specific_surface_summary.csv",index=False,float_format="%.9g")
    agreement.to_csv(OUTPUT/"area_specific_pairwise_agreement.csv",index=False,float_format="%.9g")
    grid.to_csv(OUTPUT/"area_specific_surface_grid.csv",index=False,float_format="%.9g")
    render(sessions,agreement,OUTPUT/"Figure_area_specific_registration_readiness.png")
    print(sessions.to_string(index=False));print(agreement.to_string(index=False))


def render(sessions,agreement,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True);x=np.arange(len(AREAS))
    for i,area in enumerate(AREAS):
        local=agreement.loc[agreement.area.eq(area)&agreement.surface_correlation.notna()]
        axes[0].scatter(np.full(len(local),i),local.surface_correlation,s=45,alpha=.7,
                        color="#3366aa" if area=="VISp" else "#d97736")
        if len(local):axes[0].plot([i-.22,i+.22],[local.surface_correlation.median()]*2,color="#222",lw=2)
        local_s=sessions.loc[sessions.area.eq(area)]
        axes[1].scatter(np.full(len(local_s),i),local_s.surface_spatial_sd_log2_area,s=55,alpha=.75,
                        color="#3366aa" if area=="VISp" else "#d97736")
    axes[0].axhline(0,color="#555",ls="--");axes[0].set(xticks=x,xticklabels=AREAS,ylim=(-1,1),
        ylabel="Pairwise surface correlation",title="Cross-session shape agreement")
    axes[1].set(xticks=x,xticklabels=AREAS,ylabel="Spatial SD of centered log₂ RF area",
        title="Within-area size variation across visual space")
    axes[0].tick_params(axis="x",rotation=30);axes[1].tick_params(axis="x",rotation=30)
    v1=sessions.loc[sessions.area.eq("VISp"),"surface_spatial_sd_log2_area"]
    if len(v1):axes[1].axhspan(v1.min(),v1.max(),color="#3366aa",alpha=.12,label="observed V1 session span")
    axes[1].legend(frameon=False)
    for axis in axes:axis.grid(alpha=.14)
    fig.suptitle("Area-specific analytic-aperture RF-size surfaces: pilot registration evidence",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":main()
