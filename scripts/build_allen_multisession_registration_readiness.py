#!/usr/bin/env python3
"""Build cross-session RF geometry summaries and corrected size-surface audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"03_geometry"
DEFAULT_OUTPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"
SESSIONS=(746083955,755434585,760693773,798911424)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-dir",type=Path,default=DEFAULT_GEOMETRY)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    return p.parse_args()


def kernel_surface(local,az_grid,el_grid,bandwidth=15,minimum_effective=3):
    points=local[["azimuth_deg","elevation_deg"]].to_numpy(float);values=local.standardized_log2_area.to_numpy(float)
    az,el=np.meshgrid(az_grid,el_grid);surface=np.full(az.shape,np.nan);effective=np.zeros(az.shape)
    for r in range(len(el_grid)):
        for c in range(len(az_grid)):
            distance=np.sqrt(np.sum((points-[az_grid[c],el_grid[r]])**2,axis=1))
            w=np.exp(-.5*(distance/bandwidth)**2)
            if w.sum()>0:effective[r,c]=w.sum()**2/np.square(w).sum()
            # Effective-n alone is unsafe far outside the observations because
            # many uniformly tiny weights can still produce a large value.
            if effective[r,c]>=minimum_effective and np.sum(distance<=20)>=3:
                surface[r,c]=np.average(values,weights=w)
            else:
                effective[r,c]=0
    return surface,effective


def summaries(fits):
    rows=[]
    for keys,local in fits.groupby(["session_id","group","spatial_model"],observed=True):
        inside=local.loc[local.axis_edge_distance_deg.ge(0)&~local.axis_censored.astype(bool)]
        rho=spearmanr(inside.axis_edge_distance_deg,np.log2(inside.axis_area_deg2)).correlation if len(inside)>4 else np.nan
        rotated=local.loc[local.unit_split.eq("evaluation")&local.rotation_test_gain.notna()]
        rows.append({"session_id":keys[0],"group":keys[1],"spatial_model":keys[2],"units":len(local),
            "inside_uncensored_units":len(inside),"median_axis_area_deg2":inside.axis_area_deg2.median(),
            "spearman_log2_area_vs_edge_distance":rho,"axis_censored_fraction":local.axis_censored.mean(),
            "rotation_units":len(rotated),"median_rotation_test_gain":rotated.rotation_test_gain.median(),
            "fraction_rotation_gain_positive":rotated.rotation_test_gain.gt(0).mean()})
    return pd.DataFrame(rows)


def render_cross_session(summary,path):
    colors={"point":"#3366aa","aperture":"#d97736"};markers={"V1":"o","HVA":"s"}
    fig,axes=plt.subplots(2,2,figsize=(12.4,9),constrained_layout=True)
    x=np.arange(len(SESSIONS))
    for group in ("V1","HVA"):
        for model in ("point","aperture"):
            local=summary.loc[summary.group.eq(group)&summary.spatial_model.eq(model)].set_index("session_id").loc[list(SESSIONS)]
            axes[0,0].plot(x,local.median_axis_area_deg2,marker=markers[group],color=colors[model],
                           ls="-" if group=="V1" else "--",label=f"{group} {model}")
            axes[0,1].plot(x,local.spearman_log2_area_vs_edge_distance,marker=markers[group],color=colors[model],
                           ls="-" if group=="V1" else "--",label=f"{group} {model}")
            axes[1,0].plot(x,100*local.axis_censored_fraction,marker=markers[group],color=colors[model],
                           ls="-" if group=="V1" else "--",label=f"{group} {model}")
            axes[1,1].plot(x,local.median_rotation_test_gain,marker=markers[group],color=colors[model],
                           ls="-" if group=="V1" else "--",label=f"{group} {model}")
    axes[0,0].set(yscale="log",ylabel="Median latent half-max area (deg²)",title="Point versus aperture RF size")
    axes[0,1].axhline(0,color="#555",ls="--");axes[0,1].set(ylabel="Spearman ρ: log₂ area vs edge distance",title="Residual edge association")
    axes[1,0].set(ylabel="Fits reaching bounds (%)",title="Censoring burden")
    axes[1,1].axhline(0,color="#555",ls="--");axes[1,1].set(ylabel="Median held-out rotation gain",title="Does tilt replicate?")
    for axis in axes.ravel():
        axis.set_xticks(x,[str(s) for s in SESSIONS],rotation=30);axis.grid(alpha=.14)
    axes[0,0].legend(frameon=False,ncol=2,fontsize=8)
    fig.suptitle("Cross-session RF geometry and edge audit",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def build_surfaces(fits):
    ap=fits.loc[fits.spatial_model.eq("aperture")&~fits.axis_censored.astype(bool)&
                fits.axis_edge_distance_deg.gt(10)].copy()
    ap["azimuth_deg"]=ap.axis_center_x_deg+50;ap["elevation_deg"]=ap.axis_center_y_deg+10
    ap["log2_area"]=np.log2(ap.axis_area_deg2)
    ap["standardized_log2_area"]=ap.groupby(["session_id","group"],observed=True).log2_area.transform(
        lambda x:(x-x.median())/max(float(x.quantile(.75)-x.quantile(.25)),.25))
    az_grid=np.linspace(10,90,33);el_grid=np.linspace(-30,50,33);surface_rows=[];maps={}
    for sid in SESSIONS:
        for group in ("V1","HVA"):
            local=ap.loc[ap.session_id.eq(sid)&ap.group.eq(group)]
            surface,effective=kernel_surface(local,az_grid,el_grid) if len(local) else (np.full((33,33),np.nan),np.zeros((33,33)))
            maps[(sid,group)]=(surface,effective,len(local))
            for r,elevation in enumerate(el_grid):
                for c,azimuth in enumerate(az_grid):
                    surface_rows.append({"session_id":sid,"group":group,"azimuth_deg":azimuth,"elevation_deg":elevation,
                        "standardized_log2_aperture_area":surface[r,c],"effective_units":effective[r,c],"source_units":len(local)})
    return ap,pd.DataFrame(surface_rows),maps,az_grid,el_grid


def surface_agreement(maps):
    rows=[]
    for group in ("V1","HVA"):
        for i,a in enumerate(SESSIONS):
            for b in SESSIONS[i+1:]:
                ma,ea,_=maps[(a,group)];mb,eb,_=maps[(b,group)]
                valid=np.isfinite(ma)&np.isfinite(mb)&(ea>=3)&(eb>=3)
                rows.append({"group":group,"session_a":a,"session_b":b,"overlap_cells":int(valid.sum()),
                    "surface_correlation":correlation(ma[valid],mb[valid]) if valid.sum()>=10 else np.nan})
    return pd.DataFrame(rows)


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else np.nan


def render_surfaces(maps,az_grid,el_grid,path):
    fig,axes=plt.subplots(len(SESSIONS),2,figsize=(9.5,17),sharex=True,sharey=True,constrained_layout=True)
    for r,sid in enumerate(SESSIONS):
        for c,group in enumerate(("V1","HVA")):
            surface,effective,n=maps[(sid,group)];axis=axes[r,c]
            im=axis.pcolormesh(az_grid,el_grid,surface,shading="auto",cmap="coolwarm",vmin=-1.5,vmax=1.5)
            if np.nanmax(effective)>=3:
                levels=[v for v in (3,6,12) if v<=np.nanmax(effective)]
                axis.contour(az_grid,el_grid,effective,levels=levels,colors="#333",linewidths=.6)
            axis.set(title=f"{sid} · {group} · interior n={n}",xlabel="RF azimuth (deg)",ylabel="RF elevation (deg)",aspect="equal")
    fig.colorbar(im,ax=axes,label="Within-session standardized log₂ aperture RF area",shrink=.55)
    fig.suptitle("Interior analytic-aperture RF-size surfaces for registration audit",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def render_agreement(agreement,path):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),constrained_layout=True)
    for axis,group in zip(axes,("V1","HVA")):
        matrix=pd.DataFrame(np.eye(len(SESSIONS)),index=SESSIONS,columns=SESSIONS)
        for row in agreement.loc[agreement.group.eq(group)].itertuples():
            matrix.loc[row.session_a,row.session_b]=row.surface_correlation;matrix.loc[row.session_b,row.session_a]=row.surface_correlation
        im=axis.imshow(matrix.to_numpy(float),vmin=-1,vmax=1,cmap="coolwarm")
        axis.set(xticks=np.arange(len(SESSIONS)),yticks=np.arange(len(SESSIONS)),xticklabels=SESSIONS,yticklabels=SESSIONS,title=f"{group} pairwise surface correlation")
        axis.tick_params(axis="x",rotation=35)
        for r in range(len(SESSIONS)):
            for c in range(len(SESSIONS)):
                value=matrix.iloc[r,c];axis.text(c,r,"—" if not np.isfinite(value) else f"{value:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=axes,label="Pearson correlation on shared supported grid cells",shrink=.8)
    fig.suptitle("Do corrected RF-size surfaces replicate across sessions?",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();geometry=args.geometry_dir.resolve();output=args.output_dir.resolve()
    cross=output/"06_cross_session";registration=output/"07_registration_readiness"
    cross.mkdir(parents=True,exist_ok=True);registration.mkdir(parents=True,exist_ok=True)
    fits=pd.read_csv(geometry/"all_session_unit_geometry_fits.csv",low_memory=False)
    summary=summaries(fits);summary.to_csv(cross/"cross_session_geometry_summary.csv",index=False,float_format="%.9g")
    render_cross_session(summary,cross/"Figure_cross_session_geometry.png")
    support,grid,maps,az,el=build_surfaces(fits)
    support.to_csv(registration/"interior_aperture_surface_support.csv",index=False,float_format="%.9g")
    grid.to_csv(registration/"aperture_rf_size_surface_grid.csv",index=False,float_format="%.9g")
    agreement=surface_agreement(maps);agreement.to_csv(registration/"pairwise_surface_agreement.csv",index=False,float_format="%.9g")
    render_surfaces(maps,az,el,registration/"Figure_aperture_rf_size_surfaces.png")
    render_agreement(agreement,registration/"Figure_surface_agreement.png")
    print(summary.to_string(index=False));print(agreement.to_string(index=False))


if __name__=="__main__":main()
