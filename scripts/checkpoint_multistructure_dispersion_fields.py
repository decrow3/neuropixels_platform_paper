#!/usr/bin/env python3
"""Matched V1/HVA/LGd residual RF-dispersion fields in concrete sessions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from scripts.check_v1_dispersion_support_geometry import weighted_covariance


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"/"multistructure_dispersion_fields"
FITS=ROOT/"artifacts"/"allen_full_rf_production_v1"/"03_aggregate"/"all_session_unit_geometry_fits.csv"
UNITS=ROOT/"data"/"unit_table.csv"
LGD=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"/"lgd_gabor_boundary_pilot"
SESSIONS=(755434585,760345702,754829445)
CCF=["anterior_posterior_ccf_coordinate","left_right_ccf_coordinate","dorsal_ventral_ccf_coordinate"]


def anatomical_residuals(local, bandwidth=250.):
    result=local.copy().reset_index(drop=True); predicted=np.full((len(result),2),np.nan); effective=np.zeros(len(result))
    for _,idx in result.groupby("map_area",observed=True).groups.items():
        idx=np.asarray(list(idx),int); points=result.loc[idx,CCF].to_numpy(float); rf=result.loc[idx,["rf_x","rf_y"]].to_numpy(float)
        delta=points[:,None,:]-points[None,:,:];w=np.exp(-.5*np.sum(delta**2,axis=2)/bandwidth**2);np.fill_diagonal(w,0)
        total=w.sum(1);eff=total**2/np.maximum((w*w).sum(1),1e-12);p=w@rf/np.maximum(total[:,None],1e-12);p[eff<3]=np.nan
        predicted[idx]=p;effective[idx]=eff
    result[["predicted_x","predicted_y"]]=predicted;result["mean_map_effective_n"]=effective
    result["residual_x"]=result.rf_x-result.predicted_x;result["residual_y"]=result.rf_y-result.predicted_y
    return result


def dispersion(local, rf_bandwidth=15.):
    result=local.copy(); rf=result[["rf_x","rf_y"]].to_numpy(float);res=result[["residual_x","residual_y"]].to_numpy(float)
    delta=rf[:,None,:]-rf[None,:,:];w=np.exp(-.5*np.sum(delta**2,axis=2)/rf_bandwidth**2);np.fill_diagonal(w,0)
    finite=np.isfinite(res).all(1);w[:,~finite]=0;total=w.sum(1);eff=total**2/np.maximum((w*w).sum(1),1e-12)
    cov=weighted_covariance(np.nan_to_num(res),w);trace=np.trace(cov,axis1=1,axis2=2);trace[(eff<3)|~finite]=np.nan
    result["log2_residual_trace"]=np.log2(np.maximum(trace,1e-6));result["rf_neighborhood_effective_n"]=eff
    return result


def load_populations():
    unit=pd.read_csv(UNITS,low_memory=False)
    keep=["ecephys_unit_id","ecephys_session_id","ecephys_structure_acronym","ecephys_probe_id","specimen_id",*CCF]
    unit_small=unit[keep]
    fit=pd.read_csv(FITS,low_memory=False)
    fit=fit.loc[fit.spatial_model.eq("aperture")&fit.session_id.isin(SESSIONS)].copy()
    cortex=fit.merge(unit_small,on="ecephys_unit_id",how="left",suffixes=("","_unit"))
    cortex["ecephys_session_id"]=cortex.session_id.astype(int);cortex["rf_x"]=cortex.axis_center_x_deg;cortex["rf_y"]=cortex.axis_center_y_deg
    cortex["structure_group"]=cortex.group;cortex["map_area"]=cortex.ecephys_structure_acronym
    frames=[cortex[[*keep,"rf_x","rf_y","structure_group","map_area","axis_test_deviance","axis_censored"]]]
    for sid in SESSIONS:
        f=pd.read_csv(LGD/f"session_{sid}"/"lgd_aperture_fits.csv")
        f=f.loc[f.heldout_spatial_gain>0].merge(unit_small.drop(columns="ecephys_session_id"),on="ecephys_unit_id",how="left")
        f["ecephys_session_id"]=sid
        f["rf_x"]=f.center_x_deg;f["rf_y"]=f.center_y_deg;f["structure_group"]="LGd";f["map_area"]="LGd"
        f["axis_test_deviance"]=f.test_deviance;f["axis_censored"]=f.parameter_censored
        frames.append(f[[*keep,"rf_x","rf_y","structure_group","map_area","axis_test_deviance","axis_censored"]])
    return pd.concat(frames,ignore_index=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);pop=load_populations();rows=[];audits=[]
    for (sid,group),local in pop.groupby(["ecephys_session_id","structure_group"],observed=True):
        local=local.dropna(subset=[*CCF,"rf_x","rf_y"])
        found=dispersion(anatomical_residuals(local));rows.append(found)
        audits.append({"ecephys_session_id":sid,"structure_group":group,"units":len(local),"areas":local.map_area.nunique(),"valid_mean_map":found.predicted_x.notna().sum(),"valid_dispersion":found.log2_residual_trace.notna().sum(),"median_rf_effective_n":found.rf_neighborhood_effective_n.median(),"censored_fraction":found.axis_censored.mean()})
    result=pd.concat(rows,ignore_index=True);audit=pd.DataFrame(audits)
    result.to_csv(OUT/"matched_unit_dispersion_descriptors.csv.gz",index=False,compression="gzip");audit.to_csv(OUT/"session_structure_coverage.csv",index=False)
    limits=np.nanquantile(result.log2_residual_trace,[.02,.98]);norm=Normalize(*limits)
    fig,axes=plt.subplots(len(SESSIONS),3,figsize=(13,12),sharex=True,sharey=True)
    for row,sid in enumerate(SESSIONS):
        for col,group in enumerate(("V1","HVA","LGd")):
            ax=axes[row,col];local=result.loc[result.ecephys_session_id.eq(sid)&result.structure_group.eq(group)]
            sc=ax.scatter(local.rf_x,local.rf_y,c=local.log2_residual_trace,cmap="cividis",norm=norm,s=24,alpha=.8)
            boundary=local.axis_censored.astype(bool);ax.scatter(local.loc[boundary,"rf_x"],local.loc[boundary,"rf_y"],facecolors="none",edgecolors="magenta",s=38,linewidth=.6)
            cov=audit.loc[audit.ecephys_session_id.eq(sid)&audit.structure_group.eq(group)].iloc[0]
            ax.set_aspect("equal");ax.set_xlim(-65,65);ax.set_ylim(-65,65);ax.axhline(0,color=".85",lw=.6);ax.axvline(0,color=".85",lw=.6)
            ax.set_title(f"{sid} {group}\nN={int(cov.units)}, valid dispersion={int(cov.valid_dispersion)}")
            ax.set_xlabel("RF grid x (deg)");ax.set_ylabel("RF grid y (deg)")
    fig.colorbar(sc,ax=axes.ravel().tolist(),shrink=.55,label="log2 anatomy-residual covariance trace")
    fig.suptitle("Matched residual RF-dispersion fields; magenta rings mark parameter-censored fits",y=.995)
    fig.savefig(OUT/"Figure_concrete_multistructure_dispersion_fields.png",dpi=180,bbox_inches="tight");plt.close(fig)
    selection=pd.DataFrame([
        {"ecephys_session_id":755434585,"role":"strong prior V1/LGd geometry case"},
        {"ecephys_session_id":760345702,"role":"previous V1 covariance localization success / typical LGd"},
        {"ecephys_session_id":754829445,"role":"multi-probe LGd transverse-geometry counterexample"},
    ]);selection.to_csv(OUT/"concrete_case_selection.csv",index=False)
    manifest={"anatomical_bandwidth_um":250,"rf_neighborhood_bandwidth_deg":15,"hva_mean_maps_separate_by_area":True,"hva_dispersion_pooled_after_area_specific_residualization":True,"lgd_source":"raw Gabor aperture fits with positive held-out spatial gain"}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2));print(audit.to_string(index=False))


if __name__=="__main__":main()
