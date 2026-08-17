#!/usr/bin/env python3
"""Independent target-cell-half validation of frozen multi-structure templates."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.checkpoint_joint_multistructure_dispersion_likelihood import load_all
from scripts.checkpoint_multistructure_dispersion_fields import anatomical_residuals, dispersion, CCF
from scripts.fit_joint_multistructure_dispersion_em import (
    GROUPS, grids, make_base_surfaces, translated_surfaces, loo_template,
    loss_grid_vectorized,
)
from scripts.test_v1_rf_size_corroboration import interpolator, robust_scale, best_shift

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"v1_absolute_size_dispersion_translation_checkpoint"
EM=BASE/"joint_multistructure_dispersion_em"
DESC=BASE/"joint_multistructure_dispersion_checkpoint"/"all_structure_dispersion_descriptors.csv.gz"
ELIG=BASE/"joint_multistructure_dispersion_checkpoint"/"three_structure_eligible_sessions.csv"
OUT=BASE/"joint_multistructure_cell_half_validation"
CASES=(715093703,754829445,760345702)


def stratified_halves(local,rng):
    labels=local.map_area.astype(str)+"|"+local.ecephys_probe_id.astype(str)
    halves=[[],[]]
    for _,idx in labels.groupby(labels).groups.items():
        idx=np.asarray(list(idx));rng.shuffle(idx)
        halves[0].extend(idx[::2]);halves[1].extend(idx[1::2])
    return [local.loc[x].copy() for x in halves]


def half_descriptor(local):
    usable=local.loc[~local.center_bound&local[CCF+["rf_x","rf_y"]].notna().all(axis=1)]
    if len(usable)<8:return pd.DataFrame()
    return dispersion(anatomical_residuals(usable)).dropna(subset=["log2_residual_trace"])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DESC,low_memory=False);ids=pd.read_csv(ELIG).ecephys_session_id.astype(int).tolist();data=data[data.ecephys_session_id.isin(ids)]
    summary=pd.read_csv(EM/"initialization_summary.csv");best=summary.sort_values("final_objective").iloc[0].initialization
    shifts=pd.read_csv(EM/"all_initialization_session_shifts.csv");shifts=shifts[shifts.initialization.eq(best)].set_index("ecephys_session_id")
    current={sid:shifts.loc[sid,["shift_az_deg","shift_el_deg"]].to_numpy(float) for sid in ids}
    axis,grid,_,candidates=grids();scales={g:robust_scale(data.loc[data.structure_group.eq(g),"log2_residual_trace"].dropna().to_numpy(float),.05) for g in GROUPS}
    base=make_base_surfaces(data,ids,grid,axis);surfaces=translated_surfaces(base,current,grid)
    templates={(sid,g):interpolator(loo_template(surfaces[g],sid),axis) for sid in CASES for g in GROUPS}
    pop=load_all();rng=np.random.default_rng(20260816);rows=[]
    for repeat in range(20):
        for sid in CASES:
            opt={}
            for g in GROUPS:
                local=pop.loc[pop.ecephys_session_id.eq(sid)&pop.structure_group.eq(g)].copy()
                for half,subset in enumerate(stratified_halves(local,rng)):
                    found=half_descriptor(subset)
                    if len(found)<10:continue
                    loss=loss_grid_vectorized(found[["rf_x","rf_y"]].to_numpy(float),found.log2_residual_trace.to_numpy(float),templates[(sid,g)],candidates,scales[g])
                    shift,minimum=best_shift(loss,candidates);opt[(g,half)]=shift
                    rows.append({"repeat":repeat,"ecephys_session_id":sid,"structure_group":g,"half":half,"valid_descriptor_cells":len(found),"shift_az_deg":shift[0],"shift_el_deg":shift[1],"minimum_loss":minimum})
            for g in GROUPS:
                if (g,0) in opt and (g,1) in opt:
                    rows.append({"repeat":repeat,"ecephys_session_id":sid,"structure_group":g+"_REPRO","half":-1,"valid_descriptor_cells":np.nan,"shift_az_deg":np.nan,"shift_el_deg":np.nan,"minimum_loss":np.linalg.norm(opt[(g,0)]-opt[(g,1)])})
    result=pd.DataFrame(rows);result.to_csv(OUT/"cell_half_optima.csv",index=False)
    repro=result[result.structure_group.str.endswith("_REPRO")].copy();repro["structure_group"]=repro.structure_group.str.replace("_REPRO","",regex=False)
    report=repro.groupby(["ecephys_session_id","structure_group"]).minimum_loss.agg(["count","median","mean"]).reset_index().rename(columns={"minimum_loss":"split_distance_deg"})
    report.to_csv(OUT/"cell_half_reproducibility_summary.csv",index=False)
    fig,axes=plt.subplots(1,len(CASES),figsize=(13,4),sharey=True)
    for ax,sid in zip(axes,CASES):
        local=repro[repro.ecephys_session_id.eq(sid)]
        vals=[local.loc[local.structure_group.eq(g),"minimum_loss"].to_numpy() for g in GROUPS]
        ax.boxplot(vals,labels=GROUPS);ax.axhline(10,color="tab:red",ls="--",lw=1);ax.set_title(str(sid));ax.set_ylabel("half-to-half optimum distance (deg)");ax.set_ylim(0,70)
    fig.suptitle("Frozen-template target-cell-half reproducibility");fig.tight_layout();fig.savefig(OUT/"Figure_cell_half_reproducibility.png",dpi=180);plt.close(fig)
    (OUT/"run_manifest.json").write_text(json.dumps({"cases":CASES,"repeats":20,"training_templates":"full-data damped EM, target session excluded","halves":"stratified by map area and probe; descriptors recomputed independently","best_em_initialization":best},indent=2))
    print(report.to_string(index=False))

if __name__=="__main__":main()
